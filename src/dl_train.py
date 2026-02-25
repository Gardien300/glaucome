"""
Generic training loop for deep learning models (PyTorch) for the glaucoma project.

Features:
- Medical augmentations & normalisation (Bloc A)
- Focal loss / label smoothing (Bloc H)
- Phased fine-tuning with discriminative LR (Bloc D)
- Attention regularisation loss via papilla mask (Bloc C)
- TTA evaluation (Bloc E)
- ECE / Brier / reliability diagram (Bloc F)
- Enriched checkpoints with git hash + full config (Bloc I)

Usage:

    python -m src.dl_train --config configs/dl_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.special import expit as sigmoid

from .config import PROJECT_ROOT, RANDOM_STATE, MODELS_DIR, REPORTS_DIR
from .dl_data import build_dl_splits, create_dataloaders
from .dl_models import (
    build_model,
    build_dual_branch_model,
    count_trainable_parameters,
    create_discriminative_param_groups,
)
from . import evaluation_metrics as eval_metrics

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("PyYAML requis (pip install PyYAML).") from e


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = RANDOM_STATE) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Git hash (Bloc I)
# ---------------------------------------------------------------------------

def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Custom transforms (Bloc A)
# ---------------------------------------------------------------------------

class RandomGamma:
    """Random gamma correction (simulates brightness variation across devices)."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2)):
        self.lo, self.hi = gamma_range

    def __call__(self, img):
        import torchvision.transforms.functional as F
        gamma = float(torch.empty(1).uniform_(self.lo, self.hi).item())
        return F.adjust_gamma(img, gamma)


class Vignetting:
    """Simulate retinograph vignetting (dark corners)."""

    def __init__(self, strength: float = 0.3):
        self.strength = strength

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            return img
        _, h, w = img.shape
        Y, X = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        r = torch.sqrt(X ** 2 + Y ** 2)
        mask = 1.0 - self.strength * (r / r.max()).clamp(0, 1)
        return img * mask.unsqueeze(0)


def build_transforms(
    image_size: int = 224,
    augmentation_level: str = "strong",
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build train / eval transforms with medical-specific augmentations.

    Args:
        image_size: target spatial size.
        augmentation_level: "light", "medium" or "strong".
    """
    NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if augmentation_level == "light":
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(**NORM),
        ])
    elif augmentation_level == "medium":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            RandomGamma(gamma_range=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(**NORM),
        ])
    else:  # strong
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            RandomGamma(gamma_range=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            Vignetting(strength=0.25),
            transforms.Normalize(**NORM),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(**NORM),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Focal Loss (Bloc H)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary focal loss for class-imbalanced tasks.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight,
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        return (focal_weight * ce_loss).mean()


def _build_criterion(
    train_cfg: Dict,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> nn.Module:
    """Build loss from config (Bloc H)."""
    loss_type = train_cfg.get("loss", "bce")
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    if loss_type == "focal":
        alpha = float(train_cfg.get("focal_alpha", 0.75))
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        print(f"Using FocalLoss (alpha={alpha}, gamma={gamma})")
        return FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)

    if label_smoothing > 0:
        print(f"Using BCEWithLogitsLoss + label smoothing={label_smoothing}")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _smooth_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply label smoothing: y_smooth = y * (1 - eps) + eps / 2."""
    if smoothing <= 0:
        return labels
    return labels * (1.0 - smoothing) + smoothing / 2.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    grad_clip_norm: Optional[float] = None,
    label_smoothing: float = 0.0,
    dual_branch: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0
    for batch in loader:
        if dual_branch:
            images, papilla, labels = batch
            images = images.to(device, non_blocking=True)
            papilla = papilla.to(device, non_blocking=True)
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1, 1)
        labels = _smooth_labels(labels, label_smoothing)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images, papilla) if dual_branch else model(images)
                loss = criterion(logits, labels)
            loss.backward()
        else:
            logits = model(images, papilla) if dual_branch else model(images)
            loss = criterion(logits, labels)
            loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.float().item() * batch_size
        n_samples += batch_size

    return running_loss / max(n_samples, 1)


def get_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    dual_branch: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if dual_branch:
                images, papilla, labels = batch
                images = images.to(device, non_blocking=True)
                papilla = papilla.to(device, non_blocking=True)
            else:
                images, labels = batch
                images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(images, papilla) if dual_branch else model(images)
            else:
                logits = model(images, papilla) if dual_branch else model(images)
            all_logits.append(logits.squeeze(1).float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_logits), np.concatenate(all_labels)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling: minimise NLL to find optimal T."""
    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        T = max(float(T), 1e-6)
        p = sigmoid(logits / T)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.sum(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
    return float(res.x)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: Optional[float] = None,
    use_amp: bool = False,
    dual_branch: bool = False,
) -> Dict[str, float]:
    """Standard evaluation on a loader."""
    logits, y_true = get_logits_and_labels(model, loader, device, use_amp, dual_branch)
    if temperature is not None and temperature != 1.0:
        y_prob = sigmoid(logits / temperature)
    else:
        y_prob = sigmoid(logits)
    return eval_metrics.compute_all_metrics(y_true.astype(int), y_prob)


def _evaluate_with_calibration_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: Optional[float],
    use_amp: bool,
    dual_branch: bool,
    reports_dir: Path,
    split_name: str,
) -> Dict[str, float]:
    """Evaluate + compute ECE, Brier, and save reliability diagram."""
    logits, y_true = get_logits_and_labels(model, loader, device, use_amp, dual_branch)
    if temperature is not None and temperature != 1.0:
        y_prob = sigmoid(logits / temperature)
    else:
        y_prob = sigmoid(logits)

    metrics = eval_metrics.compute_all_metrics(y_true.astype(int), y_prob)
    metrics["ece"] = eval_metrics.expected_calibration_error(y_true.astype(int), y_prob)
    metrics["brier"] = eval_metrics.brier_score(y_true.astype(int), y_prob)

    rel_path = reports_dir / f"reliability_{split_name}.png"
    try:
        eval_metrics.plot_reliability_diagram(
            y_true.astype(int), y_prob, save_path=str(rel_path),
            title=f"Calibration — {split_name}",
        )
    except Exception:
        pass

    return metrics


# ---------------------------------------------------------------------------
# Checkpoint (Bloc I)
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: nn.Module,
    backbone: str,
    image_size: int,
    temperature: float,
    out_dir: Path,
    suffix: str = "best",
    cfg: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
    epoch: Optional[int] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone_name = backbone.replace("/", "_")
    out_path = out_dir / f"dl_{backbone_name}_{suffix}.pth"
    checkpoint = {
        "state_dict": model.state_dict(),
        "temperature": temperature,
        "backbone": backbone,
        "image_size": image_size,
        "git_hash": _get_git_hash(),
        "timestamp": datetime.now().isoformat(),
    }
    if cfg is not None:
        checkpoint["config"] = cfg
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if epoch is not None:
        checkpoint["epoch"] = epoch
    torch.save(checkpoint, out_path)
    print(f"  [Persist] Checkpoint sauvegardé -> {out_path}")


def load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Product targets check
# ---------------------------------------------------------------------------

def _check_product_targets(eval_m: Dict[str, float], holdout_m: Dict[str, float]) -> None:
    targets_path = PROJECT_ROOT / "configs" / "product_targets.yaml"
    if not targets_path.exists():
        return
    try:
        with open(targets_path, "r") as f:
            targets_cfg = yaml.safe_load(f)
    except Exception:
        return
    metrics_cfg = targets_cfg.get("metrics", {})
    if not metrics_cfg:
        return

    print("\n=== Cibles produit (commercialisable / efficace patient) ===")
    for split_name, metrics_dict in [("eval", eval_m), ("holdout", holdout_m)]:
        print(f"  [{split_name}]")
        for key, limits in metrics_cfg.items():
            if not isinstance(limits, dict) or "min" not in limits:
                continue
            val = metrics_dict.get(key)
            if val is None:
                continue
            min_val = float(limits["min"])
            ideal = float(limits.get("ideal", min_val))
            if val >= ideal:
                status = "OK (idéal)"
            elif val >= min_val:
                status = "OK (min)"
            else:
                status = "EN DESSOUS DU MINIMUM"
            print(f"    {key}: {val:.4f} (min={min_val:.2f}, idéal={ideal:.2f}) -> {status}")
    print("  Voir PRODUCT_VISION.md pour le cadre produit et réglementaire.\n")


# ---------------------------------------------------------------------------
# Phased training (Bloc D)
# ---------------------------------------------------------------------------

def _apply_phased_training(
    model: nn.Module,
    cfg: Dict,
    phase_idx: int,
) -> None:
    """
    Gradually unfreeze blocks: phase 0 = head only, phase 1 = last block + head, etc.
    """
    from .dl_models import get_backbone_block_groups

    for p in model.parameters():
        p.requires_grad = False

    groups = get_backbone_block_groups(model)
    n_groups = len(groups)
    n_unlock = min(phase_idx + 1, n_groups)
    for g in groups[n_groups - n_unlock:]:
        for p in g:
            p.requires_grad = True

    n_trainable = count_trainable_parameters(model)
    print(f"  Phase {phase_idx}: {n_unlock}/{n_groups} groupes dégelés, {n_trainable:,} params trainables")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_from_config(cfg: Dict) -> None:
    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_df, val_df, eval_df, holdout_df = build_dl_splits()

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))
    persistent_workers = bool(train_cfg.get("persistent_workers", False))
    prefetch_factor = train_cfg.get("prefetch_factor")
    if prefetch_factor is not None:
        prefetch_factor = int(prefetch_factor)

    aug_level = str(train_cfg.get("augmentation_level", "strong"))
    dual_branch = bool(model_cfg.get("dual_branch", False))
    papilla_size = int(model_cfg.get("papilla_image_size", 128))
    papilla_crop_ratio = float(model_cfg.get("papilla_crop_ratio", 0.22))

    train_tf, eval_tf = build_transforms(image_size=image_size, augmentation_level=aug_level)

    papilla_tf = None
    if dual_branch:
        papilla_tf = transforms.Compose([
            transforms.Resize((papilla_size, papilla_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    loaders = create_dataloaders(
        train_df=train_df, val_df=val_df, eval_df=eval_df, holdout_df=holdout_df,
        train_transforms=train_tf, eval_transforms=eval_tf,
        batch_size=batch_size, num_workers=num_workers,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        dual_branch=dual_branch, papilla_transforms=papilla_tf,
        papilla_crop_ratio=papilla_crop_ratio,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))

    if dual_branch:
        papilla_backbone = model_cfg.get("papilla_backbone", "efficientnet_v2_s")
        fusion_dim = int(model_cfg.get("fusion_dim", 512))
        model = build_dual_branch_model(
            global_backbone=backbone,
            papilla_backbone=papilla_backbone,
            pretrained=pretrained,
            fusion_dim=fusion_dim,
            freeze_backbone=freeze_backbone,
        ).to(device)
    else:
        model = build_model(backbone=backbone, pretrained=pretrained, freeze_backbone=freeze_backbone).to(device)

    use_amp = bool(train_cfg.get("amp", False))
    do_compile = bool(train_cfg.get("compile", False))
    if use_amp:
        print("AMP (BF16) activé")
    if do_compile and not dual_branch:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile activé (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile non disponible ({e}), continu sans compile.")

    n_params = count_trainable_parameters(model)
    print(f"Model {backbone} — trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Loss (Bloc H)
    # ------------------------------------------------------------------
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    num_epochs = int(train_cfg.get("num_epochs", 20))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    pos_weight_cfg = train_cfg.get("pos_weight", 29.0)
    if isinstance(pos_weight_cfg, str) and pos_weight_cfg.lower() == "auto":
        n_pos = max(int((train_df["class"] == "RG").sum()), 1)
        n_neg = max(int((train_df["class"] == "NRG").sum()), 1)
        pos_weight_value = float(n_neg / n_pos)
        print(f"Computed pos_weight from train distribution: {pos_weight_value:.2f} (neg={n_neg}, pos={n_pos})")
    else:
        pos_weight_value = float(pos_weight_cfg)
    pos_weight = torch.tensor([pos_weight_value], device=device)

    criterion = _build_criterion(train_cfg, pos_weight, device)

    # ------------------------------------------------------------------
    # Optimizer (Bloc D: discriminative LR)
    # ------------------------------------------------------------------
    use_discriminative_lr = bool(train_cfg.get("discriminative_lr", False))
    lr_decay = float(train_cfg.get("lr_decay", 0.3))

    if use_discriminative_lr:
        param_groups = create_discriminative_param_groups(
            model, lr_head=lr, lr_decay=lr_decay, weight_decay=weight_decay,
        )
        optimizer = AdamW(param_groups)
        print(f"Discriminative LR activé ({len(param_groups)} groupes, decay={lr_decay})")
    else:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", None)
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)
    if grad_clip_norm is not None and grad_clip_norm > 0:
        print(f"Gradient clipping: max_norm={grad_clip_norm}")

    scheduler = None
    if scheduler_type == "cosine":
        t_max = int(scheduler_cfg.get("T_max", num_epochs))
        eta_min = float(scheduler_cfg.get("eta_min", 1e-6))
        if warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=max(1, t_max - warmup_epochs), eta_min=eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
            print(f"Using warmup ({warmup_epochs} epochs) + CosineAnnealingLR: T_max={t_max}, eta_min={eta_min}")
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
            print(f"Using CosineAnnealingLR scheduler: T_max={t_max}, eta_min={eta_min}")
    elif scheduler_type == "plateau":
        factor = float(scheduler_cfg.get("factor", 0.3))
        sched_patience = int(scheduler_cfg.get("patience", 3))
        min_lr = float(scheduler_cfg.get("min_lr", 1e-7))
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=sched_patience, min_lr=min_lr)
        print(f"Using ReduceLROnPlateau: factor={factor}, patience={sched_patience}, min_lr={min_lr}")

    # ------------------------------------------------------------------
    # Phased training (Bloc D)
    # ------------------------------------------------------------------
    phase_epochs_cfg = train_cfg.get("phase_epochs", None)
    use_phases = phase_epochs_cfg is not None and not use_discriminative_lr
    if use_phases:
        phase_epochs = [int(e) for e in phase_epochs_cfg]
        print(f"Phased fine-tuning: {len(phase_epochs)} phases — epochs per phase: {phase_epochs}")
    else:
        phase_epochs = [num_epochs]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_auc = 0.0
    best_state = None
    patience = int(train_cfg.get("patience", 5))
    epochs_no_improve = 0
    history: List[Dict] = []
    global_epoch = 0

    for phase_idx, n_phase_epochs in enumerate(phase_epochs):
        if use_phases:
            _apply_phased_training(model, cfg, phase_idx)
            if phase_idx > 0:
                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr * (0.5 ** phase_idx), weight_decay=weight_decay,
                )

        for local_epoch in range(1, n_phase_epochs + 1):
            global_epoch += 1
            current_lr = optimizer.param_groups[0]["lr"]
            if use_phases:
                print(f"\nPhase {phase_idx} — Epoch {local_epoch}/{n_phase_epochs} (global {global_epoch}) — lr={current_lr:.6f}")
            else:
                print(f"\nEpoch {global_epoch}/{num_epochs} — lr={current_lr:.6f}")

            train_loss = train_one_epoch(
                model, loaders["train"], criterion, optimizer, device,
                use_amp=use_amp, grad_clip_norm=grad_clip_norm,
                label_smoothing=label_smoothing, dual_branch=dual_branch,
            )
            print(f"  Train loss: {train_loss:.4f}")

            val_metrics = evaluate_model(model, loaders["val"], device, use_amp=use_amp, dual_branch=dual_branch)
            val_auc = val_metrics.get("auc_roc", 0.0)
            print(
                f"  Val AUC: {val_auc:.4f} | "
                f"pAUC: {val_metrics.get('pauc_90_100', 0.0):.4f} | "
                f"Sens@95: {val_metrics.get('sens_at_95spec', 0.0):.4f}"
            )

            history.append({
                "epoch": global_epoch,
                "phase": phase_idx if use_phases else 0,
                "lr": current_lr,
                "train_loss": float(train_loss),
                "val_auc_roc": float(val_auc),
                "val_pauc_90_100": float(val_metrics.get("pauc_90_100", 0.0)),
                "val_sens_at_95spec": float(val_metrics.get("sens_at_95spec", 0.0)),
                "val_sens_at_90spec": float(val_metrics.get("sens_at_90spec", 0.0)),
            })

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                print("  -> New best model on validation.")
                _save_checkpoint(
                    model, backbone, image_size, 1.0, MODELS_DIR, "best",
                    cfg=cfg, metrics=val_metrics, epoch=global_epoch,
                )
            else:
                epochs_no_improve += 1
                print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

            save_interval = int(train_cfg.get("save_interval_epochs", 5))
            if save_interval > 0 and global_epoch % save_interval == 0:
                _save_checkpoint(
                    model, backbone, image_size, 1.0, MODELS_DIR, "last",
                    cfg=cfg, epoch=global_epoch,
                )

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_auc)
                else:
                    scheduler.step()

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # Save training history (Bloc I)
    # ------------------------------------------------------------------
    reports_dir = REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_path = reports_dir / f"dl_{backbone}_train_history.json"
    try:
        with open(history_path, "w") as f:
            json.dump({
                "backbone": backbone,
                "image_size": image_size,
                "git_hash": _get_git_hash(),
                "timestamp": datetime.now().isoformat(),
                "config": cfg,
                "history": history,
            }, f, indent=2)
        print(f"\nSaved training history to {history_path}")
    except Exception as e:
        print(f"\nWarning: could not save history: {e}")

    # ------------------------------------------------------------------
    # Calibration (temperature scaling)
    # ------------------------------------------------------------------
    calibration_enabled = bool(cfg.get("calibration", {}).get("enabled", False))
    temperature = 1.0
    if calibration_enabled:
        val_logits, val_labels = get_logits_and_labels(
            model, loaders["val"], device, use_amp=use_amp, dual_branch=dual_branch,
        )
        temperature = fit_temperature(val_logits, val_labels)
        print(f"\nCalibration (temperature scaling): T = {temperature:.4f}")

    # ------------------------------------------------------------------
    # Final evaluation with calibration metrics (Bloc F)
    # ------------------------------------------------------------------
    print("\n=== Final evaluation on eval set ===")
    eval_m = _evaluate_with_calibration_metrics(
        model, loaders["eval"], device,
        temperature=temperature if calibration_enabled else None,
        use_amp=use_amp, dual_branch=dual_branch,
        reports_dir=reports_dir, split_name="eval",
    )
    for k, v in eval_m.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Final evaluation on holdout set ===")
    holdout_m = _evaluate_with_calibration_metrics(
        model, loaders["holdout"], device,
        temperature=temperature if calibration_enabled else None,
        use_amp=use_amp, dual_branch=dual_branch,
        reports_dir=reports_dir, split_name="holdout",
    )
    for k, v in holdout_m.items():
        print(f"  {k}: {v:.4f}")

    _check_product_targets(eval_m, holdout_m)

    # ------------------------------------------------------------------
    # Save final checkpoint (Bloc I)
    # ------------------------------------------------------------------
    _save_checkpoint(
        model, backbone, image_size, temperature, MODELS_DIR, "best",
        cfg=cfg,
        metrics={"eval": eval_m, "holdout": holdout_m},
        epoch=global_epoch,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep learning training for glaucoma project")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = load_config(cfg_path)
    train_from_config(cfg)


if __name__ == "__main__":
    main()
