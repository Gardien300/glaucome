"""
Generic training loop for deep learning models (PyTorch) for the glaucoma project.

This script is intended to be run both locally (for quick tests on CPU)
and on a GPU pod (for full training).

Usage (example):

    python -m src.dl_train --config configs/dl_baseline.yaml

The config file should define:
  - model.backbone (default: efficientnet_b0, recommended for fundus/glaucoma)
  - model.pretrained, model.freeze_backbone
  - train.batch_size, train.num_epochs, train.lr, train.weight_decay
  - train.pos_weight (for BCEWithLogitsLoss)
  - data.image_size (optional)
  - calibration.enabled (optional): if true, temperature scaling is fitted on val and applied at eval/save.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from .dl_models import build_model, count_trainable_parameters
from . import evaluation_metrics as eval_metrics


try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("PyYAML requis pour lire les fichiers de config (pip install PyYAML).") from e


def set_seed(seed: int = RANDOM_STATE) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Simple train / eval transforms using torchvision.
    """
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_tf, eval_tf


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    grad_clip_norm: Optional[float] = None,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1, 1)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, labels)
            loss.backward()
        else:
            logits = model(images)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all logits and labels from a loader (for calibration)."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(images)
            else:
                logits = model(images)
            all_logits.append(logits.squeeze(1).float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Temperature scaling: find T > 0 that minimizes NLL on (logits, labels).
    Returns the optimal temperature T (applied as probs = sigmoid(logits / T)).
    """
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
) -> Dict[str, float]:
    """
    Run inference on a loader and compute AIROGS-style metrics.
    If temperature is set, use calibrated probs: sigmoid(logits / temperature).
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(images)
            else:
                logits = model(images)
            all_logits.append(logits.squeeze(1).float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_score = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    if temperature is not None and temperature != 1.0:
        y_prob = sigmoid(y_score / temperature)
    else:
        y_prob = sigmoid(y_score)

    metrics = eval_metrics.compute_all_metrics(y_true.astype(int), y_prob)
    return metrics


def _save_checkpoint_now(
    model: nn.Module,
    backbone: str,
    image_size: int,
    temperature: float,
    out_dir: Path,
    suffix: str = "best",
) -> None:
    """Sauvegarde immédiate du checkpoint (pour persistance RunPod / manque crédits)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone_name = backbone.replace("/", "_")
    out_path = out_dir / f"dl_{backbone_name}_{suffix}.pth"
    checkpoint = {
        "state_dict": model.state_dict(),
        "temperature": temperature,
        "backbone": backbone,
        "image_size": image_size,
    }
    torch.save(checkpoint, out_path)
    print(f"  [Persist] Checkpoint sauvegardé -> {out_path}")


def load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _check_product_targets(eval_metrics: Dict[str, float], holdout_metrics: Dict[str, float]) -> None:
    """
    Affiche si les métriques eval/holdout atteignent les cibles produit
    (configs/product_targets.yaml, PRODUCT_VISION.md).
    """
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
    for split_name, metrics_dict in [("eval", eval_metrics), ("holdout", holdout_metrics)]:
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


def train_from_config(cfg: Dict) -> None:
    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_df, val_df, eval_df, holdout_df = build_dl_splits()

    train_cfg = cfg.get("train", {})
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))
    persistent_workers = bool(train_cfg.get("persistent_workers", False))
    prefetch_factor = train_cfg.get("prefetch_factor")
    if prefetch_factor is not None:
        prefetch_factor = int(prefetch_factor)

    train_tf, eval_tf = build_transforms(image_size=image_size)
    loaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        holdout_df=holdout_df,
        train_transforms=train_tf,
        eval_transforms=eval_tf,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))

    model = build_model(
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)

    use_amp = bool(train_cfg.get("amp", False))
    do_compile = bool(train_cfg.get("compile", False))
    if use_amp:
        print("AMP (BF16) activé")
    if do_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile activé (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile non disponible ({e}), continu sans compile.")

    n_params = count_trainable_parameters(model)
    print(f"Model {backbone} — trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    num_epochs = int(train_cfg.get("num_epochs", 20))

    # Gestion flexible de pos_weight : valeur numérique ou "auto" (calculée sur le train)
    pos_weight_cfg = train_cfg.get("pos_weight", 29.0)
    if isinstance(pos_weight_cfg, str) and pos_weight_cfg.lower() == "auto":
        # Calculer le ratio négatifs / positifs sur le train DL (0+1+2)
        n_pos = max(int((train_df["class"] == "RG").sum()), 1)
        n_neg = max(int((train_df["class"] == "NRG").sum()), 1)
        pos_weight_value = float(n_neg / n_pos)
        print(f"Computed pos_weight from train distribution: {pos_weight_value:.2f} (neg={n_neg}, pos={n_pos})")
    else:
        pos_weight_value = float(pos_weight_cfg)
    pos_weight = torch.tensor([pos_weight_value], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Optional scheduler (cosine annealing, with optional warmup)
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
            warmup = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=max(1, t_max - warmup_epochs), eta_min=eta_min
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
            print(
                f"Using warmup ({warmup_epochs} epochs) + CosineAnnealingLR: "
                f"T_max={t_max}, eta_min={eta_min}"
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
            print(f"Using CosineAnnealingLR scheduler: T_max={t_max}, eta_min={eta_min}")
    elif scheduler_type == "plateau":
        factor = float(scheduler_cfg.get("factor", 0.3))
        sched_patience = int(scheduler_cfg.get("patience", 3))
        min_lr = float(scheduler_cfg.get("min_lr", 1e-7))
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=factor,
            patience=sched_patience,
            min_lr=min_lr,
            verbose=True,
        )
        print(
            f"Using ReduceLROnPlateau scheduler: factor={factor}, "
            f"patience={sched_patience}, min_lr={min_lr}"
        )

    # ------------------------------------------------------------------
    # Training loop (with simple early stopping on val AUC)
    # ------------------------------------------------------------------
    best_auc = 0.0
    best_state = None
    patience = int(train_cfg.get("patience", 5))
    epochs_no_improve = 0

    # Historique des métriques par epoch pour analyse ultérieure
    history = []

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{num_epochs} — lr={current_lr:.6f}")
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
        )
        print(f"  Train loss: {train_loss:.4f}")

        val_metrics = evaluate_model(model, loaders["val"], device, use_amp=use_amp)
        val_auc = val_metrics.get("auc_roc", 0.0)
        print(
            f"  Val AUC: {val_auc:.4f} | "
            f"pAUC: {val_metrics.get('pauc_90_100', 0.0):.4f} | "
            f"Sens@95: {val_metrics.get('sens_at_95spec', 0.0):.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": float(train_loss),
                "val_auc_roc": float(val_auc),
                "val_pauc_90_100": float(val_metrics.get("pauc_90_100", 0.0)),
                "val_sens_at_95spec": float(val_metrics.get("sens_at_95spec", 0.0)),
                "val_sens_at_90spec": float(val_metrics.get("sens_at_90spec", 0.0)),
            }
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
            epochs_no_improve = 0
            print("  -> New best model on validation.")
            _save_checkpoint_now(model, backbone, image_size, 1.0, out_dir=MODELS_DIR, suffix="best")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

        # Sauvegarde périodique de secours (dernier état) pour RunPod / manque crédits
        save_interval = int(train_cfg.get("save_interval_epochs", 5))
        if save_interval > 0 and epoch % save_interval == 0:
            _save_checkpoint_now(model, backbone, image_size, 1.0, out_dir=MODELS_DIR, suffix="last")

        if scheduler is not None:
            if scheduler_type == "cosine" or warmup_epochs > 0:
                scheduler.step()
            elif scheduler_type == "plateau":
                scheduler.step(val_auc)

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Sauvegarde de l'historique d'entraînement (pour consolidation ultérieure)
    reports_dir = REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_path = reports_dir / f"dl_{backbone}_train_history.json"
    try:
        with open(history_path, "w") as f:
            json.dump(
                {
                    "backbone": backbone,
                    "image_size": image_size,
                    "train_cfg": train_cfg,
                    "history": history,
                },
                f,
                indent=2,
            )
        print(f"\nSaved training history to {history_path}")
    except Exception as e:
        print(f"\nWarning: could not save training history to {history_path}: {e}")

    # ------------------------------------------------------------------
    # Calibration (temperature scaling) sur le set de validation
    # ------------------------------------------------------------------
    calibration_enabled = bool(cfg.get("calibration", {}).get("enabled", False))
    temperature = 1.0

    if calibration_enabled:
        val_logits, val_labels = get_logits_and_labels(
            model, loaders["val"], device, use_amp=use_amp
        )
        temperature = fit_temperature(val_logits, val_labels)
        print(f"\nCalibration (temperature scaling): T = {temperature:.4f}")

    # ------------------------------------------------------------------
    # Final evaluation on eval + holdout (avec probas calibrées si activé)
    # ------------------------------------------------------------------
    print("\n=== Final evaluation on eval set ===")
    eval_metrics_dict = evaluate_model(
        model,
        loaders["eval"],
        device,
        temperature=temperature if calibration_enabled else None,
        use_amp=use_amp,
    )
    for k, v in eval_metrics_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Final evaluation on holdout set (folder 5) ===")
    holdout_metrics_dict = evaluate_model(
        model,
        loaders["holdout"],
        device,
        temperature=temperature if calibration_enabled else None,
        use_amp=use_amp,
    )
    for k, v in holdout_metrics_dict.items():
        print(f"  {k}: {v:.4f}")

    # Vérification des cibles produit (voir PRODUCT_VISION.md, configs/product_targets.yaml)
    _check_product_targets(eval_metrics_dict, holdout_metrics_dict)

    # Save checkpoint (state_dict + temperature + backbone + image_size pour inférence)
    out_dir = MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone_name = backbone.replace("/", "_")
    out_path = out_dir / f"dl_{backbone_name}_best.pth"
    checkpoint = {
        "state_dict": model.state_dict(),
        "temperature": temperature,
        "backbone": backbone,
        "image_size": image_size,
    }
    torch.save(checkpoint, out_path)
    print(f"\nSaved checkpoint to {out_path} (state_dict + temperature={temperature:.4f} + backbone)")


def main():
    parser = argparse.ArgumentParser(description="Deep learning training for glaucoma project")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/dl_baseline.yaml)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = load_config(cfg_path)
    train_from_config(cfg)


if __name__ == "__main__":
    main()

