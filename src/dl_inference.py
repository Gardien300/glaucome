"""
High-level inference helpers for the deep learning glaucoma model.

Provides:
- load_model_and_metadata: charge le checkpoint DL.
- predict_with_explanations: prédiction + Grad-CAM.
- predict_with_tta: Test-Time Augmentation pour des prédictions robustes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.special import expit as sigmoid
from torchvision import transforms

from .config import MODELS_DIR
from .dl_data import normalize_fundus
from .dl_models import build_model
from .xai import GradCAM, overlay_heatmap_on_image


def _get_default_checkpoint_path() -> Path:
    candidate = MODELS_DIR / "dl_efficientnet_v2_m_best.pth"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Checkpoint {candidate} introuvable. "
            "Entraîne d'abord le modèle DL avec src.dl_train et configs/dl_highperf.yaml."
        )
    return candidate


def load_model_and_metadata(
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
) -> Tuple[torch.nn.Module, float, Dict]:
    """Charge un modèle DL entraîné + température de calibration + métadonnées."""
    if checkpoint_path is None:
        checkpoint_path = _get_default_checkpoint_path()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt.get("backbone", "efficientnet_v2_m")
    temperature = float(ckpt.get("temperature", 1.0))
    image_size = int(ckpt.get("image_size", 384))

    state_dict = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        prefix = "_orig_mod."
        state_dict = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

    model = build_model(backbone=backbone, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata = {
        "backbone": backbone,
        "image_size": image_size,
        "checkpoint_path": str(checkpoint_path),
    }
    return model, temperature, metadata


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# TTA (Bloc E)
# ---------------------------------------------------------------------------

def _build_tta_transforms(image_size: int) -> List[transforms.Compose]:
    """Build a set of transforms for Test-Time Augmentation."""
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base_resize = transforms.Resize((image_size, image_size))

    return [
        transforms.Compose([base_resize, transforms.ToTensor(), norm]),
        transforms.Compose([base_resize, transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
        transforms.Compose([base_resize, transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), norm]),
        transforms.Compose([
            base_resize,
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
            norm,
        ]),
        transforms.Compose([
            base_resize,
            transforms.RandomRotation(degrees=(270, 270)),
            transforms.ToTensor(),
            norm,
        ]),
    ]


def predict_with_tta(
    model: torch.nn.Module,
    pil_image: Image.Image,
    image_size: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> float:
    """
    TTA: average logits across multiple augmented views, then apply sigmoid.
    Returns calibrated probability.
    """
    if device is None:
        device = next(model.parameters()).device

    processed = normalize_fundus(pil_image)
    tta_tfs = _build_tta_transforms(image_size)

    logits_list = []
    model.eval()
    with torch.no_grad():
        for tf in tta_tfs:
            tensor = tf(processed).unsqueeze(0).to(device)
            logits = model(tensor).squeeze().cpu().item()
            logits_list.append(logits)

    avg_logit = float(np.mean(logits_list))
    return float(sigmoid(avg_logit / temperature))


def evaluate_with_tta(
    model: torch.nn.Module,
    loader,
    image_size: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run TTA evaluation on a full DataLoader.
    Returns (y_prob, y_true).
    Note: slower than standard eval — intended for final holdout assessment.
    """
    if device is None:
        device = next(model.parameters()).device

    tta_tfs = _build_tta_transforms(image_size)
    norm_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_probs = []
    all_labels = []

    model.eval()
    ds = loader.dataset
    with torch.no_grad():
        for idx in range(len(ds)):
            row = ds.df.iloc[idx]
            label = 1 if row["class"] == "RG" else 0
            img_path = ds.root / Path(row["path"])
            with Image.open(img_path) as img:
                img = img.convert("RGB")
            processed = normalize_fundus(img)

            logits_list = []
            for tf in tta_tfs:
                tensor = tf(processed).unsqueeze(0).to(device)
                logit = model(tensor).squeeze().cpu().item()
                logits_list.append(logit)

            avg_logit = float(np.mean(logits_list))
            prob = float(sigmoid(avg_logit / temperature))
            all_probs.append(prob)
            all_labels.append(label)

    return np.array(all_probs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Inference with XAI
# ---------------------------------------------------------------------------

def _get_target_layer_for_gradcam(model: torch.nn.Module, backbone: str):
    backbone = backbone.lower()
    if backbone.startswith("efficientnet"):
        return model.features[-1]
    if backbone.startswith("convnext"):
        return model.features[-1]
    if backbone.startswith("resnet"):
        return model.layer4
    raise ValueError(f"Grad-CAM non supporté pour le backbone '{backbone}'.")


def predict_with_explanations(
    pil_image: Image.Image,
    checkpoint_path: Path | None = None,
    use_tta: bool = False,
) -> Tuple[float, str, np.ndarray]:
    """
    Applique le modèle DL + Grad-CAM sur une image PIL.

    Returns:
        prob_rg: probabilité prédite (calibrée) que l'image soit RG.
        label_str: texte lisible pour le clinicien.
        overlay: image RGB avec heatmap Grad-CAM superposée (uint8).
    """
    model, temperature, meta = load_model_and_metadata(checkpoint_path=checkpoint_path)
    device = next(model.parameters()).device
    image_size = int(meta["image_size"])

    preprocessed_pil = normalize_fundus(pil_image)
    np_img = np.array(preprocessed_pil.resize((image_size, image_size), Image.BILINEAR))

    if use_tta:
        prob_rg = predict_with_tta(model, pil_image, image_size, temperature, device)
    else:
        tf = _build_transform(image_size)
        input_tensor = tf(preprocessed_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
        logits_np = logits.squeeze(1).cpu().numpy()
        prob_rg = float(sigmoid(logits_np / temperature)[0])

    label_str = "Glaucome référable (RG)" if prob_rg >= 0.5 else "Non glaucomateux (NRG)"

    tf = _build_transform(image_size)
    input_tensor = tf(preprocessed_pil).unsqueeze(0).to(device)
    backbone = meta["backbone"]
    target_layer = _get_target_layer_for_gradcam(model, backbone=backbone)
    cam = GradCAM(model, target_layer)
    try:
        heatmap = cam.generate(input_tensor, target_class=1, device=device)
        overlay = overlay_heatmap_on_image(np_img, heatmap, alpha=0.4)
    finally:
        cam.close()

    return prob_rg, label_str, overlay
