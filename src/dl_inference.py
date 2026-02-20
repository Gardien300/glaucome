"""
High-level inference helpers for the deep learning glaucoma model.

Provides:
- load_model_and_metadata: charge le checkpoint DL (state_dict + T + backbone + image_size)
- predict_with_explanations: prend une image PIL, renvoie proba RG, label texte, overlay Grad-CAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.special import expit as sigmoid
from torchvision import transforms

from .config import PROJECT_ROOT, MODELS_DIR
from .dl_models import build_model
from .xai import GradCAM, overlay_heatmap_on_image


def _get_default_checkpoint_path() -> Path:
    models_dir = MODELS_DIR
    # Par défaut, on suppose le modèle produit efficientnet_v2_m
    candidate = models_dir / "dl_efficientnet_v2_m_best.pth"
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
    """
    Charge un modèle DL entraîné + température de calibration + métadonnées.

    Args:
        checkpoint_path: chemin du .pth sauvegardé par src.dl_train.
        device: device torch (cpu/cuda). Si None, déduit automatiquement.
    """
    if checkpoint_path is None:
        checkpoint_path = _get_default_checkpoint_path()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt.get("backbone", "efficientnet_v2_m")
    temperature = float(ckpt.get("temperature", 1.0))
    image_size = int(ckpt.get("image_size", 384))

    model = build_model(backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "backbone": backbone,
        "image_size": image_size,
        "checkpoint_path": str(checkpoint_path),
    }
    return model, temperature, metadata


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocess_fundus_for_inference(
    pil_image: Image.Image,
    image_size: int,
    device_id: str | None = None,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Prétraitement léger pour rapprocher l'image du domaine d'entraînement.

    Étapes :
    - conversion en RGB
    - crop centré carré (si nécessaire)
    - correction de contraste douce (CLAHE sur la luminance)
    - resize à image_size

    Args:
        pil_image: image brute fournie par le praticien.
        image_size: taille cible (côté) du modèle DL.
        device_id: identifiant optionnel du rétinographe (réservé pour ajustements futurs).

    Returns:
        preprocessed_pil: image PIL prête pour la transform torchvision.
        np_image_rgb: version numpy RGB (avant normalisation) pour XAI.
    """
    import cv2  # type: ignore[import]

    img = pil_image.convert("RGB")

    # Crop centré carré (pour limiter les formes trop exotiques)
    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))

    # Correction de contraste douce via CLAHE sur la luminance (espace LAB)
    np_img = np.array(img)
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # Resize final à image_size × image_size
    rgb_eq_pil = Image.fromarray(rgb_eq)
    rgb_eq_pil = rgb_eq_pil.resize((image_size, image_size), Image.BILINEAR)

    return rgb_eq_pil, np.array(rgb_eq_pil)


def _get_target_layer_for_gradcam(model: torch.nn.Module, backbone: str):
    """
    Retourne la couche cible à utiliser pour Grad-CAM selon le backbone.
    """
    backbone = backbone.lower()
    if backbone.startswith("efficientnet"):
        return model.features[-1]
    if backbone.startswith("convnext"):
        return model.features[-1]
    if backbone.startswith("resnet"):
        return model.layer4
    # ViT ou autres: Grad-CAM non supporté dans cette implémentation minimale
    raise ValueError(f"Grad-CAM non supporté pour le backbone '{backbone}'.")


def predict_with_explanations(
    pil_image: Image.Image,
    checkpoint_path: Path | None = None,
) -> Tuple[float, str, np.ndarray]:
    """
    Applique le modèle DL + Grad-CAM sur une image PIL.

    Retourne:
        prob_rg: probabilité prédite (calibrée) que l'image soit RG.
        label_str: texte lisible pour le clinicien.
        overlay: image RGB avec heatmap Grad-CAM superposée (uint8).
    """
    model, temperature, meta = load_model_and_metadata(checkpoint_path=checkpoint_path)
    device = next(model.parameters()).device

    image_size = int(meta["image_size"])
    tf = _build_transform(image_size)

    # Prétraitement léger + copie numpy pour l'overlay XAI
    preprocessed_pil, np_img = preprocess_fundus_for_inference(
        pil_image, image_size=image_size, device_id=None
    )

    input_tensor = tf(preprocessed_pil).unsqueeze(0).to(device)

    # Prédiction
    with torch.no_grad():
        logits = model(input_tensor)
    logits_np = logits.squeeze(1).cpu().numpy()
    prob_rg = float(sigmoid(logits_np / temperature)[0])

    label_str = "Glaucome référable (RG)" if prob_rg >= 0.5 else "Non glaucomateux (NRG)"

    # Grad-CAM
    backbone = meta["backbone"]
    target_layer = _get_target_layer_for_gradcam(model, backbone=backbone)
    cam = GradCAM(model, target_layer)
    try:
        heatmap = cam.generate(input_tensor, target_class=1, device=device)
        overlay = overlay_heatmap_on_image(np_img, heatmap, alpha=0.4)
    finally:
        cam.close()

    return prob_rg, label_str, overlay

