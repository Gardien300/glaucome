"""
Service d'inférence pour le modèle DL de dépistage du glaucome, pensé pour le web.

Ce module encapsule :
- le chargement du checkpoint (une seule fois par process),
- le prétraitement des images de fond d'œil,
- une fonction de prédiction simple (probabilité RG + classe binaire + métadonnées).

Il réutilise les briques de `src.dl_inference` (chargeur de modèle, transforms, prétraitement).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

from .dl_inference import (
    load_model_and_metadata,
    preprocess_fundus_for_inference,
    _build_transform,  # type: ignore[attr-defined]
    _get_target_layer_for_gradcam,  # type: ignore[attr-defined]
)
from .xai import GradCAM, overlay_heatmap_on_image


@dataclass
class InferenceResult:
    """
    Résultat minimal pour l'API web.
    """

    probability_rg: float
    predicted_class: str  # "RG" ou "NRG"
    threshold: float
    model_version: str


class GlaucomaInferenceService:
    """
    Service d'inférence Stateful, adapté à un process FastAPI.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        # Permettre de surcharger le checkpoint par variable d'env
        env_ckpt = os.environ.get("MODEL_CHECKPOINT_PATH")
        if env_ckpt:
            checkpoint_path = Path(env_ckpt)

        self.model, self.temperature, self.metadata = load_model_and_metadata(
            checkpoint_path=checkpoint_path, device=device
        )
        self.device = next(self.model.parameters()).device

        image_size = int(self.metadata["image_size"])
        self.transform = _build_transform(image_size)
        self.threshold = 0.5  # seuil binaire par défaut

    @property
    def model_version(self) -> str:
        """
        Version de modèle lisible pour les logs (nom de checkpoint).
        """
        ckpt_path = self.metadata.get("checkpoint_path", "")
        return Path(ckpt_path).name if ckpt_path else "unknown"

    def predict(self, pil_image: Image.Image) -> InferenceResult:
        """
        Applique le modèle sur une image PIL et retourne un InferenceResult.
        """
        image_size = int(self.metadata["image_size"])

        # Prétraitement léger cohérent avec l'entraînement
        preprocessed_pil, _ = preprocess_fundus_for_inference(
            pil_image, image_size=image_size, device_id=None
        )

        input_tensor = self.transform(preprocessed_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)

        # Calibrage éventuel par temperature scaling
        logits = logits.squeeze(1) / float(self.temperature)
        prob_rg = float(torch.sigmoid(logits)[0].item())

        predicted_class = "RG" if prob_rg >= self.threshold else "NRG"

        return InferenceResult(
            probability_rg=prob_rg,
            predicted_class=predicted_class,
            threshold=self.threshold,
            model_version=self.model_version,
        )

    def predict_with_xai(self, pil_image: Image.Image):
        """
        Variante de prédiction qui retourne aussi une overlay Grad-CAM (numpy array RGB).
        """
        image_size = int(self.metadata["image_size"])

        # Prétraitement cohérent avec l'entraînement + copie numpy pour XAI
        preprocessed_pil, np_img = preprocess_fundus_for_inference(
            pil_image, image_size=image_size, device_id=None
        )

        input_tensor = self.transform(preprocessed_pil).unsqueeze(0).to(self.device)

        # Prédiction
        with torch.no_grad():
            logits = self.model(input_tensor)

        logits = logits.squeeze(1) / float(self.temperature)
        prob_rg = float(torch.sigmoid(logits)[0].item())
        predicted_class = "RG" if prob_rg >= self.threshold else "NRG"

        inf = InferenceResult(
            probability_rg=prob_rg,
            predicted_class=predicted_class,
            threshold=self.threshold,
            model_version=self.model_version,
        )

        # Grad-CAM (classe positive RG)
        backbone = self.metadata.get("backbone", "efficientnet_v2_m")
        target_layer = _get_target_layer_for_gradcam(self.model, backbone=backbone)
        cam = GradCAM(self.model, target_layer)
        try:
            heatmap = cam.generate(input_tensor, target_class=1, device=self.device)
            overlay = overlay_heatmap_on_image(np_img, heatmap, alpha=0.4)
        finally:
            cam.close()

        return inf, overlay


@lru_cache(maxsize=1)
def get_inference_service() -> GlaucomaInferenceService:
    """
    Retourne un singleton du service d'inférence (chargé une fois par process).
    """

    return GlaucomaInferenceService()

