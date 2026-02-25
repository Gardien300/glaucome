"""
Grad-CAM utilities for explainability on fundus CNN models.

Designed for binary classification models returning a single logit
(NRG vs RG) as in src.dl_models.build_model.

Typical usage (batch size 1):

    from src.dl_models import build_model
    from src.xai import GradCAM, overlay_heatmap_on_image

    model = build_model(backbone="efficientnet_v2_m", pretrained=True)
    cam = GradCAM(model, target_layer=model.features[-1])

    heatmap = cam.generate(input_tensor, target_class=1)
    overlay = overlay_heatmap_on_image(np_image, heatmap)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


class GradCAM:
    """
    Minimal Grad-CAM implementation for a given target layer.

    - Assumes a model that returns a tensor of shape (N, 1)
      with raw logits for the positive class (RG).
    - Works best with batch size 1 for visualisation.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks on the target layer
        self._forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)  # type: ignore[arg-type]

    def _save_activations(self, module, input, output) -> None:  # noqa: D401
        """Forward hook to store activations."""
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:  # noqa: D401
        """Backward hook to store gradients."""
        # grad_output is a tuple; we take gradients w.r.t. the output of the layer
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = 1,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a given input.

        Args:
            input_tensor: tensor of shape (1, C, H, W).
            target_class: index of target class. For binary models with a single
                          logit, use 1 for the positive class (RG).
            device: optional torch.device. If None, inferred from model.

        Returns:
            heatmap: numpy array of shape (H_feat, W_feat) normalised in [0, 1].
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.to(device, non_blocking=True)

        # Forward pass
        logits = self.model(input_tensor)  # (1, 1) expected
        if logits.ndim == 2 and logits.size(1) == 1:
            # Binary logit; we backprop through the positive class score
            score = logits[0, 0]
        else:
            # Multi-logit case: choose target_class index
            if target_class is None:
                raise ValueError("target_class must be provided for multi-logit models.")
            score = logits[0, int(target_class)]

        # Backward pass for the chosen score
        score.backward(retain_graph=True)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # activations: (N, C, H, W), gradients: (N, C, H, W)
        activations = self._activations
        gradients = self._gradients

        # Batch size 1 assumed
        activations = activations[0]  # (C, H, W)
        gradients = gradients[0]  # (C, H, W)

        # Global average pooling on gradients to obtain channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activations
        cam = torch.zeros_like(activations[0])
        for w, a in zip(weights, activations):
            cam += w * a

        cam = torch.relu(cam)

        # Normalise to [0, 1]
        cam_np = cam.cpu().numpy()
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def close(self) -> None:
        """Remove hooks to avoid memory leaks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def build_papilla_mask(
    feat_h: int,
    feat_w: int,
    cx_norm: float,
    cy_norm: float,
    radius_ratio: float = 0.15,
) -> torch.Tensor:
    """
    Build a circular binary mask in feature-map space around the optic disc.

    Args:
        feat_h, feat_w: spatial dimensions of the feature map.
        cx_norm, cy_norm: normalised [0, 1] centre of the optic disc.
        radius_ratio: radius of the mask as a fraction of the feature map.

    Returns:
        mask: (feat_h, feat_w) float tensor with 1 inside the disc, 0 outside.
    """
    Y, X = torch.meshgrid(
        torch.linspace(0, 1, feat_h),
        torch.linspace(0, 1, feat_w),
        indexing="ij",
    )
    dist = torch.sqrt((X - cx_norm) ** 2 + (Y - cy_norm) ** 2)
    return (dist <= radius_ratio).float()


def compute_attention_loss_from_cam(
    cam: torch.Tensor,
    papilla_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Attention regularisation loss: penalise activations outside the papilla
    for RG images, encourage activations inside the papilla.

    Args:
        cam: (B, H_f, W_f) normalised CAM (requires grad).
        papilla_mask: (B, H_f, W_f) binary mask.
        labels: (B,) integer labels (1 = RG).

    Returns:
        Scalar loss (0 if no RG in batch).
    """
    rg_mask = (labels == 1).float()
    if rg_mask.sum() == 0:
        return torch.tensor(0.0, device=cam.device, requires_grad=True)

    inv_mask = 1.0 - papilla_mask
    loss_outside = ((cam * inv_mask) ** 2).mean(dim=(1, 2))
    loss_inside = (((1.0 - cam) * papilla_mask) ** 2).mean(dim=(1, 2))
    att_loss = (rg_mask * (loss_outside + loss_inside)).sum() / rg_mask.sum().clamp(min=1)
    return att_loss


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Superpose une heatmap Grad-CAM (H, W) sur une image RGB (H, W, 3).

    Args:
        image: image RGB en numpy array dans [0, 255] ou [0, 1].
        heatmap: heatmap 2D normalisée dans [0, 1] (taille ≈ feature map).
                 Elle sera redimensionnée aux dimensions de l'image.
        alpha: transparence de la heatmap.

    Returns:
        image_rgb: image RGB avec overlay (uint8).
    """
    import cv2  # type: ignore[import]

    if image.dtype != np.uint8:
        img = image.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.clip(0, 255).astype(np.uint8)
    else:
        img = image.copy()

    h, w = img.shape[:2]

    # Redimensionner la heatmap à la taille de l'image
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h))
    heatmap_color = cv2.applyColorMap((255 * heatmap_resized).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_color.astype(np.float32) + (1 - alpha) * img.astype(np.float32)).clip(0, 255)
    return overlay.astype(np.uint8)

