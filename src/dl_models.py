"""
Deep learning model definitions for the glaucoma project (PyTorch).

We use torchvision backbones (ResNet, EfficientNet, EfficientNetV2, ConvNeXt, ViT)
and replace the final classification head to output a single logit (binary
classification NRG vs RG, with BCEWithLogitsLoss).

Supports single-branch (standard) and dual-branch (global + papilla crop) models.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn
from PIL import Image


BackboneName = Literal[
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b4",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "convnext_tiny",
    "convnext_small",
    "vit_b_16",
]


def _replace_classification_head(
    backbone: nn.Module,
    num_features: int,
) -> nn.Module:
    """
    Replace the classification head of a backbone by a single-logit head.
    """
    head = nn.Linear(num_features, 1)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def get_backbone(
    name: BackboneName = "efficientnet_b0",
    pretrained: bool = True,
) -> nn.Module:
    """
    Create a torchvision backbone with an output head adapted for binary
    classification (one logit).

    Args:
        name: one of 'resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b4',
              'efficientnet_v2_s', 'efficientnet_v2_m', 'convnext_tiny', 'convnext_small',
              'vit_b_16'.
        pretrained: whether to use ImageNet-pretrained weights.
    """
    import torchvision.models as models

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_f = model.fc.in_features
        model.fc = _replace_classification_head(model, num_f)
        return model

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        num_f = model.fc.in_features
        model.fc = _replace_classification_head(model, num_f)
        return model

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_f = model.classifier[1].in_features
        model.classifier[1] = _replace_classification_head(model, num_f)
        return model

    if name == "efficientnet_b4":
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_f = model.classifier[1].in_features
        model.classifier[1] = _replace_classification_head(model, num_f)
        return model

    if name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_f = model.classifier[1].in_features
        model.classifier[1] = _replace_classification_head(model, num_f)
        return model

    if name == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_f = model.classifier[1].in_features
        model.classifier[1] = _replace_classification_head(model, num_f)
        return model

    if name == "efficientnet_v2_l":
        model = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_f = model.classifier[1].in_features
        model.classifier[1] = _replace_classification_head(model, num_f)
        return model

    if name == "convnext_tiny":
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_f = model.classifier[2].in_features
        model.classifier[2] = _replace_classification_head(model, num_f)
        return model

    if name == "convnext_small":
        model = models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_f = model.classifier[2].in_features
        model.classifier[2] = _replace_classification_head(model, num_f)
        return model

    if name == "vit_b_16":
        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_f = model.heads.head.in_features
        model.heads.head = _replace_classification_head(model, num_f)
        return model

    raise ValueError(f"Unknown backbone name: {name}")


def build_model(
    backbone: BackboneName = "efficientnet_b0",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    High-level helper that returns a complete model ready for training.

    Args:
        backbone: backbone architecture to use.
        pretrained: whether to load ImageNet-pretrained weights.
        freeze_backbone: if True, only the final head is trainable.
    """
    model = get_backbone(backbone, pretrained=pretrained)

    if freeze_backbone:
        for name, param in model.named_parameters():
            # keep head trainable (usually named 'fc' or 'classifier' or 'heads')
            if any(k in name for k in ["fc.", "classifier.", "heads."]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Optic disc detection & cropping (Bloc B)
# ---------------------------------------------------------------------------

def detect_optic_disc_center(pil_image: Image.Image) -> Tuple[int, int]:
    """
    Heuristic detection of the optic disc centre via luminance peak.
    Returns (cx, cy) in pixel coordinates.
    """
    import cv2

    gray = np.array(pil_image.convert("L"))
    h, w = gray.shape

    margin_x = int(w * 0.15)
    margin_y = int(h * 0.10)
    roi = gray[margin_y : h - margin_y, margin_x : w - margin_x]

    sigma = max(w // 10, 15)
    ksize = sigma * 2 + 1
    blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigmaX=sigma)

    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    cx = max_loc[0] + margin_x
    cy = max_loc[1] + margin_y
    return cx, cy


def crop_optic_disc(
    pil_image: Image.Image,
    cx: int,
    cy: int,
    crop_ratio: float = 0.22,
) -> Image.Image:
    """Extract a square crop around the estimated optic disc centre."""
    w, h = pil_image.size
    radius = int(crop_ratio * min(w, h))
    left = max(0, cx - radius)
    top = max(0, cy - radius)
    right = min(w, cx + radius)
    bottom = min(h, cy + radius)
    return pil_image.crop((left, top, right, bottom))


# ---------------------------------------------------------------------------
# Feature-extractor wrapper (removes the classification head)
# ---------------------------------------------------------------------------

def _get_feature_dim(backbone_model: nn.Module) -> int:
    """Return the feature dimension of a backbone before the classification head."""
    if hasattr(backbone_model, "classifier"):
        head = backbone_model.classifier
        if isinstance(head, nn.Sequential):
            for layer in reversed(list(head.children())):
                if isinstance(layer, nn.Linear):
                    return layer.in_features
        elif isinstance(head, nn.Linear):
            return head.in_features
    if hasattr(backbone_model, "fc") and isinstance(backbone_model.fc, nn.Linear):
        return backbone_model.fc.in_features
    if hasattr(backbone_model, "heads"):
        head = backbone_model.heads
        if hasattr(head, "head") and isinstance(head.head, nn.Linear):
            return head.head.in_features
    raise ValueError("Cannot determine feature dim for backbone.")


class _FeatureExtractor(nn.Module):
    """Wraps a classification backbone to return pooled features instead of logits."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self._remove_head()

    def _remove_head(self) -> None:
        if hasattr(self.backbone, "classifier"):
            head = self.backbone.classifier
            if isinstance(head, nn.Sequential):
                for i in range(len(head) - 1, -1, -1):
                    if isinstance(head[i], nn.Linear):
                        head[i] = nn.Identity()
                        break
            elif isinstance(head, nn.Linear):
                self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "heads"):
            if hasattr(self.backbone.heads, "head"):
                self.backbone.heads.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DualBranchModel(nn.Module):
    """
    Two-branch architecture: global fundus image + optic-disc crop.
    Feature vectors are concatenated and passed through a small fusion MLP.
    """

    def __init__(
        self,
        global_backbone: BackboneName = "efficientnet_v2_m",
        papilla_backbone: BackboneName = "efficientnet_v2_s",
        pretrained: bool = True,
        fusion_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        global_raw = get_backbone(global_backbone, pretrained=pretrained)
        papilla_raw = get_backbone(papilla_backbone, pretrained=pretrained)

        self.d_global = _get_feature_dim(global_raw)
        self.d_papilla = _get_feature_dim(papilla_raw)

        self.global_branch = _FeatureExtractor(global_raw)
        self.papilla_branch = _FeatureExtractor(papilla_raw)

        self.fusion = nn.Sequential(
            nn.Linear(self.d_global + self.d_papilla, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )
        nn.init.xavier_uniform_(self.fusion[0].weight)
        nn.init.xavier_uniform_(self.fusion[3].weight)

    def forward(
        self, x_global: torch.Tensor, x_papilla: torch.Tensor
    ) -> torch.Tensor:
        f_g = self.global_branch(x_global)
        f_p = self.papilla_branch(x_papilla)
        f = torch.cat([f_g, f_p], dim=1)
        return self.fusion(f)


def build_dual_branch_model(
    global_backbone: BackboneName = "efficientnet_v2_m",
    papilla_backbone: BackboneName = "efficientnet_v2_s",
    pretrained: bool = True,
    fusion_dim: int = 512,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> DualBranchModel:
    model = DualBranchModel(
        global_backbone=global_backbone,
        papilla_backbone=papilla_backbone,
        pretrained=pretrained,
        fusion_dim=fusion_dim,
        dropout=dropout,
    )
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fusion" not in name:
                param.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Block groups for phased fine-tuning (Bloc D)
# ---------------------------------------------------------------------------

def get_backbone_block_groups(
    model: nn.Module,
) -> List[List[nn.Parameter]]:
    """
    Split backbone parameters into groups from deepest to shallowest.
    Returns a list of param lists: [deep_params, ..., shallow_params, head_params].
    """
    head_keywords = ("fc.", "classifier.", "heads.", "fusion.")

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        blocks = list(model.features.children())
    elif hasattr(model, "layer1"):
        blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    else:
        blocks = []

    groups: List[List[nn.Parameter]] = []
    for block in blocks:
        params = [p for p in block.parameters() if p.requires_grad]
        if params:
            groups.append(params)

    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(k in n for k in head_keywords)
    ]
    if head_params:
        groups.append(head_params)

    return groups


def create_discriminative_param_groups(
    model: nn.Module,
    lr_head: float = 1e-3,
    lr_decay: float = 0.3,
    weight_decay: float = 1e-4,
) -> List[Dict]:
    """
    Create optimizer param groups with discriminative (layer-wise) learning rates.
    Deeper layers get exponentially smaller LR.
    """
    block_groups = get_backbone_block_groups(model)
    n = len(block_groups)
    param_groups = []
    for i, params in enumerate(block_groups):
        lr = lr_head * (lr_decay ** max(0, n - 1 - i))
        param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})
    return param_groups


