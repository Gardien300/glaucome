"""
Deep learning model definitions for the glaucoma project (PyTorch).

We use torchvision backbones (ResNet, EfficientNet, EfficientNetV2, ConvNeXt, ViT)
and replace the final classification head to output a single logit (binary
classification NRG vs RG, with BCEWithLogitsLoss).

Recommandation par défaut: efficientnet_b0 (voir docs/RECO_MODELES_CNN.md).
"""

from typing import Literal

import torch
from torch import nn


BackboneName = Literal[
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b4",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
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
    """
    Utility function to count the number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


