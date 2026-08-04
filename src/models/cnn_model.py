"""
PyTorch model definitions for skin lesion classification.

Two options are provided:

1. SimpleCNN     - a small from-scratch convolutional network. Useful for
                   demonstrating CNN fundamentals and as a fast baseline.
2. build_model() - a transfer-learning factory wrapping torchvision's
                   pretrained ResNet18 / EfficientNet-B0 backbones with a
                   new classification head sized for our 7 classes. This
                   is the recommended architecture given the dataset size
                   (~14k training images) and class imbalance.

Both expose `get_target_layer(model, architecture)` so the Grad-CAM module
(src/explainability/gradcam.py) knows which convolutional layer to hook.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
from torchvision import models

from src import config


class SimpleCNN(nn.Module):
    """A compact from-scratch CNN (4 conv blocks + classifier head)."""

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.conv_block1 = conv_block(3, 32)     # 224 -> 112
        self.conv_block2 = conv_block(32, 64)    # 112 -> 56
        self.conv_block3 = conv_block(64, 128)   # 56 -> 28
        self.conv_block4 = conv_block(128, 256)  # 28 -> 14

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)  # last conv feature map -> Grad-CAM target
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _build_resnet18(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_efficientnet_b0(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_model(
    architecture: str = config.DEFAULT_ARCHITECTURE,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Factory returning a ready-to-train PyTorch model.

    Args:
        architecture: 'simplecnn' | 'resnet18' | 'efficientnet_b0'
        pretrained: use ImageNet weights (ignored for simplecnn)
        freeze_backbone: freeze conv layers, only train the new head
                         (useful for quick fine-tuning on small data)
    """
    architecture = architecture.lower()
    if architecture == "simplecnn":
        model = SimpleCNN(num_classes=num_classes)
    elif architecture == "resnet18":
        model = _build_resnet18(num_classes, pretrained, freeze_backbone)
    elif architecture == "efficientnet_b0":
        model = _build_efficientnet_b0(num_classes, pretrained, freeze_backbone)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'")
    return model.to(config.DEVICE)


def count_params(model: nn.Module) -> int:
    """Total number of trainable + non-trainable parameters (used in the
    model-comparison report to weigh accuracy against model size)."""
    return sum(p.numel() for p in model.parameters())


def get_target_layer(model: nn.Module, architecture: str = config.DEFAULT_ARCHITECTURE):
    """Return the nn.Module to hook for Grad-CAM (last conv layer before pooling)."""
    architecture = architecture.lower()
    if architecture == "simplecnn":
        return model.conv_block4
    if architecture == "resnet18":
        return model.layer4[-1]
    if architecture == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unknown architecture '{architecture}'")


if __name__ == "__main__":
    for arch in ["simplecnn", "resnet18", "efficientnet_b0"]:
        m = build_model(architecture=arch, pretrained=False)
        dummy = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
        out = m(dummy)
        n_params = count_params(m)
        print(f"{arch:16s} output={tuple(out.shape)} params={n_params:,} "
              f"target_layer={get_target_layer(m, arch).__class__.__name__}")
