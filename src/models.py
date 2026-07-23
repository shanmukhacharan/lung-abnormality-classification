"""
Model builders for the three architectures compared in this project:

- Baseline CNN: ResNet50 with all convolutional layers frozen, only the
  final classification layer trained.
- Fine-Tuned CNN: ResNet50 with every layer trainable.
- Vision Transformer: ViT-B16 with every layer trainable.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import vit_b_16

from . import config


def build_baseline_cnn(num_classes: int = 4):
    """ResNet50 with frozen backbone; only the final FC layer is trainable."""
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(config.DEVICE)


def build_finetuned_cnn(num_classes: int = 4):
    """ResNet50 with every layer trainable (full fine-tuning)."""
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(config.DEVICE)


def build_vision_transformer(num_classes: int = 4):
    """ViT-B16 with every layer trainable."""
    model = vit_b_16(pretrained=True)

    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)

    return model.to(config.DEVICE)
