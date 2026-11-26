import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes: int = 15, freeze_backbone: bool = False, dropout: float = 0.0):
    """ImageNet-pretrained ResNet-18 with a new classification head.

    The defaults train all layers with a plain linear head (the E0 baseline).
    freeze_backbone=True trains only the new head; dropout > 0 adds dropout before the head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model
