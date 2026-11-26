import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) without the alpha class weights: (1 - p_t) ** gamma * cross-entropy."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_loss_function(exp_id: str, train_loader=None, device="cuda"):
    """E1a: class-weighted cross-entropy, E1b: focal loss, everything else: plain cross-entropy."""
    if exp_id == "E1a":
        # inverse-frequency weights, in ImageFolder class order (same as the model outputs)
        targets = [label for _, label in train_loader.dataset.samples]
        counts = np.bincount(targets)
        weights = 1.0 / counts
        weights = torch.tensor(weights / weights.sum(), dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    elif exp_id == "E1b":
        return FocalLoss(gamma=2.0)
    else:
        return nn.CrossEntropyLoss()
