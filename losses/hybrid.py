# losses/hybrid.py

import torch
import torch.nn as nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, label_smoothing=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0])) # Can be adjusted
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing manually for BCE if needed
        # Or use pos_weight/weight if class imbalance is severe
        
        # Binary label smoothing:
        # y_smoothed = y * (1 - alpha) + 0.5 * alpha
        smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, smoothed_targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, gamma=2.0, alpha=0.75):
        super().__init__()
        if not 0 <= focal_weight <= 1 or not 0 < alpha < 1 or not 0 <= gamma < float("inf"):
            raise ValueError("Expected focal weight in [0, 1], alpha in (0, 1), and finite nonnegative gamma")
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.alpha = alpha
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        error_probability = (-torch.expm1(-bce)).clamp_min(torch.finfo(bce.dtype).tiny)
        class_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = (class_weight * error_probability.pow(self.gamma) * bce).mean()
        return self.focal_weight * focal + (1 - self.focal_weight) * self.dice(logits, targets)
