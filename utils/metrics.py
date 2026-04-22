# utils/metrics.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_metrics(logits, targets, threshold=0.5, eps=1e-6, compute_auc=False):
    """
    Calculate comprehensive pixel-wise performance metrics.
    AUC is optional as it is computationally expensive.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Confusion Matrix Components (GPU)
    tp = (preds * targets).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()

    # Performance Indices
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    
    # IoU
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + eps) / (union + eps)

    auc = 0.0
    if compute_auc:
        # AUC Calculation (requires CPU/Numpy)
        probs_flat = probs.view(-1).cpu().detach().numpy()
        targets_flat = targets.view(-1).cpu().detach().numpy()
        try:
            if len(np.unique(targets_flat)) > 1:
                auc = roc_auc_score(targets_flat, probs_flat)
            else:
                auc = 0.5
        except Exception:
            auc = 0.5

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "auc": float(auc),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }
    }
