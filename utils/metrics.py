# utils/metrics.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_metrics(logits, targets, threshold=0.5, eps=1e-6):
    """
    Calculate comprehensive pixel-wise performance metrics, including AUC.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Flatten for global calculation
    probs_flat = probs.view(-1).cpu().detach().numpy()
    preds_flat = preds.view(-1).cpu().detach().numpy()
    targets_flat = targets.view(-1).cpu().detach().numpy()

    # Confusion Matrix Components (using numpy for consistency here)
    tp = (preds_flat * targets_flat).sum()
    fp = ((preds_flat == 1) & (targets_flat == 0)).sum()
    fn = ((preds_flat == 0) & (targets_flat == 1)).sum()
    tn = ((preds_flat == 0) & (targets_flat == 0)).sum()

    # Performance Indices
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    
    # IoU
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + eps) / (union + eps)

    # AUC Calculation (only if both classes are present in targets)
    try:
        if len(np.unique(targets_flat)) > 1:
            auc = roc_auc_score(targets_flat, probs_flat)
        else:
            auc = 0.5 # Default for single class
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