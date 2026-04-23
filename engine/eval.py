# engine/eval.py

import torch
from tqdm import tqdm
from utils.metrics import calculate_metrics


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    metrics_sum = {
        "iou": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
        "auc": 0.0
    }

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)

        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            batch_metrics = calculate_metrics(outputs, masks)
            
            for k in metrics_sum.keys():
                metrics_sum[k] += batch_metrics[k]

            pbar.set_postfix({
                "loss": loss.item()
            })

    avg_loss = total_loss / len(loader)
    avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics