# engine/eval.py

import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.metrics import calculate_metrics


def validate(model, loader, criterion, device, autocast_dtype=torch.float16):
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
    auc_probs = []
    auc_targets = []
    positive_f1 = positive_iou = negative_area = 0.0
    positive_count = negative_count = 0
    positive_f1 = positive_iou = negative_area = 0.0
    positive_count = negative_count = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)

        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)

            total_loss += loss.item()
            # A full AUC for every 16-image batch requires sorting millions of pixels.
            # Keep the threshold metrics exact and calculate one representative AUC per epoch.
            batch_metrics = calculate_metrics(outputs, masks, compute_auc=False)
            dims = (1, 2, 3)
            predicted = outputs.float().sigmoid() >= 0.5
            target = masks.bool()
            intersection = (predicted & target).sum(dims)
            predicted_area = predicted.sum(dims)
            target_area = target.sum(dims)
            nonempty = target_area > 0
            positive_f1 += (2 * intersection[nonempty] / (predicted_area + target_area)[nonempty]).sum().item()
            positive_iou += (intersection[nonempty] / (predicted_area + target_area - intersection)[nonempty]).sum().item()
            negative_area += (predicted_area[~nonempty].sum() / masks[0].numel()).item()
            positive_count += nonempty.sum().item()
            negative_count += (~nonempty).sum().item()
            dims = (1, 2, 3)
            predicted = outputs > 0
            target = masks.bool()
            intersection = (predicted & target).sum(dims)
            predicted_area = predicted.sum(dims)
            target_area = target.sum(dims)
            nonempty = target_area > 0
            positive_f1 += (2 * intersection[nonempty] / (predicted_area + target_area)[nonempty]).sum().item()
            positive_iou += (intersection[nonempty] / (predicted_area + target_area - intersection)[nonempty]).sum().item()
            negative_area += (predicted_area[~nonempty].sum() / masks[0].numel()).item()
            positive_count += nonempty.sum().item()
            negative_count += (~nonempty).sum().item()
            
            for k in metrics_sum.keys():
                if k != "auc":
                    metrics_sum[k] += batch_metrics[k]

            flat_probs = torch.sigmoid(outputs).detach().flatten()
            flat_targets = masks.detach().flatten()
            stride = max(1, flat_probs.numel() // 8_000)
            auc_probs.append(flat_probs[::stride].cpu())
            auc_targets.append(flat_targets[::stride].cpu())

            pbar.set_postfix({
                "loss": loss.item()
            })

    avg_loss = total_loss / len(loader)
    avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items() if k != "auc"}
    if positive_count:
        avg_metrics["positive_mask_f1"] = positive_f1 / positive_count
        avg_metrics["positive_mask_iou"] = positive_iou / positive_count
    if negative_count:
        avg_metrics["empty_target_mean_predicted_area"] = negative_area / negative_count
    if positive_count:
        avg_metrics["positive_mask_f1"] = positive_f1 / positive_count
        avg_metrics["positive_mask_iou"] = positive_iou / positive_count
    if negative_count:
        avg_metrics["empty_target_mean_predicted_area"] = negative_area / negative_count
    sampled_probs = torch.cat(auc_probs).numpy()
    sampled_targets = torch.cat(auc_targets).numpy()
    try:
        avg_metrics["auc"] = float(roc_auc_score(sampled_targets, sampled_probs)) if len(set(sampled_targets)) > 1 else 0.5
    except ValueError:
        avg_metrics["auc"] = 0.5

    return avg_loss, avg_metrics
