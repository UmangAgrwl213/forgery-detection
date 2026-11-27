"""Evaluate a frozen-split CASIA checkpoint and save a benchmark report."""

import argparse
import hashlib
import json
from pathlib import Path

from main import make_loader

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from data.casia import CASIADataset, read_split_manifest
from data.transforms import get_val_transforms
from predict import load_model
from utils.benchmark_stats import binomial_summary


PIXEL_THRESHOLD = 0.5
BOUNDARY_RADIUS = 2


def boundary_f1(predicted, target, radius=BOUNDARY_RADIUS):
    def edges(mask):
        outside = F.pad((~mask).float(), (1, 1, 1, 1), value=1)
        return mask & F.max_pool2d(outside, 3, stride=1).bool()

    def nearby(edge):
        return F.max_pool2d(edge.float(), 2 * radius + 1, stride=1, padding=radius).bool()

    prediction_edge, target_edge = edges(predicted), edges(target)
    dims = (1, 2, 3)
    precision = (prediction_edge & nearby(target_edge)).sum(dims) / prediction_edge.sum(dims).clamp_min(1)
    recall = (target_edge & nearby(prediction_edge)).sum(dims) / target_edge.sum(dims).clamp_min(1)
    return 2 * precision * recall / (precision + recall).clamp_min(1e-8)


def image_threshold(labels, scores, max_fpr=None):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    # Prefer the more conservative threshold when balanced accuracy ties.
    if max_fpr is None:
        index = np.argmax(tpr - fpr)
    else:
        eligible = np.flatnonzero(fpr <= max_fpr)
        index = eligible[np.argmax(tpr[eligible])]
    return float(min(thresholds[index], 2.0))


@torch.inference_mode()
def collect(model, loader, device, *, include_boundary=False):
    results = []
    boundaries = []
    for images, masks in loader:
        predicted = model(images.to(device)).sigmoid() >= PIXEL_THRESHOLD
        target = masks.to(device).bool()
        dims = (1, 2, 3)
        result = torch.stack(((predicted & target).sum(dims), predicted.sum(dims), target.sum(dims)), dim=1)
        results.append(result.cpu().numpy())
        if include_boundary:
            nonempty = target.any(dim=dims)
            boundaries.append(boundary_f1(predicted[nonempty], target[nonempty]).cpu().numpy())
    counts = np.concatenate(results, axis=0)
    return (counts, np.concatenate(boundaries)) if include_boundary else counts


def summarize(counts, labels, pixels, threshold):
    intersection, predicted, target = counts.T
    scores = predicted / pixels
    calls = scores >= threshold
    positives = target > 0
    if not positives.any():
        raise ValueError("Test masks contain no positive pixels")
    fpr = float(calls[~labels].mean())
    recall = float(calls[labels].mean())
    return {
        "images": len(labels), "tampered_images": int(labels.sum()),
        "nonempty_test_masks": int(positives.sum()),
        "pixel_f1_mean_positive_mask": float(np.mean(2 * intersection[positives] / (predicted + target)[positives])),
        "pixel_iou_mean_positive_mask": float(np.mean(intersection[positives] / (predicted + target - intersection)[positives])),
        "pixel_accuracy": float(1 - (predicted + target - 2 * intersection).sum() / (pixels * len(labels))),
        "image_accuracy": float(np.mean(calls == labels)),
        "image_balanced_accuracy": (recall + 1 - fpr) / 2,
        "image_roc_auc": float(roc_auc_score(labels, scores)),
        "image_average_precision": float(average_precision_score(labels, scores)),
        "authentic_false_positive_rate": fpr, "tampered_recall": recall,
        "false_positive_uncertainty": binomial_summary(int(calls[~labels].sum()), int((~labels).sum())),
        "recall_uncertainty": binomial_summary(int(calls[labels].sum()), int(labels.sum())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/CASIA2"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-validation-fpr", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.max_validation_fpr < 1:
        parser.error("--max-validation-fpr must be between 0 (inclusive) and 1 (exclusive)")
    metadata = json.loads((args.run_dir / "run_metadata.json").read_bytes())
    history = json.loads((args.run_dir / "log.json").read_bytes())["val_loss"]
    if len(history) != metadata["epochs"] or not np.isfinite(history).all():
        parser.error("Held-out evaluation requires a completed run with finite validation losses")
    digest = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    if digest != metadata["split_sha256"]:
        parser.error("The manifest does not match the run's recorded split")
    splits = read_split_manifest(args.split_manifest)
    report_path = args.run_dir / "heldout_eval.json"
    if report_path.exists():
        parser.error(f"Report already exists: {report_path}")
    weights = args.run_dir / "best_model.pth"
    size = metadata["image_size"]
    device = torch.device(args.device)
    model = load_model(weights, device)
    report = {"split_sha256": digest, "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "image_size": size, "pixel_threshold": PIXEL_THRESHOLD, "image_score": "predicted mask area fraction",
              "selected_epoch": int(np.argmin(history)) + 1, "max_validation_fpr": args.max_validation_fpr,
              "low_false_alarm_operating_point": {}, "boundary_tolerance_model_pixels": BOUNDARY_RADIUS}
    for part in ("validation", "test"):
        rows = splits[part]
        labels = np.array([row["mask"] is not None for row in rows])
        if not labels.any() or labels.all():
            parser.error(f"Both classes are required in {part}")
        dataset = CASIADataset(args.dataset_root, args.dataset_root / "CASIA 2 Groundtruth",
                               get_val_transforms(size), records=rows)
        loader = make_loader(dataset, 8, False, args.workers, 2, device.type == "cuda")
        counts, boundaries = collect(model, loader, device, include_boundary=True)
        if part == "validation":
            report["image_threshold"] = image_threshold(labels, counts[:, 1] / size ** 2)
            report["low_false_alarm_operating_point"]["image_threshold"] = image_threshold(
                labels, counts[:, 1] / size ** 2, args.max_validation_fpr
            )
        report[part] = summarize(counts, labels, size ** 2, report["image_threshold"])
        report[part]["pixel_boundary_f1_mean_positive_mask"] = float(boundaries.mean())
        low_false_alarm = report["low_false_alarm_operating_point"]
        low_false_alarm[part] = summarize(counts, labels, size ** 2, low_false_alarm["image_threshold"])
        del loader
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(report_path)


if __name__ == "__main__":
    main()
