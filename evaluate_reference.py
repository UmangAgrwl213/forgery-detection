"""Score external probability maps on the same frozen CASIA protocol."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from data.casia import CASIADataset, read_split_manifest
from data.transforms import get_val_transforms
from evaluate_casia import PIXEL_THRESHOLD, image_threshold, summarize


def reference_counts(dataset, rows, maps, size, map_format="npy"):
    counts = []
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        suffix = "npz" if map_format == "trufor" else "npy"
        path = maps / f"{row['image']}.{suffix}"
        contents = path.read_bytes()
        digest.update(hashlib.sha256(contents).digest())
        with path.open("rb") as file:
            if map_format == "trufor":
                with np.load(file, allow_pickle=False) as archive:
                    probabilities = archive["map"]
            else:
                probabilities = np.load(file, allow_pickle=False)
        if probabilities.ndim != 2 or probabilities.size == 0 or not np.issubdtype(probabilities.dtype, np.floating):
            raise ValueError(f"Expected a two-dimensional floating-point probability map: {path}")
        if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
            raise ValueError(f"Invalid probability range: {path}")
        predicted = cv2.resize(probabilities.astype(np.float32), (size, size)) >= PIXEL_THRESHOLD
        _, mask = dataset[index]
        target = mask[0].numpy().astype(bool)
        counts.append(((predicted & target).sum(), predicted.sum(), target.sum()))
    return np.asarray(counts), digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True, help="Directory containing Au/ and Tp/ probability maps")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--map-format", choices=("npy", "trufor"), default="npy")
    parser.add_argument("--training-overlap", choices=("unknown", "possible", "verified-disjoint"), default="unknown")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/CASIA2"))
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()
    if args.image_size < 1 or args.output.exists():
        parser.error("Use a positive image size and a fresh output path")
    splits = read_split_manifest(args.split_manifest)
    size = args.image_size
    report = {"model": args.model_name, "map_format": args.map_format, "training_overlap": args.training_overlap,
              "image_size": size, "pixel_threshold": PIXEL_THRESHOLD,
              "split_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(), "maps_sha256": {},
              "max_validation_fpr": 0.05, "low_false_alarm_operating_point": {}}
    low = report["low_false_alarm_operating_point"]
    for part in ("validation", "test"):
        rows = splits[part]
        labels = np.array([row["mask"] is not None for row in rows])
        if not labels.any() or labels.all():
            parser.error(f"Both classes are required in {part}")
        dataset = CASIADataset(args.dataset_root, args.dataset_root / "CASIA 2 Groundtruth",
                               get_val_transforms(size), records=rows)
        counts, report["maps_sha256"][part] = reference_counts(dataset, rows, args.maps, size, args.map_format)
        if part == "validation":
            scores = counts[:, 1] / size ** 2
            report["image_threshold"] = image_threshold(labels, scores)
            low["image_threshold"] = image_threshold(labels, scores, report["max_validation_fpr"])
        report[part] = summarize(counts, labels, size ** 2, report["image_threshold"])
        low[part] = summarize(counts, labels, size ** 2, low["image_threshold"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
