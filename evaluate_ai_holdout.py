"""Evaluate a selected AI checkpoint on a new generator without retuning its cutoff."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ai_detector import Images, build_model, scores, test_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory with real/ and fake/<generator>/")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    selection = json.loads((args.run_dir / "validation_eval.json").read_bytes())
    known = {name for part in ("train", "validation") for name in selection["generators"][part]}
    clean = Images(args.dataset)
    if clean.generators & known:
        parser.error("The holdout contains a generator used for training or validation")
    checkpoint = args.run_dir / "best_ai_model.pth"
    with checkpoint.open("rb") as file:
        checkpoint_hash = hashlib.file_digest(file, "sha256").hexdigest()
    if selection.get("checkpoint_sha256", checkpoint_hash) != checkpoint_hash:
        parser.error("The checkpoint hash does not match the validation selection")
    device = torch.device(args.device)
    selected = torch.load(checkpoint, map_location=device, weights_only=True)
    if selected["epoch"] != selection["epoch"] or selected["threshold"] != selection["threshold"]:
        parser.error("The checkpoint does not match the validation selection")
    model = build_model().to(device)
    model.load_state_dict(selected["weights"])
    threshold = selected["threshold"]
    report = {"checkpoint_sha256": checkpoint_hash,
              "selected_epoch": selected["epoch"], "threshold": threshold,
              "generators": sorted(clean.generators)}
    manifest = args.dataset / "manifest.json"
    if manifest.is_file():
        raw = manifest.read_bytes()
        listed = [row["output"] for row in json.loads(raw)]
        actual = sorted(path.relative_to(args.dataset).as_posix() for path, _ in clean.records)
        if sorted(listed) != actual:
            parser.error("Holdout files do not match the recorded manifest")
        report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    for name, data in (("test", clean), ("jpeg_70", Images(args.dataset, jpeg_quality=70))):
        labels, predictions = scores(model, DataLoader(data, batch_size=32), device)
        report[name] = test_metrics(labels, predictions, threshold, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
