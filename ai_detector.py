"""Train and evaluate an image-level detector with generator-disjoint splits."""

import argparse
import hashlib
import json
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop

from utils.benchmark_stats import binomial_summary


TRANSFORM = ResNet34_Weights.DEFAULT.transforms()
TRAIN_CROP = RandomResizedCrop(256, scale=(0.8, 1.0))
TRAIN_FLIP = RandomHorizontalFlip()


def build_model(pretrained=False):
    model = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


@torch.inference_mode()
def classify(image, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model().eval()
    model.load_state_dict(state["weights"])
    score = model(TRANSFORM(default_loader(image)).unsqueeze(0)).sigmoid().item()
    return score, state["threshold"]


class Images(Dataset):
    def __init__(self, directory, jpeg_quality=None, augment=False):
        self.records = []
        self.generators = set()
        self.sample_generators = []
        self.jpeg_quality = jpeg_quality
        self.augment = augment
        for label, kind in enumerate(("real", "fake")):
            paths = sorted(p for p in (directory / kind).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
            if not paths:
                raise ValueError(f"No images in {directory / kind}")
            for path in paths:
                group = path.relative_to(directory / kind).parts
                if label and len(group) < 2:
                    raise ValueError(f"Place synthetic images under fake/<generator>: {path}")
                if label:
                    self.generators.add(group[0].casefold())
                self.records.append((path, label))
                self.sample_generators.append(group[0].casefold() if label else None)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        image = default_loader(path)
        quality = self.jpeg_quality
        if self.augment:
            image = TRAIN_FLIP(TRAIN_CROP(image))
            if random.random() < 0.5:
                quality = random.randint(60, 95)
        if quality is None:
            return TRANSFORM(image), float(label)
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                return TRANSFORM(compressed.convert("RGB")), float(label)


@torch.inference_mode()
def scores(model, loader, device):
    model.eval()
    labels, predictions = [], []
    for images, targets in loader:
        labels.extend(targets.tolist())
        predictions.extend(model(images.to(device)).sigmoid().flatten().tolist())
    return np.array(labels, dtype=bool), np.array(predictions)


def metrics(labels, predictions, threshold):
    calls = predictions >= threshold
    return dict(images=len(labels), accuracy=float((calls == labels).mean()),
                balanced_accuracy=float(balanced_accuracy_score(labels, calls)),
                roc_auc=float(roc_auc_score(labels, predictions)),
                authentic_false_positive_rate=float(calls[~labels].mean()),
                generated_recall=float(calls[labels].mean()),
                authentic_images=int((~labels).sum()),
                false_positive_uncertainty=binomial_summary(int(calls[~labels].sum()), int((~labels).sum())),
                recall_uncertainty=binomial_summary(int(calls[labels].sum()), int(labels.sum())))


def select_thresholds(labels, predictions, max_fpr):
    fpr, recall, thresholds = roc_curve(labels, predictions, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= max_fpr)
    # ROC thresholds are descending, so ties favor the higher threshold.
    constrained = eligible[np.argmax(recall[eligible])]
    balanced = np.argmax(recall - fpr)
    return float(min(thresholds[balanced], 2.0)), float(min(thresholds[constrained], 2.0))


def test_metrics(labels, predictions, threshold, dataset):
    report = metrics(labels, predictions, threshold)
    groups = np.array(dataset.sample_generators, dtype=object)
    report["by_generator"] = {}
    for name in sorted(dataset.generators):
        selected = groups == name
        report["by_generator"][name] = dict(images=int(selected.sum()),
                                              recall=float((predictions[selected] >= threshold).mean()),
                                              uncertainty=binomial_summary(int((predictions[selected] >= threshold).sum()),
                                                                           int(selected.sum())))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--max-validation-fpr", type=float, default=0.05)
    parser.add_argument("--test-jpeg-quality", type=int, nargs="*", default=[70])
    parser.add_argument("--augment", action="store_true", help="Apply the same random crop, flip and JPEG policy to both training classes")
    parser.add_argument("--validation-only", action="store_true", help="Compare a candidate without loading the test partition")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not 0 <= args.max_validation_fpr < 1:
        parser.error("--max-validation-fpr must be between 0 (inclusive) and 1 (exclusive)")
    if any(not 1 <= quality <= 95 for quality in args.test_jpeg_quality):
        parser.error("JPEG quality must be between 1 and 95")
    torch.manual_seed(42)
    random.seed(42)
    parts = ("train", "validation") if args.validation_only else ("train", "validation", "test")
    sets = {part: Images(args.dataset / part, augment=args.augment and part == "train") for part in parts}
    generators = [data.generators for data in sets.values()]
    if len(set.union(*generators)) != sum(map(len, generators)):
        parser.error("Fake-image generator folders must be disjoint across splits")
    args.output.mkdir(parents=True, exist_ok=False)
    loaders = {part: DataLoader(data, batch_size=32, shuffle=part == "train") for part, data in sets.items()}
    device = torch.device(args.device)
    torch.hub.set_dir(str(Path("outputs/torch_cache").resolve()))
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    positive = sum(label for _, label in sets["train"].records)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(sets["train"]) - positive) / positive, device=device))
    best = -1.0
    checkpoint = args.output / "best_ai_model.pth"
    with (args.output / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for images, labels in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(images.to(device)).flatten(), labels.to(device, dtype=torch.float32))
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
            labels, predictions = scores(model, loaders["validation"], device)
            balanced_threshold, threshold = select_thresholds(labels, predictions, args.max_validation_fpr)
            result = metrics(labels, predictions, threshold)
            row = json.dumps(dict(epoch=epoch, train_loss=loss_sum / len(sets["train"]), threshold=threshold, **result))
            print(row, flush=True)
            log.write(row + "\n")
            log.flush()
            if result["roc_auc"] > best:
                best = result["roc_auc"]
                torch.save(dict(weights=model.state_dict(), threshold=threshold, epoch=epoch, validation=result,
                                balanced_threshold=balanced_threshold,
                                balanced_validation=metrics(labels, predictions, balanced_threshold)), checkpoint)
    selected = torch.load(checkpoint, map_location=device, weights_only=True)
    if args.validation_only:
        with checkpoint.open("rb") as file:
            checkpoint_hash = hashlib.file_digest(file, "sha256").hexdigest()
        report = dict(epoch=selected["epoch"], threshold=selected["threshold"], validation=selected["validation"],
                      max_validation_fpr=args.max_validation_fpr, training_augmentation=args.augment,
                      checkpoint_sha256=checkpoint_hash,
                      generators={part: sorted(data.generators) for part, data in sets.items()})
        (args.output / "validation_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    model.load_state_dict(selected["weights"])
    labels, predictions = scores(model, loaders["test"], device)
    report = dict(epoch=selected["epoch"], threshold=selected["threshold"], validation=selected["validation"],
                  test=test_metrics(labels, predictions, selected["threshold"], sets["test"]),
                  max_validation_fpr=args.max_validation_fpr, training_augmentation=args.augment,
                  balanced_operating_point=dict(threshold=selected["balanced_threshold"],
                                                validation=selected["balanced_validation"],
                                                test=metrics(labels, predictions, selected["balanced_threshold"])),
                  generators={part: sorted(data.generators) for part, data in sets.items()})
    report["jpeg_robustness"] = {}
    for quality in sorted(set(args.test_jpeg_quality)):
        compressed = Images(args.dataset / "test", jpeg_quality=quality)
        labels, predictions = scores(model, DataLoader(compressed, batch_size=32), device)
        report["jpeg_robustness"][quality] = {
            "constrained": test_metrics(labels, predictions, selected["threshold"], compressed),
            "balanced": test_metrics(labels, predictions, selected["balanced_threshold"], compressed),
        }
    (args.output / "heldout_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
