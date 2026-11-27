"""Score the two-camera Columbia splicing masks with a fixed pixel cutoff."""

import argparse
import hashlib
import json
from pathlib import Path

from main import make_loader  # Import before numerical libraries to limit CPU threads.

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from data.transforms import get_val_transforms
from predict import load_model


def camera_masks(rgb):
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    return ((red >= 100) & (red - green >= 70) & (red - blue >= 70),
            (green >= 100) & (green - red >= 70) & (green - blue >= 70))


def localization(probability, rgb):
    camera_a, camera_b = camera_masks(rgb)
    if not camera_a.any() or not camera_b.any():
        raise ValueError("Both camera regions must be present")
    scores = cv2.resize(probability, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    prediction = scores >= 0.5
    hit_a, hit_b = (prediction & camera_a).sum(), (prediction & camera_b).sum()
    a, b = camera_a.sum(), camera_b.sum()
    f1_a = 2 * hit_a / (a + hit_a + hit_b)
    f1_b = 2 * (a - hit_a) / (a + (a - hit_a) + (b - hit_b))
    return float(max(f1_a, f1_b)), float((a + b) / prediction.size)


class ColumbiaImages(torch.utils.data.Dataset):
    def __init__(self, images, size):
        self.images = images
        self.transform = get_val_transforms(size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as image:
            return self.transform(image=np.asarray(image.convert("RGB")))["image"]


@torch.inference_mode()
def predict_maps(images, weights, size, device):
    model = load_model(weights, device)
    loader = make_loader(ColumbiaImages(images, size), 8, False, 0, 2, device.type == "cuda")
    for batch in loader:
        yield from model(batch.to(device)).sigmoid()[:, 0].cpu().numpy()


def read_maps(root, images):
    for image in images:
        with np.load(root / image.parent.name / f"{image.name}.npz", allow_pickle=False) as archive:
            yield archive["map"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--weights", type=Path)
    choice.add_argument("--trufor-maps", type=Path)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    authentic = sorted((args.dataset / "4cam_auth").glob("*.tif"))
    spliced = sorted((args.dataset / "4cam_splc").glob("*.tif"))
    if (len(authentic), len(spliced)) != (183, 180):
        parser.error(f"Expected 183 authentic and 180 spliced images, got {len(authentic)} and {len(spliced)}")
    images = authentic + spliced
    maps = (predict_maps(images, args.weights, args.image_size, torch.device(args.device)) if args.weights else
            read_maps(args.trufor_maps, images))
    rows = []
    for index, (image, probability) in enumerate(zip(images, maps, strict=True)):
        if probability.ndim != 2 or not probability.size or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise ValueError(f"Invalid probability map for {image}")
        row = dict(image=image.relative_to(args.dataset).as_posix(), spliced=index >= len(authentic),
                   model_mask_area=float((probability >= 0.5).mean()))
        if row["spliced"]:
            with Image.open(image.parent / "edgemask" / f"{image.stem}_edgemask_3.jpg") as mask:
                rgb = np.asarray(mask.convert("RGB"))
            row["pixel_f1_polarity_invariant"], row["included_pixel_fraction"] = localization(probability, rgb)
        rows.append(row)
    report = dict(model="ResNet34-UNet" if args.weights else "TruFor", pixel_threshold=0.5,
                  protocol="native three-region mask; exclude boundary; mean best polarity F1",
                  image_mask_area_roc_auc=float(roc_auc_score([row["spliced"] for row in rows],
                                                             [row["model_mask_area"] for row in rows])),
                  pixel_f1_mean=float(np.mean([row["pixel_f1_polarity_invariant"] for row in rows if row["spliced"]])),
                  images=rows)
    if args.weights:
        with args.weights.open("rb") as file:
            report["checkpoint_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
        report["image_size"] = args.image_size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"{args.output} | F1 {report['pixel_f1_mean']:.4f} | image AUC {report['image_mask_area_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
