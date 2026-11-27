"""Locate likely forged regions in any image, including files outside CASIA."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.transforms import get_val_transforms
from models.resnet_unet import ResNetUNet
from utils.config import load_yaml_config


def load_model(weights_path, device):
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    checkpoint = {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}
    # The checkpoint already contains all weights; do not download ImageNet weights at inference.
    model = ResNetUNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def remove_small_regions(mask, min_area):
    if min_area <= 0:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 255
    return clean


def predict(image_path, model, image_size, threshold, min_area, device):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    input_tensor = get_val_transforms(image_size)(image=original)["image"].unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()
    height, width = original.shape[:2]
    probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = ((probability >= threshold).astype(np.uint8) * 255)
    return original, probability, remove_small_regions(mask, min_area)


def save_prediction(image_path, original, probability, mask, output_dir, threshold):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    mask_path = output_dir / f"{stem}_forgery_mask.png"
    overlay_path = output_dir / f"{stem}_forgery_overlay.png"
    report_path = output_dir / f"{stem}_forgery_report.png"

    overlay = original.copy()
    overlay[mask > 0] = (255, 20, 20)
    overlay = cv2.addWeighted(original, 0.62, overlay, 0.38, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 235, 0), 2)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    figure, axes = plt.subplots(1, 3, figsize=(17, 6))
    axes[0].imshow(original); axes[0].set_title("Input image")
    heatmap = axes[1].imshow(probability, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Forgery probability")
    figure.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay); axes[2].set_title(f"Detected regions (threshold {threshold:g})")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(report_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    forged_pixels = int((mask > 0).sum())
    summary = {
        "input": str(Path(image_path).resolve()), "threshold": threshold,
        "forged_area_percent": round(100 * forged_pixels / mask.size, 3),
        "mask": str(mask_path.resolve()), "overlay": str(overlay_path.resolve()),
    }
    (output_dir / f"{stem}_forgery_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report_path, summary


def main():
    parser = argparse.ArgumentParser(description="Find likely forged regions in an arbitrary image.")
    parser.add_argument("image", help="Path to an image; it may be outside the dataset.")
    parser.add_argument("--weights", default=None, help="Path to model .pth (default: outputs/best_model.pth)")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=64, help="Discard smaller regions; use 0 to disable")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    threshold = args.threshold if args.threshold is not None else config["inference"].get("threshold", 0.5)
    if not 0 < threshold < 1 or args.min_area < 0:
        parser.error("threshold must be between 0 and 1 and min-area must be non-negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.weights or str(Path(config["paths"]["output_dir"]) / "best_model.pth")
    model = load_model(weights, device)
    original, probability, mask = predict(args.image, model, config["training"]["image_size"], threshold, args.min_area, device)
    report_path, summary = save_prediction(args.image, original, probability, mask, args.output_dir, threshold)
    print(f"Saved result: {report_path}")
    print(f"Predicted forged area: {summary['forged_area_percent']}%")


if __name__ == "__main__":
    main()
