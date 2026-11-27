"""Locate likely forged regions in any image, including files outside CASIA."""

import argparse
import hashlib
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
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Handle the case where checkpoint keys may have _orig_mod prefix
    checkpoint = {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}
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


def predict(image_path, model, image_size, threshold, min_area, device, *, include_model_score=False):
    encoded = np.fromfile(image_path, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    input_tensor = get_val_transforms(image_size)(image=original)["image"].unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()
    model_score = float(np.mean(probability >= threshold)) if include_model_score else None
    height, width = original.shape[:2]
    probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = ((probability >= threshold).astype(np.uint8) * 255)
    result = original, probability, remove_small_regions(mask, min_area)
    return (*result, model_score) if include_model_score else result


def write_png(path, pixels):
    success, encoded = cv2.imencode(".png", pixels)
    if not success:
        raise OSError(f"Could not encode PNG: {path}")
    path.write_bytes(encoded.tobytes())


def save_prediction(image_path, original, probability, mask, output_dir, threshold, *, assessment=None):
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
    write_png(mask_path, mask)
    write_png(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

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
    if assessment is not None:
        summary["tampering_assessment"] = assessment
    (output_dir / f"{stem}_forgery_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report_path, summary


def main():
    parser = argparse.ArgumentParser(description="Find likely forged regions in an arbitrary image.")
    parser.add_argument("image", help="Path to an image; it may be outside the dataset.")
    parser.add_argument("--weights", default=None, help="Path to model .pth (default: outputs/best_model.pth)")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--image-size", type=int, help="Override checkpoint input size (a multiple of 32)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=64, help="Discard smaller regions; use 0 to disable")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    threshold = args.threshold if args.threshold is not None else config["inference"].get("threshold", 0.5)
    if not 0 < threshold < 1 or args.min_area < 0:
        parser.error("threshold must be between 0 and 1 and min-area must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = args.weights or str(Path(config["paths"]["output_dir"]) / "best_model.pth")
    metadata_path = Path(weights).with_name("run_metadata.json")
    image_size = config["training"]["image_size"]
    if metadata_path.is_file():
        image_size = json.loads(metadata_path.read_bytes()).get("image_size", image_size)
    if args.image_size is not None:
        image_size = args.image_size
    if image_size < 32 or image_size % 32:
        parser.error("Image size must be a positive multiple of 32")
    model = load_model(weights, device)
    original, probability, mask, score = predict(args.image, model, image_size, threshold, args.min_area, device,
                                                include_model_score=True)
    assessment = {"model_mask_area_fraction": score, "decision": "uncalibrated"}
    calibration_path = Path(weights).with_name("heldout_eval.json")
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_bytes())
        operating_point = calibration.get("low_false_alarm_operating_point", {})
        cutoff = operating_point.get("image_threshold")
        if (cutoff is not None and calibration.get("image_size") == image_size
                and calibration.get("pixel_threshold") == threshold
                and calibration.get("checkpoint_sha256") == hashlib.sha256(Path(weights).read_bytes()).hexdigest()):
            assessment["image_threshold"] = cutoff
            assessment["decision"] = ("inconclusive" if cutoff > 1 else
                                      "signal_detected" if score >= cutoff else "no_signal_detected")
    report_path, summary = save_prediction(args.image, original, probability, mask, args.output_dir, threshold,
                                           assessment=assessment)
    print(f"Saved result: {report_path}")
    decision = {"uncalibrated": "Tampering result uncalibrated", "inconclusive": "Tampering result inconclusive",
                "signal_detected": "Tampering signal detected", "no_signal_detected": "No tampering signal detected"}
    print(f"{decision[assessment['decision']]} · highlighted area {summary['forged_area_percent']}%")


if __name__ == "__main__":
    main()
