"""Run static-shape OpenVINO inference on an Intel NPU, CPU, or AUTO device."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from predict import remove_small_regions, save_prediction
from utils.config import load_yaml_config


def main():
    parser = argparse.ArgumentParser(description="Run exported forgery localization with OpenVINO.")
    parser.add_argument("image")
    parser.add_argument("--model", default="outputs/openvino/forgery_segmentation.xml")
    parser.add_argument("--device", default="NPU", help="NPU, CPU, or AUTO:NPU,CPU")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=64)
    args = parser.parse_args()
    if not 0 < (args.threshold or 0.5) < 1 or args.min_area < 0:
        parser.error("threshold must be between 0 and 1 and min-area must be non-negative")
    try:
        import openvino as ov
    except ImportError as error:
        raise SystemExit("Install optional NPU dependencies: pip install -r requirements-npu.txt") from error

    config = load_yaml_config(args.config)
    threshold = args.threshold if args.threshold is not None else config["inference"].get("threshold", 0.5)
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"OpenVINO model not found: {model_path}. Run export_openvino.py first.")
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {args.image}")
    original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_size = config["training"]["image_size"]
    resized = cv2.resize(original, (image_size, image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    normalized = (resized - np.array((0.485, 0.456, 0.406), dtype=np.float32)) / np.array((0.229, 0.224, 0.225), dtype=np.float32)
    model_input = np.transpose(normalized, (2, 0, 1))[None, ...]

    core = ov.Core()
    if args.device.upper() == "NPU" and "NPU" not in core.available_devices:
        raise RuntimeError(f"OpenVINO cannot see an NPU. Available devices: {core.available_devices}")
    compiled = core.compile_model(str(model_path), args.device)
    logits = compiled([model_input])[compiled.output(0)]
    probability = 1.0 / (1.0 + np.exp(-logits.squeeze()))
    height, width = original.shape[:2]
    probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = remove_small_regions(((probability >= threshold).astype(np.uint8) * 255), args.min_area)
    report_path, summary = save_prediction(args.image, original, probability, mask, args.output_dir, threshold)
    print(f"OpenVINO device: {args.device}")
    print(f"Saved result: {report_path}")
    print(f"Predicted forged area: {summary['forged_area_percent']}%")


if __name__ == "__main__":
    main()
