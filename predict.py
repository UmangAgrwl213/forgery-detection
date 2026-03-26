import argparse
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from datasets.transforms import get_val_transforms
from models.unet import UNet
from utils.config import (
    add_common_config_arg,
    get_config_section,
    load_yaml_config,
    resolve_setting,
)
from utils.postprocess import postprocess_mask


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run forgery segmentation inference."
    )
    add_common_config_arg(parser)
    parser.add_argument(
        "--weights-path",
        dest="weights_path",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        help="Directory of input images.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Directory for saved predictions.",
    )
    parser.add_argument(
        "--image-size",
        dest="image_size",
        type=int,
        help="Resize dimension before inference.",
    )
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        type=int,
        help="Number of random input images to process.",
    )
    parser.add_argument(
        "--percentile",
        dest="percentile",
        type=float,
        help="Percentile used for adaptive thresholding.",
    )
    parser.add_argument(
        "--min-area",
        dest="min_area",
        type=int,
        help="Minimum connected-component area to keep.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        choices=["cpu", "cuda"],
        help="Force a specific device.",
    )
    return parser


def get_settings():
    parser = build_parser()
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    predict_config = get_config_section(config, "predict")

    weights_path = resolve_setting(args, predict_config, "weights_path")
    input_dir = resolve_setting(args, predict_config, "input_dir")

    if weights_path is None or input_dir is None:
        parser.error(
            "Missing required inference paths. Provide --weights-path and "
            "--input-dir, or set them under predict: in a YAML config."
        )

    return {
        "weights_path": Path(weights_path),
        "input_dir": Path(input_dir),
        "output_dir": Path(
            resolve_setting(
                args, predict_config, "output_dir", "outputs/predictions"
            )
        ),
        "image_size": resolve_setting(args, predict_config, "image_size", 256),
        "num_samples": resolve_setting(args, predict_config, "num_samples", 5),
        "percentile": resolve_setting(args, predict_config, "percentile", 95),
        "min_area": resolve_setting(args, predict_config, "min_area", 300),
        "device": resolve_setting(args, predict_config, "device"),
    }


def load_model(weights_path, device):
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def predict_image(model, image, transform, device, percentile, min_area):
    augmented = transform(image=image)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)

    prob_map = probs.squeeze().cpu().numpy()
    bin_mask = postprocess_mask(
        prob_map,
        percentile=percentile,
        min_area=min_area,
    )
    return prob_map, bin_mask


def main():
    settings = get_settings()
    device = settings["device"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(settings["weights_path"], device)
    transform = get_val_transforms(settings["image_size"])

    valid_exts = (".jpg", ".png", ".tif", ".tiff")
    images = [
        name for name in os.listdir(settings["input_dir"])
        if name.lower().endswith(valid_exts)
    ]

    if len(images) < settings["num_samples"]:
        raise RuntimeError(
            f"Only {len(images)} images found, need {settings['num_samples']}"
        )

    images = random.sample(images, settings["num_samples"])
    print(f"Predicting on {len(images)} random images")

    for name in images:
        try:
            img_path = settings["input_dir"] / name
            image = read_image(img_path)
            prob_map, bin_mask = predict_image(
                model,
                image,
                transform,
                device,
                settings["percentile"],
                settings["min_area"],
            )

            base = Path(name).stem
            prob_path = output_dir / f"{base}_prob.png"
            mask_path = output_dir / f"{base}_mask.png"

            plt.imsave(prob_path, prob_map, cmap="jet")
            cv2.imwrite(str(mask_path), bin_mask)

            print(f"[SAVED] {prob_path}")
            print(f"[SAVED] {mask_path}")
        except Exception as error:
            print(f"[SKIPPED] {name} -> {error}")


if __name__ == "__main__":
    main()
