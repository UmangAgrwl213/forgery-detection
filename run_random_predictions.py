#!/usr/bin/env python3
"""Run predictions on random images from the CASIA dataset to test the model."""

import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add the project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from models.resnet_unet import ResNetUNet
from data.transforms import get_val_transforms
from utils.config import load_yaml_config

def load_model(weights_path, device):
    """Load the trained model."""
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Handle the case where checkpoint keys may have _orig_mod prefix
    checkpoint = {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}

    model = ResNetUNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict_single_image(model, image_path, device, image_size=384):
    """Run prediction on a single image."""
    # Read image
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Apply transforms
    transform = get_val_transforms(image_size)
    input_tensor = transform(image=original)["image"].unsqueeze(0).to(device)

    with torch.inference_mode():
        probability = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

    # Resize back to original dimensions
    height, width = original.shape[:2]
    probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)

    return original, probability

def get_random_images_from_dataset(dataset_dir, num_images=5):
    """Get random image paths from the dataset."""
    tp_dir = Path(dataset_dir) / "Tp"
    gt_dir = Path(dataset_dir) / "CASIA 2 Groundtruth"

    if not tp_dir.exists():
        raise FileNotFoundError(f"TP directory not found: {tp_dir}")

    # Get all .tif and .jpg files from Tp directory
    image_extensions = [".tif", ".jpg", ".jpeg", ".png"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(tp_dir.glob(f"*{ext}"))

    if len(image_files) == 0:
        raise FileNotFoundError("No image files found in TP directory")

    # Randomly select images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    return selected_images

def main():
    """Run predictions on random dataset images."""
    print("Starting random image prediction test...")

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    weights_path = Path("outputs/best_model.pth")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    try:
        model = load_model(weights_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get random images
    dataset_dir = Path("dataset/CASIA2")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    try:
        random_images = get_random_images_from_dataset(dataset_dir, num_images=3)
        print(f"Selected {len(random_images)} random images for testing:")
        for img in random_images:
            print(f"  - {img.name}")
    except Exception as e:
        print(f"Error getting random images: {e}")
        return

    # Run predictions
    output_dir = Path("outputs/random_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning predictions...")
    for i, image_path in enumerate(tqdm(random_images, desc="Predictions")):
        try:
            original, probability = predict_single_image(model, image_path, device)

            # Save results
            stem = Path(image_path).stem
            output_image_path = output_dir / f"{stem}_prediction.png"

            # Create overlay
            mask = ((probability > 0.5).astype(np.uint8) * 255)
            overlay = original.copy()
            overlay[mask > 0] = (255, 20, 20)  # Red overlay for detected regions
            overlay = cv2.addWeighted(original, 0.62, overlay, 0.38, 0)

            # Save results
            cv2.imwrite(str(output_image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print(f"\nPredictions completed! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()