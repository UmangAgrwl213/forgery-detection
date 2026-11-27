#!/usr/bin/env python3
"""Run the best trained model on random CASIA v2 images to demonstrate functionality."""

import os
import random
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from models.resnet_unet import ResNetUNet
from data.transforms import get_val_transforms
from utils.config import load_yaml_config

def setup_directories():
    """Setup output directories."""
    output_dir = Path("outputs/best_model_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_random_images(dataset_root="dataset/CASIA2", num_images=5):
    """Get random image paths from the dataset."""
    tp_dir = Path(dataset_root) / "Tp"

    if not tp_dir.exists():
        raise FileNotFoundError(f"TP directory not found: {tp_dir}")

    # Get all image files
    image_extensions = [".tif", ".jpg", ".jpeg", ".png"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(tp_dir.glob(f"*{ext}"))

    if len(image_files) == 0:
        raise FileNotFoundError("No image files found in TP directory")

    # Select random images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    return selected_images

def load_model(weights_path, device):
    """Load the trained model with proper error handling."""
    print(f"Loading model from: {weights_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Handle the case where checkpoint keys may have _orig_mod prefix
        checkpoint = {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}

        model = ResNetUNet(pretrained=False).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        print("✓ Model loaded successfully")
        return model

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

def process_single_image(model, image_path, device, output_dir, threshold=0.5):
    """Process a single image and save results."""
    try:
        # Read image
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"  Processing: {image_path.name}")

        # Apply transforms
        transform = get_val_transforms(384)  # Using default image size
        input_tensor = transform(image=original)["image"].unsqueeze(0).to(device)

        # Run prediction
        with torch.inference_mode():
            probability = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

        # Resize back to original dimensions
        height, width = original.shape[:2]
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)

        # Create mask
        mask = ((probability >= threshold).astype(np.uint8) * 255)

        # Create overlay
        overlay = original.copy()
        overlay[mask > 0] = (255, 20, 20)  # Red overlay for detected regions
        overlay = cv2.addWeighted(original, 0.62, overlay, 0.38, 0)

        # Save results
        stem = Path(image_path).stem
        mask_path = output_dir / f"{stem}_mask.png"
        overlay_path = output_dir / f"{stem}_overlay.png"
        prob_path = output_dir / f"{stem}_probability.npy"

        # Save images
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), mask)

        # Save probability array
        np.save(prob_path, probability)

        # Calculate basic statistics
        detected_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        detection_percentage = (detected_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        return {
            "image": image_path.name,
            "detected_pixels": int(detected_pixels),
            "total_pixels": int(total_pixels),
            "detection_percentage": round(detection_percentage, 2),
            "max_probability": float(np.max(probability)),
            "mean_probability": float(np.mean(probability))
        }

    except Exception as e:
        print(f"  ✗ Error processing {image_path.name}: {e}")
        return None

def main():
    """Run the best model on random images."""
    print("=== Running Best Model on Random Images ===")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = setup_directories()

    # Load model
    weights_path = Path("outputs/best_model.pth")
    model = load_model(weights_path, device)

    # Get random images
    try:
        random_images = get_random_images(num_images=3)
        print(f"Selected {len(random_images)} random images for testing:")
        for img in random_images:
            print(f"  - {img.name}")
    except Exception as e:
        print(f"Error getting random images: {e}")
        return

    # Process images
    print("\nProcessing images...")
    results = []

    for image_path in tqdm(random_images, desc="Processing"):
        result = process_single_image(model, image_path, device, output_dir)
        if result:
            results.append(result)

    # Generate summary
    print("\n=== Processing Summary ===")
    if results:
        for result in results:
            print(f"Image: {result['image']}")
            print(f"  Detected pixels: {result['detected_pixels']}")
            print(f"  Detection percentage: {result['detection_percentage']}%")
            print(f"  Max probability: {result['max_probability']:.3f}")
            print(f"  Mean probability: {result['mean_probability']:.3f}")
            print()

        # Overall statistics
        total_detected = sum(r['detected_pixels'] for r in results)
        total_pixels = sum(r['total_pixels'] for r in results)
        avg_detection = (total_detected / total_pixels * 100) if total_pixels > 0 else 0

        print("Overall Statistics:")
        print(f"  Total detected pixels: {total_detected}")
        print(f"  Average detection percentage: {avg_detection:.2f}%")
    else:
        print("No successful predictions completed.")

    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()