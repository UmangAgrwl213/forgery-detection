#!/usr/bin/env python3
"""Create comprehensive gallery showing real image, fake image, and mask together."""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from models.resnet_unet import ResNetUNet
from data.transforms import get_val_transforms

def load_model(weights_path, device):
    """Load the trained model."""
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Handle the case where checkpoint keys may have _orig_mod prefix
    checkpoint = {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}

    model = ResNetUNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def process_image_for_gallery(image_path, model, device, output_dir):
    """Process a single image and create comprehensive gallery."""
    try:
        # Read image
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transform = get_val_transforms(384)
        input_tensor = transform(image=original)["image"].unsqueeze(0).to(device)

        # Run prediction
        with torch.inference_mode():
            probability = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

        # Resize back to original dimensions
        height, width = original.shape[:2]
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)

        # Create binary mask
        mask = ((probability > 0.5).astype(np.uint8) * 255)

        # Create comprehensive gallery
        # Create a composite image showing:
        # 1. Original image
        # 2. Probability heatmap
        # 3. Binary mask
        # 4. Overlay with detections

        # Create probability heatmap (colormap)
        prob_heatmap = cv2.applyColorMap((probability * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Create overlay
        overlay = original.copy()
        overlay[mask > 0] = (255, 20, 20)  # Red overlay for detected regions

        # Create gallery layout: 2x2 grid
        # Resize images to same dimensions for consistent display
        max_height = max(original.shape[0], prob_heatmap.shape[0], mask.shape[0])
        max_width = max(original.shape[1], prob_heatmap.shape[1], mask.shape[1])

        # Pad images to same size
        def pad_to_size(img, target_height, target_width):
            if len(img.shape) == 3:
                padded = np.zeros((target_height, target_width, 3), dtype=img.dtype)
            else:
                padded = np.zeros((target_height, target_width), dtype=img.dtype)

            padded[:img.shape[0], :img.shape[1]] = img
            return padded

        # Resize all components to same dimensions
        orig_resized = cv2.resize(original, (max_width, max_height))
        prob_resized = pad_to_size(prob_heatmap, max_height, max_width)
        mask_resized = pad_to_size(mask, max_height, max_width)
        overlay_resized = pad_to_size(overlay, max_height, max_width)

        # Create 2x2 gallery
        top_row = np.hstack([orig_resized, prob_resized])
        bottom_row = np.hstack([mask_resized, overlay_resized])

        gallery = np.vstack([top_row, bottom_row])

        # Add labels
        cv2.putText(gallery, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(gallery, 'Probability Heatmap', (max_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(gallery, 'Binary Mask', (10, max_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(gallery, 'Detection Overlay', (max_width + 10, max_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Save gallery
        stem = Path(image_path).stem
        gallery_path = output_dir / f"{stem}_comprehensive_gallery.png"
        cv2.imwrite(str(gallery_path), cv2.cvtColor(gallery, cv2.COLOR_RGB2BGR))

        return {
            "image": image_path.name,
            "gallery_path": gallery_path,
            "detected_pixels": int(np.sum(mask > 0)),
            "max_probability": float(np.max(probability)),
            "mean_probability": float(np.mean(probability))
        }

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return None

def create_comprehensive_gallery():
    """Create comprehensive gallery for multiple images."""
    print("=== Creating Comprehensive Gallery ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup output directory
    output_dir = Path("outputs/comprehensive_gallery")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    weights_path = Path("outputs/best_model.pth")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model not found: {weights_path}")

    model = load_model(weights_path, device)

    # Get sample images from dataset
    tp_dir = Path("dataset/CASIA2/Tp")
    if not tp_dir.exists():
        raise FileNotFoundError(f"TP directory not found: {tp_dir}")

    image_extensions = [".tif", ".jpg", ".jpeg", ".png"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(tp_dir.glob(f"*{ext}"))

    if len(image_files) == 0:
        raise FileNotFoundError("No image files found in TP directory")

    # Select first few images
    selected_images = image_files[:3]  # Limit to first 3 for demo

    print(f"Selected {len(selected_images)} images for gallery creation:")
    for img in selected_images:
        print(f"  - {img.name}")

    # Process images
    print("\nCreating comprehensive galleries...")
    results = []

    for image_path in tqdm(selected_images, desc="Processing"):
        result = process_image_for_gallery(image_path, model, device, output_dir)
        if result:
            results.append(result)

    # Print summary
    print("\n=== Gallery Creation Summary ===")
    for result in results:
        print(f"Image: {result['image']}")
        print(f"  Detected pixels: {result['detected_pixels']}")
        print(f"  Max probability: {result['max_probability']:.3f}")
        print(f"  Mean probability: {result['mean_probability']:.3f}")
        print(f"  Gallery saved to: {result['gallery_path']}")
        print()

    print(f"Complete! All galleries saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    try:
        gallery_dir = create_comprehensive_gallery()
        print(f"\n🎉 Comprehensive galleries created successfully in {gallery_dir}")
        print("Open any .png file to see the complete visualization:")
        print("- Top left: Original image")
        print("- Top right: Probability heatmap")
        print("- Bottom left: Binary mask")
        print("- Bottom right: Detection overlay (red highlights)")
    except Exception as e:
        print(f"Error creating gallery: {e}")
        import traceback
        traceback.print_exc()