#!/usr/bin/env python3
"""Simple demonstration of comprehensive visualization: real, fake, and mask together."""

import cv2
import numpy as np
import torch
from pathlib import Path

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from models.resnet_unet import ResNetUNet
from data.transforms import get_val_transforms

def demo_comprehensive_visualization():
    """Demonstrate comprehensive visualization with real, fake, and mask."""

    print("=== Comprehensive Forgery Detection Visualization ===")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    weights_path = Path("outputs/best_model.pth")
    if not weights_path.exists():
        print("✗ Model file not found!")
        print("Please ensure you have trained the model first.")
        return

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        checkpoint = {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}

        model = ResNetUNet(pretrained=False).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        print("✓ Model loaded successfully")

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Find a sample image
    tp_dir = Path("dataset/CASIA2/Tp")
    if not tp_dir.exists():
        print("✗ TP directory not found!")
        return

    image_extensions = [".tif", ".jpg", ".jpeg", ".png"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(tp_dir.glob(f"*{ext}"))

    if len(image_files) == 0:
        print("✗ No images found in dataset!")
        return

    sample_image = image_files[0]
    print(f"Processing: {sample_image.name}")

    try:
        # Load and process image
        bgr = cv2.imread(str(sample_image), cv2.IMREAD_COLOR)
        if bgr is None:
            print("✗ Could not read image!")
            return

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

        # Create comprehensive visualization
        print("\nCreating comprehensive visualization...")

        # Create the 4-panel view
        # Panel 1: Original image
        panel1 = original.copy()

        # Panel 2: Probability heatmap
        prob_heatmap = cv2.applyColorMap((probability * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Panel 3: Binary mask
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Panel 4: Overlay with detections (red highlights)
        overlay = original.copy()
        overlay[mask > 0] = (255, 20, 20)  # Red overlay

        # Resize all panels to same size for consistent display
        max_height = max(panel1.shape[0], prob_heatmap.shape[0], mask_colored.shape[0], overlay.shape[0])
        max_width = max(panel1.shape[1], prob_heatmap.shape[1], mask_colored.shape[1], overlay.shape[1])

        # Function to resize panels
        def resize_panel(panel, target_height, target_width):
            if len(panel.shape) == 3:
                resized = np.zeros((target_height, target_width, 3), dtype=panel.dtype)
            else:
                resized = np.zeros((target_height, target_width), dtype=panel.dtype)
            resized[:panel.shape[0], :panel.shape[1]] = panel
            return resized

        # Resize all panels
        panel1_resized = resize_panel(panel1, max_height, max_width)
        prob_resized = resize_panel(prob_heatmap, max_height, max_width)
        mask_resized = resize_panel(mask_colored, max_height, max_width)
        overlay_resized = resize_panel(overlay, max_height, max_width)

        # Create 2x2 layout
        top_row = np.hstack([panel1_resized, prob_resized])
        bottom_row = np.hstack([mask_resized, overlay_resized])
        complete_visualization = np.vstack([top_row, bottom_row])

        # Add labels
        cv2.putText(complete_visualization, 'Original Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(complete_visualization, 'Probability Heatmap', (max_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(complete_visualization, 'Binary Mask', (10, max_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(complete_visualization, 'Detection Overlay', (max_width + 10, max_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Save result
        output_path = Path("outputs/comprehensive_demo.png")
        cv2.imwrite(str(output_path), cv2.cvtColor(complete_visualization, cv2.COLOR_RGB2BGR))

        print("✓ Comprehensive visualization created!")
        print(f"  File saved to: {output_path}")
        print("\nVisualization shows:")
        print("  • Top-left: Original image")
        print("  • Top-right: Probability heatmap (blue = low prob, red = high prob)")
        print("  • Bottom-left: Binary mask (white = detected forgery)")
        print("  • Bottom-right: Detection overlay (red highlights detected regions)")

        # Show statistics
        detected_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        detection_percentage = (detected_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        print(f"\nDetection Statistics:")
        print(f"  Detected pixels: {detected_pixels}")
        print(f"  Total pixels: {total_pixels}")
        print(f"  Detection percentage: {detection_percentage:.2f}%")
        print(f"  Max probability: {np.max(probability):.3f}")

    except Exception as e:
        print(f"✗ Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_comprehensive_visualization()