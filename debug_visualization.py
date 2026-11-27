#!/usr/bin/env python3
"""Debug script to ensure visual outputs are generated properly."""

import os
import cv2
import numpy as np
import torch
from pathlib import Path

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from models.resnet_unet import ResNetUNet
from data.transforms import get_val_transforms

def debug_model_loading():
    """Debug model loading and visualization."""
    print("=== Debugging Model Visualization ===")

    # Check if we have a model file
    weights_path = Path("outputs/best_model.pth")
    if not weights_path.exists():
        print(f"✗ Model file not found: {weights_path}")
        return False

    print(f"✓ Model file found: {weights_path}")
    print(f"  Size: {weights_path.stat().st_size / (1024*1024):.2f} MB")

    # Test loading the model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        checkpoint = {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}

        model = ResNetUNet(pretrained=False).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        print("✓ Model loaded successfully")

        # Test with a simple input
        dummy_input = torch.randn(1, 3, 384, 384).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"✗ Error in model loading/debug: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_processing():
    """Test image processing with visualization."""
    print("\n=== Testing Image Processing ===")

    # Get a sample image from dataset
    tp_dir = Path("dataset/CASIA2/Tp")
    if not tp_dir.exists():
        print(f"✗ TP directory not found: {tp_dir}")
        return False

    # Find first image file
    image_extensions = [".tif", ".jpg", ".jpeg", ".png"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(tp_dir.glob(f"*{ext}"))

    if not image_files:
        print("✗ No image files found in dataset")
        return False

    sample_image = image_files[0]
    print(f"Testing with: {sample_image.name}")

    try:
        # Load and process the image
        bgr = cv2.imread(str(sample_image), cv2.IMREAD_COLOR)
        if bgr is None:
            print("✗ Could not read image")
            return False

        original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"✓ Image loaded successfully. Shape: {original.shape}")

        # Test transforms
        transform = get_val_transforms(384)
        transformed = transform(image=original)
        print(f"✓ Transform applied. Transformed shape: {transformed['image'].shape}")

        # Show what the transformed image looks like
        transformed_img = transformed['image']
        print(f"Transformed tensor shape: {transformed_img.shape}")

        return True

    except Exception as e:
        print(f"✗ Error in image processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_visualization():
    """Create a simple visualization test."""
    print("\n=== Simple Visualization Test ===")

    try:
        # Create a dummy image for testing
        test_img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)

        # Create a simple mask (red overlay)
        mask = np.zeros((384, 384), dtype=np.uint8)
        mask[100:200, 100:200] = 255  # White square

        # Apply overlay
        overlay = test_img.copy()
        overlay[mask > 0] = [255, 20, 20]  # Red overlay

        # Save test images
        output_dir = Path("outputs/test_visualization")
        output_dir.mkdir(exist_ok=True)

        cv2.imwrite(str(output_dir / "test_original.png"), cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / "test_mask.png"), mask)
        cv2.imwrite(str(output_dir / "test_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print("✓ Simple visualization test completed")
        print(f"  Files saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"✗ Error in simple visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests."""
    print("Debugging Visualization System")
    print("=" * 40)

    # Run tests
    success1 = debug_model_loading()
    success2 = test_image_processing()
    success3 = create_simple_visualization()

    if success1 and success2 and success3:
        print("\n🎉 All debugging tests passed!")
        print("The system should generate visual outputs correctly.")
        print("\nTo run actual predictions:")
        print("python predict.py \"dataset/CASIA2/Tp/sample_image.tif\" --weights outputs/best_model.pth")
    else:
        print("\n⚠ Some issues detected in the visualization system")

if __name__ == "__main__":
    main()