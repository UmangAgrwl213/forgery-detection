#!/usr/bin/env python3
"""Quick test to verify the model environment is working."""

import torch
from pathlib import Path

def quick_test():
    """Run a quick verification test."""
    print("=== Quick Environment Test ===")

    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Check model file
    model_path = Path("outputs/best_model.pth")
    if model_path.exists():
        print(f"✓ Model file found: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    else:
        print("✗ Model file NOT found in outputs/")
        print("  Please ensure you have trained the model first")
        return False

    # Test basic import
    try:
        from models.resnet_unet import ResNetUNet
        print("✓ Model architecture imports successfully")

        # Test basic instantiation
        model = ResNetUNet(pretrained=False)
        print("✓ Model instantiation successful")

        # Test dummy forward pass
        dummy_input = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")

        print("\n🎉 Environment is ready to run predictions!")
        return True

    except Exception as e:
        print(f"✗ Error in quick test: {e}")
        return False

if __name__ == "__main__":
    quick_test()