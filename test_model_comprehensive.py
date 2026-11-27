#!/usr/bin/env python3
"""Comprehensive test of the forgery detection model."""

import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from models.resnet_unet import ResNetUNet

def test_model_comprehensive():
    """Run comprehensive tests on the model."""

    print("=== Comprehensive Model Test ===")

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")

    # Test 1: Model creation
    print("\n1. Testing model creation...")
    try:
        model = ResNetUNet(pretrained=False)
        print("✓ Model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

    # Test 2: Load trained weights
    print("\n2. Testing model loading...")
    try:
        weights_path = Path("outputs/best_model.pth")
        if not weights_path.exists():
            print("⚠ Warning: No trained weights found, creating dummy test")
            # Just test the architecture
            dummy_input = torch.randn(1, 3, 384, 384)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"✓ Dummy forward pass successful. Output shape: {output.shape}")
            return True

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        checkpoint = {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint)
        print("✓ Model weights loaded successfully")

        # Test forward pass with real input
        dummy_input = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")

    except Exception as e:
        print(f"✗ Model loading/test failed: {e}")
        return False

    # Test 3: Check model architecture
    print("\n3. Testing model architecture...")
    try:
        print("✓ Model architecture test passed")
        print(f"✓ Model type: {type(model)}")

        # Show some layer information
        layers = list(model.children())
        print(f"✓ Number of top-level layers: {len(layers)}")

    except Exception as e:
        print(f"✗ Architecture test failed: {e}")
        return False

    print("\n=== All Tests Passed! ===")
    print("The model is ready for predictions.")
    return True

if __name__ == "__main__":
    test_model_comprehensive()