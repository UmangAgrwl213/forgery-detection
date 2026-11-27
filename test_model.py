#!/usr/bin/env python3
"""Simple test to verify the model environment is working."""

import torch
from pathlib import Path
from models.resnet_unet import ResNetUNet

def test_model_loading():
    """Test that we can load and use the model."""

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a simple model instance
    try:
        model = ResNetUNet(pretrained=False)
        print("Model created successfully")

        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 384, 384)
        print(f"Input shape: {dummy_input.shape}")

        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            print("Forward pass successful!")

        # Check if we have a trained model file
        model_path = Path("outputs/best_model.pth")
        if model_path.exists():
            print(f"Found trained model at: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            print(f"Checkpoint loaded successfully. Keys: {list(checkpoint.keys())[:5]}...")  # Show first 5 keys

        else:
            print("No trained model found in outputs directory")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()