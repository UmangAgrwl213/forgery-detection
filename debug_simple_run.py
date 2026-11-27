#!/usr/bin/env python3
"""Simple debug run to verify the system works."""

import cv2
import numpy as np
import torch
from pathlib import Path

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from models.resnet_unet import ResNetUNet
from data.transforms import get_val_transforms

def debug_system():
    """Simple debug to verify system works."""
    print("=== Debug System Run ===")

    # Check if we have the model
    weights_path = Path("outputs/best_model.pth")
    print(f"Model path: {weights_path}")
    print(f"Model exists: {weights_path.exists()}")

    if not weights_path.exists():
        print("ERROR: Model file not found!")
        return False

    # Check dataset
    tp_dir = Path("dataset/CASIA2/Tp")
    print(f"Dataset path: {tp_dir}")
    print(f"Dataset exists: {tp_dir.exists()}")

    if tp_dir.exists():
        files = list(tp_dir.glob("*"))
        print(f"Found {len(files)} files in dataset")
        if files:
            print(f"Sample file: {files[0].name}")

    # Test basic model loading
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        print("✓ Model loaded successfully")

        # Simple test with dummy data
        model = ResNetUNet(pretrained=False).to(device)
        model.load_state_dict(checkpoint)
        model.eval()

        print("✓ Model initialized and loaded")

        # Test forward pass
        dummy_input = torch.randn(1, 3, 384, 384).to(device)
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        print("✓ System is working correctly!")
        return True

    except Exception as e:
        print(f"✗ Error in debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_system()