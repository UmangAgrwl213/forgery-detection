#!/usr/bin/env python3
"""Simple demonstration of how to run the forgery detection model."""

import os
import sys
from pathlib import Path

def main():
    print("=== Forgery Detection Model Demo ===")

    # Show current directory structure
    print("\n1. Current working directory:")
    print(os.getcwd())

    # Check if we have the required files
    required_files = [
        "outputs/best_model.pth",
        "dataset/CASIA2/Tp/",
        "predict.py"
    ]

    print("\n2. Checking required files:")
    for file_path in required_files:
        path = Path(file_path)
        exists = path.exists()
        print(f"  {file_path}: {'✓' if exists else '✗'}")

        if not exists and "dataset" in file_path:
            print("     Note: Dataset directory structure should be:")
            print("     dataset/CASIA2/")
            print("       ├── Tp/ (tampered images)")
            print("       └── CASIA 2 Groundtruth/ (ground truth masks)")

    # Show model information if available
    model_path = Path("outputs/best_model.pth")
    if model_path.exists():
        print(f"\n3. Model file found:")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"   Last modified: {model_path.stat().st_mtime}")

    # Show training log if available
    log_path = Path("outputs/log.json")
    if log_path.exists():
        print(f"\n4. Training log found:")
        import json
        with open(log_path, 'r') as f:
            log_data = json.load(f)

        train_loss = log_data.get('train_loss', [])
        val_loss = log_data.get('val_loss', [])

        print(f"   Training epochs: {len(train_loss)}")
        if val_loss:
            print(f"   Validation epochs: {len(val_loss)}")
        print(f"   Last training loss: {train_loss[-1]:.4f}" if train_loss else "   No training data")

    # Instructions for running predictions
    print("\n5. How to run predictions:")
    print("   Option 1: Test with a single image")
    print("     python predict.py \"dataset/CASIA2/Tp/Tp_D_CND_M_N_ani00018_sec00096_00138.tif\"")
    print("     ")
    print("   Option 2: Run comprehensive test")
    print("     python test_model_comprehensive.py")
    print("     ")
    print("   Option 3: Run predictions on random images")
    print("     python run_random_predictions.py")

    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()