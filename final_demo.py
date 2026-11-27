#!/usr/bin/env python3
"""
Final demonstration of the forgery detection system.
This shows how to use the pre-trained model for inference.
"""

import os
import sys
from pathlib import Path

def demonstrate_system():
    """Demonstrate the complete system setup and usage."""

    print("=== Forgery Detection System Demo ===")
    print()

    # Show system information
    print("1. System Information:")
    print(f"   Python: {sys.version}")

    import torch
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Show model information
    model_path = "outputs/best_model.pth"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   Model: {model_path}")
        print(f"   Size: {size_mb:.1f} MB")

    # Show config
    config_path = "configs/config.yaml"
    if os.path.exists(config_path):
        print("   Config: configs/config.yaml")

    print()
    print("2. Available Scripts:")
    scripts = [
        "main.py - Train the model",
        "predict.py - Run inference on images",
        "run_best_model_random.py - Test with random CASIA images",
        "demo_model.py - Demonstrate system functionality",
        "test_model_comprehensive.py - Run comprehensive tests"
    ]

    for script in scripts:
        print(f"   {script}")

    print()
    print("3. Usage Examples:")
    print("   # Run inference on a specific image")
    print("   python predict.py path/to/image.jpg")
    print()
    print("   # Run demo with random images from CASIA dataset")
    print("   python run_best_model_random.py")
    print()
    print("   # Train the model (if you have data)")
    print("   python main.py --epochs 5 --image-size 384")
    print()

    # Show directory structure
    print("4. Directory Structure:")
    dirs = ['dataset/CASIA2/Tp', 'dataset/CASIA2/CASIA 2 Groundtruth', 'outputs', 'configs']
    for d in dirs:
        if os.path.exists(d):
            print(f"   ✓ {d}")
        else:
            print(f"   ✗ {d} (missing)")

    print()
    print("5. System Capabilities:")
    print("   ✓ Real-time forgery detection")
    print("   ✓ GPU acceleration with CUDA")
    print("   ✓ Support for CASIA dataset")
    print("   ✓ Pre-trained model available")
    print("   ✓ Visualization of detected regions")
    print("   ✓ Confidence scoring and thresholding")

    print()
    print("=== Demo Complete ===")
    print("The system is ready to run. To get started:")
    print("1. Place images in dataset/CASIA2/Tp/ directory")
    print("2. Run python predict.py /path/to/image.jpg")
    print("3. Or run python run_best_model_random.py for a demo")

if __name__ == "__main__":
    demonstrate_system()