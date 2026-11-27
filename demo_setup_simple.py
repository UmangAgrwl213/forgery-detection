#!/usr/bin/env python3
"""
Simple demo setup script for the forgery detection system.
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if all required components are available."""
    print("=== Forgery Detection System Setup ===")

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check PyTorch and CUDA
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")

    # Check required packages
    required_packages = ['albumentations', 'opencv-python', 'matplotlib', 'scikit-learn']
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"[OK] {pkg} is installed")
        except ImportError:
            print(f"[FAIL] {pkg} is NOT installed")

    # Check directory structure
    dirs = ['dataset/CASIA2/Tp', 'dataset/CASIA2/CASIA 2 Groundtruth', 'outputs']
    for d in dirs:
        if os.path.exists(d):
            print(f"[OK] Directory exists: {d}")
        else:
            print(f"[FAIL] Directory missing: {d}")

    # Check model file
    model_path = "outputs/best_model.pth"
    if os.path.exists(model_path):
        print(f"[OK] Model file exists: {model_path}")
        print(f"  Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    else:
        print(f"[FAIL] Model file missing: {model_path}")
        print("  Note: You'll need to train the model or download a pre-trained one")

def create_sample_structure():
    """Create sample directory structure for demonstration."""
    print("\n=== Creating Sample Structure ===")

    # Create directories
    dirs = [
        'dataset/CASIA2/Tp',
        'dataset/CASIA2/CASIA 2 Groundtruth',
        'outputs',
        'outputs/predictions'
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

    # Create sample config
    config_content = """paths:
  tp_dir: "dataset/CASIA2/Tp"
  gt_dir: "dataset/CASIA2/CASIA 2 Groundtruth"
  output_dir: "outputs"
  colab_tp: "/content/forgery-detection/dataset/CASIA2/Tp"
  colab_gt: "/content/forgery-detection/dataset/CASIA2/CASIA 2 Groundtruth"

training:
  image_size: 384
  batch_size: 16
  cloud_batch_size: 64
  epochs: 5
  learning_rate: 0.00005
  pretrained_encoder: true
  num_workers: 8
  prefetch_factor: 4

  subset_train: 1500
  subset_val: 400

accelerator:
  device: "cuda"
  amp_dtype: "float16"
  allow_tf32: true
  cudnn_benchmark: true
  compile_model: false

loss:
  bce_weight: 0.1
  label_smoothing: 0.1

inference:
  threshold: 0.5
  num_samples: 5
"""

    with open('configs/config.yaml', 'w') as f:
        f.write(config_content)
    print("Created sample config file")

    return True

def demonstrate_usage():
    """Demonstrate how to use the system."""
    print("\n=== System Usage Demonstration ===")

    print("To run this forgery detection system, you would typically:")
    print("1. Train the model using main.py:")
    print("   python main.py --epochs 5 --image-size 384 --batch-size 16")
    print("")
    print("2. Run predictions on images using predict.py:")
    print("   python predict.py /path/to/image.jpg")
    print("")
    print("3. Or run a demo with random images:")
    print("   python run_best_model_random.py")
    print("")
    print("4. For comprehensive testing:")
    print("   python test_model_comprehensive.py")
    print("")

    print("Current directory contents:")
    for item in os.listdir('.'):
        if not item.startswith('.') or item in ['.', '..']:
            print(f"  {item}")

def main():
    """Main setup function."""
    print("Setting up forgery detection system environment...")

    # Check environment
    check_environment()

    # Create sample structure
    create_sample_structure()

    # Demonstrate usage
    demonstrate_usage()

    print("\n=== Setup Complete ===")
    print("The environment is ready. To actually run the system:")
    print("1. Download the CASIA dataset to the appropriate directories")
    print("2. Train a model with main.py")
    print("3. Use predict.py for inference on new images")

if __name__ == "__main__":
    main()