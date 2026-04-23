# Image Forgery Detection - AI Collaboration Guide

---

## 📌 Project Overview
This project implements an **Image Forgery Detection system** using a **ResNet34-UNet model in PyTorch**.  
The system performs **pixel-wise segmentation** to identify manipulated regions in images, optimized for the **CASIA v2 dataset**.

---

## 🛠️ Environment Setup (GPU Verified)
The system is fully configured for **GPU-accelerated training** on the **NVIDIA GTX 1650 Ti (4GB)** using CUDA 12.4.

```bash
# Environment Status: ACTIVE (PyTorch 2.6.0+cu124)
# To verify:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🚀 Key Features & Forensic Upgrades
- **High-Resolution Vision:** Upgraded from 224 to **384x384** to capture fine forensic artifacts and edge inconsistencies.
- **Forensic-Aware Transforms:** Removed blurring/color-jittering augmentations that mask forgery evidence; implemented **RandomCrop** to maintain sharpness.
- **Training Stability:** 
    - **Gradient Clipping:** Prevent `nan` losses by capping gradients at `1.0`.
    - **Optimized LR:** Set to `5e-5` for stable convergence at higher resolutions.
    - **Dice-Heavy Loss:** BCE weight reduced to `0.1` to prioritize localization of small forged regions.
- **Mixed Precision (AMP):** Faster training via `torch.amp`.
- **Advanced Optimization:** AdamW Optimizer + OneCycleLR Scheduler.

---

## 📂 Project Structure (CLEANED)

```text
forgery-detection/
  ├── datasets/
  │     ├── casia.py         # Optimized dataset loader (Tensor-safe)
  │     └── transforms.py    # Forensic-aware Albumentations pipeline
  ├── engine/
  │     ├── train.py         # Training with AMP + Clipping + OneCycle
  │     └── eval.py          # Comprehensive Metric Evaluation
  ├── models/
  │     └── resnet_unet.py   # ResNet34-UNet architecture
  ├── losses/
  │     └── hybrid.py        # BCEDiceLoss with Label Smoothing
  ├── utils/
  │     ├── metrics.py       # Full analytical metric suite (IoU, F1, AUC)
  │     └── config.py        # YAML configuration loader
  ├── outputs/               # Model weights, logs, and Gallery
  ├── configs/
  │     └── config.yaml      # Master training configuration
  ├── main.py                # Primary training entry point
  ├── create_gallery.py      # 4-panel forensic visualization tool
  ├── plot_losses.py         # Training progress visualizer
  └── predict.py             # Single image inference
```

---

## ☁️ Cloud Migration (Google Colab / Kaggle)
The project is "Cloud Ready." Use `prepare_cloud_zip.py` to create a clean bundle for upload.
*   **Target Epochs:** 30–50 (Recommended for Cloud GPUs).
*   **Resolution:** 384x384.

---

## ⚡ Technical Integrity
| Feature | Status | Benefit |
| :--- | :--- | :--- |
| **GPU Acceleration** | ✅ Enabled | 10x faster training than CPU. |
| **Triton/Compile** | ❌ Disabled | Compatibility fix for Windows. |
| **Gradient Clipping**| ✅ Active | Fixes `nan` loss during high-res training. |
| **Dice Weighting** | ✅ 0.9 Dice / 0.1 BCE | Handles authentic-to-forged pixel imbalance. |

---

## 🔮 Future Improvements & Research Roadmap
To further enhance detection accuracy and robustness, the following upgrades are planned:

- **Loss Function Evolution:** Transition to **Focal-Dice Loss** to better handle boundary pixel ambiguity and class imbalance.
- **Forensic Augmentations:** Integrate **JPEG Compression Simulation** and **Gaussian Noise** into the pipeline to improve robustness against real-world social media compressions.
- **Attention Mechanisms:** Implement **Squeeze-and-Excitation (SE) blocks** in the UNet decoder to prioritize high-signal forensic features.
- **Multi-Task Learning:** Add a **Classification Head** to the ResNet encoder to provide global context (Authentic vs. Forged) alongside local segmentation.
- **Differential Learning Rates:** Apply finer LR control, giving the pretrained encoder a smaller rate than the fresh decoder for more stable convergence.

---

## 🏁 Final Steps for Report
1.  **Run `python create_gallery.py`** after training to generate comparison visuals.
2.  **Use `python plot_losses.py`** to generate Loss/IoU curves.
3.  **Cloud Train:** Perform 50 epochs on Colab/Kaggle for peak F1-scores.
