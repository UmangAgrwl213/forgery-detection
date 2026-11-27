# Image Forgery Detection (ResNet34-UNet)

A deep learning system for pixel-level image forgery detection, optimized for the CASIA v2 dataset. This project uses a ResNet34-based U-Net architecture to identify and localize splicing and copy-move manipulations.

## 🚀 Recent Upgrades
- **High-Resolution (384x384):** Enhanced vision for fine forensic artifacts.
- **GPU Acceleration:** Fully configured for CUDA-enabled PyTorch.
- **Forensic Precision:** Optimized transforms and hybrid loss (Dice-heavy) for better mask alignment.
- **Enhanced Desktop Application:** Added clipboard support (Ctrl+V) and improved UI
- **Comprehensive Visualization Tools:** New scripts for detailed forensic analysis

## 🗺️ Roadmap
- [ ] **Focal Loss Integration:** Improve boundary detection on hard pixels.
- [ ] **Attention Modules:** Add SE-Blocks to the decoder for channel-wise feature prioritization.
- [ ] **Classification Head:** Multi-task learning for reduced false positives.
- [ ] **Advanced Augmentations:** Add JPEG artifacts and sensor noise simulations.

## 🛠️ Installation

```bash
# Create environment
python -m venv .venv
.\.venv\Scripts\activate

# Install GPU-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### RTX 5070 Ti and Intel NPU

The project is configured for CUDA training with FP16 mixed precision, TensorFloat-32,
pinned-memory loading, and optional `torch.compile`. `batch_size: 16` targets 16 GB
VRAM. If CUDA reports an out-of-memory error, lower it to 12 and then 8 in
`configs/config.yaml`.

The Intel NPU is for low-power inference, not PyTorch training. Export a finished
checkpoint to OpenVINO, then run it on the NPU:

```powershell
python -m pip install -r requirements-npu.txt
python export_openvino.py
python predict_openvino.py "C:/path/to/suspect-image.jpg" --device NPU
```

`--device AUTO:NPU,CPU` permits a CPU fallback if the NPU cannot compile the model.

## 📈 Usage

### 1. Configure Paths
The dataset is expected at `dataset/CASIA2` (the archive supplied with this project
extracts there). The included configuration already points to it:
```yaml
paths:
  tp_dir: "dataset/CASIA2/Tp"
  gt_dir: "dataset/CASIA2/CASIA 2 Groundtruth"
```

### 2. Train Model
Run the main training script. It is configured for GPU by default.
```bash
python main.py
```

### 3. Generate Forensic Gallery
After training, generate a 4-panel comparison gallery (Input | GT | Prob Map | Prediction) for your report:
```bash
python create_gallery.py
```

### 4. Plot Loss Curves
```bash
python plot_losses.py
```

This writes `outputs/training_curves.png`, so it also works on headless machines.
Add `--show` when a desktop plot window is wanted.

### 5. Inspect an image outside the dataset

After training, pass any local image path to the predictor:

```bash
python predict.py "C:/path/to/suspect-image.jpg"
```

It writes three useful artifacts to `outputs/predictions/`: a binary forgery mask,
an image overlay with outlined detected regions, and a three-panel report. Adjust
the sensitivity with `--threshold 0.4`; remove fewer small detections with
`--min-area 0`.

### 6. Desktop Application
Launch the Windows preview with:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_desktop.ps1
```

Paste an image with Ctrl+V or choose an image file; reports are stored under `outputs/desktop`. 
The app prefers the source-disjoint CASIA checkpoint with its matching held-out calibration and uses CPU inference by default so it can run during CUDA training. Use `-Weights` to select another checkpoint, `-Device cuda` for CUDA inference, and `-AiWeights` to opt into an experimental AI checkpoint. When calibration is absent or does not match the checkpoint, results are labeled unsure.

### 7. Comprehensive Visualization
For detailed forensic analysis with all components together:
```bash
python simple_comprehensive_demo.py
```

This creates a comprehensive visualization showing:
- Original image
- Probability heatmap (blue = low probability, red = high probability)
- Binary mask (white = detected forgery regions)
- Detection overlay (red highlights for detected forgeries)

## Dataset layout

The supplied archive is extracted into this layout:

```text
dataset/CASIA2/
  Au/                       # 7,437 authentic images (not used by this mask trainer)
  Tp/                       # 5,123 tampered input images
  CASIA 2 Groundtruth/      # 5,123 pixel masks
```

`main.py` trains on the matching `Tp` and ground-truth filenames (4,981 exact
pairs in this archive); unpaired files are ignored safely. The `Au` images are
kept for future classification or false-positive evaluation.

## ☁️ Cloud Migration (Colab/Kaggle)
If you need to train for more epochs (e.g., 50+), use the provided preparation script:
1. Run `python prepare_cloud_zip.py`.
2. Upload `colab_upload.zip` to Google Drive or Kaggle.
3. Use the "Smart Setup" code blocks provided during the AI collaboration phase to start cloud training.

## 📊 Evaluation Metrics
The system evaluates performance using:
- **IoU (Intersection over Union)**: Standard for localization.
- **F1-Score / Dice**: Pixel-wise accuracy.
- **AUC-ROC**: Model's ability to distinguish between forged and authentic pixels.

## 📊 Benchmark Results
Tampering inference now surfaces the low-false-alarm validation cutoff when the held-out report matches the checkpoint hash, input size, and pixel threshold. The image decision uses model-resolution mask area, consistent with benchmark scoring. This calibrates the image-level decision only; pixel heatmap scores remain uncalibrated. Older checkpoints produce an explicitly uncalibrated report.

The source-disjoint checkpoint's validation-selected low-false-alarm operating point measured a 1.08% authentic false-positive rate and 52.49% tampered recall on its 1,496-image CASIA test split (12/1,115 authentic images flagged). This is an evaluation estimate, not a guarantee on other datasets or individual images.
