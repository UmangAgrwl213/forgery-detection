# Image Forgery Detection (ResNet34-UNet)

A deep learning system for pixel-level image forgery detection, optimized for the CASIA v2 dataset. This project uses a ResNet34-based U-Net architecture to identify and localize splicing and copy-move manipulations.

## 🚀 Recent Upgrades
- **High-Resolution (384x384):** Enhanced vision for fine forensic artifacts.
- **GPU Acceleration:** Fully configured for CUDA-enabled PyTorch.
- **Forensic Precision:** Optimized transforms and hybrid loss (Dice-heavy) for better mask alignment.

## 🛠️ Installation

```bash
# Create environment
python -m venv .venv
.\.venv\Scripts\activate

# Install GPU-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 📈 Usage

### 1. Configure Paths
Edit `configs/config.yaml` to point to your dataset:
```yaml
paths:
  tp_dir: "D:/dataset/CASIA2/Tp"
  gt_dir: "D:/dataset/CASIA2/CASIA 2 Groundtruth"
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
