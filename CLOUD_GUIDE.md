# ☁️ Cloud Training Guide (Google Colab)

## 📌 Setup Cells
Copy and paste these into your Colab notebook.

### 1. Mount & Extract
```python
from google.colab import drive
drive.mount('/content/drive')

# Unzip (Update path to your zip location)
!unzip "/content/drive/MyDrive/colab_upload.zip" -d /content/
%cd /content/forgery-detection
```

### 2. Auto-Configure for Cloud
This script automatically switches your config to "High Performance Cloud Mode."
```python
import yaml
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Switch to Cloud Paths & High Batch Size
config['paths']['tp_dir'] = config['paths']['colab_tp']
config['paths']['gt_dir'] = config['paths']['colab_gt']
config['training']['batch_size'] = config['training']['cloud_batch_size']
config['training']['subset_train'] = -1 # Train on ALL images
config['training']['subset_val'] = -1

with open('configs/config.yaml', 'w') as f:
    yaml.dump(config, f)

!pip install -r requirements.txt
print("🚀 Ready for High-Speed Cloud Training.")
```

### 3. Run & Save
```python
!python main.py
!python plot_losses.py
!python create_gallery.py

# Save results back to Drive
import shutil
shutil.make_archive('/content/drive/MyDrive/forgery_run_latest', 'zip', 'outputs')
```

## 📜 Session Log (21 April 2026)
- **Run ID:** resnet34-unet-v3-384px
- **Epochs:** 30
- **Final IoU:** 56.66%
- **Final F1:** 71.97%
- **Batch Size Used:** 64 (T4 GPU)
- **Note:** Disabling AUC during validation saved ~45 mins of total training time.
