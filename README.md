# Forgery Detection

## Setup

Windows PowerShell:

```powershell
cd "D:\final year\forgery-detection"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Configure Paths

Edit [configs/base.yaml](/D:/final%20year/forgery-detection/configs/base.yaml) and set these fields to your dataset location:

```yaml
train:
  tp_dir: D:/dataset/CASIA2/Tp
  gt_dir: D:/dataset/CASIA2/CASIA 2 Groundtruth

predict:
  input_dir: D:/dataset/CASIA2/Tp
  weights_path: outputs/best_model.pth
```

## Train

Using the config file:

```powershell
.\.venv\Scripts\python.exe main.py --config configs\base.yaml
```

Using explicit CLI arguments:

```powershell
.\.venv\Scripts\python.exe main.py `
  --tp-dir "D:\dataset\CASIA2\Tp" `
  --gt-dir "D:\dataset\CASIA2\CASIA 2 Groundtruth" `
  --output-dir outputs `
  --image-size 256 `
  --batch-size 4 `
  --num-workers 4 `
  --epochs 40 `
  --lr 1e-4 `
  --bce-weight 0.3
```

Best weights are saved to `outputs\best_model.pth` by default.

## Predict

Using the config file:

```powershell
.\.venv\Scripts\python.exe predict.py --config configs\base.yaml
```

Using explicit CLI arguments:

```powershell
.\.venv\Scripts\python.exe predict.py `
  --weights-path outputs\best_model.pth `
  --input-dir "D:\dataset\CASIA2\Tp" `
  --output-dir outputs\predictions `
  --image-size 256 `
  --num-samples 5 `
  --percentile 95 `
  --min-area 300
```

Prediction outputs are saved under `outputs\predictions` by default.
