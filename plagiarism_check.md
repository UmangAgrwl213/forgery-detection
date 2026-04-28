# Project: Image Forgery Detection (ResNet34-UNet)
## Source Code for Plagiarism Check

### 1. main.py
```python
import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from datasets.casia import CASIADataset
from datasets.transforms import get_train_transforms, get_val_transforms
from models.resnet_unet import ResNetUNet
from losses.hybrid import BCEDiceLoss
from engine.train import train_one_epoch
from engine.eval import validate
from utils.config import load_yaml_config


def main():
    torch.backends.cudnn.benchmark = True

    config = load_yaml_config("configs/config.yaml")

    paths = config["paths"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]

    TP_DIR = paths["tp_dir"]
    GT_DIR = paths["gt_dir"]
    OUTPUT_DIR = paths["output_dir"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------- BACKUP PREVIOUS MODEL --------
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        import shutil
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(OUTPUT_DIR, f"best_model_prev_{timestamp}.pth")
        shutil.copy2(best_model_path, backup_path)
        print(f"📦 Previous best model backed up to: {backup_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    image_size = train_cfg["image_size"]

    # -------- DATA --------
    full_dataset = CASIADataset(
        TP_DIR, GT_DIR, transform=None # No transforms yet
    )

    total_size = len(full_dataset)
    train_limit = train_cfg.get("subset_train", total_size)
    val_limit = train_cfg.get("subset_val", total_size)

    # Shuffle indices for a fair split
    indices = torch.randperm(total_size).tolist()

    # Define split points
    # We'll take subset_train for training and subset_val for validation, ensuring they are disjoint
    train_indices = indices[:min(train_limit, total_size - val_limit)]
    val_indices = indices[len(train_indices):len(train_indices) + val_limit]

    # If the user didn't specify limits, or they overlap, we should handle it more robustly
    if not train_cfg.get("subset_train") and not train_cfg.get("subset_val"):
        # Default to 80/20 split
        train_split = int(0.8 * total_size)
        train_indices = indices[:train_split]
        val_indices = indices[train_split:]

    train_dataset = Subset(
        CASIADataset(TP_DIR, GT_DIR, get_train_transforms(image_size)),
        train_indices
    )
    val_dataset = Subset(
        CASIADataset(TP_DIR, GT_DIR, get_val_transforms(image_size)),
        val_indices
    )

    print(f"Dataset: Total={total_size}, Train={len(train_dataset)}, Val={len(val_dataset)}")

    # -------- LOADERS --------
    # Added persistent_workers=True to keep processes alive between epochs
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=True,
        persistent_workers=True if train_cfg.get("num_workers", 0) > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=True,
        persistent_workers=True if train_cfg.get("num_workers", 0) > 0 else False
    )

    # -------- MODEL --------
    model = ResNetUNet().to(device)
    
    # -------- GRAPH OPTIMIZATION (Disabled for Windows) --------
    # if hasattr(torch, "compile") and device == "cuda":
    #     try:
    #         print("🚀 Compiling model for speed...")
    #         model = torch.compile(model)
    #     except Exception as e:
    #         print(f"⚠️ Could not compile model: {e}")

    # -------- LOSS --------
    # Added Label Smoothing for better generalization
    criterion = BCEDiceLoss(
        bce_weight=loss_cfg.get("bce_weight", 0.5),
        label_smoothing=loss_cfg.get("label_smoothing", 0.1)
    )

    # -------- OPTIMIZER --------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=1e-2 # Added weight decay (standard for AdamW)
    )
    
    # -------- SCHEDULER --------
    # Using OneCycleLR for faster convergence and super-convergence
    epochs = train_cfg["epochs"]
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg["learning_rate"] * 10,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    best_val_loss = float("inf")
    log_data = {
        "train_loss": [], 
        "val_loss": [], 
        "val_iou": [], 
        "val_f1": [], 
        "val_precision": [], 
        "val_recall": [], 
        "val_accuracy": [],
        "val_auc": []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # train_one_epoch must call scheduler.step() per iteration for OneCycleLR
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler=scheduler
        )

        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # scheduler.step() is handled inside the training loop for OneCycleLR

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
            f"IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, "best_model.pth")
            )
            print("✔ Best model saved")

        log_data["train_loss"].append(train_loss)
        log_data["val_loss"].append(val_loss)
        log_data["val_iou"].append(val_metrics["iou"])
        log_data["val_f1"].append(val_metrics["f1"])
        log_data["val_precision"].append(val_metrics["precision"])
        log_data["val_recall"].append(val_metrics["recall"])
        log_data["val_accuracy"].append(val_metrics["accuracy"])
        log_data["val_auc"].append(val_metrics["auc"])

        with open(os.path.join(OUTPUT_DIR, "log.json"), "w") as f:
            json.dump(log_data, f, indent=4)


if __name__ == "__main__":
    main()
```

### 2. datasets/casia.py
```python
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CASIADataset(Dataset):
    def __init__(self, tp_dir, gt_dir, transform=None):
        self.tp_dir = tp_dir
        self.gt_dir = gt_dir
        self.transform = transform

        self.images = []

        for fname in os.listdir(tp_dir):
            if fname.lower().endswith((".jpg", ".png", ".tif", ".tiff")):
                gt_name = os.path.splitext(fname)[0] + "_gt.png"
                gt_path = os.path.join(gt_dir, gt_name)

                if os.path.exists(gt_path):
                    self.images.append(fname)

        if len(self.images) == 0:
            raise RuntimeError("No valid CASIA image-mask pairs found.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img_path = os.path.join(self.tp_dir, name)
        gt_path = os.path.join(
            self.gt_dir,
            os.path.splitext(name)[0] + "_gt.png"
        )

        # -------- Read image --------
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # -------- Read mask --------
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {gt_path}")

        mask = (mask > 0).astype("float32")

        # -------- FIX SIZE MISMATCH --------
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # -------- Apply transforms --------
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # -------- Final shape --------
        # Ensure mask is a tensor and has shape (1, H, W)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)

        if mask.ndimension() == 2:
            mask = mask.unsqueeze(0)

        # Ensure image is a tensor (handled by ToTensorV2 usually, but safe check)
        if not torch.is_tensor(image):
            image = torch.from_numpy(image.transpose(2, 0, 1))

        return image, mask
```

### 3. datasets/transforms.py
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=384):
    """
    Forensic-aware transforms.
    Removed GaussianBlur and ColorJitter as they mask forgery artifacts.
    Increased resolution to 384.
    """
    return A.Compose([
        # Resize first to a consistent base size
        A.Resize(img_size + 32, img_size + 32),

        # Randomly crop to the target size - this keeps forensic details sharp
        A.RandomCrop(img_size, img_size),

        # Geometric transforms are safe for forensics
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),

        # Standard Normalize
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])


def get_val_transforms(img_size=384):
    return A.Compose([
        A.Resize(img_size, img_size),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])
```

### 4. models/resnet_unet.py
```python
import torch
import torch.nn as nn
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ResNetUNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1 = nn.Sequential(*self.base_layers[3:4]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1 = nn.Sequential(*self.base_layers[4]) # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = nn.Sequential(*self.base_layers[5]) # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = nn.Sequential(*self.base_layers[6]) # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = nn.Sequential(*self.base_layers[7]) # size=(N, 512, x.H/32, x.W/32)

        self.decode4 = DecoderBlock(512, 256, 256)
        self.decode3 = DecoderBlock(256, 128, 128)
        self.decode2 = DecoderBlock(128, 64, 64)
        self.decode1 = DecoderBlock(64, 64, 64)
        self.decode0 = DecoderBlock(64, 0, 32)

        self.final_conv = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, input):
        e0 = self.layer0(input)    # 112x112, 64
        e0_1 = self.layer0_1(e0)   # 56x56, 64
        e1 = self.layer1(e0_1)     # 56x56, 64
        e2 = self.layer2(e1)       # 28x28, 128
        e3 = self.layer3(e2)       # 14x14, 256
        e4 = self.layer4(e3)       # 7x7, 512

        d4 = self.decode4(e4, e3)  # 14x14, 256
        d3 = self.decode3(d4, e2)  # 28x28, 128
        d2 = self.decode2(d3, e1)  # 56x56, 64
        d1 = self.decode1(d2, e0)  # 112x112, 64
        d0 = self.decode0(d1)      # 224x224, 32

        return self.final_conv(d0)
```

### 5. engine/train.py
```python
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()

    total_loss = 0.0
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward with Mixed Precision
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        # Backward with Mixed Precision
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        # Update OneCycleLR per batch
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    avg_loss = total_loss / len(loader)
    return avg_loss
```

### 6. engine/eval.py
```python
import torch
from tqdm import tqdm
from utils.metrics import calculate_metrics


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    metrics_sum = {
        "iou": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
        "auc": 0.0
    }

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)

        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            batch_metrics = calculate_metrics(outputs, masks)
            
            for k in metrics_sum.keys():
                metrics_sum[k] += batch_metrics[k]

            pbar.set_postfix({
                "loss": loss.item()
            })

    avg_loss = total_loss / len(loader)
    avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics
```

### 7. losses/hybrid.py
```python
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, label_smoothing=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0])) # Can be adjusted
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing manually for BCE if needed
        # Or use pos_weight/weight if class imbalance is severe
        
        # Binary label smoothing:
        # y_smoothed = y * (1 - alpha) + 0.5 * alpha
        smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, smoothed_targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
```

### 8. utils/metrics.py
```python
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_metrics(logits, targets, threshold=0.5, eps=1e-6):
    """
    Calculate comprehensive pixel-wise performance metrics, including AUC.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Flatten for global calculation
    probs_flat = probs.view(-1).cpu().detach().numpy()
    preds_flat = preds.view(-1).cpu().detach().numpy()
    targets_flat = targets.view(-1).cpu().detach().numpy()

    # Confusion Matrix Components (using numpy for consistency here)
    tp = (preds_flat * targets_flat).sum()
    fp = ((preds_flat == 1) & (targets_flat == 0)).sum()
    fn = ((preds_flat == 0) & (targets_flat == 1)).sum()
    tn = ((preds_flat == 0) & (targets_flat == 0)).sum()

    # Performance Indices
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    
    # IoU
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + eps) / (union + eps)

    # AUC Calculation (only if both classes are present in targets)
    try:
        if len(np.unique(targets_flat)) > 1:
            auc = roc_auc_score(targets_flat, probs_flat)
        else:
            auc = 0.5 # Default for single class
    except Exception:
        auc = 0.5

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "auc": float(auc),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }
    }
```

### 9. utils/config.py
```python
import argparse
from pathlib import Path
import yaml


# ---------- Load YAML ----------
def load_yaml_config(config_path):
    if config_path is None:
        return {}

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    return data


# ---------- Get section ----------
def get_config_section(config, section):
    data = config.get(section, {})

    if not isinstance(data, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")

    return data


# ---------- CLI arg ----------
def add_common_config_arg(parser):
    parser.add_argument(
        "--config",
        type=str,   # safer than Path for CLI
        default=None,
        help="Path to YAML config file"
    )


# ---------- Resolve priority ----------
def resolve_setting(args, config_section, key, default=None):
    """
    Priority:
    1. CLI argument
    2. YAML config
    3. Default value
    """
    value = getattr(args, key, None)

    if value is not None:
        return value

    if key in config_section:
        return config_section[key]

    return default
```

### 10. create_gallery.py
```python
import os
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.resnet_unet import ResNetUNet
from datasets.transforms import get_val_transforms
from utils.config import load_yaml_config
from utils.metrics import calculate_metrics

def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    model = ResNetUNet().to(device)
    # Handle possible torch.compile prefix
    state_dict = torch.load(path, map_location=device, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def read_gt(name, gt_dir):
    base = os.path.splitext(name)[0]
    # Try common CASIA suffixes
    potential_paths = [
        os.path.join(gt_dir, base + "_gt.png"),
        os.path.join(gt_dir, base + "_gt.jpg"),
        os.path.join(gt_dir, base + ".png")
    ]
    for p in potential_paths:
        if os.path.exists(p):
            gt = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                return (gt > 0).astype(np.uint8)
    return None

def main():
    config = load_yaml_config("configs/config.yaml")
    
    tp_dir = config["paths"]["tp_dir"]
    gt_dir = config["paths"]["gt_dir"]
    output_dir = os.path.join(config["paths"]["output_dir"], "gallery")
    weights_path = os.path.join(config["paths"]["output_dir"], "best_model.pth")
    image_size = config["training"]["image_size"]
    threshold = config["inference"].get("threshold", 0.5)

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🎨 Generating Gallery (Resolution: {image_size})")
    
    try:
        model = load_model(weights_path, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Tip: Make sure you have trained the model at least once and 'best_model.pth' exists.")
        return

    transform = get_val_transforms(image_size)

    # Get list of images
    all_images = [f for f in os.listdir(tp_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]
    num_samples = config["inference"].get("num_samples", 10)
    samples = random.sample(all_images, min(num_samples, len(all_images)))

    for name in samples:
        path = os.path.join(tp_dir, name)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = read_gt(name, gt_dir)

        # Prepare for model
        augmented = transform(image=image)
        img_tensor = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)

        prob_map = probs.squeeze().cpu().numpy()
        pred_mask = (prob_map > threshold).astype(np.uint8)

        # Metrics for the title
        metric_str = ""
        if gt_mask is not None:
            gt_resized = cv2.resize(gt_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            gt_tensor = torch.from_numpy(gt_resized).unsqueeze(0).unsqueeze(0).float().to(device)
            m = calculate_metrics(logits, gt_tensor, threshold=threshold)
            metric_str = f"F1: {m['f1']:.2f} | IoU: {m['iou']:.2f} | AUC: {m['auc']:.2f}"

        # Visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        # 1. Original
        axes[0].imshow(image)
        axes[0].set_title("Input Image", fontsize=14)
        axes[0].axis("off")

        # 2. GT
        if gt_mask is not None:
            axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth", fontsize=14)
        axes[1].axis("off")

        # 3. Probability (Heatmap)
        im3 = axes[2].imshow(prob_map, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Probability Map", fontsize=14)
        axes[2].axis("off")
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # 4. Prediction (Thresholded)
        axes[3].imshow(pred_mask, cmap="gray")
        axes[3].set_title(f"Prediction (T={threshold})", fontsize=14)
        axes[3].axis("off")

        plt.suptitle(f"File: {name}\n{metric_str}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(output_dir, os.path.splitext(name)[0] + "_result.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✅ Saved: {name} (Metrics: {metric_str})")

    print(f"\n✨ All results saved to: {output_dir}")

if __name__ == "__main__":
    main()
```

### 11. plot_losses.py
```python
import os
import json
import matplotlib.pyplot as plt


LOG_FILE = "outputs/log.json"


def main():
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError("log.json not found. Run training first.")

    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    train = logs.get("train_loss", [])
    val = logs.get("val_loss", [])

    if len(train) == 0:
        raise RuntimeError("No data in log file.")

    plt.figure(figsize=(8, 5))

    plt.plot(train, label="Train Loss")
    if len(val) > 0:
        plt.plot(val, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
```

### 12. predict.py
```python
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.resnet_unet import ResNetUNet
from datasets.transforms import get_val_transforms


WEIGHTS_PATH = "outputs/best_model.pth"
IMAGE_PATH = "D:/final year/dataset/CASIA2/Tp/Tp_D_NNN_M_N_sec20047_arc20001_02128.tif"
GT_DIR = "D:/final year/dataset/CASIA2/CASIA 2 Groundtruth"
IMAGE_SIZE = 224
THRESHOLD = 0.5


def load_model(path, device):
    model = ResNetUNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def read_gt(image_path):
    name = os.path.basename(image_path)
    base = os.path.splitext(name)[0]
    gt_path = os.path.join(GT_DIR, base + "_gt.png")

    if not os.path.exists(gt_path):
        return None

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    return (gt > 0).astype(np.uint8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(WEIGHTS_PATH, device)
    transform = get_val_transforms(IMAGE_SIZE)

    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_mask = read_gt(IMAGE_PATH)

    augmented = transform(image=image)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)

    prob_map = probs.squeeze().cpu().numpy()
    pred_mask = (prob_map > THRESHOLD).astype("uint8")

    # -------- Metrics --------
    if gt_mask is not None:
        # Resize GT to match model output size
        gt_mask = cv2.resize(
            gt_mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )
        
        from utils.metrics import calculate_metrics
        # Need tensors for calculate_metrics
        gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float().to(device)
        metrics = calculate_metrics(logits, gt_tensor, threshold=THRESHOLD)
        
        print("\nPerformance Indices:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"AUC Score: {metrics['auc']:.4f}")
        print(f"IoU Score: {metrics['iou']:.4f}")
        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"TP: {cm['tp']} | FP: {cm['fp']}")
        print(f"FN: {cm['fn']} | TN: {cm['tn']}")

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    if gt_mask is not None:
        plt.subplot(1, 4, 2)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")
    else:
        plt.subplot(1, 4, 2)
        plt.text(0.5, 0.5, "No GT Found", ha='center', va='center')
        plt.title("Ground Truth")
        plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(prob_map, cmap="jet")
    plt.title("Probability")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

### 13. prepare_cloud_zip.py
```python
import zipfile
import os

def zip_project(output_filename="colab_upload.zip"):
    # 1. CORE CODE DIRECTORIES
    code_items = [
        'app_api', 'configs', 'datasets', 'engine', 'losses', 
        'metrics', 'models', 'utils', 'main.py', 'create_gallery.py', 
        'plot_losses.py', 'predict.py', 'requirements.txt'
    ]
    
    # 2. DATASET DIRECTORY
    # Looking for 'dataset' in the parent directory as per your structure
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "dataset"))
    
    exclude_ext = ['.pyc', '.pth', '.json', '.sh']
    exclude_dir = ['__pycache__', 'outputs', 'logs', '.git', '.venv']

    print(f"📦 Starting ZIP creation: {output_filename}")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # --- Add Code ---
        for item in code_items:
            if os.path.isfile(item):
                zipf.write(item, arcname=os.path.join("forgery-detection", item))
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    dirs[:] = [d for d in dirs if d not in exclude_dir]
                    for file in files:
                        if not any(file.endswith(ext) for ext in exclude_ext):
                            file_path = os.path.join(root, file)
                            # Store in a consistent 'forgery-detection' subfolder
                            archive_name = os.path.join("forgery-detection", file_path)
                            zipf.write(file_path, archive_name)
        print("✅ Code added.")

        # --- Add Dataset ---
        if os.path.exists(dataset_dir):
            print(f"📂 Found dataset at: {dataset_dir}. Adding to ZIP...")
            count = 0
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.tif')):
                        file_path = os.path.join(root, file)
                        # Relative path from the parent of 'dataset'
                        rel_path = os.path.relpath(file_path, os.path.join(dataset_dir, ".."))
                        zipf.write(file_path, arcname=rel_path)
                        count += 1
            print(f"✅ Dataset added ({count} images).")
        else:
            print("⚠️ Dataset folder not found in parent directory. Only code was zipped.")

    print(f"\n✨ SUCCESS! Upload '{output_filename}' to your Google Drive.")

if __name__ == "__main__":
    zip_project()
```

### 14. models/unet.py (Legacy/Alternative)
```python
import torch
import torch.nn as nn


# ---------- Basic Block ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ---------- UNet ----------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)
```

### 15. losses/dice.py (Legacy/Alternative)
```python
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice
```
