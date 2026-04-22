# main.py

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

    # -------- FINAL SUMMARY --------
    best_idx = log_data["val_loss"].index(min(log_data["val_loss"]))
    summary = (
        "📊 --- FINAL TRAINING SUMMARY ---\n"
        f"Best Epoch: {best_idx + 1}\n"
        f"Peak IoU:   {max(log_data['val_iou']):.4f}\n"
        f"Peak F1:    {max(log_data['val_f1']):.4f}\n"
        f"Final Loss: {log_data['val_loss'][-1]:.4f}\n"
        "----------------------------------\n"
    )
    with open(os.path.join(OUTPUT_DIR, "final_report.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()