"""Train the forgery-localization model with CUDA-aware defaults."""

import json
import os
import shutil
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data.casia import CASIADataset
from data.transforms import get_train_transforms, get_val_transforms
from engine.eval import validate
from engine.train import train_one_epoch
from losses.hybrid import BCEDiceLoss
from models.resnet_unet import ResNetUNet
from plot_losses import save_training_plots
from utils.config import load_yaml_config
from utils.runtime import amp_dtype, configure_accelerator


def make_loader(dataset, batch_size, shuffle, workers, prefetch_factor, pin_memory):
    options = {
        "batch_size": batch_size, "shuffle": shuffle, "num_workers": workers,
        "pin_memory": pin_memory, "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **options)


def main():
    config = load_yaml_config("configs/config.yaml")
    paths, train_cfg, loss_cfg = config["paths"], config["training"], config["loss"]
    accelerator_cfg = config.get("accelerator", {})
    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    best_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_path):
        backup = os.path.join(output_dir, f"best_model_prev_{datetime.now():%Y%m%d_%H%M%S}.pth")
        shutil.copy2(best_path, backup)
        print(f"Previous best model backed up to: {backup}")

    device = configure_accelerator(accelerator_cfg)
    precision = amp_dtype(accelerator_cfg.get("amp_dtype", "float16"))
    image_size = train_cfg["image_size"]
    full_dataset = CASIADataset(paths["tp_dir"], paths["gt_dir"])
    total_size = len(full_dataset)
    indices = torch.randperm(total_size).tolist()

    train_limit = train_cfg.get("subset_train")
    val_limit = train_cfg.get("subset_val")
    if train_limit is None and val_limit is None:
        split = int(total_size * 0.8)
        train_indices, val_indices = indices[:split], indices[split:]
    else:
        val_limit = min(val_limit or int(total_size * 0.2), total_size - 1)
        train_limit = min(train_limit or total_size - val_limit, total_size - val_limit)
        train_indices = indices[:train_limit]
        val_indices = indices[train_limit:train_limit + val_limit]
    if not train_indices or not val_indices:
        raise RuntimeError("Training and validation splits must each contain at least one sample.")

    train_dataset = Subset(CASIADataset(paths["tp_dir"], paths["gt_dir"], get_train_transforms(image_size)), train_indices)
    val_dataset = Subset(CASIADataset(paths["tp_dir"], paths["gt_dir"], get_val_transforms(image_size)), val_indices)
    print(f"Dataset: Total={total_size}, Train={len(train_dataset)}, Val={len(val_dataset)}")
    workers = train_cfg.get("num_workers", 0)
    batch_size = train_cfg["batch_size"]
    prefetch_factor = train_cfg.get("prefetch_factor", 2)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_dataset, batch_size, True, workers, prefetch_factor, pin_memory)
    val_loader = make_loader(val_dataset, batch_size, False, workers, prefetch_factor, pin_memory)

    model = ResNetUNet().to(device)
    if accelerator_cfg.get("compile_model", False) and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            print("Compiling model for CUDA training...")
            model = torch.compile(model)
        except Exception as error:
            print(f"Model compilation skipped: {error}")

    criterion = BCEDiceLoss(bce_weight=loss_cfg.get("bce_weight", 0.5), label_smoothing=loss_cfg.get("label_smoothing", 0.1))
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=1e-2)
    epochs = train_cfg["epochs"]
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_cfg["learning_rate"] * 10,
                                              steps_per_epoch=len(train_loader), epochs=epochs,
                                              pct_start=0.3, anneal_strategy="cos")
    logs = {key: [] for key in ("train_loss", "val_loss", "val_iou", "val_f1", "val_precision", "val_recall", "val_accuracy", "val_auc")}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler, precision)
        val_loss, metrics = validate(model, val_loader, criterion, device, precision)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {metrics['iou']:.4f} | F1: {metrics['f1']:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_owner = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(state_owner.state_dict(), best_path)
            print("Best model saved")
        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_loss)
        for name, value in metrics.items():
            logs[f"val_{name}"].append(value)
        with open(os.path.join(output_dir, "log.json"), "w", encoding="utf-8") as file:
            json.dump(logs, file, indent=2)
        save_training_plots(os.path.join(output_dir, "log.json"))


if __name__ == "__main__":
    main()
