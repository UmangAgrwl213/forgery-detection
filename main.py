"""Train the forgery-localization model with CUDA-aware defaults."""

import os

# Windows DataLoader workers each import NumPy/scikit-learn. Limit their BLAS pools
# so eight workers do not exhaust RAM or native-thread resources during validation.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data.casia import CASIADataset, read_split_manifest
from data.transforms import get_train_transforms, get_val_transforms
from engine.eval import validate
from engine.train import train_one_epoch
from losses.hybrid import BCEDiceLoss, FocalDiceLoss
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


def make_random_split(paths, train_cfg, full_dataset, forensic_augmentations=False):
    image_size = train_cfg["image_size"]
    images, masks = paths["tp_dir"], paths["gt_dir"]
    train_data = CASIADataset(images, masks, get_train_transforms(image_size, forensic=forensic_augmentations))
    val_data = CASIADataset(images, masks, get_val_transforms(image_size))
    total = len(train_data)
    indices = torch.randperm(total).tolist()
    train_limit, val_limit = train_cfg.get("subset_train"), train_cfg.get("subset_val")
    if full_dataset or (train_limit in (None, -1) and val_limit in (None, -1)):
        train_limit = int(total * 0.8)
        val_limit = total - train_limit
    else:
        val_limit = min(val_limit or int(total * 0.2), total - 1)
        train_limit = min(train_limit or total - val_limit, total - val_limit)
    if min(train_limit, val_limit) <= 0:
        raise ValueError("Training and validation splits must be nonempty")
    return Subset(train_data, indices[:train_limit]), Subset(val_data, indices[train_limit:train_limit + val_limit]), total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--image-size", type=int, help="Square model input in pixels, divisible by 32.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--loss", choices=("bce-dice", "focal-dice"), default="bce-dice")
    parser.add_argument("--attention", action="store_true", help="Enable decoder squeeze-and-excitation")
    parser.add_argument("--forensic-augmentations", action="store_true", help="Apply training-only JPEG and sensor-like noise")
    parser.add_argument("--output-dir")
    split_options = parser.add_mutually_exclusive_group()
    split_options.add_argument("--full-dataset", action="store_true", help="Use an 80/20 split of all matched pairs.")
    split_options.add_argument("--split-manifest", type=Path, help="Train and validate on frozen CASIA partitions.")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    config = load_yaml_config("configs/config.yaml")
    paths, train_cfg, loss_cfg = config["paths"], config["training"], config["loss"]
    if args.image_size is not None:
        train_cfg["image_size"] = args.image_size
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if train_cfg["image_size"] < 32 or train_cfg["image_size"] % 32:
        parser.error("Image size must be a positive multiple of 32")
    if train_cfg["batch_size"] < 1:
        parser.error("Batch size must be positive")
    accelerator_cfg = config.get("accelerator", {})
    output_dir = args.output_dir or paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if args.split_manifest:
        manifest = read_split_manifest(args.split_manifest)
        run_path = Path(output_dir)
        if (run_path / "best_model.pth").exists() or (run_path / "log.json").exists():
            parser.error("Choose a fresh --output-dir for a manifest-based run")
        with (run_path / "run_metadata.json").open("x", encoding="utf-8") as file:
            json.dump({"split_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
                       "image_size": train_cfg["image_size"], "batch_size": train_cfg["batch_size"], "seed": args.seed,
                       "epochs": args.epochs or train_cfg["epochs"], "loss": args.loss,
                       "decoder_attention": args.attention, "forensic_augmentations": args.forensic_augmentations}, file, indent=2)
    # Keep framework caches inside the project instead of protected user-profile folders.
    torch_cache = os.path.abspath(os.path.join(paths["output_dir"], "torch_cache"))
    os.makedirs(torch_cache, exist_ok=True)
    torch.hub.set_dir(torch_cache)

    best_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_path):
        backup = os.path.join(output_dir, f"best_model_prev_{datetime.now():%Y%m%d_%H%M%S}.pth")
        shutil.copy2(best_path, backup)
        print(f"Previous best model backed up to: {backup}")

    device = configure_accelerator(accelerator_cfg)
    precision = amp_dtype(accelerator_cfg.get("amp_dtype", "float16"))
    image_size = train_cfg["image_size"]
    if args.split_manifest:
        root = Path(paths["tp_dir"]).parent
        train_transform = get_train_transforms(image_size, forensic=args.forensic_augmentations)
        train_dataset = CASIADataset(root, paths["gt_dir"], train_transform, records=manifest["train"])
        val_dataset = CASIADataset(root, paths["gt_dir"], get_val_transforms(image_size), records=manifest["validation"])
        total_size = sum(map(len, manifest.values()))
    else:
        train_dataset, val_dataset, total_size = make_random_split(paths, train_cfg, args.full_dataset, args.forensic_augmentations)
    print(f"Dataset: Total={total_size}, Train={len(train_dataset)}, Val={len(val_dataset)}")
    workers = train_cfg.get("num_workers", 0)
    batch_size = train_cfg["batch_size"]
    prefetch_factor = train_cfg.get("prefetch_factor", 2)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_dataset, batch_size, True, workers, prefetch_factor, pin_memory)
    val_loader = make_loader(val_dataset, batch_size, False, workers, prefetch_factor, pin_memory)

    model = ResNetUNet(pretrained=train_cfg.get("pretrained_encoder", True), attention=args.attention).to(device)
    if accelerator_cfg.get("compile_model", False) and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            print("Compiling model for CUDA training...")
            model = torch.compile(model)
        except Exception as error:
            print(f"Model compilation skipped: {error}")

    if args.loss == "focal-dice":
        criterion = FocalDiceLoss(focal_weight=loss_cfg.get("focal_weight", 0.5),
                                  gamma=loss_cfg.get("focal_gamma", 2.0), alpha=loss_cfg.get("focal_alpha", 0.75))
    else:
        criterion = BCEDiceLoss(bce_weight=loss_cfg.get("bce_weight", 0.5), label_smoothing=loss_cfg.get("label_smoothing", 0.1))
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=1e-2)
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
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
            logs.setdefault(f"val_{name}", []).append(value)
        with open(os.path.join(output_dir, "log.json"), "w", encoding="utf-8") as file:
            json.dump(logs, file, indent=2)
        save_training_plots(os.path.join(output_dir, "log.json"))


if __name__ == "__main__":
    main()
