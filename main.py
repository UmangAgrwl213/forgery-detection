import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.casia import CASIADataset
from datasets.transforms import get_train_transforms, get_val_transforms
from engine.eval import validate
from engine.train import train_one_epoch
from losses.hybrid import BCEDiceLoss
from models.unet import UNet
from utils.config import (
    add_common_config_arg,
    get_config_section,
    load_yaml_config,
    resolve_setting,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the forgery segmentation model."
    )
    add_common_config_arg(parser)
    parser.add_argument("--tp-dir", dest="tp_dir", help="CASIA Tp directory.")
    parser.add_argument(
        "--gt-dir",
        dest="gt_dir",
        help="CASIA ground-truth directory.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Directory for checkpoints and artifacts.",
    )
    parser.add_argument(
        "--image-size",
        dest="image_size",
        type=int,
        help="Resize dimension for training and validation.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Batch size for both train and validation.",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--bce-weight",
        dest="bce_weight",
        type=float,
        help="BCE contribution in the hybrid loss.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        choices=["cpu", "cuda"],
        help="Force a specific device.",
    )
    return parser


def get_settings():
    parser = build_parser()
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    train_config = get_config_section(config, "train")

    tp_dir = resolve_setting(args, train_config, "tp_dir")
    gt_dir = resolve_setting(args, train_config, "gt_dir")

    if tp_dir is None or gt_dir is None:
        parser.error(
            "Missing required dataset paths. Provide --tp-dir and --gt-dir, "
            "or set them under train: in a YAML config."
        )

    settings = {
        "tp_dir": Path(tp_dir),
        "gt_dir": Path(gt_dir),
        "output_dir": Path(
            resolve_setting(args, train_config, "output_dir", "outputs")
        ),
        "image_size": resolve_setting(args, train_config, "image_size", 256),
        "batch_size": resolve_setting(args, train_config, "batch_size", 4),
        "num_workers": resolve_setting(args, train_config, "num_workers", 4),
        "epochs": resolve_setting(args, train_config, "epochs", 40),
        "lr": resolve_setting(args, train_config, "lr", 1e-4),
        "bce_weight": resolve_setting(args, train_config, "bce_weight", 0.3),
        "device": resolve_setting(args, train_config, "device"),
    }

    return settings


def main():
    settings = get_settings()
    torch.backends.cudnn.benchmark = True

    device = settings["device"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"

    train_dataset = CASIADataset(
        tp_dir=str(settings["tp_dir"]),
        gt_dir=str(settings["gt_dir"]),
        transform=get_train_transforms(settings["image_size"]),
    )

    val_dataset = CASIADataset(
        tp_dir=str(settings["tp_dir"]),
        gt_dir=str(settings["gt_dir"]),
        transform=get_val_transforms(settings["image_size"]),
    )

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        pin_memory=pin_memory,
    )

    model = UNet().to(device)
    criterion = BCEDiceLoss(bce_weight=settings["bce_weight"])
    optimizer = optim.Adam(model.parameters(), lr=settings["lr"])

    best_val_loss = float("inf")

    for epoch in range(settings["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{settings['epochs']}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
