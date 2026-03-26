import os
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.unet import UNet
from datasets.transforms import get_val_transforms
from utils.postprocess import postprocess_mask


# ---------------- CONFIG ----------------
WEIGHTS_PATH = "outputs/best_model.pth"
TP_DIR = r"D:\dataset\CASIA2\Tp"
GT_DIR = r"D:\dataset\CASIA2\CASIA 2 Groundtruth"

OUTPUT_DIR = "outputs/visualizations"
IMAGE_SIZE = 256
NUM_SAMPLES = 5

PERCENTILE = 95
MIN_AREA = 300
# ----------------------------------------


def load_model(weights_path, device):
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gt(tp_filename, gt_dir):
    base = os.path.splitext(tp_filename)[0]
    gt_path = os.path.join(gt_dir, base + "_gt.png")

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"GT not found: {gt_path}")

    return (gt > 0).astype(np.uint8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(WEIGHTS_PATH, device)
    transform = get_val_transforms(IMAGE_SIZE)

    valid_exts = (".jpg", ".png", ".tif", ".tiff")
    images = []

    # 🔒 Only keep TP images that have GT
    for f in os.listdir(TP_DIR):
        if f.lower().endswith(valid_exts):
            gt_path = os.path.join(
                GT_DIR, os.path.splitext(f)[0] + "_gt.png"
            )
            if os.path.exists(gt_path):
                images.append(f)

    if len(images) < NUM_SAMPLES:
        raise RuntimeError(
            f"Only {len(images)} images with GT found, need {NUM_SAMPLES}"
        )

    selected = random.sample(images, NUM_SAMPLES)

    for idx, name in enumerate(selected, 1):
        print(f"[{idx}/{NUM_SAMPLES}] {name}")

        tp_path = os.path.join(TP_DIR, name)

        image = read_rgb(tp_path)
        gt_mask = read_gt(name, GT_DIR)

        augmented = transform(image=image)
        img_tensor = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)

        prob_map = probs.squeeze().cpu().numpy()

        pred_mask = postprocess_mask(
            prob_map,
            percentile=PERCENTILE,
            min_area=MIN_AREA
        )

        # -------- Save side-by-side --------
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(prob_map, cmap="jet")
        plt.title("Probability Map")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.tight_layout()

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{os.path.splitext(name)[0]}_comparison.png"
        )

        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
