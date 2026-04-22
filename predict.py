# predict.py

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