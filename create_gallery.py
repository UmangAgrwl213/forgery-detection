# create_gallery.py

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
