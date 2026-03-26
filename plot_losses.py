import json
import matplotlib.pyplot as plt
import cv2
import numpy as np

def postprocess_mask(prob_map, threshold=0.25, min_area=100):
    """
    prob_map: HxW float array (0–1)
    returns: clean binary mask (HxW, uint8)
    """

    # 1. Threshold
    binary = (prob_map > threshold).astype(np.uint8) * 255

    # 2. Morphological closing (fill gaps)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


with open("outputs/loss_curve_hybrid.json") as f:
    data = json.load(f)

train = data["train"]
val = data["val"]

plt.figure(figsize=(6, 4))
plt.plot(train, label="Train Loss")
plt.plot(val, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Hybrid BCE + Dice Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
