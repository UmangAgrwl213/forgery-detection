from datasets.casia import CASIADataset
from datasets.transforms import get_train_transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

TP_DIR = r"D:\dataset\CASIA2\Tp"
GT_DIR = r"D:\dataset\CASIA2\CASIA 2 Groundtruth"

dataset = CASIADataset(
    tp_dir=TP_DIR,
    gt_dir=GT_DIR,
    transform=get_train_transforms(256)
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

image, mask = next(iter(loader))

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

import numpy as np

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

img = image[0].permute(1, 2, 0).cpu().numpy()
img = (img * std + mean).clip(0, 1)

msk = mask[0][0].cpu().numpy()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(msk, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.show()
