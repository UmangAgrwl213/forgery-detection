import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class CASIADataset(Dataset):
    """Pairs CASIA tampered images with masks using common CASIA file naming."""

    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(self, image_dir, ground_truth_dir, transform=None, *, records=None):
        self.image_dir = Path(image_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.transform = transform
        if not self.image_dir.is_dir() or not self.ground_truth_dir.is_dir():
            raise FileNotFoundError("tp_dir and gt_dir must both be existing directories.")

        self.samples = []
        if records is not None:
            for record in records:
                image = self.image_dir / record["image"]
                mask_name = record["mask"]
                mask = self.ground_truth_dir / mask_name if mask_name is not None else None
                if not image.is_file() or (mask is not None and not mask.is_file()):
                    raise FileNotFoundError(f"Missing manifest pair: {image}, {mask}")
                self.samples.append((image, mask))
            if not self.samples:
                raise ValueError("Manifest partition is empty")
            return
        for image_path in sorted(self.image_dir.iterdir()):
            if image_path.suffix.lower() not in self.extensions:
                continue
            mask_path = self._find_mask(image_path.stem)
            if mask_path:
                self.samples.append((image_path, mask_path))
        if not self.samples:
            raise RuntimeError("No image/mask pairs found. Check CASIA paths and mask filenames.")

    def _find_mask(self, stem):
        for suffix in ("_gt.png", "_gt.jpg", ".png", ".jpg", ".tif"):
            candidate = self.ground_truth_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path is not None
                else np.zeros(image.shape[:2], dtype=np.uint8) if image is not None else None)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read pair: {image_path}, {mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # A few CASIA masks have a different stored resolution from their source image.
        # Align them before geometric augmentation; nearest-neighbor preserves mask labels.
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            return transformed["image"], transformed["mask"].unsqueeze(0).float()
        return image, mask


def read_split_manifest(path):
    splits = json.loads(Path(path).read_bytes())["splits"]
    seen_images = set()
    seen_sources = set()
    for partition in ("train", "validation", "test"):
        rows = splits[partition]
        images = {row["image"].casefold() for row in rows}
        sources = {source for row in rows for source in row["sources"]}
        if not rows or len(images) != len(rows) or images & seen_images or sources & seen_sources:
            raise ValueError(f"Invalid or overlapping manifest partition: {partition}")
        seen_images.update(images)
        seen_sources.update(sources)
    return splits
