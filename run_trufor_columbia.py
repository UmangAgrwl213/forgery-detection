"""Export official TruFor probability maps for local Columbia research."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official TruFor_train_test directory")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    source, weights = args.source.resolve(), args.weights.resolve()
    destination = args.output.resolve()
    images = [image.resolve() for group in ("4cam_auth", "4cam_splc")
              for image in sorted((args.dataset / group).glob("*.tif"))]
    if len(images) != 363:
        parser.error(f"Expected 363 Columbia TIFF images, got {len(images)}")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        images = images[:args.limit]

    sys.path.insert(0, str(source))
    os.chdir(source)
    from dataset.dataset_test import TestDataset
    from lib.config import config, update_config
    from lib.utils import get_model

    settings = argparse.Namespace(experiment="trufor_ph3", opts=["TEST.MODEL_FILE", str(weights)])
    update_config(config, settings)
    device = torch.device(args.device)
    # Official legacy weights store scalar metadata using the old NumPy module name.
    allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
               np.dtypes.Float32DType, np.dtypes.Float64DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model = get_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    dataset = TestDataset(list_img=[str(image) for image in images])
    with torch.inference_mode():
        for index, image in enumerate(images):
            path = destination / image.parent.name / f"{image.name}.npz"
            if path.is_file():
                continue
            rgb, _ = dataset[index]
            logits, _, detection, _ = model(rgb[None].to(device), save_np=False)
            probability = logits[0].softmax(dim=0)[1].cpu().numpy()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as file:
                np.savez_compressed(file, map=probability, score=detection.sigmoid().item())
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(images):
                print(f"Exported {index + 1}/{len(images)}: {image.name}", flush=True)


if __name__ == "__main__":
    main()
