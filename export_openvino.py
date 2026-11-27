"""Export a trained PyTorch checkpoint to OpenVINO's static-shape IR format."""

import argparse
from pathlib import Path

import torch

from models.resnet_unet import ResNetUNet
from utils.config import load_yaml_config


def main():
    parser = argparse.ArgumentParser(description="Export best_model.pth for Intel NPU inference.")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="outputs/openvino")
    args = parser.parse_args()
    try:
        import openvino as ov
    except ImportError as error:
        raise SystemExit("Install optional NPU dependencies: pip install -r requirements-npu.txt") from error

    config = load_yaml_config(args.config)
    weights = Path(args.weights or Path(config["paths"]["output_dir"]) / "best_model.pth")
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = config["training"]["image_size"]

    model = ResNetUNet(pretrained=False).cpu().eval()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in state.items()})
    onnx_path = output_dir / "forgery_segmentation.onnx"
    sample = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(model, sample, onnx_path, input_names=["image"], output_names=["logits"], opset_version=17)
    ov_model = ov.convert_model(str(onnx_path))
    xml_path = output_dir / "forgery_segmentation.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=True)
    print(f"Saved OpenVINO model: {xml_path}")
    print("Run NPU inference with: python predict_openvino.py IMAGE --model " + str(xml_path) + " --device NPU")


if __name__ == "__main__":
    main()
