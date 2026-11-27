"""Hardware selection and safe CUDA performance settings."""

import torch


def configure_accelerator(settings):
    """Return the selected device and configure CUDA only when available."""
    requested = settings.get("device", "cuda").lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("accelerator.device must be auto, cuda, or cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see an NVIDIA GPU.")
    use_cuda = requested != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        allow_tf32 = settings.get("allow_tf32", True)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = settings.get("cudnn_benchmark", True)
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        properties = torch.cuda.get_device_properties(0)
        print(f"Using CUDA: {properties.name} ({properties.total_memory / 2**30:.1f} GiB VRAM)")
    else:
        print("Using CPU; CUDA acceleration is unavailable or disabled.")
    return device


def amp_dtype(name):
    if name.lower() == "float16":
        return torch.float16
    if name.lower() == "bfloat16":
        return torch.bfloat16
    raise ValueError("accelerator.amp_dtype must be float16 or bfloat16")
