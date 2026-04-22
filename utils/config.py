# utils/config.py

import argparse
from pathlib import Path
import yaml


# ---------- Load YAML ----------
def load_yaml_config(config_path):
    if config_path is None:
        return {}

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    return data


# ---------- Get section ----------
def get_config_section(config, section):
    data = config.get(section, {})

    if not isinstance(data, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")

    return data


# ---------- CLI arg ----------
def add_common_config_arg(parser):
    parser.add_argument(
        "--config",
        type=str,   # safer than Path for CLI
        default=None,
        help="Path to YAML config file"
    )


# ---------- Resolve priority ----------
def resolve_setting(args, config_section, key, default=None):
    """
    Priority:
    1. CLI argument
    2. YAML config
    3. Default value
    """
    value = getattr(args, key, None)

    if value is not None:
        return value

    if key in config_section:
        return config_section[key]

    return default