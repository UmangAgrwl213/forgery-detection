import argparse
from pathlib import Path

import yaml


def load_yaml_config(config_path):
    if not config_path:
        return {}

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    return data


def get_config_section(config, section):
    data = config.get(section, {})
    if not isinstance(data, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")
    return data


def add_common_config_arg(parser):
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML config file.",
    )


def resolve_setting(args, config_section, key, default=None):
    value = getattr(args, key, None)
    if value is not None:
        return value
    return config_section.get(key, default)
