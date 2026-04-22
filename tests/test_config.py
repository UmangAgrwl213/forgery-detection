import pytest
import yaml
import argparse
from pathlib import Path
from utils.config import load_yaml_config, get_config_section, resolve_setting, add_common_config_arg

def test_load_yaml_config_none_path():
    assert load_yaml_config(None) == {}
    assert load_yaml_config("") == {}

def test_load_yaml_config_valid(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: value", encoding="utf-8")
    assert load_yaml_config(str(config_file)) == {"key": "value"}

def test_load_yaml_config_empty(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")
    assert load_yaml_config(str(config_file)) == {}

def test_load_yaml_config_not_mapping(tmp_path):
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(ValueError, match="Config file must contain a top-level mapping."):
        load_yaml_config(str(config_file))

def test_load_yaml_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("{ ", encoding="utf-8")
    # yaml.safe_load("{ ") raises yaml.parser.ParserError, which inherits from YAMLError
    with pytest.raises(yaml.YAMLError):
        load_yaml_config(str(config_file))

def test_load_yaml_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_yaml_config("non_existent_file.yaml")

def test_get_config_section_valid():
    config = {"section": {"key": "value"}}
    assert get_config_section(config, "section") == {"key": "value"}

def test_get_config_section_missing():
    config = {"other": "value"}
    assert get_config_section(config, "section") == {}

def test_get_config_section_not_mapping():
    config = {"section": "not a mapping"}
    with pytest.raises(ValueError, match="Config section 'section' must be a mapping."):
        get_config_section(config, "section")

def test_resolve_setting_cli_precedence():
    args = argparse.Namespace(key="cli_value")
    config_section = {"key": "config_value"}
    assert resolve_setting(args, config_section, "key", "default") == "cli_value"

def test_resolve_setting_config_precedence():
    args = argparse.Namespace(key=None)
    config_section = {"key": "config_value"}
    assert resolve_setting(args, config_section, "key", "default") == "config_value"

def test_resolve_setting_default():
    args = argparse.Namespace(key=None)
    config_section = {}
    assert resolve_setting(args, config_section, "key", "default") == "default"

def test_add_common_config_arg():
    parser = argparse.ArgumentParser()
    add_common_config_arg(parser)
    args = parser.parse_args(["--config", "config.yaml"])
    assert args.config == Path("config.yaml")
