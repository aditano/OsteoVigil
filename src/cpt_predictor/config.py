from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "default.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config_path:
        override_path = Path(config_path)
        with override_path.open("r", encoding="utf-8") as handle:
            config = _deep_merge(config, yaml.safe_load(handle) or {})

    if overrides:
        config = _deep_merge(config, overrides)

    return config


def ensure_output_dir(output_dir: Optional[str], config: Dict[str, Any]) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        path = ROOT_DIR / config["project"]["output_root"] / "run"
    path.mkdir(parents=True, exist_ok=True)
    return path

