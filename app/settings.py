"""YAML-based configuration loader for ChainInsight.

Loads configs/chaininsight.yaml and merges with environment variable overrides.
This module provides structured access to all configuration values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "chaininsight.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config. Defaults to configs/chaininsight.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def get_data_config() -> dict[str, Any]:
    """Get data generation configuration."""
    return load_config().get("data", {})


def get_model_config(model_name: str | None = None) -> dict[str, Any]:
    """Get model configuration, optionally for a specific model."""
    model_cfg = load_config().get("model", {})
    if model_name:
        return model_cfg.get(model_name, {})
    return model_cfg


def get_eval_config() -> dict[str, Any]:
    """Get evaluation configuration."""
    return load_config().get("evaluation", {})


def get_capacity_config() -> dict[str, Any]:
    """Get capacity planning configuration."""
    return load_config().get("capacity", {})


def get_sensing_config() -> dict[str, Any]:
    """Get demand sensing configuration."""
    return load_config().get("sensing", {})


def get_sop_config() -> dict[str, Any]:
    """Get S&OP simulation configuration."""
    return load_config().get("sop", {})


def get_supply_chain_config() -> dict[str, Any]:
    """Get supply chain configuration."""
    return load_config().get("supply_chain", {})


def get_monitoring_config() -> dict[str, Any]:
    """Get monitoring configuration."""
    return load_config().get("monitoring", {})


def get_chart_config() -> dict[str, Any]:
    """Get chart visualization configuration."""
    return load_config().get("chart", {})
