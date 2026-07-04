"""Config loader — reads config.yaml and resolves all paths."""

import os
from pathlib import Path

import yaml

try:
    import torch
except ImportError:
    torch = None


def _repo_root() -> Path:
    """Return the repo root (parent of anpr/ package)."""
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml and resolve all relative paths against repo root.

    Args:
        path: Path to config.yaml. If None, looks for config.yaml in repo root.

    Returns:
        Dict with all paths resolved to absolute paths.
    """
    if path is None:
        path = _repo_root() / "config.yaml"
    else:
        path = Path(path)

    with open(path) as f:
        cfg = yaml.safe_load(f)

    root = path.resolve().parent

    for key, val in cfg.get("paths", {}).items():
        if val and not os.path.isabs(val):
            cfg["paths"][key] = str(root / val)

    return cfg


def resolve_device(device_spec: str) -> str:
    """Resolve 'auto' device to the best available, or pass through explicit values."""
    if device_spec != "auto":
        return device_spec

    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_path(config_value: str) -> str:
    """Resolve a model path from config (may be relative or HF hub name).

    If the value looks like a HuggingFace model name (contains '/'), return as-is.
    Otherwise resolve relative to repo root.
    """
    if "/" in config_value and not os.path.exists(config_value):
        if not config_value.startswith(("weights/", "./", "../", "/")):
            return config_value  # HF hub name like "facebook/dinov2-base"

    if os.path.isabs(config_value):
        return config_value

    return str(_repo_root() / config_value)
