"""Shared utilities — image loading, path helpers, JSON I/O."""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load an image as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> dict | list:
    """Load JSON file. Returns empty dict on missing/corrupt file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, FileNotFoundError) as e:
        print(f"Warning: Could not load {path}: {e}", file=sys.stderr)
        return {}


def save_json(path: str | Path, data: dict | list) -> bool:
    """Save JSON file with error handling. Returns True on success."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except OSError as e:
        print(f"Error: Could not save {path}: {e}", file=sys.stderr)
        return False


def image_stem(filename: str) -> str:
    """Return filename without extension: 'img_crop0.jpg' -> 'img_crop0'."""
    return os.path.splitext(filename)[0]
