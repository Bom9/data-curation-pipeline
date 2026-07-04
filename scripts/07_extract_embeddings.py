#!/usr/bin/env python3
"""
Extract DINOv2 CLS token embeddings for all non-excluded images.

Reads:  config.yaml -> paths.images_dir, clustering.embedding.*
Reads:  data/output/02_excluded.json (if exists)
Writes: data/output/embeddings.npy, data/output/image_filenames.npy
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.config import load_config
from anpr.models.dinov2 import DINOv2Extractor
from anpr.utils import ensure_dir, load_json


def main():
    cfg = load_config()
    images_dir = Path(cfg["paths"]["images_dir"])
    output_dir = ensure_dir(cfg["paths"]["output_dir"])

    excluded_path = output_dir / "02_excluded.json"
    excluded = set(load_json(excluded_path)) if excluded_path.exists() else set()

    image_paths = sorted(images_dir.glob("*.jpg"))
    to_process = [p for p in image_paths if p.name not in excluded]

    print(f"Total images:     {len(image_paths):,}")
    print(f"Excluded (pre):   {len(excluded):,}")
    print(f"To embed:         {len(to_process):,}")

    if not to_process:
        print("No images to process.")
        return

    extractor = DINOv2Extractor()
    embeddings, filenames = extractor.extract_batch(to_process)

    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "image_filenames.npy", filenames)

    print(f"\nDone! Embeddings: {embeddings.shape} saved to {output_dir}/")


if __name__ == "__main__":
    main()
