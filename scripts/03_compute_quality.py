#!/usr/bin/env python3
"""
Compute enabled quality descriptors for every non-excluded image.

Reads:  config.yaml -> paths.images_dir, paths.output_dir, quality.enabled
Reads:  data/output/02_excluded.json (if exists, skip those images)
Writes: data/output/03_quality.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.config import load_config
from anpr.utils import ensure_dir, load_json
from descriptors import compute_all


def main():
    cfg = load_config()
    images_dir = Path(cfg["paths"]["images_dir"])
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    enabled = cfg["quality"]["enabled"]

    # Load exclusion list
    excluded_path = output_dir / "02_excluded.json"
    excluded = set(load_json(excluded_path)) if excluded_path.exists() else set()

    image_paths = sorted(images_dir.glob("*.jpg"))
    to_process = [p for p in image_paths if p.name not in excluded]

    print(f"Total images:     {len(image_paths):,}")
    print(f"Excluded (pre):   {len(excluded):,}")
    print(f"To process:       {len(to_process):,}")
    print(f"Descriptors:      {enabled}")

    out_path = output_dir / "03_quality.csv"
    fieldnames = None

    with open(out_path, "w", newline="") as f:
        writer = None
        for i, img_path in enumerate(to_process):
            result = compute_all(str(img_path), enabled=enabled)

            if writer is None:
                fieldnames = list(result.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            writer.writerow(result)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1:,}/{len(to_process):,} images...")

    print(f"\nDone! {len(to_process):,} images written to {out_path}")


if __name__ == "__main__":
    main()
