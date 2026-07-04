#!/usr/bin/env python3
"""
Identify images smaller than the configured file-size threshold.

Reads:  config.yaml -> paths.images_dir, prefilter.size.max_kb
Writes: data/output/01_size_excluded.json  (list of filenames to exclude)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.config import load_config
from anpr.utils import ensure_dir


def main():
    cfg = load_config()
    images_dir = Path(cfg["paths"]["images_dir"])
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    max_kb = cfg["prefilter"]["size"]["max_kb"]
    max_bytes = int(max_kb * 1024)

    excluded = []
    image_paths = sorted(images_dir.glob("*.jpg"))
    for img_path in image_paths:
        if img_path.stat().st_size < max_bytes:
            excluded.append(img_path.name)

    out_path = output_dir / "01_size_excluded.json"
    with open(out_path, "w") as f:
        json.dump(excluded, f, indent=2)

    print(f"Images scanned:  {len(image_paths):,}")
    print(f"Threshold:       < {max_kb} KB")
    print(f"Excluded (size): {len(excluded):,}")
    print(f"Remaining:       {len(image_paths) - len(excluded):,}")
    print(f"Output:          {out_path}")


if __name__ == "__main__":
    main()
