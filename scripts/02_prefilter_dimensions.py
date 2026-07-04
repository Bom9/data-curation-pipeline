#!/usr/bin/env python3
"""
Identify images in the bottom percentile by class-specific dimensions,
merge with size exclusions from script 01 into a single exclusion list.

Reads:  config.yaml -> paths.labels_dir, prefilter.dimensions
Reads:  data/output/01_size_excluded.json (if exists)
Writes: data/output/02_excluded.json  (merged exclusion list)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.config import load_config
from anpr.utils import ensure_dir, load_json


def percentile(values, percent):
    values = sorted(values)
    if not values:
        return 0
    index = (len(values) - 1) * (percent / 100)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def main():
    cfg = load_config()
    labels_dir = Path(cfg["paths"]["labels_dir"])
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    dims_cfg = cfg["prefilter"]["dimensions"]

    # Load size exclusions from script 01
    size_excluded_path = output_dir / "01_size_excluded.json"
    size_excluded = set(load_json(size_excluded_path) if size_excluded_path.exists() else [])

    # Collect dimension data from label JSONs
    rows = []
    for label_path in sorted(labels_dir.glob("*.json")):
        with label_path.open() as f:
            data = json.load(f)

        sam3 = data.get("crop_info", {}).get("sam3", {})
        bbox = sam3.get("original_vehicle_bbox")
        if bbox is None:
            bbox = data.get("bbox", data.get("vehicle_bbox"))

        if bbox is None or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = bbox[:4]
        predicted_class = sam3.get("predicted_class", "others")
        group = "motorbike" if predicted_class == "motorbike" else "others"

        rows.append({
            "image_file": f"{label_path.stem}.jpg",
            "group": group,
            "width": abs(x2 - x1),
            "height": abs(y2 - y1),
        })

    # Compute thresholds per group
    dim_excluded = set()
    for group, group_cfg in dims_cfg.items():
        metric = group_cfg["metric"]
        pct = group_cfg["percentile_max"]
        group_values = [r[metric] for r in rows if r["group"] == group]
        threshold = percentile(group_values, pct)
        print(f"{group.title()}: {metric} <= {threshold:.2f} (P{pct})")

        for r in rows:
            if r["group"] == group and r[metric] <= threshold:
                dim_excluded.add(r["image_file"])

    combined = sorted(size_excluded | dim_excluded)
    out_path = output_dir / "02_excluded.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Size excluded:     {len(size_excluded):,}")
    print(f"Dimension excluded: {len(dim_excluded):,}")
    print(f"Total excluded:    {len(combined):,}")
    print(f"Output:            {out_path}")


if __name__ == "__main__":
    main()
