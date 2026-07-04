#!/usr/bin/env python3
"""
Generate histograms, percentile charts, and outlier lists from quality CSV.

Reads:  data/output/03_quality.csv
Writes: data/output/04_quality_report/  (PNG charts + summary.txt)
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.config import load_config
from anpr.utils import ensure_dir


def main():
    cfg = load_config()
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    csv_path = output_dir / "03_quality.csv"
    report_dir = ensure_dir(output_dir / "04_quality_report")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run script 03 first.")
        sys.exit(1)

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No data in CSV.")
        return

    print(f"Loaded {len(rows):,} rows from {csv_path}")

    # Collect numeric columns
    numeric_cols = []
    for col in reader.fieldnames:
        if col == "image_file":
            continue
        try:
            float(rows[0][col])
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue

    print(f"Numeric columns: {numeric_cols}")

    # Generate histogram per numeric column
    for col in numeric_cols:
        values = [float(r[col]) for r in rows if r[col] is not None]
        if not values:
            continue
        values = np.array(values)
        p5, p50, p95 = np.percentile(values, [5, 50, 95])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(values, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(p5, color="red", ls="--", label=f"P5 = {p5:.2f}")
        ax.axvline(p50, color="green", ls="--", label=f"P50 = {p50:.2f}")
        ax.axvline(p95, color="orange", ls="--", label=f"P95 = {p95:.2f}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {col} (n={len(values):,})")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(report_dir / f"hist_{col}.png", dpi=120)
        plt.close()

    # Write summary
    summary = [f"Quality Analysis Summary", f"{'='*40}", f"Images: {len(rows):,}", ""]
    for col in numeric_cols:
        values = [float(r[col]) for r in rows if r[col] is not None]
        if not values:
            continue
        arr = np.array(values)
        summary.append(f"{col}: min={arr.min():.2f} max={arr.max():.2f} mean={arr.mean():.2f} std={arr.std():.2f}")
        p1, p5, p50, p95, p99 = np.percentile(arr, [1, 5, 50, 95, 99])
        summary.append(f"  P1={p1:.2f} P5={p5:.2f} P50={p50:.2f} P95={p95:.2f} P99={p99:.2f}")

    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    with open(report_dir / "summary.txt", "w") as f:
        f.write(summary_text + "\n")

    print(f"\nReport saved to {report_dir}/")


if __name__ == "__main__":
    main()
