#!/usr/bin/env python3
"""
Run the full data curation pipeline, or selected stages.

Usage:
    python run_pipeline.py                         # full pipeline
    python run_pipeline.py --skip quality          # skip quality stages
    python run_pipeline.py --only cluster          # only clustering stages
    python run_pipeline.py --start inference       # from inference onward
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

STAGES = {
    "prefilter": ["01_prefilter_size.py", "02_prefilter_dimensions.py"],
    "quality": ["03_compute_quality.py", "04_analyze_quality.py"],
    "inference": ["05_run_inference.py", "06_build_labels.py"],
    "cluster": ["07_extract_embeddings.py", "08_cluster.py"],
    "review": ["09_cluster_review_gui.py", "10_quality_review_gui.py"],
}

STAGE_ORDER = ["prefilter", "quality", "inference", "cluster", "review"]


def run_script(script_name):
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"  Warning: {script_name} not found, skipping.")
        return True
    print(f"\n{'='*60}")
    print(f"  Running: {script_name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(script_path)],
                            cwd=str(Path(__file__).resolve().parent))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Data Curation Pipeline Orchestrator")
    parser.add_argument("--skip", type=str, nargs="+", default=[],
                        choices=list(STAGES.keys()), help="Stages to skip")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        choices=list(STAGES.keys()), help="Only run these stages")
    parser.add_argument("--start", type=str, default=None,
                        choices=list(STAGES.keys()), help="Start from this stage onward")
    args = parser.parse_args()

    if args.only:
        stages_to_run = [s for s in STAGE_ORDER if s in args.only]
    elif args.start:
        start_idx = STAGE_ORDER.index(args.start)
        stages_to_run = STAGE_ORDER[start_idx:]
    else:
        stages_to_run = list(STAGE_ORDER)

    stages_to_run = [s for s in stages_to_run if s not in args.skip]

    print("Data Curation Pipeline")
    print(f"Stages to run: {stages_to_run}")
    print(f"Skipped: {args.skip if args.skip else 'none'}")

    for stage in stages_to_run:
        for script in STAGES[stage]:
            success = run_script(script)
            if not success:
                print(f"\n  Error running {script}. Pipeline stopped.")
                sys.exit(1)

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
