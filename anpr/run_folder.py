"""Run end-to-end inference over a folder of images.

Each image's predictions are written as one JSON line to stdout (or
``--output``). Useful for batch processing or building a result CSV.

Usage:
    python -m anpr.run_folder \\
        --image-dir /path/to/images \\
        --output results.jsonl \\
        --use-checksum-recovery

Recurses into subdirectories. Skips files that aren't images.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from anpr.pipeline import predict_image_e2e
from anpr.models.svtrv2 import build_ocr
from anpr.models.yolo26 import build_detector

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DETECTOR_WEIGHTS = _REPO_ROOT / "weights/yolo26/yolo26n_ft2_motorcycles.pth"
_DEFAULT_OCR_WEIGHTS = _REPO_ROOT / "weights/SVTRv2/anpr_finetune_9_best_375.pth"

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--detector-weights", type=Path, default=_DEFAULT_DETECTOR_WEIGHTS)
    ap.add_argument("--ocr-weights", type=Path, default=_DEFAULT_OCR_WEIGHTS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use-checksum-recovery", action="store_true")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSONL output here. Default: stdout.",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conf-thresh", type=float, default=None, help="Override detector conf.")
    ap.add_argument("--input-size", type=int, default=None, help="Override detector input size (square).")
    args = ap.parse_args()

    det_overrides = {}
    if args.conf_thresh is not None:
        det_overrides["conf_thresh"] = args.conf_thresh
    if args.input_size is not None:
        det_overrides["input_size"] = (args.input_size, args.input_size)

    print(f"Loading detector from {args.detector_weights}", file=sys.stderr)
    detector = build_detector(args.detector_weights, args.device, **det_overrides)
    print(f"Loading OCR from {args.ocr_weights}", file=sys.stderr)
    ocr = build_ocr(args.ocr_weights, args.device)

    paths = iter_images(args.image_dir)
    if args.limit is not None:
        paths = paths[: args.limit]
    print(f"Found {len(paths)} images", file=sys.stderr)

    out_fp = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    t0 = time.time()
    try:
        for i, p in enumerate(paths):
            img = cv2.imread(str(p))
            if img is None:
                print(f"  skip (unreadable): {p}", file=sys.stderr)
                continue
            preds = predict_image_e2e(
                detector, ocr, img, use_checksum_recovery=args.use_checksum_recovery
            )
            record = {
                "image": str(p.relative_to(args.image_dir)),
                "width": img.shape[1],
                "height": img.shape[0],
                "predictions": [asdict(pred) for pred in preds],
            }
            out_fp.write(json.dumps(record) + "\n")
            out_fp.flush()
            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(paths)}] {time.time() - t0:.1f}s", file=sys.stderr)
    finally:
        if args.output:
            out_fp.close()

    print(f"Done in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
