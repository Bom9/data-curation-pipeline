#!/usr/bin/env python3
"""
Run YOLO detection + SVTRv2 OCR on every non-excluded image.

Reads:  config.yaml -> paths.images_dir, inference.*
Reads:  data/output/02_excluded.json (if exists)
Writes: data/output/predictions.jsonl  (one JSON object per line)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.models.yolo import load_yolo
from anpr.models.svtr import load_ocr
from anpr.config import load_config
from anpr.utils import ensure_dir, load_json, load_image_bgr
from anpr.pipeline import predict_image_e2e


def main():
    cfg = load_config()
    images_dir = Path(cfg["paths"]["images_dir"])
    output_dir = ensure_dir(cfg["paths"]["output_dir"])

    # Load exclusion list
    excluded_path = output_dir / "02_excluded.json"
    excluded = set(load_json(excluded_path)) if excluded_path.exists() else set()

    image_paths = sorted(images_dir.glob("*.jpg"))
    to_process = [p for p in image_paths if p.name not in excluded]

    print(f"Loading models...")
    detector = load_yolo()
    ocr = load_ocr()

    out_path = output_dir / "predictions.jsonl"
    processed = 0

    with open(out_path, "w") as f:
        for img_path in to_process:
            image = load_image_bgr(str(img_path))
            preds = predict_image_e2e(detector, ocr, image)

            entry = {
                "image": img_path.name,
                "predictions": [
                    {
                        "bbox_xyxy": list(p.bbox_xyxy),
                        "text": p.text,
                        "text_raw": p.text_raw,
                        "ocr_confidence": p.ocr_confidence,
                        "per_char_confidence": p.per_char_confidence,
                        "detection_confidence": p.detection_confidence,
                    }
                    for p in preds
                ],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            processed += 1

            if processed % 500 == 0:
                print(f"  Processed {processed:,}/{len(to_process):,} images...")

    print(f"\nDone! {processed:,} images written to {out_path}")


if __name__ == "__main__":
    main()
