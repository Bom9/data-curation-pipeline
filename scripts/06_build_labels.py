#!/usr/bin/env python3
"""
Merge predictions.jsonl + all_labels + image folder structure
into a unified combined_labels.json with checksum validation.

Reads:  config.yaml -> paths.*
Reads:  data/output/predictions.jsonl
Writes: data/output/combined_labels.json
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.config import load_config
from anpr.checksum import validate_plate
from anpr.utils import ensure_dir


def load_predictions(path):
    preds = {}
    if not os.path.exists(path):
        return preds
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            preds[d["image"]] = d.get("predictions", [])
    return preds


def build_bbox(pred):
    bbox = pred.get("bbox_xyxy", [0, 0, 0, 0])
    return {
        "x1": bbox[0],
        "y1": bbox[1],
        "x2": bbox[2],
        "y2": bbox[3],
        "svtrv2_text": pred.get("text", ""),
        "svtrv2_conf": pred.get("ocr_confidence", 0.0),
    }


def main():
    cfg = load_config()
    labels_dir = Path(cfg["paths"]["labels_dir"])
    images_dir = Path(cfg["paths"]["images_dir"])
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    predictions_file = output_dir / "predictions.jsonl"

    # Scan images for folder assignment
    folder_lookup = {}
    if images_dir.exists():
        for fname in os.listdir(images_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                folder_lookup[fname] = "all_images"

    predictions = load_predictions(predictions_file)
    print(f"Loaded {len(predictions)} prediction entries")

    output = {}
    processed = 0

    for label_path in sorted(labels_dir.glob("*.json")):
        stem = label_path.stem
        img_fname = f"{stem}.jpg"

        folder_label = folder_lookup.get(img_fname, "unknown")
        note = "" if folder_label == "all_images" else folder_label

        img_preds = predictions.get(img_fname, [])

        if img_preds:
            bboxes = [build_bbox(p) for p in img_preds]
            svtrv2_pred = img_preds[0].get("text", "")
            ocr_actual = svtrv2_pred
            chk = validate_plate(ocr_actual)
            ocr_correct = chk
            checksum_valid = chk
        else:
            bboxes = []
            svtrv2_pred = ""
            ocr_actual = ""
            ocr_correct = None
            checksum_valid = None

        entry = {
            "filename": img_fname,
            "bboxes": bboxes,
            "deleted_bboxes": [],
            "svtrv2_pred": svtrv2_pred,
            "ocr_actual": ocr_actual,
            "ocr_correct": ocr_correct,
            "ocr_unsure": False,
            "checksum_valid": checksum_valid,
            "note": note,
            "human_label": None,
        }
        output[stem] = entry
        processed += 1

        if processed % 5000 == 0:
            print(f"  Processed {processed} labels...")

    out_path = output_dir / "combined_labels.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done! {processed} entries written to {out_path}")


if __name__ == "__main__":
    main()
