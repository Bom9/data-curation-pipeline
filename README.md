# Data Curation Pipeline

End-to-end pipeline for curating ANPR (Automatic Number Plate Recognition) datasets: pre-filtering, quality analysis, OCR inference, DINOv2 embedding, PCA clustering, and human review.

Designed to be dataset-independent: change one `config.yaml` file and the entire pipeline runs on new data.

## Pipeline Overview

```
all_images/ --> [01,02 Pre-filter] --> exclusion list
    |
    +--> [03,04 Quality] --> quality.csv + charts
    |
    +--> [05,06 Inference] --> predictions.jsonl + combined_labels.json
    |
    +--> [07,08 Clustering] --> DINOv2 embeddings + cluster_labels.json
    |
    +--> [09,10 Review] --> human_label annotations
```

## Quickstart

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt

# 2. Get model weights (not in repo)
#    Obtain weights/ folder from authorized personnel, place at repo root.
#    See weights/README.md for the required layout.

# 3. Place your data
#    data/all_images/   <-- plate crop images (.jpg)
#    data/all_labels/   <-- JSON metadata files (.json)

# 4. Edit config.yaml for your dataset paths

# 5. Run the pipeline
python run_pipeline.py

# Or step by step:
python scripts/01_prefilter_size.py
python scripts/02_prefilter_dimensions.py
...
```

## Data Preparation

### Images (`data/all_images/`)

Plain `.jpg` plate crop images. Filename stem must match the label JSON stem (e.g. `sr_2760_rid_1488717_origin_img_3_crop0.jpg`).

### Labels (`data/all_labels/`)

One `.json` per image. Script 02 (dimension prefilter) reads `crop_info.sam3` for vehicle bounding box and class. Required structure:

```json
{
  "crop_info": {
    "sam3": {
      "original_vehicle_bbox": [x1, y1, x2, y2],
      "predicted_class": "motorbike" | "others"
    }
  }
}
```

| Field | Type | Required by | Description |
|---|---|---|---|
| `crop_info.sam3.original_vehicle_bbox` | `[int, int, int, int]` | Script 02 | Vehicle bounding box `[x1, y1, x2, y2]` in original frame pixels |
| `crop_info.sam3.predicted_class` | `str` | Script 02 | `"motorbike"` or `"others"` — determines which dimension metric to apply |

If labels lack these fields, script 02 simply skips that image (no crash). Scripts 03, 05, 07 do not read labels at all — they only need images.

## Requirements

- Python 3.11+
- PyTorch 2.0+ (CUDA, MPS, or CPU)
- Model weights: YOLOv26, SVTRv2 (from authorized personnel), DINOv2-base (auto-downloaded from HuggingFace)

## Configuration

Edit `config.yaml` to switch datasets. See [INDEX.md](INDEX.md) for the full script reference and data schema.

## Repo Layout

- `anpr/` — Shared core (models, checksum, config loader)
- `scripts/` — Pipeline stages (runnable independently)
- `descriptors/` — Image quality descriptor modules
- `config.yaml` — All paths and thresholds
- `run_pipeline.py` — Orchestrator
