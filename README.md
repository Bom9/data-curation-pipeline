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

```
data-curation-pipeline/
├── config.yaml                 # all paths and thresholds — change this per dataset
├── README.md                   # this file
├── INDEX.md                    # every script, what it reads, what it writes
├── requirements.txt            # frozen Python dependencies
├── run_pipeline.py             # optional orchestrator (--skip, --only, --start)
├── anpr/                       # shared core (dataset-independent)
│   ├── config.py               # load_config() — reads config.yaml, resolves paths
│   ├── checksum.py             # SG plate checksum logic
│   ├── utils.py                # image loading, JSON I/O, ensure_dir
│   ├── pipeline.py             # end-to-end detect → OCR inference
│   ├── models/
│   │   ├── _types.py           # BoundingBox, DetectionResult, OcrResult (pydantic)
│   │   ├── yolo.py             # YOLOv26 detector wrapper (config-integrated)
│   │   ├── yolo26/             # YOLOv26 model implementation
│   │   ├── svtr.py             # SVTRv2 OCR wrapper (config-integrated)
│   │   ├── svtrv2/             # SVTRv2 model implementation
│   │   └── dinov2.py           # DINOv2 embedding extractor
│   └── post_process/
│       ├── checksum_recovery.py
│       └── lta_checksum.py
├── descriptors/                # quality descriptor modules (reusable)
│   ├── brightness.py
│   ├── contrast.py
│   ├── laplacian_blur.py
│   ├── dark_pixel_ratio.py
│   ├── bright_pixel_ratio.py
│   └── file_size.py
├── scripts/                    # pipeline stages (runnable independently)
│   ├── 01_prefilter_size.py
│   ├── 02_prefilter_dimensions.py
│   ├── 03_compute_quality.py
│   ├── 04_analyze_quality.py
│   ├── 05_run_inference.py
│   ├── 06_build_labels.py
│   ├── 07_extract_embeddings.py
│   ├── 08_cluster.py
│   ├── 09_cluster_review_gui.py
│   └── 10_quality_review_gui.py
├── weights/                    # (gitignored) model weights
│   └── README.md               # expected layout documentation
└── data/                       # (gitignored contents) images, labels, outputs
    ├── all_images/             # plate crop images (.jpg)
    ├── all_labels/             # JSON metadata files (.json)
    └── output/                 # pipeline writes all artifacts here
```
