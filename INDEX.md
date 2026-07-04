# Data Curation Pipeline — Script & Artifact Index

All scripts in `scripts/`. All artifacts go to `data/output/` (configurable via config.yaml).

## Scripts

| Script | Reads | Writes | Description |
|---|---|---|---|
| `01_prefilter_size.py` | `images_dir` | `01_size_excluded.json` | Identify images below file-size threshold |
| `02_prefilter_dimensions.py` | `labels_dir` + `01_size_excluded.json` | `02_excluded.json` | Filter by class-specific dimension percentiles, merge exclusions |
| `03_compute_quality.py` | `images_dir` + `02_excluded.json` | `03_quality.csv` | Compute brightness, contrast, blur, pixel ratios, file size |
| `04_analyze_quality.py` | `03_quality.csv` | `04_quality_report/` | Histograms, percentile charts, summary stats |
| `05_run_inference.py` | `images_dir` + `02_excluded.json` | `predictions.jsonl` | YOLO detection + SVTRv2 OCR |
| `06_build_labels.py` | predictions + labels + images | `combined_labels.json` | Merge all sources with checksum validation |
| `07_extract_embeddings.py` | `images_dir` + `02_excluded.json` | `embeddings.npy`, `image_filenames.npy` | DINOv2 CLS token embeddings |
| `08_cluster.py` | `embeddings.npy` | `cluster_labels.json` | StandardScaler -> PCA -> KMeans |
| `09_cluster_review_gui.py` | clusters + labels + images | updates `combined_labels.json` | Tkinter cluster review (pass/trash) |
| `10_quality_review_gui.py` | quality + labels + images | updates `combined_labels.json` | Tkinter quality review (pass/trash) |

## Artifacts

| File | Format | Produced by | Description |
|---|---|---|---|
| `01_size_excluded.json` | JSON list | 01 | Filenames excluded by file size |
| `02_excluded.json` | JSON list | 02 | Merged exclusion list (size + dimensions) |
| `03_quality.csv` | CSV | 03 | Quality descriptors per image |
| `04_quality_report/` | PNG + TXT | 04 | Analysis charts and summary |
| `predictions.jsonl` | JSONL | 05 | YOLO + SVTRv2 predictions per image |
| `combined_labels.json` | JSON dict | 06, 09, 10 | Canonical metadata, updated by review GUIs |
| `embeddings.npy` | NPY | 07 | DINOv2 embeddings (N, 768) |
| `image_filenames.npy` | NPY | 07 | Corresponding filenames |
| `cluster_labels.json` | JSON dict | 08 | {stem: cluster_id} mapping |

## combined_labels.json Schema

```json
{
  "image_stem": {
    "filename": "image_stem.jpg",
    "bboxes": [{"x1": 0, "y1": 0, "x2": 100, "y2": 50, "svtrv2_text": "ABC123", "svtrv2_conf": 0.95}],
    "deleted_bboxes": [],
    "svtrv2_pred": "ABC123",
    "ocr_actual": "ABC123",
    "ocr_correct": true,
    "ocr_unsure": false,
    "checksum_valid": true,
    "note": "",
    "human_label": "pass"
  }
}
```

## Rebuild Order

When running step by step:

```bash
cd scripts
python 01_prefilter_size.py
python 02_prefilter_dimensions.py
python 03_compute_quality.py     # can run parallel with 05, 07
python 04_analyze_quality.py
python 05_run_inference.py       # can run parallel with 03, 07
python 06_build_labels.py
python 07_extract_embeddings.py  # can run parallel with 03, 05
python 08_cluster.py
python 09_cluster_review_gui.py
python 10_quality_review_gui.py
```

Steps 03, 05, 07 can run in parallel after 02 completes.
