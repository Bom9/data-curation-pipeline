# Weights

Model weights are not distributed with this repo. Obtain the contents of this folder from authorized personnel and place them here before running any inference-dependent step.

## Expected layout

```
weights/
├── yolo26/
│   ├── yolo26n_ft2_motorcycles.pth         # fine-tuned detector weights (required)
│   ├── yolo26n_ft2_motorcycles.config.json # model architecture config (required)
│   └── lp_names.txt                        # class labels (required)
├── SVTRv2/
│   ├── anpr_finetune_9_best_375.pth        # fine-tuned OCR weights (required)
│   ├── anpr_finetune_9_best_375.config.json # model architecture config (required)
│   ├── EN_symbol_dict.txt                  # character dictionary (required)
│   ├── substitution_map_ft9_375.json       # OCR error substitution map
│   └── confusion_matrix_ft9_375.json       # OCR confusion matrix
└── dinov2_model/                           # (optional) local DINOv2-base copy
    ├── model.safetensors
    ├── config.json
    └── preprocessor_config.json
```

## DINOv2

By default the pipeline downloads `facebook/dinov2-base` from HuggingFace automatically on first run. If you have a local copy (e.g. for offline machines), place it here and update `clustering.embedding.model` in config.yaml.
