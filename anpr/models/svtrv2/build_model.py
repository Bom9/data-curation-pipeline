"""Build an SVTRv2 OCR from a checkpoint + sidecar config.

The architecture kwargs (``dims``, ``depths``, ``num_heads``, ``mixer``,
``sub_k``, ``num_convs``) and ``character_dict_path`` live in a JSON
sidecar next to the ``.pth`` — ``<weights_path>.config.json``. Drop in a
different checkpoint plus its matching config and this function loads it
without any code change.

Relative paths in the config (e.g. ``"character_dict_path": "EN_symbol_dict.txt"``)
are resolved against the config file's directory.
"""

import json
from pathlib import Path

from anpr.models.svtrv2.ocr import SVTRv2OCR


def build_ocr(weights_path: str | Path, device: str = "cpu", **overrides) -> SVTRv2OCR:
    """Build an SVTRv2 OCR. ``**overrides`` win over the sidecar config."""
    weights_path = Path(weights_path)
    cfg_path = weights_path.with_suffix(".config.json")
    cfg = json.loads(cfg_path.read_text())

    for k in ("character_dict_path",):
        if k in cfg and not Path(cfg[k]).is_absolute():
            cfg[k] = cfg_path.parent / cfg[k]

    cfg.update(overrides)
    return SVTRv2OCR(weights_path=weights_path, device=device, **cfg)
