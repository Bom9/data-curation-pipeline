"""Build a YOLO26 detector from a checkpoint + sidecar config.

The architecture kwargs (``nc``, ``depth_mult``, ``width_mult``,
``max_channels``, ``force_c3k``, ``label_file``, ``conf_thresh``, ``add_nms``)
live in a JSON sidecar next to the ``.pth`` — ``<weights_path>.config.json``.
Drop in a different checkpoint plus its matching config and this function
loads it without any code change.

Relative paths in the config (e.g. ``"label_file": "lp_names.txt"``) are
resolved against the config file's directory.
"""

import json
from pathlib import Path

from anpr.models.yolo26.detector import YOLO26Detector


def build_detector(weights_path: str | Path, device: str = "cpu", **overrides) -> YOLO26Detector:
    """Build a YOLO26 detector. ``**overrides`` win over the sidecar config."""
    weights_path = Path(weights_path)
    cfg_path = weights_path.with_suffix(".config.json")
    cfg = json.loads(cfg_path.read_text())

    for k in ("label_file",):
        if k in cfg and not Path(cfg[k]).is_absolute():
            cfg[k] = cfg_path.parent / cfg[k]

    cfg.update(overrides)
    return YOLO26Detector(weights_path=weights_path, device=device, **cfg)
