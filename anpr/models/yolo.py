"""YOLO detector wrapper — loads config, builds detector, predicts."""

from pathlib import Path

from anpr.config import load_config, resolve_device, resolve_model_path
from anpr.models.yolo26 import build_detector
from anpr.models.yolo26.detector import YOLO26Detector


def load_yolo(
    model_path: str | None = None,
    device: str = "auto",
    conf_threshold: float = 0.25,
    label_file: str | None = None,
) -> YOLO26Detector:
    """Load a YOLO detector from a checkpoint + sidecar config.

    Args:
        model_path: Path to .pth checkpoint. If None, reads from config.yaml.
        device: 'auto', 'cuda', 'mps', or 'cpu'.
        conf_threshold: Minimum detection confidence.
        label_file: Path to label names file. If None, inferred from checkpoint dir.

    Returns:
        Configured YOLO26Detector instance.
    """
    if model_path is None:
        cfg = load_config()
        model_path = resolve_model_path(cfg["inference"]["yolo"]["model"])
        conf_threshold = cfg["inference"]["yolo"].get("conf_threshold", conf_threshold)
        device = resolve_device(cfg["inference"]["yolo"].get("device", device))
    else:
        device = resolve_device(device)

    if label_file is None:
        label_dir = Path(model_path).parent
        candidate = label_dir / "lp_names.txt"
        if candidate.exists():
            label_file = str(candidate)

    return build_detector(
        model_path,
        device=device,
        conf_thresh=conf_threshold,
        label_file=label_file,
    )
