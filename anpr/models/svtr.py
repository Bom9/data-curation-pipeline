"""SVTRv2 OCR wrapper — loads config, builds OCR, predicts."""

from anpr.config import load_config, resolve_device, resolve_model_path
from anpr.models.svtrv2 import build_ocr
from anpr.models.svtrv2.ocr import SVTRv2OCR


def load_ocr(
    model_path: str | None = None,
    dict_path: str | None = None,
    device: str = "auto",
) -> SVTRv2OCR:
    """Load an SVTRv2 OCR model from a checkpoint + sidecar config.

    Args:
        model_path: Path to .pth checkpoint. If None, reads from config.yaml.
        dict_path: Path to character dictionary. If None, uses config or sidecar.
        device: 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        Configured SVTRv2OCR instance.
    """
    if model_path is None:
        cfg = load_config()
        model_path = resolve_model_path(cfg["inference"]["ocr"]["model"])
        dict_path = resolve_model_path(cfg["inference"]["ocr"]["dict_path"])
        device = resolve_device(cfg["inference"]["ocr"].get("device", device))
    else:
        device = resolve_device(device)

    return build_ocr(model_path, device=device, character_dict_path=dict_path)
