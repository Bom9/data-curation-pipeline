from anpr.checksum import compute_check_letter, parse_plate, validate_plate
from anpr.config import load_config, resolve_device, resolve_model_path
from anpr.models import (
    BoundingBox,
    DetectionResult,
    DINOv2Extractor,
    OcrResult,
    SVTRv2OCR,
    YOLO26Detector,
    build_detector,
    build_ocr,
)
from anpr.models.yolo import load_yolo
from anpr.models.svtr import load_ocr
from anpr.post_process import apply_checksum_recovery, verify_plate
from anpr.utils import ensure_dir, image_stem, load_image_bgr, load_json, save_json
