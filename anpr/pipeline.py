"""End-to-end license-plate inference: detect → crop → OCR → optional checksum repair."""

from dataclasses import dataclass

import numpy as np

from anpr.models.svtrv2 import SVTRv2OCR
from anpr.models.yolo26 import YOLO26Detector
from anpr.post_process import apply_checksum_recovery


@dataclass
class PlatePrediction:
    """One detected plate's prediction.

    Attributes:
        text: Final plate string (after checksum repair if enabled).
        text_raw: Raw OCR string before any checksum repair.
        ocr_confidence: SVTRv2 aggregate confidence in [0, 1].
        per_char_confidence: Per-character confidence list, or None if unavailable.
        bbox_xyxy: ``(x1, y1, x2, y2)`` pixel coordinates of the detection.
        detection_confidence: YOLO26 detection score in [0, 1].
    """

    text: str
    text_raw: str
    ocr_confidence: float
    per_char_confidence: list[float] | None
    bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float


def predict_image_e2e(
    detector: YOLO26Detector,
    ocr: SVTRv2OCR,
    image_bgr: np.ndarray,
    *,
    use_checksum_recovery: bool = False,
) -> list[PlatePrediction]:
    """Run the full pipeline on a single BGR image.

    Returns one ``PlatePrediction`` per detected plate, sorted by detection
    confidence (descending). Empty list if the detector finds nothing or
    every crop fails OCR.
    """
    detections = detector.predict(image_bgr)
    detections.sort(key=lambda d: -d.confidence)

    out: list[PlatePrediction] = []
    for det in detections:
        bb = det.bounding_box
        crop = image_bgr[int(bb.y1) : int(bb.y2), int(bb.x1) : int(bb.x2)]
        if crop.size == 0:
            continue
        ocr_res = ocr.predict(crop)
        if ocr_res is None:
            continue

        text_raw = ocr_res.text
        text = text_raw
        if use_checksum_recovery:
            text = apply_checksum_recovery(text_raw, ocr_res.per_char_confidence)
        out.append(
            PlatePrediction(
                text=text,
                text_raw=text_raw,
                ocr_confidence=ocr_res.confidence,
                per_char_confidence=ocr_res.per_char_confidence,
                bbox_xyxy=(bb.x1, bb.y1, bb.x2, bb.y2),
                detection_confidence=det.confidence,
            )
        )
    return out
