"""Detection and OCR result types.

Lifted verbatim from fast-alpr's infer_eval schemas. Kept as pydantic so
the detector/OCR code can be copied with minimal edits.
"""

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _validate_coords(self) -> "BoundingBox":
        for name in ("x1", "y1", "x2", "y2"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)}")
        if self.x2 <= self.x1:
            raise ValueError(f"x2 must be > x1, got x1={self.x1}, x2={self.x2}")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 must be > y1, got y1={self.y1}, y2={self.y2}")
        return self

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class DetectionResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    label: str
    confidence: float
    bounding_box: BoundingBox
    keypoints: tuple[tuple[float, ...], ...] | None = None

    @model_validator(mode="after")
    def _validate_confidence(self) -> "DetectionResult":
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        return self

    @field_validator("keypoints", mode="before")
    @classmethod
    def _coerce_keypoints(cls, v: Any) -> Any:
        if v is None:
            return None
        return tuple(tuple(kp) for kp in v)


class OcrResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    text: str
    confidence: float
    per_char_confidence: list[float] | None = None

    @model_validator(mode="after")
    def _validate_ocr(self) -> "OcrResult":
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.per_char_confidence is not None:
            if len(self.per_char_confidence) != len(self.text):
                raise ValueError(
                    f"per_char_confidence length ({len(self.per_char_confidence)})"
                    f" != text length ({len(self.text)})"
                )
            bad = {v for v in self.per_char_confidence if not 0.0 <= v <= 1.0}
            if bad:
                raise ValueError(f"per_char_confidence values must be in [0.0, 1.0], got {bad}")
        return self
