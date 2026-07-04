"""YOLO26 detector — pure-PyTorch, no ultralytics runtime dep."""

from pathlib import Path

import cv2
import numpy as np
import torch

from anpr.models._types import BoundingBox, DetectionResult
from anpr.models.yolo26.model import YOLO26Model

_DEFAULT_LABEL_FILE = Path(__file__).parent / "coco_names.txt"


def _torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Pure-torch NMS matching torchvision behaviour (no torchvision dependency).

    Adapted from ultralytics ``TorchNMS.nms``.

    Args:
        boxes: ``(N, 4)`` in xyxy format.
        scores: ``(N,)`` confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Indices of boxes to keep.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(0, descending=True)

    keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
    keep_idx = 0
    while order.numel() > 0:
        i = order[0]
        keep[keep_idx] = i
        keep_idx += 1
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp_(min=0) * (yy2 - yy1).clamp_(min=0)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]

    return keep[:keep_idx]


def _load_label_file(path: str | Path) -> list[str]:
    """Read one class name per line from a text file."""
    return Path(path).read_text().strip().splitlines()


class YOLO26Detector:
    """Pure-PyTorch YOLO26 detector.

    Accepts a state-dict ``.pth`` converted from ultralytics via
    ``convert_weights.py``.  No dependency on the ``ultralytics`` package
    at inference time.

    Architecture parameters (``depth_mult``, ``width_mult``,
    ``max_channels``, ``force_c3k``) are passed through from the pipeline
    YAML config, so a single class supports all model scales.

    Args:
        weights_path: Path to the converted ``.pth`` state-dict.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``, …).
        nc: Number of classes (must match the checkpoint).
        depth_mult: Scales block repeat counts.
        width_mult: Scales channel widths.
        max_channels: Maximum channel count.
        force_c3k: Use C3k sub-blocks everywhere (True for m/l/x scales).
        reg_max: DFL bins (1 for YOLO26, 16 for older YOLO variants).
        end2end: Use the NMS-free one-to-one head (YOLO26 default).
        input_size: ``(width, height)`` for letterbox resize.
        conf_thresh: Minimum detection confidence.
        iou_thresh: IoU threshold for NMS (only used when ``end2end=False``).
        label_file: Path to a text file with one class name per line.
            Defaults to the bundled ``coco_names.txt`` (80 COCO classes).
        filter_classes: If given, only return detections whose label is in
            this set.
        add_nms: Apply post-hoc NMS to the end2end head output.  YOLO26
            uses an NMS-free one-to-one matching head, but it can still
            produce near-duplicate boxes.  Enable this to filter them out
            using ``iou_thresh``.  Has no effect when ``end2end=False``
            (that path already runs NMS).
    """

    def __init__(  # noqa: PLR0913
        self,
        weights_path: str | Path,
        device: str = "cpu",
        nc: int = 80,
        depth_mult: float = 0.50,
        width_mult: float = 0.25,
        max_channels: int = 1024,
        force_c3k: bool = False,
        reg_max: int = 1,
        end2end: bool = True,
        input_size: tuple[int, int] = (640, 640),
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        label_file: str | Path | None = None,
        filter_classes: list[str] | None = None,
        add_nms: bool = False,
    ) -> None:
        self._weights_path = Path(weights_path)
        self._device = torch.device(device)
        self._end2end = end2end
        self._add_nms = add_nms
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_w, self.input_h = input_size
        self._class_names = _load_label_file(label_file or _DEFAULT_LABEL_FILE)
        self._filter_classes = set(filter_classes) if filter_classes else None

        self._model = YOLO26Model(
            nc=nc,
            depth_mult=depth_mult,
            width_mult=width_mult,
            max_channels=max_channels,
            force_c3k=force_c3k,
            reg_max=reg_max,
            end2end=end2end,
        )
        self._model.load_ultralytics_weights(str(self._weights_path))
        self._model.to(self._device)
        self._model.eval()

    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        img_h, img_w = frame.shape[:2]
        blob, ratio, pad_w, pad_h = self._letterbox(frame)
        tensor = torch.from_numpy(blob).to(self._device)

        with torch.no_grad():
            output = self._model(tensor)

        if self._end2end:
            return self._postprocess_end2end(output, ratio, pad_w, pad_h, img_h, img_w)
        return self._postprocess_nms(output, ratio, pad_w, pad_h, img_h, img_w)

    # ---- preprocessing ----------------------------------------------------

    def _letterbox(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float, int, int]:
        """Resize keeping aspect ratio, pad to ``input_size`` with gray."""
        img_h, img_w = image.shape[:2]
        ratio = min(self.input_w / img_w, self.input_h / img_h)

        new_w = round(img_w * ratio)
        new_h = round(img_h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = self.input_w - new_w
        dh = self.input_h - new_h
        pad_left = dw // 2
        pad_top = dh // 2

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            dh - pad_top,
            pad_left,
            dw - pad_left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # BGR → RGB, float32 [0, 1], HWC → CHW, add batch dim
        rgb = padded[..., ::-1]
        blob = np.ascontiguousarray(rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]

        return blob, ratio, dw, dh

    # ---- postprocessing ---------------------------------------------------

    def _postprocess_end2end(
        self,
        output: torch.Tensor,
        ratio: float,
        pad_w: int,
        pad_h: int,
        orig_h: int,
        orig_w: int,
    ) -> list[DetectionResult]:
        """Post-process end2end output ``[1, max_det, 6]``.

        YOLO26 uses an NMS-free one-to-one matching head, but it can still
        emit near-identical boxes at different confidence levels.  When
        ``add_nms`` is enabled, a standard IoU-based NMS pass removes these
        duplicates before building the result list.
        """
        raw = output[0]  # [max_det, 6]
        if self._add_nms:
            mask = raw[:, 4] >= self.conf_thresh
            if mask.any():
                candidates = raw[mask]
                kept = _torch_nms(candidates[:, :4], candidates[:, 4], self.iou_thresh)
                raw = candidates[kept]
            else:
                raw = raw[:0]

        dets = raw.cpu().numpy()
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        results: list[DetectionResult] = []
        for det in dets:
            x1, y1, x2, y2, score, cls_id = det
            if score < self.conf_thresh:
                continue

            cls_idx = int(cls_id)
            if cls_idx < 0 or cls_idx >= len(self._class_names):
                continue
            label = self._class_names[cls_idx]
            if self._filter_classes and label not in self._filter_classes:
                continue

            x1 = round(max(0.0, min((x1 - pad_left) / ratio, orig_w)))
            y1 = round(max(0.0, min((y1 - pad_top) / ratio, orig_h)))
            x2 = round(max(0.0, min((x2 - pad_left) / ratio, orig_w)))
            y2 = round(max(0.0, min((y2 - pad_top) / ratio, orig_h)))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                DetectionResult(
                    label=label,
                    confidence=float(score),
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return results

    def _postprocess_nms(
        self,
        output: torch.Tensor,
        ratio: float,
        pad_w: int,
        pad_h: int,
        orig_h: int,
        orig_w: int,
    ) -> list[DetectionResult]:
        """Post-process raw output ``[1, 4+nc, total]`` with NMS."""
        pred = output[0].cpu().numpy()  # [4+nc, total]
        pred = pred.T  # [total, 4+nc]

        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]

        max_scores = scores_all.max(axis=1)
        class_ids = scores_all.argmax(axis=1)

        mask = max_scores > self.conf_thresh
        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        # NMS (cv2 expects top-left xywh)
        tl_xywh = boxes_xywh.copy()
        tl_xywh[:, :2] -= tl_xywh[:, 2:] / 2
        indices = cv2.dnn.NMSBoxes(
            tl_xywh.tolist(),
            max_scores.tolist(),
            self.conf_thresh,
            self.iou_thresh,
        )
        if len(indices) == 0:
            return []
        kept = np.array(indices).flatten()

        # Center xywh → xyxy
        xy = boxes_xywh[kept, :2]
        wh = boxes_xywh[kept, 2:]
        boxes_xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
        scores_kept = max_scores[kept]
        cls_kept = class_ids[kept]

        pad_left = pad_w // 2
        pad_top = pad_h // 2

        results: list[DetectionResult] = []
        for i in range(len(kept)):
            bx1, by1, bx2, by2 = boxes_xyxy[i]
            cls_idx = int(cls_kept[i])
            if cls_idx < 0 or cls_idx >= len(self._class_names):
                continue
            label = self._class_names[cls_idx]
            if self._filter_classes and label not in self._filter_classes:
                continue

            x1 = round(max(0.0, min((bx1 - pad_left) / ratio, orig_w)))
            y1 = round(max(0.0, min((by1 - pad_top) / ratio, orig_h)))
            x2 = round(max(0.0, min((bx2 - pad_left) / ratio, orig_w)))
            y2 = round(max(0.0, min((by2 - pad_top) / ratio, orig_h)))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                DetectionResult(
                    label=label,
                    confidence=float(scores_kept[i]),
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return results
