"""YOLO26 model architecture — pure PyTorch, no ultralytics dependency.

The architecture matches ``ultralytics/cfg/models/26/yolo26.yaml`` exactly so
that pretrained weights (converted via ``convert_weights.py``) load directly.

Architecture parameters (``depth_mult``, ``width_mult``, ``max_channels``,
``force_c3k``) are passed through from the pipeline YAML config, so a single
class supports all model scales (n / s / m / l / x).
"""

import copy
import math
from typing import ClassVar

import torch
from torch import nn

from anpr.models.yolo26.modules import (
    C2PSA,
    DFL,
    SPPF,
    C3k2,
    Conv,
    DWConv,
    dist2bbox,
    make_anchors,
)


def _ch(c: int, width_mult: float, max_channels: int) -> int:
    """Scale channel count and round to nearest multiple of 8."""
    return math.ceil(min(c, max_channels) * width_mult / 8) * 8


def _reps(n: int, depth_mult: float) -> int:
    """Scale repeat count, minimum 1."""
    return max(round(n * depth_mult), 1)


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------


class DetectHead(nn.Module):
    """YOLO26 detection head (end-to-end NMS-free by default).

    For *end2end* mode the model produces final detections via a one-to-one
    assignment head — no NMS is needed.  For non-end2end mode the raw
    per-anchor predictions are returned for external NMS.
    """

    max_det = 300

    def __init__(self, nc: int = 80, reg_max: int = 1, end2end: bool = True, ch: tuple = ()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4

        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))

        # One-to-many heads (used during training; kept for weight compat).
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        self._end2end = end2end
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

        # Stride is set after construction by YOLO26Model.
        self.stride = [8.0, 16.0, 32.0]

        # Runtime caches (recomputed on shape change).
        self._cached_shape = None
        self._cached_anchors = torch.empty(0)
        self._cached_strides = torch.empty(0)

    # ---- public -----------------------------------------------------------

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        if self._end2end:
            return self._forward_end2end(x)
        return self._forward_raw(x)

    # ---- end2end path (NMS-free) ------------------------------------------

    def _forward_end2end(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return ``[B, max_det, 6]`` with ``[x1, y1, x2, y2, score, cls]``."""
        bs = x[0].shape[0]
        boxes = torch.cat(
            [self.one2one_cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [self.one2one_cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        dbox = self._decode(boxes, x)  # [B, 4, total] in xyxy
        y = torch.cat((dbox, scores.sigmoid()), dim=1)  # [B, 4+nc, total]
        y = y.permute(0, 2, 1)  # [B, total, 4+nc]
        return self._topk(y)

    def _topk(self, y: torch.Tensor) -> torch.Tensor:
        """Select top-k detections from ``[B, total, 4+nc]``."""
        boxes, scores = y.split([4, self.nc], dim=-1)
        best_scores, best_cls = scores.max(dim=-1)  # [B, total]
        k = min(self.max_det, best_scores.shape[1])
        topk_scores, topk_idx = best_scores.topk(k, dim=1)
        idx = topk_idx.unsqueeze(-1)
        topk_boxes = boxes.gather(1, idx.expand(-1, -1, 4))
        topk_cls = best_cls.gather(1, topk_idx).unsqueeze(-1).float()
        return torch.cat([topk_boxes, topk_scores.unsqueeze(-1), topk_cls], dim=-1)

    # ---- raw path (requires external NMS) ---------------------------------

    def _forward_raw(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return ``[B, 4+nc, total]`` (xywh boxes + sigmoid scores)."""
        bs = x[0].shape[0]
        boxes = torch.cat(
            [self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        dbox = self._decode(boxes, x)  # [B, 4, total] in xywh
        return torch.cat((dbox, scores.sigmoid()), dim=1)

    # ---- shared -----------------------------------------------------------

    def _decode(self, boxes: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        shape = feats[0].shape
        if self._cached_shape != shape:
            anchors, strides = make_anchors(feats, self.stride, 0.5)
            self._cached_anchors = anchors.transpose(0, 1)
            self._cached_strides = strides.transpose(0, 1)
            self._cached_shape = shape
        return (
            dist2bbox(
                self.dfl(boxes),
                self._cached_anchors.unsqueeze(0),
                xywh=not self._end2end,
                dim=1,
            )
            * self._cached_strides
        )


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class YOLO26Model(nn.Module):
    """Complete YOLO26 detection model (backbone + neck + head).

    Architecture mirrors ``yolo26.yaml``.  All sizing parameters are
    explicit so that each model scale is fully described by its YAML
    config (no hidden ``scale`` lookup).

    Args:
        nc: Number of classes.
        depth_mult: Scales block repeat counts.
        width_mult: Scales channel widths.
        max_channels: Maximum channel count (caps width scaling).
        force_c3k: Use C3k sub-blocks for *all* C3k2 layers (required
            for m / l / x scales; n / s use plain Bottleneck in early
            layers).
        reg_max: DFL bins (1 for YOLO26, 16 for older YOLO variants).
        end2end: NMS-free one-to-one detection head (YOLO26 default).
    """

    def __init__(
        self,
        nc: int = 80,
        depth_mult: float = 0.50,
        width_mult: float = 0.25,
        max_channels: int = 1024,
        force_c3k: bool = False,
        reg_max: int = 1,
        end2end: bool = True,
    ):
        super().__init__()

        def ch(c):
            return _ch(c, width_mult, max_channels)

        def reps(n):
            return _reps(n, depth_mult)

        # --- Backbone (layers 0-10) ----------------------------------------
        self.b0 = Conv(3, ch(64), 3, 2)
        self.b1 = Conv(ch(64), ch(128), 3, 2)
        self.b2 = C3k2(ch(128), ch(256), reps(2), c3k=force_c3k, e=0.25)
        self.b3 = Conv(ch(256), ch(256), 3, 2)
        self.b4 = C3k2(ch(256), ch(512), reps(2), c3k=force_c3k, e=0.25)
        self.b5 = Conv(ch(512), ch(512), 3, 2)
        self.b6 = C3k2(ch(512), ch(512), reps(2), c3k=True)
        self.b7 = Conv(ch(512), ch(1024), 3, 2)
        self.b8 = C3k2(ch(1024), ch(1024), reps(2), c3k=True)
        self.b9 = SPPF(ch(1024), ch(1024), 5, 3, True)
        self.b10 = C2PSA(ch(1024), ch(1024), reps(2))

        # --- Neck (layers 11-22) -------------------------------------------
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down path
        self.n13 = C3k2(ch(1024) + ch(512), ch(512), reps(2), c3k=True)
        self.n16 = C3k2(ch(512) + ch(512), ch(256), reps(2), c3k=True)

        # Bottom-up path
        self.n17 = Conv(ch(256), ch(256), 3, 2)
        self.n19 = C3k2(ch(256) + ch(512), ch(512), reps(2), c3k=True)
        self.n20 = Conv(ch(512), ch(512), 3, 2)
        self.n22 = C3k2(
            ch(512) + ch(1024),
            ch(1024),
            reps(1),
            c3k=True,
            e=0.5,
            attn=True,
        )

        # --- Detect head (layer 23) ----------------------------------------
        self.detect = DetectHead(
            nc=nc,
            reg_max=reg_max,
            end2end=end2end,
            ch=(ch(256), ch(512), ch(1024)),
        )

    # ---- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        b4 = self.b4(x)
        x = self.b5(b4)
        b6 = self.b6(x)
        x = self.b7(b6)
        x = self.b8(x)
        x = self.b9(x)
        b10 = self.b10(x)

        # Neck — top-down
        x = torch.cat([self.up(b10), b6], 1)
        n13 = self.n13(x)
        x = torch.cat([self.up(n13), b4], 1)
        p3 = self.n16(x)

        # Neck — bottom-up
        x = torch.cat([self.n17(p3), n13], 1)
        p4 = self.n19(x)
        x = torch.cat([self.n20(p4), b10], 1)
        p5 = self.n22(x)

        return self.detect([p3, p4, p5])

    # ---- weight loading ---------------------------------------------------

    # Mapping from ultralytics sequential indices → our named attributes.
    _KEY_MAP: ClassVar[dict[str, str]] = {
        "model.0.": "b0.",
        "model.1.": "b1.",
        "model.2.": "b2.",
        "model.3.": "b3.",
        "model.4.": "b4.",
        "model.5.": "b5.",
        "model.6.": "b6.",
        "model.7.": "b7.",
        "model.8.": "b8.",
        "model.9.": "b9.",
        "model.10.": "b10.",
        # 11 = Upsample, 12 = Concat (no parameters)
        "model.13.": "n13.",
        # 14 = Upsample, 15 = Concat
        "model.16.": "n16.",
        "model.17.": "n17.",
        # 18 = Concat
        "model.19.": "n19.",
        "model.20.": "n20.",
        # 21 = Concat
        "model.22.": "n22.",
        "model.23.": "detect.",
    }

    def load_ultralytics_weights(self, weights_path: str) -> None:
        """Load a state-dict ``.pth`` file produced by ``convert_weights.py``.

        Keys are remapped from the ultralytics sequential numbering
        (``model.0.*``, ``model.1.*``, …) to the named layers used here.
        Missing / unexpected keys are silently ignored so that one-to-many
        head weights (unused at inference) and training-only keys are skipped.
        """
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Ultralytics training checkpoints wrap the model inside a dict.
        if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            state_dict = ckpt["model"].state_dict()
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Already a flat state-dict (e.g. from convert_weights.py).
            state_dict = ckpt

        remapped: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            for old_prefix, new_prefix in self._KEY_MAP.items():
                if k.startswith(old_prefix):
                    remapped[new_prefix + k[len(old_prefix) :]] = v
                    break

        info = self.load_state_dict(remapped, strict=False)
        if info.unexpected_keys:
            pass  # one-to-many head / training-only params — expected
