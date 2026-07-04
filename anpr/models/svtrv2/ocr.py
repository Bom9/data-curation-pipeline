"""SVTRv2 OCR implementation using OpenOCR model components."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from anpr.models._types import OcrResult
from anpr.models.svtrv2.decoder import RCTCDecoder
from anpr.models.svtrv2.encoder import SVTRv2LNConvTwo33
from anpr.models.svtrv2.postprocess import CTCLabelDecode

# Aspect-ratio-aware resize targets (width, height).
_BASE_SHAPES: dict[int, tuple[int, int]] = {
    1: (64, 64),
    2: (96, 48),
    3: (112, 40),
    4: (128, 32),
}
_MAX_RATIO = 12


class _SVTRv2Model(nn.Module):
    """Encoder + RCTCDecoder, no training-only branches."""

    def __init__(self, encoder: SVTRv2LNConvTwo33, decoder: RCTCDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SVTRv2OCR:
    """OCR using SVTRv2 with RCTC decoding.

    Architecture parameters (dims, depths, num_heads, mixer, etc.) are
    passed through from the pipeline YAML config, so a single class
    supports all model sizes (T, S, B).

    Args:
        weights_path: Path to the checkpoint file.
        device: Torch device string.
        character_dict_path: Path to character dictionary. Required — must
            match the dict the checkpoint was trained with (it defines the
            classifier output dim). For the LP-finetuned ft9 checkpoint,
            use ``weights/SVTRv2/EN_symbol_dict.txt``.
        use_space_char: Whether to include space in the vocab.
        dims: Per-stage embedding dimensions.
        depths: Per-stage block depths.
        num_heads: Per-stage attention head counts.
        mixer: Per-stage block types (``"Conv"``, ``"Global"``, etc.).
        sub_k: Per-stage downsampling strides.
        use_pos_embed: Whether to use positional embeddings.
        feat2d: Whether to output 2D feature maps from the encoder.
        num_convs: Per-stage conv block counts (optional).
    """

    def __init__(
        self,
        weights_path: str | Path,
        character_dict_path: str | Path,
        device: str = "cpu",
        use_space_char: bool = False,
        pad_to_ratio: bool = False,
        # Encoder architecture params
        dims: list[int] | None = None,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        mixer: list[list[str]] | None = None,
        sub_k: list[list[int]] | None = None,
        use_pos_embed: bool = False,
        feat2d: bool = True,
        num_convs: list[list[int]] | None = None,
    ) -> None:
        self._weights_path = Path(weights_path)
        self._device = torch.device(device)
        self._pad_to_ratio = pad_to_ratio

        self._postprocess = CTCLabelDecode(str(character_dict_path), use_space_char=use_space_char)
        n_classes = len(self._postprocess.character)

        encoder_kwargs: dict[str, Any] = {
            "in_channels": 3,
            "use_pos_embed": use_pos_embed,
            "last_stage": False,
            "feat2d": feat2d,
        }
        if dims is not None:
            encoder_kwargs["dims"] = dims
        if depths is not None:
            encoder_kwargs["depths"] = depths
        if num_heads is not None:
            encoder_kwargs["num_heads"] = num_heads
        if mixer is not None:
            encoder_kwargs["mixer"] = mixer
        if sub_k is not None:
            encoder_kwargs["sub_k"] = sub_k
        if num_convs is not None:
            encoder_kwargs["num_convs"] = num_convs

        encoder = SVTRv2LNConvTwo33(**encoder_kwargs)
        decoder = RCTCDecoder(in_channels=encoder.out_channels, out_channels=n_classes)

        self._model = _SVTRv2Model(encoder, decoder)
        self._load_weights()
        self._model.to(self._device)
        self._model.eval()

        self._normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def _load_weights(self) -> None:
        checkpoint = torch.load(self._weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        mapped: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("Student."):
                k = k[len("Student.") :]

            if k.startswith("decoder.ctc_decoder."):
                mapped["decoder." + k[len("decoder.ctc_decoder.") :]] = v
            elif k.startswith("encoder."):
                mapped[k] = v
            elif k.startswith("decoder."):
                # Skip training-only decoder weights (GTC/SMTR)
                continue
            elif k.startswith("backbone."):
                mapped["encoder." + k[len("backbone.") :]] = v
            elif k.startswith("head.ctc_decoder."):
                mapped["decoder." + k[len("head.ctc_decoder.") :]] = v
            elif k.startswith("head."):
                continue
            else:
                mapped[k] = v

        self._model.load_state_dict(mapped, strict=False)

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Aspect-ratio-aware resize + normalize to [-1, 1].

        When ``pad_to_ratio`` is enabled, the image is zero-padded to
        match the target bucket's aspect ratio before resizing, so
        features are not warped by stretching.
        """
        h, w = image_bgr.shape[:2]
        if h == 0 or w == 0:
            raise ValueError(f"Image has zero dimension: {w}x{h}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        ratio = max(1, round(w / h))
        ratio = min(ratio, _MAX_RATIO)
        if ratio in _BASE_SHAPES:
            target_w, target_h = _BASE_SHAPES[ratio]
        else:
            target_w, target_h = 32 * ratio, 32

        if self._pad_to_ratio:
            # Pad to match target aspect ratio before resizing.
            target_ar = target_w / target_h
            current_ar = w / max(h, 1)
            _, cur_h, cur_w = tensor.shape
            if current_ar < target_ar:
                # Too tall — pad width
                new_w = int(cur_h * target_ar)
                pad_left = (new_w - cur_w) // 2
                pad_right = new_w - cur_w - pad_left
                tensor = torch.nn.functional.pad(tensor, [pad_left, pad_right, 0, 0])
            elif current_ar > target_ar:
                # Too wide — pad height
                new_h = int(cur_w / target_ar)
                pad_top = (new_h - cur_h) // 2
                pad_bottom = new_h - cur_h - pad_top
                tensor = torch.nn.functional.pad(tensor, [0, 0, pad_top, pad_bottom])

        tensor = TF.resize(tensor, [target_h, target_w], interpolation=InterpolationMode.BICUBIC)
        tensor = self._normalize(tensor)
        return tensor.unsqueeze(0)

    def predict(self, cropped_image: np.ndarray) -> OcrResult | None:
        if cropped_image is None or cropped_image.size == 0:
            return None

        tensor = self._preprocess(cropped_image).to(self._device)

        with torch.no_grad():
            preds = self._model(tensor)

        results = self._postprocess(preds)
        if not results:
            return None

        text, confidence, per_char = results[0]
        if not text:
            return None

        return OcrResult(text=text, confidence=confidence, per_char_confidence=per_char)

    def predict_batch(
        self, cropped_images: list[np.ndarray]
    ) -> list[OcrResult | None]:
        """Run OCR on multiple plate crops in one batched forward pass.

        Each crop may have a different aspect ratio so they are preprocessed
        individually, padded to a common size, stacked, then run through
        the model once.

        Args:
            cropped_images: List of BGR plate crops.

        Returns:
            List of OcrResult (or None for empty/failed crops), same order
            as the input list.
        """
        if not cropped_images:
            return []

        # 1) Preprocess each crop w/o batch dim.
        tensors: list[torch.Tensor | None] = []
        for img in cropped_images:
            if img is None or img.size == 0:
                tensors.append(None)
                continue

            h, w = img.shape[:2]
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

            ratio = max(1, round(w / h))
            ratio = min(ratio, _MAX_RATIO)
            if ratio in _BASE_SHAPES:
                target_w, target_h = _BASE_SHAPES[ratio]
            else:
                target_w, target_h = 32 * ratio, 32

            if self._pad_to_ratio:
                target_ar = target_w / target_h
                current_ar = w / max(h, 1)
                if current_ar < target_ar:
                    new_w = int(h * target_ar)
                    pad_left = (new_w - w) // 2
                    pad_right = new_w - w - pad_left
                    t = torch.nn.functional.pad(t, [pad_left, pad_right, 0, 0])
                elif current_ar > target_ar:
                    new_h = int(w / target_ar)
                    pad_top = (new_h - h) // 2
                    pad_bottom = new_h - h - pad_top
                    t = torch.nn.functional.pad(t, [0, 0, pad_top, pad_bottom])

            t = TF.resize(t, [target_h, target_w],
                          interpolation=InterpolationMode.BICUBIC)
            t = self._normalize(t)          # (C, H, W) — NO batch dim
            tensors.append(t)

        # 2) Filter out empties, pad to common size, stack.
        valid: list[tuple[int, torch.Tensor]] = [
            (i, t) for i, t in enumerate(tensors) if t is not None
        ]
        if not valid:
            return [None] * len(cropped_images)

        max_h = max(t.shape[1] for _, t in valid)
        max_w = max(t.shape[2] for _, t in valid)

        batched: list[torch.Tensor] = []
        for _, t in valid:
            dh = max_h - t.shape[1]
            dw = max_w - t.shape[2]
            if dh > 0 or dw > 0:
                t = torch.nn.functional.pad(t, [0, dw, 0, dh])
            batched.append(t)

        batch_tensor = torch.stack(batched).to(self._device)   # (B, 3, H, W)

        # 3) One forward pass.
        with torch.no_grad():
            preds = self._model(batch_tensor)                   # (B, T, n_classes)

        # 4) Postprocess (CTCLabelDecode already handles batch dim).
        raw_results = self._postprocess(preds)     # list of (text, conf, per_char)

        # 5) Map back to original crop order.
        out: list[OcrResult | None] = [None] * len(cropped_images)
        for (orig_idx, _), (text, conf, per_char) in zip(valid, raw_results):
            if text:
                out[orig_idx] = OcrResult(
                    text=text, confidence=conf, per_char_confidence=per_char,
                )
        return out
