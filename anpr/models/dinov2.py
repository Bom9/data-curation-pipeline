"""DINOv2 embedding extractor — loads from HF hub or local path."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from anpr.config import load_config, resolve_device, resolve_model_path


class DINOv2Extractor:
    """Extract DINOv2 CLS token embeddings from images.

    Args:
        model_name: HF hub name (e.g. "facebook/dinov2-base") or local path.
        device: 'auto', 'cuda', 'mps', or 'cpu'.
        batch_size: Images per forward pass.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        batch_size: int = 64,
    ):
        if model_name is None:
            cfg = load_config()
            model_name = resolve_model_path(cfg["clustering"]["embedding"]["model"])
            device = resolve_device(cfg["clustering"]["embedding"].get("device", device))
            batch_size = cfg["clustering"]["embedding"].get("batch_size", batch_size)
        else:
            device = resolve_device(device)

        self._device = torch.device(device)
        self._batch_size = batch_size

        print(f"Loading DINOv2 model: {model_name} ...")
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        print(f"  Model loaded on {self._device}")

    def extract_one(self, image_path: str | Path) -> np.ndarray:
        """Extract embedding for a single image. Returns (768,) float32 array."""
        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        cls_embed = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
        return cls_embed.cpu().numpy().squeeze(0).astype(np.float32)

    def extract_batch(self, image_paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings for a list of image paths.

        Returns:
            (embeddings, filenames) where embeddings is (N, 768) and
            filenames is (N,) string array of successfully processed filenames.
        """
        all_embeddings = []
        all_filenames = []

        for i in range(0, len(image_paths), self._batch_size):
            batch_paths = image_paths[i : i + self._batch_size]

            batch_pil = []
            batch_names = []
            for p in batch_paths:
                try:
                    with Image.open(p) as img:
                        batch_pil.append(img.convert("RGB"))
                    batch_names.append(str(Path(p).name))
                except Exception as e:
                    print(f"  WARNING: Could not load {p}: {e}", file=sys.stderr)
                    continue

            if not batch_pil:
                continue

            inputs = self._processor(images=batch_pil, return_tensors="pt").to(self._device)

            with torch.inference_mode():
                outputs = self._model(**inputs)

            cls_embeds = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
            cls_embeds = cls_embeds.cpu().numpy().astype(np.float32)

            for j, fname in enumerate(batch_names):
                all_embeddings.append(cls_embeds[j])
                all_filenames.append(fname)

            if (i // self._batch_size + 1) % 10 == 0:
                print(f"  Processed {len(all_embeddings)}/{len(image_paths)} images...")

        return np.array(all_embeddings), np.array(all_filenames)
