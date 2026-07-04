"""CTC postprocessing for SVTRv2.

Extracted from OpenOCR (https://github.com/Topdu/OpenOCR).
"""

import numpy as np
import torch


class CTCLabelDecode:
    """Decode CTC output probabilities to text."""

    def __init__(self, character_dict_path: str, use_space_char: bool = False):
        self.character_str: list[str] = []

        with open(character_dict_path, "rb") as fin:
            for line in fin.readlines():
                self.character_str.append(line.decode("utf-8").strip("\n").strip("\r\n"))

        if use_space_char:
            self.character_str.append(" ")

        # CTC blank at index 0
        self.character = ["blank"] + list(self.character_str)

    def __call__(self, preds: torch.Tensor) -> list[tuple[str, float]]:
        preds_np = preds.detach().cpu().numpy()
        preds_idx = preds_np.argmax(axis=2)
        preds_prob = preds_np.max(axis=2)

        results: list[tuple[str, float]] = []
        for batch_idx in range(len(preds_idx)):
            selection = np.ones(len(preds_idx[batch_idx]), dtype=bool)
            # Remove consecutive duplicates
            selection[1:] = preds_idx[batch_idx][1:] != preds_idx[batch_idx][:-1]
            # Remove blank token (index 0)
            selection &= preds_idx[batch_idx] != 0

            char_list = [self.character[idx] for idx in preds_idx[batch_idx][selection]]
            conf_list = preds_prob[batch_idx][selection]

            text = "".join(char_list)
            per_char = [float(c) for c in conf_list]
            confidence = float(np.mean(conf_list)) if len(conf_list) > 0 else 0.0
            results.append((text, confidence, per_char))

        return results
