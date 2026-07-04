"""SG plate checksum-based OCR error recovery (v4).

Single-character substitution gated by per-character confidence and a
data-driven substitution map. Replaces an OCR prediction with a checksum-
valid alternative when:
  * the prediction looks like an SG plate
  * the last character (check letter) is high-confidence
  * exactly one other position has low confidence
  * a substitution from the confusion map yields a plate that passes the
    LTA checksum

If no valid candidate is found, the original prediction is returned.
"""

import json
import re
from pathlib import Path

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WEIGHTS = [9, 4, 5, 4, 3, 2]
CHECKSUM_TABLE = list("AZYXUTSRPMLKJHGEDCB")
SG_PLATE_RE = re.compile(r"^([A-Z]{1,3})(\d{1,4})([A-Z])$")
SG_PREFIX_CHARS = set("SFGEXPYQCRT")

# Sub map lives next to the OCR checkpoint it was derived from. Walks
# from anpr/post_process/checksum_recovery.py up 3 levels to repo root
# then into weights/SVTRv2/. Pass ``path=`` to override.
_DEFAULT_SUB_MAP_PATH = (
    Path(__file__).resolve().parents[2]
    / "weights" / "SVTRv2" / "substitution_map_ft9_375.json"
)

SubMap = dict[str, list[tuple[str, float]]]


def verify_sg(plate: str) -> bool | None:
    """True if plate passes LTA checksum; False if it doesn't; None if not SG-format."""
    plate = plate.strip().upper().replace("-", "").replace(" ", "")
    m = SG_PLATE_RE.match(plate)
    if not m:
        return None
    prefix, digits_str, cc = m.group(1), m.group(2), m.group(3)
    if len(prefix) == 1:
        pv = [0, ALPHABET.index(prefix[0]) + 1]
    elif len(prefix) == 2:
        pv = [ALPHABET.index(c) + 1 for c in prefix]
    else:
        pv = [ALPHABET.index(c) + 1 for c in prefix[1:]]
    dv = [0] * (4 - len(digits_str)) + [int(c) for c in digits_str]
    ws = sum(w * v for w, v in zip(WEIGHTS, pv + dv))
    return cc == CHECKSUM_TABLE[ws % 19]


def load_substitution_map(path: Path | str | None = None) -> SubMap:
    """Load substitution map JSON into ``{char: [(alt_char, p_error), ...]}``.

    Default path is ``substitution_map_ft9_375.json`` next to this module —
    built for the ``anpr_finetune_9 / best_375.pth`` OCR checkpoint.

    The JSON file has shape ``{char: [{"char": alt, "p_error": float}, ...]}``.
    Entries are kept in file order (already sorted by descending p_error).
    """
    p = Path(path) if path is not None else _DEFAULT_SUB_MAP_PATH
    raw = json.loads(p.read_text())
    return {k: [(e["char"], e["p_error"]) for e in v] for k, v in raw.items()}


def apply_checksum_recovery(
    pred_text: str,
    per_char_conf: list[float] | None,
    sub_map: SubMap | None = None,
    conf_threshold: float = 0.99,
) -> str:
    """v4 single-plate checksum recovery.

    Returns the corrected plate string, or ``pred_text`` unchanged if no
    valid recovery is found / the plate isn't a recovery candidate.

    The lazy default loads ``substitution_map.json`` next to this file on
    first call; passing ``sub_map`` explicitly is faster for batches.
    """
    if not pred_text or per_char_conf is None:
        return pred_text
    if pred_text[0] not in SG_PREFIX_CHARS:
        return pred_text
    if len(per_char_conf) < 2 or per_char_conf[-1] < 0.99:
        return pred_text
    if SG_PLATE_RE.match(pred_text) is None:
        return pred_text
    if verify_sg(pred_text) is True:
        return pred_text

    if sub_map is None:
        sub_map = load_substitution_map()

    # Positions with conf < threshold, excluding the last (anchor) char,
    # sorted by ascending confidence (least confident first).
    positions = [
        (i, per_char_conf[i])
        for i in range(0, min(len(pred_text) - 1, len(per_char_conf) - 1))
        if per_char_conf[i] < conf_threshold
    ]
    positions.sort(key=lambda x: x[1])

    for pos, _ in positions:
        for alt, _ in sub_map.get(pred_text[pos], []):
            cand = pred_text[:pos] + alt + pred_text[pos + 1 :]
            if SG_PLATE_RE.match(cand) and verify_sg(cand):
                return cand
    return pred_text
