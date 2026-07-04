"""Singapore LTA license plate checksum verification.

Format: [1-3 letter prefix] [1-4 digit number] [check letter]

The check letter is computed using a weighted sum modulo 19:
  - For 3-letter prefixes, the first letter (typically 'S') is ignored
  - The number is zero-padded to 4 digits
  - Weights: [9, 4, 5, 4, 3, 2]
  - Remainder maps to check letter via lookup table

Reference test cases (all valid): ES321P, EA5Y, SMD1001R, SNE588D
"""

import re

WEIGHTS = [9, 4, 5, 4, 3, 2]
CHECKSUM_TABLE = "AZYXUTSRPMLKJHGEDCB"

_PLATE_RE = re.compile(r"^([A-Z]{1,3})(\d{1,4})([A-Z])?$")


def compute_check_letter(prefix: str, digits: str) -> str:
    """Compute the LTA check letter for a given prefix and number."""
    if len(prefix) == 3:
        prefix = prefix[1:]
    elif len(prefix) == 1:
        prefix = " " + prefix

    letter_vals = [ord(c) - 64 for c in prefix]  # A=1, B=2, ...
    digit_vals = [int(d) for d in digits.zfill(4)]

    total = sum(v * w for v, w in zip(letter_vals + digit_vals, WEIGHTS))
    return CHECKSUM_TABLE[total % 19]


def validate_plate(plate: str) -> bool | None:
    """Verify a Singapore license plate check letter.

    Returns True/False/None (None = not SG format).
    """
    plate = plate.strip().upper().replace("-", "").replace(" ", "")
    m = _PLATE_RE.match(plate)
    if not m:
        return None

    prefix = m.group(1)
    digits = m.group(2)
    given_check = m.group(3)

    if given_check is None:
        return None

    expected = compute_check_letter(prefix, digits)
    return given_check == expected


def parse_plate(plate: str) -> tuple[str, str, str] | None:
    """Parse a Singapore plate into (prefix, digits, check_letter)."""
    plate = plate.strip().upper().replace("-", "").replace(" ", "")
    m = _PLATE_RE.match(plate)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3) or ""
