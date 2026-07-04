"""
LTA Singapore license plate checksum verification.

Singapore vehicle registration plates follow the format:
    [2-3 letter prefix] [1-4 digit number] [check letter]

The check letter is computed using a weighted sum modulo 19:
    - For 3-letter prefixes, the first letter (typically 'S') is ignored
      for checksum purposes; only the last 2 prefix letters are used.
    - The number is zero-padded to 4 digits.
    - Weights:  [9, 4, 5, 4, 3, 2]
    - Positions: [prefix_letter_1, prefix_letter_2, d1, d2, d3, d4]
    - Each letter value: A=1, B=2, ..., Z=26
    - Remainder of weighted sum mod 19 maps to the check letter.

Reference test cases (all valid):
    ES321P, EA5Y, SMD1001R, SNE588D
"""

import re
from typing import NamedTuple

WEIGHTS = [9, 4, 5, 4, 3, 2]
CHECK_LETTERS = "AZYXUTSRPMLKJHGEDCB"  # index 0-18

# Plate regex: 2-3 uppercase letters, 1-4 digits, 1 uppercase check letter
_PLATE_RE = re.compile(r"^([A-Z]{2,3})(\d{1,4})([A-Z])$")


class PlateParseResult(NamedTuple):
    prefix: str      # full prefix letters (e.g. "SMD")
    number: str       # raw digit string (e.g. "1001")
    check_letter: str # trailing check letter (e.g. "R")


def parse_plate(plate: str) -> PlateParseResult | None:
    """Parse a Singapore plate string into components, or None if invalid format."""
    m = _PLATE_RE.match(plate.strip().upper())
    if not m:
        return None
    return PlateParseResult(prefix=m.group(1), number=m.group(2), check_letter=m.group(3))


def compute_check_letter(prefix: str, number: str) -> str:
    """Compute the LTA check letter for a given prefix and number.

    Args:
        prefix: 2-3 letter prefix (e.g. "ES", "SMD").
        number: 1-4 digit number string (e.g. "321", "1001").

    Returns:
        The expected check letter.
    """
    # Use last 2 letters of prefix for checksum
    p1, p2 = prefix[-2], prefix[-1]
    digits = number.zfill(4)

    values = [
        ord(p1) - ord("A") + 1,
        ord(p2) - ord("A") + 1,
        int(digits[0]),
        int(digits[1]),
        int(digits[2]),
        int(digits[3]),
    ]

    weighted_sum = sum(v * w for v, w in zip(values, WEIGHTS))
    remainder = weighted_sum % 19
    return CHECK_LETTERS[remainder]


def verify_plate(plate: str) -> bool | None:
    """Verify a Singapore license plate's check letter.

    Returns:
        True if checksum matches, False if it doesn't, None if the plate
        doesn't match Singapore format.
    """
    parsed = parse_plate(plate)
    if parsed is None:
        return None
    expected = compute_check_letter(parsed.prefix, parsed.number)
    return parsed.check_letter == expected


if __name__ == "__main__":
    test_cases = ["ES321P", "EA5Y", "SMD1001R", "SNE588D"]
    for plate in test_cases:
        result = verify_plate(plate)
        parsed = parse_plate(plate)
        expected = compute_check_letter(parsed.prefix, parsed.number)
        status = "PASS" if result else "FAIL"
        print(f"{plate}: {status}  (expected={expected}, got={parsed.check_letter})")
