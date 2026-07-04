"""Image quality descriptor modules.

Each module has a compute(image_path: str) -> dict function.
To register a new descriptor, add its name (matching the module filename)
to the enabled list in config.yaml under the quality section.
"""

import importlib

_DESCRIPTOR_NAMES = [
    "brightness",
    "contrast",
    "laplacian_blur",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "file_size",
]


def compute_all(image_path: str, enabled: list[str] | None = None) -> dict:
    """Run all enabled descriptors on one image, merge results into a single dict."""
    if enabled is None:
        enabled = _DESCRIPTOR_NAMES

    result = {"image_file": image_path}
    for name in enabled:
        try:
            mod = importlib.import_module(f"descriptors.{name}")
            vals = mod.compute(image_path)
            result.update(vals)
        except Exception as e:
            print(f"  Warning: descriptor '{name}' failed for {image_path}: {e}")
            result[name] = None
    return result
