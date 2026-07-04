"""File size descriptor."""
import os

def compute(image_path: str) -> dict:
    size_bytes = os.path.getsize(image_path)
    return {"file_size_bytes": size_bytes, "file_size_kb": round(size_bytes / 1024, 2)}
