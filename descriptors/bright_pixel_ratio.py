"""Bright pixel ratio descriptor."""
import cv2
import numpy as np

def compute(image_path: str, threshold: int = 230) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {"bright_pixel_ratio": float(np.mean(gray > threshold))}
