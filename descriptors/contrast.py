"""RMS and Michelson contrast descriptors."""
import cv2
import numpy as np

def compute(image_path: str) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rms_contrast = float(np.std(gray))
    min_val, max_val = gray.min(), gray.max()
    if max_val + min_val > 0:
        michelson = float((max_val - min_val) / (max_val + min_val))
    else:
        michelson = 0.0
    return {"rms_contrast": rms_contrast, "michelson_contrast": michelson}
