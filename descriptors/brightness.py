"""Mean brightness descriptor."""
import cv2

def compute(image_path: str) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {"brightness": float(gray.mean())}
