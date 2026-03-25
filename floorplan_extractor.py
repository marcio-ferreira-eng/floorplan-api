import cv2
import numpy as np
from PIL import Image
import io
import math

def preprocess(img_array: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cleaned

def extract_walls(binary: np.ndarray, orig_w: int, orig_h: int):
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=40,
        maxLineGap=10
    )
    walls = []
    if lines is None:
        return walls
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        angle = math.atan2(dy, dx) * 180 / math.pi
        if angle < 10 or angle > 80:  # orthogonal only
            walls.append({
                "x1": round(x1 / orig_w, 4),
                "y1": round(y1 / orig_h, 4),
                "x2": round(x2 / orig_w, 4),
                "y2": round(y2 / orig_h, 4),
            })
    return walls

def extract_floorplan(image_bytes: bytes, filename: str = "") -> dict:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    h, w = img.shape[:2]
    binary = preprocess(img)
    walls = extract_walls(binary, w, h)

    return {
        "imageWidth": w,
        "imageHeight": h,
        "walls": walls,
        "doors": [],
        "windows": [],
        "sinks": [],
    }
