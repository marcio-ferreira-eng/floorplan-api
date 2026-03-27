import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
import tempfile
import os

@dataclass
class WallSegment:
    x1: float
    y1: float
    x2: float
    y2: float

def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morphed

def extract_walls(img: np.ndarray) -> list[WallSegment]:
    processed = preprocess(img)
    h, w = processed.shape
    min_len = int(min(h, w) * 0.03)
    lines = cv2.HoughLinesP(
        processed,
        rho=1, theta=np.pi/180,
        threshold=50,
        minLineLength=min_len,
        maxLineGap=10
    )
    walls = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx < 5 or dy < 5:  # orthogonal only
                walls.append(WallSegment(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2)
                ))
    return walls

def extract_floorplan(file_path: str) -> dict:
    # Read bytes from file
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    # Handle PDF: convert first page to image
    if file_path.lower().endswith('.pdf'):
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(file_path)
            page = pdf[0]
            bitmap = page.render(scale=2.0)
            pil_img = bitmap.to_pil()
            img_array = np.array(pil_img.convert('RGB'))
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return {"error": f"PDF conversion failed: {str(e)}", "walls": [], "objects": []}
    else:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image", "walls": [], "objects": []}

    walls = extract_walls(img)
    h, w = img.shape[:2]

    return {
        "image_size": {"width": w, "height": h},
        "walls": [asdict(w) for w in walls],
        "objects": []
    }
