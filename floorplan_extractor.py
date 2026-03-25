import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
from ultralytics import YOLO

PRETRAINED_MODEL_ID = "keremberke/yolov8n-floor-plan-detection"
_model_cache: Optional[YOLO] = None

@dataclass
class WallSegment:
    x1: int; y1: int
    x2: int; y2: int
    angle: str
    length_px: int

@dataclass
class FloorObject:
    cls: str
    x: int; y: int
    w: int; h: int
    confidence: float

def _get_model() -> YOLO:
    global _model_cache
    if _model_cache is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id=PRETRAINED_MODEL_ID, filename="best.pt")
        _model_cache = YOLO(ckpt)
    return _model_cache

def preprocess(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Imagem não encontrada: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)
    return img, binary

def extract_walls(binary, min_length_px=40):
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/360,
                             threshold=60, minLineLength=min_length_px, maxLineGap=8)
    walls = []
    if lines is None:
        return walls
    for line in lines:
        x1, y1, x2, y2 = line[0].tolist()
        angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)))
        if angle <= 8:
            direction = "horizontal"
        elif angle >= 82:
            direction = "vertical"
        else:
            continue
        walls.append(WallSegment(x1, y1, x2, y2, direction,
                                  int(np.hypot(x2-x1, y2-y1))))
    return walls

def detect_objects(image_path: str, conf=0.35):
    model = _get_model()
    results = model.predict(image_path, conf=conf, verbose=False)
    objects = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            objects.append(FloorObject(
                cls=model.names[int(box.cls[0])],
                x=x1, y=y1, w=x2-x1, h=y2-y1,
                confidence=round(float(box.conf[0]), 3)
            ))
    return objects

def extract_floorplan(image_path: str) -> dict:
    img, binary = preprocess(image_path)
    h, w = img.shape[:2]
    walls   = extract_walls(binary)
    objects = detect_objects(image_path)
    return {
        "image_width":  w,
        "image_height": h,
        "walls":   [asdict(wall) for wall in walls],
        "objects": [asdict(obj)  for obj  in objects],
    }
