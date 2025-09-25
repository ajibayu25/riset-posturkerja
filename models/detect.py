"""YOLOv8 object detection wrapper for ROSA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from ultralytics import YOLO

BBox = Tuple[int, int, int, int]


@dataclass
class ObjectDetector:
    model_path: str = "yolov8n.pt"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.model_path)

    def predict(self, frame) -> any:
        return self.model.predict(source=frame, device=self.device, verbose=False)[0]

    @staticmethod
    def pick_monitor_bbox(prediction) -> Optional[BBox]:
        if prediction is None or prediction.boxes is None or len(prediction.boxes) == 0:
            return None
        names = prediction.names
        boxes = prediction.boxes
        candidates = []
        for idx in range(len(boxes)):
            label = names.get(int(boxes.cls[idx]), "")
            if label in {"tv", "laptop", "screen"}:
                coords = boxes.xyxy[idx].tolist()
                area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                candidates.append((area, coords))
        if not candidates:
            return None
        _, best_coords = max(candidates, key=lambda item: item[0])
        x1, y1, x2, y2 = map(int, best_coords)
        return x1, y1, x2, y2

    @staticmethod
    def pick_phone_bbox(prediction) -> Optional[BBox]:
        if prediction is None or prediction.boxes is None or len(prediction.boxes) == 0:
            return None
        names = prediction.names
        boxes = prediction.boxes
        best_conf = -1.0
        best_coords = None
        for idx in range(len(boxes)):
            label = names.get(int(boxes.cls[idx]), "")
            if label == "cell phone":
                conf = float(boxes.conf[idx].item())
                if conf > best_conf:
                    best_conf = conf
                    best_coords = boxes.xyxy[idx].tolist()
        if best_coords is None:
            return None
        x1, y1, x2, y2 = map(int, best_coords)
        return x1, y1, x2, y2
