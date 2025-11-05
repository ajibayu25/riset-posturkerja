"""YOLOv8 object detection wrapper for ROSA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

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
    def pick_mouse_bbox(prediction, labels: Tuple[str, ...] = ("mouse",)) -> Optional[BBox]:
        """Return the highest-confidence mouse bounding box from detector output."""
        if (
            prediction is None
            or getattr(prediction, "boxes", None) is None
            or len(prediction.boxes) == 0
        ):
            return None
        names = prediction.names
        boxes = prediction.boxes
        best_conf = -1.0
        best_coords = None
        for idx in range(len(boxes)):
            label = names.get(int(boxes.cls[idx]), "")
            if labels and label not in labels:
                continue
            conf = float(boxes.conf[idx].item())
            if conf <= best_conf:
                continue
            best_conf = conf
            best_coords = boxes.xyxy[idx].tolist()
        if best_coords is None:
            return None
        x1, y1, x2, y2 = map(int, best_coords)
        return x1, y1, x2, y2

    @staticmethod
    def collect_hand_bboxes(prediction, labels: Optional[Tuple[str, ...]] = None, min_conf: float = 0.15) -> List[BBox]:
        """Extract hand bounding boxes from a detector prediction."""
        if (
            prediction is None
            or getattr(prediction, "boxes", None) is None
            or len(prediction.boxes) == 0
        ):
            return []
        names = prediction.names
        boxes = prediction.boxes
        hand_boxes: List[BBox] = []
        for idx in range(len(boxes)):
            label = names.get(int(boxes.cls[idx]), "")
            conf = float(boxes.conf[idx].item())
            if conf < min_conf:
                continue
            if labels and label not in labels:
                continue
            coords = tuple(map(int, boxes.xyxy[idx].tolist()))
            hand_boxes.append(coords)  # type: ignore[arg-type]
        return hand_boxes

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

    @staticmethod
    def pick_audio_devices(prediction, extra_predictions: Optional[Iterable] = None, labels: Tuple[str, ...] = ("cell phone", "earbud", "earphone", "headset")) -> List[Tuple[str, float, BBox]]:
        devices: List[Tuple[str, float, BBox]] = []
        predictions = [prediction]
        if extra_predictions:
            predictions.extend(extra_predictions)
        for pred in predictions:
            if pred is None or pred.boxes is None or len(pred.boxes) == 0:
                continue
            names = pred.names
            boxes = pred.boxes
            for idx in range(len(boxes)):
                label = names.get(int(boxes.cls[idx]), "")
                if label not in labels:
                    continue
                coords = boxes.xyxy[idx].tolist()
                conf = float(boxes.conf[idx].item())
                bbox = tuple(map(int, coords))
                devices.append((label, conf, bbox))  # type: ignore[arg-type]
        return devices

    @staticmethod
    def pick_table_candidate(
        prediction,
        labels: Tuple[str, ...] = ("dining table", "table", "desk", "bench", "kitchen table"),
        min_conf: float = 0.10,
    ) -> Optional[Tuple[BBox, float]]:
        """Return the highest scoring table/desk candidate (bbox, confidence)."""
        if (
            prediction is None
            or getattr(prediction, "boxes", None) is None
            or len(prediction.boxes) == 0
        ):
            return None
        names = prediction.names
        boxes = prediction.boxes
        candidates: List[Tuple[float, float, List[float]]] = []
        for idx in range(len(boxes)):
            label = names.get(int(boxes.cls[idx]), "")
            if label not in labels:
                continue
            conf = float(boxes.conf[idx].item())
            if conf < min_conf:
                continue
            coords = boxes.xyxy[idx].tolist()
            width = max(coords[2] - coords[0], 1.0)
            height = max(coords[3] - coords[1], 1.0)
            area = width * height
            score = conf * area
            candidates.append((score, conf, coords))
        if not candidates:
            return None
        _, best_conf, best_coords = max(candidates, key=lambda item: item[0])
        x1, y1, x2, y2 = map(int, best_coords)
        return (x1, y1, x2, y2), best_conf

    @staticmethod
    def pick_chair_candidate(
        prediction,
        labels: Tuple[str, ...] = ("chair", "armchair", "couch"),
        min_conf: float = 0.10,
    ) -> Optional[Tuple[BBox, float]]:
        """Return the most confident chair-like candidate (bbox, confidence)."""
        if (
            prediction is None
            or getattr(prediction, "boxes", None) is None
            or len(prediction.boxes) == 0
        ):
            return None
        names = prediction.names
        boxes = prediction.boxes
        best_score = -1.0
        best_entry: Optional[Tuple[List[float], float]] = None
        for idx in range(len(boxes)):
            label = names.get(int(boxes.cls[idx]), "")
            if label not in labels:
                continue
            conf = float(boxes.conf[idx].item())
            if conf < min_conf:
                continue
            coords = boxes.xyxy[idx].tolist()
            width = max(coords[2] - coords[0], 1.0)
            height = max(coords[3] - coords[1], 1.0)
            area = width * height
            score = conf * area
            if score > best_score:
                best_score = score
                best_entry = (coords, conf)
        if best_entry is None:
            return None
        coords, conf = best_entry
        x1, y1, x2, y2 = map(int, coords)
        return (x1, y1, x2, y2), conf
