"""YOLOv8 pose estimator wrapper for ROSA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class PoseEstimator:
    model_path: str = "yolov8n-pose.pt"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.model_path)

    def predict(self, frame) -> any:
        return self.model.predict(source=frame, device=self.device, verbose=False)[0]

    def predict_xy(self, frame) -> Optional[np.ndarray]:
        prediction = self.predict(frame)
        if prediction.keypoints is None or len(prediction.keypoints) == 0:
            return None
        keypoints = prediction.keypoints.xy[0]
        return keypoints.detach().cpu().numpy()
