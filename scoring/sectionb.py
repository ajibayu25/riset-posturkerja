from __future__ import annotations

"""Live scoring for ROSA Section B (monitor & telephone)."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import DET_MODEL, DEVICE, EXPORT_CSV, EXPORT_JSONL, POSE_MODEL
from constants.grids import MONITOR_PHONE_GRID, SECTION_B_MONITOR_AXIS, SECTION_B_PHONE_AXIS
from constants.thresholds import SECTION_B_ADJUSTMENTS, SECTION_B_THRESHOLDS
from core.geometry import Skeleton2D, clamp, distance
from core.smoothing import EMA
from core.timers import duration_adjust
from rosa_io.exporters import export_csv, export_json
from models.detect import ObjectDetector
from models.pose import PoseEstimator

BBox = Tuple[int, int, int, int]


@dataclass
class AxisScore:
    name: str
    base: int
    min_value: int
    max_value: int
    adjustments: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> int:
        value = self.base + sum(self.adjustments.values())
        return int(clamp(value, self.min_value, self.max_value))

    def as_dict(self) -> Dict[str, float]:
        out = {f"{self.name}_base": self.base, f"{self.name}_total": self.total}
        for key, val in self.metrics.items():
            out[f"{self.name}_{key}"] = val
        for key, val in self.adjustments.items():
            out[f"{self.name}_adj_{key}"] = val
        return out


@dataclass
class SectionBResult:
    timestamp: float
    monitor: AxisScore
    phone: AxisScore
    horizontal_axis: int
    vertical_axis: int
    duration_adjustment: int
    section_score: int

    def to_row(self) -> Dict[str, float]:
        row = {
            "ts": self.timestamp,
            "section": "B",
            "horizontal_axis": self.horizontal_axis,
            "vertical_axis": self.vertical_axis,
            "duration_adjustment": self.duration_adjustment,
            "section_score": self.section_score,
        }
        row.update(self.monitor.as_dict())
        row.update(self.phone.as_dict())
        return row


class SectionBScorer:
    def __init__(self) -> None:
        self.monitor_axis_min = SECTION_B_MONITOR_AXIS[0]
        self.monitor_axis_max = SECTION_B_MONITOR_AXIS[-1]
        self.phone_axis_min = SECTION_B_PHONE_AXIS[0]
        self.phone_axis_max = SECTION_B_PHONE_AXIS[-1]

    def score(
        self,
        skeleton: Skeleton2D,
        monitor_bbox: Optional[BBox],
        phone_bbox: Optional[BBox],
        frame_shape: Tuple[int, int, int],
        total_seconds: float,
        continuous_seconds: float,
    ) -> SectionBResult:
        monitor_score = self._score_monitor(skeleton, monitor_bbox)
        phone_score = self._score_phone(skeleton, phone_bbox, frame_shape)

        duration_adj = duration_adjust(total_seconds, continuous_seconds)

        horizontal_axis = clamp(
            monitor_score.total + duration_adj,
            self.monitor_axis_min,
            self.monitor_axis_max,
        )
        vertical_axis = clamp(
            phone_score.total + duration_adj,
            self.phone_axis_min,
            self.phone_axis_max,
        )

        h_idx = int(horizontal_axis - self.monitor_axis_min)
        v_idx = int(vertical_axis - self.phone_axis_min)
        section_score = int(MONITOR_PHONE_GRID[v_idx, h_idx])

        return SectionBResult(
            timestamp=time.time(),
            monitor=monitor_score,
            phone=phone_score,
            horizontal_axis=int(horizontal_axis),
            vertical_axis=int(vertical_axis),
            duration_adjustment=duration_adj,
            section_score=section_score,
        )

    def _score_monitor(self, skeleton: Skeleton2D, monitor_bbox: Optional[BBox]) -> AxisScore:
        cfg = SECTION_B_THRESHOLDS["monitor"]
        adjustments = {}
        metrics: Dict[str, float] = {}
        base = 0

        neck_vec = None
        shoulder_mid = skeleton.shoulder_mid()
        nose = skeleton.point("nose")
        if shoulder_mid is not None and nose is not None:
            neck_vec = nose - shoulder_mid
            metrics["neck_vertical"] = float(neck_vec[1])
            metrics["neck_horizontal"] = float(neck_vec[0])
            if neck_vec[1] > cfg["vertical_angle_deg"]["too_low_max"]:
                base = max(base, 1)  # looking down
                adjustments.setdefault("too_low", SECTION_B_ADJUSTMENTS["monitor"].get("too_low", 1))
            if neck_vec[1] < -cfg["vertical_angle_deg"]["too_high_min"]:
                base = max(base, 2)
                adjustments.setdefault("too_high", SECTION_B_ADJUSTMENTS["monitor"].get("too_high", 1))

        if monitor_bbox is not None and shoulder_mid is not None:
            x1, y1, x2, y2 = monitor_bbox
            width = max(1.0, float(x2 - x1))
            height = max(1.0, float(y2 - y1))
            center_y = (y1 + y2) / 2.0
            metrics["monitor_width"] = width
            metrics["monitor_height"] = height
            metrics["monitor_center_y"] = center_y
            metrics["monitor_offset_y"] = center_y - shoulder_mid[1]
            shoulder_width = skeleton.shoulder_width()
            if not np.isnan(shoulder_width):
                ratio = shoulder_width / width
                metrics["distance_ratio"] = ratio
                if ratio > cfg["distance_cm"]["too_far_min"] / 10.0:  # rough pixel heuristic
                    adjustments.setdefault("too_far", SECTION_B_ADJUSTMENTS["monitor"].get("too_far", 1))

        return AxisScore(
            name="monitor",
            base=int(base),
            min_value=self.monitor_axis_min,
            max_value=self.monitor_axis_max,
            adjustments=adjustments,
            metrics=metrics,
        )

    def _score_phone(
        self,
        skeleton: Skeleton2D,
        phone_bbox: Optional[BBox],
        frame_shape: Tuple[int, int, int],
    ) -> AxisScore:
        cfg = SECTION_B_THRESHOLDS["telephone"]
        adjustments = {}
        metrics: Dict[str, float] = {}
        base = 0

        sidebend = skeleton.neck_sidebend()
        metrics["neck_sidebend"] = sidebend
        if not np.isnan(sidebend) and abs(sidebend) > cfg["neck_sidebend_deg"]:
            adjustments.setdefault("neck_shoulder_hold", SECTION_B_ADJUSTMENTS["telephone"].get("neck_shoulder_hold", 2))

        if phone_bbox is not None:
            x1, y1, x2, y2 = phone_bbox
            phone_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
            metrics["phone_center_x"] = phone_center[0]
            metrics["phone_center_y"] = phone_center[1]
            ref_point = skeleton.point("right_shoulder") or skeleton.point("nose")
            if ref_point is not None:
                reach = distance(phone_center, ref_point)
                metrics["reach_pixels"] = reach
                height, width = frame_shape[0], frame_shape[1]
                diag = (width**2 + height**2) ** 0.5
                if reach > 0.3 * diag:
                    base = max(base, 2)
                    adjustments.setdefault("outside_reach", SECTION_B_ADJUSTMENTS["telephone"].get("outside_reach", 2))
        else:
            metrics["phone_detected"] = 0.0

        return AxisScore(
            name="phone",
            base=int(base),
            min_value=self.phone_axis_min,
            max_value=self.phone_axis_max,
            adjustments=adjustments,
            metrics=metrics,
        )


class LiveSectionBApp:
    def __init__(
        self,
        cam_index: int = 0,
        pose_model: Optional[str] = None,
        det_model: Optional[str] = None,
        device: Optional[str] = None,
        export_mode: str = "csv",
        smoothing_alpha: float = 0.3,
        detection_stride: int = 5,
    ) -> None:
        self.cam_index = cam_index
        self.export_mode = export_mode
        self.pose = PoseEstimator(model_path=pose_model or POSE_MODEL, device=device or DEVICE)
        self.detector = ObjectDetector(model_path=det_model or DET_MODEL, device=device or DEVICE)
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.ema: Optional[EMA] = EMA(alpha=smoothing_alpha) if smoothing_alpha else None
        self.scorer = SectionBScorer()
        self.session_start = time.time()
        self.continuous_start = self.session_start
        self.last_export_ts = 0.0
        self.last_result: Optional[SectionBResult] = None
        self.detection_stride = max(1, detection_stride)
        self.frame_count = 0
        self.last_monitor_bbox: Optional[BBox] = None
        self.last_phone_bbox: Optional[BBox] = None

    def _apply_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        if self.ema is None:
            return keypoints
        flat = keypoints.reshape(-1)
        smoothed = self.ema.update(flat)
        return smoothed.reshape(keypoints.shape)

    def _export(self, result: SectionBResult) -> None:
        if self.export_mode == "none":
            return
        row = result.to_row()
        if self.export_mode == "csv":
            export_csv(EXPORT_CSV, row)
        elif self.export_mode == "json":
            export_json(EXPORT_JSONL, row)

    def _format_overlay(self, result: SectionBResult) -> List[str]:
        lines = [
            f"Section B score: {result.section_score} (dur {result.duration_adjustment:+d})",
            f"Monitor axis (H): {result.horizontal_axis} | Phone axis (V): {result.vertical_axis}",
            f"Monitor total {result.monitor.total} (base {result.monitor.base})",
            f"Phone total {result.phone.total} (base {result.phone.base})",
        ]
        risk = "OK" if result.section_score < 5 else "High"
        lines.append(f"Risk level: {risk}")
        return lines

    def _maybe_run_detection(self, frame: np.ndarray) -> None:
        if self.frame_count % self.detection_stride != 0:
            return
        detections = self.detector.predict(frame)
        self.last_monitor_bbox = ObjectDetector.pick_monitor_bbox(detections)
        self.last_phone_bbox = ObjectDetector.pick_phone_bbox(detections)

    def run(self) -> None:
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {self.cam_index} cannot be opened")
        window_name = "ROSA Section B"
        cv2.namedWindow(window_name)
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                self.frame_count += 1
                self._maybe_run_detection(frame)
                keypoints = self.pose.predict_xy(frame)
                overlay_lines: List[str] = []
                display = frame
                if keypoints is not None:
                    keypoints = self._apply_smoothing(keypoints[:, :2])
                    display = frame.copy()
                    skeleton = Skeleton2D.from_array(keypoints)
                    now = time.time()
                    total_seconds = now - self.session_start
                    continuous_seconds = now - self.continuous_start
                    result = self.scorer.score(
                        skeleton,
                        self.last_monitor_bbox,
                        self.last_phone_bbox,
                        frame.shape,
                        total_seconds,
                        continuous_seconds,
                    )
                    self.last_result = result
                    overlay_lines = self._format_overlay(result)
                    if now - self.last_export_ts > 5.0:
                        self._export(result)
                        self.last_export_ts = now
                elif self.last_result is not None:
                    overlay_lines = ["No pose detected"]
                for idx, text in enumerate(overlay_lines):
                    cv2.putText(
                        display,
                        text,
                        (20, 40 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if self.last_monitor_bbox is not None:
                    x1, y1, x2, y2 = self.last_monitor_bbox
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)
                if self.last_phone_bbox is not None:
                    x1, y1, x2, y2 = self.last_phone_bbox
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 120, 0), 2)
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    self.continuous_start = time.time()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


__all__ = [
    "AxisScore",
    "LiveSectionBApp",
    "SectionBResult",
    "SectionBScorer",
]



