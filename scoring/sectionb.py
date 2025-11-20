from __future__ import annotations

"""Live scoring for ROSA Section B (monitor & telephone)."""

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    DET_MODEL,
    EARPHONE_MODEL,
    DEVICE,
    EXPORT_CSV,
    EXPORT_JSONL,
    POSE_MODEL,
    CAMERA_TARGET_FPS,
    CAMERA_FRAME_WIDTH,
    CAMERA_FRAME_HEIGHT,
    DATA_CAPTURE_INTERVAL,
)
from constants.grids import MONITOR_PHONE_GRID, SECTION_B_MONITOR_AXIS, SECTION_B_PHONE_AXIS
from core.geometry import Skeleton2D, clamp
from core.smoothing import EMA
from core.timers import duration_adjust
from rosa_io.exporters import export_csv, export_json
from models.detect import ObjectDetector
from models.pose import PoseEstimator
from sensory.front import monitor_components, phone_components

BBox = Tuple[int, int, int, int]


@dataclass
class AxisScore:
    """Score details for a single axis (monitor or phone) including adjustments."""
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
    """Structured result for Section B evaluation."""
    timestamp: float
    monitor: AxisScore
    phone: AxisScore
    horizontal_axis: int
    vertical_axis: int
    duration_adjustment: int
    section_score: int
    query_breakdown: Dict[str, int]

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
    """Combine pose and detection measurements into ROSA Section B scores.

    Heuristic summary:
    - Monitor axis: neck flex/extension, shoulder/elbow gaps, wrist span, distance to monitor.
    - Telephone axis: neck sidebend (shoulder hold), reach to phone, hands-free device near ear.
    - Document holder: bundle vs holder detections plus head twist/offset from monitor.
    Scores are clamped and looked up in the Section B matrix.
    """
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
        audio_devices: Iterable[Tuple[str, float, BBox]],
        frame_shape: Tuple[int, int, int],
        total_seconds: float,
        continuous_seconds: float,
        document_artifacts: Optional[Dict[str, List[BBox]]] = None,
    ) -> SectionBResult:
        """Main entry: compute axes, grid lookup, and duration adjustments.

        The ROSA manual treats monitor and telephone axes independently; we
        mirror that separation here so the GUI/export can break down each axis
        while still feeding the combined score into the Section B matrix."""
        monitor_comp = monitor_components(skeleton, monitor_bbox, document_artifacts)
        phone_comp = phone_components(skeleton, phone_bbox, audio_devices, frame_shape)

        monitor_score = AxisScore(
            name="monitor",
            base=int(monitor_comp.base),
            min_value=self.monitor_axis_min,
            max_value=self.monitor_axis_max,
            adjustments=dict(monitor_comp.adjustments),
            metrics=dict(monitor_comp.metrics),
        )
        phone_score = AxisScore(
            name="phone",
            base=int(phone_comp.base),
            min_value=self.phone_axis_min,
            max_value=self.phone_axis_max,
            adjustments=dict(phone_comp.adjustments),
            metrics=dict(phone_comp.metrics),
        )
        query_breakdown: Dict[str, int] = {}
        query_breakdown.update(monitor_comp.queries)
        query_breakdown.update(phone_comp.queries)

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
            query_breakdown=query_breakdown,
        )


class LiveSectionBApp:
    """Legacy OpenCV loop to visualise Section B scoring without the Tk GUI."""
    def __init__(
        self,
        cam_index: int = 0,
        pose_model: Optional[str] = None,
        det_model: Optional[str] = None,
        ear_model: Optional[str] = None,
        device: Optional[str] = None,
        export_mode: str = "csv",
        smoothing_alpha: float = 0.3,
        detection_stride: int = 5,
    ) -> None:
        self.cam_index = cam_index
        self.export_mode = export_mode
        self.pose = PoseEstimator(model_path=pose_model or POSE_MODEL, device=device or DEVICE)
        self.detector = ObjectDetector(model_path=det_model or DET_MODEL, device=device or DEVICE)
        self.audio_detector = ObjectDetector(model_path=ear_model or EARPHONE_MODEL, device=device or DEVICE)
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
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
        self.last_audio_devices = []
        self.document_artifacts: Dict[str, List[BBox]] = {"holders": [], "bundles": []}

    def _apply_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to raw pose keypoints."""
        if self.ema is None:
            return keypoints
        flat = keypoints.reshape(-1)
        smoothed = self.ema.update(flat)
        return smoothed.reshape(keypoints.shape)

    def _export(self, result: SectionBResult) -> None:
        """Write Section B row to persistent CSV/JSON logs."""
        if self.export_mode == "none":
            return
        row = result.to_row()
        if self.export_mode == "csv":
            export_csv(EXPORT_CSV, row)
        elif self.export_mode == "json":
            export_json(EXPORT_JSONL, row)

    def _format_overlay(self, result: SectionBResult) -> List[str]:
        """Render human-readable summary shown on video preview."""
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
        """Throttle object detection to every n-th frame for speed."""
        if self.frame_count % self.detection_stride != 0:
            return
        detections = self.detector.predict(frame)
        self.last_monitor_bbox = ObjectDetector.pick_monitor_bbox(detections)
        self.last_phone_bbox = ObjectDetector.pick_phone_bbox(detections)
        ear_pred = self.audio_detector.predict(frame)
        self.last_audio_devices = ObjectDetector.pick_audio_devices(detections, [ear_pred])
        self.document_artifacts = ObjectDetector.detect_document_artifacts(detections)

    def run(self) -> None:
        """Main loop capturing frames, scoring, and updating the preview window."""
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
                        self.last_audio_devices,
                        frame.shape,
                        total_seconds,
                        continuous_seconds,
                        document_artifacts=self.document_artifacts,
                    )
                    self.last_result = result
                    overlay_lines = self._format_overlay(result)
                    if now - self.last_export_ts > DATA_CAPTURE_INTERVAL:
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



