from __future__ import annotations

"""Live scoring for ROSA Section C (mouse & keyboard)."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    DEVICE,
    EXPORT_CSV,
    EXPORT_JSONL,
    HAND_MODEL,
    POSE_MODEL,
    CAMERA_TARGET_FPS,
    CAMERA_FRAME_WIDTH,
    CAMERA_FRAME_HEIGHT,
    DATA_CAPTURE_INTERVAL,
)
from constants.grids import SECTIONC_MOUSE_KEYBOARD_GRID, SECTION_C_KEYBOARD_AXIS, SECTION_C_MOUSE_AXIS
from core.geometry import Skeleton2D, clamp, distance
from core.smoothing import EMA
from core.timers import duration_adjust
from rosa_io.exporters import export_csv, export_json
from models.detect import ObjectDetector, BBox
from models.pose import PoseEstimator
from sensory.overhead import keyboard_components, mouse_components


@dataclass
class AxisScore:
    """Capture base score plus adjustments for mouse/keyboard axes."""
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
class SectionCResult:
    """Point-in-time ROSA Section C result."""
    timestamp: float
    mouse: AxisScore
    keyboard: AxisScore
    vertical_axis: int
    horizontal_axis: int
    duration_adjustment: int
    section_score: int
    query_breakdown: Dict[str, int]

    def to_row(self) -> Dict[str, float]:
        row = {
            "ts": self.timestamp,
            "section": "C",
            "vertical_axis": self.vertical_axis,
            "horizontal_axis": self.horizontal_axis,
            "duration_adjustment": self.duration_adjustment,
            "section_score": self.section_score,
        }
        row.update(self.mouse.as_dict())
        row.update(self.keyboard.as_dict())
        return row


class SectionCScorer:
    """Compute Section C scores from a pose skeleton.

    Heuristic summary:
    - Mouse axis: lateral offset/reach/abduction (px→cm from shoulder breadth), surface mismatch,
      pinch-grip via hand–mouse overlap (with pose-based fallback hand boxes).
    - Keyboard axis: wrist deviation during typing using forearm vs hand vectors.
    Scores map onto the Section C mouse/keyboard matrix.
    """
    def __init__(self) -> None:
        self.mouse_axis_min = SECTION_C_MOUSE_AXIS[0]
        self.mouse_axis_max = SECTION_C_MOUSE_AXIS[-1]
        self.keyboard_axis_min = SECTION_C_KEYBOARD_AXIS[0]
        self.keyboard_axis_max = SECTION_C_KEYBOARD_AXIS[-1]

    def score(
        self,
        skeleton: Skeleton2D,
        hand_preference: str,
        total_seconds: float,
        continuous_seconds: float,
        *,
        mouse_bbox: Optional[Tuple[int, int, int, int]] = None,
        hand_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> SectionCResult:
        """Main entry to produce Section C score and breakdown.

        The scorer intentionally keeps state minimal—mouse/keyboard sub-scores
        are calculated independently and their query maps are merged afterward.
        Duration adjustments happen here so the same behaviour applies whether
        the scorer is driven from the GUI or the CLI helper.
        """
        mouse_comp = mouse_components(
            skeleton,
            mouse_bbox,
            hand_bboxes or [],
            hand_preference,
        )
        keyboard_comp = keyboard_components(skeleton, hand_bboxes or [])

        mouse_score = AxisScore(
            name="mouse",
            base=int(mouse_comp.base),
            min_value=self.mouse_axis_min,
            max_value=self.mouse_axis_max,
            adjustments=dict(mouse_comp.adjustments),
            metrics=dict(mouse_comp.metrics),
        )
        keyboard_score = AxisScore(
            name="keyboard",
            base=int(keyboard_comp.base),
            min_value=self.keyboard_axis_min,
            max_value=self.keyboard_axis_max,
            adjustments=dict(keyboard_comp.adjustments),
            metrics=dict(keyboard_comp.metrics),
        )
        query_breakdown: Dict[str, int] = {}
        query_breakdown.update(mouse_comp.queries)
        query_breakdown.update(keyboard_comp.queries)

        duration_adj = duration_adjust(total_seconds, continuous_seconds)
        vertical_axis = clamp(
            mouse_score.total + duration_adj,
            self.mouse_axis_min,
            self.mouse_axis_max,
        )
        horizontal_axis = clamp(
            keyboard_score.total + duration_adj,
            self.keyboard_axis_min,
            self.keyboard_axis_max,
        )

        v_idx = int(vertical_axis - self.mouse_axis_min)
        h_idx = int(horizontal_axis - self.keyboard_axis_min)
        section_score = int(SECTIONC_MOUSE_KEYBOARD_GRID[v_idx, h_idx])

        return SectionCResult(
            timestamp=time.time(),
            mouse=mouse_score,
            keyboard=keyboard_score,
            vertical_axis=int(vertical_axis),
            horizontal_axis=int(horizontal_axis),
            duration_adjustment=duration_adj,
            section_score=section_score,
            query_breakdown=query_breakdown,
        )

class LiveSectionCApp:
    """Minimal OpenCV UI to preview Section C scoring."""
    def __init__(
        self,
        cam_index: int = 0,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        export_mode: str = "csv",
        smoothing_alpha: float = 0.3,
        hand_preference: str = "right",
    ) -> None:
        self.cam_index = cam_index
        self.export_mode = export_mode
        self.hand_preference = hand_preference
        self.pose = PoseEstimator(model_path=model_name or POSE_MODEL, device=device or DEVICE)
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
        self.ema: Optional[EMA] = EMA(alpha=smoothing_alpha) if smoothing_alpha else None
        self.scorer = SectionCScorer()
        self.hand_detector = ObjectDetector(model_path=HAND_MODEL, device=device or DEVICE)
        self._hand_bboxes: List[BBox] = []
        self._frame_idx = 0
        self._hand_stride = 6
        self.session_start = time.time()
        self.continuous_start = self.session_start
        self.last_export_ts = 0.0
        self.last_result: Optional[SectionCResult] = None

    def _apply_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        """Smooth raw pose keypoints to reduce jitter."""
        if self.ema is None:
            return keypoints
        flat = keypoints.reshape(-1)
        smoothed = self.ema.update(flat)
        return smoothed.reshape(keypoints.shape)

    def _export(self, result: SectionCResult) -> None:
        """Append the latest Section C row to CSV/JSON logs."""
        if self.export_mode == "none":
            return
        row = result.to_row()
        if self.export_mode == "csv":
            export_csv(EXPORT_CSV, row)
        elif self.export_mode == "json":
            export_json(EXPORT_JSONL, row)

    def _format_overlay(self, result: SectionCResult) -> List[str]:
        """Build textual overlay summarising risk posture."""
        lines = [
            f"Section C score: {result.section_score} (dur {result.duration_adjustment:+d})",
            f"Mouse axis (V): {result.vertical_axis} | Keyboard axis (H): {result.horizontal_axis}",
            f"Mouse total {result.mouse.total} (base {result.mouse.base})",
            f"Keyboard total {result.keyboard.total} (base {result.keyboard.base})",
        ]
        risk = "OK" if result.section_score < 5 else "High"
        lines.append(f"Risk level: {risk}")
        return lines

    def run(self) -> None:
        """Open capture loop, compute scores, and display overlays."""
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {self.cam_index} cannot be opened")
        window_name = "ROSA Section C"
        cv2.namedWindow(window_name)
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                keypoints = self.pose.predict_xy(frame)
                overlay_lines: List[str] = []
                display = frame
                if keypoints is not None:
                    keypoints = self._apply_smoothing(keypoints[:, :2])
                    display = frame.copy()
                    skeleton = Skeleton2D.from_array(keypoints)
                    self._frame_idx += 1
                    if self._frame_idx % self._hand_stride == 0 or not self._hand_bboxes:
                        hand_pred = self.hand_detector.predict(frame)
                        self._hand_bboxes = ObjectDetector.collect_hand_bboxes(hand_pred)
                    now = time.time()
                    total_seconds = now - self.session_start
                    continuous_seconds = now - self.continuous_start
                    result = self.scorer.score(
                        skeleton,
                        self.hand_preference,
                        total_seconds,
                        continuous_seconds,
                        hand_bboxes=self._hand_bboxes,
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
                        (200, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
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
    "LiveSectionCApp",
    "SectionCResult",
    "SectionCScorer",
]
