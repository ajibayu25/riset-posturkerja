from __future__ import annotations

"""Live scoring for ROSA Section C (mouse & keyboard)."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import DEVICE, EXPORT_CSV, EXPORT_JSONL, POSE_MODEL
from constants.grids import SECTIONC_MOUSE_KEYBOARD_GRID, SECTION_C_KEYBOARD_AXIS, SECTION_C_MOUSE_AXIS
from constants.thresholds import SECTION_C_ADJUSTMENTS, SECTION_C_THRESHOLDS
from core.geometry import Skeleton2D, clamp, distance
from core.smoothing import EMA
from core.timers import duration_adjust
from rosa_io.exporters import export_csv, export_json
from models.pose import PoseEstimator


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
class SectionCResult:
    timestamp: float
    mouse: AxisScore
    keyboard: AxisScore
    vertical_axis: int
    horizontal_axis: int
    duration_adjustment: int
    section_score: int

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
    ) -> SectionCResult:
        mouse_score = self._score_mouse(skeleton, hand_preference)
        keyboard_score = self._score_keyboard(skeleton)

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
        )

    def _score_mouse(self, skeleton: Skeleton2D, hand_preference: str) -> AxisScore:
        cfg = SECTION_C_THRESHOLDS["mouse"]
        adjustments = {}
        metrics: Dict[str, float] = {}
        base = 0

        side = hand_preference
        shoulder = skeleton.point(f"{side}_shoulder")
        wrist = skeleton.point(f"{side}_wrist")
        if shoulder is not None and wrist is not None:
            offset = abs(wrist[0] - shoulder[0])
            metrics["lateral_offset_px"] = offset
            shoulder_width = skeleton.shoulder_width()
            if not np.isnan(shoulder_width) and shoulder_width > 1e-6:
                ratio = offset / shoulder_width
                metrics["lateral_ratio"] = ratio
                if ratio > 0.35:
                    base = max(base, 1)
                    adjustments.setdefault("reach", SECTION_C_ADJUSTMENTS["mouse"].get("reach", 2))
        shoulder_mid = skeleton.shoulder_mid()
        hip_mid = skeleton.hip_mid()
        if shoulder_mid is not None and hip_mid is not None:
            vertical = shoulder_mid[1] - hip_mid[1]
            metrics["shoulder_elevation"] = vertical
            if vertical < -20:
                adjustments.setdefault("shoulder_shrug", SECTION_C_ADJUSTMENTS["keyboard"].get("shoulder_shrug", 1))

        return AxisScore(
            name="mouse",
            base=int(base),
            min_value=self.mouse_axis_min,
            max_value=self.mouse_axis_max,
            adjustments=adjustments,
            metrics=metrics,
        )

    def _score_keyboard(self, skeleton: Skeleton2D) -> AxisScore:
        cfg = SECTION_C_THRESHOLDS["keyboard"]
        adjustments = {}
        metrics: Dict[str, float] = {}
        base = 0

        elbow_angles: List[float] = []
        for side in ("left", "right"):
            angle = skeleton.elbow_angle(side)
            metrics[f"{side}_elbow_angle"] = angle
            if np.isnan(angle):
                continue
            elbow_angles.append(angle)
        if elbow_angles:
            avg_angle = float(np.mean(elbow_angles))
            metrics["avg_elbow_angle"] = avg_angle
            if abs(avg_angle - 90.0) > 15.0:
                base = max(base, 1)
            if abs(avg_angle - 90.0) > 30.0:
                base = max(base, 2)
        shoulder_diff = skeleton.shoulder_height_diff()
        metrics["shoulder_height_diff"] = shoulder_diff
        if not np.isnan(shoulder_diff) and abs(shoulder_diff) > cfg["shoulder_shrug_deg"]:
            adjustments.setdefault("shoulder_shrug", SECTION_C_ADJUSTMENTS["keyboard"].get("shoulder_shrug", 1))

        return AxisScore(
            name="keyboard",
            base=int(base),
            min_value=self.keyboard_axis_min,
            max_value=self.keyboard_axis_max,
            adjustments=adjustments,
            metrics=metrics,
        )


class LiveSectionCApp:
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
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.ema: Optional[EMA] = EMA(alpha=smoothing_alpha) if smoothing_alpha else None
        self.scorer = SectionCScorer()
        self.session_start = time.time()
        self.continuous_start = self.session_start
        self.last_export_ts = 0.0
        self.last_result: Optional[SectionCResult] = None

    def _apply_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        if self.ema is None:
            return keypoints
        flat = keypoints.reshape(-1)
        smoothed = self.ema.update(flat)
        return smoothed.reshape(keypoints.shape)

    def _export(self, result: SectionCResult) -> None:
        if self.export_mode == "none":
            return
        row = result.to_row()
        if self.export_mode == "csv":
            export_csv(EXPORT_CSV, row)
        elif self.export_mode == "json":
            export_json(EXPORT_JSONL, row)

    def _format_overlay(self, result: SectionCResult) -> List[str]:
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
                    now = time.time()
                    total_seconds = now - self.session_start
                    continuous_seconds = now - self.continuous_start
                    result = self.scorer.score(
                        skeleton,
                        self.hand_preference,
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
