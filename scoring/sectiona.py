from __future__ import annotations

"""Live scoring for ROSA Section A (chair) with pose-based heuristics."""

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from config import DEVICE, EXPORT_CSV, EXPORT_JSONL, POSE_MODEL
from constants.grids import (
    SECTION_A_GRID,
    SECTION_A_HORIZONTAL_AXIS,
    SECTION_A_VERTICAL_AXIS,
)
from constants.thresholds import SECTION_A_THRESHOLDS
from core.geometry import Skeleton2D, clamp, distance
from core.smoothing import EMA
from core.timers import duration_adjust
from rosa_io.exporters import export_csv, export_json
from models.pose import PoseEstimator


@dataclass
class SubScore:
    name: str
    base: int
    adjustments: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def adjustment_total(self) -> int:
        return sum(self.adjustments.values())

    @property
    def total(self) -> int:
        value = self.base + self.adjustment_total
        return max(0, int(round(value)))

    def as_dict(self) -> Dict[str, float]:
        out = {f"{self.name}_base": self.base, f"{self.name}_total": self.total}
        for key, val in self.metrics.items():
            out[f"{self.name}_{key}"] = val
        for key, val in self.adjustments.items():
            out[f"{self.name}_adj_{key}"] = val
        return out


@dataclass
class SectionAResult:
    timestamp: float
    seat_height: SubScore
    seat_depth: SubScore
    armrest: SubScore
    back_support: SubScore
    vertical_axis: int
    horizontal_axis: int
    chair_score_base: int
    duration_adjustment: int
    chair_score_final: int

    def to_row(self) -> Dict[str, float]:
        row = {
            "ts": self.timestamp,
            "section": "A",
            "vertical_axis": self.vertical_axis,
            "horizontal_axis": self.horizontal_axis,
            "chair_score_base": self.chair_score_base,
            "duration_adjustment": self.duration_adjustment,
            "chair_score_final": self.chair_score_final,
        }
        for subs in (self.seat_height, self.seat_depth, self.armrest, self.back_support):
            row.update(subs.as_dict())
        return row


def _nanmean(values: Iterable[float]) -> float:
    arr = [v for v in values if not np.isnan(v)]
    if not arr:
        return float("nan")
    return float(np.mean(arr))


def _score_seat_height(skeleton: Skeleton2D) -> SubScore:
    knee_cfg = SECTION_A_THRESHOLDS["seat_height"]["knee_angle_deg"]
    metrics: Dict[str, float] = {}
    base = 1
    knee_angles: List[float] = []
    for side in ("left", "right"):
        angle = skeleton.knee_angle(side)
        metrics[f"{side}_knee_angle"] = angle
        if not np.isnan(angle):
            knee_angles.append(angle)
    avg_angle = _nanmean(knee_angles)
    metrics["avg_knee_angle"] = avg_angle
    if not np.isnan(avg_angle):
        if avg_angle < knee_cfg["too_low_max"]:
            base = 2
            metrics["classification"] = 1.0  # too low
        elif avg_angle > knee_cfg["too_high_min"]:
            base = 2
            metrics["classification"] = 2.0  # too high
        else:
            metrics["classification"] = 0.0  # neutral
    # detect feet off ground by comparing ankle drop relative to leg length
    foot_contact = True
    for side in ("left", "right"):
        hip = skeleton.point(f"{side}_hip")
        knee = skeleton.point(f"{side}_knee")
        ankle = skeleton.point(f"{side}_ankle")
        if hip is None or knee is None or ankle is None:
            continue
        leg_len = distance(hip, ankle)
        drop = ankle[1] - knee[1]
        metrics[f"{side}_ankle_drop"] = drop
        if leg_len > 1e-3 and drop < 0.1 * leg_len:
            foot_contact = False
    if not foot_contact:
        base = max(base, 3)
        metrics["foot_contact"] = 0.0
    else:
        metrics["foot_contact"] = 1.0
    return SubScore("seat_height", base, {}, metrics)


def _score_seat_depth(skeleton: Skeleton2D) -> SubScore:
    metrics: Dict[str, float] = {}
    thigh_lengths: List[float] = []
    horiz_span: List[float] = []
    for side in ("left", "right"):
        hip = skeleton.point(f"{side}_hip")
        knee = skeleton.point(f"{side}_knee")
        if hip is None or knee is None:
            continue
        thigh_lengths.append(distance(hip, knee))
        horiz_span.append(abs(knee[0] - hip[0]))
    if thigh_lengths:
        metrics["avg_thigh_len"] = float(np.mean(thigh_lengths))
    if horiz_span:
        metrics["avg_thigh_span_x"] = float(np.mean(horiz_span))
    # automated detection for seat depth is pending, keep neutral score
    return SubScore("seat_depth", 1, {}, metrics)


def _score_armrest(skeleton: Skeleton2D) -> SubScore:
    elbow_cfg = SECTION_A_THRESHOLDS["armrest"]["elbow_angle_deg"]
    metrics: Dict[str, float] = {}
    flags: List[str] = []
    for side in ("left", "right"):
        angle = skeleton.elbow_angle(side)
        metrics[f"{side}_elbow_angle"] = angle
        if np.isnan(angle):
            continue
        if angle < elbow_cfg["neutral_min"]:
            flags.append("too_high")
        elif angle > elbow_cfg["neutral_max"]:
            flags.append("too_low")
    base = 1
    if flags:
        base = 2
        metrics["classification"] = 1.0 if flags[0] == "too_high" else 2.0
    else:
        metrics["classification"] = 0.0
    return SubScore("armrest", base, {}, metrics)


def _score_back_support(skeleton: Skeleton2D) -> SubScore:
    cfg = SECTION_A_THRESHOLDS["back_support"]
    metrics: Dict[str, float] = {}
    inclination = skeleton.trunk_inclination()
    metrics["trunk_inclination"] = inclination
    base = 1
    if not np.isnan(inclination):
        if inclination > cfg["forward_flex_deg"]:
            base = 2
            metrics["classification"] = 1.0  # leaning forward
        else:
            metrics["classification"] = 0.0
    else:
        metrics["classification"] = -1.0
    return SubScore("back_support", base, {}, metrics)


class SectionAScorer:
    def score(
        self,
        skeleton: Skeleton2D,
        total_seconds: float,
        continuous_seconds: float,
    ) -> SectionAResult:
        seat_height = _score_seat_height(skeleton)
        seat_depth = _score_seat_depth(skeleton)
        armrest = _score_armrest(skeleton)
        back_support = _score_back_support(skeleton)

        vertical_axis = seat_height.total + seat_depth.total
        horizontal_axis = armrest.total + back_support.total

        vertical_axis = int(clamp(vertical_axis, SECTION_A_VERTICAL_AXIS[0], SECTION_A_VERTICAL_AXIS[-1]))
        horizontal_axis = int(clamp(horizontal_axis, SECTION_A_HORIZONTAL_AXIS[0], SECTION_A_HORIZONTAL_AXIS[-1]))

        v_idx = vertical_axis - SECTION_A_VERTICAL_AXIS[0]
        h_idx = horizontal_axis - SECTION_A_HORIZONTAL_AXIS[0]
        chair_score_base = int(SECTION_A_GRID[v_idx, h_idx])

        duration_adj = duration_adjust(total_seconds, continuous_seconds)
        chair_score_final = int(clamp(chair_score_base + duration_adj, 1, 10))

        return SectionAResult(
            timestamp=time.time(),
            seat_height=seat_height,
            seat_depth=seat_depth,
            armrest=armrest,
            back_support=back_support,
            vertical_axis=vertical_axis,
            horizontal_axis=horizontal_axis,
            chair_score_base=chair_score_base,
            duration_adjustment=duration_adj,
            chair_score_final=chair_score_final,
        )


COCO_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def _draw_keypoints(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    for x, y in keypoints:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
    for a, b in COCO_EDGES:
        if a < len(keypoints) and b < len(keypoints):
            pa = keypoints[a]
            pb = keypoints[b]
            cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 200, 255), 2)
    return vis


class LiveSectionAApp:
    def __init__(
        self,
        cam_index: int = 0,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        export_mode: str = "csv",
        smoothing_alpha: float = 0.3,
    ) -> None:
        self.cam_index = cam_index
        self.export_mode = export_mode
        self.pose = PoseEstimator(model_path=model_name or POSE_MODEL, device=device or DEVICE)
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.ema: Optional[EMA] = EMA(alpha=smoothing_alpha) if smoothing_alpha else None
        self.scorer = SectionAScorer()
        self.session_start = time.time()
        self.continuous_start = self.session_start
        self.last_export_ts = 0.0
        self.last_result: Optional[SectionAResult] = None

    def _apply_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        if self.ema is None:
            return keypoints
        flat = keypoints.reshape(-1)
        smoothed = self.ema.update(flat)
        return smoothed.reshape(keypoints.shape)

    def _export(self, result: SectionAResult) -> None:
        if self.export_mode == "none":
            return
        row = result.to_row()
        if self.export_mode == "csv":
            export_csv(EXPORT_CSV, row)
        elif self.export_mode == "json":
            export_json(EXPORT_JSONL, row)

    def _format_overlay(self, result: SectionAResult) -> List[str]:
        lines = [
            f"Section A chair score: {result.chair_score_final} (base {result.chair_score_base}, dur {result.duration_adjustment:+d})",
            f"Vertical axis (seat): {result.vertical_axis} | Horizontal axis (arm/back): {result.horizontal_axis}",
            f"Seat height base {result.seat_height.base} total {result.seat_height.total}",
            f"Seat depth base {result.seat_depth.base} total {result.seat_depth.total}",
            f"Armrest base {result.armrest.base} total {result.armrest.total}",
            f"Back support base {result.back_support.base} total {result.back_support.total}",
        ]
        risk = "OK" if result.chair_score_final < 5 else "High"
        lines.append(f"Risk level: {risk}")
        return lines

    def run(self) -> None:
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {self.cam_index} cannot be opened")
        window_name = "ROSA Section A"
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
                    display = _draw_keypoints(frame, keypoints)
                    skeleton = Skeleton2D.from_array(keypoints)
                    now = time.time()
                    total_seconds = now - self.session_start
                    continuous_seconds = now - self.continuous_start
                    result = self.scorer.score(skeleton, total_seconds, continuous_seconds)
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
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    # mark a break in continuous exposure
                    self.continuous_start = time.time()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


__all__ = [
    "LiveSectionAApp",
    "SectionAScorer",
    "SectionAResult",
    "SubScore",
]
