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
from core.geometry import Skeleton2D, clamp
from core.smoothing import EMA
from core.timers import duration_adjust
from rosa_io.exporters import export_csv, export_json
from models.pose import PoseEstimator
from sensory.side import (
    armrest_components,
    back_support_components,
    seat_depth_components,
    seat_height_components,
    get_work_surface_flag,
)


@dataclass
class SubScore:
    """Container holding base score, adjustments, and sensor metrics for one indicator."""
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
    """Full scoring outcome for Section A at a single timestamp."""
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
    query_breakdown: Dict[str, int]

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


class SectionAScorer:
    """Transform pose skeletons into ROSA Section A scores."""
    def score(
        self,
        skeleton: Skeleton2D,
        total_seconds: float,
        continuous_seconds: float,
        desk_info: Optional[Tuple[Tuple[int, int, int, int], float]] = None,
        chair_info: Optional[Tuple[Tuple[int, int, int, int], float]] = None,
    ) -> SectionAResult:
        seat_height_comp = seat_height_components(skeleton, desk_info)
        seat_depth_comp = seat_depth_components(skeleton, chair_info)
        armrest_comp = armrest_components(skeleton)
        back_support_comp = back_support_components(skeleton)

        seat_height = SubScore("seat_height", seat_height_comp.base, seat_height_comp.adjustments, seat_height_comp.metrics)
        seat_depth = SubScore("seat_depth", seat_depth_comp.base, seat_depth_comp.adjustments, seat_depth_comp.metrics)
        armrest = SubScore("armrest", armrest_comp.base, armrest_comp.adjustments, armrest_comp.metrics)
        back_support = SubScore("back_support", back_support_comp.base, back_support_comp.adjustments, back_support_comp.metrics)

        query_breakdown: Dict[str, int] = {}
        for comp in (seat_height_comp, seat_depth_comp, armrest_comp, back_support_comp):
            query_breakdown.update(comp.queries)
        work_flag = int(get_work_surface_flag())
        query_breakdown["work_surface_too_high"] = work_flag

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
            query_breakdown=query_breakdown,
        )


COCO_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def _draw_keypoints(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Overlay COCO skeleton on top of frame for debug visualisation."""
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
    """Standalone OpenCV window for Section A live scoring (debug helper)."""
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
        """Optionally smooth keypoints frame-to-frame."""
        if self.ema is None:
            return keypoints
        flat = keypoints.reshape(-1)
        smoothed = self.ema.update(flat)
        return smoothed.reshape(keypoints.shape)

    def _export(self, result: SectionAResult) -> None:
        """Persist the structured result to CSV or JSON logs."""
        if self.export_mode == "none":
            return
        row = result.to_row()
        if self.export_mode == "csv":
            export_csv(EXPORT_CSV, row)
        elif self.export_mode == "json":
            export_json(EXPORT_JSONL, row)

    def _format_overlay(self, result: SectionAResult) -> List[str]:
        """Compose status text overlay summarising key risk points."""
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
        """Open the webcam loop until the user quits, updating scores live."""
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
