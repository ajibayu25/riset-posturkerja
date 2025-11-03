"""Tkinter GUI for ROSA live scoring (Sections A, B, C)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
import platform
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from config import CAMERA_INDEX, CAMERA_PRESETS, DET_MODEL, EARPHONE_MODEL, DEVICE, EXPORT_CSV, EXPORT_JSONL, POSE_MODEL, SECTIONC_HAND
from constants.grids import (
    MONITOR_PHONE_GRID,
    SECTIONC_MOUSE_KEYBOARD_GRID,
    SECTION_A_GRID,
    SECTION_A_HORIZONTAL_AXIS,
    SECTION_A_VERTICAL_AXIS,
    SECTION_B_MONITOR_AXIS,
    SECTION_B_PHONE_AXIS,
    SECTION_C_KEYBOARD_AXIS,
    SECTION_C_MOUSE_AXIS,
)
from constants.thresholds import (
    SECTION_A_THRESHOLDS,
    SECTION_B_THRESHOLDS,
    SECTION_C_THRESHOLDS,
)
from core.geometry import Skeleton2D, clamp, distance
from core.smoothing import KeypointSmoother
from core.timers import duration_adjust
from rosa_io.exporters import export_csv, export_json
from models.detect import ObjectDetector
from models.pose import PoseEstimator
from scoring.sectiona import SectionAScorer
from scoring.sectionb import SectionBScorer
from scoring.sectionc import SectionCScorer

QUERY_DISPLAY = {
    'front': [
        ('elbows_supported', 'Elbows supported in line with shoulder, shoulders relaxed', 1),
        ('shoulder_height_issue', 'Too High (Shoulders Shrugged) / Low (Arms Unsupported)', 2),
        ('too_wide', 'Too Wide', 1),
        ('work_surface_high', 'Work Surface too High', 1),
        ('headset_or_phone_posture', 'Headset / One Hand on Phone & Neutral Neck Posture', 1),
        ('too_far_reach', 'Too Far of Reach (outside of 30 cm)', 2),
        ('neck_shoulder_hold', 'Neck and Shoulder Hold', 2),
        ('no_hands_free', 'No Hands-Free Options', 1),
        ('mouse_inline_shoulder', 'Mouse in line with shoulder', 1),
        ('reaching_to_mouse', 'Reaching to mouse', 2),
        ('mouse_keyboard_diff_surfaces', 'Mouse/Keyboard on different surfaces', 2),
        ('pinch_grip_mouse', 'Pinch grip on mouse', 1),
        ('keyboard_too_high', 'Keyboard too high - shoulders shrugged', 1),
    ],
    'side': [
        ('knees_at_90', 'Knees at 90 deg', 1),
        ('knee_angle_too_low', 'Too Low - Knee Angle < 90 deg', 2),
        ('knee_angle_too_high', 'Too High - Knee Angle > 90 deg', 2),
        ('no_foot_contact', 'No foot contact on ground', 2),
        ('insufficient_under_desk_space', 'Insufficient space under desk - ability to cross legs', 1),
        ('knee_to_seat_gap_ok', 'Approximately 3 inches of space between knee and edge of seat', 1),
        ('seat_too_long', 'Too Long - Less than 3 inches of space', 2),
        ('seat_too_short', 'Too Short - More than 3 inches of space', 2),
        ('adequate_lumbar_support', 'Adequate lumbar support - chair reclined between 95-110 deg', 1),
        ('poor_lumbar_position', 'No lumbar support or lumbar support not positioned in small of back', 2),
        ('chair_angle_out_of_range', 'Angled too far back (>110 deg) or too far forward (<95 deg)', 2),
        ('no_back_support', 'No back support (e.g., stool or worker leaning forward)', 3),
        ('hard_surface', 'Hard / damaged surface', 1),
    ],
    'overhead': [
        ('neck_twist_gt_30', 'Neck twist greater than 30 deg', 1),
        ('typing_deviation', 'Deviation while typing', 1),
    ],
}

BBox = Tuple[int, int, int, int]
COCO_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Return a copy of frame with pose keypoints/edges drawn for debugging."""
    vis = frame.copy()
    for x, y in keypoints:
        cv2.circle(vis, (int(x), int(y)), 4, color, -1)
    for a, b in COCO_EDGES:
        if a < len(keypoints) and b < len(keypoints):
            pa, pb = keypoints[a], keypoints[b]
            cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 200, 255), 2)
    return vis


def put_text_lines(frame: np.ndarray, lines: List[str], origin: Tuple[int, int] = (16, 32), color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Overlay a list of strings onto the frame starting at origin."""
    vis = frame.copy()
    x, y = origin
    for idx, text in enumerate(lines):
        cv2.putText(
            vis,
            text,
            (x, y + idx * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis


@dataclass
class PipelineResult:
    """Capture the latest processed frame and metadata summary."""
    frame: np.ndarray
    summary: Dict[str, float]


class BasePipeline:
    """Common webcam→pose→scoring loop shared by each section pipeline."""
    def __init__(self, cam_index: int, export_mode: str = "csv", smoothing_alpha: float = 0.3) -> None:
        """Open camera stream and prepare pose estimator & smoothing."""
        self.cam_index = cam_index
        self.export_mode = export_mode
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.pose = PoseEstimator(model_path=POSE_MODEL, device=DEVICE)
        self.smoother = KeypointSmoother(alpha=smoothing_alpha)
        self.session_start = time.time()
        self.continuous_start = self.session_start
        self.last_export_ts = 0.0
        self.export_interval = 5.0
        self.eval_interval = 10.0
        self.last_eval_ts = 0.0

    def is_opened(self) -> bool:
        """Return True if the capture device is ready."""
        return self.cap.isOpened()

    def reset_continuous(self) -> None:
        """Reset continuous exposure timer (triggered when breaks occur)."""
        self.continuous_start = time.time()

    def release(self) -> None:
        """Release underlying camera resource."""
        if self.cap.isOpened():
            self.cap.release()

    def _maybe_export(self, row: Dict[str, float]) -> None:
        """Write CSV/JSON rows at most every export_interval seconds."""
        if self.export_mode == "none":
            return
        now = time.time()
        if now - self.last_export_ts < self.export_interval:
            return
        self.last_export_ts = now
        if self.export_mode == "csv":
            export_csv(EXPORT_CSV, row)
        elif self.export_mode == "json":
            export_json(EXPORT_JSONL, row)

    def step(self) -> Optional[PipelineResult]:
        """Process next frame and return processed result or None if stream ended."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        ts = time.time()
        keypoints = self.pose.predict_xy(frame)
        if keypoints is not None:
            keypoints = self.smoother.update(keypoints[:, :2], timestamp=ts)
        evaluate = False
        if self.last_eval_ts == 0.0 or ts - self.last_eval_ts >= self.eval_interval:
            evaluate = True
            self.last_eval_ts = ts
        result = self.process_frame(frame, keypoints, ts, evaluate)
        remaining = max(0.0, self.eval_interval - (ts - self.last_eval_ts))
        result.summary.setdefault("next_update_in", remaining)
        return result

    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float, evaluate: bool) -> PipelineResult:
        raise NotImplementedError


class SectionAPipeline(BasePipeline):
    """Pipeline handling side camera (Section A chair scoring)."""
    def __init__(self, cam_index: int, export_mode: str = "csv", smoothing_alpha: float = 0.3) -> None:
        """Initialise ROSA Section A scorer for a dedicated camera feed."""
        super().__init__(cam_index, export_mode, smoothing_alpha)
        self.scorer = SectionAScorer()
        self._last_result: Optional[SectionAResult] = None
        self._last_summary: Optional[Dict[str, float]] = None
        self._last_updated_ts: float = 0.0

    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float, evaluate: bool) -> PipelineResult:
        """Render overlays and compute Section A metrics for the current frame."""
        display = frame.copy()
        if keypoints is None:
            display = put_text_lines(display, ["No pose detected"], color=(0, 0, 255))
            summary = self._last_summary.copy() if self._last_summary else {"score": float("nan"), "queries": {}, "updated_at": 0.0}
            return PipelineResult(display, summary)

        display = draw_skeleton(display, keypoints)
        skeleton = Skeleton2D.from_array(keypoints)

        if evaluate:
            total_seconds = timestamp - self.session_start
            continuous_seconds = timestamp - self.continuous_start
            result = self.scorer.score(
                skeleton,
                self.last_monitor_bbox,
                self.last_phone_bbox,
                self.last_audio_devices,
                frame.shape,
                total_seconds,
                continuous_seconds,
            )
            self._last_result = result
            self._last_updated_ts = result.timestamp
            self._maybe_export(result.to_row())
        result_obj = self._last_result

        if result_obj is not None:
            lines = [
                f"Section A score {result_obj.chair_score_final} (base {result_obj.chair_score_base}, dur {result_obj.duration_adjustment:+d})",
                f"Vertical axis: {result_obj.vertical_axis} | Horizontal axis: {result_obj.horizontal_axis}",
                f"Seat height {result_obj.seat_height.total} | Seat depth {result_obj.seat_depth.total}",
                f"Armrest {result_obj.armrest.total} | Back support {result_obj.back_support.total}",
            ]
            risk = "OK" if result_obj.chair_score_final < 5 else "High"
            lines.append(f"Risk: {risk}")
            display = put_text_lines(display, lines)
            summary = {
                "score": result_obj.chair_score_final,
                "vertical_axis": result_obj.vertical_axis,
                "horizontal_axis": result_obj.horizontal_axis,
                "queries": result_obj.query_breakdown,
                "updated_at": self._last_updated_ts,
            }
        else:
            summary = {"score": float("nan"), "queries": {}, "updated_at": 0.0}
        self._last_summary = summary
        return PipelineResult(display, summary)


class SectionBPipeline(BasePipeline):
    """Front camera pipeline scoring monitor/phone posture (Section B)."""
    def __init__(
        self,
        cam_index: int,
        export_mode: str = "csv",
        smoothing_alpha: float = 0.3,
        detection_stride: int = 5,
    ) -> None:
        """Attach detector and scorer needed for monitor & phone analysis."""
        super().__init__(cam_index, export_mode, smoothing_alpha)
        self.detector = ObjectDetector(model_path=DET_MODEL, device=DEVICE)
        self.audio_detector = ObjectDetector(model_path=EARPHONE_MODEL, device=DEVICE)
        self.scorer = SectionBScorer()
        self.detection_stride = max(1, detection_stride)
        self.frame_count = 0
        self.last_monitor_bbox: Optional[BBox] = None
        self.last_phone_bbox: Optional[BBox] = None
        self.last_audio_devices = []
        self._last_result: Optional[SectionBResult] = None
        self._last_summary: Optional[Dict[str, float]] = None
        self._last_updated_ts: float = 0.0

    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float, evaluate: bool) -> PipelineResult:
        """Blend detection + pose data to produce Section B overlays and metrics."""
        display = frame.copy()
        summary: Dict[str, float] = self._last_summary.copy() if self._last_summary else {"score": float("nan"), "queries": {}, "updated_at": 0.0}
        self.frame_count += 1
        if self.frame_count % self.detection_stride == 0:
            detections = self.detector.predict(frame)
            ear_detections = self.audio_detector.predict(frame)
            self.last_monitor_bbox = ObjectDetector.pick_monitor_bbox(detections)
            self.last_phone_bbox = ObjectDetector.pick_phone_bbox(detections)
            self.last_audio_devices = ObjectDetector.pick_audio_devices(detections, [ear_detections])

        if keypoints is None:
            if self.last_monitor_bbox is not None:
                x1, y1, x2, y2 = self.last_monitor_bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)
            if self.last_phone_bbox is not None:
                x1, y1, x2, y2 = self.last_phone_bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 120, 0), 2)
            for _label, _conf, bbox in self.last_audio_devices:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 127), 2)
            display = put_text_lines(display, ["No pose detected"], color=(0, 0, 255))
            return PipelineResult(display, summary)

        display = draw_skeleton(display, keypoints)
        skeleton = Skeleton2D.from_array(keypoints)
        if evaluate:
            total_seconds = timestamp - self.session_start
            continuous_seconds = timestamp - self.continuous_start
            result = self.scorer.score(
                skeleton,
                self.last_monitor_bbox,
                self.last_phone_bbox,
                self.last_audio_devices,
                frame.shape,
                total_seconds,
                continuous_seconds,
            )
            self._last_result = result
            self._last_updated_ts = result.timestamp
            self._maybe_export(result.to_row())
        result = self._last_result

        if self.last_monitor_bbox is not None:
            x1, y1, x2, y2 = self.last_monitor_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)
        if self.last_phone_bbox is not None:
            x1, y1, x2, y2 = self.last_phone_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 120, 0), 2)
        for _label, _conf, bbox in self.last_audio_devices:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 127), 2)

        if result is not None:
            lines = [
                f"Section B score {result.section_score} (dur {result.duration_adjustment:+d})",
                f"Monitor axis: {result.horizontal_axis} | Phone axis: {result.vertical_axis}",
                f"Monitor total {result.monitor.total} (base {result.monitor.base})",
                f"Phone total {result.phone.total} (base {result.phone.base})",
            ]
            risk = "OK" if result.section_score < 5 else "High"
            lines.append(f"Risk: {risk}")
            display = put_text_lines(display, lines)
            summary = {
                "score": result.section_score,
                "horizontal_axis": result.horizontal_axis,
                "vertical_axis": result.vertical_axis,
                "queries": result.query_breakdown,
                "updated_at": self._last_updated_ts,
            }
        self._last_summary = summary
        return PipelineResult(display, summary)


class SectionCPipeline(BasePipeline):
    """Overhead camera pipeline that scores mouse/keyboard posture (Section C)."""
    def __init__(
        self,
        cam_index: int,
        export_mode: str = "csv",
        smoothing_alpha: float = 0.3,
        hand_preference: str = "right",
    ) -> None:
        """Store settings for mouse-hand dominance and instantiate scorer."""
        super().__init__(cam_index, export_mode, smoothing_alpha)
        self.scorer = SectionCScorer()
        self.hand_preference = hand_preference.lower()
        self._last_result: Optional[SectionCResult] = None
        self._last_summary: Optional[Dict[str, float]] = None
        self._last_updated_ts: float = 0.0

    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float, evaluate: bool) -> PipelineResult:
        """Score current frame for Section C and build GUI-facing summary."""
        display = frame.copy()
        summary: Dict[str, float] = self._last_summary.copy() if self._last_summary else {"score": float("nan"), "queries": {}, "updated_at": 0.0}
        if keypoints is None:
            display = put_text_lines(display, ["No pose detected"], color=(0, 0, 255))
            return PipelineResult(display, summary)

        display = draw_skeleton(display, keypoints, color=(0, 255, 120))
        skeleton = Skeleton2D.from_array(keypoints)
        if evaluate:
            total_seconds = timestamp - self.session_start
            continuous_seconds = timestamp - self.continuous_start
            result = self.scorer.score(
                skeleton,
                self.hand_preference,
                total_seconds,
                continuous_seconds,
            )
            self._last_result = result
            self._last_updated_ts = result.timestamp
            self._maybe_export(result.to_row())
        result = self._last_result

        if result is not None:
            lines = [
                f"Section C score {result.section_score} (dur {result.duration_adjustment:+d})",
                f"Mouse axis: {result.vertical_axis} | Keyboard axis: {result.horizontal_axis}",
                f"Mouse total {result.mouse.total} (base {result.mouse.base})",
                f"Keyboard total {result.keyboard.total} (base {result.keyboard.base})",
            ]
            risk = "OK" if result.section_score < 5 else "High"
            lines.append(f"Risk: {risk}")
            display = put_text_lines(display, lines)

            summary = {
                "score": result.section_score,
                "vertical_axis": result.vertical_axis,
                "horizontal_axis": result.horizontal_axis,
                "queries": result.query_breakdown,
                "updated_at": self._last_updated_ts,
            }
        self._last_summary = summary
        return PipelineResult(display, summary)


PIPELINE_FACTORIES: Dict[str, Callable[..., BasePipeline]] = {
    "A": SectionAPipeline,
    "B": SectionBPipeline,
    "C": SectionCPipeline,
}


class ROSATkApp:
    """Simple single-section Tk application (legacy fallback)."""
    def __init__(self, root: tk.Tk) -> None:
        """Wire Tk widgets for manual Section selection and preview."""
        self.root = root
        self.root.title("ROSA Live Scoring")
        self.section_var = tk.StringVar(value="A")
        self.export_var = tk.StringVar(value="csv")
        self.cam_var = tk.IntVar(value=0)
        self.hand_var = tk.StringVar(value="right")
        self.status_var = tk.StringVar(value="Idle")
        self.score_var = tk.StringVar(value="Score: -")
        self.running = False
        self.pipeline: Optional[BasePipeline] = None
        self.photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        """Create control bar, preview label, and info widgets."""
        control_frame = ttk.Frame(self.root, padding=8)
        control_frame.grid(row=0, column=0, sticky="ew")
        control_frame.columnconfigure(5, weight=1)

        ttk.Label(control_frame, text="Section:").grid(row=0, column=0, padx=4)
        ttk.OptionMenu(control_frame, self.section_var, self.section_var.get(), "A", "B", "C").grid(row=0, column=1)

        ttk.Label(control_frame, text="Export:").grid(row=0, column=2, padx=4)
        ttk.OptionMenu(control_frame, self.export_var, self.export_var.get(), "csv", "json", "none").grid(row=0, column=3)

        ttk.Label(control_frame, text="Camera:").grid(row=0, column=4, padx=4)
        ttk.Entry(control_frame, textvariable=self.cam_var, width=4).grid(row=0, column=5)

        ttk.Label(control_frame, text="Hand:").grid(row=0, column=6, padx=4)
        ttk.OptionMenu(control_frame, self.hand_var, self.hand_var.get(), "right", "left").grid(row=0, column=7)

        ttk.Button(control_frame, text="Start", command=self.start).grid(row=0, column=8, padx=4)
        ttk.Button(control_frame, text="Stop", command=self.stop).grid(row=0, column=9, padx=4)
        ttk.Button(control_frame, text="Break", command=self.mark_break).grid(row=0, column=10, padx=4)

        ttk.Label(control_frame, textvariable=self.status_var, foreground="blue").grid(row=0, column=11, padx=8)

        self.video_label = ttk.Label(self.root)
        self.video_label.grid(row=1, column=0, padx=8, pady=8)

        info_frame = ttk.Frame(self.root, padding=8)
        info_frame.grid(row=2, column=0, sticky="ew")
        ttk.Label(info_frame, textvariable=self.score_var, font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")

    def start(self) -> None:
        """Start the chosen section pipeline and begin refreshing the UI."""
        if self.running:
            return
        section = self.section_var.get().upper()
        factory = PIPELINE_FACTORIES.get(section)
        if factory is None:
            messagebox.showerror("Error", f"Unsupported section {section}")
            return
        try:
            pipeline_kwargs = {
                "cam_index": int(self.cam_var.get()),
                "export_mode": self.export_var.get(),
            }
            if section == "B":
                pipeline_kwargs["detection_stride"] = 5
            if section == "C":
                pipeline_kwargs["hand_preference"] = self.hand_var.get()
            self.pipeline = factory(**pipeline_kwargs)
            if not self.pipeline.is_opened():
                raise RuntimeError("Camera cannot be opened")
        except Exception as exc:
            if self.pipeline:
                self.pipeline.release()
                self.pipeline = None
            messagebox.showerror("Error", str(exc))
            return

        self.running = True
        self.status_var.set(f"Running Section {section}")
        self._update_loop()

    def stop(self) -> None:
        """Stop active pipeline and reset UI state."""
        self.running = False
        if self.pipeline:
            self.pipeline.release()
            self.pipeline = None
        self.status_var.set("Stopped")
        self.score_var.set("Score: -")

    def mark_break(self) -> None:
        """Reset exposure timers to represent a user break."""
        if self.pipeline:
            self.pipeline.reset_continuous()
            self.status_var.set("Break marked - timer reset")

    def _update_loop(self) -> None:
        """Periodic Tk callback that fetches frames and updates the preview."""
        if not self.running or self.pipeline is None:
            return
        result = self.pipeline.step()
        if result is None:
            self.stop()
            return
        frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo)

        score = result.summary.get("score", float("nan"))
        if np.isnan(score):
            self.score_var.set("Score: -")
        else:
            if self.section_var.get().upper() == "A":
                self.score_var.set(
                    f"Score: {score:.0f} | Vertical {result.summary.get('vertical_axis', 0):.0f} | Horizontal {result.summary.get('horizontal_axis', 0):.0f}"
                )
            elif self.section_var.get().upper() == "B":
                self.score_var.set(
                    f"Score: {score:.0f} | Monitor {result.summary.get('horizontal_axis', 0):.0f} | Phone {result.summary.get('vertical_axis', 0):.0f}"
                )
            else:
                self.score_var.set(
                    f"Score: {score:.0f} | Mouse {result.summary.get('vertical_axis', 0):.0f} | Keyboard {result.summary.get('horizontal_axis', 0):.0f}"
                )
        self.root.after(33, self._update_loop)

    def on_close(self) -> None:
        """Stop running pipelines and close the Tk window cleanly."""
        self.stop()
        self.root.destroy()


class MultiSectionTkApp:
    """Display and log all ROSA sections simultaneously (three cameras)."""

    def __init__(self, root: tk.Tk, export_mode: str = "csv") -> None:
        """Set up shared Tk variables, placeholders, and panel layout state."""
        self.root = root
        self.export_mode = export_mode
        self.view_specs = [
            {"key": "B", "label": "Front", "query_key": "front"},
            {"key": "A", "label": "Side", "query_key": "side"},
            {"key": "C", "label": "Overhead", "query_key": "overhead"},
        ]
        self.section_order = [spec["key"] for spec in self.view_specs]
        self.section_specs = {spec["key"]: spec for spec in self.view_specs}
        self.status_var = tk.StringVar(value="Ready. Toggle cameras to begin.")
        self.photo_refs: Dict[str, Optional[ImageTk.PhotoImage]] = {sec: None for sec in self.section_order}
        self.score_vars: Dict[str, tk.StringVar] = {}
        self.video_labels: Dict[str, ttk.Label] = {}
        self.toggle_buttons: Dict[str, ttk.Button] = {}
        self.indicator_text_vars: Dict[str, tk.StringVar] = {}
        self.pipelines: Dict[str, BasePipeline] = {}
        self.section_running: Dict[str, bool] = {sec: False for sec in self.section_order}
        self.camera_presets = self._detect_available_cameras()
        self.camera_label_to_index = dict(self.camera_presets)
        self.camera_selection_vars: Dict[str, tk.StringVar] = {}
        self.preview_max_size: Tuple[int, int] = (360, 270)
        placeholder_image = Image.new("RGB", self.preview_max_size, color=(30, 30, 30))
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_image)

        self._build_ui()
        self._update_status()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._update_loop)

    def _system_camera_names(self) -> List[str]:
        """Fetch camera device names from the OS when running on Windows."""
        if platform.system() != "Windows":
            return []
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_PnPEntity | Where-Object {$_.PNPClass -eq 'Camera'} | Select-Object -ExpandProperty Name",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if result.returncode != 0:
                return []
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except Exception:
            return []

    def _detect_available_cameras(self) -> List[Tuple[str, Optional[int]]]:
        """Filter CAMERA_PRESETS, keeping only entries that open successfully."""
        names = self._system_camera_names()
        working_indices: List[int] = []
        for idx in range(8):
            if idx in working_indices:
                continue
            if platform.system() == "Windows":
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(idx)
            opened = cap.isOpened()
            cap.release()
            if opened:
                working_indices.append(idx)

        available: List[Tuple[str, Optional[int]]] = [("None", None)]
        if not working_indices:
            return available

        if names:
            label_counts: Dict[str, int] = {}
            for label, idx in zip(names, working_indices):
                count = label_counts.get(label, 0)
                label_counts[label] = count + 1
                display = f"{label} ({count + 1})" if count else label
                available.append((display, idx))
            for idx in working_indices[len(names):]:
                available.append((f"Camera {idx}", idx))
        else:
            for idx in working_indices:
                available.append((f"Camera {idx}", idx))
        return available

    def _default_camera_choice(self, section: str) -> str:
        """Pick initial dropdown value based on config CAMERA_INDEX."""
        desired = CAMERA_INDEX.get(section)
        for label, value in self.camera_presets:
            if value == desired:
                return label
        return self.camera_presets[0][0]

    def _resolve_camera_index(self, section: str) -> Optional[int]:
        """Translate selected camera label into an OpenCV index (or None)."""
        var = self.camera_selection_vars.get(section)
        if var is None:
            return None
        label = var.get()
        return self.camera_label_to_index.get(label)

    def _build_ui(self) -> None:
        """Create shared toolbar and section-specific panels."""
        control = ttk.Frame(self.root, padding=8)
        control.grid(row=0, column=0, sticky="ew")
        control.columnconfigure(3, weight=1)

        ttk.Button(control, text="Start All", command=self.start_all).grid(row=0, column=0, padx=4)
        ttk.Button(control, text="Stop All", command=self.stop).grid(row=0, column=1, padx=4)
        ttk.Button(control, text="Break", command=self.mark_break).grid(row=0, column=2, padx=4)
        ttk.Label(control, textvariable=self.status_var, foreground="blue").grid(row=0, column=3, padx=8, sticky="w")

        container = ttk.Frame(self.root, padding=8)
        container.grid(row=1, column=0, sticky="nsew")
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        for idx, section in enumerate(self.section_order):
            spec = self.section_specs[section]
            frame = ttk.LabelFrame(container, text=spec["label"], padding=4)
            frame.grid(row=0, column=idx, padx=6, pady=4, sticky="nsew")
            container.columnconfigure(idx, weight=1)

            camera_default = self._default_camera_choice(section)
            cam_var = tk.StringVar(value=camera_default)
            self.camera_selection_vars[section] = cam_var
            camera_row = ttk.Frame(frame)
            camera_row.pack(fill="x", padx=4, pady=(2, 4))
            ttk.Label(camera_row, text="Camera:").pack(side="left")
            camera_labels = [label for label, _ in self.camera_presets]
            ttk.OptionMenu(camera_row, cam_var, cam_var.get(), *camera_labels).pack(side="left", fill="x", expand=True, padx=(4, 0))

            video = tk.Label(
                frame,
                image=self.placeholder_photo,
                text="Camera off",
                compound="center",
                bg="#202020",
                fg="#f0f0f0",
            )
            video.pack(padx=4, pady=(0, 6))
            self.video_labels[section] = video
            self.photo_refs[section] = self.placeholder_photo

            friendly_label = spec["label"]
            score_var = tk.StringVar(value=f"{friendly_label} indicators total: -")
            self.score_vars[section] = score_var
            ttk.Label(
                frame,
                textvariable=score_var,
                font=("Segoe UI", 11, "bold"),
            ).pack(anchor="w", padx=4, pady=(0, 2))

            indicator_var = tk.StringVar(value=self._format_indicator_block(section, None))
            indicator_label = ttk.Label(
                frame,
                textvariable=indicator_var,
                anchor="w",
                justify="left",
                wraplength=320,
                font=("Segoe UI", 9),
            )
            indicator_label.pack(fill="x", padx=4, pady=(2, 4))
            self.indicator_text_vars[section] = indicator_var

            btn = ttk.Button(frame, text="Start", command=lambda s=section: self.toggle_section(s))
            btn.pack(pady=(0, 6))
            self.toggle_buttons[section] = btn

    def _format_indicator_block(self, section: str, summary: Optional[Dict[str, Any]]) -> str:
        """Build multiline string describing latest query values for a section."""
        spec = self.section_specs[section]
        lines: List[str] = []
        updated_at = None
        next_in = None
        if summary:
            updated_at = summary.get("updated_at")
            next_in = summary.get("next_update_in")
        if updated_at:
            lines.append(f"Last update: {time.strftime('%H:%M:%S', time.localtime(updated_at))}")
        else:
            lines.append("Last update: -")
        if isinstance(next_in, (int, float)):
            lines.append(f"Next capture: {max(0, int(round(next_in)))}s")
        else:
            lines.append("Next capture: -")
        lines.append("")
        query_scores: Dict[str, int] = {}
        if summary:
            queries = summary.get("queries")
            if isinstance(queries, dict):
                query_scores = {str(k): int(v) for k, v in queries.items()}
        for key, label, weight in QUERY_DISPLAY.get(spec["query_key"], []):
            value = int(query_scores.get(key, 0))
            marker = "[x]" if value > 0 else "[ ]"
            if weight is not None:
                lines.append(f"{marker} score={value} of {weight} - {label}")
            else:
                lines.append(f"{marker} score={value} - {label}")
        return "\n".join(lines)

    def _create_pipeline(self, section: str, cam_index: Optional[int] = None) -> BasePipeline:
        """Instantiate the appropriate pipeline for a section key."""
        if cam_index is None:
            cam_index = self._resolve_camera_index(section)
        if cam_index is None:
            friendly = self.section_specs[section]["label"]
            raise RuntimeError(f"Camera untuk panel {friendly} belum dipilih.")
        if section == "A":
            pipeline = SectionAPipeline(cam_index=cam_index, export_mode=self.export_mode)
        elif section == "B":
            pipeline = SectionBPipeline(cam_index=cam_index, export_mode=self.export_mode)
        else:
            pipeline = SectionCPipeline(
                cam_index=cam_index,
                export_mode=self.export_mode,
                hand_preference=SECTIONC_HAND,
            )
        if not pipeline.is_opened():
            pipeline.release()
            raise RuntimeError(f"Kamera {cam_index} untuk section {section} tidak dapat dibuka")
        return pipeline

    def toggle_section(self, section: str) -> None:
        """Start or stop a section feed depending on current state."""
        if self.section_running.get(section):
            self._stop_section(section)
        else:
            self._start_section(section)

    def _start_section(self, section: str) -> None:
        """Attempt to open camera and begin polling for a given section."""
        if self.section_running.get(section):
            return
        cam_index = self._resolve_camera_index(section)
        if cam_index is None:
            friendly = self.section_specs[section]["label"]
            messagebox.showinfo(
                "Camera disabled",
                f"Panel {friendly} disetel ke 'None'. Pilih kamera sebelum menyalakan.",
            )
            return
        try:
            pipeline = self._create_pipeline(section, cam_index=cam_index)
        except Exception as exc:
            messagebox.showerror("Camera error", str(exc))
            return
        self.pipelines[section] = pipeline
        self.section_running[section] = True
        self.toggle_buttons[section].configure(text="Stop")
        friendly = self.section_specs[section]["label"]
        self.score_vars[section].set(f"{friendly} indicators total: initializing...")
        self.indicator_text_vars[section].set(self._format_indicator_block(section, None))
        self.video_labels[section].configure(image=self.placeholder_photo, text="Connecting...", compound="center")
        self.photo_refs[section] = self.placeholder_photo
        self._update_status()

    def _stop_section(self, section: str) -> None:
        """Tear down camera pipeline and reset UI indicators."""
        pipeline = self.pipelines.pop(section, None)
        if pipeline is not None:
            pipeline.release()
        self.section_running[section] = False
        self.toggle_buttons[section].configure(text="Start")
        friendly = self.section_specs[section]["label"]
        self.score_vars[section].set(f"{friendly} indicators total: -")
        self.indicator_text_vars[section].set(self._format_indicator_block(section, None))
        self.photo_refs[section] = self.placeholder_photo
        self.video_labels[section].configure(image=self.placeholder_photo, text="Camera off", compound="center")
        self._update_status()

    def _update_status(self) -> None:
        """Refresh status bar text with currently running sections."""
        running = [self.section_specs[sec]["label"] for sec in self.section_order if self.section_running.get(sec)]
        if running:
            self.status_var.set("Running: " + ", ".join(running))
        else:
            self.status_var.set("Ready. Toggle cameras to begin.")

    def start_all(self) -> None:
        """Start every section that is not disabled and not yet running."""
        for section in self.section_order:
            if not self.section_running.get(section):
                self._start_section(section)
        self._update_status()

    def mark_break(self) -> None:
        """Propagate break marker to all active pipelines."""
        for pipeline in self.pipelines.values():
            pipeline.reset_continuous()
        self.status_var.set("Break recorded")

    def stop(self) -> None:
        """Stop all active sections."""
        for section in list(self.section_order):
            if self.section_running.get(section):
                self._stop_section(section)

    def _update_loop(self) -> None:
        """Tk polling loop that refreshes video previews and indicator text."""
        for section, pipeline in list(self.pipelines.items()):
            result = pipeline.step()
            if result is None:
                messagebox.showwarning("Camera warning", f"Stream section {section} terputus. Kamera dimatikan.")
                self._stop_section(section)
                continue
            frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            max_w, max_h = self.preview_max_size
            if image.width > max_w or image.height > max_h:
                image.thumbnail(self.preview_max_size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=image)
            self.photo_refs[section] = photo
            self.video_labels[section].configure(image=photo, text="", compound="center")

            score = result.summary.get("score", float("nan"))
            friendly = self.section_specs[section]["label"]
            if np.isnan(score):
                text = f"{friendly} indicators total: -"
            else:
                text = f"{friendly} indicators total: {score:.0f}"
            updated_at = result.summary.get("updated_at")
            if updated_at:
                text += f" | updated {time.strftime('%H:%M:%S', time.localtime(updated_at))}"
            next_in = result.summary.get("next_update_in")
            if isinstance(next_in, (int, float)):
                text += f" | next {max(0, int(round(next_in)))}s"
            self.score_vars[section].set(text)
            self.indicator_text_vars[section].set(self._format_indicator_block(section, result.summary))
        self.root.after(33, self._update_loop)

    def on_close(self) -> None:
        """Handle window close by stopping all pipelines first."""
        self.stop()
        self.root.destroy()

def main(multi: bool = True) -> None:
    """Launch Tkinter ROSA application in multi or single-section mode."""
    root = tk.Tk()
    if multi:
        MultiSectionTkApp(root)
    else:
        ROSATkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main(multi=True)

