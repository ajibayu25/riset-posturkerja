"""Tkinter GUI for ROSA live scoring (Sections A, B, C)."""



from __future__ import annotations



import copy

import threading

import time


from dataclasses import dataclass
from pathlib import Path

import platform

import subprocess

import json

from typing import Any, Callable, Dict, List, Optional, Tuple



import cv2

import numpy as np

import tkinter as tk

from tkinter import ttk, messagebox, simpledialog

from PIL import Image, ImageTk



# Path constants for image assets
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
HARD_SURFACE_IMAGE = ASSETS_DIR / "hard_surface.png"

from config import (
    CAMERA_DEFAULTS,
    CAMERA_INDEX,
    CAMERA_PRESETS,
    DET_MODEL,
    EARPHONE_MODEL,
    HAND_MODEL,
    DEVICE,
    EXPORT_CSV,
    EXPORT_JSONL,
    EXPORT_XLSX,
    GLARE_SERIAL_PORT,
    GLARE_BAUDRATE,
    POSE_MODEL,
    SECTIONC_HAND,
)

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

from rosa_io.exporters import export_csv, export_json, export_excel_row

from rosa_io.excel_schema import EXCEL_HEADERS, build_excel_row

from models.detect import ObjectDetector, BBox

from models.pose import PoseEstimator

from scoring.monitor_peripherals import MonitorPeripheralScorer

from scoring.rosa_total import ROSATotalScorer

from scoring.sectiona import SectionAScorer, SectionAResult

from scoring.sectionb import SectionBScorer, SectionBResult

from scoring.sectionc import SectionCScorer, SectionCResult
from sensory.side import (
    assess_mouse_keyboard_surfaces,
    assess_work_surface_elevation,
    detect_palmrest_side,
)
from sensory.glare_serial import GlareSerialClient



QUERY_DISPLAY = {

    'front': [

        ('elbows_supported_in_line_with_shoulder_shoulders_relaxed', 'Elbows supported in line with shoulder, shoulders relaxed', 1),

        ('too_high_shoulders_shrugged_or_low_arms_unsupported', 'Too High (Shoulders Shrugged) / Low (Arms Unsupported)', 2),

        ('too_wide', 'Too Wide', 1),

        ('headset_or_one_hand_on_phone_neutral_neck_posture', 'Headset / One Hand on Phone & Neutral Neck Posture', 1),

        ('too_far_of_reach_outside_30_cm', 'Too Far of Reach (outside of 30 cm)', 2),

        ('neck_and_shoulder_hold', 'Neck and Shoulder Hold', 2),
        ('neck_twist_greater_than_30_deg', 'Neck twist greater than 30 deg', 1),
        ('no_hands_free_options', 'No Hands-Free Options', 1),

        ('keyboard_too_high_shoulders_shrugged', 'Keyboard too high - shoulders shrugged', 1),

    ],

    'side': [

        ('knees_at_90_deg', 'Knees at 90 degree', 1),

        ('too_low_knee_angle_less_than_90_deg', 'Too Low - Knee Angle < 90 deg', 2),

        ('too_high_knee_angle_greater_than_90_deg', 'Too High - Knee Angle > 90 deg', 2),

        ('no_foot_contact_on_ground', 'No foot contact on ground', 2),

        ('insufficient_space_under_desk_ability_to_cross_legs', 'Insufficient space under desk - ability to cross legs', 1),

        ('approximately_three_inches_between_knee_and_seat_edge', 'Approximately 3 inches of space between knee and edge of seat', 1),

        ('too_long_less_than_three_inches_of_space', 'Too Long - Less than 3 inches of space', 2),

        ('too_short_more_than_three_inches_of_space', 'Too Short - More than 3 inches of space', 2),

        ('adequate_lumbar_support_chair_reclined_between_95_110_deg', 'Adequate lumbar support - chair reclined between 95-110 deg', 1),

        ('no_lumbar_support_or_not_positioned_in_small_of_back', 'No lumbar support or lumbar support not positioned in small of back', 2),

        ('angled_too_far_back_greater_than_110_or_too_far_forward_less_than_95', 'Angled too far back (>110 deg) or too far forward (<95 deg)', 2),

        ('no_back_support_or_worker_leaning_forward', 'No back support (e.g., stool or worker leaning forward)', 3),

        ('hard_or_damaged_surface', 'Hard / damaged surface', 1),
        ('work_surface_too_high', 'Work Surface too High', 1),
        ('armrests_too_wide', 'Armrests too wide', 1),
        ('palmrest_in_front_of_mouse', 'Palmrest in front of mouse', 1),
        ('mouse_keyboard_on_different_surfaces', 'Mouse/Keyboard on different surfaces', 2),

    ],

    'overhead': [

        ('deviation_while_typing', 'Deviation while typing', 1),
        ('mouse_in_line_with_shoulder', 'Mouse in line with shoulder', 1),
        ('reaching_to_mouse', 'Reaching to mouse', 2),
        ('pinch_grip_on_mouse', 'Pinch grip on mouse', 1),
        ('documents_used_no_document_holder', 'Documents used - no document holder', 1),

    ],

}

SNAPSHOT_ROOT = Path("snapshots")


def save_snapshot(section: str, frame: np.ndarray, timestamp: float) -> None:
    """Persist a frame to disk for later review (best-effort)."""
    try:
        folder = SNAPSHOT_ROOT / section
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        filename = folder / f"{stamp}.jpg"
        cv2.imwrite(str(filename), frame)
    except Exception:
        # Snapshot failures shouldn't break scoring loop.
        return


class IndicatorPanel(ttk.Frame):
    """Scrollable collection of colored indicator rows."""

    def __init__(self, master: tk.Widget, width: int = 360, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self._preferred_width = width
        self.canvas = tk.Canvas(self, highlightthickness=0, width=width, bg="#f7f7f7")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg="#f7f7f7")
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind(
            "<Configure>",
            lambda e: setattr(self, "_preferred_width", max(e.width, 240)),
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.rows: List[tk.Widget] = []

    def set_rows(self, rows: List[Tuple[str, str, str]]) -> None:
        """Replace current rows with colored status badges + labels."""
        for row in self.rows:
            row.destroy()
        self.rows.clear()

        color_map = {
            "ok": {
                "badge_bg": "#0fb268",
                "badge_fg": "#ffffff",
                "row_bg": "#ffffff",
                "border": "#c5eeda",
            },
            "alert": {
                "badge_bg": "#f2453d",
                "badge_fg": "#ffffff",
                "row_bg": "#ffffff",
                "border": "#f8caca",
            },
            "unknown": {
                "badge_bg": "#a7a7a7",
                "badge_fg": "#ffffff",
                "row_bg": "#f6f6f6",
                "border": "#dcdcdc",
            },
        }
        for status_text, description, tag in rows:
            colors = color_map.get(tag, color_map["unknown"])
            row_canvas = tk.Canvas(
                self.inner,
                bg=colors["row_bg"],
                highlightthickness=0,
                height=44,
                width=max(self._preferred_width - 12, 240),
            )
            row_canvas.pack(fill="x", padx=6, pady=3)
            row_canvas.update_idletasks()
            width = max(row_canvas.winfo_width(), self._preferred_width - 12, 240)
            row_canvas.config(width=width)
            row_canvas.create_rectangle(
                0,
                0,
                width,
                44,
                outline=colors["border"],
                width=1,
                fill=colors["row_bg"],
            )
            badge_width = 86
            row_canvas.create_rectangle(
                12,
                10,
                12 + badge_width,
                34,
                outline="",
                fill=colors["badge_bg"],
            )
            row_canvas.create_text(
                12 + badge_width / 2,
                22,
                text=status_text,
                fill=colors["badge_fg"],
                font=("Segoe UI", 10, "bold"),
            )
            row_canvas.create_text(
                12 + badge_width + 14,
                22,
                text=description,
                fill="#0f0f0f",
                anchor="w",
                font=("Segoe UI", 10),
                width=width - (12 + badge_width + 20),
            )
            self.rows.append(row_canvas)


class ArmrestSurfaceDialog(simpledialog.Dialog):
    """Modal dialog presenting armrest surface conditions for hard-surface query."""

    OPTIONS = [
        ("Empuk (berpadding / ada bantalan)", "empuk"),
        ("Licin (tidak ada tekstur / terasa keras)", "licin"),
        ("Keras / rusak (langsung kayu, plastik, metal)", "keras"),
    ]

    def __init__(
        self,
        master: tk.Widget,
        default: Optional[str] = None,
        image_path: Optional[Path] = None,
    ) -> None:
        self._selection = tk.StringVar()
        normalized = (default or "").strip().lower()
        if normalized in {"empuk", "licin", "keras"}:
            self._selection.set(normalized)
        else:
            self._selection.set("empuk")
        self._image_path = image_path
        self._photo: Optional[ImageTk.PhotoImage] = None
        super().__init__(master, title="Kondisi permukaan armrest")

    def body(self, master: tk.Widget) -> None:
        if self._image_path and self._image_path.exists():
            try:
                image = Image.open(self._image_path)
                image.thumbnail((360, 200), Image.LANCZOS)
                self._photo = ImageTk.PhotoImage(image)
                ttk.Label(master, image=self._photo).pack(anchor="center", pady=(0, 4))
            except Exception:
                pass

        ttk.Label(
            master,
            text="Pilih kondisi permukaan sandaran tangan (armrest) sesuai pengamatan:",
            padding=(0, 4),
            wraplength=360,
            justify="left",
        ).pack(anchor="w")
        for label, value in self.OPTIONS:
            ttk.Radiobutton(
                master,
                text=label,
                value=value,
                variable=self._selection,
            ).pack(anchor="w", padx=4, pady=2)

    def apply(self) -> None:
        self.result = self._selection.get()

class PostSessionDialog(simpledialog.Dialog):
    """Modal dialog to capture all questionnaire-based ROSA items on exit."""

    def __init__(self, master: tk.Widget) -> None:
        self.results: Dict[str, Any] = {}
        self._vars: Dict[str, tk.Variable] = {}
        self._photos: Dict[str, ImageTk.PhotoImage] = {}
        super().__init__(master, title="Kuesioner Akhir Sesi")

    def body(self, master: tk.Widget) -> None:
        main_frame = ttk.Frame(master)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # --- Armrest Surface ---
        armrest_frame = ttk.LabelFrame(main_frame, text="1. Kondisi Permukaan Armrest", padding=6)
        armrest_frame.pack(fill="x", pady=(0, 10))

        if HARD_SURFACE_IMAGE.exists():
            try:
                image = Image.open(HARD_SURFACE_IMAGE)
                image.thumbnail((240, 135), Image.LANCZOS)
                self._photos["armrest_surface"] = ImageTk.PhotoImage(image)
                ttk.Label(armrest_frame, image=self._photos["armrest_surface"]).pack(anchor="center", pady=(0, 6))
            except Exception:
                pass
        armrest_var = tk.StringVar(value="empuk")
        self._vars["hard_or_damaged_surface"] = armrest_var
        for text, val in ArmrestSurfaceDialog.OPTIONS:
            ttk.Radiobutton(armrest_frame, text=text, value=val, variable=armrest_var).pack(anchor="w")

        # --- Non-Adjustable Components ---
        adj_frame = ttk.LabelFrame(main_frame, text="2. Komponen yang Tidak Dapat Disesuaikan", padding=6)
        adj_frame.pack(fill="x")
        
        adj_items = [
            ("seat_height_non_adjustable", "Tinggi Kursi", "non_adjustable_seat_height.png"),
            ("seat_depth_non_adjustable", "Kedalaman Kursi", "non_adjustable_seat_depth.png"),
            ("armrest_non_adjustable", "Sandaran Tangan (Armrest)", "non_adjustable_armrest.png"),
            ("back_support_non_adjustable", "Sandaran Punggung", "non_adjustable_back_support.png"),
            ("keyboard_platform_non_adjustable", "Platform Keyboard", "non_adjustable_keyboard.png"),
        ]

        for key, label, img_name in adj_items:
            item_frame = ttk.Frame(adj_frame)
            item_frame.pack(fill="x", pady=4)

            img_path = ASSETS_DIR / img_name
            if img_path.exists():
                try:
                    image = Image.open(img_path)
                    image.thumbnail((120, 80), Image.LANCZOS)
                    self._photos[key] = ImageTk.PhotoImage(image)
                    img_label = ttk.Label(item_frame, image=self._photos[key])
                    img_label.pack(side="left", padx=(0, 10))
                except Exception:
                    pass

            var = tk.BooleanVar(value=False)
            self._vars[key] = var
            cb = ttk.Checkbutton(item_frame, text=label, variable=var)
            cb.pack(side="left", anchor="w")

        return main_frame

    def apply(self) -> None:
        self.results = {}
        # Armrest surface
        armrest_choice = self._vars["hard_or_damaged_surface"].get()
        self.results["hard_or_damaged_surface"] = 1 if armrest_choice != "empuk" else 0

        # Non-adjustable flags
        adj_keys = [
            "seat_height_non_adjustable",
            "seat_depth_non_adjustable",
            "armrest_non_adjustable",
            "back_support_non_adjustable",
            "keyboard_platform_non_adjustable",
        ]
        for key in adj_keys:
            if self._vars[key].get():
                self.results[key] = 1


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
    summary: Dict[str, Any]


class BasePipeline:
    """Common webcam-to-pose scoring loop shared by each section pipeline."""

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

    def __init__(
        self,
        cam_index: int,
        export_mode: str = "csv",
        smoothing_alpha: float = 0.3,
        draw_skeleton: bool = True,
    ) -> None:
        super().__init__(cam_index, export_mode, smoothing_alpha)
        self.scorer = SectionAScorer()
        self.detector = ObjectDetector(model_path=DET_MODEL, device=DEVICE)
        self._last_result: Optional[SectionAResult] = None
        self._last_summary: Optional[Dict[str, Any]] = None
        self._last_updated_ts: float = 0.0
        self._last_mouse_bbox: Optional[BBox] = None
        self._last_desk_info: Optional[Tuple[BBox, float]] = None
        self._last_chair_info: Optional[Tuple[BBox, float]] = None
        self._draw_skeleton = draw_skeleton
        self._floor_line_y: Optional[float] = None
        # Run heavy desk/chair detection sparingly to keep UI responsive.
        self._desk_detection_stride = 15
        self._frame_counter = 0

    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float, evaluate: bool) -> PipelineResult:
        """Render overlays and compute Section A metrics for the current frame."""
        display = frame.copy()
        previous = self._last_summary or {}
        prev_queries = previous.get("queries", {})
        prev_metrics = previous.get("metrics", {})

        # Only run YOLO desk/chair detection occasionally to keep FPS stable.
        self._frame_counter += 1
        run_detection = (
            evaluate
            or self._last_desk_info is None
            or self._last_chair_info is None
            or self._frame_counter % self._desk_detection_stride == 0
        )
        if run_detection:
            detections = self.detector.predict(frame)
            self._last_desk_info = ObjectDetector.pick_table_candidate(detections)
            self._last_chair_info = ObjectDetector.pick_chair_candidate(detections)
            self._update_floor_line(frame.shape[0])

        if self._last_desk_info is not None:
            (dx1, dy1, dx2, dy2), _conf = self._last_desk_info
            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), (180, 120, 20), 2)
        if self._last_chair_info is not None:
            (cx1, cy1, cx2, cy2), _ = self._last_chair_info
            cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (90, 180, 255), 2)

        if keypoints is None:
            summary = {
                "score": previous.get("score", float("nan")),
                "queries": dict(prev_queries) if isinstance(prev_queries, dict) else {},
                "metrics": dict(prev_metrics) if isinstance(prev_metrics, dict) else {},
                "updated_at": previous.get("updated_at", 0.0),
                "vertical_axis": previous.get("vertical_axis"),
                "horizontal_axis": previous.get("horizontal_axis"),
                "section_result": previous.get("section_result"),
                "just_updated": False,
            }
            display = put_text_lines(display, ["No pose detected"], color=(0, 0, 255))
            return PipelineResult(display, summary)

        if self._draw_skeleton:
            display = draw_skeleton(display, keypoints)
        skeleton = Skeleton2D.from_array(keypoints)

        # Heuristic palmrest detection uses the same side feed; returns coarse metrics.
        palm_detection = detect_palmrest_side(frame, skeleton, SECTIONC_HAND)
        palmrest_flag = palm_detection.get("flag")
        palmrest_metrics = palm_detection.get("metrics", {})

        surface_metrics = assess_mouse_keyboard_surfaces(skeleton, SECTIONC_HAND)
        work_metrics = assess_work_surface_elevation(skeleton, SECTIONC_HAND)

        just_updated = False
        if evaluate:
            total_seconds = timestamp - self.session_start
            continuous_seconds = timestamp - self.continuous_start
            save_snapshot("A", frame, timestamp)
            result = self.scorer.score(
                skeleton,
                total_seconds,
                continuous_seconds,
                desk_info=self._last_desk_info,
                chair_info=self._last_chair_info,
                palmrest_flag=palmrest_flag,
                palmrest_metrics=palmrest_metrics,
            )
            self._last_result = result
            self._last_updated_ts = result.timestamp
            self._maybe_export(result.to_row())
            just_updated = True
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
            metrics = result_obj.seat_height.metrics
            floor_y = None
            if isinstance(metrics, dict):
                val = metrics.get("estimated_floor_y")
                if val is not None:
                    floor_y = float(val)
                contact_flag = int(metrics.get("foot_contact_flag", 1))
            if floor_y is None:
                floor_y = self._floor_line_y
            floor_y = float(floor_y) if floor_y is not None else float("nan")
            if not np.isnan(floor_y):
                floor_y_int = int(floor_y)
                no_foot_flag = int(result_obj.query_breakdown.get("no_foot_contact_on_ground", 0))
                floor_color = (0, 210, 0) if no_foot_flag == 0 else (20, 40, 220)
                cv2.line(display, (0, floor_y_int), (display.shape[1], floor_y_int), floor_color, 2)
                lines.append(f"Floor ref y={floor_y_int} ({'contact' if no_foot_flag == 0 else 'no contact'})")
            display = put_text_lines(display, lines)
            summary = {
                "score": result_obj.chair_score_final,
                "vertical_axis": result_obj.vertical_axis,
                "horizontal_axis": result_obj.horizontal_axis,
                "queries": dict(result_obj.query_breakdown),
                "metrics": {},
                "updated_at": self._last_updated_ts,
                "section_result": result_obj,
                "just_updated": just_updated,
            }
        else:
            summary = {
                "score": float("nan"),
                "queries": dict(prev_queries) if isinstance(prev_queries, dict) else {},
                "metrics": dict(prev_metrics) if isinstance(prev_metrics, dict) else {},
                "updated_at": previous.get("updated_at", 0.0),
                "vertical_axis": previous.get("vertical_axis"),
                "horizontal_axis": previous.get("horizontal_axis"),
                "section_result": None,
                "just_updated": False,
            }

        queries = summary.get("queries", {})
        queries["mouse_keyboard_on_different_surfaces"] = int(surface_metrics.get("mouse_keyboard_surface_flag", 0))
        queries["work_surface_too_high"] = int(work_metrics.get("work_surface_flag", 0))
        if palmrest_flag is True:
            queries["palmrest_in_front_of_mouse"] = 1
        elif palmrest_flag is False:
            queries.setdefault("palmrest_in_front_of_mouse", 0)
        summary["queries"] = queries

        metrics = summary.get("metrics", {})
        metrics.update(surface_metrics)
        metrics.update(work_metrics)
        if palmrest_metrics:
            metrics.update({f"palmrest_{k}": float(v) for k, v in palmrest_metrics.items()})
        if self._floor_line_y is not None:
            metrics["floor_line_y"] = float(self._floor_line_y)
        summary["metrics"] = metrics

        self._maybe_calibrate_floor_line(skeleton, result_obj)
        self._apply_floor_contact_override(skeleton, summary)
        self._last_summary = summary
        return PipelineResult(display, summary)

    def _update_floor_line(self, frame_height: int) -> None:
        baseline = frame_height - 8
        if self._last_chair_info is not None:
            (cx1, cy1, cx2, cy2), _ = self._last_chair_info
            baseline = min(frame_height - 4, cy2 + 4)
        if self._floor_line_y is None:
            self._floor_line_y = float(baseline)
        else:
            self._floor_line_y = 0.8 * self._floor_line_y + 0.2 * float(baseline)

    def _apply_floor_contact_override(self, skeleton: Skeleton2D, summary: Dict[str, Any]) -> None:
        if self._floor_line_y is None:
            return
        cfg = SECTION_A_THRESHOLDS["seat_height"]
        tol_ratio = cfg.get("floor_line_tolerance_ratio", 0.08)
        tol_px = cfg.get("floor_line_tolerance_px", 12.0)
        contacts: List[bool] = []
        for side in ("left", "right"):
            ankle = skeleton.point(f"{side}_ankle")
            if ankle is None:
                continue
            hip = skeleton.point(f"{side}_hip")
            knee = skeleton.point(f"{side}_knee")
            leg_len = float("nan")
            if hip is not None:
                leg_len = float(distance(hip, ankle))
            elif knee is not None:
                leg_len = float(distance(knee, ankle))
            tol = max(tol_px, (leg_len if not np.isnan(leg_len) else 0.0) * tol_ratio)
            gap = float(self._floor_line_y) - float(ankle[1])
            contacts.append(gap <= tol)
        if not contacts:
            return
        foot_contact = any(contacts)
        value = 0 if foot_contact else 2
        summary.setdefault("queries", {})["no_foot_contact_on_ground"] = value
        metrics = summary.setdefault("metrics", {})
        metrics["floordet_contacts"] = int(foot_contact)
        metrics["floor_line_y"] = float(self._floor_line_y)
        result = summary.get("section_result")
        if result is not None and hasattr(result, "query_breakdown"):
            result.query_breakdown["no_foot_contact_on_ground"] = value
            seat_metrics = getattr(result.seat_height, "metrics", None)
            if isinstance(seat_metrics, dict):
                seat_metrics["estimated_floor_y"] = float(self._floor_line_y)

    def _maybe_calibrate_floor_line(self, skeleton: Skeleton2D, section_result: Optional[SectionAResult]) -> None:
        if self._floor_line_y is None or skeleton is None or section_result is None:
            return
        metrics = getattr(section_result.seat_height, "metrics", None)
        if not isinstance(metrics, dict):
            return
        contact_flag = metrics.get("foot_contact_flag")
        if contact_flag is None:
            return
        try:
            contact_flag = int(contact_flag)
        except (TypeError, ValueError):
            return
        if contact_flag <= 0:
            return
        ankle_points = [skeleton.point("left_ankle"), skeleton.point("right_ankle")]
        y_values = [float(pt[1]) for pt in ankle_points if pt is not None]
        if not y_values:
            return
        target = max(y_values)
        self._floor_line_y = 0.7 * self._floor_line_y + 0.3 * target

class SectionBPipeline(BasePipeline):

    """Front camera pipeline scoring monitor/phone posture (Section B)."""

    def __init__(

        self,

        cam_index: int,

        export_mode: str = "csv",

        smoothing_alpha: float = 0.3,

        detection_stride: int = 10,

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
        self.document_artifacts: Dict[str, List[BBox]] = {"holders": [], "bundles": []}

        self._last_result: Optional[SectionBResult] = None

        self._last_summary: Optional[Dict[str, float]] = None

        self._last_updated_ts: float = 0.0



    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float, evaluate: bool) -> PipelineResult:

        """Blend detection + pose data to produce Section B overlays and metrics."""

        display = frame.copy()

        previous = self._last_summary or {}

        prev_queries = previous.get("queries", {})

        summary: Dict[str, Any] = {

            "score": previous.get("score", float("nan")),

            "queries": dict(prev_queries) if isinstance(prev_queries, dict) else {},

            "updated_at": previous.get("updated_at", 0.0),

            "horizontal_axis": previous.get("horizontal_axis"),

            "vertical_axis": previous.get("vertical_axis"),

            "section_result": previous.get("section_result"),

            "just_updated": False,

        }

        # Only invoke the detector when necessary to keep throughput high.
        # Throttle YOLO inference so the front camera remains responsive.
        self.frame_count += 1

        run_detection = (
            evaluate
            or self.frame_count % self.detection_stride == 0
            or self.last_monitor_bbox is None
            or self.last_phone_bbox is None
        )

        if run_detection:

            detections = self.detector.predict(frame)

            ear_detections = self.audio_detector.predict(frame)

            self.last_monitor_bbox = ObjectDetector.pick_monitor_bbox(detections)

            self.last_phone_bbox = ObjectDetector.pick_phone_bbox(detections)

            self.last_audio_devices = ObjectDetector.pick_audio_devices(detections, [ear_detections])
            self.document_artifacts = ObjectDetector.detect_document_artifacts(detections)



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

        just_updated = False

        if evaluate:

            total_seconds = timestamp - self.session_start

            continuous_seconds = timestamp - self.continuous_start

            # Store a still for auditing / dataset building.
            save_snapshot("B", frame, timestamp)

            result = self.scorer.score(
                skeleton,
                self.last_monitor_bbox,
                self.last_phone_bbox,
                self.last_audio_devices,
                frame.shape,
                total_seconds,
                continuous_seconds,
                self.document_artifacts,
            )

            self._last_result = result

            self._last_updated_ts = result.timestamp

            self._maybe_export(result.to_row())

            just_updated = True

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

            summary.update(

                {

                    "score": result.section_score,

                    "horizontal_axis": result.horizontal_axis,

                    "vertical_axis": result.vertical_axis,

                    "queries": result.query_breakdown,

                    "updated_at": self._last_updated_ts,

                    "section_result": result,

                    "just_updated": just_updated,

                }

            )

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
        detection_stride: int = 12,
        draw_skeleton: bool = True,

    ) -> None:

        """Store settings for mouse-hand dominance and instantiate scorer."""

        super().__init__(cam_index, export_mode, smoothing_alpha)

        self.detector = ObjectDetector(model_path=DET_MODEL, device=DEVICE)

        self.hand_detector = ObjectDetector(model_path=HAND_MODEL, device=DEVICE)

        self.scorer = SectionCScorer()

        self.hand_preference = hand_preference.lower()

        self._last_result: Optional[SectionCResult] = None

        self._last_summary: Optional[Dict[str, float]] = None

        self._last_updated_ts: float = 0.0

        self._last_mouse_bbox: Optional[BBox] = None

        self._last_hand_bboxes: List[BBox] = []

        self.detection_stride = max(1, detection_stride)
        self._frame_index = 0
        self._draw_skeleton = draw_skeleton



    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float, evaluate: bool) -> PipelineResult:

        """Score current frame for Section C and build GUI-facing summary."""

        display = frame.copy()

        previous = self._last_summary or {}

        prev_queries = previous.get("queries", {})

        summary: Dict[str, Any] = {

            "score": previous.get("score", float("nan")),

            "queries": dict(prev_queries) if isinstance(prev_queries, dict) else {},

            "updated_at": previous.get("updated_at", 0.0),

            "vertical_axis": previous.get("vertical_axis"),

            "horizontal_axis": previous.get("horizontal_axis"),

            "section_result": previous.get("section_result"),

            "just_updated": False,

        }

        # Spread out YOLO inference so the overhead stream stays responsive.
        # Keep a lightweight detection cadence on the overhead feed as well.
        self._frame_index += 1
        run_detection = (
            evaluate
            or self._last_mouse_bbox is None
            or self._frame_index % self.detection_stride == 0
        )
        if run_detection:
            detection = self.detector.predict(frame)
            self._last_mouse_bbox = ObjectDetector.pick_mouse_bbox(detection)
            hand_prediction = self.hand_detector.predict(frame)
            self._last_hand_bboxes = ObjectDetector.collect_hand_bboxes(hand_prediction)

        if keypoints is None:

            display = put_text_lines(display, ["No pose detected"], color=(0, 0, 255))

            return PipelineResult(display, summary)



        if self._draw_skeleton:
            display = draw_skeleton(display, keypoints, color=(0, 255, 120))

        skeleton = Skeleton2D.from_array(keypoints)

        surface_metrics = assess_mouse_keyboard_surfaces(skeleton, SECTIONC_HAND)

        surface_flag = int(surface_metrics.get("mouse_keyboard_surface_flag", 0))

        queries_dict = summary.setdefault("queries", {})

        queries_dict["mouse_keyboard_on_different_surfaces"] = surface_flag

        metrics = summary.get("metrics", {})
        metrics.update(surface_metrics)
        summary["metrics"] = metrics

        just_updated = False

        if evaluate:

            total_seconds = timestamp - self.session_start

            continuous_seconds = timestamp - self.continuous_start

            # Save reference frame for post-hoc review (overhead top view).
            save_snapshot("C", frame, timestamp)

            result = self.scorer.score(

                skeleton,

                self.hand_preference,

                total_seconds,

                continuous_seconds,

                mouse_bbox=self._last_mouse_bbox,

                hand_bboxes=self._last_hand_bboxes,

            )

            self._last_result = result

            self._last_updated_ts = result.timestamp

            self._maybe_export(result.to_row())

            just_updated = True

        result = self._last_result

        mouse_bbox = self._last_mouse_bbox
        if mouse_bbox is not None:
            x1, y1, x2, y2 = map(int, mouse_bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 120, 255), 2)

        for hand_bbox in self._last_hand_bboxes:
            hx1, hy1, hx2, hy2 = map(int, hand_bbox)
            cv2.rectangle(display, (hx1, hy1), (hx2, hy2), (40, 200, 120), 2)

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

            summary.update(

                {

                    "score": result.section_score,

                    "vertical_axis": result.vertical_axis,

                    "horizontal_axis": result.horizontal_axis,

                    "queries": result.query_breakdown,

                    "updated_at": self._last_updated_ts,

                    "section_result": result,

                    "just_updated": just_updated,

                }

            )

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

        self.indicator_panels: Dict[str, IndicatorPanel] = {}
        self.last_update_vars: Dict[str, tk.StringVar] = {}
        self.next_capture_vars: Dict[str, tk.StringVar] = {}

        self.pipelines: Dict[str, BasePipeline] = {}

        self.section_running: Dict[str, bool] = {sec: False for sec in self.section_order}

        self.camera_presets = self._detect_available_cameras()

        self.camera_label_to_index = dict(self.camera_presets)

        self.camera_selection_vars: Dict[str, tk.StringVar] = {}

        self.preview_max_size: Tuple[int, int] = (360, 270)

        placeholder_image = Image.new("RGB", self.preview_max_size, color=(30, 30, 30))

        self.placeholder_photo = ImageTk.PhotoImage(placeholder_image)

        self.latest_results: Dict[str, Optional[Any]] = {sec: None for sec in self.section_order}
        self.manual_query_overrides: Dict[str, Dict[str, int]] = {}
        self.latest_summaries: Dict[str, Optional[Dict[str, Any]]] = {sec: None for sec in self.section_order}
        self.armrest_surface_choice: Optional[str] = None
        self.armrest_prompt_done = False

        self.latest_timestamps: Dict[str, float] = {sec: 0.0 for sec in self.section_order}

        self.monitor_peripheral_scorer = MonitorPeripheralScorer()

        self.rosa_total_scorer = ROSATotalScorer()

        self.last_excel_signature: Optional[Tuple[float, float, float]] = None

        self.glare_client: Optional[GlareSerialClient] = None
        self.glare_status_var = tk.StringVar(value="Glare detector nonaktif.")
        self.glare_detail_var = tk.StringVar(
            value="Set GLARE_SERIAL_PORT di config.py untuk menghubungkan Arduino glare sensor."
        )
        self.glare_last_update_var = tk.StringVar(value="Last update: -")
        if GLARE_SERIAL_PORT:
            self.glare_client = GlareSerialClient(GLARE_SERIAL_PORT, baudrate=GLARE_BAUDRATE)
            self.glare_client.start()
            self.glare_status_var.set(f"Menghubungkan glare detector di {GLARE_SERIAL_PORT}...")
            self.glare_detail_var.set("Menunggu data dari sensor.")



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
                (
                    "Get-CimInstance Win32_PnPEntity "
                    "| Where-Object {$_.PNPClass -eq 'Camera'} "
                    "| Select-Object Name, Manufacturer "
                    "| ConvertTo-Json -Depth 2"
                ),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode != 0 or not result.stdout.strip():
                return []
            payload = result.stdout.strip()
            data = json.loads(payload)
            if isinstance(data, dict):
                records = [data]
            else:
                records = data
            names: List[str] = []
            for entry in records:
                name = (entry or {}).get("Name", "")
                manufacturer = (entry or {}).get("Manufacturer", "")
                label = (name or "").strip()
                if manufacturer:
                    manu = manufacturer.strip()
                    if manu and manu.lower() not in label.lower():
                        label = f"{manu} {label}".strip()
                if label:
                    names.append(label)
            return names
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



        used_indices: Dict[int, str] = {}

        for label, idx in CAMERA_PRESETS:

            if idx is None:

                continue

            if idx in working_indices and idx not in used_indices:

                available.append((label, idx))

                used_indices[idx] = label



        name_iter = iter(names)

        label_counts: Dict[str, int] = {}

        for idx in working_indices:

            if idx in used_indices:

                continue

            raw_name = next(name_iter, None)

            if raw_name:

                count = label_counts.get(raw_name, 0)

                label_counts[raw_name] = count + 1

                display = f"{raw_name} ({count + 1})" if count else raw_name

            else:

                display = f"Camera {idx}"

            available.append((display, idx))

        return available



    def _default_camera_choice(self, section: str) -> str:
        """Pick initial dropdown value using flexible name preferences."""
        preferences = CAMERA_DEFAULTS.get(section, [])
        for pref in preferences:
            pref_norm = (pref or "").strip().lower()
            if not pref_norm:
                continue
            for label, _ in self.camera_presets:
                label_norm = (label or "").strip().lower()
                if pref_norm in label_norm:
                    return label

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



        scroll_holder = ttk.Frame(self.root)
        scroll_holder.grid(row=1, column=0, sticky="nsew")
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        main_canvas = tk.Canvas(scroll_holder, highlightthickness=0)
        main_scroll = ttk.Scrollbar(scroll_holder, orient="vertical", command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=main_scroll.set)
        main_canvas.grid(row=0, column=0, sticky="nsew")
        main_scroll.grid(row=0, column=1, sticky="ns")
        scroll_holder.rowconfigure(0, weight=1)
        scroll_holder.columnconfigure(0, weight=1)
        container = ttk.Frame(main_canvas, padding=8)
        main_canvas.create_window((0, 0), window=container, anchor="nw")
        container.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")),
        )
        self._main_scroll_canvas = main_canvas



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



            last_var = tk.StringVar(value="Last update: -")
            next_var = tk.StringVar(value="Next capture: -")
            ttk.Label(frame, textvariable=last_var, font=("Segoe UI", 8, "italic")).pack(anchor="w", padx=4)
            ttk.Label(frame, textvariable=next_var, font=("Segoe UI", 8, "italic")).pack(anchor="w", padx=4, pady=(0, 4))
            self.last_update_vars[section] = last_var
            self.next_capture_vars[section] = next_var

            panel = IndicatorPanel(frame)
            panel.pack(fill="both", expand=True, padx=4, pady=(2, 4))
            self.indicator_panels[section] = panel
            panel.set_rows(self._build_indicator_rows(section, None))



            btn = ttk.Button(frame, text="Start", command=lambda s=section: self.toggle_section(s))

            btn.pack(pady=(0, 6))

            self.toggle_buttons[section] = btn

        glare_frame = ttk.LabelFrame(
            container,
            text="Glare Detector",
            padding=8,
        )
        glare_frame.grid(
            row=1,
            column=0,
            columnspan=len(self.section_order),
            sticky="ew",
            pady=(12, 0),
        )
        container.rowconfigure(1, weight=0)
        ttk.Label(
            glare_frame,
            textvariable=self.glare_status_var,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            glare_frame,
            textvariable=self.glare_last_update_var,
            font=("Segoe UI", 9, "italic"),
        ).pack(anchor="w", pady=(0, 2))
        ttk.Label(
            glare_frame,
            textvariable=self.glare_detail_var,
            font=("Segoe UI", 10),
            wraplength=620,
            justify="left",
        ).pack(anchor="w")



    def _build_indicator_rows(self, section: str, summary: Optional[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """Convert query scores into badge rows for the indicator panel."""
        spec = self.section_specs[section]
        rows: List[Tuple[str, str, str]] = []
        query_scores: Dict[str, int] = {}
        data_ready = False
        updated_at: Optional[float] = None
        next_in: Optional[float] = None

        if summary:
            queries = summary.get("queries")
            if isinstance(queries, dict):
                query_scores = {str(k): int(v) for k, v in queries.items()}
            updated_at = summary.get("updated_at")
            next_in = summary.get("next_update_in")
            if isinstance(updated_at, (int, float)) and updated_at > 0:
                data_ready = True
            elif any(query_scores.values()):
                data_ready = True

        overrides = self.manual_query_overrides.get(section)
        if overrides:
            query_scores.update({str(k): int(v) for k, v in overrides.items()})

        row_specs = QUERY_DISPLAY.get(spec["query_key"], [])
        if not row_specs:
            return [("0 of 0", "No indicators configured", "unknown")]

        for key, label, weight in row_specs:
            if (
                section == "C"
                and key == "documents_used_no_document_holder"
                and key not in query_scores
            ):
                front_summary = self.latest_summaries.get("B")
                if front_summary:
                    front_queries = front_summary.get("queries", {})
                    if isinstance(front_queries, dict) and key in front_queries:
                        query_scores[key] = int(front_queries.get(key, 0))
            if not data_ready and weight is not None:
                status_text = "null"
                value = 0
            else:
                value = int(query_scores.get(key, 0))
                if weight is None:
                    status_text = "info"
                else:
                    status_text = f"{value} of {weight}"
            tag = self._indicator_tag_for_values(value, weight, data_ready)
            rows.append((status_text, label, tag))

        # Update timestamp labels alongside the badge rows.
        last_var = self.last_update_vars.get(section)
        next_var = self.next_capture_vars.get(section)
        if last_var is not None:
            if isinstance(updated_at, (int, float)) and updated_at > 0:
                last_var.set(f"Last update: {time.strftime('%H:%M:%S', time.localtime(updated_at))}")
            else:
                last_var.set("Last update: -")
        if next_var is not None:
            if isinstance(next_in, (int, float)):
                next_var.set(f"Next capture: {max(0, int(round(next_in)))}s")
            else:
                next_var.set("Next capture: -")

        return rows

    def _apply_manual_overrides_to_summary(
        self,
        section: str,
        summary: Dict[str, Any],
        section_result: Optional[Any],
    ) -> None:
        """Ensure manual questionnaire answers propagate to summaries/results."""
        overrides = self.manual_query_overrides.get(section)
        if not overrides:
            return
        queries = summary.setdefault("queries", {})
        for key, value in overrides.items():
            queries[key] = int(value)
        if section_result is not None and hasattr(section_result, "query_breakdown"):
            section_result.query_breakdown.update({k: int(v) for k, v in overrides.items()})

    def _update_indicator_panel(self, section: str, summary: Optional[Dict[str, Any]]) -> None:
        """Refresh a section's indicator panel with the latest badge colors."""
        panel = self.indicator_panels.get(section)
        if panel is None:
            return
        panel.set_rows(self._build_indicator_rows(section, summary))

    @staticmethod
    def _indicator_tag_for_values(value: int, weight: Optional[int], data_ready: bool) -> str:
        """Return badge color tag (ok/alert/unknown) for an indicator row."""
        if not data_ready:
            return "unknown"
        if weight is None or weight <= 0:
            return "unknown"
        return "alert" if value > 0 else "ok"

    def _set_manual_override(self, section: str, key: str, value: int) -> None:
        """Record manual responses and refresh cached summaries."""
        overrides = self.manual_query_overrides.setdefault(section, {})
        overrides[key] = int(value)
        cached = self.latest_summaries.get(section)
        if cached is not None:
            patched = copy.deepcopy(cached)
            patched.setdefault("queries", {})[key] = int(value)
            self.latest_summaries[section] = patched
            self._update_indicator_panel(section, patched)
        result_obj = self.latest_results.get(section)
        if result_obj is not None and hasattr(result_obj, "query_breakdown"):
            result_obj.query_breakdown[key] = int(value)


    def _refresh_glare_status(self) -> None:
        """Update glare detector status box."""
        if self.glare_client is None:
            self.glare_status_var.set("Glare detector nonaktif (GLARE_SERIAL_PORT=None).")
            self.glare_detail_var.set(
                "Atur GLARE_SERIAL_PORT di config.py bila ingin membaca sensor glare berbasis Arduino."
            )
            self.glare_last_update_var.set("Last update: -")
            return

        snapshot = self.glare_client.snapshot()
        connected = bool(snapshot.get("connected"))
        port = GLARE_SERIAL_PORT or "-"
        status = "Connected" if connected else "Disconnected"
        self.glare_status_var.set(f"Glare detector: {status} ({port})")

        updated_at = snapshot.get("updated_at")
        if isinstance(updated_at, (int, float)) and updated_at > 0:
            ts = time.strftime("%H:%M:%S", time.localtime(updated_at))
            self.glare_last_update_var.set(f"Last update: {ts}")
        else:
            self.glare_last_update_var.set("Last update: -")

        glare_flag = snapshot.get("glare")
        ratio = snapshot.get("ratio")
        lux_screen = snapshot.get("lux_screen")
        lux_room = snapshot.get("lux_room")
        msg_parts = [str(snapshot.get("message", ""))]
        if glare_flag is True:
            msg_parts.append("Status: GLARE terdeteksi.")
        elif glare_flag is False:
            msg_parts.append("Status: aman (tidak glare).")
        else:
            msg_parts.append("Menunggu pembacaan pertama...")
        if isinstance(ratio, (int, float)):
            msg_parts.append(f"Rasio layar/ambient {ratio:.2f}")
        if isinstance(lux_screen, (int, float)) and isinstance(lux_room, (int, float)):
            msg_parts.append(f"Lux layar {lux_screen:.0f} | ambient {lux_room:.0f}")
        self.glare_detail_var.set(" ".join(part for part in msg_parts if part))



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

        if section == "A":
            self.manual_query_overrides.pop("A", None)
            self.armrest_surface_choice = None
            self.armrest_prompt_done = False

        cam_index = self._resolve_camera_index(section)

        if cam_index is None:

            friendly = self.section_specs[section]["label"]

            messagebox.showinfo(

                "Camera disabled",

                f"Panel {friendly} disetel ke 'None'. Pilih kamera sebelum menyalakan.",

            )

            return

        self.latest_results[section] = None

        self.latest_timestamps[section] = 0.0

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

        self._update_indicator_panel(section, None)

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

        self._update_indicator_panel(section, None)

        self.photo_refs[section] = self.placeholder_photo

        self.video_labels[section].configure(image=self.placeholder_photo, text="Camera off", compound="center")

        self.latest_results[section] = None

        self.latest_timestamps[section] = 0.0

        self.latest_summaries[section] = None

        self._update_status()



    def _update_status(self) -> None:

        """Refresh status bar text with currently running sections."""

        running = [self.section_specs[sec]["label"] for sec in self.section_order if self.section_running.get(sec)]

        if running:

            self.status_var.set("Running: " + ", ".join(running))

        else:

            self.status_var.set("Ready. Toggle cameras to begin.")



    def _maybe_export_excel(self) -> None:

        """Append a combined ROSA snapshot to the Excel workbook when data refreshes."""

        if not EXPORT_XLSX:

            return

        required = ("A", "B", "C")

        if any(self.latest_results.get(sec) is None for sec in required):

            return

        timestamps = [self.latest_timestamps.get(sec, 0.0) for sec in required]

        if any(ts <= 0.0 for ts in timestamps):

            return

        if max(timestamps) - min(timestamps) > 2.0:

            return

        section_a = self.latest_results["A"]

        section_b = self.latest_results["B"]

        section_c = self.latest_results["C"]

        signature = (

            getattr(section_a, "timestamp", 0.0),

            getattr(section_b, "timestamp", 0.0),

            getattr(section_c, "timestamp", 0.0),

        )

        if self.last_excel_signature == signature:

            return

        try:

            monitor_peripherals = self.monitor_peripheral_scorer.score(section_b, section_c)

            rosa_total = self.rosa_total_scorer.score(section_a, monitor_peripherals)

            row = build_excel_row(section_a, section_b, section_c, monitor_peripherals, rosa_total)

            export_excel_row(EXPORT_XLSX, row, EXCEL_HEADERS)

        except Exception as exc:

            messagebox.showwarning("Export warning", f"Gagal menyimpan Excel: {exc}")

            return

        self.last_excel_signature = signature



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

        if self.section_running.get("A"):
            self._prompt_armrest_surface()

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

            section_result = result.summary.get("section_result")
            self._apply_manual_overrides_to_summary(section, result.summary, section_result)
            self._update_indicator_panel(section, result.summary)
            self.latest_summaries[section] = copy.deepcopy(result.summary)

            if result.summary.get("just_updated"):

                if section_result is not None:

                    self.latest_results[section] = section_result

                    updated_at = result.summary.get("updated_at", time.time())

                    if isinstance(updated_at, (int, float)):

                        self.latest_timestamps[section] = float(updated_at)

                    else:

                        self.latest_timestamps[section] = time.time()

                    self._maybe_export_excel()

        self._refresh_glare_status()
        self.root.after(33, self._update_loop)



    def on_close(self) -> None:

        """Handle window close: prompt for final inputs, then stop all pipelines."""
        dialog = PostSessionDialog(self.root)
        results = getattr(dialog, "results", None)
        if results:
            # Section A overrides
            self._set_manual_override("A", "hard_or_damaged_surface", results.get("hard_or_damaged_surface", 0))
            self._set_manual_override("A", "seat_height_non_adjustable", results.get("seat_height_non_adjustable", 0))
            self._set_manual_override("A", "seat_depth_non_adjustable", results.get("seat_depth_non_adjustable", 0))
            self._set_manual_override("A", "armrest_non_adjustable", results.get("armrest_non_adjustable", 0))
            self._set_manual_override("A", "back_support_non_adjustable", results.get("back_support_non_adjustable", 0))
            
            # Section C override
            self._set_manual_override("C", "keyboard_platform_non_adjustable", results.get("keyboard_platform_non_adjustable", 0))

        self.stop()
        if self.glare_client is not None:
            self.glare_client.stop()

        self.root.destroy()



def main(multi: bool = True) -> None:

    """Launch Tkinter ROSA application in multi or single-section mode."""

    root = tk.Tk()
    try:
        root.state("zoomed")
    except Exception:
        root.geometry("1280x720")

    if multi:

        MultiSectionTkApp(root)

    else:

        ROSATkApp(root)

    root.mainloop()





if __name__ == "__main__":

    main(multi=True)
