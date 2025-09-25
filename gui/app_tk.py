"""Tkinter GUI for ROSA live scoring (Sections A, B, C)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from config import CAMERA_INDEX, DET_MODEL, DEVICE, EXPORT_CSV, EXPORT_JSONL, POSE_MODEL, SECTIONC_HAND
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

BBox = Tuple[int, int, int, int]
COCO_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    vis = frame.copy()
    for x, y in keypoints:
        cv2.circle(vis, (int(x), int(y)), 4, color, -1)
    for a, b in COCO_EDGES:
        if a < len(keypoints) and b < len(keypoints):
            pa, pb = keypoints[a], keypoints[b]
            cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 200, 255), 2)
    return vis


def put_text_lines(frame: np.ndarray, lines: List[str], origin: Tuple[int, int] = (16, 32), color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
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
    frame: np.ndarray
    summary: Dict[str, float]


class BasePipeline:
    def __init__(self, cam_index: int, export_mode: str = "csv", smoothing_alpha: float = 0.3) -> None:
        self.cam_index = cam_index
        self.export_mode = export_mode
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

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def reset_continuous(self) -> None:
        self.continuous_start = time.time()

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

    def _maybe_export(self, row: Dict[str, float]) -> None:
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
        ok, frame = self.cap.read()
        if not ok:
            return None
        ts = time.time()
        keypoints = self.pose.predict_xy(frame)
        if keypoints is not None:
            keypoints = self.smoother.update(keypoints[:, :2], timestamp=ts)
        return self.process_frame(frame, keypoints, ts)

    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float) -> PipelineResult:
        raise NotImplementedError


class SectionAPipeline(BasePipeline):
    def __init__(self, cam_index: int, export_mode: str = "csv", smoothing_alpha: float = 0.3) -> None:
        super().__init__(cam_index, export_mode, smoothing_alpha)
        self.scorer = SectionAScorer()

    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float) -> PipelineResult:
        display = frame.copy()
        summary: Dict[str, float] = {"score": float("nan")}
        if keypoints is None:
            display = put_text_lines(display, ["No pose detected"], color=(0, 0, 255))
            return PipelineResult(display, summary)

        display = draw_skeleton(display, keypoints)
        skeleton = Skeleton2D.from_array(keypoints)
        total_seconds = timestamp - self.session_start
        continuous_seconds = timestamp - self.continuous_start
        result = self.scorer.score(skeleton, total_seconds, continuous_seconds)

        lines = [
            f"Section A score {result.chair_score_final} (base {result.chair_score_base}, dur {result.duration_adjustment:+d})",
            f"Vertical axis: {result.vertical_axis} | Horizontal axis: {result.horizontal_axis}",
            f"Seat height {result.seat_height.total} | Seat depth {result.seat_depth.total}",
            f"Armrest {result.armrest.total} | Back support {result.back_support.total}",
        ]
        risk = "OK" if result.chair_score_final < 5 else "High"
        lines.append(f"Risk: {risk}")
        display = put_text_lines(display, lines)

        summary = {
            "score": result.chair_score_final,
            "vertical_axis": result.vertical_axis,
            "horizontal_axis": result.horizontal_axis,
        }
        self._maybe_export(result.to_row())
        return PipelineResult(display, summary)


class SectionBPipeline(BasePipeline):
    def __init__(
        self,
        cam_index: int,
        export_mode: str = "csv",
        smoothing_alpha: float = 0.3,
        detection_stride: int = 5,
    ) -> None:
        super().__init__(cam_index, export_mode, smoothing_alpha)
        self.detector = ObjectDetector(model_path=DET_MODEL, device=DEVICE)
        self.scorer = SectionBScorer()
        self.detection_stride = max(1, detection_stride)
        self.frame_count = 0
        self.last_monitor_bbox: Optional[BBox] = None
        self.last_phone_bbox: Optional[BBox] = None

    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float) -> PipelineResult:
        display = frame.copy()
        summary: Dict[str, float] = {"score": float("nan")}
        self.frame_count += 1
        if self.frame_count % self.detection_stride == 0:
            detections = self.detector.predict(frame)
            self.last_monitor_bbox = ObjectDetector.pick_monitor_bbox(detections)
            self.last_phone_bbox = ObjectDetector.pick_phone_bbox(detections)

        if keypoints is None:
            if self.last_monitor_bbox is not None:
                x1, y1, x2, y2 = self.last_monitor_bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)
            if self.last_phone_bbox is not None:
                x1, y1, x2, y2 = self.last_phone_bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 120, 0), 2)
            display = put_text_lines(display, ["No pose detected"], color=(0, 0, 255))
            return PipelineResult(display, summary)

        display = draw_skeleton(display, keypoints)
        skeleton = Skeleton2D.from_array(keypoints)
        total_seconds = timestamp - self.session_start
        continuous_seconds = timestamp - self.continuous_start
        result = self.scorer.score(
            skeleton,
            self.last_monitor_bbox,
            self.last_phone_bbox,
            frame.shape,
            total_seconds,
            continuous_seconds,
        )

        if self.last_monitor_bbox is not None:
            x1, y1, x2, y2 = self.last_monitor_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)
        if self.last_phone_bbox is not None:
            x1, y1, x2, y2 = self.last_phone_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 120, 0), 2)

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
        }
        self._maybe_export(result.to_row())
        return PipelineResult(display, summary)


class SectionCPipeline(BasePipeline):
    def __init__(
        self,
        cam_index: int,
        export_mode: str = "csv",
        smoothing_alpha: float = 0.3,
        hand_preference: str = "right",
    ) -> None:
        super().__init__(cam_index, export_mode, smoothing_alpha)
        self.scorer = SectionCScorer()
        self.hand_preference = hand_preference.lower()

    def process_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray], timestamp: float) -> PipelineResult:
        display = frame.copy()
        summary: Dict[str, float] = {"score": float("nan")}
        if keypoints is None:
            display = put_text_lines(display, ["No pose detected"], color=(0, 0, 255))
            return PipelineResult(display, summary)

        display = draw_skeleton(display, keypoints, color=(0, 255, 120))
        skeleton = Skeleton2D.from_array(keypoints)
        total_seconds = timestamp - self.session_start
        continuous_seconds = timestamp - self.continuous_start
        result = self.scorer.score(
            skeleton,
            self.hand_preference,
            total_seconds,
            continuous_seconds,
        )

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
        }
        self._maybe_export(result.to_row())
        return PipelineResult(display, summary)


PIPELINE_FACTORIES: Dict[str, Callable[..., BasePipeline]] = {
    "A": SectionAPipeline,
    "B": SectionBPipeline,
    "C": SectionCPipeline,
}


class ROSATkApp:
    def __init__(self, root: tk.Tk) -> None:
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
        self.running = False
        if self.pipeline:
            self.pipeline.release()
            self.pipeline = None
        self.status_var.set("Stopped")
        self.score_var.set("Score: -")

    def mark_break(self) -> None:
        if self.pipeline:
            self.pipeline.reset_continuous()
            self.status_var.set("Break marked - timer reset")

    def _update_loop(self) -> None:
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
        self.stop()
        self.root.destroy()


class MultiSectionTkApp:
    """Display and log all ROSA sections simultaneously (three cameras)."""

    def __init__(self, root: tk.Tk, export_mode: str = "csv") -> None:
        self.root = root
        self.export_mode = export_mode
        self.section_order = ["A", "B", "C"]
        self.status_var = tk.StringVar(value="Ready. Toggle cameras to begin.")
        self.photo_refs: Dict[str, Optional[ImageTk.PhotoImage]] = {sec: None for sec in self.section_order}
        self.score_vars: Dict[str, tk.StringVar] = {}
        self.video_labels: Dict[str, ttk.Label] = {}
        self.toggle_buttons: Dict[str, ttk.Button] = {}
        self.pipelines: Dict[str, BasePipeline] = {}
        self.section_running: Dict[str, bool] = {sec: False for sec in self.section_order}

        self._build_ui()
        self._update_status()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._update_loop)

    def _build_ui(self) -> None:
        control = ttk.Frame(self.root, padding=8)
        control.grid(row=0, column=0, sticky="ew")
        control.columnconfigure(3, weight=1)

        ttk.Button(control, text="Stop All", command=self.stop).grid(row=0, column=0, padx=4)
        ttk.Button(control, text="Break", command=self.mark_break).grid(row=0, column=1, padx=4)
        ttk.Label(control, textvariable=self.status_var, foreground="blue").grid(row=0, column=2, padx=8, sticky="w")

        container = ttk.Frame(self.root, padding=8)
        container.grid(row=1, column=0, sticky="nsew")
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        for idx, section in enumerate(self.section_order):
            frame = ttk.LabelFrame(container, text=f"Section {section}", padding=4)
            frame.grid(row=0, column=idx, padx=6, pady=4, sticky="nsew")
            container.columnconfigure(idx, weight=1)

            video = ttk.Label(frame, text="Camera off", anchor="center")
            video.pack(fill="both", expand=True)
            self.video_labels[section] = video

            score_var = tk.StringVar(value="Score: -")
            self.score_vars[section] = score_var
            ttk.Label(frame, textvariable=score_var, font=("Segoe UI", 11, "bold")).pack(pady=(6, 2))

            btn = ttk.Button(frame, text="Start", command=lambda s=section: self.toggle_section(s))
            btn.pack(pady=(0, 4))
            self.toggle_buttons[section] = btn

    def _create_pipeline(self, section: str) -> BasePipeline:
        cam_index = CAMERA_INDEX.get(section)
        if cam_index is None:
            raise RuntimeError(f"Camera index untuk section {section} belum diatur di config")
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
        if self.section_running.get(section):
            self._stop_section(section)
        else:
            self._start_section(section)

    def _start_section(self, section: str) -> None:
        try:
            pipeline = self._create_pipeline(section)
        except Exception as exc:
            messagebox.showerror("Camera error", str(exc))
            return
        self.pipelines[section] = pipeline
        self.section_running[section] = True
        self.toggle_buttons[section].configure(text="Stop")
        self.score_vars[section].set("Initializing...")
        self.video_labels[section].configure(text="Connecting...", image="")
        self.photo_refs[section] = None
        self._update_status()

    def _stop_section(self, section: str) -> None:
        pipeline = self.pipelines.pop(section, None)
        if pipeline is not None:
            pipeline.release()
        self.section_running[section] = False
        self.toggle_buttons[section].configure(text="Start")
        self.score_vars[section].set("Score: -")
        self.photo_refs[section] = None
        self.video_labels[section].configure(image="", text="Camera off")
        self._update_status()

    def _update_status(self) -> None:
        running = [sec for sec in self.section_order if self.section_running.get(sec)]
        if running:
            self.status_var.set("Running: " + ", ".join(running))
        else:
            self.status_var.set("Ready. Toggle cameras to begin.")

    def mark_break(self) -> None:
        for pipeline in self.pipelines.values():
            pipeline.reset_continuous()
        self.status_var.set("Break recorded")

    def stop(self) -> None:
        for section in list(self.section_order):
            if self.section_running.get(section):
                self._stop_section(section)

    def _update_loop(self) -> None:
        for section, pipeline in list(self.pipelines.items()):
            result = pipeline.step()
            if result is None:
                messagebox.showwarning("Camera warning", f"Stream section {section} terputus. Kamera dimatikan.")
                self._stop_section(section)
                continue
            frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            self.photo_refs[section] = photo
            self.video_labels[section].configure(image=photo, text="")

            score = result.summary.get("score", float("nan"))
            if np.isnan(score):
                text = "Score: -"
            else:
                if section == "A":
                    text = (
                        f"Score: {score:.0f} | Vertical {result.summary.get('vertical_axis', 0):.0f} "
                        f"| Horizontal {result.summary.get('horizontal_axis', 0):.0f}"
                    )
                elif section == "B":
                    text = (
                        f"Score: {score:.0f} | Monitor {result.summary.get('horizontal_axis', 0):.0f} "
                        f"| Phone {result.summary.get('vertical_axis', 0):.0f}"
                    )
                else:
                    text = (
                        f"Score: {score:.0f} | Mouse {result.summary.get('vertical_axis', 0):.0f} "
                        f"| Keyboard {result.summary.get('horizontal_axis', 0):.0f}"
                    )
            self.score_vars[section].set(text)
        self.root.after(33, self._update_loop)

    def on_close(self) -> None:
        self.stop()
        self.root.destroy()

def main(multi: bool = True) -> None:
    root = tk.Tk()
    if multi:
        MultiSectionTkApp(root)
    else:
        ROSATkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main(multi=True)

