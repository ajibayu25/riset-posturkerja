"""Front-camera sensory heuristics for ROSA Section B (monitor & phone)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from constants.thresholds import SECTION_B_ADJUSTMENTS, SECTION_B_THRESHOLDS
from core.geometry import Skeleton2D, distance
from . import ComponentOutput

BBox = Tuple[int, int, int, int]


def monitor_components(skeleton: Skeleton2D, bbox: Optional[BBox]) -> ComponentOutput:
    """Derive monitor posture indicators from skeleton and detection."""
    cfg = SECTION_B_THRESHOLDS["monitor"]
    queries: Dict[str, int] = {
        "elbows_supported": 0,
        "shoulder_height_issue": 0,
        "too_wide": 0,
        "work_surface_high": 0,
        "too_far_reach": 0,
        "mouse_inline_shoulder": 0,
        "reaching_to_mouse": 0,
        "mouse_keyboard_diff_surfaces": 0,
        "pinch_grip_mouse": 0,
        "keyboard_too_high": 0,
    }
    metrics: Dict[str, float] = {}
    adjustments: Dict[str, int] = {}
    base = 0

    shoulder_mid = skeleton.shoulder_mid()
    nose = skeleton.point("nose")
    if shoulder_mid is not None and nose is not None:
        neck_vec = nose - shoulder_mid
        metrics["neck_vertical"] = float(neck_vec[1])
        if neck_vec[1] > cfg["vertical_angle_deg"]["too_low_max"]:
            base = max(base, 1)
            adjustments["too_low"] = SECTION_B_ADJUSTMENTS["monitor"].get("too_low", 1)
        if neck_vec[1] < -cfg["vertical_angle_deg"]["too_high_min"]:
            base = max(base, 2)
            adjustments["too_high"] = SECTION_B_ADJUSTMENTS["monitor"].get("too_high", 1)

    if bbox is not None and shoulder_mid is not None:
        x1, y1, x2, y2 = bbox
        width = max(1.0, float(x2 - x1))
        metrics["monitor_width"] = width
        shoulder_width = skeleton.shoulder_width()
        if not np.isnan(shoulder_width) and width > 1e-6:
            ratio = shoulder_width / width
            metrics["distance_ratio"] = ratio
            if ratio > cfg["distance_cm"]["too_far_min"] / 10.0:
                adjustments["too_far"] = SECTION_B_ADJUSTMENTS["monitor"].get("too_far", 1)
                queries["too_far_reach"] = 2

    return ComponentOutput(
        base=base,
        adjustments=adjustments,
        metrics=metrics,
        queries=queries,
    )


def phone_components(
    skeleton: Skeleton2D,
    phone_bbox: Optional[BBox],
    audio_devices: Iterable[Tuple[str, float, BBox]],
    frame_shape: Tuple[int, int, int],
) -> ComponentOutput:
    """Derive telephone/earphone posture indicators."""
    cfg = SECTION_B_THRESHOLDS["telephone"]
    queries: Dict[str, int] = {
        "headset_or_phone_posture": 0,
        "too_far_reach": 0,
        "neck_shoulder_hold": 0,
        "no_hands_free": 0,
    }
    metrics: Dict[str, float] = {}
    adjustments: Dict[str, int] = {}
    base = 0

    sidebend = skeleton.neck_sidebend()
    metrics["neck_sidebend"] = sidebend
    if not np.isnan(sidebend) and abs(sidebend) > cfg["neck_sidebend_deg"]:
        adjustments["neck_shoulder_hold"] = SECTION_B_ADJUSTMENTS["telephone"].get("neck_shoulder_hold", 2)
        queries["neck_shoulder_hold"] = 2

    neutral_neck = not np.isnan(sidebend) and abs(sidebend) <= cfg["neck_sidebend_deg"]
    ear_points = [skeleton.point("left_ear"), skeleton.point("right_ear"), skeleton.point("nose")]

    if phone_bbox is not None:
        x1, y1, x2, y2 = phone_bbox
        phone_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
        metrics["phone_center_x"] = phone_center[0]
        metrics["phone_center_y"] = phone_center[1]
        ref_point = skeleton.point("right_shoulder")
        if ref_point is None:
            ref_point = skeleton.point("nose")
        if ref_point is not None:
            reach = distance(phone_center, ref_point)
            metrics["reach_pixels"] = reach
            height, width = frame_shape[0], frame_shape[1]
            diag = (width**2 + height**2) ** 0.5
            if reach > 0.3 * diag:
                base = max(base, 2)
                adjustments["outside_reach"] = SECTION_B_ADJUSTMENTS["telephone"].get("outside_reach", 2)
                queries["too_far_reach"] = 2

    diag = (frame_shape[0] ** 2 + frame_shape[1] ** 2) ** 0.5
    head_threshold = 0.18 * diag
    for label, conf, bbox in audio_devices:
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
        distances = [distance(center, pt) for pt in ear_points if pt is not None]
        if distances and min(distances) <= head_threshold and neutral_neck:
            queries["headset_or_phone_posture"] = 1
            break

    return ComponentOutput(
        base=base,
        adjustments=adjustments,
        metrics=metrics,
        queries=queries,
    )


__all__ = ["monitor_components", "phone_components"]
