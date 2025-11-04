"""Front-camera sensory heuristics for ROSA Section B (monitor & phone)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from constants.thresholds import SECTION_B_ADJUSTMENTS, SECTION_B_THRESHOLDS
from core.geometry import Skeleton2D, distance
from . import ComponentOutput

BBox = Tuple[int, int, int, int]


def monitor_components(skeleton: Skeleton2D, bbox: Optional[BBox]) -> ComponentOutput:
    """Derive monitor posture indicators from skeleton and detection.

    Measurements follow ROSA Section B guidance combined with CSA Z412/ISO 9241 ergonomic
    targets.  Shoulder–elbow gaps are compared against the thresholds stored in
    ``constants.thresholds``.  All geometric checks are normalised by an estimated torso
    length so that the heuristics remain scale-agnostic when only pixel data are available.
    """
    cfg = SECTION_B_THRESHOLDS["monitor"]
    queries: Dict[str, int] = {
        "elbows_supported_in_line_with_shoulder_shoulders_relaxed": 0,
        "too_high_shoulders_shrugged_or_low_arms_unsupported": 0,
        "too_wide": 0,
        "work_surface_too_high": 0,
        "too_far_of_reach_outside_30_cm": 0,
        "mouse_in_line_with_shoulder": 0,
        "reaching_to_mouse": 0,
        "mouse_keyboard_on_different_surfaces": 0,
        "pinch_grip_on_mouse": 0,
        "keyboard_too_high_shoulders_shrugged": 0,
    }
    metrics: Dict[str, float] = {}
    adjustments: Dict[str, int] = {}
    base = 0

    shoulder_mid = skeleton.shoulder_mid()
    hip_mid = skeleton.hip_mid()
    torso_len = float(distance(shoulder_mid, hip_mid)) if shoulder_mid is not None and hip_mid is not None else float("nan")
    if np.isnan(torso_len) or torso_len < 1e-3:
        torso_len = skeleton.shoulder_width()
    if np.isnan(torso_len) or torso_len < 1e-3:
        torso_len = 100.0

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

    left_shoulder = skeleton.point("left_shoulder")
    right_shoulder = skeleton.point("right_shoulder")
    left_elbow = skeleton.point("left_elbow")
    right_elbow = skeleton.point("right_elbow")
    left_wrist = skeleton.point("left_wrist")
    right_wrist = skeleton.point("right_wrist")

    # Compare elbow height against shoulders to flag relaxed vs shrugged shoulders.
    offsets = []
    for shoulder, elbow in ((left_shoulder, left_elbow), (right_shoulder, right_elbow)):
        if shoulder is None or elbow is None:
            continue
        offsets.append(float(elbow[1] - shoulder[1]))
    if offsets:
        avg_offset = float(np.mean(offsets))
        spread = float(np.std(offsets))
        metrics["elbow_shoulder_offset_avg"] = avg_offset
        metrics["elbow_shoulder_offset_std"] = spread
        if all(0.05 * torso_len <= off <= 0.40 * torso_len for off in offsets):
            queries["elbows_supported_in_line_with_shoulder_shoulders_relaxed"] = 1
        if any(off < -0.05 * torso_len for off in offsets) or abs((left_shoulder[1] if left_shoulder is not None else 0.0) - (right_shoulder[1] if right_shoulder is not None else 0.0)) > 0.15 * torso_len:
            queries["too_high_shoulders_shrugged_or_low_arms_unsupported"] = 2

    shoulder_width = skeleton.shoulder_width()
    wrist_span = float(np.abs(left_wrist[0] - right_wrist[0])) if left_wrist is not None and right_wrist is not None else float("nan")
    if not np.isnan(shoulder_width) and not np.isnan(wrist_span):
        metrics["wrist_span"] = wrist_span
        metrics["shoulder_width"] = shoulder_width
        if wrist_span > 1.25 * shoulder_width:
            queries["too_wide"] = 1

    # Compare wrist elevation to elbow height; large positive differences imply arms
    # are lifted to reach a high work surface (ROSA Section B monitor axis).
    if left_wrist is not None and right_wrist is not None and left_elbow is not None and right_elbow is not None:
        wrist_heights = [float(elbow[1] - wrist[1]) for wrist, elbow in ((left_wrist, left_elbow), (right_wrist, right_elbow))]
        metrics["wrist_above_elbow_avg"] = float(np.mean(wrist_heights))
        if all(diff > 0.12 * torso_len for diff in wrist_heights):
            queries["work_surface_too_high"] = 1
        if all(diff > 0.18 * torso_len for diff in wrist_heights):
            queries["keyboard_too_high_shoulders_shrugged"] = 1
        wrist_height_delta = abs(wrist_heights[0] - wrist_heights[1])
        if wrist_height_delta > 0.12 * torso_len:
            queries["mouse_keyboard_on_different_surfaces"] = 2

    # Horizontal deviation of the dominant wrist from the shoulder indicates
    # whether the mouse stays in-line or requires reaching.
    if right_wrist is not None and right_shoulder is not None and not np.isnan(shoulder_width):
        horiz = float(right_wrist[0] - right_shoulder[0])
        metrics["mouse_horizontal_offset"] = horiz
        if abs(horiz) <= 0.20 * shoulder_width:
            queries["mouse_in_line_with_shoulder"] = 1
        if abs(horiz) > 0.45 * shoulder_width:
            queries["reaching_to_mouse"] = 2

    if left_wrist is not None and right_wrist is not None and left_elbow is not None and right_elbow is not None:
        elbow_span = float(np.abs(left_elbow[0] - right_elbow[0]))
        metrics["elbow_span"] = elbow_span
        # A small difference between elbow span and wrist span indicates the
        # user is keeping the mouse close to the body's midline and pinching the device.
        if abs(wrist_span - elbow_span) < 0.10 * shoulder_width:
            queries["pinch_grip_on_mouse"] = 1

    if bbox is not None and shoulder_mid is not None:
        x1, y1, x2, y2 = bbox
        width = max(1.0, float(x2 - x1))
        metrics["monitor_width"] = width
        if not np.isnan(shoulder_width) and width > 1e-6:
            ratio = shoulder_width / width
            metrics["distance_ratio"] = ratio
            if ratio > cfg["distance_cm"]["too_far_min"] / 10.0:
                adjustments["too_far"] = SECTION_B_ADJUSTMENTS["monitor"].get("too_far", 1)
                queries["too_far_of_reach_outside_30_cm"] = 2

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
    """Derive telephone/earphone posture indicators.

    Logic mirrors ROSA Section B telephone checklist: we observe lateral neck
    flexion for shoulder holds, reaching distance for handset placement, and the
    presence of hands-free devices near the ear region.
    """
    cfg = SECTION_B_THRESHOLDS["telephone"]
    cfg_front = SECTION_B_THRESHOLDS["front"]
    queries: Dict[str, int] = {
        "headset_or_one_hand_on_phone_neutral_neck_posture": 0,
        "too_far_of_reach_outside_30_cm": 0,
        "neck_and_shoulder_hold": 0,
        "no_hands_free_options": 0,
    }
    metrics: Dict[str, float] = {}
    adjustments: Dict[str, int] = {}
    base = 0

    sidebend = skeleton.neck_sidebend()
    metrics["neck_sidebend"] = sidebend
    if not np.isnan(sidebend) and abs(sidebend) > cfg["neck_sidebend_deg"]:
        adjustments["neck_and_shoulder_hold"] = SECTION_B_ADJUSTMENTS["telephone"].get("neck_and_shoulder_hold", 2)
        queries["neck_and_shoulder_hold"] = 2

    neutral_neck = not np.isnan(sidebend) and abs(sidebend) <= cfg["neck_sidebend_deg"]
    ear_points = [skeleton.point("left_ear"), skeleton.point("right_ear"), skeleton.point("nose")]

    shoulder_width = skeleton.shoulder_width()
    if not np.isnan(shoulder_width) and shoulder_width > 1e-3:
        px_per_cm = shoulder_width / cfg_front["shoulder_breadth_cm"]
    else:
        px_per_cm = 10.0
    metrics["phone_px_per_cm_est"] = px_per_cm

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
            reach_cm = reach / max(px_per_cm, 1e-3)
            metrics["reach_cm"] = reach_cm
            if reach_cm > cfg_front["phone_reach_max_cm"]:
                base = max(base, 2)
                adjustments["outside_reach"] = SECTION_B_ADJUSTMENTS["telephone"].get("outside_reach", 2)
                queries["too_far_of_reach_outside_30_cm"] = 2

    # Use frame diagonal to estimate proximity of audio devices to the head region.
    diag = (frame_shape[0] ** 2 + frame_shape[1] ** 2) ** 0.5
    head_threshold = 0.18 * diag
    saw_audio_contact = False
    # Detect headsets / earbuds by checking if an object detection bounding box
    # falls near either ear while the neck stays neutral.
    for label, conf, bbox in audio_devices:
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
        distances = [distance(center, pt) for pt in ear_points if pt is not None]
        if distances and min(distances) <= head_threshold and neutral_neck:
            queries["headset_or_one_hand_on_phone_neutral_neck_posture"] = 1
            saw_audio_contact = True
            break

    if not saw_audio_contact and phone_bbox is not None:
        queries["no_hands_free_options"] = 1

    return ComponentOutput(
        base=base,
        adjustments=adjustments,
        metrics=metrics,
        queries=queries,
    )


__all__ = ["monitor_components", "phone_components"]
