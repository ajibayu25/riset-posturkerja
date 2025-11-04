"""Side-camera sensory heuristics for ROSA Section A (chair)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from constants.thresholds import SECTION_A_THRESHOLDS, SECTION_A_ADJUSTMENTS, SECTION_C_THRESHOLDS
from core.geometry import Skeleton2D, distance
from . import ComponentOutput


_MOUSE_SURFACE_FLAG = 0
_WORK_SURFACE_FLAG = 0

def seat_height_components(skeleton: Skeleton2D) -> ComponentOutput:
    """Evaluate knee flexion, foot contact, and under-desk space."""
    cfg = SECTION_A_THRESHOLDS["seat_height"]
    queries: Dict[str, int] = {
        "knees_at_90_deg": 0,
        "too_low_knee_angle_less_than_90_deg": 0,
        "too_high_knee_angle_greater_than_90_deg": 0,
        "no_foot_contact_on_ground": 0,
        "insufficient_space_under_desk_ability_to_cross_legs": 0,
    }
    metrics: Dict[str, float] = {}
    adjustments: Dict[str, int] = {}
    base = 1

    # ROSA Section A expects knees close to 90 degrees. Capture both sides and average.
    angles: List[float] = []
    knee_cfg = cfg["knee_angle_deg"]
    for side in ("left", "right"):
        angle = skeleton.knee_angle(side)
        metrics[f"{side}_knee_angle"] = angle
        if np.isnan(angle):
            continue
        angles.append(angle)

    if angles:
        avg_angle = float(np.mean(angles))
        metrics["avg_knee_angle"] = avg_angle
        if knee_cfg["neutral_min"] <= avg_angle <= knee_cfg["neutral_max"]:
            queries["knees_at_90_deg"] = 1
        if avg_angle < knee_cfg["too_low_max"]:
            base = 2
            queries["too_low_knee_angle_less_than_90_deg"] = 2
        if avg_angle > knee_cfg["too_high_min"]:
            base = 2
            queries["too_high_knee_angle_greater_than_90_deg"] = 2

    # CSA/ROSA expect feet flat on floor; use ankle drop relative to leg length
    # to flag loss of contact.
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
        queries["no_foot_contact_on_ground"] = 2

    return ComponentOutput(
        base=base,
        adjustments=adjustments,
        metrics=metrics,
        queries=queries,
    )


def seat_depth_components(skeleton: Skeleton2D) -> ComponentOutput:
    """Placeholder heuristics for seat-pan depth related indicators."""
    queries: Dict[str, int] = {
        "approximately_three_inches_between_knee_and_seat_edge": 1,
        "too_long_less_than_three_inches_of_space": 0,
        "too_short_more_than_three_inches_of_space": 0,
    }
    return ComponentOutput(
        base=1,
        adjustments={},
        metrics={},
        queries=queries,
    )


def armrest_components(skeleton: Skeleton2D) -> ComponentOutput:
    """Score armrest height using shoulder–elbow gap per CSA Z412 / ROSA."""
    thresholds = SECTION_A_THRESHOLDS["armrest"]["shoulder_elbow_gap_cm"]
    queries: Dict[str, int] = {
        "elbows_supported_in_line_with_shoulder_shoulders_relaxed": 0,
        "too_high_shoulders_shrugged_or_low_arms_unsupported": 0,
        "armrests_too_wide": 0,
    }
    metrics: Dict[str, float] = {}
    adjustments: Dict[str, int] = {}
    base = 1

    left_shoulder = skeleton.point("left_shoulder")
    right_shoulder = skeleton.point("right_shoulder")
    left_elbow = skeleton.point("left_elbow")
    right_elbow = skeleton.point("right_elbow")

    if any(pt is None for pt in (left_shoulder, right_shoulder, left_elbow, right_elbow)):
        return ComponentOutput(base=base, adjustments={}, metrics=metrics, queries=queries)

    shoulder_y = float((left_shoulder[1] + right_shoulder[1]) / 2.0)
    elbow_y = float((left_elbow[1] + right_elbow[1]) / 2.0)
    metrics["shoulder_y_px"] = shoulder_y
    metrics["elbow_y_px"] = elbow_y

    shoulder_width = skeleton.shoulder_width()
    PX_PER_CM_FALLBACK = 38.0  # typical biacromial breadth in cm.
    # When the scene is not calibrated we approximate pixel-to-centimetre scaling
    # by assuming a nominal shoulder breadth (38 cm). This keeps the gap thresholds
    # meaningful even without explicit calibration.
    if not np.isnan(shoulder_width) and shoulder_width > 1e-3:
        px_per_cm = shoulder_width / PX_PER_CM_FALLBACK
    else:
        px_per_cm = 10.0
    metrics["px_per_cm_est"] = px_per_cm

    dy_cm = (elbow_y - shoulder_y) / max(px_per_cm, 1e-3)
    metrics["shoulder_elbow_gap_cm"] = dy_cm

    # Evaluate shoulder-elbow height gap against ergonomic thresholds.
    if dy_cm < thresholds["too_high_max"]:
        base = 2
        queries["too_high_shoulders_shrugged_or_low_arms_unsupported"] = 2
        metrics["armrest_classification"] = "too_high"
    elif dy_cm > thresholds["too_low_min"]:
        base = 2
        queries["too_high_shoulders_shrugged_or_low_arms_unsupported"] = 2
        metrics["armrest_classification"] = "too_low"
    else:
        queries["elbows_supported_in_line_with_shoulder_shoulders_relaxed"] = 1
        metrics["armrest_classification"] = "neutral"

    # Detect armrests that force the elbows beyond ~20% wider than shoulder breadth.
    elbow_span = abs(left_elbow[0] - right_elbow[0])
    if not np.isnan(shoulder_width) and shoulder_width > 1e-3:
        abduction_ratio = (elbow_span / shoulder_width) - 1.0
        metrics["armrest_abduction_ratio"] = abduction_ratio
        if abduction_ratio > SECTION_A_THRESHOLDS["armrest"]["max_abduction_ratio"]:
            adjustments = {"too_wide_armrest_spacing": SECTION_A_ADJUSTMENTS["armrest"].get("too_wide_armrest_spacing", 1)}
            queries["armrests_too_wide"] = 1
        else:
            adjustments = {}
    else:
        metrics["armrest_abduction_ratio"] = float("nan")

    return ComponentOutput(base=base, adjustments=adjustments, metrics=metrics, queries=queries)


def back_support_components(skeleton: Skeleton2D) -> ComponentOutput:
    """Evaluate trunk inclination relative to lumbar support expectations."""
    cfg = SECTION_A_THRESHOLDS["back_support"]["recline_deg"]
    queries: Dict[str, int] = {
        "adequate_lumbar_support_chair_reclined_between_95_110_deg": 0,
        "no_lumbar_support_or_not_positioned_in_small_of_back": 0,
        "angled_too_far_back_greater_than_110_or_too_far_forward_less_than_95": 0,
        "no_back_support_or_worker_leaning_forward": 0,
        "hard_or_damaged_surface": 0,
    }
    metrics: Dict[str, float] = {}
    adjustments: Dict[str, int] = {}
    base = 1

    inclination = skeleton.trunk_inclination()
    metrics["trunk_inclination"] = inclination
    if not np.isnan(inclination):
        if cfg["neutral_min"] <= inclination <= cfg["neutral_max"]:
            queries["adequate_lumbar_support_chair_reclined_between_95_110_deg"] = 1
        elif inclination < cfg["forward_cutoff"] or inclination > cfg["rear_cutoff"]:
            base = 2
            queries["angled_too_far_back_greater_than_110_or_too_far_forward_less_than_95"] = 2

    return ComponentOutput(
        base=base,
        adjustments={},
        metrics=metrics,
        queries=queries,
    )




def assess_mouse_keyboard_surfaces(
    skeleton: Skeleton2D,
    hand_preference: str = "right",
) -> Dict[str, float]:
    """Estimate height gap between mouse and keyboard surfaces from side view."""
    cfg = SECTION_C_THRESHOLDS["mouse"]
    metrics: Dict[str, float] = {}
    dominant = hand_preference.lower()
    if dominant not in {"left", "right"}:
        dominant = "right"
    wrist_dom = skeleton.point(f"{dominant}_wrist")
    wrist_other = skeleton.point("left_wrist" if dominant == "right" else "right_wrist")
    if wrist_dom is None or wrist_other is None:
        _set_mouse_surface_flag(0)
        return metrics
    shoulder_width = skeleton.shoulder_width()
    if not np.isnan(shoulder_width) and shoulder_width > 1e-3:
        px_per_cm = shoulder_width / cfg.get("shoulder_breadth_cm", 38.0)
    else:
        px_per_cm = 10.0
    diff_cm = abs(float(wrist_dom[1] - wrist_other[1])) / max(px_per_cm, 1e-3)
    metrics["mouse_keyboard_surface_diff_cm"] = diff_cm
    flag = 2 if diff_cm > cfg["surface_height_diff_cm"] else 0
    metrics["mouse_keyboard_surface_flag"] = flag
    _set_mouse_surface_flag(flag)
    return metrics


def assess_work_surface_elevation(
    skeleton: Skeleton2D,
    hand_preference: str = "right",
) -> Dict[str, float]:
    """Estimate shoulder shrug posture indicating work surface too high."""
    metrics: Dict[str, float] = {}
    shoulder_mid = skeleton.shoulder_mid()
    hip_mid = skeleton.hip_mid()
    if shoulder_mid is None or hip_mid is None:
        _set_work_surface_flag(0)
        return metrics
    torso_len = float(distance(shoulder_mid, hip_mid))
    if torso_len < 1e-3 or np.isnan(torso_len):
        torso_len = skeleton.shoulder_width()
    if torso_len < 1e-3 or np.isnan(torso_len):
        torso_len = 100.0

    left_elbow = skeleton.point("left_elbow")
    right_elbow = skeleton.point("right_elbow")
    left_wrist = skeleton.point("left_wrist")
    right_wrist = skeleton.point("right_wrist")
    if any(v is None for v in (left_elbow, right_elbow, left_wrist, right_wrist)):
        _set_work_surface_flag(0)
        return metrics

    wrist_heights = [
        float(elbow[1] - wrist[1])
        for elbow, wrist in ((left_elbow, left_wrist), (right_elbow, right_wrist))
    ]
    metrics["work_surface_wrist_diff_px"] = float(np.mean(wrist_heights))

    flag = 0
    if wrist_heights and all(diff > 0.12 * torso_len for diff in wrist_heights):
        flag = 1
    metrics["work_surface_flag"] = flag
    _set_work_surface_flag(flag)
    return metrics


def _set_mouse_surface_flag(value: int) -> None:
    global _MOUSE_SURFACE_FLAG
    _MOUSE_SURFACE_FLAG = value


def get_mouse_surface_flag() -> int:
    return _MOUSE_SURFACE_FLAG


def _set_work_surface_flag(value: int) -> None:
    global _WORK_SURFACE_FLAG
    _WORK_SURFACE_FLAG = value


def get_work_surface_flag() -> int:
    return _WORK_SURFACE_FLAG


__all__ = [
    "seat_height_components",
    "seat_depth_components",
    "armrest_components",
    "back_support_components",
    "assess_mouse_keyboard_surfaces",
    "assess_work_surface_elevation",
    "get_mouse_surface_flag",
    "get_work_surface_flag",
]
