"""Side-camera sensory heuristics for ROSA Section A (chair)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from constants.thresholds import SECTION_A_THRESHOLDS, SECTION_A_ADJUSTMENTS, SECTION_C_THRESHOLDS
from core.geometry import Skeleton2D, distance
from . import ComponentOutput

BBox = Tuple[int, int, int, int]

_MOUSE_SURFACE_FLAG = 0
_WORK_SURFACE_FLAG = 0

def seat_height_components(
    skeleton: Skeleton2D,
    desk_info: Optional[Tuple[BBox, float]] = None,
) -> ComponentOutput:
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
    leg_lengths: List[float] = []
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
        target = knee_cfg.get("target", 90.0)
        tolerance = knee_cfg.get("ideal_tolerance_deg", 5.0)
        metrics["target_knee_angle"] = target
        metrics["knee_angle_deviation"] = abs(avg_angle - target)
        low_cutoff = knee_cfg["too_low_max"]
        high_cutoff = knee_cfg["too_high_min"]
        if avg_angle > high_cutoff:
            base = 2
            queries["too_high_knee_angle_greater_than_90_deg"] = 2
        elif avg_angle < low_cutoff:
            base = 2
            queries["too_low_knee_angle_less_than_90_deg"] = 2
        elif abs(avg_angle - target) <= tolerance:
            queries["knees_at_90_deg"] = 1

    # CSA/ROSA expect feet flat on floor; use ankle drop relative to leg length
    # (hip-to-ankle) to flag loss of contact.
    foot_contact = True
    min_ratio = cfg.get("foot_contact_min_ratio", 0.0)
    min_drop_px = cfg.get("foot_contact_min_drop_px", 0.0)
    for side in ("left", "right"):
        hip = skeleton.point(f"{side}_hip")
        knee = skeleton.point(f"{side}_knee")
        ankle = skeleton.point(f"{side}_ankle")
        if hip is None or knee is None or ankle is None:
            continue
        drop = ankle[1] - knee[1]
        metrics[f"{side}_ankle_drop_px"] = drop
        # Immediate checks that do not require scaled ratios.
        if drop <= 0.0 or drop < min_drop_px:
            foot_contact = False

        leg_len = distance(hip, ankle)
        if leg_len <= 1e-3:
            continue
        if not np.isnan(leg_len):
            leg_lengths.append(leg_len)
        drop_ratio = drop / max(leg_len, 1e-3)
        metrics[f"{side}_ankle_drop_ratio"] = drop_ratio
        # Flag loss of contact if ankle fails ratio threshold as well.
        if drop_ratio < min_ratio:
            foot_contact = False
    if not foot_contact:
        base = max(base, 3)
        queries["no_foot_contact_on_ground"] = 2

    avg_leg_len = float(np.mean(leg_lengths)) if leg_lengths else float("nan")
    metrics["avg_leg_length_px"] = avg_leg_len

    desk_clearance_px = float("nan")
    desk_clearance_cm = float("nan")
    desk_clearance_ratio = float("nan")
    metrics["desk_detection_available"] = 1 if desk_info is not None else 0

    if desk_info is not None:
        desk_bbox, desk_conf = desk_info
        metrics["desk_detection_conf"] = float(desk_conf)
        knee_mid = skeleton.knee_mid()
        if knee_mid is not None:
            x1, y1, x2, y2 = desk_bbox
            # Only evaluate if knee is roughly within desk vertical span
            vertical_margin = 40.0
            if (y1 - vertical_margin) <= knee_mid[1] <= (y2 + vertical_margin):
                desk_center_x = 0.5 * (x1 + x2)
                if knee_mid[0] <= desk_center_x:
                    desk_edge_x = float(x1)
                    clearance_px = desk_edge_x - knee_mid[0]
                else:
                    desk_edge_x = float(x2)
                    clearance_px = knee_mid[0] - desk_edge_x
                desk_clearance_px = clearance_px
                metrics["desk_edge_x_px"] = desk_edge_x
                # Estimate pixels per centimetre using leg length
                px_per_cm = float("nan")
                leg_fallback_cm = cfg.get("leg_length_cm_fallback", 90.0)
                if not np.isnan(avg_leg_len) and avg_leg_len > 1e-3 and leg_fallback_cm > 1e-3:
                    px_per_cm = avg_leg_len / leg_fallback_cm
                metrics["px_per_cm_leg_est"] = px_per_cm
                if not np.isnan(px_per_cm) and px_per_cm > 1e-3:
                    desk_clearance_cm = clearance_px / px_per_cm
                ratio_min = cfg.get("desk_clearance_ratio_min", 0.05)
                if not np.isnan(avg_leg_len) and avg_leg_len > 1e-3:
                    desk_clearance_ratio = clearance_px / avg_leg_len
                insufficient = clearance_px < 0
                if not insufficient:
                    clearance_goal_cm = cfg.get("legroom_clearance_cm_min", 5.0)
                    if not np.isnan(desk_clearance_cm):
                        insufficient = desk_clearance_cm < clearance_goal_cm
                    elif not np.isnan(desk_clearance_ratio):
                        insufficient = desk_clearance_ratio < ratio_min
                if insufficient:
                    queries["insufficient_space_under_desk_ability_to_cross_legs"] = 1
                    adjustments["insufficient_legroom"] = SECTION_A_ADJUSTMENTS["seat_height"].get("insufficient_legroom", 1)
            else:
                metrics["desk_edge_x_px"] = float("nan")
        else:
            metrics["desk_edge_x_px"] = float("nan")
    else:
        metrics["desk_detection_conf"] = float("nan")
        metrics["desk_edge_x_px"] = float("nan")

    metrics["desk_clearance_px"] = desk_clearance_px
    metrics["desk_clearance_cm"] = desk_clearance_cm
    metrics["desk_clearance_ratio"] = desk_clearance_ratio

    return ComponentOutput(
        base=base,
        adjustments=adjustments,
        metrics=metrics,
        queries=queries,
    )


def seat_depth_components(
    skeleton: Skeleton2D,
    chair_info: Optional[Tuple[BBox, float]] = None,
) -> ComponentOutput:
    """Estimate seat-pan clearance relative to the knee using chair detection."""
    queries: Dict[str, int] = {
        "approximately_three_inches_between_knee_and_seat_edge": 0,
        "too_long_less_than_three_inches_of_space": 0,
        "too_short_more_than_three_inches_of_space": 0,
    }
    metrics: Dict[str, float] = {
        "chair_detection_available": 1 if chair_info is not None else 0,
    }
    base = 1

    if chair_info is None:
        metrics.update(
            {
                "chair_detection_conf": float("nan"),
                "seat_clearance_px": float("nan"),
                "seat_clearance_cm": float("nan"),
                "seat_clearance_ratio": float("nan"),
                "seat_edge_x_px": float("nan"),
                "px_per_cm_thigh_est": float("nan"),
                "avg_thigh_length_px": float("nan"),
                "seat_depth_unclassified": 1,
            }
        )
        return ComponentOutput(base=base, adjustments={}, metrics=metrics, queries=queries)

    chair_bbox, chair_conf = chair_info
    metrics["chair_detection_conf"] = float(chair_conf)

    knee_mid = skeleton.knee_mid()
    if knee_mid is None:
        metrics.update(
            {
                "seat_clearance_px": float("nan"),
                "seat_clearance_cm": float("nan"),
                "seat_clearance_ratio": float("nan"),
                "seat_edge_x_px": float("nan"),
                "px_per_cm_thigh_est": float("nan"),
                "avg_thigh_length_px": float("nan"),
                "seat_depth_unclassified": 1,
            }
        )
        return ComponentOutput(base=base, adjustments={}, metrics=metrics, queries=queries)

    cx1, cy1, cx2, cy2 = chair_bbox
    seat_center_x = 0.5 * (cx1 + cx2)
    if knee_mid[0] >= seat_center_x:
        seat_edge_x = float(cx2)
        clearance_px = float(knee_mid[0] - seat_edge_x)
        orientation = 1.0
    else:
        seat_edge_x = float(cx1)
        clearance_px = float(seat_edge_x - knee_mid[0])
        orientation = -1.0

    metrics["seat_edge_x_px"] = seat_edge_x
    metrics["seat_clearance_px"] = clearance_px
    metrics["seat_orientation"] = orientation

    thigh_lengths: List[float] = []
    for side in ("left", "right"):
        hip = skeleton.point(f"{side}_hip")
        knee = skeleton.point(f"{side}_knee")
        if hip is None or knee is None:
            continue
        length = distance(hip, knee)
        if not np.isnan(length) and length > 1e-3:
            thigh_lengths.append(length)

    avg_thigh = float(np.mean(thigh_lengths)) if thigh_lengths else float("nan")
    metrics["avg_thigh_length_px"] = avg_thigh

    seat_depth_cfg = SECTION_A_THRESHOLDS["seat_depth"]
    thigh_fallback = seat_depth_cfg.get("thigh_length_cm_fallback", 46.0)
    px_per_cm = float("nan")
    if not np.isnan(avg_thigh) and thigh_fallback > 1e-3:
        px_per_cm = avg_thigh / thigh_fallback
    metrics["px_per_cm_thigh_est"] = px_per_cm

    clearance_cm_limits = seat_depth_cfg["clearance_cm"]
    clearance_cm = float("nan")
    if not np.isnan(px_per_cm) and px_per_cm > 1e-3:
        clearance_cm = clearance_px / px_per_cm
    metrics["seat_clearance_cm"] = clearance_cm

    clearance_ratio = float("nan")
    if not np.isnan(avg_thigh) and avg_thigh > 1e-3:
        clearance_ratio = clearance_px / avg_thigh
    metrics["seat_clearance_ratio"] = clearance_ratio

    classified = False

    if not np.isnan(clearance_cm):
        classified = True
        if clearance_px < 0 or clearance_cm < clearance_cm_limits["too_long_max"]:
            base = 2
            queries["too_long_less_than_three_inches_of_space"] = 2
        elif clearance_cm > clearance_cm_limits["too_short_min"]:
            base = 2
            queries["too_short_more_than_three_inches_of_space"] = 2
        else:
            queries["approximately_three_inches_between_knee_and_seat_edge"] = 1
    else:
        ratio_limits = seat_depth_cfg.get("clearance_ratio_limits", {})
        min_ratio = ratio_limits.get("min")
        max_ratio = ratio_limits.get("max")
        if min_ratio is not None and max_ratio is not None and not np.isnan(clearance_ratio):
            classified = True
            if clearance_px < 0 or clearance_ratio < float(min_ratio):
                base = 2
                queries["too_long_less_than_three_inches_of_space"] = 2
            elif clearance_ratio > float(max_ratio):
                base = 2
                queries["too_short_more_than_three_inches_of_space"] = 2
            else:
                queries["approximately_three_inches_between_knee_and_seat_edge"] = 1

    metrics["seat_depth_unclassified"] = 0 if classified else 1

    return ComponentOutput(
        base=base,
        adjustments={},
        metrics=metrics,
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
