"""Side-camera sensory heuristics for ROSA Section A (chair)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from constants.thresholds import SECTION_A_THRESHOLDS
from core.geometry import Skeleton2D, distance
from . import ComponentOutput


def seat_height_components(skeleton: Skeleton2D) -> ComponentOutput:
    """Evaluate knee flexion, foot contact, and under-desk space."""
    cfg = SECTION_A_THRESHOLDS["seat_height"]
    queries: Dict[str, int] = {
        "knees_at_90": 0,
        "knee_angle_too_low": 0,
        "knee_angle_too_high": 0,
        "no_foot_contact": 0,
        "insufficient_under_desk_space": 0,
    }
    metrics: Dict[str, float] = {}
    adjustments: Dict[str, int] = {}
    base = 1

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
            queries["knees_at_90"] = 1
        if avg_angle < knee_cfg["too_low_max"]:
            base = 2
            queries["knee_angle_too_low"] = 2
        if avg_angle > knee_cfg["too_high_min"]:
            base = 2
            queries["knee_angle_too_high"] = 2

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
        queries["no_foot_contact"] = 2

    return ComponentOutput(
        base=base,
        adjustments=adjustments,
        metrics=metrics,
        queries=queries,
    )


def seat_depth_components(skeleton: Skeleton2D) -> ComponentOutput:
    """Placeholder heuristics for seat-pan depth related indicators."""
    queries: Dict[str, int] = {
        "knee_to_seat_gap_ok": 1,
        "seat_too_long": 0,
        "seat_too_short": 0,
    }
    return ComponentOutput(
        base=1,
        adjustments={},
        metrics={},
        queries=queries,
    )


def armrest_components(skeleton: Skeleton2D) -> ComponentOutput:
    """Arm-support placeholder for completeness (Section A horizontal axis)."""
    return ComponentOutput(
        base=1,
        adjustments={},
        metrics={},
        queries={},
    )


def back_support_components(skeleton: Skeleton2D) -> ComponentOutput:
    """Evaluate trunk inclination relative to lumbar support expectations."""
    cfg = SECTION_A_THRESHOLDS["back_support"]["recline_deg"]
    queries: Dict[str, int] = {
        "adequate_lumbar_support": 0,
        "poor_lumbar_position": 0,
        "chair_angle_out_of_range": 0,
        "no_back_support": 0,
        "hard_surface": 0,
    }
    metrics: Dict[str, float] = {}
    base = 1

    inclination = skeleton.trunk_inclination()
    metrics["trunk_inclination"] = inclination
    if not np.isnan(inclination):
        if cfg["neutral_min"] <= inclination <= cfg["neutral_max"]:
            queries["adequate_lumbar_support"] = 1
        elif inclination < cfg["forward_cutoff"] or inclination > cfg["rear_cutoff"]:
            base = 2
            queries["chair_angle_out_of_range"] = 2

    return ComponentOutput(
        base=base,
        adjustments={},
        metrics=metrics,
        queries=queries,
    )


__all__ = [
    "seat_height_components",
    "seat_depth_components",
    "armrest_components",
    "back_support_components",
]

