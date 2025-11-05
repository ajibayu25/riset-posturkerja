"""Overhead-camera sensory heuristics for ROSA Section C."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import numpy as np

from constants.thresholds import SECTION_C_THRESHOLDS, SECTION_C_ADJUSTMENTS
from core.geometry import Skeleton2D, angle_between, distance
from sensory.side import get_mouse_surface_flag
from . import ComponentOutput

BBox = Tuple[int, int, int, int]


def mouse_components(
    skeleton: Skeleton2D,
    mouse_bbox: Optional[BBox],
    hand_bboxes: Optional[List[BBox]],
    hand_preference: str = "right",
) -> ComponentOutput:
    """Evaluate mouse posture for ROSA Section C (mouse axis)."""
    cfg = SECTION_C_THRESHOLDS["mouse"]
    hand_bboxes = hand_bboxes or []
    queries: Dict[str, int] = {
        "mouse_in_line_with_shoulder": 0,
        "reaching_to_mouse": 0,
        "pinch_grip_on_mouse": 0,
        "mouse_keyboard_on_different_surfaces": 0,
    }
    metrics: Dict[str, float] = {}
    adjustments: Dict[str, int] = {}
    base = 1

    shoulder_width = skeleton.shoulder_width()
    if not np.isnan(shoulder_width) and shoulder_width > 1e-3:
        px_per_cm = shoulder_width / cfg.get("shoulder_breadth_cm", 38.0)
    else:
        px_per_cm = 10.0
    metrics["px_per_cm_est"] = px_per_cm

    dominant = hand_preference.lower()
    if dominant not in {"left", "right"}:
        dominant = "right"
    shoulder = skeleton.point(f"{dominant}_shoulder")
    elbow = skeleton.point(f"{dominant}_elbow")
    wrist = skeleton.point(f"{dominant}_wrist")
    hip = skeleton.point(f"{dominant}_hip")

    if any(pt is None for pt in (shoulder, elbow, wrist, hip)):
        return ComponentOutput(base=base, adjustments=adjustments, metrics=metrics, queries=queries)

    if mouse_bbox is not None:
        x1, y1, x2, y2 = mouse_bbox
        mouse = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
    else:
        mouse = np.array(wrist, dtype=float)
    metrics["mouse_center_x"] = float(mouse[0])
    metrics["mouse_center_y"] = float(mouse[1])

    offset_cm = abs(float(mouse[0] - shoulder[0])) / max(px_per_cm, 1e-3)
    reach_cm = distance(mouse, wrist) / max(px_per_cm, 1e-3)
    arm_vec = elbow - shoulder
    torso_vec = hip - shoulder
    abd_deg = float(angle_between(arm_vec, torso_vec))
    if np.isnan(abd_deg):
        abd_deg = 0.0
    metrics["offset_cm"] = offset_cm
    metrics["reach_cm"] = reach_cm
    metrics["abduction_deg"] = abd_deg

    inline = (
        offset_cm <= cfg["lateral_offset_cm"]["inline_max"]
        and reach_cm <= cfg["reach_inline_cm_max"]
        and abd_deg <= cfg["inline_abduction_deg"]
    )
    if inline:
        queries["mouse_in_line_with_shoulder"] = 1

    reaching = (
        offset_cm >= cfg["lateral_offset_cm"]["reach_min"]
        or reach_cm >= cfg["reach_cm_min"]
        or abd_deg >= cfg["reaching_abduction_deg"]
    )
    if reaching and not inline:
        base = max(base, 2)
        queries["reaching_to_mouse"] = 2

    surface_flag = get_mouse_surface_flag()
    if surface_flag:
        queries["mouse_keyboard_on_different_surfaces"] = surface_flag
        if surface_flag >= 2:
            base = max(base, 2)
            adjustments["different_surface"] = SECTION_C_ADJUSTMENTS["mouse"].get("different_surface", 2)

    pinch_flag = 0
    best_contact_ratio = float("inf")
    metrics["hand_bbox_count"] = float(len(hand_bboxes))
    if mouse_bbox is not None and hand_bboxes:
        mx1, my1, mx2, my2 = map(float, mouse_bbox)
        mouse_area = max(1.0, (mx2 - mx1) * (my2 - my1))
        for hb in hand_bboxes:
            hx1, hy1, hx2, hy2 = map(float, hb)
            inter_x1 = max(mx1, hx1)
            inter_y1 = max(my1, hy1)
            inter_x2 = min(mx2, hx2)
            inter_y2 = min(my2, hy2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
            intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            hand_area = max(1.0, (hx2 - hx1) * (hy2 - hy1))
            ratio = intersection / hand_area
            best_contact_ratio = min(best_contact_ratio, ratio)
            if ratio <= cfg["contact_area_ratio_max"]:
                pinch_flag = 1
                adjustments["pinch_grip"] = SECTION_C_ADJUSTMENTS["mouse"].get("pinch_grip", 1)
                break
        if best_contact_ratio < float("inf"):
            metrics["pinch_contact_ratio"] = best_contact_ratio
        metrics["mouse_area_px"] = mouse_area
    queries["pinch_grip_on_mouse"] = pinch_flag

    return ComponentOutput(base=base, adjustments=adjustments, metrics=metrics, queries=queries)


def _match_hand_centers(
    skeleton: Skeleton2D,
    hand_bboxes: List[BBox],
) -> Dict[str, np.ndarray]:
    """Assign detected hand boxes to left/right wrists."""
    assignments: Dict[str, np.ndarray] = {}
    remaining = list(enumerate(hand_bboxes))
    for side in ("left", "right"):
        wrist = skeleton.point(f"{side}_wrist")
        if wrist is None or not remaining:
            continue
        wrist_arr = np.asarray(wrist, dtype=float)
        best_idx = None
        best_dist = float("inf")
        for idx, bbox in remaining:
            x1, y1, x2, y2 = map(float, bbox)
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
            dist = float(np.linalg.norm(center - wrist_arr))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            continue
        # Remove matched entry from remaining
        for i, (original_idx, bbox) in enumerate(remaining):
            if original_idx == best_idx:
                x1, y1, x2, y2 = map(float, bbox)
                assignments[side] = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
                remaining.pop(i)
                break
    return assignments


def keyboard_components(
    skeleton: Skeleton2D,
    hand_bboxes: Optional[List[BBox]] = None,
) -> ComponentOutput:
    """Detect excessive radial/ulnar deviation while typing."""
    queries: Dict[str, int] = {
        "deviation_while_typing": 0,
    }
    cfg = SECTION_C_THRESHOLDS["keyboard"]
    metrics: Dict[str, float] = {}
    adjustments: Dict[str, int] = {}

    hand_bboxes = hand_bboxes or []
    assignments = _match_hand_centers(skeleton, hand_bboxes)

    deviation_threshold = cfg.get("wrist_deviation_deg", 10.0)
    max_deviation = 0.0
    deviation_hits = 0

    for side in ("left", "right"):
        elbow = skeleton.point(f"{side}_elbow")
        wrist = skeleton.point(f"{side}_wrist")
        hand_center = assignments.get(side)
        key = f"{side}_typing_deviation_deg"
        if elbow is None or wrist is None or hand_center is None:
            metrics[key] = float("nan")
            continue
        forearm_vec = np.asarray(wrist, dtype=float) - np.asarray(elbow, dtype=float)
        hand_vec = hand_center - np.asarray(wrist, dtype=float)
        if np.linalg.norm(forearm_vec) < 1e-3 or np.linalg.norm(hand_vec) < 1e-3:
            metrics[key] = float("nan")
            continue
        deviation_deg = float(angle_between(forearm_vec, hand_vec))
        metrics[key] = deviation_deg
        max_deviation = max(max_deviation, deviation_deg)
        if deviation_deg > deviation_threshold:
            deviation_hits += 1

    metrics["typing_max_deviation_deg"] = max_deviation
    metrics["typing_deviation_hits"] = float(deviation_hits)
    metrics["hand_bbox_count_keyboard"] = float(len(hand_bboxes))

    if deviation_hits > 0:
        queries["deviation_while_typing"] = 1
        adjustments["wrist_deviation"] = SECTION_C_ADJUSTMENTS["keyboard"].get("wrist_deviation", 1)

    return ComponentOutput(base=0, adjustments=adjustments, metrics=metrics, queries=queries)


__all__ = ["mouse_components", "keyboard_components"]
