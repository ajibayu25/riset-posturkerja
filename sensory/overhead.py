"""Overhead-camera sensory heuristics for ROSA Section C."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import cv2
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
    palmrest_flag: Optional[bool] = None,
    palmrest_metrics: Optional[Dict[str, float]] = None,
) -> ComponentOutput:
    """Evaluate mouse posture for ROSA Section C (mouse axis)."""
    cfg = SECTION_C_THRESHOLDS["mouse"]
    hand_bboxes = hand_bboxes or []
    queries: Dict[str, int] = {
        "mouse_in_line_with_shoulder": 0,
        "reaching_to_mouse": 0,
        "pinch_grip_on_mouse": 0,
        "mouse_keyboard_on_different_surfaces": 0,
        "palmrest_in_front_of_mouse": 0,
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

    # Palmrest heuristic (smoothed upstream)
    if palmrest_metrics:
        for key, value in palmrest_metrics.items():
            metrics[f"palmrest_{key}"] = value
    if palmrest_flag is True:
        metrics["palmrest_detected"] = 1.0
        queries["palmrest_in_front_of_mouse"] = 1
        adjustments["palmrest_front"] = SECTION_C_ADJUSTMENTS["mouse"].get("palmrest_front", 1)
    elif palmrest_flag is False:
        metrics["palmrest_detected"] = 0.0
    else:
        metrics["palmrest_detected"] = float("nan")

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


def detect_palmrest(
    frame: Optional[np.ndarray],
    skeleton: Skeleton2D,
    hand_bboxes: Optional[List[BBox]],
    hand_preference: str,
    mouse_bbox: Optional[BBox],
) -> Dict[str, object]:
    """Heuristic palmrest detection in front of the mouse using an overhead view."""
    palm_cfg = SECTION_C_THRESHOLDS["mouse"].get("palmrest", {})
    result: Dict[str, object] = {
        "valid": False,
        "flag": False,
        "confidence": 0.0,
        "bbox": None,
        "corridor": None,
        "px_per_cm": float("nan"),
        "metrics": {},
    }
    if frame is None:
        return result

    hand_bboxes = hand_bboxes or []
    assignments = _match_hand_centers(skeleton, hand_bboxes)

    shoulder_width = skeleton.shoulder_width()
    if not np.isnan(shoulder_width) and shoulder_width > 1e-3:
        px_per_cm = shoulder_width / SECTION_C_THRESHOLDS["mouse"].get("shoulder_breadth_cm", 38.0)
    else:
        px_per_cm = 10.0
    result["px_per_cm"] = px_per_cm

    dominant = hand_preference.lower()
    if dominant not in {"left", "right"}:
        dominant = "right"

    wrist = skeleton.point(f"{dominant}_wrist")
    hand_center = assignments.get(dominant)
    if wrist is None:
        return result
    wrist_arr = np.asarray(wrist, dtype=float)

    mouse_center: Optional[np.ndarray] = None
    if mouse_bbox is not None:
        mx1, my1, mx2, my2 = map(float, mouse_bbox)
        mouse_center = np.array([(mx1 + mx2) / 2.0, (my1 + my2) / 2.0], dtype=float)
    elif hand_center is not None:
        mouse_center = hand_center + (hand_center - np.asarray(wrist, dtype=float))

    direction_vec: Optional[np.ndarray] = None
    if hand_center is not None:
        direction_vec = hand_center - wrist_arr
    if direction_vec is None and mouse_center is not None:
        direction_vec = mouse_center - wrist_arr
    if direction_vec is None:
        return result

    direction_norm = np.linalg.norm(direction_vec)
    if direction_norm < 1e-3:
        return result
    d = direction_vec / direction_norm
    v = np.array([-d[1], d[0]])

    def cm_to_px_range(key: str) -> Tuple[float, float]:
        lo, hi = palm_cfg.get(key, (0.0, 0.0))
        return float(lo) * px_per_cm, float(hi) * px_per_cm

    length_lo_px, length_hi_px = cm_to_px_range("length_cm")
    width_lo_px, width_hi_px = cm_to_px_range("width_cm")
    start_offset_px = float(palm_cfg.get("start_offset_cm", 0.5)) * px_per_cm
    length_px = length_hi_px
    half_width_px = max(width_lo_px, width_hi_px) / 2.0

    p0 = wrist_arr + d * start_offset_px
    corners = np.array(
        [
            p0 + v * half_width_px,
            p0 - v * half_width_px,
            p0 - v * half_width_px + d * length_px,
            p0 + v * half_width_px + d * length_px,
        ],
        dtype=np.int32,
    )
    result["corridor"] = corners

    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(corners)
    rect_w = max(1, rect_w)
    rect_h = max(1, rect_h)
    local_corners = corners.copy()
    local_corners[:, 0] -= rect_x
    local_corners[:, 1] -= rect_y

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w]
    mask = np.zeros((rect_h, rect_w), np.uint8)
    cv2.fillPoly(mask, [local_corners], 255)
    roi = cv2.bitwise_and(roi_gray, roi_gray, mask=mask)
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    if blur.size == 0:
        return result
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_height_cm = palm_cfg.get("min_height_cm", 3.0)
    max_height_cm = palm_cfg.get("max_height_cm", 6.0)
    min_width_cm = palm_cfg.get("min_width_cm", 8.0)
    max_width_cm = palm_cfg.get("max_width_cm", 20.0)
    min_area_cm2 = palm_cfg.get("min_area_cm2", 25.0)
    ratio_min = palm_cfg.get("aspect_ratio_min", 1.2)
    ratio_max = palm_cfg.get("aspect_ratio_max", 4.0)
    max_texture_var = palm_cfg.get("max_texture_var", 120.0)

    best_bbox: Optional[BBox] = None
    best_proj = -np.inf
    best_confidence = 0.0
    samples = 0

    def px_to_cm(value: float) -> float:
        return value / max(px_per_cm, 1e-6)

    for contour in contours:
        if len(contour) < 8:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        w_cm = px_to_cm(float(w))
        h_cm = px_to_cm(float(h))
        area_cm2 = w_cm * h_cm
        if not (min_height_cm <= h_cm <= max_height_cm):
            continue
        if not (min_width_cm <= w_cm <= max_width_cm):
            continue
        if area_cm2 < min_area_cm2:
            continue
        ratio = w / h if h > 0 else 0.0
        if not (ratio_min <= ratio <= ratio_max):
            continue
        patch = roi_gray[y : y + h, x : x + w]
        if patch.size == 0:
            continue
        lap_var = cv2.Laplacian(patch, cv2.CV_64F).var()
        if lap_var > max_texture_var:
            continue
        center = np.array(
            [rect_x + x + w / 2.0, rect_y + y + h / 2.0],
            dtype=float,
        )
        proj = float(np.dot(center - wrist_arr, d))
        if proj <= 0:
            continue
        if mouse_center is not None:
            m_proj = float(np.dot(mouse_center - wrist_arr, d))
            if proj >= m_proj:
                continue
        lateral = abs(float(np.dot(center - wrist_arr, v)))
        if lateral > half_width_px * 1.5:
            continue
        confidence = min(1.0, area_cm2 / max(min_area_cm2, 1e-6))
        samples += 1
        if proj > best_proj:
            best_proj = proj
            best_bbox = (int(rect_x + x), int(rect_y + y), int(w), int(h))
            best_confidence = confidence

    result["metrics"] = {
        "palmrest_candidate_count": float(samples),
        "palmrest_best_projection_px": float(best_proj if best_proj > -np.inf else 0.0),
        "palmrest_confidence_raw": float(best_confidence),
    }

    if best_bbox is not None:
        result.update(
            {
                "valid": True,
                "flag": True,
                "confidence": best_confidence,
                "bbox": best_bbox,
            }
        )
    else:
        result.update({"valid": True, "flag": False, "confidence": 0.0})

    return result


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


__all__ = ["mouse_components", "keyboard_components", "detect_palmrest"]
