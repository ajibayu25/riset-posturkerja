"""Geometric helpers for ROSA posture analysis.

The utilities here operate on 2D pose keypoints (x, y[, score]) as produced by
YOLOv8-Pose / COCO-format estimators.  They provide convenient ways to extract
joint positions, compute body angles, distances, and other derived metrics used
by the ROSA scoring pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

# --- Keypoint naming -------------------------------------------------------

COCO_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
COCO_KEYPOINT_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(COCO_KEYPOINT_NAMES)}

PointLike = Union[Sequence[float], np.ndarray]
PointArray = np.ndarray

# --- Basic utilities -------------------------------------------------------

def _ensure_array(point: PointLike) -> Optional[np.ndarray]:
    if point is None:
        return None
    arr = np.asarray(point, dtype=float)
    if arr.size == 0 or np.any(np.isnan(arr[:2])):
        return None
    if arr.ndim == 1:
        return arr[:2]
    return arr.reshape(-1)[:2]


def distance(p1: PointLike, p2: PointLike) -> float:
    a, b = _ensure_array(p1), _ensure_array(p2)
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(a - b))


def midpoint(*points: PointLike) -> Optional[np.ndarray]:
    valid = [_ensure_array(p) for p in points if _ensure_array(p) is not None]
    if not valid:
        return None
    return np.stack(valid, axis=0).mean(axis=0)


def centroid(points: Iterable[PointLike]) -> Optional[np.ndarray]:
    pts = [_ensure_array(p) for p in points if _ensure_array(p) is not None]
    if not pts:
        return None
    return np.stack(pts, axis=0).mean(axis=0)


def vector(p1: PointLike, p2: PointLike) -> Optional[np.ndarray]:
    a, b = _ensure_array(p1), _ensure_array(p2)
    if a is None or b is None:
        return None
    return b - a


def normalize(v: PointLike) -> Optional[np.ndarray]:
    arr = _ensure_array(v)
    if arr is None:
        return None
    norm = np.linalg.norm(arr)
    if norm < 1e-6:
        return None
    return arr / norm


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def angle_deg(a: PointLike, b: PointLike, c: PointLike) -> float:
    """Return the interior angle ABC in degrees (0-180)."""
    pa, pb, pc = _ensure_array(a), _ensure_array(b), _ensure_array(c)
    if pa is None or pb is None or pc is None:
        return float("nan")
    v1, v2 = pa - pb, pc - pb
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def angle_between(v1: PointLike, v2: PointLike) -> float:
    """Unsigned angle in degrees between two vectors."""
    a, b = normalize(v1), normalize(v2)
    if a is None or b is None:
        return float("nan")
    cosang = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def signed_angle_between(v1: PointLike, v2: PointLike) -> float:
    """Signed angle in degrees from v1 to v2 around +Z axis (right-hand rule)."""
    a, b = normalize(v1), normalize(v2)
    if a is None or b is None:
        return float("nan")
    angle = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b)))
    return float(angle)


def angle_to_horizontal(p1: PointLike, p2: PointLike) -> float:
    """Angle of the vector p1->p2 relative to the +X axis."""
    v = vector(p1, p2)
    if v is None:
        return float("nan")
    return float(np.degrees(np.arctan2(v[1], v[0])))


def angle_to_vertical(p1: PointLike, p2: PointLike) -> float:
    """Angle (absolute) of vector p1->p2 relative to the +Y axis."""
    v = vector(p1, p2)
    if v is None:
        return float("nan")
    vertical = np.array([0.0, 1.0])
    return angle_between(v, vertical)


def vertical_displacement(p1: PointLike, p2: PointLike) -> float:
    a, b = _ensure_array(p1), _ensure_array(p2)
    if a is None or b is None:
        return float("nan")
    return float(b[1] - a[1])


def horizontal_displacement(p1: PointLike, p2: PointLike) -> float:
    a, b = _ensure_array(p1), _ensure_array(p2)
    if a is None or b is None:
        return float("nan")
    return float(b[0] - a[0])


def bounding_box(points: Iterable[PointLike]) -> Optional[Tuple[float, float, float, float]]:
    pts = [_ensure_array(p) for p in points if _ensure_array(p) is not None]
    if not pts:
        return None
    arr = np.stack(pts, axis=0)
    xmin, ymin = np.min(arr[:, 0]), np.min(arr[:, 1])
    xmax, ymax = np.max(arr[:, 0]), np.max(arr[:, 1])
    return float(xmin), float(ymin), float(xmax), float(ymax)


# --- Skeleton helper -------------------------------------------------------

@dataclass
class Skeleton2D:
    """Convenience wrapper around a 2D keypoint array/dict."""

    keypoints: Dict[str, np.ndarray]
    confidence: Dict[str, float]

    @classmethod
    def from_array(
        cls,
        keypoints: np.ndarray,
        names: Sequence[str] = COCO_KEYPOINT_NAMES,
        min_confidence: float = 0.0,
    ) -> "Skeleton2D":
        arr = np.asarray(keypoints, dtype=float)
        if arr.ndim != 2:
            raise ValueError("keypoints must be shape (N, D)")
        if arr.shape[0] != len(names):
            raise ValueError("unexpected keypoint count")
        dims = arr.shape[1]
        points: Dict[str, np.ndarray] = {}
        conf: Dict[str, float] = {}
        for idx, name in enumerate(names):
            point = arr[idx]
            score = float(point[2]) if dims > 2 else 1.0
            if score < min_confidence:
                continue
            points[name] = point[:2].astype(float)
            conf[name] = score
        return cls(points, conf)

    def has(self, name: str) -> bool:
        return name in self.keypoints

    def point(self, name: str) -> Optional[np.ndarray]:
        return self.keypoints.get(name)

    def confidence_of(self, name: str) -> float:
        return self.confidence.get(name, 0.0)

    # --- paired helpers -------------------------------------------------
    def pair_midpoint(self, left: str, right: str) -> Optional[np.ndarray]:
        return midpoint(self.point(left), self.point(right))

    def shoulder_mid(self) -> Optional[np.ndarray]:
        return self.pair_midpoint("left_shoulder", "right_shoulder")

    def hip_mid(self) -> Optional[np.ndarray]:
        return self.pair_midpoint("left_hip", "right_hip")

    def knee_mid(self) -> Optional[np.ndarray]:
        return self.pair_midpoint("left_knee", "right_knee")

    # --- joint angles ---------------------------------------------------
    def elbow_angle(self, side: str) -> float:
        joint = f"{side}_elbow"
        shoulder = f"{side}_shoulder"
        wrist = f"{side}_wrist"
        return angle_deg(self.point(shoulder), self.point(joint), self.point(wrist))

    def knee_angle(self, side: str) -> float:
        joint = f"{side}_knee"
        hip = f"{side}_hip"
        ankle = f"{side}_ankle"
        return angle_deg(self.point(hip), self.point(joint), self.point(ankle))

    def hip_angle(self, side: str) -> float:
        hip = f"{side}_hip"
        knee = f"{side}_knee"
        shoulder = f"{side}_shoulder"
        return angle_deg(self.point(shoulder), self.point(hip), self.point(knee))

    def shoulder_abduction(self, side: str) -> float:
        shoulder = f"{side}_shoulder"
        hip = f"{side}_hip"
        elbow = f"{side}_elbow"
        return angle_deg(self.point(hip), self.point(shoulder), self.point(elbow))

    # --- posture angles -------------------------------------------------
    def trunk_inclination(self) -> float:
        hip_mid = self.hip_mid()
        shoulder_mid = self.shoulder_mid()
        if hip_mid is None or shoulder_mid is None:
            return float("nan")
        return angle_to_vertical(hip_mid, shoulder_mid)

    def neck_flexion(self) -> float:
        shoulder_mid = self.shoulder_mid()
        nose = self.point("nose")
        if shoulder_mid is None or nose is None:
            return float("nan")
        vertical = np.array([0.0, -1.0])
        direction = nose - shoulder_mid
        return angle_between(direction, vertical)

    def neck_sidebend(self) -> float:
        shoulder_mid = self.shoulder_mid()
        nose = self.point("nose")
        if shoulder_mid is None or nose is None:
            return float("nan")
        # Signed; positive when nose is to worker's right (camera view dependant)
        vertical = np.array([0.0, -1.0])
        direction = nose - shoulder_mid
        return signed_angle_between(vertical, direction)

    def shoulder_height_diff(self) -> float:
        left, right = self.point("left_shoulder"), self.point("right_shoulder")
        if left is None or right is None:
            return float("nan")
        return float(left[1] - right[1])

    def shoulder_width(self) -> float:
        left, right = self.point("left_shoulder"), self.point("right_shoulder")
        if left is None or right is None:
            return float("nan")
        return distance(left, right)

    def hip_width(self) -> float:
        left, right = self.point("left_hip"), self.point("right_hip")
        if left is None or right is None:
            return float("nan")
        return distance(left, right)

    def limb_length(self, side: str, proximal: str, distal: str) -> float:
        a = f"{side}_{proximal}"
        b = f"{side}_{distal}"
        return distance(self.point(a), self.point(b))

    def ankle_height(self, side: str) -> float:
        ankle = self.point(f"{side}_ankle")
        if ankle is None:
            return float("nan")
        return float(ankle[1])


__all__ = [
    "COCO_KEYPOINT_NAMES",
    "COCO_KEYPOINT_INDEX",
    "Skeleton2D",
    "angle_deg",
    "angle_between",
    "angle_to_horizontal",
    "angle_to_vertical",

    "bounding_box",
    "centroid",
    "clamp",
    "distance",
    "horizontal_displacement",
    "midpoint",
    "normalize",
    "signed_angle_between",
    "vector",
    "vertical_displacement",
]
