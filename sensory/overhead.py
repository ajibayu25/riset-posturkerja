"""Overhead-camera sensory heuristics for ROSA Section C."""

from __future__ import annotations

from typing import Dict

import numpy as np

from constants.thresholds import SECTION_C_THRESHOLDS
from core.geometry import Skeleton2D
from . import ComponentOutput


def mouse_components(skeleton: Skeleton2D) -> ComponentOutput:
    """Placeholder mouse metrics (extend with real heuristics as available)."""
    queries: Dict[str, int] = {
        "neck_twist_gt_30": 0,
    }
    return ComponentOutput(base=0, adjustments={}, metrics={}, queries=queries)


def keyboard_components(skeleton: Skeleton2D) -> ComponentOutput:
    """Use neck twist and typing deviation heuristics."""
    queries: Dict[str, int] = {
        "typing_deviation": 0,
    }
    cfg = SECTION_C_THRESHOLDS["keyboard"]
    metrics: Dict[str, float] = {}

    neck_twist = skeleton.neck_sidebend()
    metrics["neck_sidebend"] = neck_twist
    if not np.isnan(neck_twist) and abs(neck_twist) > cfg["wrist_deviation_deg"]:
        queries["typing_deviation"] = 1

    return ComponentOutput(base=0, adjustments={}, metrics=metrics, queries=queries)


__all__ = ["mouse_components", "keyboard_components"]

