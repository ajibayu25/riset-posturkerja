"""Combine Section B (monitor/telephone) and Section C (mouse/keyboard) into the ROSA peripherals score."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from constants.grids import MONITOR_PERIPHERALS_AXIS, MONITOR_PERIPHERALS_GRID
from core.geometry import clamp
from scoring.sectionb import SectionBResult
from scoring.sectionc import SectionCResult


@dataclass
class MonitorPeripheralResult:
    """Aggregated ROSA score for monitor/telephone vs mouse/keyboard."""

    timestamp: float
    monitor_section_score: int
    peripherals_section_score: int
    combined_score: int


class MonitorPeripheralScorer:
    """Lookup helper that merges Section B and Section C scores into the combo grid.

    Heuristic summary:
    - Clamp Section B and Section C scores to the 1–9 axis.
    - Use MONITOR_PERIPHERALS_GRID to get the combined monitor+peripheral score.
    """

    def score(
        self,
        section_b: SectionBResult,
        section_c: SectionCResult,
    ) -> MonitorPeripheralResult:
        monitor_score = int(clamp(section_b.section_score, MONITOR_PERIPHERALS_AXIS[0], MONITOR_PERIPHERALS_AXIS[-1]))
        peripheral_score = int(clamp(section_c.section_score, MONITOR_PERIPHERALS_AXIS[0], MONITOR_PERIPHERALS_AXIS[-1]))

        v_idx = peripheral_score - MONITOR_PERIPHERALS_AXIS[0]
        h_idx = monitor_score - MONITOR_PERIPHERALS_AXIS[0]
        combined = int(MONITOR_PERIPHERALS_GRID[v_idx, h_idx])

        return MonitorPeripheralResult(
            timestamp=time.time(),
            monitor_section_score=monitor_score,
            peripherals_section_score=peripheral_score,
            combined_score=combined,
        )


__all__ = ["MonitorPeripheralResult", "MonitorPeripheralScorer"]

