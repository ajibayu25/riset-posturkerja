"""Compute the final ROSA score by combining chair score with monitor/peripheral score."""

from __future__ import annotations

import time
from dataclasses import dataclass

from constants.grids import ROSA_FINAL_AXIS, ROSA_FINAL_GRID
from core.geometry import clamp
from scoring.monitor_peripherals import MonitorPeripheralResult
from scoring.sectiona import SectionAResult


@dataclass
class ROSATotalResult:
    """Final ROSA assessment combining chair and peripherals scores."""

    timestamp: float
    chair_score: int
    monitor_peripheral_score: int
    rosa_total: int


class ROSATotalScorer:
    """Lookup helper for the final ROSA matrix."""

    def score(self, section_a: SectionAResult, monitor_peripherals: MonitorPeripheralResult) -> ROSATotalResult:
        chair_score = int(clamp(section_a.chair_score_final, ROSA_FINAL_AXIS[0], ROSA_FINAL_AXIS[-1]))
        mp_score = int(clamp(monitor_peripherals.combined_score, ROSA_FINAL_AXIS[0], ROSA_FINAL_AXIS[-1]))

        v_idx = chair_score - ROSA_FINAL_AXIS[0]
        h_idx = mp_score - ROSA_FINAL_AXIS[0]
        total = int(ROSA_FINAL_GRID[v_idx, h_idx])

        return ROSATotalResult(
            timestamp=time.time(),
            chair_score=chair_score,
            monitor_peripheral_score=mp_score,
            rosa_total=total,
        )


__all__ = ["ROSATotalResult", "ROSATotalScorer"]

