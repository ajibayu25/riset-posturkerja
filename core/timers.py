"""Timing helpers for ROSA exposure tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from constants.thresholds import DURATION_RULE

__all__ = [
    "ExposureTimer",
    "duration_adjust",
    "format_hms",
    "seconds_since",
]


def seconds_since(ts: float) -> float:
    """Return elapsed seconds from the supplied timestamp to now."""
    return max(0.0, time.time() - ts)


def format_hms(seconds: float) -> str:
    """Format seconds to H:MM:SS."""
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def duration_adjust(total_s: float, max_cont_s: float) -> int:
    """ROSA duration scoring adjustment (-1, 0, +1)."""
    h_total = total_s / 3600.0
    cont = max_cont_s / 60.0
    if h_total < DURATION_RULE["short_daily_hours"] or cont < DURATION_RULE["short_continuous_minutes"]:
        return -1
    if h_total > DURATION_RULE["long_daily_hours"] or cont > DURATION_RULE["long_continuous_minutes"]:
        return +1
    return 0


@dataclass
class ExposureTimer:
    """Track total and continuous exposure durations for ROSA scoring."""

    start_ts: float = field(default_factory=time.time)
    continuous_ts: float = field(default_factory=time.time)
    total_override: Optional[float] = None
    continuous_override: Optional[float] = None

    def reset(self) -> None:
        now = time.time()
        self.start_ts = now
        self.continuous_ts = now
        self.total_override = None
        self.continuous_override = None

    def mark_break(self) -> None:
        self.continuous_ts = time.time()
        self.continuous_override = None

    def set_manual(self, total_hours: Optional[float] = None, continuous_minutes: Optional[float] = None) -> None:
        if total_hours is not None:
            self.total_override = float(max(0.0, total_hours * 3600.0))
        if continuous_minutes is not None:
            self.continuous_override = float(max(0.0, continuous_minutes * 60.0))

    def total_seconds(self) -> float:
        if self.total_override is not None:
            return self.total_override
        return seconds_since(self.start_ts)

    def continuous_seconds(self) -> float:
        if self.continuous_override is not None:
            return self.continuous_override
        return seconds_since(self.continuous_ts)

    def duration_adjustment(self) -> int:
        return duration_adjust(self.total_seconds(), self.continuous_seconds())

    def summary(self) -> str:
        total = format_hms(self.total_seconds())
        continuous = format_hms(self.continuous_seconds())
        adj = self.duration_adjustment()
        return f"Total {total} | Continuous {continuous} | Adj {adj:+d}"
