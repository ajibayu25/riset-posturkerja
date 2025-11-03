"""Sensor post-processing helpers for front/side/overhead camera views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ComponentOutput:
    """Structured output of a sensory component analysis."""

    base: int
    adjustments: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    queries: Dict[str, int] = field(default_factory=dict)


__all__ = ["ComponentOutput"]

