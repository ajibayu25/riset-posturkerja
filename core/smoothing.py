"""Temporal smoothing utilities for pose/keypoint streams."""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np


class EMA:
    """Simple exponential moving average for numpy arrays."""

    def __init__(self, alpha: float = 0.4) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def update(self, value) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if self._state is None:
            self._state = arr.copy()
        else:
            self._state = self.alpha * arr + (1.0 - self.alpha) * self._state
        return self._state.copy()


class MedianK:
    """Running median over the last *k* samples."""

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
        self._buf: deque[np.ndarray] = deque(maxlen=self.k)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, value) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        self._buf.append(arr)
        stack = np.stack(self._buf, axis=0)
        return np.median(stack, axis=0)


class OneEuroFilter:
    """One Euro filter (Casiez et al.) for smooth yet responsive tracking."""

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        if freq <= 0:
            raise ValueError("freq must be positive")
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._prev: Optional[np.ndarray] = None
        self._prev_deriv: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._prev = None
        self._prev_deriv = None
        self._prev_t = None

    def update(self, value, timestamp: Optional[float] = None) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if timestamp is None:
            dt = 1.0 / self.freq
        else:
            if self._prev_t is None:
                dt = 1.0 / self.freq
            else:
                dt = max(1e-6, timestamp - self._prev_t)
            self._prev_t = float(timestamp)

        if self._prev is None:
            deriv = np.zeros_like(arr)
        else:
            deriv = (arr - self._prev) / dt

        if self._prev_deriv is None:
            deriv_hat = deriv
        else:
            alpha_d = self._alpha(self.d_cutoff, dt)
            deriv_hat = alpha_d * deriv + (1.0 - alpha_d) * self._prev_deriv

        cutoff = self.min_cutoff + self.beta * np.abs(deriv_hat)
        alpha = self._alpha(np.maximum(cutoff, 1e-3), dt)
        if self._prev is None:
            filtered = arr
        else:
            filtered = alpha * arr + (1.0 - alpha) * self._prev

        self._prev = filtered
        self._prev_deriv = deriv_hat
        return filtered.copy()


class KeypointSmoother:
    """Blend frame-to-frame keypoints with optional confidence gating."""

    def __init__(
        self,
        alpha: float = 0.4,
        min_confidence: float = 0.0,
        use_one_euro: bool = False,
        one_euro_kwargs: Optional[dict] = None,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self._ema = EMA(alpha) if not use_one_euro else None
        self._one_euro = OneEuroFilter(**(one_euro_kwargs or {})) if use_one_euro else None
        self._last: Optional[np.ndarray] = None

    def reset(self) -> None:
        if self._ema:
            self._ema.reset()
        if self._one_euro:
            self._one_euro.reset()
        self._last = None

    def update(self, keypoints, confidence: Optional[np.ndarray] = None, timestamp: Optional[float] = None) -> np.ndarray:
        arr = np.asarray(keypoints, dtype=float)
        if confidence is not None and self._last is not None:
            conf = np.asarray(confidence, dtype=float)
            mask = conf < self.min_confidence
            if mask.ndim == 1:
                mask = mask[:, None]
            arr = np.where(mask, self._last, arr)
        if self._one_euro is not None:
            smoothed = self._one_euro.update(arr, timestamp=timestamp)
        else:
            smoothed = self._ema.update(arr)
        self._last = smoothed
        return smoothed.copy()


__all__ = [
    "EMA",
    "KeypointSmoother",
    "MedianK",
    "OneEuroFilter",
]
