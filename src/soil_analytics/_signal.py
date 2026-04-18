"""Lightweight 1D peak helpers (numpy only; no SciPy)."""


from __future__ import annotations

import numpy as np


def count_prominent_extrema(
    y: np.ndarray,
    *,
    invert: bool,
    prominence_scale: float = 0.3,
) -> int:
    """
    Count local maxima (absorbance) or minima (transmittance) with approximate prominence.

    `prominence_scale` multiplies std(y) as a minimum height above adjacent samples.
    """
    if len(y) < 3:
        return 0
    yy = -y if invert else y
    prom = float(np.std(y)) * prominence_scale + 1e-12
    n = 0
    for i in range(1, len(yy) - 1):
        if yy[i] > yy[i - 1] and yy[i] >= yy[i + 1]:
            left = max(yy[:i])
            right = max(yy[i + 1 :])
            base = max(left, right)
            if yy[i] - base >= prom:
                n += 1
    return n


def has_prominent_peak(y: np.ndarray, *, invert: bool, prominence_scale: float = 0.3) -> bool:
    return count_prominent_extrema(y, invert=invert, prominence_scale=prominence_scale) > 0
