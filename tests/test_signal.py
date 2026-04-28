"""Peak detection helpers."""

import numpy as np

from lime_analytics._signal import has_prominent_peak


def test_has_prominent_peak_absorbance() -> None:
    y = np.array([0.0, 0.1, 0.5, 0.2, 0.1])
    assert has_prominent_peak(y, invert=False, prominence_scale=0.05)
