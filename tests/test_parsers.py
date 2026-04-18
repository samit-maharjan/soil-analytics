"""Tests for CSV parsers."""

from pathlib import Path

import numpy as np

from soil_analytics.parsers import parse_ftir_csv, parse_tga_csv, parse_xrd_csv

FIX = Path(__file__).parent / "fixtures"


def test_parse_ftir() -> None:
    raw = (FIX / "sample_ftir.csv").read_bytes()
    s = parse_ftir_csv(raw, source_name="sample")
    assert s.y_label == "absorbance"
    assert len(s.wavenumber_cm1) == len(s.y)
    assert np.all(np.diff(s.wavenumber_cm1) >= 0)


def test_parse_xrd() -> None:
    raw = (FIX / "sample_xrd.csv").read_bytes()
    p = parse_xrd_csv(raw)
    assert len(p.two_theta_deg) == len(p.intensity)


def test_parse_tga() -> None:
    raw = (FIX / "sample_tga.csv").read_bytes()
    c = parse_tga_csv(raw)
    assert c.dtg is not None
    assert len(c.temperature_c) == len(c.mass)
