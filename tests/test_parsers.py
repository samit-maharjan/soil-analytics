"""Tests for CSV parsers."""

from pathlib import Path

import numpy as np

from soil_analytics.parsers import parse_ftir_csv, parse_tga_csv, parse_xrd_asc, parse_xrd_bytes, parse_xrd_csv

FIX = Path(__file__).parent / "fixtures"


def test_parse_ftir() -> None:
    raw = (FIX / "sample_ftir.csv").read_bytes()
    s = parse_ftir_csv(raw, source_name="sample")
    assert s.y_label == "absorbance"
    assert len(s.wavenumber_cm1) == len(s.y)
    assert np.all(np.diff(s.wavenumber_cm1) >= 0)


def test_parse_ftir_jcamp() -> None:
    raw = b"""##TITLE=Test
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=%T
1000.0\t50.0
2000.0\t60.0
3000.0\t70.0
"""
    s = parse_ftir_csv(raw, source_name="jcamp")
    assert s.y_label == "transmittance"
    assert len(s.wavenumber_cm1) == 3
    assert np.allclose(s.wavenumber_cm1, [1000.0, 2000.0, 3000.0])


def test_parse_xrd() -> None:
    raw = (FIX / "sample_xrd.csv").read_bytes()
    p = parse_xrd_csv(raw)
    assert len(p.two_theta_deg) == len(p.intensity)


def test_parse_xrd_asc() -> None:
    raw = b"5.0 100.0\n5.1 110.0\n5.2 105.0\n"
    p = parse_xrd_asc(raw, source_name="t.asc")
    assert len(p.two_theta_deg) == 3
    assert p.source_name == "t.asc"


def test_parse_xrd_bytes_dispatch() -> None:
    raw = b"10.0 1.0\n10.1 2.0\n"
    a = parse_xrd_bytes(raw, filename="x.asc")
    assert len(a.two_theta_deg) == 2
    csv = (FIX / "sample_xrd.csv").read_bytes()
    b = parse_xrd_bytes(csv, filename="sample_xrd.csv")
    assert len(b.two_theta_deg) == len(b.intensity)


def test_parse_tga() -> None:
    raw = (FIX / "sample_tga.csv").read_bytes()
    c = parse_tga_csv(raw)
    assert c.dtg is not None
    assert len(c.temperature_c) == len(c.mass)
