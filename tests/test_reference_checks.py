"""Tests for reference checks."""

from pathlib import Path

from soil_analytics.parsers import parse_ftir_csv, parse_tga_csv, parse_xrd_csv
from soil_analytics.reference_checks import check_ftir, check_tga, check_xrd

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]


def test_ftir_checks() -> None:
    raw = (FIX / "sample_ftir.csv").read_bytes()
    s = parse_ftir_csv(raw)
    cfg = ROOT / "config" / "reference_ranges" / "ftir_bands.yaml"
    results = check_ftir(s, config_path=cfg)
    assert len(results) >= 1
    ids = {r.check_id for r in results}
    assert "aliphatic_ch" in ids


def test_xrd_checks() -> None:
    raw = (FIX / "sample_xrd.csv").read_bytes()
    p = parse_xrd_csv(raw)
    cfg = ROOT / "config" / "reference_ranges" / "xrd_peaks.yaml"
    results = check_xrd(p, config_path=cfg)
    assert any(r.check_id == "quartz_101" for r in results)


def test_tga_checks() -> None:
    raw = (FIX / "sample_tga.csv").read_bytes()
    c = parse_tga_csv(raw)
    cfg = ROOT / "config" / "reference_ranges" / "tga_windows.yaml"
    results = check_tga(c, config_path=cfg)
    assert len(results) >= 1
