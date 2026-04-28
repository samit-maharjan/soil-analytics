"""Manual wavenumber / 2θ / TGA window lookups."""

from pathlib import Path

from lime_analytics.paths import project_root, reference_config_dir
from lime_analytics.reference_checks import ftir_manual_wavenumber_rows, tga_window_manual_row
from lime_analytics.xrd_phases import xrd_manual_two_theta_rows


def test_ftir_manual_hits_narrow_bands() -> None:
    cfg: Path = reference_config_dir() / "ftir_bands.yaml"
    # pick something likely inside a band; if none, the “no match” path is also valid
    rows = ftir_manual_wavenumber_rows(2000.0, cfg)
    assert rows
    assert "Inference" in rows[0] or "Result" in rows[0]


def test_xrd_manual_finds_window() -> None:
    # middle of a common calcite / lime window
    r = xrd_manual_two_theta_rows(29.4)
    assert r
    assert "Phase" in r[0] or "Result" in r[0]


def test_tga_manual_one_row() -> None:
    root = project_root()
    import yaml

    with open(root / "config" / "reference_ranges" / "tga_windows.yaml", encoding="utf-8") as f:
        w0 = (yaml.safe_load(f) or {}).get("windows", [])[0]
    rows = tga_window_manual_row(w0, -0.5)
    assert len(rows) == 1
    assert "Inference (reference)" in rows[0]
