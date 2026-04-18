"""Literature-based range checks for FTIR, XRD, TGA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from soil_analytics._signal import has_prominent_peak
from soil_analytics.paths import project_root
from soil_analytics.schemas import FTIRSeries, TGACurve, XRDPattern

CheckStatus = Literal["pass", "warn", "fail", "info"]


@dataclass
class CheckResult:
    check_id: str
    label: str
    status: CheckStatus
    message: str
    evidence: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_ftir(
    series: FTIRSeries,
    config_path: Path | None = None,
) -> list[CheckResult]:
    path = config_path or project_root() / "config" / "reference_ranges" / "ftir_bands.yaml"
    data = _load_yaml(path)
    results: list[CheckResult] = []
    wn = series.wavenumber_cm1
    y = series.y
    invert = series.y_label == "transmittance"

    for band in data.get("bands", []):
        lo, hi = float(band["wavenumber_min"]), float(band["wavenumber_max"])
        mask = (wn >= lo) & (wn <= hi)
        if not np.any(mask):
            results.append(
                CheckResult(
                    check_id=band["id"],
                    label=band["label"],
                    status="warn",
                    message=f"No data in {lo}-{hi} cm⁻¹",
                    evidence={"wavenumber_range": [lo, hi]},
                )
            )
            continue
        seg_wn = wn[mask]
        seg_y = y[mask]
        peak_idx = int(np.argmax(seg_y) if not invert else np.argmin(seg_y))
        peak_wn = float(seg_wn[peak_idx])
        peak_val = float(seg_y[peak_idx])
        if len(seg_y) >= 5:
            has_peak = has_prominent_peak(seg_y, invert=invert, prominence_scale=0.3)
        else:
            has_peak = True

        status: CheckStatus = "pass" if has_peak else "warn"
        results.append(
            CheckResult(
                check_id=band["id"],
                label=band["label"],
                status=status,
                message=f"Extremum at {peak_wn:.1f} cm⁻¹ (value {peak_val:.4f})",
                evidence={
                    "peak_wavenumber_cm1": peak_wn,
                    "peak_value": peak_val,
                    "wavenumber_range": [lo, hi],
                    "source": band.get("source", ""),
                    "notes": band.get("notes", ""),
                },
            )
        )
    return results


def ftir_inference_rows(results: list[CheckResult]) -> list[dict[str, str]]:
    """
    Turn FTIR band check results into rows for a band table: peak wavenumber and qualitative inference.
    Inference text comes from YAML ``notes`` when present; otherwise the check message.
    """
    rows: list[dict[str, str]] = []
    for r in results:
        ev = r.evidence or {}
        peak = ev.get("peak_wavenumber_cm1")
        notes = (ev.get("notes") or "").strip()
        wn_str = f"{float(peak):.1f}" if peak is not None else "—"
        inference = notes if notes else r.message
        rows.append(
            {
                "Band": r.label,
                "Peak wavenumber (cm⁻¹)": wn_str,
                "Inference": inference,
                "Status": r.status,
            }
        )
    return rows


def check_xrd(
    pattern: XRDPattern,
    config_path: Path | None = None,
) -> list[CheckResult]:
    path = config_path or project_root() / "config" / "reference_ranges" / "xrd_peaks.yaml"
    data = _load_yaml(path)
    tt = pattern.two_theta_deg
    intens = pattern.intensity
    results: list[CheckResult] = []

    for mineral in data.get("minerals", []):
        lo, hi = float(mineral["two_theta_min"]), float(mineral["two_theta_max"])
        mask = (tt >= lo) & (tt <= hi)
        if not np.any(mask):
            results.append(
                CheckResult(
                    check_id=mineral["id"],
                    label=mineral["label"],
                    status="info",
                    message=f"No scan coverage in {lo}-{hi}° 2θ",
                    evidence={"two_theta_range": [lo, hi]},
                )
            )
            continue
        seg_i = intens[mask]
        seg_tt = tt[mask]
        peak_idx = int(np.argmax(seg_i))
        peak_tt = float(seg_tt[peak_idx])
        peak_i = float(seg_i[peak_idx])
        # prominence vs local baseline (min in window)
        prom = peak_i - float(np.min(seg_i))
        status: CheckStatus = "pass" if prom > 0 and peak_i > np.median(intens) else "warn"
        results.append(
            CheckResult(
                check_id=mineral["id"],
                label=mineral["label"],
                status=status,
                message=f"Max intensity in window at {peak_tt:.2f}° 2θ (I≈{peak_i:.2f})",
                evidence={
                    "peak_two_theta": peak_tt,
                    "peak_intensity": peak_i,
                    "two_theta_range": [lo, hi],
                    "source": mineral.get("source", ""),
                    "notes": mineral.get("notes", ""),
                    "assumptions": data.get("assumptions", {}),
                },
            )
        )
    return results


def check_tga(
    curve: TGACurve,
    config_path: Path | None = None,
) -> list[CheckResult]:
    path = config_path or project_root() / "config" / "reference_ranges" / "tga_windows.yaml"
    data = _load_yaml(path)
    t = curve.temperature_c
    mass = curve.mass
    m0 = float(mass[0]) if len(mass) else 1.0
    if abs(m0) < 1e-12:
        m0 = 1.0
    results: list[CheckResult] = []

    for win in data.get("windows", []):
        lo, hi = float(win["temp_min_c"]), float(win["temp_max_c"])
        mask = (t >= lo) & (t <= hi)
        if not np.any(mask):
            results.append(
                CheckResult(
                    check_id=win["id"],
                    label=win["label"],
                    status="info",
                    message=f"No data in {lo}-{hi} °C",
                    evidence={"temp_range_c": [lo, hi]},
                )
            )
            continue
        m_start = float(mass[mask][0])
        m_end = float(mass[mask][-1])
        delta_frac = (m_start - m_end) / abs(m0)
        status: CheckStatus = "pass" if delta_frac > 1e-6 else "warn"
        results.append(
            CheckResult(
                check_id=win["id"],
                label=win["label"],
                status=status,
                message=(
                    f"Approx. fractional mass loss in window: {delta_frac * 100:.3f}% of initial"
                ),
                evidence={
                    "delta_mass_fraction": delta_frac,
                    "temp_range_c": [lo, hi],
                    "source": win.get("source", ""),
                    "notes": win.get("notes", ""),
                    "assumptions": data.get("assumptions", {}),
                },
            )
        )
    return results
