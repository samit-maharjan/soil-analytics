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


def ftir_inference_rows(
    results: list[CheckResult],
    sample: str | None = None,
) -> list[dict[str, str]]:
    """
    Turn FTIR band check results into rows for a band table: peak wavenumber and qualitative inference.
    Inference text comes from YAML ``notes`` when present; otherwise the check message.
    If ``sample`` is set, a ``Sample`` column is included (for multi-spectrum tables).
    """
    rows: list[dict[str, str]] = []
    for r in results:
        ev = r.evidence or {}
        peak = ev.get("peak_wavenumber_cm1")
        notes = (ev.get("notes") or "").strip()
        wn_str = f"{float(peak):.1f}" if peak is not None else "—"
        inference = notes if notes else r.message
        row: dict[str, str] = {
            "Band": r.label,
            "Peak wavenumber (cm⁻¹)": wn_str,
            "Inference": inference,
            "Status": r.status,
        }
        if sample is not None:
            rows.append({"Sample": sample, **row})
        else:
            rows.append(row)
    return rows


def ftir_manual_wavenumber_rows(wavenumber_cm1: float, config_path: Path) -> list[dict[str, str]]:
    """
    For a user-entered wavenumber, return rows for any configured band that contains that value
    (reference inference text from the band ``notes`` in ``ftir_bands.yaml``).
    """
    data = _load_yaml(config_path)
    wn = float(wavenumber_cm1)
    rows: list[dict[str, str]] = []
    for band in data.get("bands", []):
        lo, hi = float(band["wavenumber_min"]), float(band["wavenumber_max"])
        if lo <= wn <= hi:
            notes = (band.get("notes") or "").strip() or "—"
            rows.append(
                {
                    "Band": str(band.get("label", band.get("id", "—"))),
                    "Wavenumber range (cm⁻¹)": f"{lo:g}–{hi:g}",
                    "Value (cm⁻¹)": f"{wn:.1f}",
                    "Inference": notes,
                }
            )
    if not rows:
        return [
            {
                "Result": f"No reference band in the list contains {wn:.1f} cm⁻¹. "
                "Try another value, or use the file upload to scan extrema per window.",
            }
        ]
    return rows


def tga_range_display_str(window: dict) -> str:
    """
    User-facing range text for TGA windows. If ``range_label`` is set in the YAML (e.g. ``<120``,
    ``120–250``), it is used; otherwise ``temp_min_c``/``temp_max_c`` is formatted.
    """
    rlab = (window.get("range_label") or "").strip()
    if rlab:
        return rlab
    lo, hi = float(window["temp_min_c"]), float(window["temp_max_c"])
    return f"{lo:g}–{hi:g}"


def tga_window_manual_row(
    window: dict,
    user_delta_tg: float,
) -> list[dict[str, str]]:
    """
    One row: reference temperature range + label, user-provided ΔTG, and the window ``notes`` as
    inference text. ``window`` is a dict from ``tga_windows.yaml`` (``windows`` list item).
    """
    label = str(window.get("label", "—"))
    inf = " ".join(str(window.get("notes", "") or "").split())
    return [
        {
            "Range (°C)": tga_range_display_str(window),
            "Phase / compound (reference)": label,
            "Your ΔTG (as entered)": f"{user_delta_tg:.6f}",
            "Inference (reference)": inf or "—",
        }
    ]


def ftir_merged_inference_rows(results_per_sample: list[list[CheckResult]]) -> list[dict[str, str]]:
    """
    One row per band across all samples: peak wavenumbers as comma-separated values (sample order
    matches ``results_per_sample``). Columns: Band, Peak wavenumber (cm⁻¹), Inference.
    """
    if not results_per_sample:
        return []
    n = len(results_per_sample[0])
    for res in results_per_sample[1:]:
        if len(res) != n:
            raise ValueError("Inconsistent FTIR band count across samples (use the same ftir_bands.yaml).")
    rows: list[dict[str, str]] = []
    for i in range(n):
        r0 = results_per_sample[0][i]
        peaks: list[str] = []
        for res in results_per_sample:
            r = res[i]
            if r.check_id != r0.check_id:
                raise ValueError("FTIR band ordering differs across samples.")
            ev = r.evidence or {}
            p = ev.get("peak_wavenumber_cm1")
            peaks.append(f"{float(p):.1f}" if p is not None else "—")
        peak_str = ", ".join(peaks)
        ev0 = r0.evidence or {}
        notes = (ev0.get("notes") or "").strip()
        inference = notes if notes else r0.message
        rows.append(
            {
                "Peak wavenumber (cm⁻¹)": peak_str,
                "Band": r0.label,
                "Inference": inference,
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
