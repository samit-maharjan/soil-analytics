"""XRD phase matching to literature 2θ windows for plots and phase summary tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soil_analytics.schemas import XRDPattern


@dataclass(frozen=True)
class PhaseWindow:
    """One searchable 2θ interval for a phase line."""

    phase: str
    symbol: str
    tt_lo: float
    tt_hi: float
    significance: str


# Laboratory 2θ windows (lime mortar / carbonation; Cu Kα-style reporting).
XRD_PHASE_WINDOWS: tuple[PhaseWindow, ...] = (
    PhaseWindow("Portlandite", "P", 18.0, 18.5, "Residual or unreacted lime"),
    PhaseWindow("Calcite", "C", 22.5, 23.5, "Carbonated lime phase"),
    PhaseWindow("Vaterite", "Vt", 24.5, 25.5, "Metastable calcium carbonate phase"),
    PhaseWindow("Quartz", "Q", 25.5, 26.5, "Siliceous aggregate phase"),
    PhaseWindow("Aragonite", "Ar", 26.0, 27.0, "Calcium carbonate polymorph"),
    PhaseWindow("Calcite (major peak)", "C", 29.0, 30.0, "Dominant stable binder phase"),
    PhaseWindow("Calcite / Aragonite", "C/A", 30.5, 31.5, "Carbonation-related phase"),
    PhaseWindow("Vaterite", "Vt", 32.0, 33.0, "Intermediate carbonation product"),
    PhaseWindow("Mullite", "M", 33.0, 34.0, "Aluminosilicate phase"),
    PhaseWindow("Portlandite", "P", 34.0, 34.5, "Partial carbonation / lime residue"),
    PhaseWindow("Aragonite", "Ar", 36.0, 37.0, "Secondary carbonate polymorph peak"),
    PhaseWindow("Calcite", "C", 39.0, 40.5, "Stable crystalline lime phase"),
    PhaseWindow("Mullite", "M", 40.0, 41.0, "Clay / pozzolanic contribution"),
    PhaseWindow("Calcite", "C", 42.5, 43.5, "Lime mortar matrix confirmation"),
    PhaseWindow("Mullite", "M", 45.0, 46.0, "Aluminosilicate contribution"),
    PhaseWindow("Calcite / Quartz", "C+Q", 47.0, 48.5, "Binder–aggregate interaction"),
    PhaseWindow("Vaterite", "Vt", 49.0, 50.0, "Minor metastable carbonate phase"),
    PhaseWindow("Mullite", "M", 54.0, 55.0, "Secondary mullite peak"),
    PhaseWindow("Calcite", "C", 56.0, 58.0, "Mature carbonation"),
    PhaseWindow("Calcite", "C", 59.0, 61.0, "Crystalline stability"),
    PhaseWindow("Mullite / Aragonite", "M+Ar", 60.0, 62.0, "Minor phase confirmation"),
    PhaseWindow("Quartz / Silicates", "Q+S", 64.0, 66.0, "Aggregate phase"),
    PhaseWindow("Calcite / Mullite", "C+M", 72.0, 75.0, "Minor crystalline phases"),
)


@dataclass(frozen=True)
class PhaseHit:
    phase: str
    symbol: str
    two_theta_deg: float
    intensity: float
    significance: str
    window_id: str


def _dynamic_range(intensity: np.ndarray) -> float:
    return float(np.max(intensity) - np.min(intensity)) if len(intensity) else 1.0


def find_phase_hits(pattern: XRDPattern, *, min_prominence_frac: float = 0.004) -> list[PhaseHit]:
    """
    For each reference window, take the scan maximum in that 2θ range if the rise above the
    window minimum is meaningful versus the overall pattern dynamic range (filters flat noise).
    """
    tt = pattern.two_theta_deg
    iy = pattern.intensity.astype(float)
    full_scale = _dynamic_range(iy)
    floor = max(full_scale * 1e-9, 1e-12)
    hits: list[PhaseHit] = []

    for j, w in enumerate(XRD_PHASE_WINDOWS):
        mask = (tt >= w.tt_lo) & (tt <= w.tt_hi)
        if not np.any(mask):
            continue
        seg_tt = tt[mask]
        seg_i = iy[mask]
        peak_rel = int(np.argmax(seg_i))
        peak_i = float(seg_i[peak_rel])
        peak_tt = float(seg_tt[peak_rel])
        base = float(np.min(seg_i))
        prom = peak_i - base
        if prom < min_prominence_frac * max(full_scale, floor):
            continue
        wid = f"{w.symbol}_{j}_{w.tt_lo:.1f}-{w.tt_hi:.1f}"
        hits.append(
            PhaseHit(
                phase=w.phase,
                symbol=w.symbol,
                two_theta_deg=peak_tt,
                intensity=peak_i,
                significance=w.significance,
                window_id=wid,
            )
        )
    return hits


def xrd_manual_two_theta_rows(two_theta_deg: float) -> list[dict[str, str]]:
    """
    For a user-entered 2θ (degrees), return reference rows for every **laboratory window** that
    contains that value (``XRD_PHASE_WINDOWS``).
    """
    tt = float(two_theta_deg)
    out: list[dict[str, str]] = []
    for w in XRD_PHASE_WINDOWS:
        if w.tt_lo <= tt <= w.tt_hi:
            out.append(
                {
                    "Phase": w.phase,
                    "Symbol": w.symbol,
                    "2θ range (°)": f"{w.tt_lo:.1f}–{w.tt_hi:.1f}",
                    "Your 2θ (°)": f"{tt:.2f}",
                    "Reference note": w.significance,
                }
            )
    if not out:
        return [
            {
                "Result": f"No 2θ window in the current lab table includes {tt:.2f}°. "
                "Widen the value or use a full scan to match maxima in windows."
            }
        ]
    return out


_PHASE_ORDER = {
    "Portlandite": 0,
    "Quartz": 1,
    "Mullite": 2,
    "Calcite": 3,
    "Calcite (major peak)": 4,
    "Vaterite": 5,
    "Aragonite": 6,
    "Calcite / Aragonite": 7,
    "Calcite / Quartz": 8,
    "Mullite / Aragonite": 9,
    "Quartz / Silicates": 10,
    "Calcite / Mullite": 11,
}


def merge_xrd_phase_rows(hits_per_sample: list[list[PhaseHit]]) -> list[dict[str, str]]:
    """
    One row per phase: unique 2θ values across all samples, sorted, comma-separated.
    Columns: Phase, Symbol, 2θ values (°), Phase significance.
    """
    phase_to_deg: dict[str, set[float]] = {}
    phase_sym: dict[str, str] = {}
    phase_sig: dict[str, set[str]] = {}

    for hits in hits_per_sample:
        for h in hits:
            phase_to_deg.setdefault(h.phase, set()).add(round(h.two_theta_deg, 2))
            phase_sym[h.phase] = h.symbol
            phase_sig.setdefault(h.phase, set()).add(h.significance)

    def sort_key(p: str) -> tuple[int, str]:
        return (_PHASE_ORDER.get(p, 99), p)

    rows: list[dict[str, str]] = []
    for phase in sorted(phase_to_deg.keys(), key=sort_key):
        degs = sorted(phase_to_deg[phase])
        deg_str = ", ".join(f"{d:.2f}" for d in degs)
        sig_str = "; ".join(sorted(phase_sig.get(phase, set())))
        rows.append(
            {
                "Phase": phase,
                "Symbol": phase_sym.get(phase, "—"),
                "2θ values (°)": deg_str,
                "Phase significance": sig_str,
            }
        )
    return rows
