"""XRD phase matching to literature 2θ windows (P / M / C) for stacked plots and tables."""

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


# Reference windows derived from qualitative tables (Cu Kα-style reporting; same assumptions as app notes).
XRD_PHASE_WINDOWS: tuple[PhaseWindow, ...] = (
    PhaseWindow("Portlandite", "P", 17.0, 19.0, "Residual/unreacted lime (Ca(OH)₂); indicates partial carbonation"),
    PhaseWindow("Portlandite", "P", 33.0, 35.0, "Residual/unreacted lime (Ca(OH)₂); indicates partial carbonation"),
    PhaseWindow("Mullite", "M", 15.5, 17.5, "Fired clay mineral; confirms brick powder (surkhi) addition"),
    PhaseWindow("Mullite", "M", 25.5, 27.5, "Fired clay mineral; confirms brick powder (surkhi) addition"),
    PhaseWindow("Mullite", "M", 39.5, 41.5, "Fired clay mineral; confirms brick powder (surkhi) addition"),
    PhaseWindow("Calcite", "C", 22.0, 24.0, "Major carbonation product (CaCO₃); confirms lime carbonation"),
    PhaseWindow("Calcite", "C", 28.7, 30.2, "Major carbonation product (CaCO₃); confirms lime carbonation"),
    PhaseWindow("Calcite", "C", 30.2, 31.8, "Major carbonation product (CaCO₃); confirms lime carbonation"),
    PhaseWindow("Calcite", "C", 38.5, 40.0, "Major carbonation product (CaCO₃); confirms lime carbonation"),
    PhaseWindow("Calcite", "C", 42.0, 44.0, "Major carbonation product (CaCO₃); confirms lime carbonation"),
    PhaseWindow("Calcite", "C", 46.5, 48.5, "Major carbonation product (CaCO₃); confirms lime carbonation"),
    PhaseWindow("Calcite", "C", 56.0, 58.0, "Major carbonation product (CaCO₃); confirms lime carbonation"),
    PhaseWindow("Calcite", "C", 59.0, 61.0, "Major carbonation product (CaCO₃); confirms lime carbonation"),
    PhaseWindow("Calcite", "C", 63.5, 65.5, "Major carbonation product (CaCO₃); confirms lime carbonation"),
    PhaseWindow("Calcite", "C", 71.5, 75.5, "Major carbonation product (CaCO₃); confirms lime carbonation"),
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


_PHASE_ORDER = {"Portlandite": 0, "Mullite": 1, "Calcite": 2}


def merge_xrd_phase_rows(hits_per_sample: list[list[PhaseHit]]) -> list[dict[str, str]]:
    """
    One row per phase: unique 2θ values across all samples, sorted, comma-separated.
    Columns: Phase, Symbol, 2θ values (°), Phase significance.
    """
    phase_to_deg: dict[str, set[float]] = {}
    phase_sym: dict[str, str] = {}
    phase_sig: dict[str, str] = {}

    for hits in hits_per_sample:
        for h in hits:
            phase_to_deg.setdefault(h.phase, set()).add(round(h.two_theta_deg, 2))
            phase_sym[h.phase] = h.symbol
            phase_sig[h.phase] = h.significance

    def sort_key(p: str) -> tuple[int, str]:
        return (_PHASE_ORDER.get(p, 99), p)

    rows: list[dict[str, str]] = []
    for phase in sorted(phase_to_deg.keys(), key=sort_key):
        degs = sorted(phase_to_deg[phase])
        deg_str = ", ".join(f"{d:.2f}" for d in degs)
        rows.append(
            {
                "Phase": phase,
                "Symbol": phase_sym.get(phase, "—"),
                "2θ values (°)": deg_str,
                "Phase significance": phase_sig.get(phase, ""),
            }
        )
    return rows
