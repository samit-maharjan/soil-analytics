"""Matplotlib figures for FTIR, XRD, TGA (lighter than Plotly for this app)."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from soil_analytics.schemas import FTIRSeries, TGACurve, XRDPattern
from soil_analytics.xrd_phases import PhaseHit


def plot_ftir_multi(
    series_list: list[FTIRSeries],
    labels: list[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Overlay several spectra with distinct colors and a legend."""
    if not series_list:
        raise ValueError("series_list must not be empty")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if labels is None:
        labels = [s.source_name or f"series_{i}" for i, s in enumerate(series_list)]
    if len(labels) != len(series_list):
        raise ValueError("labels must match series_list length")

    palette = [f"C{i}" for i in range(10)]
    wn_mins: list[float] = []
    wn_maxs: list[float] = []
    for i, s in enumerate(series_list):
        c = palette[i % len(palette)]
        ax.plot(s.wavenumber_cm1, s.y, color=c, lw=1.0, label=labels[i])
        wn_mins.append(float(s.wavenumber_cm1.min()))
        wn_maxs.append(float(s.wavenumber_cm1.max()))

    ax.set_title(title or "FTIR spectra (overlay)")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    y0 = series_list[0]
    if all(s.y_label == y0.y_label for s in series_list):
        if y0.y_label == "transmittance":
            yl = "Transmittance (%T)" if float(max(s.y.max() for s in series_list)) > 1.5 else "Transmittance"
        elif y0.y_label == "absorbance":
            yl = "Absorbance"
        elif y0.y_label == "reflectance":
            yl = "Reflectance"
        else:
            yl = y0.y_label.capitalize()
    else:
        yl = "Response (mixed units)"
    ax.set_ylabel(yl)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.5)
    ax.set_xlim(max(wn_maxs), min(wn_mins))
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_ftir(series: FTIRSeries, title: str | None = None) -> Figure:
    """Spectrum plot: decreasing wavenumber left-to-right (typical MIR convention)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.wavenumber_cm1, series.y, color="#1f77b4", lw=1.0)
    ax.set_title(title or (series.source_name or "FTIR spectrum"))
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    if series.y_label == "transmittance":
        yl = "Transmittance (%T)" if float(series.y.max()) > 1.5 else "Transmittance"
    elif series.y_label == "absorbance":
        yl = "Absorbance"
    elif series.y_label == "reflectance":
        yl = "Reflectance"
    else:
        yl = series.y_label.capitalize()
    ax.set_ylabel(yl)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.5)
    ax.set_xlim(float(series.wavenumber_cm1.max()), float(series.wavenumber_cm1.min()))
    fig.tight_layout()
    return fig


def plot_xrd(pattern: XRDPattern) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(pattern.two_theta_deg, pattern.intensity, color="C0", lw=1.0)
    ax.set_title("XRD")
    ax.set_xlabel("2θ (degrees)")
    ax.set_ylabel("Intensity")
    fig.tight_layout()
    return fig


# If two window maxima are within this (2θ, °), treat as one peak: one multiline label at the summit.
_XRD_SAME_PEAK_MAX_GAP_DEG = 0.5


def _xrd_cluster_hits_by_neighboring_maxima(
    pat_hits: list[PhaseHit], tt: np.ndarray
) -> list[list[PhaseHit]]:
    """Group hits that land on the same (or immediately neighboring) 2θ maximum in sorted order."""
    if not pat_hits:
        return []
    with_idx: list[tuple[PhaseHit, int, float]] = []
    for h in pat_hits:
        idx = int(np.argmin(np.abs(tt - h.two_theta_deg)))
        with_idx.append((h, idx, float(tt[idx])))
    with_idx.sort(key=lambda t: t[2])
    clusters: list[list[tuple[PhaseHit, int, float]]] = []
    cur: list[tuple[PhaseHit, int, float]] = [with_idx[0]]
    for item in with_idx[1:]:
        ttp = item[2]
        if ttp - cur[-1][2] <= _XRD_SAME_PEAK_MAX_GAP_DEG:
            cur.append(item)
        else:
            clusters.append(cur)
            cur = [item]
    clusters.append(cur)
    return [[h for h, _, _ in c] for c in clusters]


def plot_xrd_multi(
    patterns: list[XRDPattern],
    labels: list[str],
    hits_per_pattern: list[list[PhaseHit]],
    *,
    title: str | None = None,
) -> Figure:
    """Offset vertical layout for several patterns; labels at each maximum, merged per peak (full symbols, e.g. M+Ar)."""
    if not patterns:
        raise ValueError("patterns must not be empty")
    if len(labels) != len(patterns) or len(hits_per_pattern) != len(patterns):
        raise ValueError("labels and hits_per_pattern must match patterns length")

    spans = [float(p.intensity.max()) - float(p.intensity.min()) for p in patterns]
    span = max(spans) if spans else 1.0
    pad = max(span * 0.12, 1.0)
    stack_step = span + pad

    # Bottom to top: black, red, blue (then cycle matplotlib colors).
    fixed = ["#1a1a1a", "#d62728", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for i, p in enumerate(patterns):
        tt = p.two_theta_deg
        iy = p.intensity.astype(float)
        y0 = float(iy.min())
        y_norm = iy - y0
        offset = i * stack_step
        y_plot = y_norm + offset
        color = fixed[i] if i < len(fixed) else f"C{i}"
        ax.plot(tt, y_plot, color=color, lw=0.9, label=labels[i])

        pat_hits = hits_per_pattern[i]
        for group in _xrd_cluster_hits_by_neighboring_maxima(pat_hits, tt):
            group = sorted(group, key=lambda h: (h.two_theta_deg, h.symbol))
            idxs: set[int] = set()
            for h in group:
                j = int(np.argmin(np.abs(tt - h.two_theta_deg)))
                idxs.add(j)
            best_idx = max(idxs, key=lambda j: float(y_plot[j]))
            seen: set[str] = set()
            lines: list[str] = []
            for h in group:
                s = h.symbol
                if not s or s in seen:
                    continue
                seen.add(s)
                lines.append(s)
            if not lines:
                continue
            label = "\n".join(lines) if len(lines) > 1 else lines[0]
            x_p = float(tt[best_idx])
            y_p = float(y_plot[best_idx])
            ax.annotate(
                label,
                xy=(x_p, y_p),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                linespacing=1.2,
            )

    ax.set_title(title or (labels[0] if (len(patterns) == 1 and labels) else "XRD"))
    ax.set_xlabel("2θ (degrees)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(10.0, 80.0)
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.5)
    if len(patterns) > 1:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def tga_mass_at_temp(temperature_c: np.ndarray, mass: np.ndarray, t_query: float) -> float:
    return float(
        np.interp(
            t_query,
            temperature_c.astype(float),
            mass.astype(float),
            left=float(mass[0]),
            right=float(mass[-1]),
        )
    )


def plot_tga_multi_reference(
    curves: list[TGACurve],
    windows: list[dict[str, Any]],
    *,
    title: str | None = None,
    y_axis_label: str | None = None,
) -> Figure:
    """
    Overlay TG curves (temperature vs mass or TG %) with bracket-style mass-change
    annotations per reference temperature window (from ``tga_windows.yaml`` ``windows``).
    """
    if not curves:
        raise ValueError("curves must not be empty")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    palette = [f"C{i % 10}" for i in range(max(len(curves), 10))]

    y_label = y_axis_label
    if y_label is None:
        ml0 = curves[0].mass_label.lower()
        if "%" in ml0 or all(float(c.mass.max()) < 200 for c in curves):
            y_label = "TG / %"
        else:
            y_label = curves[0].mass_label

    for i, c in enumerate(curves):
        color = palette[i]
        lab = c.source_name or f"series_{i + 1}"
        ax.plot(c.temperature_c, c.mass, color=color, lw=1.2, label=lab)

    t_all = np.concatenate([c.temperature_c for c in curves])
    m_all = np.concatenate([c.mass for c in curves])
    t_min, t_max = float(np.min(t_all)), float(np.max(t_all))
    m_min, m_max = float(np.min(m_all)), float(np.max(m_all))
    pad_y = max((m_max - m_min) * 0.08, 1.0)
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(m_min - pad_y, m_max + pad_y)

    x_span = max(t_max - t_min, 1.0)
    bracket_half = 0.018 * x_span

    for wi, win in enumerate(windows):
        lo = float(win["temp_min_c"])
        hi = float(win["temp_max_c"])
        if hi < t_min or lo > t_max:
            continue
        lo_c = max(lo, t_min)
        hi_c = min(hi, t_max)
        t_mid = (lo_c + hi_c) / 2.0

        for ci, c in enumerate(curves):
            color = palette[ci]
            m_lo = tga_mass_at_temp(c.temperature_c, c.mass, lo_c)
            m_hi = tga_mass_at_temp(c.temperature_c, c.mass, hi_c)
            delta = m_hi - m_lo
            dx = (ci - 0.5 * (len(curves) - 1)) * bracket_half * 0.9
            tm = t_mid + dx

            ax.plot([tm - bracket_half, tm], [m_lo, m_lo], color=color, lw=1.0, zorder=4)
            ax.plot([tm - bracket_half, tm], [m_hi, m_hi], color=color, lw=1.0, zorder=4)
            ax.annotate(
                "",
                xy=(tm, m_hi),
                xytext=(tm, m_lo),
                arrowprops={
                    "arrowstyle": "<->",
                    "color": color,
                    "lw": 1.0,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=5,
            )
            tx = tm + bracket_half * 1.1 + wi * bracket_half * 0.15
            ax.text(
                tx,
                (m_lo + m_hi) / 2.0,
                f"Mass change: {delta:+.2f} %",
                color=color,
                fontsize=7.5,
                va="center",
                ha="left",
                zorder=6,
            )

    for ci, c in enumerate(curves):
        color = palette[ci]
        t_last = float(c.temperature_c[-1])
        m_last = float(c.mass[-1])
        ax.plot([t_last], [m_last], marker="+", ms=9, mew=1.2, color=color, zorder=7)
        ax.annotate(
            f"Residual mass: {m_last:.2f} % ({t_last:.1f} °C)",
            xy=(t_last, m_last),
            xytext=(12, 10 + ci * 14),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            ha="left",
        )

    ax.set_title(title or "Thermogravimetric analysis (TG)")
    ax.set_xlabel("Temperature / °C")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.5)
    if len(curves) > 1:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_tga(curve: TGACurve) -> Figure:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(curve.temperature_c, curve.mass, color="C0", lw=1.2, label=curve.mass_label)
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel(curve.mass_label)
    ax1.set_title("TGA")
    if curve.dtg is not None:
        ax2 = ax1.twinx()
        ax2.plot(
            curve.temperature_c,
            curve.dtg,
            color="C1",
            lw=1.0,
            alpha=0.85,
            label="DTG (dm/dT)",
        )
        ax2.set_ylabel("DTG")
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right")
    fig.tight_layout()
    return fig


def figure_to_embed_html(fig: Figure, dpi: int = 120) -> str:
    """PNG fragment for HTML reports (no external JS)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f'<p><img src="data:image/png;base64,{b64}" alt="plot" /></p>'
