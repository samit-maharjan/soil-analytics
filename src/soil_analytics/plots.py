"""Matplotlib figures for FTIR, XRD, TGA (lighter than Plotly for this app)."""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from soil_analytics.schemas import FTIRSeries, TGACurve, XRDPattern


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
