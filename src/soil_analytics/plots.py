"""Matplotlib figures for FTIR, XRD, TGA (lighter than Plotly for this app)."""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from soil_analytics.schemas import FTIRSeries, TGACurve, XRDPattern


def plot_ftir(series: FTIRSeries) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(series.wavenumber_cm1, series.y, color="C0", lw=1.2)
    ax.set_title("FTIR")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel(series.y_label)
    ax.invert_xaxis()
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
