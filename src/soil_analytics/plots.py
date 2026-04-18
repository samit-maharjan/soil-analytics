"""Plotly figures for FTIR, XRD, TGA."""

from __future__ import annotations

import plotly.graph_objects as go

from soil_analytics.schemas import FTIRSeries, TGACurve, XRDPattern


def plot_ftir(series: FTIRSeries) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.wavenumber_cm1,
            y=series.y,
            mode="lines",
            name=series.y_label,
        )
    )
    fig.update_layout(
        title="FTIR",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title=series.y_label,
        xaxis=dict(autorange="reversed"),
        height=450,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


def plot_xrd(pattern: XRDPattern) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pattern.two_theta_deg,
            y=pattern.intensity,
            mode="lines",
            name="Intensity",
        )
    )
    fig.update_layout(
        title="XRD",
        xaxis_title="2θ (degrees)",
        yaxis_title="Intensity",
        height=450,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


def plot_tga(curve: TGACurve) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve.temperature_c,
            y=curve.mass,
            mode="lines",
            name=curve.mass_label,
            yaxis="y",
        )
    )
    if curve.dtg is not None:
        fig.add_trace(
            go.Scatter(
                x=curve.temperature_c,
                y=curve.dtg,
                mode="lines",
                name="DTG (dm/dT)",
                yaxis="y2",
            )
        )
    layout_updates: dict = dict(
        title="TGA",
        xaxis_title="Temperature (°C)",
        height=450,
        margin=dict(l=50, r=60, t=50, b=50),
    )
    if curve.dtg is not None:
        layout_updates["yaxis"] = dict(title=curve.mass_label)
        layout_updates["yaxis2"] = dict(title="DTG", overlaying="y", side="right")
    else:
        layout_updates["yaxis_title"] = curve.mass_label
    fig.update_layout(**layout_updates)
    return fig


def figure_to_embed_html(fig: go.Figure) -> str:
    """Fragment HTML for embedding in a larger report."""
    return fig.to_html(full_html=False, include_plotlyjs="cdn")
