"""FTIR upload and band inference table."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from soil_analytics.parsers import parse_ftir_csv
from soil_analytics.paths import reference_config_dir
from soil_analytics.schemas import FTIRSeries
from soil_analytics.plots import figure_to_embed_html, plot_ftir_multi
from soil_analytics.reference_checks import check_ftir, ftir_manual_wavenumber_rows, ftir_merged_inference_rows
from soil_analytics.report import build_html_report
from soil_analytics.streamlit_readability import inject_readability_css
from soil_analytics.streamlit_tables import scrollable_dataframe

st.set_page_config(page_title="FTIR", layout="wide")
inject_readability_css()
st.title("FTIR")
st.markdown(
    """
**Fourier-transform infrared (FTIR)** records absorbance or transmittance vs **wavenumber** (cm⁻¹).

Use **manual wavenumber lookup** below (no file), or upload **`.txt`** spectra for overlays and per-band extrema.
"""
)
st.divider()

cfg = reference_config_dir() / "ftir_bands.yaml"

st.subheader("Manual peak wavenumber")
st.caption("Enter a peak position in cm⁻¹ to see which reference band it falls into and the inference text from the config.")
c_man1, c_man2 = st.columns([2.2, 1], vertical_alignment="bottom", gap="medium")
with c_man1:
    wn_manual = st.number_input("Peak wavenumber (cm⁻¹)", value=1000.0, format="%.2f", key="ftir_wn_manual")
with c_man2:
    do_manual = st.button("Show band inference", type="primary", key="ftir_manual_btn", use_container_width=True)
if do_manual:
    man_rows = ftir_manual_wavenumber_rows(wn_manual, cfg)
    scrollable_dataframe(man_rows)
st.markdown("  \n  ")

st.subheader("Upload spectra")
up = st.file_uploader(
    "Spectrum files (.txt)",
    type=["txt"],
    accept_multiple_files=True,
    key="ftir_files",
    help="Plain text: tab/space-separated, or JCAMP-DX style.",
)
st.markdown("")

if not up:
    st.info("Upload one or more `.txt` files to plot overlays and per-window extrema.")
    st.stop()

parsed: list[tuple[str, FTIRSeries]] = []
for u in up:
    try:
        s = parse_ftir_csv(u.getvalue(), source_name=u.name)
        parsed.append((u.name, s))
    except Exception as e:
        st.error(f"**{u.name}** — {e}")

if not parsed:
    st.warning("No files could be parsed. Check the format and column order.")
    st.stop()

st.subheader("Overlay")
series_list = [s for _, s in parsed]
labels = [n for n, _ in parsed]
multi = len(parsed) > 1
fig = plot_ftir_multi(series_list, labels=labels, title="FTIR overlay" if multi else labels[0])
plot_html = figure_to_embed_html(fig)
st.pyplot(fig)
plt.close(fig)

st.divider()
all_results = [check_ftir(s, config_path=cfg) for _, s in parsed]
merged_inference = ftir_merged_inference_rows(all_results)
st.subheader("Band inferences (from uploaded spectra)")
st.caption("Peaks = extrema within each window")
scrollable_dataframe(merged_inference)
st.markdown("")

html = build_html_report(
    "FTIR report",
    [
        (
            "Summary",
            "Qualitative comparison to configured band windows",
            None,
        )
    ],
    figure_html=plot_html,
    inference_rows=merged_inference,
)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name="ftir_report.html",
    mime="text/html",
)
