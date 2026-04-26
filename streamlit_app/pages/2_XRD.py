"""XRD upload: ASC 2θ vs intensity, offset multi-file plot, phase hints from 2θ windows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from soil_analytics.parsers import parse_xrd_bytes
from soil_analytics.plots import figure_to_embed_html, plot_xrd_multi
from soil_analytics.schemas import XRDPattern
from soil_analytics.report import build_html_report
from soil_analytics.xrd_phases import find_phase_hits, merge_xrd_phase_rows, xrd_manual_two_theta_rows
from soil_analytics.streamlit_readability import inject_readability_css
from soil_analytics.streamlit_tables import scrollable_dataframe

st.set_page_config(page_title="XRD", layout="wide")
inject_readability_css()
st.title("XRD")
st.markdown(
    """
**X-ray diffraction (XRD):** 2θ vs **intensity**. Maxima in reference 2θ windows are matched for **qualitative** comments.

Use **manual 2θ** below, or **upload `.asc`** (two columns) for one or more patterns.
"""
)
st.divider()

st.subheader("Manual 2θ value")
st.caption("Enter a 2θ value in degrees. Any laboratory reference window that includes this angle is listed with its short note.")
c1, c2 = st.columns([2.2, 1], vertical_alignment="bottom", gap="medium")
with c1:
    tt_man = st.number_input("2θ (degrees)", value=29.0, format="%.3f", key="xrd_tt_manual")
with c2:
    do_man = st.button("Show window match", type="primary", key="xrd_man_btn", use_container_width=True)
if do_man:
    mrows = xrd_manual_two_theta_rows(tt_man)
    scrollable_dataframe(mrows)
st.markdown("  \n  ")

st.subheader("Upload data")
up = st.file_uploader(
    "Data files (.asc, 2θ vs intensity)",
    type=["asc"],
    accept_multiple_files=True,
    key="xrd_files",
    help="Two columns: 2θ and intensity (or counts).",
)
st.markdown("")

if not up:
    st.info("Upload one or more `.asc` files to plot and summarize phase windows.")
    st.stop()

parsed: list[tuple[str, XRDPattern]] = []
for u in up:
    try:
        p = parse_xrd_bytes(u.getvalue(), source_name=u.name, filename=u.name)
        parsed.append((u.name, p))
    except Exception as e:
        st.error(f"**{u.name}** — {e}")

if not parsed:
    st.warning("No files could be parsed. Check two columns 2θ and y.")
    st.stop()

st.subheader("2θ – intensity")
labels = [n for n, _ in parsed]
patterns = [p for _, p in parsed]
hits_per = [find_phase_hits(p) for p in patterns]
multi = len(parsed) > 1
fig = plot_xrd_multi(
    patterns,
    labels,
    hits_per,
    title="XRD" if multi else labels[0],
)
plot_html = figure_to_embed_html(fig)
st.pyplot(fig)
plt.close(fig)

st.divider()
merged = merge_xrd_phase_rows(hits_per)
st.subheader("Phase summary (from file maxima in windows)")
st.caption("One row per phase: 2θ values = scan maxima in windows, combined across files.")
scrollable_dataframe(merged)
st.markdown("")

html = build_html_report(
    "XRD report",
    [
        (
            "Summary",
            "Qualitative labels from the lab 2θ table (not a substitute for Rietveld or clay packages).",
            None,
        )
    ],
    figure_html=plot_html,
    inference_rows=merged,
    inference_heading="Phase summary",
    inference_intro="Listed 2θ = scan maxima in each window, combined across files.",
)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name="xrd_report.html",
    mime="text/html",
)
