"""XRD upload: ASC / CSV 2θ vs intensity, stacked multi-sample plot, phase table (P / M / C)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from soil_analytics.parsers import parse_xrd_bytes
from soil_analytics.plots import figure_to_embed_html, plot_xrd_stacked
from soil_analytics.schemas import XRDPattern
from soil_analytics.report import build_html_report
from soil_analytics.xrd_phases import find_phase_hits, merge_xrd_phase_rows

st.set_page_config(page_title="XRD", layout="wide")
st.title("XRD")
st.caption(
    "Upload **ASC** (two-column 2θ vs intensity), **CSV**, or **TXT** tables. "
    "Multiple files stack vertically with distinct colors; phase letters use literature 2θ windows (P / M / C)."
)

up = st.file_uploader(
    "XRD files (ASC, CSV, or TXT)",
    type=["asc", "csv", "txt"],
    accept_multiple_files=True,
    key="xrd_files",
)
if not up:
    st.info("Upload one or more files to begin.")
    st.stop()

parsed: list[tuple[str, XRDPattern]] = []
for u in up:
    try:
        p = parse_xrd_bytes(u.getvalue(), source_name=u.name, filename=u.name)
        parsed.append((u.name, p))
    except Exception as e:
        st.error(f"**{u.name}**: {e}")

if not parsed:
    st.warning("No files could be parsed.")
    st.stop()

labels = [n for n, _ in parsed]
patterns = [p for _, p in parsed]
hits_per = [find_phase_hits(p) for p in patterns]

multi = len(parsed) > 1
fig = plot_xrd_stacked(
    patterns,
    labels,
    hits_per,
    title="XRD (stacked)" if multi else labels[0],
)
plot_html = figure_to_embed_html(fig)
st.pyplot(fig)
plt.close(fig)

merged = merge_xrd_phase_rows(hits_per)
st.subheader("Phase summary")
st.dataframe(merged, use_container_width=True, hide_index=True)

html = build_html_report(
    "XRD report",
    [
        (
            "Summary",
            "Qualitative labels from 2θ windows for Portlandite (P), Mullite (M), and Calcite (C); "
            "not a substitute for full Rietveld or clay protocols.",
            None,
        )
    ],
    figure_html=plot_html,
    inference_rows=merged,
    inference_heading="Phase summary",
    inference_intro=(
        "Listed 2θ values are scan maxima within each reference window, combined across uploaded samples."
    ),
)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name="xrd_report.html",
    mime="text/html",
)
