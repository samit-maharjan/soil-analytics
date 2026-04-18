"""XRD upload and reference checks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from soil_analytics.parsers import parse_xrd_csv
from soil_analytics.paths import reference_config_dir
from soil_analytics.plots import figure_to_embed_html, plot_xrd
from soil_analytics.reference_checks import check_xrd
from soil_analytics.report import build_html_report

st.set_page_config(page_title="XRD", layout="wide")
st.title("XRD")
st.caption(
    "Upload CSV with 2θ (degrees) and intensity. Assumptions in config (e.g. Cu Kα) apply to qualitative flags only."
)

up = st.file_uploader("XRD CSV", type=["csv", "txt"], key="xrd_csv")
if up is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    pattern = parse_xrd_csv(up.getvalue(), source_name=up.name)
except Exception as e:
    st.error(f"Could not parse file: {e}")
    st.stop()

fig = plot_xrd(pattern)
plot_html = figure_to_embed_html(fig)
st.pyplot(fig)
plt.close(fig)

cfg = reference_config_dir() / "xrd_peaks.yaml"
results = check_xrd(pattern, config_path=cfg)
df = pd.DataFrame(
    [{"id": r.check_id, "label": r.label, "status": r.status, "message": r.message} for r in results]
)
st.subheader("Reference checks")
st.dataframe(df, use_container_width=True)

html = build_html_report(
    "XRD report",
    [
        (
            "Summary",
            "Window-based screening for common soil minerals — not a substitute for full clay protocols.",
            results,
        )
    ],
    figure_html=plot_html,
)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name="xrd_report.html",
    mime="text/html",
)
