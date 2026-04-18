"""TGA upload and reference checks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from soil_analytics.parsers import parse_tga_csv
from soil_analytics.paths import reference_config_dir
from soil_analytics.plots import figure_to_embed_html, plot_tga
from soil_analytics.reference_checks import check_tga
from soil_analytics.report import build_html_report

st.set_page_config(page_title="TGA", layout="wide")
st.title("TGA / DTG")
st.caption(
    "Upload CSV with temperature (°C) and mass (or mass fraction). DTG is computed as dm/dT if not provided."
)

up = st.file_uploader("TGA CSV", type=["csv", "txt"], key="tga_csv")
if up is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    curve = parse_tga_csv(up.getvalue(), source_name=up.name)
except Exception as e:
    st.error(f"Could not parse file: {e}")
    st.stop()

fig = plot_tga(curve)
plot_html = figure_to_embed_html(fig)
st.pyplot(fig)
plt.close(fig)

cfg = reference_config_dir() / "tga_windows.yaml"
results = check_tga(curve, config_path=cfg)
df = pd.DataFrame(
    [{"id": r.check_id, "label": r.label, "status": r.status, "message": r.message} for r in results]
)
st.subheader("Reference checks")
st.dataframe(df, use_container_width=True)

html = build_html_report(
    "TGA report",
    [
        (
            "Summary",
            "Approximate mass-loss summaries per configured temperature windows — compare only with matching methods.",
            results,
        )
    ],
    figure_html=plot_html,
)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name="tga_report.html",
    mime="text/html",
)
