"""FTIR upload and reference checks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from soil_analytics.parsers import parse_ftir_csv
from soil_analytics.paths import reference_config_dir
from soil_analytics.plots import figure_to_embed_html, plot_ftir
from soil_analytics.reference_checks import check_ftir, ftir_inference_rows
from soil_analytics.report import build_html_report

st.set_page_config(page_title="FTIR", layout="wide")
st.title("FTIR")
st.caption(
    "Upload a **CSV** (wavenumber + absorbance or transmittance columns) or a **JCAMP-style text** file "
    "(e.g. `##DATA TYPE=INFRARED SPECTRUM`, `##YUNITS=%T`, then wavenumber and transmittance columns)."
)

up = st.file_uploader("FTIR file (CSV or JCAMP / text spectrum)", type=["csv", "txt"], key="ftir_csv")
if up is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    series = parse_ftir_csv(up.getvalue(), source_name=up.name)
except Exception as e:
    st.error(f"Could not parse file: {e}")
    st.stop()

fig = plot_ftir(series, title=up.name)
plot_html = figure_to_embed_html(fig)
st.pyplot(fig)
plt.close(fig)

cfg = reference_config_dir() / "ftir_bands.yaml"
results = check_ftir(series, config_path=cfg)
inference_rows = ftir_inference_rows(results)
inf_df = pd.DataFrame(inference_rows)

st.subheader("Band inferences (reference ranges)")
st.markdown(
    "Peak wavenumbers are extrema within each band window (minimum transmittance / maximum absorbance). "
    "Inferences follow the `notes` field in `config/reference_ranges/ftir_bands.yaml`."
)
st.dataframe(inf_df, use_container_width=True, hide_index=True)

st.subheader("Reference check status")
qc_df = pd.DataFrame(
    [{"id": r.check_id, "label": r.label, "status": r.status, "message": r.message} for r in results]
)
st.dataframe(qc_df, use_container_width=True)

html = build_html_report(
    "FTIR report",
    [
        (
            "Summary",
            "Qualitative comparison to configured band windows (see config/reference_ranges/ftir_bands.yaml).",
            results,
        )
    ],
    figure_html=plot_html,
    inference_rows=inference_rows,
)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name="ftir_report.html",
    mime="text/html",
)
