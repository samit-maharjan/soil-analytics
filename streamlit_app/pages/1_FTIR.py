"""FTIR upload and reference checks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from soil_analytics.parsers import parse_ftir_csv
from soil_analytics.paths import reference_config_dir
from soil_analytics.schemas import FTIRSeries
from soil_analytics.plots import figure_to_embed_html, plot_ftir_multi
from soil_analytics.reference_checks import check_ftir, ftir_merged_inference_rows
from soil_analytics.report import build_html_report

st.set_page_config(page_title="FTIR", layout="wide")
st.title("FTIR")
st.caption(
    "Upload one or more **CSV** or **JCAMP-style text** files (wavenumber vs absorbance/transmittance). "
    "Spectra are overlaid with distinct colors; band tables combine all samples."
)

up = st.file_uploader(
    "FTIR files (CSV or JCAMP / text spectrum)",
    type=["csv", "txt"],
    accept_multiple_files=True,
    key="ftir_csv",
)
if not up:
    st.info("Upload one or more files to begin.")
    st.stop()

parsed: list[tuple[str, FTIRSeries]] = []
for u in up:
    try:
        s = parse_ftir_csv(u.getvalue(), source_name=u.name)
        parsed.append((u.name, s))
    except Exception as e:
        st.error(f"**{u.name}**: {e}")

if not parsed:
    st.warning("No files could be parsed.")
    st.stop()

series_list = [s for _, s in parsed]
labels = [n for n, _ in parsed]
multi = len(parsed) > 1

fig = plot_ftir_multi(series_list, labels=labels, title="FTIR overlay" if multi else labels[0])
plot_html = figure_to_embed_html(fig)
st.pyplot(fig)
plt.close(fig)

cfg = reference_config_dir() / "ftir_bands.yaml"
all_results = [check_ftir(s, config_path=cfg) for _, s in parsed]
merged_inference = ftir_merged_inference_rows(all_results)

merged_qc: list[dict[str, str]] = []
for name, results in zip([n for n, _ in parsed], all_results, strict=True):
    for r in results:
        qc_row: dict[str, str] = {
            "id": r.check_id,
            "label": r.label,
            "status": r.status,
            "message": r.message,
        }
        if multi:
            qc_row = {"Sample": name, **qc_row}
        merged_qc.append(qc_row)

inf_df = pd.DataFrame(merged_inference)
st.subheader("Band inferences (reference ranges)")
st.markdown(
    "Peak wavenumbers are extrema within each band window (minimum transmittance / maximum absorbance). "
    "Inferences follow the `notes` field in `config/reference_ranges/ftir_bands.yaml`."
)
st.dataframe(inf_df, use_container_width=True, hide_index=True)

st.subheader("Reference check status")
st.dataframe(pd.DataFrame(merged_qc), use_container_width=True)

html = build_html_report(
    "FTIR report",
    [
        (
            "Summary",
            "Qualitative comparison to configured band windows (see config/reference_ranges/ftir_bands.yaml).",
            None,
        )
    ],
    figure_html=plot_html,
    inference_rows=merged_inference,
    qc_rows=merged_qc,
)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name="ftir_report.html",
    mime="text/html",
)
