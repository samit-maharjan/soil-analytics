"""TGA upload: TG curves, reference temperature windows, and per-window summaries."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

from soil_analytics.parsers import parse_tga_csv
from soil_analytics.paths import reference_config_dir
from soil_analytics.plots import figure_to_embed_html, plot_tga_multi_reference, tga_mass_at_temp
from soil_analytics.report import build_html_report

st.set_page_config(page_title="TGA", layout="wide")
st.title("TGA / TG")
st.caption(
    "Upload one or more CSV exports (plain columns or NETZSCH-style). "
    "Only temperature and mass (TG %) are used for the overlay plot. "
    "Reference windows and phase notes come from ``config/reference_ranges/tga_windows.yaml``."
)

cfg_path = reference_config_dir() / "tga_windows.yaml"
with open(cfg_path, encoding="utf-8") as f:
    tga_cfg = yaml.safe_load(f)
windows: list = list(tga_cfg.get("windows", []))

ups = st.file_uploader(
    "TGA CSV (one or more files)",
    type=["csv", "txt"],
    accept_multiple_files=True,
    key="tga_csv",
)
if not ups:
    st.info("Upload at least one file to begin.")
    st.stop()

curves = []
parse_errors: list[tuple[str, str]] = []
for up in ups:
    try:
        curves.append(
            parse_tga_csv(up.getvalue(), source_name=up.name, include_dtg=False),
        )
    except Exception as e:
        parse_errors.append((up.name, str(e)))

for name, msg in parse_errors:
    st.error(f"{name}: could not parse ({msg}).")

if not curves:
    st.stop()

fig = plot_tga_multi_reference(curves, windows)
plot_html = figure_to_embed_html(fig)
st.pyplot(fig)
plt.close(fig)

summary_rows: list[dict[str, str | float]] = []
for c in curves:
    for win in windows:
        lo = float(win["temp_min_c"])
        hi = float(win["temp_max_c"])
        m_lo = tga_mass_at_temp(c.temperature_c, c.mass, lo)
        m_hi = tga_mass_at_temp(c.temperature_c, c.mass, hi)
        delta = m_hi - m_lo
        notes = win.get("notes") or ""
        summary_rows.append(
            {
                "Range (°C)": f"{lo:g}–{hi:g}",
                "Phase / compound": win.get("label", ""),
                "ΔTG (% pts)": round(delta, 4),
                "Inference (reference)": " ".join(str(notes).split()),
            },
        )

st.subheader("Reference windows and mass change")
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

html = build_html_report(
    "TGA report",
    [],
    figure_html=plot_html,
    inference_rows=[{str(k): str(v) for k, v in row.items()} for row in summary_rows],
    inference_heading="Temperature windows (reference assignments)",
    inference_intro=(
        "ΔTG is the change in the uploaded TG trace between the window bounds "
        "(linear interpolation), in the same units as the mass column "
        "(typically percentage points for TG %)."
    ),
)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name="tga_report.html",
    mime="text/html",
)
