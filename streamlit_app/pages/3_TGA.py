"""TGA upload: TG curves, reference temperature windows, and per-window summaries."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

from lime_analytics.parsers import parse_tga_csv
from lime_analytics.paths import reference_config_dir
from lime_analytics.plots import figure_to_embed_html, plot_tga_multi_reference, tga_mass_at_temp
from lime_analytics.reference_checks import tga_range_display_str, tga_window_manual_row
from lime_analytics.report import build_html_report
from lime_analytics.streamlit_readability import inject_readability_css
from lime_analytics.streamlit_tables import scrollable_dataframe

st.set_page_config(page_title="TGA", layout="wide")
inject_readability_css()
st.title("TGA / TG")
st.markdown(
    """
**Thermogravimetric analysis (TGA):** mass (or normalized TG) vs **temperature**.

**Manual** mode: pick a **reference range** and enter a **ΔTG** value. **File** mode: upload a curve and
compute ΔTG between the bounds of each window in the table.
"""
)
st.divider()

cfg_path = reference_config_dir() / "tga_windows.yaml"
with open(cfg_path, encoding="utf-8") as f:
    tga_cfg = yaml.safe_load(f)
windows: list = list(tga_cfg.get("windows", []))

# Labels for selectbox: unique display strings
def _win_label(i: int, w: dict) -> str:
    r = tga_range_display_str(w)
    lab = (w.get("label") or w.get("id") or "—")[:72]
    return f"{i + 1}. {r} °C — {lab}"


st.subheader("Manual range and ΔTG")
st.caption("Choose a temperature **window** from the reference set, enter your **ΔTG** in the same units as your data (e.g. percentage points), then get the **reference** inference for that range.")
if not windows:
    st.warning("No temperature windows in `tga_windows.yaml` — manual inference is unavailable.")
else:
    c1, c2, c3 = st.columns([1.4, 0.85, 0.5], vertical_alignment="bottom", gap="small")
    with c1:
        idx = st.selectbox("Reference range", list(range(len(windows))), format_func=lambda i: _win_label(i, windows[i]), key="tga_sel_win")
    with c2:
        d_manual = st.number_input("Your ΔTG", value=0.0, format="%.6f", help="Value you assign for this range", key="tga_d")
    with c3:
        do_tga_man = st.button("Show inference", type="primary", key="tga_man", use_container_width=True)
    if do_tga_man:
        trows = tga_window_manual_row(windows[idx], d_manual)
        scrollable_dataframe(trows)
st.markdown("  \n  ")

st.subheader("Upload TGA / TG data")
ups = st.file_uploader(
    "TG / TGA curves (.csv)",
    type=["csv"],
    accept_multiple_files=True,
    key="tga_csv",
    help="Column names parsed flexibly; needs temperature and mass (or TG %).",
)
st.markdown("")

if not ups:
    st.info("Add at least one `.csv` to plot the curve and tabulate per-window mass change.")
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
    st.error(f"**{name}** — {msg}")

if not curves:
    st.stop()

st.subheader("Curves and reference windows")
fig = plot_tga_multi_reference(curves, windows)
plot_html = figure_to_embed_html(fig)
st.pyplot(fig)
plt.close(fig)

st.divider()
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
                "Range (°C)": tga_range_display_str(win),
                "Phase / compound": win.get("label", ""),
                "ΔTG (% pts)": round(delta, 4),
                "Inference (reference)": " ".join(str(notes).split()),
            },
        )

st.subheader("Per-window change (from file)")
st.caption("ΔTG = change in the trace between the window bounds; units follow your mass column.")
scrollable_dataframe(summary_rows)
st.markdown("")

html = build_html_report(
    "TGA report",
    [],
    figure_html=plot_html,
    inference_rows=[{str(k): str(v) for k, v in row.items()} for row in summary_rows],
    inference_heading="Temperature windows (reference assignments)",
    inference_intro="ΔTG is interpolated on the uploaded TG; typically % points for TG %.",
)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name="tga_report.html",
    mime="text/html",
)
