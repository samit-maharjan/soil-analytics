"""Streamlit table display: true text wrap for long “Inference” cells (not Glide’s clipped single line)."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

# Inline styles so text wraps even if Streamlit’s HTML sanitizer strips `class` on table cells
# (global CSS in `inject_readability_css` is still applied when classes are kept).
_WRAP_TABLE = (
    "width:100%;table-layout:fixed;border-collapse:collapse;font-size:0.95rem;margin:0.35rem 0 0.9rem 0;"
)
_WRAP_DIV = "width:100%;max-width:100%;overflow-x:hidden;"
_WRAP_CELL = (
    "white-space:normal;word-wrap:break-word;overflow-wrap:anywhere;word-break:break-word;"
    "min-width:0;vertical-align:top;text-align:left;line-height:1.45;padding:0.5rem 0.65rem;"
    "border:1px solid rgba(128, 128, 128, 0.4);"
)
_WRAP_TH = _WRAP_CELL + "font-weight:600;background:rgba(128, 128, 128, 0.14);"


def _cell_display(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and pd.isna(v):
        return "—"
    s = str(v).strip()
    return s if s else "—"


def _build_wrapped_table_html(data: pd.DataFrame) -> str:
    """HTML table with class + inline wrap styles; cells escaped."""
    ths = "".join(
        f'<th class="sa-inf-th" style="{_WRAP_TH}">{html.escape(c, quote=True)}</th>' for c in data.columns
    )
    body_rows: list[str] = []
    for _, row in data.iterrows():
        tds = "".join(
            f'<td class="sa-inf-td" style="{_WRAP_CELL}">{html.escape(_cell_display(v), quote=True)}</td>'
            for v in row
        )
        body_rows.append(f"<tr>{tds}</tr>")
    return (
        f'<div class="sa-inf-table-wrap" data-testid="inference-wrapped-table" style="{_WRAP_DIV}">'
        f'<table class="sa-inf-table" style="{_WRAP_TABLE}">'
        f"<thead><tr>{ths}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def scrollable_dataframe(
    data: pd.DataFrame | list[dict],
    *,
    hide_index: bool = True,
    use_container_width: bool = True,  # noqa: ARG001 — API compatibility
) -> None:
    """
    Show tabular data with **wrapped** text in cells (inference, notes, etc.).

    Streamlit’s ``st.dataframe`` uses a canvas grid that often **clips** long strings on one
    line; this uses an HTML table + CSS in ``inject_readability_css`` so long text wraps.
    """
    if isinstance(data, list):
        data = pd.DataFrame(data)
    if data.empty:
        st.caption("No rows to show.")
        return
    if not hide_index:
        data = data.reset_index()
    st.html(_build_wrapped_table_html(data), width="stretch")
