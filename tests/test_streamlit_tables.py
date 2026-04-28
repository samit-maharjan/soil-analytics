"""Inference table HTML (no Streamlit runtime)."""

import pandas as pd

from lime_analytics.streamlit_tables import _build_wrapped_table_html, _cell_display


def test_cell_display_none_and_nan() -> None:
    assert _cell_display(None) == "—"
    assert _cell_display(float("nan")) == "—"


def test_wrapped_table_escapes_html() -> None:
    df = pd.DataFrame([{"Band": "Test", "Inference": "<script>alert(1)</script> long text " * 5}])
    h = _build_wrapped_table_html(df)
    assert "<script>" not in h
    assert "&lt;" in h
    assert "sa-inf-table" in h
