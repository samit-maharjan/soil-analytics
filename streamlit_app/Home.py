"""Soil Analytics Platform — main entry."""

import streamlit as st

st.set_page_config(
    page_title="Soil Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Soil Analytics Platform")
st.markdown(
    """
Use the sidebar to open **FTIR**, **XRD**, **TGA**, **FESEM**, or **About**.

- **FTIR / XRD / TGA**: upload CSV, view plots, run literature-based range checks, download an HTML report.
- **FESEM**: reference table, **morphology → phase MCQs**, and short **soil / microstructure interpretation** text; optional download; nothing is stored by the app.

Run from the repository root: `streamlit run streamlit_app/Home.py`
"""
)
