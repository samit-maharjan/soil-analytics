"""Soil Analytics — main entry (production-style landing)."""

import streamlit as st

from soil_analytics.streamlit_readability import inject_readability_css

st.set_page_config(
    page_title="Soil Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_readability_css()

# Logo and title side by side
logo_col, title_col = st.columns([0.07, 0.93], gap="small")
with logo_col:
    st.write("")  # Add some vertical spacing
    st.image("assets/logo.svg", width=60)
with title_col:
    st.markdown(
        """
        <h1 style="margin-top: 0.15rem; margin-bottom: 0; margin-left: -1rem;">Soil Analytics</h1>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "Interpret **laboratory exports** of vibrational, diffraction, and thermal data against reference "
    "windows. Use a **structured FESEM guide** to connect texture to likely phase."
)

st.divider()
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**FTIR**  \nGroup frequencies and band inferences from your export.")
    st.page_link("pages/1_FTIR.py", label="Open FTIR", icon="📡", use_container_width=True)
with c2:
    st.markdown("**XRD**  \n2θ–intensity patterns and phase hints from 2θ windows.")
    st.page_link("pages/2_XRD.py", label="Open XRD", icon="📈", use_container_width=True)
with c3:
    st.markdown("**TGA**  \nMass vs temperature with reference window summaries.")
    st.page_link("pages/3_TGA.py", label="Open TGA", icon="🌡️", use_container_width=True)
with c4:
    st.markdown("**FESEM**  \nHabit questions that narrow a suggested phase.")
    st.page_link("pages/4_FESEM.py", label="Open FESEM", icon="🔬", use_container_width=True)