"""Methodology and limitations."""

import streamlit as st

st.set_page_config(page_title="About", layout="wide")
st.title("About this app")

st.markdown(
    """
### FTIR, XRD, TGA

- Reference bands and windows live under `config/reference_ranges/` as YAML with short citations.
- Results are **indicators** for exploration and QC, not certified quantification.
- **XRD**: mineral identification from bulk patterns is often **ambiguous** without oriented clay mounts and treatments.
- **TGA**: mass-loss steps depend on **heating rate, atmosphere, and crucible** — always record and match your lab method.

### FESEM

- **Supervised** models are trained on your chosen labeled image folder or transfer from public data; treat outputs as **assistive** until validated on your soil images.
- **Unsupervised** clustering and anomaly scores help prioritize images for expert review; they are not ground truth labels.

### Running locally

```text
pip install -e ".[dev]"
streamlit run streamlit_app/Home.py
```

Ensure you run the command from the repository root so `config/` resolves correctly.
"""
)
