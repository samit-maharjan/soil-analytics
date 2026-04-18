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

- **Supervised** models are trained on your labeled image folder or transfer from public data; treat outputs as **assistive** until validated on your soil images.
- **Unsupervised** clustering and anomaly scores help prioritize images for expert review; they are not ground truth labels.
- FESEM requires the optional ML stack: `pip install soil-analytics[ml]`

### Running locally

```text
pip install -e ".[dev]"
# FTIR/XRD/TGA only: no extras needed.
# With FESEM: pip install -e ".[dev,ml]"
streamlit run streamlit_app/Home.py
```

Ensure you run the command from the repository root so `config/` resolves correctly.

### Optional: Kaggle downloads

```text
pip install soil-analytics[kaggle]
python scripts/download_kaggle_dataset.py username/dataset-name
```
"""
)
