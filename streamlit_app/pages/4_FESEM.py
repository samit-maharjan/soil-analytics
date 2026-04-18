"""FESEM: supervised inference and unsupervised clustering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from soil_analytics.ml.supervised import predict_images_from_bytes
from soil_analytics.ml.unsupervised import run_unsupervised_pipeline
from soil_analytics.paths import models_dir, project_root

st.set_page_config(page_title="FESEM", layout="wide")
st.title("FESEM")
st.caption(
    "Supervised: load a model trained with scripts/train_fesem_supervised.py. "
    "Unsupervised: embeddings with a pretrained backbone (no labels required)."
)

default_model = models_dir() / "fesem_supervised"

tab_sup, tab_unsup = st.tabs(["Supervised inference", "Unsupervised analysis"])

with tab_sup:
    st.markdown(
        f"Default model directory: `{default_model}`. "
        "Train with: `python scripts/train_fesem_supervised.py --data-dir /path/to/ImageFolder`"
    )
    model_path = Path(
        st.text_input("Model directory", value=str(default_model), key="fesem_model_dir")
    )
    meta = model_path / "meta.json"
    if not meta.exists():
        st.warning("No trained model found (missing meta.json). Train a model or set the path above.")
    up_sup = st.file_uploader(
        "FESEM images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="fesem_sup",
    )
    if up_sup and meta.exists():
        files = [(u.name, u.getvalue()) for u in up_sup]
        if st.button("Run supervised inference", key="run_sup"):
            with st.spinner("Running inference…"):
                preds = predict_images_from_bytes(files, model_path)
            df = pd.DataFrame(preds)
            st.dataframe(df, use_container_width=True)
            for row in preds:
                st.subheader(row["path"])
                probs = row["probabilities"]
                fig = go.Figure(go.Bar(x=list(probs.keys()), y=list(probs.values())))
                fig.update_layout(title="Class probabilities", height=320)
                st.plotly_chart(fig, use_container_width=True)

with tab_unsup:
    backbone = st.selectbox("Backbone (ImageNet pretrained)", ["resnet18", "efficientnet_b0"], index=0)
    up_un = st.file_uploader(
        "FESEM images (batch)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="fesem_unsup",
    )
    if up_un and st.button("Run unsupervised pipeline", key="run_unsup"):
        files = [(u.name, u.getvalue()) for u in up_un]
        with st.spinner("Extracting embeddings and clustering (first run may download weights)…"):
            out = run_unsupervised_pipeline(files, backbone=backbone)
        if out.get("error"):
            st.error(out["error"])
        else:
            df = pd.DataFrame(
                {
                    "file": out["names"],
                    "cluster": out["cluster_labels"],
                    "anomaly_score": out["anomaly_score"],
                }
            )
            st.dataframe(df, use_container_width=True)
            umap_coords = out.get("umap")
            if umap_coords is not None and len(out["names"]) >= 5:
                fig = go.Figure(
                    go.Scatter(
                        x=umap_coords[:, 0],
                        y=umap_coords[:, 1],
                        mode="markers+text",
                        text=out["names"],
                        marker=dict(size=10, color=out["cluster_labels"], colorscale="Viridis"),
                    )
                )
                fig.update_layout(title="UMAP of embeddings", height=500)
                st.plotly_chart(fig, use_container_width=True)
            elif len(out["names"]) < 5:
                st.info("UMAP needs at least 5 images; showing table only.")

st.markdown(f"Project root resolved as: `{project_root()}`")
