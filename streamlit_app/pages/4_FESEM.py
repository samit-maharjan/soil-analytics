"""FESEM: supervised inference and unsupervised clustering (optional [ml] extra)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from soil_analytics.paths import fesem_supervised_data_dir, models_dir, project_root

st.set_page_config(page_title="FESEM", layout="wide")
st.title("FESEM")
st.caption(
    "Supervised: load a model trained with scripts/train_fesem_supervised.py. "
    "Unsupervised: embeddings with a pretrained backbone (no labels required)."
)

try:
    from soil_analytics.ml.supervised import predict_images_from_bytes
    from soil_analytics.ml.unsupervised import run_unsupervised_pipeline
except ImportError as e:
    st.error(
        "FESEM ML dependencies are not installed. "
        "Install with: `pip install soil-analytics[ml]` "
        f"(import error: {e})"
    )
    st.stop()

default_model = models_dir() / "fesem_supervised"

tab_sup, tab_unsup = st.tabs(["Supervised inference", "Unsupervised analysis"])

with tab_sup:
    data_root = fesem_supervised_data_dir()
    st.markdown(
        f"Default model directory: `{default_model}`. "
        f"Supervised training data (ImageFolder): `{data_root}` — add class subfolders and images, then run "
        "`python scripts/train_fesem_supervised.py` (or pass `--data-dir` for another folder). "
        "See `data/fesem_supervised/README.md`."
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
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(list(probs.keys()), list(probs.values()), color="steelblue")
                ax.set_ylabel("Probability")
                ax.set_title("Class probabilities")
                plt.xticks(rotation=25, ha="right")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

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
            emb = out.get("embedding_2d")
            if emb is not None and len(out["names"]) >= 3:
                fig, ax = plt.subplots(figsize=(7, 5))
                sc = ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=out["cluster_labels"],
                    cmap="viridis",
                    s=60,
                    alpha=0.85,
                )
                for i, name in enumerate(out["names"]):
                    ax.annotate(name, (emb[i, 0], emb[i, 1]), fontsize=7, alpha=0.8)
                ax.set_title("2D embedding (PCA of standardized features)")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                fig.colorbar(sc, ax=ax, label="cluster")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            elif len(out["names"]) < 3:
                st.info("PCA scatter needs at least 3 images; showing table only.")

st.markdown(f"Project root resolved as: `{project_root()}`")
