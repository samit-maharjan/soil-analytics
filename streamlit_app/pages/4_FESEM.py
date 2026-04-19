"""FESEM: supervised inference and unsupervised clustering (optional [ml] extra)."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

from soil_analytics.paths import (
    fesem_supervised_data_dir,
    models_dir,
    project_root,
    reference_config_dir,
)

st.set_page_config(page_title="FESEM", layout="wide")
st.title("FESEM")
st.caption(
    "Phase morphology table below uses only config YAML. "
    "Supervised / unsupervised tabs need `pip install soil-analytics[ml]`: "
    "load a trained model or run clustering on uploads."
)

_remarks_path = reference_config_dir() / "fesem_remarks.yaml"
if _remarks_path.is_file():
    with open(_remarks_path, encoding="utf-8") as f:
        _fesem_ref = yaml.safe_load(f) or {}
    _rows = _fesem_ref.get("phases", [])
    if _rows:
        with st.expander(
            "Phase morphology reference (from config/reference_ranges/fesem_remarks.yaml)",
            expanded=False,
        ):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Phase": r.get("label", r.get("id", "")),
                            "Formula": r.get("chemical_formula") or "—",
                            "Morphology": r.get("morphology", ""),
                            "Notes": r.get("notes", ""),
                        }
                        for r in _rows
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

try:
    from soil_analytics.ml.supervised import (
        predict_images_from_bytes,
        predict_images_from_bytes_with_annotation,
    )
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
        f"Training data root: `{data_root}` — use **ImageFolder** class subfolders "
        "*or* a CSV manifest "
        "(see `data/fesem_supervised/README.md`, `scripts/fesem_labels.example.csv`). "
        "Train with `python scripts/train_fesem_supervised.py` "
        "(optional `--manifest …`, `--crop-bottom-fraction …`)."
    )
    model_path = Path(
        st.text_input("Model directory", value=str(default_model), key="fesem_model_dir")
    )
    meta_path = model_path / "meta.json"
    crop_note = ""
    if meta_path.is_file():
        try:
            with open(meta_path, encoding="utf-8") as f:
                _m = json.load(f)
            c = float(_m.get("crop_bottom_fraction") or 0.0)
            if c > 0:
                crop_note = (
                    f" Inference uses bottom crop **{c:.3f}** of image height "
                    "(matches training)."
                )
        except (json.JSONDecodeError, OSError):
            pass
    if not meta_path.exists():
        st.warning(
            "No trained model found (missing meta.json). Train a model or set the path above."
        )
    if crop_note:
        st.info(
            "To reduce reliance on overlays and the instrument bar, "
            "training can strip the bottom strip of pixels."
            + crop_note
        )
    st.caption(
        "**On-image result** draws the predicted class and an arrow (Grad-CAM saliency) "
        "like annotated FESEM figures."
    )
    overlay_style = st.checkbox(
        "On-image label + arrow + heat tint (reference figure style)",
        value=True,
        key="fesem_overlay_style",
    )
    heatmap_alpha = st.slider(
        "Saliency heatmap blend (0 = off)",
        min_value=0.0,
        max_value=0.75,
        value=0.38,
        step=0.02,
        key="fesem_hm_alpha",
    )
    blend_hm = overlay_style and heatmap_alpha > 1e-6
    tta_infer = st.checkbox(
        "Stabilize scores (TTA: average with mirror view — more consistent on the same image)",
        value=True,
        key="fesem_tta",
    )
    up_sup = st.file_uploader(
        "FESEM images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="fesem_sup",
    )
    if up_sup and meta_path.exists():
        files = [(u.name, u.getvalue()) for u in up_sup]
        if st.button("Run supervised inference", key="run_sup"):
            with st.spinner("Running inference…"):
                if overlay_style:
                    preds = predict_images_from_bytes_with_annotation(
                        files,
                        model_path,
                        blend_heatmap=blend_hm,
                        heatmap_alpha=heatmap_alpha,
                        tta=tta_infer,
                    )
                else:
                    preds = predict_images_from_bytes(files, model_path, tta=tta_infer)
            df = pd.DataFrame(
                [{k: v for k, v in r.items() if k != "annotated_png"} for r in preds]
            )
            st.dataframe(df, use_container_width=True)
            for row, upl in zip(preds, up_sup, strict=True):
                st.divider()
                col_img, col_out = st.columns([1, 1])
                with col_img:
                    if overlay_style and "annotated_png" in row:
                        st.image(
                            BytesIO(row["annotated_png"]),
                            caption=f"{row['path']} — annotated",
                            use_container_width=True,
                        )
                        with st.expander("Original upload"):
                            st.image(
                                BytesIO(upl.getvalue()),
                                caption=row["path"],
                                use_container_width=True,
                            )
                    else:
                        st.image(
                            BytesIO(upl.getvalue()),
                            caption=row["path"],
                            use_container_width=True,
                        )
                with col_out:
                    st.markdown(
                        f"**Predicted:** `{row['predicted_class']}`  \n"
                        f"**Confidence:** {row['confidence']:.3f}"
                    )
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
    backbone = st.selectbox(
        "Backbone (ImageNet pretrained)",
        ["resnet18", "efficientnet_b0"],
        index=0,
    )
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
