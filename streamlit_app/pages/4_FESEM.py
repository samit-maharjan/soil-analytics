"""FESEM: qualitative morphology reference (YAML) and paired micrographs + analysis text."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from soil_analytics.fesem_catalog import (
    catalog_summary_rows,
    hamming_similarity_fraction,
    load_fesem_catalog,
    match_upload_by_image_similarity,
)
from soil_analytics.paths import fesem_supervised_data_dir, project_root, reference_config_dir

st.set_page_config(page_title="FESEM", layout="wide")
st.title("FESEM")
st.caption(
    "Phase morphology uses config YAML. Catalogued micrographs live under "
    "`data/fesem_supervised/micrographs/` with matching analysis text in `analysis/` "
    "(same base name: e.g. `sample.png` ↔ `sample.txt`). "
    "Uploads are mapped to that catalog **by image similarity** (perceptual hash), not file name."
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

data_root_raw = st.text_input(
    "Catalog root",
    value=str(fesem_supervised_data_dir()),
    key="fesem_data_root_input",
    help="Must contain micrographs/ and optionally analysis/ with paired text files.",
)
try:
    data_root = Path(data_root_raw).expanduser().resolve()
except OSError:
    data_root = fesem_supervised_data_dir()

pairs = load_fesem_catalog(data_root)
st.markdown(
    f"**micrographs:** `{data_root / 'micrographs'}`  \n"
    f"**analysis:** `{data_root / 'analysis'}`"
)

if not pairs:
    st.info(
        "No micrographs found. Add images under **micrographs/** and optional "
        "**analysis/<same-stem>.txt** or **.md** (see `data/fesem_supervised/README.md`)."
    )
else:
    st.subheader("Catalog")
    st.dataframe(
        pd.DataFrame(catalog_summary_rows(pairs)),
        use_container_width=True,
        hide_index=True,
    )
    labels = [p.name for p in pairs]
    choice = st.selectbox("Open micrograph", options=labels, key="fesem_pick")
    sel = next(p for p in pairs if p.name == choice)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(str(sel.image_path), caption=sel.name, use_container_width=True)
    with c2:
        if sel.analysis_text:
            st.markdown(sel.analysis_text)
        elif sel.analysis_path:
            st.warning("Analysis file is empty.")
        else:
            st.warning(
                f"No analysis file — add `{data_root / 'analysis' / (sel.image_path.stem + '.txt')}`."
            )

st.subheader("Upload — map by image similarity")
st.caption(
    "Each catalog micrograph is still tied 1:1 to an analysis file on disk. "
    "Your upload is compared **visually** to all catalog images; the **closest** match "
    "(perceptual hash Hamming distance) selects which analysis to show."
)
up = st.file_uploader(
    "Micrograph file",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    key="fesem_upload",
)
if up and st.button("Find best matching catalog entry", key="fesem_sim"):
    raw = up.getvalue()
    m = match_upload_by_image_similarity(raw, data_root=data_root)
    if m.status == "similarity_no_catalog":
        st.error("Catalog is empty — add micrographs under micrographs/.")
    elif m.status == "similarity_unreadable_upload":
        st.error("Could not decode the upload as an image.")
    elif m.status == "similarity_unreadable_catalog_entry":
        st.error("Catalog images could not be read for comparison.")
    elif m.pair is None:
        st.error("No match.")
    else:
        sim = hamming_similarity_fraction(m.hamming, m.hash_bits)
        if m.hamming is not None:
            st.metric(
                "Best-match Hamming distance (pHash)",
                m.hamming,
                help="Lower is more similar; 0 often means same or nearly identical appearance.",
            )
            if sim is not None:
                st.caption(f"Approximate similarity score (from distance): **{sim:.2%}**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(BytesIO(raw), caption="Your upload", use_container_width=True)
        with col_b:
            st.image(str(m.pair.image_path), caption=f"Closest catalog: {m.pair.name}", use_container_width=True)
        if m.status == "similarity_no_analysis":
            st.warning(
                f"Closest catalog image is **{m.pair.name}**, but there is **no analysis text** "
                f"for that entry yet."
            )
        else:
            st.markdown(m.pair.analysis_text or "")

st.markdown(f"Project root: `{project_root()}`")
