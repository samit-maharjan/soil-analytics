"""FESEM: qualitative morphology reference (YAML) and paired micrographs + analysis text."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from soil_analytics.fesem_catalog import (
    catalog_summary_rows,
    load_fesem_catalog,
    match_upload_to_catalog,
)
from soil_analytics.paths import fesem_supervised_data_dir, project_root, reference_config_dir

st.set_page_config(page_title="FESEM", layout="wide")
st.title("FESEM")
st.caption(
    "Phase morphology uses config YAML. Catalogued micrographs live under "
    "`data/fesem_supervised/micrographs/` with matching analysis text in `analysis/` "
    "(same base name: e.g. `sample.png` ↔ `sample.txt`)."
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

st.subheader("Upload lookup")
st.caption(
    "Matches uploads to catalog micrographs **by file name**. "
    "If the bytes match the file on disk, the paired analysis is shown as verified."
)
up = st.file_uploader(
    "Micrograph file",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    key="fesem_upload",
)
if up and st.button("Look up in catalog", key="fesem_lookup"):
    raw = up.getvalue()
    pair, status = match_upload_to_catalog(up.name, raw, data_root=data_root)
    if status == "catalog_no_such_file":
        st.error(f"No catalog micrograph named `{Path(up.name).name}`.")
    elif status == "catalog_match_name_only":
        st.warning(
            "File name matches a catalog micrograph, but **file contents differ** from disk. "
            "Showing the catalog entry for that name anyway."
        )
        if pair and pair.analysis_text:
            st.markdown(pair.analysis_text)
        elif pair:
            st.warning("That micrograph has no analysis text yet.")
    elif status == "catalog_no_analysis":
        st.success("Catalog file matches on disk (verified).")
        st.warning("No analysis text is linked for this micrograph.")
    else:
        st.success("Catalog file matches on disk (verified).")
        if pair and pair.analysis_text:
            st.markdown(pair.analysis_text)

st.markdown(f"Project root: `{project_root()}`")
