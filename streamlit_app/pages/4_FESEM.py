"""FESEM: table + MCQs from ``fesem_remarks.yaml`` only (no image file, no vision/ML)."""

from __future__ import annotations

import yaml
import streamlit as st
import pandas as pd

from soil_analytics.fesem_mcq import parse_phases, shuffled_mcq_labels
from soil_analytics.paths import project_root, reference_config_dir

st.set_page_config(page_title="FESEM", layout="wide")
st.title("Identification of mineral phases and morphology (FESEM)")
st.caption(
    "Use the reference table, then answer the **shape → phase** questions. "
    "Feedback links each correct association to what it **represents in the soil / microstructure**. "
    "Nothing is stored on the server; export a text summary at the end."
)

_ref = reference_config_dir() / "fesem_remarks.yaml"
if not _ref.is_file():
    st.error(f"Missing config: `{_ref}`")
    st.stop()

with open(_ref, encoding="utf-8") as f:
    _doc = yaml.safe_load(f) or {}
_rows = _doc.get("phases", [])
_narr = _doc.get("interpretation_narrative") or {}

specs = parse_phases(_rows)
if not specs:
    st.error("No phases in `fesem_remarks.yaml`.")
    st.stop()

all_labels = [p.label for p in specs]

st.subheader("Table: identification of mineral phases and morphology (FESEM)")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Mineral phase": p.label,
                "Chemical formula": p.chemical_formula or "—",
                "Morphology / description": p.morphology,
                "Context (from reference)": p.notes,
            }
            for p in specs
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

with st.expander("What these parameters represent (process, properties, environment)", expanded=False):
    st.markdown(
        f"**Process indicators:**  \n{_narr.get('process_indicators', '—')}\n\n"
        f"**Physical properties / microstructure:**  \n{_narr.get('physical_properties', '—')}\n\n"
        f"**Environment / exposure:**  \n{_narr.get('environment', '—')}"
    )

st.subheader("Questionnaire: morphology → phase (MCQ)")
st.caption(
    "For each **observed morphology** (as in the table), select the **mineral / phase** it best matches. "
    "Then read the interpretation: what that association suggests in the **soil or material**."
)

answers: dict[str, str] = {}
with st.form("fesem_mcq_form", clear_on_submit=False):
    for n, p in enumerate(specs, start=1):
        opts = shuffled_mcq_labels(p.label, all_labels, phase_id=p.id)
        st.markdown(f"**Question {n} of {len(specs)}**")
        st.markdown(
            f"Document this **appearance (shape / texture) in FESEM:**  \n*{p.morphology}*  \n\n"
            "**Which mineral / phase in the table is most consistent with that morphology?**"
        )
        choice = st.radio(
            "Choose one",
            options=opts,
            key=f"fe_mcq_{p.id}",
            label_visibility="collapsed",
        )
        answers[p.id] = choice
        st.divider()

    col1, col2 = st.columns(2)
    with col1:
        sample_id = st.text_input("Sample / site ID (optional)", placeholder="e.g. S3-02b")
    with col2:
        extra = st.text_input("Run / image label (optional)", placeholder="e.g. micrograph_12")

    submitted = st.form_submit_button("Score answers and show what it means for the soil / structure")

if submitted:
    correct_n = 0
    lines: list[str] = [
        "FESEM morphology → phase questionnaire",
        f"Sample / run: {sample_id or '—'} / {extra or '—'}",
        "",
    ]
    detail_blocks: list[str] = []

    for p in specs:
        chosen = answers.get(p.id, "")
        ok = chosen == p.label
        if ok:
            correct_n += 1
        short = p.morphology[:72] + ("…" if len(p.morphology) > 72 else "")
        lines.append(
            f"- Shape: “{short}” → table row **{p.label}** — "
            f"{'✓' if ok else '✗'} (you chose: {chosen or '—'})"
        )
        detail_blocks.append(
            f"**Row: {p.label}** (morphology you were shown: *{p.morphology}*)  \n"
            f"- You selected: **{chosen or '—'}** — **{'Correct' if ok else 'Incorrect'}** "
            f"(reference answer: **{p.label}**).  \n"
            f"- **What this phase tends to represent in the soil / material:** {p.soil_interpretation}"
        )

    st.success(f"Score: **{correct_n} / {len(specs)}** correct (morphology ↔ phase).")
    st.subheader("What your answers imply (per phase)")
    for block in detail_blocks:
        st.markdown(block)
        st.markdown("---")

    report = "\n".join(lines) + "\n\n— Per-phase soil / structure notes —\n"
    for p in specs:
        report += f"\n{p.label}:\n{p.soil_interpretation}\n"

    st.code("\n".join(lines) + "\n", language="text")
    st.download_button(
        "Download report (.txt)",
        data=report,
        file_name="fesem_mcq_interpretation.txt",
        mime="text/plain",
    )

st.caption(f"Config: `{_ref}` · Project: `{project_root()}`")
