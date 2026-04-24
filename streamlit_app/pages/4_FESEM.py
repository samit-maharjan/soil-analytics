"""FESEM: reference table and progressive phase guide (YAML; no image / ML)."""

from __future__ import annotations

import yaml
import streamlit as st

from soil_analytics.fesem_mcq import parse_phases
from soil_analytics.fesem_wizard import find_option, load_wizard, WizardNode
from soil_analytics.paths import reference_config_dir
from soil_analytics.streamlit_readability import inject_readability_css
from soil_analytics.streamlit_tables import scrollable_dataframe

REF = reference_config_dir() / "fesem_remarks.yaml"
WIZ = reference_config_dir() / "fesem_wizard.yaml"
_WIZ_MAX_STEPS = 5

st.set_page_config(page_title="FESEM", layout="wide")
inject_readability_css(emphasize_radio=True)
st.title("Mineral phase guide (FESEM)")

if not REF.is_file() or not WIZ.is_file():
    st.error("Missing `fesem_remarks.yaml` and/or `fesem_wizard.yaml` in `config/reference_ranges/`.")
    st.stop()

with open(REF, encoding="utf-8") as f:
    fesem_doc = yaml.safe_load(f) or {}
WIZARD = load_wizard(WIZ)
narr = fesem_doc.get("interpretation_narrative") or {}
phase_list = parse_phases(fesem_doc.get("phases", []))
if not phase_list:
    st.error("No phases in `fesem_remarks.yaml`.")
    st.stop()
by_id = {p.id: p for p in phase_list}


def apply_path() -> tuple[str | None, str | None, list[tuple[WizardNode, str]]]:
    steps: list[dict] = st.session_state.get("fe_steps", [])
    nid: str = WIZARD.start
    trail: list[tuple[WizardNode, str]] = []
    for stp in steps:
        node = WIZARD.nodes.get(nid)
        if not node:
            return None, None, trail
        o = find_option(node, str(stp.get("key", "")))
        if not o:
            return None, None, trail
        trail.append((node, o.label))
        if o.result:
            return None, o.result, trail
        if o.next:
            nid = o.next
    return nid, None, trail


if "fe_steps" not in st.session_state:
    st.session_state.fe_steps = []
st.session_state.pop("fe_field", None)

st.caption(
    "One step at a time. You get a **suggested** phase and a short line from the reference list—not a substitute for expert ID."
)
st.divider()

cur, phase_id, _ = apply_path()

if phase_id and phase_id in by_id:
    p = by_id[phase_id]
    st.subheader("Suggested match")
    st.markdown(
        f"**{p.label}**  \n{p.chemical_formula or '—'}  \n\n**Typical appearance (reference)**  \n{p.morphology}  \n\n"
        f"**Interpretation (reference)**  \n{p.soil_interpretation}"
    )
    b1, b2, _ = st.columns([0.32, 0.32, 0.36])
    with b1:
        if st.button("Back one step", key="fe_bk"):
            st.session_state.fe_steps = st.session_state.fe_steps[:-1]
            st.rerun()
    with b2:
        if st.button("Start over", type="primary", key="fe_re"):
            st.session_state.fe_steps = []
            st.rerun()

elif cur is None and not phase_id:
    st.error("The path is invalid. Reset to start over.")
    if st.button("Reset", key="fe_rez0"):
        st.session_state.fe_steps = []
        st.rerun()

elif cur is None and phase_id and phase_id not in by_id:
    st.error(f"Unknown result id: {phase_id!r}.")
    if st.button("Reset", key="fe_rez1"):
        st.session_state.fe_steps = []
        st.rerun()

elif cur is not None:
    node = WIZARD.nodes[cur]
    step_idx = len(st.session_state.fe_steps) + 1
    st.progress(min(1.0, (step_idx - 1) / float(_WIZ_MAX_STEPS)))
    st.markdown("")

    st.markdown(f"### Step {step_idx} — {node.title}")
    st.markdown(node.prompt)
    st.markdown("  \n  ")

    cands = [o for o in node.options if o.key and o.label]
    if not cands:
        st.error("Invalid node (no options) in the wizard file.")
        st.stop()

    wk = f"w_{node.id}_q{len(st.session_state.fe_steps)}"
    ch = st.radio(
        "Options",
        options=cands,
        format_func=lambda o: o.label,
        key=wk,
        label_visibility="collapsed",
    )
    st.markdown("  \n  ")

    b1, b2, _rest = st.columns([0.2, 0.2, 0.6])
    with b1:
        if st.button("Back", disabled=not st.session_state.fe_steps, key="fe_p"):
            st.session_state.fe_steps = st.session_state.fe_steps[:-1]
            st.rerun()
    with b2:
        nxt = st.button("Next", type="primary", key="fe_n")
    if nxt:
        st.session_state.fe_steps = [
            *st.session_state.fe_steps,
            {"node": cur, "key": ch.key},
        ]
        st.rerun()

st.divider()
st.subheader("Reference: phases and typical look")
scrollable_dataframe(
    [
        {
            "Mineral": p.label,
            "Formula": p.chemical_formula or "—",
            "Typical look": p.morphology,
            "Note": p.notes,
        }
        for p in phase_list
    ],
)
with st.expander("Context: process, structure, and environment (qualitative)", expanded=False):
    st.markdown(
        f"**Process**  \n{narr.get('process_indicators', '—')}\n\n"
        f"**Structure**  \n{narr.get('physical_properties', '—')}\n\n"
        f"**Environment**  \n{narr.get('environment', '—')}"
    )
