"""FESEM morphology ↔ phase MCQs from YAML reference rows (no ML)."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseSpec:
    id: str
    label: str
    chemical_formula: str | None
    morphology: str
    notes: str
    soil_interpretation: str


def _stable_seed(phase_id: str) -> int:
    h = 0
    for c in phase_id:
        h = (h * 31 + ord(c)) & 0x7FFFFFFF
    return h or 1


def parse_phases(yaml_block: list[dict]) -> list[PhaseSpec]:
    out: list[PhaseSpec] = []
    for r in yaml_block or []:
        if not r.get("id") or not (r.get("label") or r.get("id")):
            continue
        out.append(
            PhaseSpec(
                id=str(r["id"]),
                label=str(r.get("label") or r.get("id")),
                chemical_formula=r.get("chemical_formula"),
                morphology=str(r.get("morphology") or ""),
                notes=str(r.get("notes") or ""),
                soil_interpretation=str(
                    r.get("soil_interpretation")
                    or r.get("notes")
                    or "—"
                ),
            )
        )
    return out


def shuffled_mcq_labels(correct: str, all_labels: list[str], *, phase_id: str) -> list[str]:
    """
    One correct phase name plus three decoys, shuffled (reproducible for this phase_id).
    """
    others = [L for L in all_labels if L != correct]
    r = random.Random(_stable_seed(phase_id + ":labels"))
    k = min(3, len(others))
    decoys = r.sample(others, k=k) if k else []
    pool = [correct, *decoys]
    r.shuffle(pool)
    return pool
