"""FESEM: phase table rows from YAML (no ML)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseSpec:
    id: str
    label: str
    chemical_formula: str | None
    morphology: str
    notes: str
    soil_interpretation: str


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
