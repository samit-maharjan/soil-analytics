"""FESEM phase table helpers."""

from lime_analytics.fesem_mcq import parse_phases


def test_parse_phases_minimal() -> None:
    rows = [
        {
            "id": "a",
            "label": "Alpha",
            "chemical_formula": "X",
            "morphology": "fibrous",
            "notes": "n1",
            "lime_interpretation": "lime line",
        }
    ]
    ps = parse_phases(rows)
    assert len(ps) == 1
    assert ps[0].label == "Alpha"
    assert ps[0].lime_interpretation == "lime line"
