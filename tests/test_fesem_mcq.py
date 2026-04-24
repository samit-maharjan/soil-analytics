"""FESEM MCQ helpers."""

from soil_analytics.fesem_mcq import parse_phases, shuffled_mcq_labels


def test_parse_phases_minimal() -> None:
    rows = [
        {
            "id": "a",
            "label": "Alpha",
            "chemical_formula": "X",
            "morphology": "fibrous",
            "notes": "n1",
            "soil_interpretation": "soil line",
        }
    ]
    ps = parse_phases(rows)
    assert len(ps) == 1
    assert ps[0].label == "Alpha"
    assert ps[0].soil_interpretation == "soil line"


def test_shuffled_mcq_four_options() -> None:
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    opts = shuffled_mcq_labels("C", labels, phase_id="test-id")
    assert len(set(opts)) == 4
    assert "C" in opts
