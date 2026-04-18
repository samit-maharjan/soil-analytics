"""Report HTML builder."""

from soil_analytics.reference_checks import CheckResult
from soil_analytics.report import build_html_report


def test_build_html_report() -> None:
    checks = [
        CheckResult(
            check_id="a",
            label="Test",
            status="pass",
            message="ok",
            evidence={},
        )
    ]
    html = build_html_report("T", [("Sec", "intro", checks)], figure_html=None)
    assert "T" in html
    assert "pass" in html

    html2 = build_html_report(
        "T",
        [("Sec", "intro", checks)],
        figure_html=None,
        inference_rows=[{"Band": "b", "Peak wavenumber (cm⁻¹)": "100", "Inference": "note"}],
    )
    assert "Band inferences" in html2
    assert "note" in html2

    html3 = build_html_report(
        "T",
        [("Sec", "intro", None)],
        figure_html=None,
        inference_rows=[{"Sample": "a.txt", "Band": "b", "Peak wavenumber (cm⁻¹)": "100", "Inference": "n", "Status": "pass"}],
        qc_rows=[{"Sample": "a.txt", "id": "x", "label": "L", "status": "pass", "message": "m"}],
    )
    assert "Reference check status" in html3
    assert "a.txt" in html3
