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
        inference_rows=[{"Band": "b", "Peak wavenumber (cm⁻¹)": "100", "Inference": "note", "Status": "pass"}],
    )
    assert "Band inferences" in html2
    assert "note" in html2
