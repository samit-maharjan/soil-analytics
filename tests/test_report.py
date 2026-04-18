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
