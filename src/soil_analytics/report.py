"""Simple HTML report export."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any

from soil_analytics.reference_checks import CheckResult


def checks_to_rows(results: list[CheckResult]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        rows.append(
            {
                "id": r.check_id,
                "label": r.label,
                "status": r.status,
                "message": r.message,
            }
        )
    return rows


def build_html_report(
    title: str,
    sections: list[tuple[str, str, list[CheckResult] | None]],
    figure_html: str | None = None,
) -> str:
    """sections: (heading, intro markdown text, optional checks)."""
    parts: list[str] = []
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append(
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;}"
        "table{border-collapse:collapse;width:100%;}td,th{border:1px solid #ccc;padding:6px;}"
        "tr:nth-child(even){background:#f8f8f8;}</style></head><body>"
    )
    parts.append(f"<h1>{html.escape(title)}</h1><p><small>Generated {html.escape(ts)}</small></p>")

    for heading, intro, checks in sections:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        parts.append(f"<p>{html.escape(intro)}</p>")
        if checks:
            parts.append("<table><thead><tr><th>ID</th><th>Label</th><th>Status</th><th>Message</th></tr></thead><tbody>")
            for c in checks:
                parts.append(
                    "<tr>"
                    f"<td>{html.escape(c.check_id)}</td>"
                    f"<td>{html.escape(c.label)}</td>"
                    f"<td>{html.escape(c.status)}</td>"
                    f"<td>{html.escape(c.message)}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")

    if figure_html:
        parts.append("<h2>Figure</h2>")
        parts.append(figure_html)

    parts.append("</body></html>")
    return "\n".join(parts)
