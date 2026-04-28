"""XRD CSV and Bruker-style ASC (two-column 2θ vs intensity) parsers."""

from __future__ import annotations

import io
from typing import BinaryIO

import numpy as np
import pandas as pd

from lime_analytics.parsers._io import normalize_column, read_csv_flexible
from lime_analytics.schemas import XRDPattern


def _find_two_theta_column(columns: list[str]) -> str | None:
    for c in columns:
        cl = c.lower().replace(" ", "_")
        if cl in ("two_theta", "2theta", "2-theta", "twotheta", "deg", "x"):
            return c
        if "two_theta" in cl or "2theta" in cl or "2θ" in c:
            return c
    return None


def _find_intensity_column(columns: list[str], tt: str) -> str | None:
    for c in columns:
        if c == tt:
            continue
        cl = c.lower()
        if cl in ("intensity", "counts", "i", "y", "cps"):
            return c
    others = [c for c in columns if c != tt]
    return others[0] if others else None


def parse_xrd_csv(raw: bytes | BinaryIO, source_name: str | None = None) -> XRDPattern:
    df = read_csv_flexible(raw)
    df.columns = [normalize_column(str(c)) for c in df.columns]
    columns = list(df.columns)

    tt_col = _find_two_theta_column(columns)
    if tt_col is None:
        for i, c in enumerate(columns):
            if i + 1 < len(columns):
                try:
                    pd.to_numeric(df[c], errors="raise")
                    pd.to_numeric(df[columns[i + 1]], errors="raise")
                    tt_col, int_col = c, columns[i + 1]
                    break
                except Exception:
                    continue
        if tt_col is None:
            raise ValueError("Could not find 2theta column (e.g. two_theta, 2theta).")
    else:
        int_col = _find_intensity_column(columns, tt_col)
        if int_col is None:
            raise ValueError("Could not find intensity column.")

    two_theta = df[tt_col].to_numpy(dtype=float)
    intensity = df[int_col].to_numpy(dtype=float)
    order = np.argsort(two_theta)
    return XRDPattern(
        two_theta_deg=two_theta[order],
        intensity=intensity[order],
        source_name=source_name,
    )


def parse_xrd_asc(raw: bytes | BinaryIO, source_name: str | None = None) -> XRDPattern:
    """Whitespace-separated two-column text: 2θ (degrees), intensity (one pair per line)."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw.read().decode("utf-8", errors="replace")
    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            tt = float(parts[0])
            iy = float(parts[1])
        except ValueError:
            continue
        rows.append((tt, iy))
    if not rows:
        raise ValueError("No numeric 2θ / intensity rows found (expected ASC-style two columns).")
    arr = np.asarray(rows, dtype=float)
    two_theta = arr[:, 0]
    intensity = arr[:, 1]
    order = np.argsort(two_theta)
    return XRDPattern(
        two_theta_deg=two_theta[order],
        intensity=intensity[order],
        source_name=source_name,
    )


def _first_non_empty_line(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith(";"):
            return s
    return None


def _line_is_two_numeric_columns(line: str) -> bool:
    parts = line.replace(",", " ").split()
    if len(parts) != 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
    except ValueError:
        return False
    return True


def parse_xrd_bytes(
    raw: bytes | BinaryIO,
    *,
    source_name: str | None = None,
    filename: str | None = None,
) -> XRDPattern:
    """Dispatch: ``.asc``, two-column numeric text, or CSV with headers."""
    if isinstance(raw, bytes):
        blob = raw
    else:
        blob = raw.read()
        raw.seek(0)
    name = (filename or "").lower()
    if name.endswith(".asc"):
        return parse_xrd_asc(blob, source_name=source_name)
    head = blob[:8192].decode("utf-8", errors="replace")
    first = _first_non_empty_line(head)
    if first and _line_is_two_numeric_columns(first):
        return parse_xrd_asc(blob, source_name=source_name)
    return parse_xrd_csv(io.BytesIO(blob), source_name=source_name)
