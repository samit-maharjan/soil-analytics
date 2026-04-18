"""XRD CSV parser."""

from __future__ import annotations

from typing import BinaryIO

import numpy as np
import pandas as pd

from soil_analytics.parsers._io import normalize_column, read_csv_flexible
from soil_analytics.schemas import XRDPattern


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
