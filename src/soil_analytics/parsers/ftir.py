"""FTIR CSV parser."""

from __future__ import annotations

from typing import BinaryIO

import numpy as np
import pandas as pd

from soil_analytics.parsers._io import normalize_column, read_csv_flexible
from soil_analytics.schemas import FTIRSeries


def _find_wavenumber_column(columns: list[str]) -> str | None:
    for c in columns:
        cl = c.lower()
        if cl in ("wavenumber", "wavenumber_cm-1", "wavenumber_cm1", "cm-1", "cm_1", "x"):
            return c
        if "wavenumber" in cl or cl.endswith("cm-1") or cl.endswith("cm1"):
            return c
    return None


def _find_y_column(columns: list[str], wn: str) -> str | None:
    y_hints = ("absorbance", "abs", "transmittance", "reflectance", "t", "r", "y", "intensity")
    for c in columns:
        if c == wn:
            continue
        cl = c.lower()
        if cl in y_hints or any(h in cl for h in ("abs", "trans", "refl")):
            return c
    others = [c for c in columns if c != wn]
    return others[0] if others else None


def parse_ftir_csv(raw: bytes | BinaryIO, source_name: str | None = None) -> FTIRSeries:
    df = read_csv_flexible(raw)
    df.columns = [normalize_column(str(c)) for c in df.columns]
    columns = list(df.columns)

    wn_col = _find_wavenumber_column(columns)
    if wn_col is None:
        # first two columns if numeric
        for i, c in enumerate(columns):
            if i + 1 < len(columns):
                try:
                    pd.to_numeric(df[c], errors="raise")
                    pd.to_numeric(df[columns[i + 1]], errors="raise")
                    wn_col, y_col = c, columns[i + 1]
                    break
                except Exception:
                    continue
        if wn_col is None:
            raise ValueError("Could not find wavenumber column (e.g. wavenumber, cm-1).")
    else:
        y_col = _find_y_column(columns, wn_col)
        if y_col is None:
            raise ValueError("Could not find absorbance/transmittance column.")

    wn = df[wn_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    order = np.argsort(wn)
    wn = wn[order]
    y = y[order]

    y_label = "unknown"
    lc = y_col.lower()
    if "abs" in lc:
        y_label = "absorbance"
    elif "trans" in lc:
        y_label = "transmittance"
    elif "refl" in lc:
        y_label = "reflectance"

    return FTIRSeries(
        wavenumber_cm1=wn,
        y=y,
        y_label=y_label,
        source_name=source_name,
    )
