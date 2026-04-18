"""FTIR parsers: CSV and JCAMP-style text (e.g. ##DATA TYPE=INFRARED SPECTRUM)."""

from __future__ import annotations

import io
import re
from typing import BinaryIO, Literal, cast

import numpy as np
import pandas as pd

from soil_analytics.parsers._io import normalize_column, read_csv_flexible
from soil_analytics.schemas import FTIRSeries

_YLabel = Literal["absorbance", "transmittance", "reflectance", "unknown"]


def _raw_to_bytes(raw: bytes | BinaryIO) -> bytes:
    if isinstance(raw, bytes):
        return raw
    return raw.read()


def parse_ftir_jcamp(raw: bytes | BinaryIO, source_name: str | None = None) -> FTIRSeries:
    """
    Parse JCAMP-DX style IR text: header lines starting with ##, then ``wavenumber<TAB>y`` pairs.
    Recognizes ``##YUNITS=%T`` / ABSORBANCE-style hints for the y axis.
    """
    data = _raw_to_bytes(raw)
    text = data.decode("utf-8", errors="replace")
    y_label: str = "unknown"
    rows: list[tuple[float, float]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("##"):
            ul = line.upper()
            if "YUNITS" in ul:
                if "%T" in line or "TRANSMITTANCE" in ul:
                    y_label = "transmittance"
                elif "ABS" in ul:
                    y_label = "absorbance"
                elif "REFLECT" in ul:
                    y_label = "reflectance"
            continue
        parts = re.split(r"[\s,]+", line.strip())
        if len(parts) < 2:
            continue
        try:
            wn = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        rows.append((wn, y))

    if not rows:
        raise ValueError("No numeric wavenumber / y pairs found (expected JCAMP IR or two-column text).")

    arr = np.array(rows, dtype=float)
    wn = arr[:, 0]
    y = arr[:, 1]
    order = np.argsort(wn)
    wn = wn[order]
    y = y[order]

    if y_label == "unknown":
        mx, mn = float(np.nanmax(y)), float(np.nanmin(y))
        if mn >= -0.05 and mx <= 105.0:
            y_label = "transmittance"
        elif mn >= -0.05 and mx <= 5.0 and np.nanmean(y) < 3:
            y_label = "absorbance"

    return FTIRSeries(
        wavenumber_cm1=wn,
        y=y,
        y_label=cast(_YLabel, y_label),
        source_name=source_name,
    )


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


def _looks_like_jcamp_ir(raw: bytes) -> bool:
    head = raw[:16384].lstrip()
    if head.startswith(b"##"):
        return True
    return b"##DATA TYPE=INFRARED" in raw[: min(len(raw), 32768)]


def parse_ftir_csv(raw: bytes | BinaryIO, source_name: str | None = None) -> FTIRSeries:
    data = _raw_to_bytes(raw)
    if _looks_like_jcamp_ir(data):
        return parse_ftir_jcamp(data, source_name=source_name)

    bio = io.BytesIO(data)
    df = read_csv_flexible(bio)
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
        y_label=cast(_YLabel, y_label),
        source_name=source_name,
    )
