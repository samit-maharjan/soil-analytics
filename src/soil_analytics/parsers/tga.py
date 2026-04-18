"""TGA CSV parser and optional DTG computation."""

from __future__ import annotations

from typing import BinaryIO

import numpy as np
import pandas as pd

from soil_analytics.parsers._io import normalize_column, read_csv_flexible
from soil_analytics.schemas import TGACurve


def _find_temp_column(columns: list[str]) -> str | None:
    for c in columns:
        cl = c.lower()
        if cl in ("temperature_c", "temp_c", "temperature", "t_c", "x"):
            return c
        if "temp" in cl and "time" not in cl:
            return c
    return None


def _find_mass_column(columns: list[str], temp: str) -> str | None:
    for c in columns:
        if c == temp:
            continue
        cl = c.lower()
        if any(
            x in cl
            for x in ("mass", "weight", "mg", "fraction", "percent", "pct", "tg", "remaining")
        ):
            return c
    others = [c for c in columns if c != temp]
    return others[0] if others else None


def _compute_dtg(temperature_c: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """dm/dT using numpy.gradient (works for non-uniform temperature steps)."""
    return np.gradient(mass, temperature_c)


def parse_tga_csv(
    raw: bytes | BinaryIO,
    source_name: str | None = None,
    include_dtg: bool = True,
) -> TGACurve:
    df = read_csv_flexible(raw)
    df.columns = [normalize_column(str(c)) for c in df.columns]
    columns = list(df.columns)

    t_col = _find_temp_column(columns)
    if t_col is None:
        for i, c in enumerate(columns):
            if i + 1 < len(columns):
                try:
                    pd.to_numeric(df[c], errors="raise")
                    pd.to_numeric(df[columns[i + 1]], errors="raise")
                    t_col, m_col = c, columns[i + 1]
                    break
                except Exception:
                    continue
        if t_col is None:
            raise ValueError("Could not find temperature column.")
    else:
        m_col = _find_mass_column(columns, t_col)
        if m_col is None:
            raise ValueError("Could not find mass column.")

    temperature_c = df[t_col].to_numpy(dtype=float)
    mass = df[m_col].to_numpy(dtype=float)
    order = np.argsort(temperature_c)
    temperature_c = temperature_c[order]
    mass = mass[order]

    mass_label = m_col
    dtg_arr: np.ndarray | None = None
    if include_dtg:
        dtg_arr = _compute_dtg(temperature_c, mass)

    return TGACurve(
        temperature_c=temperature_c,
        mass=mass,
        mass_label=mass_label,
        dtg=dtg_arr,
        source_name=source_name,
    )
