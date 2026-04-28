"""TGA CSV parser and optional DTG computation."""

from __future__ import annotations

import io
from typing import BinaryIO

import numpy as np
import pandas as pd

from lime_analytics.parsers._io import normalize_column, read_csv_flexible
from lime_analytics.schemas import TGACurve


def _read_tga_dataframe_netzsch(raw: bytes) -> pd.DataFrame:
    """
    NETZSCH / Proteus-style ASCII: metadata lines starting with '#', then a
    '##Temp...' header row followed by comma-separated numeric data.
    """
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    header_idx: int | None = None
    header_line: str | None = None
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip()
        if not stripped or stripped.lower().startswith("exporttype"):
            continue
        low = stripped.lower()
        if "temp" in low and ("mass" in low or "/mg" in low or "tg" in low):
            header_idx = i
            header_line = stripped
            break
    if header_idx is None or header_line is None:
        raise ValueError("Could not find TGA table header (temperature + mass columns).")

    block = [header_line]
    for line in lines[header_idx + 1 :]:
        sl = line.strip()
        if not sl or sl.startswith("#"):
            continue
        parts = [p.strip() for p in sl.split(",")]
        if len(parts) < 2:
            continue
        try:
            float(parts[0].replace(",", "."))
        except ValueError:
            break
        block.append(sl)

    return pd.read_csv(io.StringIO("\n".join(block)), sep=",", engine="python")


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
    bio: bytes
    if isinstance(raw, bytes):
        bio = raw
    else:
        pos = raw.tell()
        bio = raw.read()
        raw.seek(pos)

    df: pd.DataFrame
    try:
        cand = read_csv_flexible(bio)
        cand.columns = [normalize_column(str(c)) for c in cand.columns]
        cols = list(cand.columns)
        tc = _find_temp_column(cols)
        mc = _find_mass_column(cols, tc) if tc else None
        if tc is None or mc is None:
            raise ValueError("missing TGA columns in generic CSV")
        df = cand
    except Exception:
        df = _read_tga_dataframe_netzsch(bio)

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
