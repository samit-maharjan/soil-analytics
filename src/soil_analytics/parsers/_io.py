"""Shared CSV loading helpers."""

from __future__ import annotations

import io
from typing import BinaryIO

import pandas as pd


def read_csv_flexible(raw: bytes | BinaryIO) -> pd.DataFrame:
    """Read CSV with comma or tab delimiter."""
    if isinstance(raw, bytes):
        bio = io.BytesIO(raw)
    else:
        bio = raw
    head = bio.read(4096)
    bio.seek(0)
    sep = "\t" if b"\t" in head and b"," not in head.split(b"\n")[0] else ","
    return pd.read_csv(bio, sep=sep, engine="python")


def normalize_column(name: str) -> str:
    return name.strip().lower().replace(" ", "_")
