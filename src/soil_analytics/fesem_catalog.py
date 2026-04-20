"""FESEM catalog: one micrograph file ↔ one analysis text under data/fesem_supervised/."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imagehash
from PIL import Image, UnidentifiedImageError

from soil_analytics.paths import fesem_supervised_data_dir

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


@dataclass(frozen=True)
class FesemPair:
    """A micrograph in ``micrographs/`` and its optional ``analysis/<stem>.txt|md``."""

    name: str
    image_path: Path
    analysis_path: Path | None
    analysis_text: str | None


def _analysis_path_for_stem(analysis_dir: Path, stem: str) -> Path | None:
    for ext in (".txt", ".md"):
        p = analysis_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def load_fesem_catalog(data_root: Path | None = None) -> list[FesemPair]:
    """
    Pair every image in ``micrographs/`` with ``analysis/<same-stem>.txt`` or ``.md`` if present.
    """
    root = data_root if data_root is not None else fesem_supervised_data_dir()
    mg = root / "micrographs"
    an = root / "analysis"
    if not mg.is_dir():
        return []
    pairs: list[FesemPair] = []
    for image_path in sorted(mg.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = image_path.stem
        ap = _analysis_path_for_stem(an, stem) if an.is_dir() else None
        text: str | None
        if ap is not None:
            text = ap.read_text(encoding="utf-8").strip() or None
        else:
            text = None
        pairs.append(
            FesemPair(
                name=image_path.name,
                image_path=image_path,
                analysis_path=ap,
                analysis_text=text,
            )
        )
    return pairs


def phash_image_bytes(data: bytes) -> imagehash.ImageHash | None:
    """Perceptual hash (pHash); ``None`` if the bytes are not a decodable image."""
    try:
        pil = Image.open(io.BytesIO(data))
        pil.load()
        return imagehash.phash(pil.convert("RGB"))
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def phash_image_path(path: Path) -> imagehash.ImageHash | None:
    try:
        with Image.open(path) as pil:
            pil.load()
            return imagehash.phash(pil.convert("RGB"))
    except (OSError, UnidentifiedImageError, ValueError):
        return None


SimilarityStatus = Literal[
    "similarity_match",
    "similarity_no_catalog",
    "similarity_unreadable_upload",
    "similarity_unreadable_catalog_entry",
    "similarity_no_analysis",
]


@dataclass(frozen=True)
class SimilarityMatch:
    """Best catalog row for an upload by minimum pHash Hamming distance."""

    pair: FesemPair | None
    hamming: int | None
    hash_bits: int
    status: SimilarityStatus


def match_upload_by_image_similarity(
    file_bytes: bytes,
    data_root: Path | None = None,
) -> SimilarityMatch:
    """
    Map an uploaded micrograph to the **catalog micrograph with the most similar appearance**
    (perceptual hash; lower Hamming distance = closer match).

    Each on-disk micrograph still has a 1:1 pairing with its ``analysis/<stem>`` file; this
    function chooses **which** micrograph the upload resembles, then returns that row’s analysis.
    """
    root = data_root if data_root is not None else fesem_supervised_data_dir()
    pairs = load_fesem_catalog(root)
    if not pairs:
        return SimilarityMatch(pair=None, hamming=None, hash_bits=64, status="similarity_no_catalog")

    q = phash_image_bytes(file_bytes)
    if q is None:
        return SimilarityMatch(pair=None, hamming=None, hash_bits=64, status="similarity_unreadable_upload")

    best: tuple[int, FesemPair] | None = None
    for p in pairs:
        h = phash_image_path(p.image_path)
        if h is None:
            continue
        d = int(h - q)
        if best is None or d < best[0]:
            best = (d, p)

    if best is None:
        return SimilarityMatch(pair=None, hamming=None, hash_bits=64, status="similarity_unreadable_catalog_entry")

    dist, pair = best
    # pHash uses 8×8 = 64 bits (see ``imagehash.phash``).
    bits = int(q.hash.size)
    if not (pair.analysis_text and str(pair.analysis_text).strip()):
        return SimilarityMatch(
            pair=pair, hamming=dist, hash_bits=bits, status="similarity_no_analysis"
        )
    return SimilarityMatch(pair=pair, hamming=dist, hash_bits=bits, status="similarity_match")


def hamming_similarity_fraction(hamming: int | None, hash_bits: int) -> float | None:
    """Map Hamming distance to a simple similarity score in ``[0, 1]`` (1 = identical hash)."""
    if hamming is None or hash_bits <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - float(hamming) / float(hash_bits)))


def catalog_summary_rows(pairs: list[FesemPair]) -> list[dict[str, str]]:
    """Rows for a small table (e.g. Streamlit / DataFrame)."""
    rows: list[dict[str, str]] = []
    for p in pairs:
        if p.analysis_path and p.analysis_text:
            status = "OK"
        elif p.analysis_path:
            status = "empty"
        else:
            status = "missing"
        rows.append(
            {
                "Micrograph": p.name,
                "Analysis": p.analysis_path.name if p.analysis_path else "—",
                "Status": status,
            }
        )
    return rows
