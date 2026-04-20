"""FESEM catalog: one micrograph file ↔ one analysis text under data/fesem_supervised/."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def match_upload_to_catalog(
    filename: str,
    file_bytes: bytes,
    data_root: Path | None = None,
) -> tuple[FesemPair | None, str]:
    """
    Match an upload to a catalog micrograph by **basename** (path segments ignored).

    Returns ``(pair, status)`` where status is:
    - ``catalog_match_verified`` — on-disk file exists and bytes match
    - ``catalog_match_name_only`` — same name on disk but content differs
    - ``catalog_no_such_file`` — no micrographs/ file with that name
    - ``catalog_no_analysis`` — file exists but no analysis text
    """
    root = data_root if data_root is not None else fesem_supervised_data_dir()
    safe_name = Path(filename).name
    target = (root / "micrographs" / safe_name).resolve()
    try:
        target.relative_to((root / "micrographs").resolve())
    except ValueError:
        return None, "catalog_no_such_file"
    if not target.is_file():
        return None, "catalog_no_such_file"
    on_disk = target.read_bytes()
    pair_list = [p for p in load_fesem_catalog(root) if p.name == safe_name]
    pair = pair_list[0] if pair_list else None
    if pair is None:
        return None, "catalog_no_such_file"
    if _sha256(file_bytes) != _sha256(on_disk):
        return pair, "catalog_match_name_only"
    if not (pair.analysis_text and str(pair.analysis_text).strip()):
        return pair, "catalog_no_analysis"
    return pair, "catalog_match_verified"


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
