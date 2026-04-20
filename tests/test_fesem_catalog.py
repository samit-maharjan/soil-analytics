"""Tests for FESEM micrograph ↔ analysis pairing and perceptual similarity."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from soil_analytics.fesem_catalog import (
    catalog_summary_rows,
    hamming_similarity_fraction,
    load_fesem_catalog,
    match_upload_by_image_similarity,
    phash_image_bytes,
)


def _png_bytes(fill: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color=fill).save(buf, format="PNG")
    return buf.getvalue()


def test_load_fesem_catalog_pairs_txt(tmp_path: Path) -> None:
    mg = tmp_path / "micrographs"
    an = tmp_path / "analysis"
    mg.mkdir()
    an.mkdir()
    (mg / "a.png").write_bytes(_png_bytes((200, 100, 50)))
    (an / "a.txt").write_text("Needle-like aragonite.", encoding="utf-8")
    pairs = load_fesem_catalog(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].name == "a.png"
    assert pairs[0].analysis_text == "Needle-like aragonite."
    rows = catalog_summary_rows(pairs)
    assert rows[0]["Status"] == "OK"


def test_load_fesem_catalog_prefers_txt_over_md(tmp_path: Path) -> None:
    mg = tmp_path / "micrographs"
    an = tmp_path / "analysis"
    mg.mkdir()
    an.mkdir()
    (mg / "x.png").write_bytes(_png_bytes((10, 20, 30)))
    (an / "x.txt").write_text("from txt", encoding="utf-8")
    (an / "x.md").write_text("from md", encoding="utf-8")
    pairs = load_fesem_catalog(tmp_path)
    assert pairs[0].analysis_text == "from txt"


def test_similarity_matches_identical_upload(tmp_path: Path) -> None:
    mg = tmp_path / "micrographs"
    an = tmp_path / "analysis"
    mg.mkdir()
    an.mkdir()
    img_b = _png_bytes((99, 150, 40))
    (mg / "only.png").write_bytes(img_b)
    (an / "only.txt").write_text("Calcite rhombs.", encoding="utf-8")
    # Second unrelated catalog image so similarity must pick the right one
    (mg / "other.png").write_bytes(_png_bytes((5, 5, 5)))
    (an / "other.txt").write_text("Other phase.", encoding="utf-8")

    m = match_upload_by_image_similarity(img_b, data_root=tmp_path)
    assert m.status == "similarity_match"
    assert m.pair is not None
    assert m.pair.name == "only.png"
    assert m.hamming == 0
    assert m.pair.analysis_text == "Calcite rhombs."


def test_similarity_no_analysis_still_returns_pair(tmp_path: Path) -> None:
    mg = tmp_path / "micrographs"
    mg.mkdir()
    img_b = _png_bytes((77, 88, 99))
    (mg / "naked.png").write_bytes(img_b)
    m = match_upload_by_image_similarity(img_b, data_root=tmp_path)
    assert m.status == "similarity_no_analysis"
    assert m.pair is not None and m.pair.name == "naked.png"
    assert m.hamming == 0


def test_similarity_unreadable_upload(tmp_path: Path) -> None:
    mg = tmp_path / "micrographs"
    an = tmp_path / "analysis"
    mg.mkdir()
    an.mkdir()
    (mg / "a.png").write_bytes(_png_bytes((1, 2, 3)))
    (an / "a.txt").write_text("x", encoding="utf-8")
    m = match_upload_by_image_similarity(b"not an image", data_root=tmp_path)
    assert m.status == "similarity_unreadable_upload"


def test_hamming_similarity_fraction() -> None:
    assert hamming_similarity_fraction(0, 64) == 1.0
    assert hamming_similarity_fraction(64, 64) == 0.0
    assert hamming_similarity_fraction(None, 64) is None


def test_phash_roundtrip_bytes() -> None:
    b = _png_bytes((42, 43, 44))
    assert phash_image_bytes(b) is not None


def test_load_empty_when_no_micrographs_dir(tmp_path: Path) -> None:
    assert load_fesem_catalog(tmp_path) == []
