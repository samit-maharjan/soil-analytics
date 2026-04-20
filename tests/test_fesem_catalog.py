"""Tests for FESEM micrograph ↔ analysis pairing."""

from __future__ import annotations

from pathlib import Path

from soil_analytics.fesem_catalog import (
    catalog_summary_rows,
    load_fesem_catalog,
    match_upload_to_catalog,
)


def test_load_fesem_catalog_pairs_txt(tmp_path: Path) -> None:
    mg = tmp_path / "micrographs"
    an = tmp_path / "analysis"
    mg.mkdir()
    an.mkdir()
    (mg / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
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
    (mg / "x.png").write_bytes(b"x")
    (an / "x.txt").write_text("from txt", encoding="utf-8")
    (an / "x.md").write_text("from md", encoding="utf-8")
    pairs = load_fesem_catalog(tmp_path)
    assert pairs[0].analysis_text == "from txt"


def test_match_upload_verified(tmp_path: Path) -> None:
    mg = tmp_path / "micrographs"
    an = tmp_path / "analysis"
    mg.mkdir()
    an.mkdir()
    img = b"fakeimagebytes"
    (mg / "m.png").write_bytes(img)
    (an / "m.txt").write_text("Morphology notes.", encoding="utf-8")
    pair, status = match_upload_to_catalog("m.png", img, data_root=tmp_path)
    assert status == "catalog_match_verified"
    assert pair is not None
    assert pair.analysis_text == "Morphology notes."


def test_match_upload_name_only(tmp_path: Path) -> None:
    mg = tmp_path / "micrographs"
    an = tmp_path / "analysis"
    mg.mkdir()
    an.mkdir()
    (mg / "m.png").write_bytes(b"a")
    (an / "m.txt").write_text("ok", encoding="utf-8")
    _, status = match_upload_to_catalog("m.png", b"b", data_root=tmp_path)
    assert status == "catalog_match_name_only"


def test_match_upload_path_traversal_ignored(tmp_path: Path) -> None:
    mg = tmp_path / "micrographs"
    an = tmp_path / "analysis"
    mg.mkdir()
    an.mkdir()
    (mg / "safe.png").write_bytes(b"x")
    (an / "safe.txt").write_text("ok", encoding="utf-8")
    pair, status = match_upload_to_catalog("../../micrographs/safe.png", b"x", data_root=tmp_path)
    assert status == "catalog_match_verified"
    assert pair is not None


def test_load_empty_when_no_micrographs_dir(tmp_path: Path) -> None:
    assert load_fesem_catalog(tmp_path) == []
