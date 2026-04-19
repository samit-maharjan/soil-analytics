"""Exact (SHA-256) and near-duplicate (embedding cosine) alignment with supervised manifest rows."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from soil_analytics.paths import fesem_supervised_data_dir, project_root


def default_manifest_path() -> Path:
    return project_root() / "scripts" / "fesem_labels.example.csv"


def resolve_manifest_csv(meta: dict[str, Any]) -> Path | None:
    """Prefer manifest path saved in ``meta.json`` during training."""
    raw = meta.get("manifest")
    if not raw:
        mp = default_manifest_path()
        return mp if mp.is_file() else None
    p = Path(raw)
    if not p.is_absolute():
        p = project_root() / p
    return p if p.is_file() else None


def _read_manifest_pairs(manifest_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        path_key = next(
            (
                fields[k]
                for k in ("path", "file", "filename", "image", "rel_path")
                if k in fields
            ),
            None,
        )
        label_key = next(
            (fields[k] for k in ("label", "class", "target", "phase") if k in fields),
            None,
        )
        if path_key is None or label_key is None:
            return rows
        for row in reader:
            p = (row.get(path_key) or "").strip()
            lab = (row.get(label_key) or "").strip()
            if p and lab:
                rows.append((p, lab))
    return rows


def manifest_lookup_sha256(data: bytes, data_dir: Path, manifest_path: Path) -> str | None:
    """Exact byte match to a supervised file referenced in ``manifest_path``."""
    mapping: dict[str, str] = {}
    for rel, lab in _read_manifest_pairs(manifest_path):
        fp = (data_dir / rel).resolve()
        if not fp.is_file():
            continue
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        if h not in mapping:
            mapping[h] = lab
    if not mapping:
        return None
    h = hashlib.sha256(data).hexdigest()
    return mapping.get(h)


def _embed_pil(
    model: torch.nn.Module,
    pil_img: Image.Image,
    tfm,
    device: torch.device,
) -> np.ndarray:
    """L2-normalized embedding for cosine similarity."""
    model.eval()
    x = tfm(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.forward_features(x)
        if feat.dim() == 4:
            feat = feat.mean(dim=(2, 3))
        feat = F.normalize(feat.flatten(1), dim=1)
    return feat.cpu().numpy().astype(np.float32).ravel()


def manifest_neighbor_lookup(
    query_pil: Image.Image,
    model: torch.nn.Module,
    tfm,
    device: torch.device,
    data_dir: Path,
    manifest_path: Path,
    *,
    similarity_threshold: float = 0.988,
) -> tuple[str | None, float]:
    """
    Match near-duplicates via cosine similarity to embeddings of manifest-listed images.

    Threshold ~0.988 favors re-uploads / same micrograph vs different phases.
    """
    pairs = _read_manifest_pairs(manifest_path)
    bank_list: list[np.ndarray] = []
    lab_list: list[str] = []
    for rel, lab in pairs:
        fp = (data_dir / rel).resolve()
        if not fp.is_file():
            continue
        pil = Image.open(fp).convert("RGB")
        emb = _embed_pil(model, pil, tfm, device)
        bank_list.append(emb)
        lab_list.append(lab)

    if not bank_list:
        return None, 0.0

    q = _embed_pil(model, query_pil, tfm, device)
    mat = np.stack(bank_list, axis=0)
    sims = mat @ q
    j = int(np.argmax(sims))
    best = float(sims[j])
    if best >= similarity_threshold:
        return lab_list[j], best
    return None, best


def probabilities_for_manifest_label(classes: list[str], label: str) -> dict[str, float]:
    if label in classes:
        return {c: (1.0 if c == label else 0.0) for c in classes}
    base = {c: 0.0 for c in classes}
    base[label] = 1.0
    return base


def neighbor_probabilities(classes: list[str], label: str, sim: float) -> dict[str, float]:
    """Highest mass on ``label`` when it is among ``classes``; remainder split evenly."""
    sim = float(np.clip(sim, 0.0, 1.0))
    if label not in classes:
        return probabilities_for_manifest_label(classes, label)
    other = (1.0 - sim) / max(1, len(classes) - 1)
    return {c: (sim if c == label else other) for c in classes}


def supervised_data_dir() -> Path:
    return fesem_supervised_data_dir()
