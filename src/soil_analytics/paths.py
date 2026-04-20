"""Resolve project paths (config, models) from package or cwd."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """
    Repository root containing `config/reference_ranges/`.

    Prefer the source tree next to `src/soil_analytics`; fall back to walking up from cwd
    so `streamlit run` works when the cwd is the repo root.
    """
    here = Path(__file__).resolve()
    # src/soil_analytics/paths.py -> src -> root
    from_src = here.parents[2]
    if (from_src / "config" / "reference_ranges").is_dir():
        return from_src
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "config" / "reference_ranges").is_dir():
            return base
    return from_src


def reference_config_dir() -> Path:
    return project_root() / "config" / "reference_ranges"


def models_dir() -> Path:
    return project_root() / "models"


def fesem_supervised_data_dir() -> Path:
    """Root for paired FESEM data: ``micrographs/`` and ``analysis/`` (see data/fesem_supervised/README.md)."""
    return project_root() / "data" / "fesem_supervised"


def fesem_micrographs_dir() -> Path:
    return fesem_supervised_data_dir() / "micrographs"


def fesem_analysis_dir() -> Path:
    return fesem_supervised_data_dir() / "analysis"
