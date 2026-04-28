"""Resolve project paths (config) from package or cwd."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """
    Repository root containing `config/reference_ranges/`.

    Prefer the source tree next to `src/lime_analytics`; fall back to walking up from cwd
    so `streamlit run` works when the cwd is the repo root.
    """
    here = Path(__file__).resolve()
    # src/lime_analytics/paths.py -> src -> root
    from_src = here.parents[2]
    if (from_src / "config" / "reference_ranges").is_dir():
        return from_src
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "config" / "reference_ranges").is_dir():
            return base
    return from_src


def reference_config_dir() -> Path:
    return project_root() / "config" / "reference_ranges"
