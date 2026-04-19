#!/usr/bin/env python3
"""Build a CSV manifest by assigning phases (from fesem_remarks.yaml) to images in supervised/."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from soil_analytics.paths import fesem_supervised_data_dir, project_root, reference_config_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Round-robin assign phase labels from config/reference_ranges/fesem_remarks.yaml "
            "to images in data/fesem_supervised/supervised/. Edit the CSV before training if "
            "you know the true phase per micrograph."
        ),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Dataset root containing supervised/ (default: {fesem_supervised_data_dir()})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: print to stdout only)",
    )
    p.add_argument(
        "--short-labels",
        action="store_true",
        help='Use id-style labels (e.g. "csh" for C–S–H) instead of YAML display names.',
    )
    args = p.parse_args()

    root = project_root()
    data_dir = args.data_dir if args.data_dir is not None else fesem_supervised_data_dir()
    sup = data_dir / "supervised"
    if not sup.is_dir():
        raise SystemExit(f"Missing folder: {sup}")

    yaml_path = reference_config_dir() / "fesem_remarks.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    phases = cfg.get("phases") or []
    if not phases:
        raise SystemExit(f"No phases in {yaml_path}")

    labels: list[str] = []
    for row in phases:
        if args.short_labels:
            labels.append(str(row.get("id", row.get("label", ""))).strip())
        else:
            labels.append(str(row.get("label", row.get("id", ""))).strip())
    labels = [x for x in labels if x]
    if len(labels) < 2:
        raise SystemExit("Need at least two phase entries in fesem_remarks.yaml.")

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    raw: list[Path] = []
    for fp in sup.iterdir():
        if fp.is_file() and fp.suffix.lower() in exts:
            raw.append(fp)

    def sort_key(p: Path) -> tuple[int, int | str]:
        stem = p.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem.lower())

    images = sorted(raw, key=sort_key)

    if len(images) < 2:
        raise SystemExit(f"Need at least two images under {sup}")

    rows: list[tuple[str, str]] = []
    for i, img in enumerate(images):
        rel = img.relative_to(data_dir).as_posix()
        lab = labels[i % len(labels)]
        rows.append((rel, lab))

    lines = ["path,label"]
    lines.extend(f"{a},{b}" for a, b in rows)

    text = "\n".join(lines) + "\n"
    if args.out is not None:
        out_path = args.out if args.out.is_absolute() else root / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {out_path}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
