#!/usr/bin/env python3
"""Train supervised FESEM classifier; optionally emit a round-robin manifest CSV draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from soil_analytics.ml.supervised import train_supervised
from soil_analytics.paths import (
    fesem_supervised_data_dir,
    project_root,
    reference_config_dir,
)


def write_round_robin_manifest(
    *,
    data_dir: Path,
    out_path: Path | None,
    short_labels: bool,
    root: Path,
) -> None:
    """Assign phase labels from ``fesem_remarks.yaml`` round-robin to ``supervised/*`` images."""
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
        if short_labels:
            labels.append(str(row.get("id", row.get("label", ""))).strip())
        else:
            labels.append(str(row.get("label", row.get("id", ""))).strip())
    labels = [x for x in labels if x]
    if len(labels) < 2:
        raise SystemExit("Need at least two phase entries in fesem_remarks.yaml.")

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    raw_paths: list[Path] = []
    for fp in sup.iterdir():
        if fp.is_file() and fp.suffix.lower() in exts:
            raw_paths.append(fp)

    def sort_key(p: Path) -> tuple[int, int | str]:
        stem = p.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem.lower())

    images = sorted(raw_paths, key=sort_key)

    if len(images) < 2:
        raise SystemExit(f"Need at least two images under {sup}")

    lines = ["path,label"]
    for i, img in enumerate(images):
        rel = img.relative_to(data_dir).as_posix()
        lab = labels[i % len(labels)]
        lines.append(f"{rel},{lab}")

    text = "\n".join(lines) + "\n"
    if out_path is not None:
        out_full = out_path if out_path.is_absolute() else root / out_path
        out_full.parent.mkdir(parents=True, exist_ok=True)
        out_full.write_text(text, encoding="utf-8")
        print(f"Wrote {len(images)} rows to {out_full}")
    else:
        print(text, end="")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Train supervised FESEM classifier from ImageFolder or a CSV manifest; "
            "or use --write-manifest to draft a CSV from fesem_remarks.yaml "
            "(see data/fesem_supervised/README.md)."
        ),
    )
    p.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "Write a round-robin path,label CSV from config/reference_ranges/fesem_remarks.yaml "
            "and images under <data-dir>/supervised/. Use --manifest-out PATH or stdout."
        ),
    )
    p.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="With --write-manifest: output file (omit to print CSV to stdout).",
    )
    p.add_argument(
        "--short-labels",
        action="store_true",
        help='With --write-manifest: use id-style labels (e.g. "csh") vs YAML display names.',
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Dataset root (default: {fesem_supervised_data_dir()})",
    )
    p.add_argument("--out-dir", type=Path, default=Path("models/fesem_supervised"))
    p.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        help=(
            "timm model name with ImageNet pretrained weights (e.g. efficientnet_b0, "
            "resnet50, convnext_tiny, regnety_032)."
        ),
    )
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV with path,label columns; paths relative to --data-dir "
            "(flat supervised/ layout)."
        ),
    )
    p.add_argument(
        "--crop-bottom-fraction",
        type=float,
        default=0.0,
        help=(
            "Remove bottom fraction of each image height before resize "
            "(reduces SEM scale bar/metadata leakage)."
        ),
    )
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable geometric/color augmentation during training.",
    )
    p.add_argument(
        "--no-strong-augment",
        action="store_true",
        help="Disable extra FESEM-oriented augmentation (affine, jitter, blur, random erasing).",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of images for validation (stratified when possible).",
    )
    p.add_argument(
        "--no-stratified",
        action="store_true",
        help="Use a random train/val split instead of stratified.",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.08,
        help="Cross-entropy label smoothing (0–0.35).",
    )
    p.add_argument(
        "--no-balance-sampler",
        action="store_true",
        help="Disable class-balanced sampling for imbalanced batches.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=14,
        help="Early stopping patience on validation accuracy (0 disables).",
    )
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--train-duplicates",
        type=int,
        default=1,
        help=(
            "Repeat each training sample this many times in the training set each epoch "
            "(more optimizer steps on the same labeled images; try 2–4 for tiny datasets)."
        ),
    )
    p.add_argument(
        "--neighbor-similarity-threshold",
        type=float,
        default=0.988,
        help=(
            "Stored in meta.json for inference: cosine similarity gate for embedding-neighbor "
            "(exact manifest bytes match still wins)."
        ),
    )
    p.add_argument(
        "--no-small-set-auto",
        action="store_true",
        help=(
            "Disable automatic settings for tiny train splits (≤28 unique train images): "
            "normally strong augment is reduced, repeats are bumped, label smoothing capped."
        ),
    )
    p.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.15,
        help=(
            "Multiplier on --lr for pretrained backbone weights (classification head uses full "
            "--lr). Typical fine-tuning: 0.05–0.25. Ignored with --uniform-lr."
        ),
    )
    p.add_argument(
        "--uniform-lr",
        action="store_true",
        help="Use one learning rate for all layers (legacy behavior).",
    )
    p.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help=(
            "Train only the classification head for this many epochs (backbone frozen), "
            "then fine-tune the full network with backbone LR multiplier. Useful for tiny sets."
        ),
    )
    args = p.parse_args()

    root = project_root()
    data_dir = args.data_dir if args.data_dir is not None else fesem_supervised_data_dir()

    if args.write_manifest:
        write_round_robin_manifest(
            data_dir=data_dir,
            out_path=args.manifest_out,
            short_labels=args.short_labels,
            root=root,
        )
        return

    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    manifest = args.manifest
    if manifest is not None and not manifest.is_absolute():
        manifest = root / manifest

    out = train_supervised(
        data_dir=data_dir,
        out_dir=out_dir,
        backbone=args.backbone,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        manifest=manifest,
        crop_bottom_fraction=float(args.crop_bottom_fraction),
        augment=not args.no_augment,
        strong_augment=not args.no_strong_augment,
        val_fraction=float(args.val_fraction),
        stratified_split=not args.no_stratified,
        label_smoothing=float(args.label_smoothing),
        balance_sampler=not args.no_balance_sampler,
        patience=int(args.patience),
        split_seed=int(args.split_seed),
        train_sample_duplicates=max(1, int(args.train_duplicates)),
        neighbor_similarity_threshold=float(args.neighbor_similarity_threshold),
        small_set_auto=not args.no_small_set_auto,
        backbone_lr_mult=float(args.backbone_lr_mult),
        uniform_lr=args.uniform_lr,
        freeze_backbone_epochs=int(args.freeze_backbone_epochs),
    )
    print(out)


if __name__ == "__main__":
    main()
