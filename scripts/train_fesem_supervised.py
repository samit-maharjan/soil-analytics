#!/usr/bin/env python3
"""Train supervised FESEM classifier from an ImageFolder tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from soil_analytics.ml.supervised import train_supervised
from soil_analytics.paths import fesem_supervised_data_dir, project_root


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Train supervised FESEM classifier from a local ImageFolder "
            "(see data/fesem_supervised/README.md)."
        ),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"ImageFolder root with class subfolders (default: {fesem_supervised_data_dir()})",
    )
    p.add_argument("--out-dir", type=Path, default=Path("models/fesem_supervised"))
    p.add_argument("--backbone", type=str, default="resnet18")
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
    args = p.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else fesem_supervised_data_dir()
    out_dir = args.out_dir if args.out_dir.is_absolute() else project_root() / args.out_dir
    manifest = args.manifest
    if manifest is not None and not manifest.is_absolute():
        manifest = project_root() / manifest

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
    )
    print(out)


if __name__ == "__main__":
    main()
