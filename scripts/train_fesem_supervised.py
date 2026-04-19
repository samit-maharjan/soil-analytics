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
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
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
        help="Disable light random flips/rotations during training.",
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
    )
    print(out)


if __name__ == "__main__":
    main()
