#!/usr/bin/env python3
"""Train supervised FESEM classifier from an ImageFolder tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from soil_analytics.ml.supervised import train_supervised


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="ImageFolder root with class subfolders",
    )
    p.add_argument("--out-dir", type=Path, default=Path("models/fesem_supervised"))
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    args = p.parse_args()

    out = train_supervised(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        backbone=args.backbone,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(out)


if __name__ == "__main__":
    main()
