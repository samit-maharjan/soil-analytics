#!/usr/bin/env python3
"""Download a Kaggle dataset into cache using kagglehub (optional [kaggle] extra)."""

from __future__ import annotations

import argparse


def main() -> None:
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "kagglehub is not installed. Run: pip install soil-analytics[kaggle]"
        ) from e

    parser = argparse.ArgumentParser(description="Download a Kaggle dataset by slug.")
    parser.add_argument(
        "slug",
        help='Dataset slug, e.g. "username/dataset-name"',
    )
    args = parser.parse_args()
    path = kagglehub.dataset_download(args.slug)
    print("Downloaded to:", path)


if __name__ == "__main__":
    main()
