#!/usr/bin/env python3
"""Download a Kaggle dataset into data/external/ using kagglehub."""

from __future__ import annotations

import argparse

import kagglehub


def main() -> None:
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
