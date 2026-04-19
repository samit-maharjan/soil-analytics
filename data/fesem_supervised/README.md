# FESEM supervised training images

Use either:

## A. ImageFolder layout

One subfolder per class name (PyTorch **ImageFolder**).

```text
data/fesem_supervised/
  README.md          (this file)
  Aragonite/
    sample_01.png
  Calcite/
    sample_02.png
```

## B. Flat folder + CSV manifest

Keep images under a folder (e.g. `supervised/`) and list **path** and **label** in a CSV at the repo root or next to data. Example format: `scripts/fesem_labels.example.csv`.

```text
data/fesem_supervised/
  supervised/
    1.png
    2.png
  labels.csv        # optional location; pass --manifest to the train script
```

Requirements:

- **At least two classes** and **at least two images** total (train/validation split).
- Supported extensions follow `torchvision.datasets.ImageFolder` defaults (e.g. `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`).

Train from the repository root (defaults use this folder):

```text
pip install -e ".[ml]"
python scripts/train_fesem_supervised.py
```

Manifest + bottom crop (reduces reliance on the SEM info bar at the bottom of micrographs):

```text
python scripts/train_fesem_supervised.py --manifest scripts/fesem_labels.example.csv --crop-bottom-fraction 0.08
```

Or pass another ImageFolder directory:

```text
python scripts/train_fesem_supervised.py --data-dir /path/to/ImageFolder
```

Outputs are written under `models/fesem_supervised/` by default.
