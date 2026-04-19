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

Keep images under a folder (e.g. `supervised/`) and list **path** and **label** in a CSV. Example: `scripts/fesem_labels.example.csv` includes all `supervised/1.png`–`11.png` rows for reference; **edit every label to match your microscopy** (the example alternation is only a template, not ground truth).

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

Manifest + bottom crop (reduces reliance on the SEM info bar) and the default **robust** training settings (stratified val split, strong augment, label smoothing, cosine LR, class-balanced sampling when imbalanced, early stopping):

```text
python scripts/train_fesem_supervised.py --manifest scripts/fesem_labels.example.csv --crop-bottom-fraction 0.08
```

Tuning (optional): `--epochs 50 --patience 20 --val-fraction 0.25 --no-strong-augment` (simpler aug if you have very few images).

Repeat exposure to each labeled training file per epoch (helps small sets converge more consistently):

```text
python scripts/train_fesem_supervised.py --manifest scripts/fesem_labels.example.csv --train-duplicates 3
```

After training, in the FESEM app you can enable **TTA** (mirror-average at inference) so the same upload gets more stable probability scores.

Or pass another ImageFolder directory:

```text
python scripts/train_fesem_supervised.py --data-dir /path/to/ImageFolder
```

Outputs are written under `models/fesem_supervised/` by default.
