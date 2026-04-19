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

Keep images under a folder (e.g. `supervised/`) and list **path** and **label** in a CSV. Example: `scripts/fesem_labels.example.csv` lists all `supervised/1.png`–`11.png` with **seven** phase names (Aragonite, Vaterite, Portlandite, Calcite, CSH, Ettringite, ACC) in a round-robin — **you must correct each row** to the true phase for that micrograph. The model can only predict classes that appear in the manifest; more classes need more (or relabeled) images.

**Regenerate a round-robin draft** from `config/reference_ranges/fesem_remarks.yaml` and whatever files are in `supervised/`:

```text
python scripts/train_fesem_supervised.py --write-manifest --manifest-out scripts/my_fesem_labels.csv
```

Omit `--manifest-out` to print the CSV to stdout instead of a file.

Then edit `my_fesem_labels.csv` and train with `--manifest scripts/my_fesem_labels.csv`.

The default backbone is **EfficientNet-B0** (ImageNet pretrained via timm). Alternatives include `--backbone resnet50`, `--backbone convnext_tiny`, or `--backbone regnety_032`; the first download of weights may take a minute.

```text
data/fesem_supervised/
  supervised/
    1.png
    2.png
  labels.csv        # optional location; pass --manifest to the train script
```

Requirements:

- **At least two classes** in the label column and **at least two images** total (train/validation split). You can have as many classes as you have distinct `label` values (e.g. all cement / CaCO₃–related phases you care about), as long as the split still leaves at least one image in train and one in val.
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
