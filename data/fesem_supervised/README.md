# FESEM supervised training images

Place labeled images here using PyTorch **ImageFolder** layout: one subfolder per class name.

```text
data/fesem_supervised/
  README.md          (this file)
  class_a/           # example — use your own labels
    img001.png
    img002.tif
  class_b/
    img010.jpg
```

Requirements:

- **At least two classes** (two subfolders) with at least one image each.
- Supported extensions follow `torchvision.datasets.ImageFolder` defaults (e.g. `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`).

Train from the repository root (defaults use this folder):

```text
pip install -e ".[ml]"
python scripts/train_fesem_supervised.py
```

Or pass another directory explicitly:

```text
python scripts/train_fesem_supervised.py --data-dir /path/to/ImageFolder
```

Outputs are written under `models/fesem_supervised/` by default.
