"""Supervised FESEM / SEM image classifier training and inference."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _BottomCropFraction:
    """Remove the bottom fraction of the image height (e.g. SEM metadata bar)."""

    def __init__(self, fraction: float) -> None:
        if not 0.0 <= fraction < 0.95:
            raise ValueError("crop_bottom_fraction must be in [0, 0.95).")
        self.fraction = fraction

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.fraction <= 0:
            return img
        w, h = img.size
        new_h = max(1, int(h * (1.0 - self.fraction)))
        return img.crop((0, 0, w, new_h))


def _build_transforms(
    img_size: int,
    *,
    crop_bottom_fraction: float = 0.0,
    train: bool = False,
    augment: bool = True,
) -> transforms.Compose:
    crop = []
    if crop_bottom_fraction > 0:
        crop.append(_BottomCropFraction(crop_bottom_fraction))

    geo: list[Any] = []
    if train and augment:
        geo.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=12),
            ]
        )

    tail = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose([*crop, *geo, *tail])


def _read_manifest_rows(manifest_path: Path) -> list[tuple[str, str]]:
    """Return (relative_path, label) from CSV with header."""
    rows: list[tuple[str, str]] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Manifest CSV has no header row.")
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        path_key = next(
            (
                fields[k]
                for k in ("path", "file", "filename", "image", "rel_path")
                if k in fields
            ),
            None,
        )
        label_key = next(
            (fields[k] for k in ("label", "class", "target", "phase") if k in fields),
            None,
        )
        if path_key is None or label_key is None:
            raise ValueError(
                "Manifest CSV needs columns for path "
                "(path/file/filename/…) and label (label/class/…)."
            )
        for row in reader:
            p = (row.get(path_key) or "").strip()
            lab = (row.get(label_key) or "").strip()
            if not p or not lab:
                continue
            rows.append((p, lab))
    if not rows:
        raise ValueError("Manifest CSV has no data rows.")
    return rows


class _ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        root: Path,
        samples: list[tuple[Path, int]],
        transform: transforms.Compose,
    ) -> None:
        self.root = root
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, target


def _resolve_manifest_samples(
    data_dir: Path, manifest_path: Path
) -> tuple[list[tuple[Path, int]], list[str]]:
    raw = _read_manifest_rows(manifest_path)
    classes = sorted({lab for _, lab in raw})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples: list[tuple[Path, int]] = []
    for rel, lab in raw:
        fp = (data_dir / rel).resolve()
        if not fp.is_file():
            raise FileNotFoundError(f"Manifest path not found: {rel} -> {fp}")
        samples.append((fp, class_to_idx[lab]))
    return samples, classes


def train_supervised(
    data_dir: Path | str,
    out_dir: Path | str,
    backbone: str = "resnet18",
    img_size: int = 224,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    manifest: Path | str | None = None,
    crop_bottom_fraction: float = 0.0,
    augment: bool = True,
) -> dict[str, Any]:
    """
    Train a classifier on either:

    - PyTorch ImageFolder layout: ``data_dir/class_name/*.png``
    - Or a CSV manifest (``path,label`` columns) with image paths relative to ``data_dir``.

    Saves model.pth, meta.json.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()

    classes: list[str]
    train_ds: Dataset[tuple[torch.Tensor, int]]
    val_ds: Dataset[tuple[torch.Tensor, int]]

    if manifest is not None:
        manifest_path = Path(manifest)
        samples, classes = _resolve_manifest_samples(data_dir, manifest_path)
        if len(classes) < 2:
            raise ValueError(
                "Need at least 2 distinct labels in the manifest for supervised training."
            )
        n = len(samples)
        if n < 2:
            raise ValueError(
                "Need at least 2 manifest rows for supervised training with a validation split."
            )
        n_val = max(1, int(0.15 * n))
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(n, generator=g).tolist()
        val_idx = set(perm[:n_val])
        train_pairs = [samples[i] for i in range(n) if i not in val_idx]
        val_pairs = [samples[i] for i in range(n) if i in val_idx]

        train_tfm = _build_transforms(
            img_size,
            crop_bottom_fraction=crop_bottom_fraction,
            train=True,
            augment=augment,
        )
        val_tfm = _build_transforms(
            img_size, crop_bottom_fraction=crop_bottom_fraction, train=False, augment=False
        )
        if len(train_pairs) < 1 or len(val_pairs) < 1:
            raise ValueError(
                "Train/val split produced an empty split; "
                "add more labeled images or adjust the manifest."
            )
        train_ds = _ManifestDataset(data_dir, train_pairs, train_tfm)
        val_ds = _ManifestDataset(data_dir, val_pairs, val_tfm)
    else:
        train_tfm = _build_transforms(
            img_size,
            crop_bottom_fraction=crop_bottom_fraction,
            train=True,
            augment=augment,
        )
        val_tfm = _build_transforms(
            img_size, crop_bottom_fraction=crop_bottom_fraction, train=False, augment=False
        )
        ds = datasets.ImageFolder(str(data_dir), transform=val_tfm)
        if len(ds.classes) < 2:
            raise ValueError("Need at least 2 classes (subfolders) for supervised training.")
        classes = ds.classes
        n = len(ds)
        if n < 2:
            raise ValueError(
                "Need at least 2 images total for supervised training with a validation split."
            )
        n_val = max(1, int(0.15 * n))
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(n, generator=g).tolist()
        val_idx = set(perm[:n_val])
        train_idx = [i for i in range(n) if i not in val_idx]
        val_indices = list(val_idx)

        # Build train subset with augmentation by wrapping samples (reload from disk).
        train_samples: list[tuple[Path, int]] = []
        for i in train_idx:
            path, target = ds.samples[i]
            train_samples.append((Path(path), target))
        val_samples = [ds.samples[i] for i in val_indices]
        train_ds = _ManifestDataset(
            data_dir,
            [(Path(p), t) for p, t in train_samples],
            train_tfm,
        )
        val_ds = _ManifestDataset(
            data_dir,
            [(Path(p), t) for p, t in val_samples],
            val_tfm,
        )
        if len(train_ds) < 1 or len(val_ds) < 1:
            raise ValueError(
                "Train/val split produced an empty split; "
                "add more images per class or consolidate classes."
            )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = timm.create_model(backbone, pretrained=True, num_classes=len(classes))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                val_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        val_loss /= len(val_ds)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "backbone": backbone,
        "classes": classes,
        "class_to_idx": {k: int(i) for i, k in enumerate(classes)},
        "img_size": img_size,
        "epochs": epochs,
        "best_val_loss": float(best_val),
        "crop_bottom_fraction": float(crop_bottom_fraction),
        "manifest": str(Path(manifest).as_posix()) if manifest is not None else None,
    }
    torch.save(model.state_dict(), out_dir / "model.pth")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"out_dir": str(out_dir), "meta": meta}


def _load_model_for_inference(
    out_dir: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any], transforms.Compose]:
    with open(out_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    backbone = meta["backbone"]
    num_classes = len(meta["classes"])
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    weights_path = out_dir / "model.pth"
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    img_size = int(meta.get("img_size", 224))
    crop = float(meta.get("crop_bottom_fraction", 0.0))
    tfm = _build_transforms(img_size, crop_bottom_fraction=crop, train=False, augment=False)
    return model, meta, tfm


def predict_images(
    image_paths: list[Path | str],
    model_dir: Path | str,
) -> list[dict[str, Any]]:
    """Return class probabilities for each image path."""
    model_dir = Path(model_dir)
    device = _device()
    model, meta, tfm = _load_model_for_inference(model_dir, device)
    classes: list[str] = meta["classes"]
    results: list[dict[str, Any]] = []

    for p in image_paths:
        p = Path(p)
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy().ravel()
        top = int(np.argmax(prob))
        results.append(
            {
                "path": str(p),
                "predicted_class": classes[top],
                "confidence": float(prob[top]),
                "probabilities": {classes[i]: float(prob[i]) for i in range(len(classes))},
            }
        )
    return results


def predict_images_from_bytes(
    files: list[tuple[str, bytes]],
    model_dir: Path | str,
) -> list[dict[str, Any]]:
    """Predict from (filename, bytes) pairs (e.g. Streamlit uploads)."""
    model_dir = Path(model_dir)
    device = _device()
    model, meta, tfm = _load_model_for_inference(model_dir, device)
    classes: list[str] = meta["classes"]
    results: list[dict[str, Any]] = []
    for name, data in files:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy().ravel()
        top = int(np.argmax(prob))
        results.append(
            {
                "path": name,
                "predicted_class": classes[top],
                "confidence": float(prob[top]),
                "probabilities": {classes[i]: float(prob[i]) for i in range(len(classes))},
            }
        )
    return results
