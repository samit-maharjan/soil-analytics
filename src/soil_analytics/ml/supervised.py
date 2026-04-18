"""Supervised FESEM / SEM image classifier training and inference."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def train_supervised(
    data_dir: Path | str,
    out_dir: Path | str,
    backbone: str = "resnet18",
    img_size: int = 224,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> dict[str, Any]:
    """
    Train a classifier on an ImageFolder layout: data_dir/class_name/*.png|jpg.

    Saves model.pth, meta.json, and optional metrics.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()

    tfm = _build_transforms(img_size)
    ds = datasets.ImageFolder(str(data_dir), transform=tfm)
    if len(ds.classes) < 2:
        raise ValueError("Need at least 2 classes (subfolders) for supervised training.")

    n = len(ds)
    n_val = max(1, int(0.15 * n))
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = set(perm[:n_val])
    train_idx = [i for i in range(n) if i not in val_idx]
    val_indices = list(val_idx)

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = timm.create_model(backbone, pretrained=True, num_classes=len(ds.classes))
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
        "classes": ds.classes,
        "class_to_idx": {k: int(v) for k, v in ds.class_to_idx.items()},
        "img_size": img_size,
        "epochs": epochs,
        "best_val_loss": float(best_val),
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
    state = torch.load(out_dir / "model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    img_size = int(meta.get("img_size", 224))
    tfm = _build_transforms(img_size)
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
