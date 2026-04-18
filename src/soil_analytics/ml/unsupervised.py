"""Unsupervised FESEM: embeddings, clustering, UMAP, simple anomaly scores."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None

try:
    import hdbscan
except ImportError:  # pragma: no cover
    hdbscan = None


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _imagenet_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_embeddings_from_bytes(
    files: list[tuple[str, bytes]],
    backbone: str = "resnet18",
    img_size: int = 224,
    batch_size: int = 8,
) -> tuple[list[str], np.ndarray]:
    """ImageNet-pretrained timm model with num_classes=0 returns a pooled embedding."""
    device = _device()
    model = timm.create_model(backbone, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    tfm = _imagenet_transform(img_size)
    names: list[str] = []
    vecs: list[np.ndarray] = []

    batch_x: list[torch.Tensor] = []
    batch_n: list[str] = []

    def flush() -> None:
        nonlocal batch_x, batch_n
        if not batch_x:
            return
        xb = torch.stack(batch_x, dim=0).to(device)
        with torch.no_grad():
            emb = model(xb).cpu().numpy()
        for i in range(emb.shape[0]):
            vecs.append(emb[i])
            names.append(batch_n[i])
        batch_x = []
        batch_n = []

    for name, data in files:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        batch_x.append(tfm(img))
        batch_n.append(name)
        if len(batch_x) >= batch_size:
            flush()
    flush()

    if not vecs:
        return [], np.zeros((0, 0), dtype=np.float32)
    mat = np.stack(vecs, axis=0).astype(np.float32)
    return names, mat


def run_unsupervised_pipeline(
    files: list[tuple[str, bytes]],
    backbone: str = "resnet18",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Embeddings -> standardize -> optional UMAP -> HDBSCAN clusters -> Mahalanobis anomaly score.
    """
    names, X = extract_embeddings_from_bytes(files, backbone=backbone)
    if X.size == 0:
        return {"names": [], "embeddings": X, "error": "No images processed."}

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    umap_coords: np.ndarray | None = None
    if umap is not None and len(names) >= 5:
        reducer = umap.UMAP(n_components=2, random_state=random_state, min_dist=0.1)
        umap_coords = reducer.fit_transform(Z)

    labels = np.full(len(names), -1, dtype=int)
    if hdbscan is not None and len(names) >= 3:
        min_cs = max(2, min(5, len(names) // 3))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs)
        labels = clusterer.fit_predict(Z)

    anomalies: list[float] = []
    try:
        cov = EmpiricalCovariance().fit(Z)
        mean = np.mean(Z, axis=0)
        prec = np.linalg.pinv(cov.covariance_)
        for i in range(Z.shape[0]):
            d = Z[i] - mean
            anomalies.append(float(np.sqrt(d @ prec @ d)))
    except Exception:
        anomalies = [0.0] * len(names)

    return {
        "names": names,
        "embeddings": X,
        "umap": umap_coords,
        "cluster_labels": labels.tolist(),
        "anomaly_score": anomalies,
        "backbone": backbone,
    }
