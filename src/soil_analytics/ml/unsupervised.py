"""Unsupervised FESEM: embeddings, PCA 2D view, clustering, Mahalanobis-style anomaly scores."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import transforms


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
    Embeddings -> standardize -> PCA (2D) -> k-means clusters -> Mahalanobis anomaly score.

    Uses PCA instead of UMAP to avoid an extra heavy dependency; interpret as linear projection.
    """
    names, X = extract_embeddings_from_bytes(files, backbone=backbone)
    if X.size == 0:
        return {"names": [], "embeddings": X, "error": "No images processed."}

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    embedding_2d: np.ndarray | None = None
    if len(names) >= 3:
        n_comp = min(2, Z.shape[0] - 1, Z.shape[1])
        if n_comp >= 1:
            pca = PCA(n_components=n_comp, random_state=random_state)
            embedding_2d = pca.fit_transform(Z)
            if embedding_2d.shape[1] == 1:
                embedding_2d = np.column_stack([embedding_2d, np.zeros(len(names))])

    labels = np.full(len(names), -1, dtype=int)
    if len(names) >= 3:
        k = max(2, min(5, len(names) // 2))
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(Z)

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
        "embedding_2d": embedding_2d,
        "cluster_labels": labels.tolist(),
        "anomaly_score": anomalies,
        "backbone": backbone,
    }
