"""Supervised FESEM / SEM image classifier training and inference."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from soil_analytics.ml.fesem_overlay import annotate_prediction_dict
from soil_analytics.paths import fesem_supervised_data_dir, project_root


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
    strong_augment: bool = False,
    random_erasing_p: float = 0.12,
) -> transforms.Compose:
    crop = []
    if crop_bottom_fraction > 0:
        crop.append(_BottomCropFraction(crop_bottom_fraction))

    geo: list[Any] = []
    if train and augment:
        geo.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.35),
                transforms.RandomRotation(degrees=18),
            ]
        )
        if strong_augment:
            geo.extend(
                [
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.06, 0.08),
                        scale=(0.84, 1.14),
                        shear=(-5, 5),
                    ),
                    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.12),
                    transforms.RandomApply(
                        [
                            transforms.GaussianBlur(kernel_size=3, sigma=(0.15, 1.1)),
                        ],
                        p=0.28,
                    ),
                ]
            )

    tail: list[Any] = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train and augment and strong_augment and random_erasing_p > 0:
        tail.append(transforms.RandomErasing(p=random_erasing_p, scale=(0.02, 0.12), ratio=(0.5, 2.0)))

    return transforms.Compose([*crop, *geo, *tail])


def _train_val_indices_stratified(targets: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Stratified split when possible; otherwise random shuffle split."""
    n = len(targets)
    if n < 2:
        raise ValueError("Need at least 2 samples for train/val split.")
    idx = np.arange(n, dtype=np.int64)
    y = np.array(targets, dtype=np.int64)
    n_val = max(1, int(round(val_fraction * n)))
    n_val = min(n_val, n - 1)

    counts = Counter(targets)
    min_per_class = min(counts.values()) if counts else 0
    use_stratify = min_per_class >= 2 and len(counts) >= 2

    if use_stratify:
        try:
            from sklearn.model_selection import train_test_split

            train_idx, val_idx = train_test_split(
                idx,
                test_size=n_val / n,
                stratify=y,
                random_state=seed,
                shuffle=True,
            )
            return train_idx.tolist(), val_idx.tolist()
        except ValueError:
            pass

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _pairs_from_indices(
    samples: list[tuple[Path, int]], train_idx: list[int], val_idx: list[int]
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    train_pairs = [samples[i] for i in train_idx]
    val_pairs = [samples[i] for i in val_idx]
    return train_pairs, val_pairs


def _expand_train_duplicates(
    train_pairs: list[tuple[Path, int]], duplicates: int
) -> list[tuple[Path, int]]:
    if duplicates < 1:
        raise ValueError("train_sample_duplicates must be >= 1.")
    if duplicates == 1:
        return train_pairs
    return [p for p in train_pairs for _ in range(duplicates)]


def _probabilities_for_pil(
    pil_img: Image.Image,
    model: nn.Module,
    tfm: transforms.Compose,
    device: torch.device,
    *,
    tta: bool,
) -> np.ndarray:
    """
    Class probabilities; if ``tta`` is True, average logits with a horizontal flip
    (stabler, more repeatable scores on the same micrograph).
    """
    x0 = tfm(pil_img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        l0 = model(x0)
        if not tta:
            prob = torch.softmax(l0, dim=1)
        else:
            xf = torch.flip(x0, dims=[3])
            lf = model(xf)
            prob = torch.softmax((l0 + lf) * 0.5, dim=1)
    return prob.cpu().numpy().ravel()


def _manifest_csv_pairs(manifest_path: Path, *, strict: bool = True) -> list[tuple[str, str]]:
    """Parse path/label columns; ``strict`` raises if the CSV is unusable for training."""
    rows: list[tuple[str, str]] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            if strict:
                raise ValueError("Manifest CSV has no header row.")
            return rows
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
            if strict:
                raise ValueError(
                    "Manifest CSV needs columns for path "
                    "(path/file/filename/…) and label (label/class/…)."
                )
            return rows
        for row in reader:
            p = (row.get(path_key) or "").strip()
            lab = (row.get(label_key) or "").strip()
            if p and lab:
                rows.append((p, lab))
    if strict and not rows:
        raise ValueError("Manifest CSV has no data rows.")
    return rows


def _read_manifest_rows(manifest_path: Path) -> list[tuple[str, str]]:
    """Return (relative_path, label) from CSV with header."""
    return _manifest_csv_pairs(manifest_path, strict=True)


def default_manifest_path() -> Path:
    return project_root() / "scripts" / "fesem_labels.example.csv"


def resolve_manifest_csv(meta: dict[str, Any]) -> Path | None:
    """Prefer manifest path saved in ``meta.json`` during training."""
    raw = meta.get("manifest")
    if not raw:
        mp = default_manifest_path()
        return mp if mp.is_file() else None
    p = Path(raw)
    if not p.is_absolute():
        p = project_root() / p
    return p if p.is_file() else None


def manifest_lookup_sha256(data: bytes, data_dir: Path, manifest_path: Path) -> str | None:
    """Exact byte match to a supervised file referenced in ``manifest_path``."""
    mapping: dict[str, str] = {}
    for rel, lab in _manifest_csv_pairs(manifest_path, strict=False):
        fp = (data_dir / rel).resolve()
        if not fp.is_file():
            continue
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        if h not in mapping:
            mapping[h] = lab
    if not mapping:
        return None
    h = hashlib.sha256(data).hexdigest()
    return mapping.get(h)


def _embed_pil(
    model: nn.Module,
    pil_img: Image.Image,
    tfm,
    device: torch.device,
) -> np.ndarray:
    """L2-normalized embedding for cosine similarity."""
    model.eval()
    x = tfm(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.forward_features(x)
        if feat.dim() == 4:
            feat = feat.mean(dim=(2, 3))
        feat = F.normalize(feat.flatten(1), dim=1)
    return feat.cpu().numpy().astype(np.float32).ravel()


def manifest_neighbor_lookup(
    query_pil: Image.Image,
    model: nn.Module,
    tfm,
    device: torch.device,
    data_dir: Path,
    manifest_path: Path,
    *,
    similarity_threshold: float = 0.988,
) -> tuple[str | None, float]:
    """Near-duplicate match via cosine similarity to embeddings of manifest-listed images."""
    pairs = _manifest_csv_pairs(manifest_path, strict=False)
    bank_list: list[np.ndarray] = []
    lab_list: list[str] = []
    for rel, lab in pairs:
        fp = (data_dir / rel).resolve()
        if not fp.is_file():
            continue
        pil = Image.open(fp).convert("RGB")
        emb = _embed_pil(model, pil, tfm, device)
        bank_list.append(emb)
        lab_list.append(lab)

    if not bank_list:
        return None, 0.0

    q = _embed_pil(model, query_pil, tfm, device)
    mat = np.stack(bank_list, axis=0)
    sims = mat @ q
    j = int(np.argmax(sims))
    best = float(sims[j])
    if best >= similarity_threshold:
        return lab_list[j], best
    return None, best


def probabilities_for_manifest_label(classes: list[str], label: str) -> dict[str, float]:
    if label in classes:
        return {c: (1.0 if c == label else 0.0) for c in classes}
    base = {c: 0.0 for c in classes}
    base[label] = 1.0
    return base


def neighbor_probabilities(classes: list[str], label: str, sim: float) -> dict[str, float]:
    """Highest mass on ``label`` when it is among ``classes``; remainder split evenly."""
    sim = float(np.clip(sim, 0.0, 1.0))
    if label not in classes:
        return probabilities_for_manifest_label(classes, label)
    other = (1.0 - sim) / max(1, len(classes) - 1)
    return {c: (sim if c == label else other) for c in classes}


def supervised_data_dir() -> Path:
    return fesem_supervised_data_dir()


def _split_classifier_vs_backbone(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Split timm model parameters into backbone vs classification head for transfer LR.
    Uses ``get_classifier()`` when available (ResNet, EfficientNet, ConvNeXt, …).
    """
    if hasattr(model, "get_classifier"):
        cls_mod = model.get_classifier()
        cls_ids = {id(p) for p in cls_mod.parameters()}
        backbone_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []
        for p in model.parameters():
            if id(p) in cls_ids:
                head_params.append(p)
            else:
                backbone_params.append(p)
        if head_params:
            return backbone_params, head_params

    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        nl = name.lower()
        if (
            "classifier" in nl
            or nl.startswith("head.")
            or ".head." in nl
            or nl == "fc.weight"
            or nl == "fc.bias"
            or ".fc." in nl
        ):
            head_params.append(p)
        else:
            backbone_params.append(p)
    if not head_params:
        return list(model.parameters()), []
    return backbone_params, head_params


def _optimizer_param_groups(
    model: nn.Module,
    lr: float,
    backbone_lr_mult: float,
    *,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Lower LR on backbone weights (ImageNet pretrain); full ``lr`` on the new head."""
    bp, hp = _split_classifier_vs_backbone(model)
    if not hp or backbone_lr_mult >= 0.999:
        return [{"params": list(model.parameters()), "lr": float(lr), "weight_decay": weight_decay}]
    return [
        {"params": bp, "lr": float(lr) * float(backbone_lr_mult), "weight_decay": weight_decay},
        {"params": hp, "lr": float(lr), "weight_decay": weight_decay},
    ]


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
    backbone: str = "efficientnet_b0",
    img_size: int = 224,
    epochs: int = 35,
    batch_size: int = 8,
    lr: float = 1e-4,
    manifest: Path | str | None = None,
    crop_bottom_fraction: float = 0.0,
    augment: bool = True,
    strong_augment: bool = True,
    val_fraction: float = 0.2,
    stratified_split: bool = True,
    label_smoothing: float = 0.08,
    balance_sampler: bool = True,
    patience: int = 14,
    split_seed: int = 42,
    random_erasing_p: float = 0.12,
    train_sample_duplicates: int = 1,
    small_set_auto: bool = True,
    neighbor_similarity_threshold: float = 0.988,
    *,
    backbone_lr_mult: float = 0.15,
    uniform_lr: bool = False,
    freeze_backbone_epochs: int = 0,
) -> dict[str, Any]:
    """
    Train a classifier on either:

    - PyTorch ImageFolder layout: ``data_dir/class_name/*.png``
    - Or a CSV manifest (``path,label`` columns) with image paths relative to ``data_dir``.

    **Transfer learning:** ``backbone`` is loaded from **ImageNet-pretrained** weights via timm
    (e.g. ``efficientnet_b0``, ``resnet50``, ``convnext_tiny``, ``regnety_032``).
    By default the optimizer uses a **lower LR on backbone** (``backbone_lr_mult``) vs the
    classification head — standard fine-tuning for small microscopy sets.

    Uses stratified train/val split when possible, FESEM-oriented augmentation, label smoothing,
    cosine LR decay, checkpoint by validation accuracy, and optional early stopping.

    Set ``train_sample_duplicates`` > 1 to list each training image that many times per epoch.

    With ``small_set_auto`` (default True), tiny training splits (≤28 unique train images before
    duplication) reduce strong augmentation and label smoothing and increase duplicate exposure.

    Saves model.pth, meta.json.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()

    classes: list[str]
    train_ds: Dataset[tuple[torch.Tensor, int]]
    val_ds: Dataset[tuple[torch.Tensor, int]]

    use_stratify = stratified_split
    eff_strong = strong_augment if augment else False
    aug_reported = eff_strong
    dup_eff = train_sample_duplicates
    effective_dup = train_sample_duplicates
    label_smooth_eff = float(np.clip(label_smoothing, 0.0, 0.35))
    small_set_applied = False

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
        targets = [t for _, t in samples]
        if use_stratify:
            train_ix, val_ix = _train_val_indices_stratified(targets, val_fraction, split_seed)
        else:
            rng = np.random.default_rng(split_seed)
            perm = rng.permutation(n).tolist()
            n_val = max(1, int(round(val_fraction * n)))
            n_val = min(n_val, n - 1)
            val_ix = perm[:n_val]
            train_ix = perm[n_val:]
        train_pairs, val_pairs = _pairs_from_indices(samples, train_ix, val_ix)

        pre_train_n = len(train_pairs)
        if small_set_auto and pre_train_n <= 28:
            eff_strong = False
            dup_eff = max(train_sample_duplicates, 4)
            dup_eff = min(dup_eff, 12)
            label_smooth_eff = min(label_smooth_eff, 0.045)
            small_set_applied = True

        train_tfm = _build_transforms(
            img_size,
            crop_bottom_fraction=crop_bottom_fraction,
            train=True,
            augment=augment,
            strong_augment=eff_strong,
            random_erasing_p=random_erasing_p,
        )
        val_tfm = _build_transforms(
            img_size,
            crop_bottom_fraction=crop_bottom_fraction,
            train=False,
            augment=False,
            strong_augment=False,
            random_erasing_p=0.0,
        )
        if len(train_pairs) < 1 or len(val_pairs) < 1:
            raise ValueError(
                "Train/val split produced an empty split; "
                "add more labeled images or adjust the manifest."
            )
        train_pairs = _expand_train_duplicates(train_pairs, dup_eff)
        effective_dup = dup_eff
        aug_reported = eff_strong
        train_ds = _ManifestDataset(data_dir, train_pairs, train_tfm)
        val_ds = _ManifestDataset(data_dir, val_pairs, val_tfm)
    else:
        val_tfm = _build_transforms(
            img_size,
            crop_bottom_fraction=crop_bottom_fraction,
            train=False,
            augment=False,
            strong_augment=False,
            random_erasing_p=0.0,
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
        samples = [(Path(ds.samples[i][0]), ds.samples[i][1]) for i in range(n)]
        targets = [t for _, t in samples]
        if use_stratify:
            train_ix, val_ix = _train_val_indices_stratified(targets, val_fraction, split_seed)
        else:
            rng = np.random.default_rng(split_seed)
            perm = rng.permutation(n).tolist()
            n_val = max(1, int(round(val_fraction * n)))
            n_val = min(n_val, n - 1)
            val_ix = perm[:n_val]
            train_ix = perm[n_val:]
        train_pairs, val_pairs = _pairs_from_indices(samples, train_ix, val_ix)

        pre_train_n = len(train_pairs)
        dup_eff_img = train_sample_duplicates
        eff_strong_img = eff_strong
        if small_set_auto and pre_train_n <= 28:
            eff_strong_img = False
            dup_eff_img = max(train_sample_duplicates, 4)
            dup_eff_img = min(dup_eff_img, 12)
            label_smooth_eff = min(label_smooth_eff, 0.045)
            small_set_applied = True

        train_tfm = _build_transforms(
            img_size,
            crop_bottom_fraction=crop_bottom_fraction,
            train=True,
            augment=augment,
            strong_augment=eff_strong_img,
            random_erasing_p=random_erasing_p,
        )
        train_pairs = _expand_train_duplicates(train_pairs, dup_eff_img)
        effective_dup = dup_eff_img
        aug_reported = eff_strong_img
        train_ds = _ManifestDataset(data_dir, train_pairs, train_tfm)
        val_ds = _ManifestDataset(data_dir, val_pairs, val_tfm)
        if len(train_ds) < 1 or len(val_ds) < 1:
            raise ValueError(
                "Train/val split produced an empty split; "
                "add more images per class or consolidate classes."
            )

    train_targets = [t for _, t in train_pairs]
    cnt = Counter(train_targets)
    use_balance = balance_sampler and len(cnt) >= 2
    if use_balance:
        mx = max(cnt.values())
        mn = min(cnt.values())
        if mn <= 0 or mx / mn < 1.35:
            use_balance = False

    if use_balance:
        class_weight = {c: 1.0 / cnt[c] for c in cnt}
        w_tensor = torch.DoubleTensor([class_weight[t] for t in train_targets])
        sampler = WeightedRandomSampler(
            weights=w_tensor, num_samples=len(train_targets), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    wd = 0.02
    mult = 1.0 if uniform_lr else float(backbone_lr_mult)

    model = timm.create_model(backbone, pretrained=True, num_classes=len(classes))
    model = model.to(device)

    backbone_params, head_params = _split_classifier_vs_backbone(model)
    freeze_epochs = max(0, min(int(freeze_backbone_epochs), epochs))
    if freeze_epochs > 0 and not head_params:
        freeze_epochs = 0

    ls = float(np.clip(label_smooth_eff, 0.0, 0.35))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)

    opt: torch.optim.Optimizer | None = None
    scheduler: Any = None

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None
    stalled = 0
    epoch_ran = 0

    for epoch in range(epochs):
        epoch_ran = epoch + 1

        if freeze_epochs > 0 and epoch == 0:
            for p in backbone_params:
                p.requires_grad = False
            for p in head_params:
                p.requires_grad = True
            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
            eta_frozen = lr * 0.02
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, freeze_epochs), eta_min=eta_frozen
            )
        elif (
            freeze_epochs > 0
            and epoch == freeze_epochs
            and freeze_epochs < epochs
        ):
            for p in backbone_params:
                p.requires_grad = True
            for p in head_params:
                p.requires_grad = True
            opt = torch.optim.AdamW(
                _optimizer_param_groups(model, lr, mult, weight_decay=wd),
            )
            rest = max(1, epochs - freeze_epochs)
            eta_tail = lr * mult * 0.02
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rest, eta_min=eta_tail)

        elif freeze_epochs == 0 and epoch == 0:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(
                _optimizer_param_groups(model, lr, mult, weight_decay=wd),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, epochs), eta_min=lr * mult * 0.02
            )

        assert opt is not None and scheduler is not None

        model.train()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch_ran}/{epochs}"):
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
                loss = F.cross_entropy(logits, y)
                val_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        val_loss /= len(val_ds)
        val_acc = correct / total if total else 0.0

        improved = val_acc > best_val_acc + 1e-7 or (
            abs(val_acc - best_val_acc) < 1e-7 and val_loss < best_val_loss - 1e-7
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1

        scheduler.step()

        if patience > 0 and stalled >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "backbone": backbone,
        "classes": classes,
        "class_to_idx": {k: int(i) for i, k in enumerate(classes)},
        "img_size": img_size,
        "epochs": epochs,
        "epochs_run": epoch_ran,
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "crop_bottom_fraction": float(crop_bottom_fraction),
        "manifest": str(Path(manifest).as_posix()) if manifest is not None else None,
        "neighbor_similarity_threshold": float(neighbor_similarity_threshold),
        "pretrained_weights": "imagenet",
        "training": {
            "backbone_lr_mult": float(backbone_lr_mult) if not uniform_lr else 1.0,
            "uniform_lr": bool(uniform_lr),
            "freeze_backbone_epochs": int(freeze_epochs),
            "strong_augment": aug_reported,
            "label_smoothing": ls,
            "val_fraction": val_fraction,
            "stratified_split": use_stratify,
            "balance_sampler": use_balance,
            "patience": patience,
            "early_stopped": bool(patience > 0 and stalled >= patience and epoch_ran < epochs),
            "train_sample_duplicates": int(train_sample_duplicates),
            "effective_train_duplicates": int(effective_dup),
            "small_set_auto_applied": small_set_applied,
        },
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


def _prediction_from_supervised_or_model(
    pil_img: Image.Image,
    raw_bytes: bytes | None,
    *,
    classes: list[str],
    meta: dict[str, Any],
    model: nn.Module,
    tfm,
    device: torch.device,
    tta: bool,
) -> tuple[dict[str, Any], bool]:
    """
    Prefer manifest byte match, then embedding neighbor match to supervised files,
    else CNN. Returns extra fields and ``simple_overlay`` when Grad-CAM can be skipped.
    """
    thresh = float(meta.get("neighbor_similarity_threshold", 0.988))
    manifest_path = resolve_manifest_csv(meta)
    data_dir = supervised_data_dir()

    if manifest_path is not None and raw_bytes is not None:
        lab_exact = manifest_lookup_sha256(raw_bytes, data_dir, manifest_path)
        if lab_exact is not None:
            probs = probabilities_for_manifest_label(classes, lab_exact)
            return (
                {
                    "predicted_class": lab_exact,
                    "confidence": 1.0,
                    "probabilities": probs,
                    "match_source": "manifest_sha256",
                },
                True,
            )

    if manifest_path is not None:
        nlab, nsim = manifest_neighbor_lookup(
            pil_img,
            model,
            tfm,
            device,
            data_dir,
            manifest_path,
            similarity_threshold=thresh,
        )
        if nlab is not None:
            probs = neighbor_probabilities(classes, nlab, nsim)
            pred = max(classes, key=lambda c: probs[c])
            return (
                {
                    "predicted_class": pred,
                    "confidence": float(probs[pred]),
                    "probabilities": probs,
                    "match_source": "manifest_neighbor",
                },
                True,
            )

    prob = _probabilities_for_pil(pil_img, model, tfm, device, tta=tta)
    top = int(np.argmax(prob))
    return (
        {
            "predicted_class": classes[top],
            "confidence": float(prob[top]),
            "probabilities": {classes[i]: float(prob[i]) for i in range(len(classes))},
            "match_source": "model",
        },
        False,
    )


def predict_images(
    image_paths: list[Path | str],
    model_dir: Path | str,
    *,
    tta: bool = False,
) -> list[dict[str, Any]]:
    """Return class probabilities for each image path."""
    model_dir = Path(model_dir)
    device = _device()
    model, meta, tfm = _load_model_for_inference(model_dir, device)
    classes: list[str] = meta["classes"]
    results: list[dict[str, Any]] = []

    for p in image_paths:
        p = Path(p)
        raw = p.read_bytes()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        core, _simple = _prediction_from_supervised_or_model(
            img,
            raw,
            classes=classes,
            meta=meta,
            model=model,
            tfm=tfm,
            device=device,
            tta=tta,
        )
        results.append({"path": str(p), **core})
    return results


def predict_images_from_bytes(
    files: list[tuple[str, bytes]],
    model_dir: Path | str,
    *,
    tta: bool = False,
) -> list[dict[str, Any]]:
    """Predict from (filename, bytes) pairs (e.g. Streamlit uploads)."""
    model_dir = Path(model_dir)
    device = _device()
    model, meta, tfm = _load_model_for_inference(model_dir, device)
    classes: list[str] = meta["classes"]
    results: list[dict[str, Any]] = []
    for name, data in files:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        core, _simple = _prediction_from_supervised_or_model(
            img,
            data,
            classes=classes,
            meta=meta,
            model=model,
            tfm=tfm,
            device=device,
            tta=tta,
        )
        results.append({"path": name, **core})
    return results


def predict_images_from_bytes_with_annotation(
    files: list[tuple[str, bytes]],
    model_dir: Path | str,
    *,
    blend_heatmap: bool = True,
    heatmap_alpha: float = 0.38,
    tta: bool = False,
) -> list[dict[str, Any]]:
    """
    Same as ``predict_images_from_bytes``, plus ``annotated_png`` bytes per item:
    class label + confidence on the image, optional Grad-CAM heat tint,
    arrow toward salient region.
    """
    model_dir = Path(model_dir)
    device = _device()
    model, meta, tfm = _load_model_for_inference(model_dir, device)
    classes: list[str] = meta["classes"]
    results: list[dict[str, Any]] = []
    for name, data in files:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        core, simple_ov = _prediction_from_supervised_or_model(
            img,
            data,
            classes=classes,
            meta=meta,
            model=model,
            tfm=tfm,
            device=device,
            tta=tta,
        )
        row: dict[str, Any] = {"path": name, **core}
        row["annotated_png"] = annotate_prediction_dict(
            data,
            row,
            model,
            meta,
            tfm,
            device,
            blend_heatmap=blend_heatmap,
            heatmap_alpha=heatmap_alpha,
            simple_overlay=simple_ov,
        )
        results.append(row)
    return results
