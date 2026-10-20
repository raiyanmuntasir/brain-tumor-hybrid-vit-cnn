"""Data loading for the public Brain Tumor MRI dataset (glioma / meningioma / no tumour / pituitary).

Expected layout (the Kaggle "Brain Tumor MRI Dataset" by M. Nickparvar unpacks like this)::

    data/brain-tumor-mri/
      Training/{glioma,meningioma,notumor,pituitary}/*.jpg
      Testing/{glioma,meningioma,notumor,pituitary}/*.jpg

A stratified validation split is carved out of ``Training``. ``Testing`` is
only touched once, for the final evaluation. Run :mod:`btvit.dedup` first: the
public dataset merges several sources and contains near-duplicate slices across
the train/test folders, which inflates reported accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def train_transform(size: int):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform(size: int):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def make_loaders(root: str | Path, img_size: int = 224, batch_size: int = 32, val_fraction: float = 0.15,
                 num_workers: int = 4, seed: int = 42):
    root = Path(root)
    train_full = datasets.ImageFolder(root / "Training", transform=train_transform(img_size))
    train_eval = datasets.ImageFolder(root / "Training", transform=eval_transform(img_size))
    test = datasets.ImageFolder(root / "Testing", transform=eval_transform(img_size))
    if train_full.classes != test.classes:
        raise ValueError(f"class folders differ: {train_full.classes} vs {test.classes}")
    idx = np.arange(len(train_full))
    tr_idx, va_idx = train_test_split(idx, test_size=val_fraction, stratify=train_full.targets, random_state=seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(Subset(train_full, tr_idx), shuffle=True, drop_last=True, **kw),
        DataLoader(Subset(train_eval, va_idx), shuffle=False, **kw),
        DataLoader(test, shuffle=False, **kw),
        train_full.classes,
    )


def make_fake_loaders(img_size: int = 64, batch_size: int = 8, n: int = 64, num_classes: int = 4):
    """Random images for smoke tests and CI. No real data needed."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    mk = lambda size, seed: datasets.FakeData(size=size, image_size=(3, img_size, img_size),  # noqa: E731
                                              num_classes=num_classes, transform=tf, random_offset=seed)
    kw = dict(batch_size=batch_size, num_workers=0)
    classes = ["glioma", "meningioma", "notumor", "pituitary"][:num_classes]
    return (DataLoader(mk(n, 0), shuffle=True, drop_last=True, **kw), DataLoader(mk(n // 2, 10_000), **kw),
            DataLoader(mk(n // 2, 20_000), **kw), classes)
