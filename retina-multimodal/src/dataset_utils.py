"""
dataset_utils.py
Fundus image + optional tabular CSV loader and transforms.
Expected CSV columns:
- 'image' (or 'filename' / 'id')
- 'label' (int) or 'diagnosis' (renamed to label)
- Optional numeric tabular columns (auto-detected if not provided)
"""
from __future__ import annotations
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _find_image(img_dir: str, stem_or_name: str) -> str:
    """Find an image by stem or full name inside img_dir."""
    direct = os.path.join(img_dir, stem_or_name)
    if os.path.exists(direct):
        return direct
    stem = os.path.splitext(stem_or_name)[0]
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Image not found for '{stem_or_name}' in '{img_dir}'.")


def build_transforms(img_size: int = 224, aug: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(int(img_size * 1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)) if aug else transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip() if aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    valid_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tfms, valid_tfms


class FundusTabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, tfms: transforms.Compose,
                 tab_features: Optional[List[str]] = None):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.tfms = tfms

        # normalize col names
        lower = {c.lower(): c for c in self.df.columns}
        # label column
        if "label" in lower:
            lab_col = lower["label"]
        elif "diagnosis" in lower:
            lab_col = lower["diagnosis"]
            self.df.rename(columns={lab_col: "label"}, inplace=True)
        else:
            raise ValueError("CSV must contain 'label' or 'diagnosis' column.")
        # image column
        if "image" in lower:
            img_col = lower["image"]
        elif "filename" in lower:
            img_col = lower["filename"]
        elif "id" in lower:
            img_col = lower["id"]
        else:
            raise ValueError("CSV must contain one of: 'image', 'filename', 'id'.")

        self.df.rename(columns={img_col: "image"}, inplace=True)
        self.df["label"] = self.df["label"].astype(int)

        # tabular columns
        if tab_features is None:
            numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.tab_features = [c for c in numeric if c not in {"label"}]
        else:
            self.tab_features = tab_features
        self.has_tab = len(self.tab_features) > 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = _find_image(self.img_dir, str(row["image"]))
        img = Image.open(img_path).convert("RGB")
        img = self.tfms(img)

        if self.has_tab:
            tab = torch.tensor(row[self.tab_features].values.astype("float32"))
        else:
            tab = torch.zeros(0)

        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return img, tab, y, str(row["image"])


def make_loaders(csv_path: str, img_dir: str, img_size: int = 224, batch_size: int = 16,
                 num_workers: int = 2, val_split: float = 0.2,
                 tab_features: Optional[List[str]] = None, shuffle_seed: int = 42):
    """Return ((train_loader, val_loader), tab_dim, num_classes)."""
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    n_val = int(len(df) * val_split)
    df_val = df.iloc[:n_val].reset_index(drop=True)
    df_train = df.iloc[n_val:].reset_index(drop=True)

    train_tfms, valid_tfms = build_transforms(img_size)
    ds_tr = FundusTabularDataset(df_train, img_dir, train_tfms, tab_features)
    ds_va = FundusTabularDataset(df_val,   img_dir, valid_tfms, tab_features)

    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    num_classes = int((df["label"] if "label" in df.columns else df["diagnosis"]).max()) + 1
    tab_dim = len(ds_tr.tab_features) if ds_tr.has_tab else 0
    return (train_loader, val_loader), tab_dim, num_classes

