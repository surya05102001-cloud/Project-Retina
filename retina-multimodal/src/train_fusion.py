"""
train_fusion.py
End-to-end training for the multimodal model (images + optional tabular data).

Example:
python src/train_fusion.py \
  --csv retina-multimodal/data/labels.csv \
  --img_dir retina-multimodal/data/images \
  --epochs 10 --batch_size 16 --img_size 224
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_utils import make_loaders
from models import FusionModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to labels CSV")
    p.add_argument("--img_dir", type=str, required=True, help="Directory with images")
    p.add_argument("--outputs", type=str, default="retina-multimodal/outputs", help="Output folder")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--backbone", type=str, default="efficientnet_b0")
    p.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet weights")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for imgs, tabs, y, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        tabs = tabs.to(device, non_blocking=True) if tabs.numel() else torch.zeros(imgs.size(0), 0, device=device)
        logits = model(imgs, tabs)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        (y_true, y_pred),
    )


def main():
    args = parse_args()
    Path(args.outputs).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (train_loader, val_loader), tab_dim, num_classes = make_loaders(
        csv_path=args.csv,
        img_dir=args.img_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        tab_features=None,  # auto-detect numeric columns except label
    )

    model = FusionModel(
        num_classes=num_classes,
        tab_in_dim=tab_dim,
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_acc = 0.0
    best_path = os.path.join(args.outputs, "best_model.pt")
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        for imgs, tabs, y, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True) if tabs.numel() else torch.zeros(imgs.size(0), 0, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs, tabs)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(losses)) if losses else 0.0

        val_acc, val_f1, (y_true, y_pred) = evaluate(model, val_loader, device)
        dt = time.time() - t0
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc, "val_f1": val_f1, "secs": dt})
        print(f"[{epoch:03d}] loss={train_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  ({dt:.1f}s)")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {"model": model.state_dict(),
                 "num_classes": num_classes,
                 "tab_dim": tab_dim,
                 "backbone": args.backbone},
                best_path,
            )

    pd.DataFrame(log_rows).to_csv(os.path.join(args.outputs, "train_log.csv"), index=False)
    print(f"\nBest ACC: {best_acc:.4f} | saved -> {best_path}")
    print("\nLast-epoch validation report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()

