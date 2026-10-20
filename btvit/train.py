"""Training script.

Real data (GPU recommended)::

    python -m btvit.train --data-root data/brain-tumor-mri --arch hybrid --pretrained --epochs 30

Smoke test on random images (CPU, about a minute)::

    python -m btvit.train --fake --epochs 1 --img-size 64 --backbone light --out runs/smoke
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import make_fake_loaders, make_loaders
from .evaluate import compute_metrics, predict, save_confusion
from .explain import save_gallery
from .model import build_model, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="btvit.train")
    ap.add_argument("--data-root", default="data/brain-tumor-mri")
    ap.add_argument("--fake", action="store_true", help="use random images (smoke test / CI)")
    ap.add_argument("--arch", choices=["hybrid", "cnn", "vit"], default="hybrid")
    ap.add_argument("--backbone", choices=["resnet18", "light"], default="resnet18")
    ap.add_argument("--pretrained", action="store_true", help="ImageNet weights for the ResNet-18 stem")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=float, default=2)
    ap.add_argument("--patience", type=int, default=8, help="early stopping on validation macro-F1")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    out = args.out or Path("runs") / args.arch
    out.mkdir(parents=True, exist_ok=True)

    if args.fake:
        train_loader, val_loader, test_loader, classes = make_fake_loaders(args.img_size, args.batch_size)
    else:
        train_loader, val_loader, test_loader, classes = make_loaders(
            args.data_root, args.img_size, args.batch_size, num_workers=args.num_workers, seed=args.seed)

    model_kwargs = dict(num_classes=len(classes), img_size=args.img_size, backbone=args.backbone,
                        pretrained=args.pretrained, dim=args.dim, depth=args.depth, heads=args.heads,
                        dropout=args.dropout)
    model = build_model(args.arch, **dict(model_kwargs)).to(args.device)
    n_params = count_parameters(model)
    print(f"{args.arch}: {n_params / 1e6:.2f}M trainable parameters on {args.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train_loader)
    sched = warmup_cosine(opt, int(args.warmup_epochs * len(train_loader)), steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "classes": classes, "model_kwargs": model_kwargs, "n_params": n_params}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    best, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(y)
            n += len(y)
        yv, pv = predict(model, val_loader, args.device)
        val = compute_metrics(yv, pv, classes)
        row = {"epoch": epoch, "train_loss": running / max(n, 1), "val_accuracy": val["accuracy"],
               "val_macro_f1": val["macro_f1"], "lr": sched.get_last_lr()[0], "seconds": time.time() - t0}
        history.append(row)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if val["macro_f1"] > best:
            best, bad_epochs = val["macro_f1"], 0
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch}, out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # ---- final, single evaluation on the held-out test set -----------------------
    ck = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    yt, pt = predict(model, test_loader, args.device)
    test = compute_metrics(yt, pt, classes)
    test["best_epoch"] = ck["epoch"]
    test["best_val_macro_f1"] = best
    (out / "test_metrics.json").write_text(json.dumps(test, indent=2))
    save_confusion(yt, pt, classes, out / "confusion_matrix.png", f"{args.arch}: test")
    save_gallery(model, test_loader.dataset, classes, out / "explanations.png", n=6, device=args.device)
    print(json.dumps({k: v for k, v in test.items() if k != "per_class"}, indent=2))
    return test


if __name__ == "__main__":
    main()
