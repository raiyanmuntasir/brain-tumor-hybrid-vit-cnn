"""Evaluation metrics and a checkpoint evaluation CLI.

``python -m btvit.evaluate --checkpoint runs/hybrid/best.pt --data-root data/brain-tumor-mri``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(model(x.to(device)), 1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def compute_metrics(y: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict:
    pred = proba.argmax(1)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "per_class": classification_report(y, pred, labels=list(range(len(classes))), target_names=classes,
                                           output_dict=True, zero_division=0),
    }
    if len(np.unique(y)) == len(classes):
        out["macro_roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    return out


def save_confusion(y, proba, classes, path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y, proba.argmax(1), labels=list(range(len(classes))),
                                            display_labels=classes, ax=ax, colorbar=False, xticks_rotation=30)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    from .data import make_loaders
    from .model import build_model

    ap = argparse.ArgumentParser(prog="btvit.evaluate")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["arch"], **cfg["model_kwargs"]).to(args.device)
    model.load_state_dict(ck["state_dict"])
    _, _, test_loader, classes = make_loaders(args.data_root, cfg["img_size"], batch_size=64, num_workers=4)
    y, p = predict(model, test_loader, args.device)
    m = compute_metrics(y, p, classes)
    out = Path(args.checkpoint).parent
    (out / "test_metrics.json").write_text(json.dumps(m, indent=2))
    save_confusion(y, p, classes, out / "confusion_matrix.png", f"{cfg['arch']} - test")
    print(json.dumps({k: v for k, v in m.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
