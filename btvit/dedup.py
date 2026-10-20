"""Find near-duplicate images across the Training and Testing folders.

The public brain-tumour MRI collections were assembled from several sources,
and identical or near-identical slices can appear in both splits. That leaks
test information into training and inflates accuracy. This tool hashes every
image with a 64-bit difference hash (dHash) and reports or moves test images
whose hash is within ``--max-distance`` bits of any training image.

    python -m btvit.dedup --data-root data/brain-tumor-mri            # report only
    python -m btvit.dedup --data-root data/brain-tumor-mri --quarantine  # move leaks out of Testing/
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dhash(path: Path, size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: np.ndarray, b: int) -> np.ndarray:
    x = np.bitwise_xor(a, np.uint64(b))
    # popcount on uint64 via byte view
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def find_leaks(root: Path, max_distance: int = 4):
    train = [p for p in (root / "Training").rglob("*") if p.suffix.lower() in EXTS]
    test = [p for p in (root / "Testing").rglob("*") if p.suffix.lower() in EXTS]
    th = np.array([dhash(p) for p in train], dtype=np.uint64)
    leaks = []
    for p in test:
        d = hamming(th, dhash(p))
        j = int(d.argmin())
        if d[j] <= max_distance:
            leaks.append((p, train[j], int(d[j])))
    return leaks, len(train), len(test)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="btvit.dedup")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--max-distance", type=int, default=4)
    ap.add_argument("--quarantine", action="store_true", help="move leaked test images to <root>/Quarantine")
    args = ap.parse_args(argv)
    leaks, n_tr, n_te = find_leaks(args.data_root, args.max_distance)
    print(f"{len(leaks)} of {n_te} test images are near-duplicates of a training image ({n_tr} training images)")
    for test_p, train_p, d in leaks[:20]:
        print(f"  d={d}  {test_p.relative_to(args.data_root)}  ~  {train_p.relative_to(args.data_root)}")
    if args.quarantine and leaks:
        q = args.data_root / "Quarantine"
        for test_p, _, _ in leaks:
            dest = q / test_p.relative_to(args.data_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(test_p), dest)
        print(f"moved {len(leaks)} files to {q}")


if __name__ == "__main__":
    main()
