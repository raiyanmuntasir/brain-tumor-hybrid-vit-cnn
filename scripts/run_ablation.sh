#!/usr/bin/env bash
# Train the hybrid model and both ablation baselines with identical settings,
# then print a results table. Usage: bash scripts/run_ablation.sh [data_root]
set -euo pipefail
ROOT=${1:-data/brain-tumor-mri}
for ARCH in hybrid cnn vit; do
  python -m btvit.train --data-root "$ROOT" --arch "$ARCH" --pretrained --epochs 30 --out "runs/$ARCH"
done
python - <<'PY'
import json
from pathlib import Path
print("| model | accuracy | macro-F1 | Cohen's kappa | macro ROC-AUC |")
print("|---|---|---|---|---|")
for arch in ("hybrid", "cnn", "vit"):
    m = json.loads(Path(f"runs/{arch}/test_metrics.json").read_text())
    print(f"| {arch} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['cohen_kappa']:.4f} | {m.get('macro_roc_auc_ovr', float('nan')):.4f} |")
PY
