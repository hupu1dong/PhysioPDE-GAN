#!/usr/bin/env bash
# Strategy C (data-scale ablation) and B (targeted augmentation), both on A-aligned synthetic data.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPZ=data/processed/g12ec_beats.npz
echo "[ablation] start $(date)"

# ---------- C: 10%/25%/50% of real data ----------
for F in 0 1 2 3 4; do
  for R in 10 25 50; do
    python -u src/train_clf.py --npz ${NPZ} --fold ${F} --real-frac 0.${R} \
      --tag "c${R}-real" --arch resnet --epochs 30 \
      > logs/clf_c${R}_real_f${F}.log 2>&1
    python -u src/train_clf.py --npz ${NPZ} --fold ${F} --real-frac 0.${R} \
      --tag "c${R}-pdeal" --arch resnet --epochs 30 \
      --syn data/synthetic/pde_g12_f${F}_aligned.npz \
      > logs/clf_c${R}_pdeal_f${F}.log 2>&1
  done
done
echo "[ablation] C done $(date)"

# ---------- B: augment LAnFB / NSIVCB only ----------
for F in 0 1 2 3 4; do
  python -u src/train_clf.py --npz ${NPZ} --fold ${F} \
    --tag "g12-real+pdesel" --arch resnet --epochs 30 \
    --syn data/synthetic/pde_g12_f${F}_aligned.npz --syn-classes "LAnFB,NSIVCB" \
    > logs/clf_pdesel_f${F}.log 2>&1
done
echo "[ablation] B done $(date)"

# ---------- aggregation ----------
for R in 10 25 50; do
  python -u src/evaluate.py --glob "metrics/probs_c${R}-*.npz" --baseline-tag "c${R}-real" \
    --out metrics/table_c${R}.csv > logs/eval_c${R}.log 2>&1
  python -u src/paired_test.py --glob "metrics/probs_c${R}-*.npz" --baseline-tag "c${R}-real" \
    --out metrics/paired_c${R}.csv >> logs/eval_c${R}.log 2>&1
done
python -u src/evaluate.py --glob 'metrics/probs_g12-*.npz' --baseline-tag g12-real \
  --out metrics/table_g12ec_5fold.csv > logs/eval_g12_5fold.log 2>&1
python -u src/paired_test.py --glob 'metrics/probs_g12-*.npz' \
  --out metrics/paired_ttest_g12.csv > logs/paired_test_g12.log 2>&1
echo "[ablation] done $(date)"
