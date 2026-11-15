#!/usr/bin/env bash
# Full reproduction pipeline (run inside the container)
# Usage: bash scripts/run_pipeline.sh [fold] [epochs_gan] [epochs_clf]
set -euo pipefail

FOLD=${1:-0}
EGAN=${2:-100}
ECLF=${3:-30}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/6] preprocessing (downloaded records -> per-beat npz)"
python src/data/preprocess.py

echo "[2/6] train GAN (fold ${FOLD})"
python src/train_gan.py --fold "${FOLD}" --epochs "${EGAN}" \
    --out-dir checkpoints/f${FOLD} --log logs/train_gan_f${FOLD}.log

echo "[3/6] generate synthetic beats"
python src/generate.py --ckpt "checkpoints/f${FOLD}/physiopde_gan_best.pt" --fold "${FOLD}" \
    --n 33000 --out "data/synthetic/pde_f${FOLD}.npz"

echo "[4/6] classifier: real"
python src/train_clf.py --fold "${FOLD}" --tag real --arch resnet --epochs "${ECLF}"

echo "[5/6] classifier: real + PhysioPDE-GAN"
python src/train_clf.py --fold "${FOLD}" --tag "real+pde" --arch resnet --epochs "${ECLF}" \
    --syn "data/synthetic/pde_f${FOLD}.npz"

echo "[6/6] aggregation"
python src/evaluate.py
