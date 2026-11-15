#!/usr/bin/env bash
# PDE component ablation: same backbone, same budget (20 epochs), one component removed at a time.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPZ=data/processed/g12ec_beats.npz
COMMON="--npz ${NPZ} --fold 0 --epochs 20 --batch 128 --steps-per-epoch 100 \
  --lr 3e-4 --lr-d 1e-4 --loss hinge --r1 10 --hidden 32 --lambda-spec 5.0 --lambda-align 5.0"
echo "[abl] start $(date)"

run_case () {
  NAME=$1; shift
  if [ ! -f "checkpoints/abl_${NAME}/physiopde_gan_last.pt" ]; then
    echo "[abl] train ${NAME} $(date)"
    python -u src/train_gan.py ${COMMON} "$@" \
      --out-dir checkpoints/abl_${NAME} --log logs/train_abl_${NAME}.log \
      > logs/train_abl_${NAME}.out 2>&1
  fi
  if [ ! -f "data/synthetic/abl_${NAME}.npz" ]; then
    python -u src/generate.py --npz ${NPZ} --ckpt checkpoints/abl_${NAME}/physiopde_gan_last.pt \
      --fold 0 --n 2000 --out data/synthetic/abl_${NAME}.npz > logs/gen_abl_${NAME}.log 2>&1
  fi
  python -u src/diagnostics.py --real ${NPZ} --syn data/synthetic/abl_${NAME}.npz \
    --n 2000 --tag abl_${NAME} >> logs/diag_abl.log 2>&1
  echo "[abl] ${NAME} done $(date)"
}

run_case full   --spatial both
run_case nosp   --spatial conv
run_case nocoup --spatial both --no-coupling
run_case notmp  --spatial both --lambda-temporal 0

echo "[abl] all done $(date)"
cat logs/diag_abl.log
