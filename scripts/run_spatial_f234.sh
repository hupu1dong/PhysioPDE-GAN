#!/usr/bin/env bash
# Spatial-modelling variant (--spatial both) for folds 2/3/4 with final aggregation.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPZ=data/processed/g12ec_beats.npz
echo "[sp] start $(date)"

for F in 2 3 4; do
  if [ ! -f "checkpoints/g12_sp_f${F}/physiopde_gan_last.pt" ]; then
    echo "[sp] train GAN fold ${F} $(date)"
    python -u src/train_gan.py --npz ${NPZ} --fold ${F} --epochs 40 --batch 128 \
      --steps-per-epoch 100 --lr 3e-4 --lr-d 1e-4 --loss hinge --r1 10 --hidden 32 \
      --lambda-spec 5.0 --lambda-align 5.0 --spatial both \
      --out-dir checkpoints/g12_sp_f${F} --log logs/train_gan_g12_sp_f${F}.log \
      > logs/train_gan_g12_sp_f${F}.out 2>&1
  fi
  if [ ! -f "data/synthetic/pde_sp_f${F}.npz" ]; then
    echo "[sp] generate fold ${F} $(date)"
    python -u src/generate.py --npz ${NPZ} --ckpt checkpoints/g12_sp_f${F}/physiopde_gan_last.pt \
      --fold ${F} --n 110000 --out data/synthetic/pde_sp_f${F}.npz \
      > logs/generate_sp_f${F}.log 2>&1
  fi
  if [ ! -f "metrics/probs_g12-real+sp_f${F}_resnet.npz" ]; then
    echo "[sp] clf fold ${F} $(date)"
    python -u src/train_clf.py --npz ${NPZ} --fold ${F} --tag "g12-real+sp" --arch resnet --epochs 30 \
      --syn data/synthetic/pde_sp_f${F}.npz > logs/clf_sp_f${F}.log 2>&1
  fi
done

echo "[sp] evaluate $(date)"
python -u src/evaluate.py --glob 'metrics/probs_g12-*.npz' --baseline-tag g12-real \
  --out metrics/table_g12ec_all.csv > logs/eval_g12_all.log 2>&1
python -u src/paired_test.py --glob 'metrics/probs_g12-*.npz' \
  --out metrics/paired_ttest_g12.csv > logs/paired_test_g12.log 2>&1
echo "[sp] done $(date)"
