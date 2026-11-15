#!/usr/bin/env bash
# Full 5-fold pipeline on G12EC (serial, to avoid GPU contention).
# Skips folds whose GAN / synthetic data / classifier outputs already exist.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPZ=data/processed/g12ec_beats.npz
GAN_ARGS="--npz ${NPZ} --epochs 40 --batch 128 --steps-per-epoch 100 --lr 3e-4 --lr-d 1e-4 --loss hinge --r1 10 --hidden 32 --lambda-spec 5.0"

echo "[5fold] start $(date)"

# ---- 1) GAN training and synthetic sampling per fold ----
for F in 0 1 2 3 4; do
  if [ ! -f "checkpoints/g12_f${F}/physiopde_gan_last.pt" ]; then
    echo "[5fold] train GAN fold ${F} $(date)"
    python -u src/train_gan.py ${GAN_ARGS} --fold ${F} \
      --out-dir checkpoints/g12_f${F} --log logs/train_gan_g12_f${F}.log \
      > logs/train_gan_g12_f${F}.out 2>&1
  fi
  if [ ! -f "data/synthetic/pde_g12_f${F}.npz" ]; then
    echo "[5fold] generate fold ${F} $(date)"
    python -u src/generate.py --npz ${NPZ} --ckpt checkpoints/g12_f${F}/physiopde_gan_last.pt \
      --fold ${F} --n 110000 --out data/synthetic/pde_g12_f${F}.npz \
      > logs/generate_g12_f${F}.log 2>&1
  fi
done

# ---- 2) downstream classifier: real / real+pde / real+dcgan ----
for F in 0 1 2 3 4; do
  if [ ! -f "metrics/probs_g12-real_f${F}_resnet.npz" ]; then
    echo "[5fold] clf real fold ${F} $(date)"
    python -u src/train_clf.py --npz ${NPZ} --fold ${F} --tag g12-real --arch resnet --epochs 30 \
      > logs/train_clf_g12_real_f${F}.log 2>&1
  fi
  if [ ! -f "metrics/probs_g12-real+pde_f${F}_resnet.npz" ]; then
    echo "[5fold] clf real+pde fold ${F} $(date)"
    python -u src/train_clf.py --npz ${NPZ} --fold ${F} --tag "g12-real+pde" --arch resnet --epochs 30 \
      --syn data/synthetic/pde_g12_f${F}.npz > logs/train_clf_g12_realpde_f${F}.log 2>&1
  fi
  if [ ! -f "metrics/probs_g12-real+dcgan_f${F}_resnet.npz" ]; then
    echo "[5fold] dcgan fold ${F} $(date)"
    python -u src/train_baseline.py --npz ${NPZ} --model dcgan --fold ${F} --epochs 40 --batch 128 \
      --out-dir checkpoints --log logs/train_dcgan_g12_f${F}.log > logs/train_dcgan_g12_f${F}.out 2>&1
    python -u src/generate.py --npz ${NPZ} --ckpt checkpoints/dcgan_best.pt --fold ${F} --n 110000 \
      --out data/synthetic/dcgan_g12_f${F}.npz > logs/generate_dcgan_g12_f${F}.log 2>&1
    python -u src/train_clf.py --npz ${NPZ} --fold ${F} --tag "g12-real+dcgan" --arch resnet --epochs 30 \
      --syn data/synthetic/dcgan_g12_f${F}.npz > logs/train_clf_g12_realdcgan_f${F}.log 2>&1
  fi
done

# ---- 3) aggregation ----
echo "[5fold] evaluate $(date)"
python -u src/evaluate.py --glob 'metrics/probs_g12-*.npz' --baseline-tag g12-real \
  --out metrics/table_g12ec_5fold.csv > logs/eval_g12_5fold.log 2>&1
python -u src/paired_test.py --glob 'metrics/probs_g12-*.npz' \
  --out metrics/paired_ttest_g12.csv > logs/paired_test_g12.log 2>&1

echo "[5fold] done $(date)"
