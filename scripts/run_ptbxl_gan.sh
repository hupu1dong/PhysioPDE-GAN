#!/bin/bash
set +e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
NPZ=data/processed/ptbxl_beats.npz
LOG=logs/ptbxl_gan.log
: > "$LOG"
for F in 0 1 2 3 4; do
  echo "===== [$(date '+%m-%d %H:%M')] GAN fold $F =====" | tee -a "$LOG"
  python -u src/train_gan.py --npz ${NPZ} --fold ${F} \
    --epochs 40 --batch 128 --steps-per-epoch 100 \
    --lr 3e-4 --lr-d 1e-4 --loss hinge --r1 10 --hidden 32 --lambda-spec 5.0 \
    --out-dir checkpoints/ptbxl_f${F} --log logs/train_gan_ptbxl_f${F}.log \
    >> "$LOG" 2>&1

  echo "===== [$(date '+%m-%d %H:%M')] generate fold $F =====" | tee -a "$LOG"
  python -u src/generate.py --npz ${NPZ} --fold ${F} \
    --ckpt checkpoints/ptbxl_f${F}/physiopde_gan_last.pt --n 45000 \
    --out data/synthetic/pde_ptbxl_f${F}.npz >> "$LOG" 2>&1

  echo "===== [$(date '+%m-%d %H:%M')] align fold $F =====" | tee -a "$LOG"
  python -u src/realign_synthetic.py --npz data/synthetic/pde_ptbxl_f${F}.npz \
    --out data/synthetic/pde_ptbxl_f${F}_aligned.npz --mode global >> "$LOG" 2>&1
done
echo "[PTBXL GAN DONE] $(date '+%m-%d %H:%M')" | tee -a "$LOG"
