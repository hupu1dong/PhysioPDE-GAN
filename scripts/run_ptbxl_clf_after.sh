#!/bin/bash
set +e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
LOG=logs/ptbxl_clf_after.log
: > "$LOG"
NPZ=data/processed/ptbxl_beats.npz

# wait for the 5-fold GAN run (at most 9 hours)
echo "[wait] waiting for PTBXL GAN to finish $(date '+%H:%M')" | tee -a "$LOG"
for i in $(seq 1 108); do
  if grep -q "PTBXL GAN DONE" logs/ptbxl_gan.log 2>/dev/null; then
    echo "[wait] completion signal detected $(date '+%H:%M')" | tee -a "$LOG"
    break
  fi
  sleep 300
done

# verify that all five aligned sample files exist
for F in 0 1 2 3 4; do
  if [ ! -f "data/synthetic/pde_ptbxl_f${F}_aligned.npz" ]; then
    echo "[abort] fold $F synthetic samples missing, stopping" | tee -a "$LOG"
    exit 1
  fi
done

# 1) strategy D: local synthetic pretraining + real finetuning
for F in 0 1 2 3 4; do
  echo "===== [$(date '+%H:%M')] strategy D fold $F =====" | tee -a "$LOG"
  python -u src/train_clf.py --npz ${NPZ} --fold $F \
    --syn data/synthetic/pde_ptbxl_f${F}_aligned.npz \
    --pretrain-epochs 10 --syn-for-pretrain \
    --tag ptbxl-pdept2 >> "$LOG" 2>&1
done

# 2) control: 1:1 mixing
for F in 0 1 2 3 4; do
  echo "===== [$(date '+%H:%M')] 1:1 mixing fold $F =====" | tee -a "$LOG"
  python -u src/train_clf.py --npz ${NPZ} --fold $F \
    --syn data/synthetic/pde_ptbxl_f${F}_aligned.npz \
    --tag ptbxl-pde2 >> "$LOG" 2>&1
done

echo "[PTBXL CLF DONE] $(date '+%H:%M')" | tee -a "$LOG"
