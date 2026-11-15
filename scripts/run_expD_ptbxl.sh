#!/bin/bash
set +e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
LOG=logs/expD_ptbxl.log
: > "$LOG"
for f in 0 1 2 3 4; do
  echo "===== fold $f  ptbxl real  $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python -u src/train_clf.py --npz data/processed/ptbxl_beats.npz --fold $f \
    --tag ptbxl-real >> "$LOG" 2>&1
  echo "===== fold $f  ptbxl D(pretrain)  $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python -u src/train_clf.py --npz data/processed/ptbxl_beats.npz --fold $f \
    --syn data/synthetic/pde_g12_f${f}_aligned.npz \
    --pretrain-epochs 10 --syn-for-pretrain \
    --tag ptbxl-pdept >> "$LOG" 2>&1
done
echo "[PTBXL D DONE] $(date '+%H:%M:%S')" | tee -a "$LOG"
