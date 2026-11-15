#!/bin/bash
set +e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
LOG=logs/expA.log
: > "$LOG"
for f in 0 1 2 3 4; do
  echo "===== fold $f  real+attn  $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python -u src/train_clf.py --npz data/processed/g12ec_beats.npz --fold $f --arch attn --tag g12-real >> "$LOG" 2>&1
  echo "===== fold $f  pdeal+attn  $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python -u src/train_clf.py --npz data/processed/g12ec_beats.npz --fold $f --arch attn --syn data/synthetic/pde_g12_f${f}_aligned.npz --syn-ratio 1.0 --tag g12-pdeal >> "$LOG" 2>&1
done
echo "[EXP A DONE] $(date '+%H:%M:%S')" | tee -a "$LOG"
