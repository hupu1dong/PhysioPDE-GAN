#!/bin/bash
set +e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
LOG=logs/expD.log
: > "$LOG"
for f in 0 1 2 3 4; do
  echo "===== fold $f  synth-pretrain + real-finetune  $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python -u src/train_clf.py --npz data/processed/g12ec_beats.npz --fold $f \
    --syn data/synthetic/pde_g12_f${f}_aligned.npz \
    --pretrain-epochs 10 --syn-for-pretrain \
    --tag g12-pdept >> "$LOG" 2>&1
done
echo "[EXP D DONE] $(date '+%H:%M:%S')" | tee -a "$LOG"
