#!/bin/bash
set +e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
LOG=logs/expBC.log
: > "$LOG"
for f in 0 1 2 3 4; do
  for r in 0.5 2.0; do
    T=$(echo $r | tr -d '.')
    echo "===== fold $f  ratio=$r  $(date '+%H:%M:%S') =====" | tee -a "$LOG"
    python -u src/train_clf.py --npz data/processed/g12ec_beats.npz --fold $f \
      --syn data/synthetic/pde_g12_f${f}_aligned.npz --syn-ratio $r \
      --tag g12-pdealr${T} >> "$LOG" 2>&1
  done
  echo "===== fold $f  weak-only  $(date '+%H:%M:%S') =====" | tee -a "$LOG"
  python -u src/train_clf.py --npz data/processed/g12ec_beats.npz --fold $f \
    --syn data/synthetic/pde_g12_f${f}_aligned.npz --syn-ratio 1.0 \
    --syn-classes LAnFB,NSIVCB --tag g12-pdealweak >> "$LOG" 2>&1
done
echo "[EXP BC DONE] $(date '+%H:%M:%S')" | tee -a "$LOG"
