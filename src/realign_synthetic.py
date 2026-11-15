"""Strategy A: align the QRS of synthetic beats to the R-peak position of the real window (index 140).

Diagnostics show the generated QRS lands at 0.3-0.5 s (index 150-250) while real data is fixed at 140.
This script detects the R peak of each synthetic beat and applies a circular shift to move it to 140.

Two modes:
    --mode global  one shift for all samples (median), preserving consistency between samples
    --mode per     one shift per sample

Output: an npz in the same format as the input (beats/labels), usable by train_clf --syn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R_IDX = 140


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["global", "per"], default="global")
    ap.add_argument("--max-shift", type=int, default=160, help="drop samples whose shift exceeds this value")
    args = ap.parse_args()

    d = np.load(args.npz)
    beats = d["beats"]
    labels = d["labels"]
    lead = beats[:, 1, :].astype(np.float32)
    # the QRS is the largest complex in the window; approximate the R peak with argmax of |x|
    r = np.abs(lead).argmax(axis=1)
    shift = R_IDX - r

    print(f"[align] n={len(beats)}  R-peak position: median={np.median(r):.0f} "
          f"p10={np.percentile(r, 10):.0f} p90={np.percentile(r, 90):.0f}  "
          f"shift: median={np.median(shift):.0f} std={shift.std():.1f}", flush=True)

    if args.mode == "global":
        g = int(np.median(shift))
        sh = np.full(len(beats), g, dtype=np.int64)
        print(f"[align] global shift {g} samples ({g/500*1000:.1f} ms)", flush=True)
    else:
        sh = shift.astype(np.int64)

    out_all = np.empty_like(beats)
    for i in range(len(beats)):
        out_all[i] = np.roll(beats[i], int(sh[i]), axis=-1)
    r_after = np.abs(out_all[:, 1, :].astype(np.float32)).argmax(axis=1)
    resid = r_after - R_IDX
    keep = np.abs(resid) <= args.max_shift
    print(f"[align] dropped {(~keep).sum()} samples with residual |d|>{args.max_shift} "
          f"({(~keep).mean()*100:.1f}%)", flush=True)
    out = out_all[keep]
    labels_k = labels[keep]

    # verify after shifting
    r2 = np.abs(out[:, 1, :].astype(np.float32)).argmax(axis=1)
    print(f"[align] R-peak position after shift median={np.median(r2):.0f} (target {R_IDX})", flush=True)

    np.savez_compressed(args.out, beats=out, labels=labels_k,
                        classes=d["classes"] if "classes" in d else np.array([]))
    print(f"[align] saved {args.out} beats={out.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
