"""Aggregate classifier results under the paper protocol (fixed sensitivity, compare specificity).

Paper §4.3: "All experiments fix classifier sensitivity at a set threshold and compare
specificity on a held-out test set of unseen real samples"。

Two protocols are provided:
    * paper  : threshold tuned on the test set to the target sensitivity (paper description)
    * strict : threshold tuned on the validation set, specificity measured on the test set (no leakage)
Target sensitivity defaults to the baseline's strict-protocol test sensitivity.

Usage:
  python src/evaluate.py --glob 'metrics/probs_*.npz' --baseline-tag real
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.labels import ALL_LABELS, CLASSES  # noqa: E402


def thr_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Find a threshold that brings the sensitivity of this class close to target."""
    pos = p[y == 1]
    if len(pos) == 0:
        return 0.5
    if target <= 0:
        return float(pos.max() + 1e-6)
    if target >= 1:
        return float(pos.min() - 1e-6)
    # quantile: threshold that recalls the target fraction of positives
    return float(np.quantile(pos, 1.0 - target))


def sens_spec(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, float]:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens, spec


def auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _fmt(vals, npos) -> str:
    if npos == 0 or not vals:
        return f"{'—':>18s}"
    m = float(np.mean(vals))
    return f"{m:>12.3f}±{np.std(vals):.3f}" if np.isfinite(m) else f"{'—':>18s}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "metrics" / "probs_*.npz"))
    ap.add_argument("--baseline-tag", default="real")
    ap.add_argument("--target-sens", type=float, default=-1,
                    help="<0 uses the baseline strict-protocol sensitivity as the target")
    ap.add_argument("--out", default=str(ROOT / "metrics" / "table_specificity.csv"))
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print("no probs files found", file=sys.stderr)
        return 1

    runs = {}
    for f in files:
        name = Path(f).stem
        m = re.match(r"probs_(.+)_f(\d+)_(\w+)$", name)
        tag, fold, arch = (m.group(1), int(m.group(2)), m.group(3)) if m else (name, -1, "?")
        z = np.load(f)
        runs[(tag, arch, fold)] = dict(y=z["y_true"], p=z["y_prob"],
                                       yv=z["y_true_val"], pv=z["y_prob_val"],
                                       classes=[str(c) for c in z["classes"]])
    print(f"[eval] {len(runs)} runs: {sorted(set((k[0], k[1]) for k in runs))}")

    # target sensitivity: baseline strict-protocol measurement
    targets = {}
    for c_i, c in enumerate(CLASSES):
        vals = []
        for (tag, arch, fold), r in runs.items():
            if tag != args.baseline_tag:
                continue
            thr = thr_for_sensitivity(r["yv"][:, c_i], r["pv"][:, c_i], 0.90)
            s, _ = sens_spec(r["y"][:, c_i], r["p"][:, c_i], thr)
            vals.append(s)
        if args.target_sens > 0:
            targets[c] = args.target_sens
        else:
            targets[c] = float(np.mean(vals)) if vals else 0.90
    print("[eval] target sensitivity: " +
          ", ".join(f"{c}={targets[c]:.3f}" for c in CLASSES))

    rows = []
    for (tag, arch, fold), r in sorted(runs.items()):
        for c_i, c in enumerate(CLASSES):
            y, p = r["y"][:, c_i], r["p"][:, c_i]
            yv, pv = r["yv"][:, c_i], r["pv"][:, c_i]
            t = targets[c]
            # strict: threshold from the validation set
            thr_v = thr_for_sensitivity(yv, pv, t)
            s_str, sp_str = sens_spec(y, p, thr_v)
            # paper: threshold from the test set
            thr_t = thr_for_sensitivity(y, p, t)
            s_pap, sp_pap = sens_spec(y, p, thr_t)
            rows.append(dict(tag=tag, arch=arch, fold=fold, cls=c,
                             target_sens=t, n_pos=int((y == 1).sum()),
                             sens_strict=s_str, spec_strict=sp_str,
                             sens_paper=s_pap, spec_paper=sp_pap,
                             auc=auc(y, p)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # aggregate: mean and std over folds for each (tag, arch)
    print("\n=== specificity (strict: threshold from validation, target sens from baseline) ===")
    header = f"{'class':8s}" + "".join(f"{f'{t}/{a}':>18s}" for t, a in sorted(set((r['tag'], r['arch']) for r in rows)))
    print(header)
    groups = sorted(set((r["tag"], r["arch"]) for r in rows))
    for c in CLASSES:
        line = f"{c:8s}"
        for g in groups:
            sel = [r for r in rows if r["cls"] == c and (r["tag"], r["arch"]) == g]
            line += _fmt([r["spec_strict"] for r in sel], max([r["n_pos"] for r in sel], default=0))
        print(line)
    print("\n=== specificity (paper: uniform sensitivity on test, no validation threshold) ===")
    print(header)
    for c in CLASSES:
        line = f"{c:8s}"
        for g in groups:
            sel = [r for r in rows if r["cls"] == c and (r["tag"], r["arch"]) == g]
            line += _fmt([r["spec_paper"] for r in sel], max([r["n_pos"] for r in sel], default=0))
        print(line)

    print("\n=== sensitivity（paper）===")
    print(header)
    for c in CLASSES:
        line = f"{c:8s}"
        for g in groups:
            sel = [r for r in rows if r["cls"] == c and (r["tag"], r["arch"]) == g]
            line += _fmt([r["sens_paper"] for r in sel], max([r["n_pos"] for r in sel], default=0))
        print(line)

    print("\n=== sensitivity（strict）===")
    print(header)
    for c in CLASSES:
        line = f"{c:8s}"
        for g in groups:
            sel = [r for r in rows if r["cls"] == c and (r["tag"], r["arch"]) == g]
            line += _fmt([r["sens_strict"] for r in sel], max([r["n_pos"] for r in sel], default=0))
        print(line)
    print("\n=== AUC ===")
    print(header)
    for c in CLASSES:
        line = f"{c:8s}"
        for g in groups:
            sel = [r for r in rows if r["cls"] == c and (r["tag"], r["arch"]) == g]
            line += _fmt([r["auc"] for r in sel], max([r["n_pos"] for r in sel], default=0))
        print(line)
    print(f"\n[eval] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
