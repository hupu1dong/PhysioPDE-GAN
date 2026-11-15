"""5-fold paired t-test: per class, compare specificity of real+synthetic against real.

Same protocol as evaluate.py in paper mode: threshold tuned on the test set to the target
sensitivity (the baseline strict-protocol measurement), then specificity is compared.

Usage:
  python src/paired_test.py --glob 'metrics/probs_g12-*.npz' --out metrics/paired_ttest_g12.csv
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

from src.data.labels import CLASSES  # noqa: E402
from src.evaluate import sens_spec, thr_for_sensitivity  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "metrics" / "probs_g12-*.npz"))
    ap.add_argument("--baseline-tag", default="g12-real")
    ap.add_argument("--target-sens", type=float, default=-1)
    ap.add_argument("--out", default=str(ROOT / "metrics" / "paired_ttest_g12.csv"))
    args = ap.parse_args()

    runs: dict[tuple[str, int], dict] = {}
    for f in sorted(glob.glob(args.glob)):
        m = re.match(r"probs_(.+)_f(\d+)_(\w+)$", Path(f).stem)
        if not m:
            continue
        tag, fold = m.group(1), int(m.group(2))
        z = np.load(f)
        runs[(tag, fold)] = dict(y=z["y_true"], p=z["y_prob"],
                                 yv=z["y_true_val"], pv=z["y_prob_val"])
    folds = sorted({k[1] for k in runs})
    tags = sorted({k[0] for k in runs if k[0] != args.baseline_tag})
    print(f"[t] folds={folds} comparisons={tags}")

    # target sensitivity: mean of the baseline strict measurement across folds
    targets = {}
    for c_i, c in enumerate(CLASSES):
        vals = []
        for fo in folds:
            r = runs.get((args.baseline_tag, fo))
            if r is None:
                continue
            thr = thr_for_sensitivity(r["yv"][:, c_i], r["pv"][:, c_i], 0.90)
            s, _ = sens_spec(r["y"][:, c_i], r["p"][:, c_i], thr)
            vals.append(s)
        targets[c] = args.target_sens if args.target_sens > 0 else (float(np.mean(vals)) if vals else 0.90)

    rows = []
    from scipy.stats import ttest_1samp
    for c_i, c in enumerate(CLASSES):
        base = {}
        for fo in folds:
            r = runs.get((args.baseline_tag, fo))
            if r is None:
                continue
            thr = thr_for_sensitivity(r["y"][:, c_i], r["p"][:, c_i], targets[c])
            base[fo] = sens_spec(r["y"][:, c_i], r["p"][:, c_i], thr)[1]
        for tag in tags:
            deltas = []
            for fo in folds:
                r = runs.get((tag, fo))
                if r is None or fo not in base:
                    continue
                thr = thr_for_sensitivity(r["y"][:, c_i], r["p"][:, c_i], targets[c])
                sp = sens_spec(r["y"][:, c_i], r["p"][:, c_i], thr)[1]
                deltas.append(sp - base[fo])
            if len(deltas) < 2:
                rows.append(dict(cls=c, compare=tag, n=len(deltas), mean_delta=float("nan"),
                                 std=float("nan"), t=float("nan"), p=float("nan"), deltas=""))
                continue
            d = np.asarray(deltas)
            t, p = ttest_1samp(d, 0.0)
            rows.append(dict(cls=c, compare=tag, n=len(d), mean_delta=float(d.mean()),
                             std=float(d.std(ddof=1)), t=float(t), p=float(p),
                             deltas=";".join(f"{x:.4f}" for x in d)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cls", "compare", "n", "mean_delta",
                                          "std", "t", "p", "deltas"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'class':8s}{'compare':18s}{'n':>3s}{'mean Δspec':>12s}{'std':>9s}{'t':>8s}{'p':>9s}")
    for r in rows:
        print(f"{r['cls']:8s}{r['compare']:18s}{r['n']:>3d}{r['mean_delta']:>12.4f}"
              f"{r['std']:>9.4f}{r['t']:>8.2f}{r['p']:>9.4f}")
    print(f"\n[t] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
