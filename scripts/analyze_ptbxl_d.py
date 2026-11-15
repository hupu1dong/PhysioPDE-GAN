"""5-fold results of strategy D (synthetic pretraining + real finetuning) on PTB-XL."""
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

CLASSES = ["IAVB", "RBBB", "LBBB", "NSIVCB", "LAnFB", "AFL"]

d = defaultdict(lambda: defaultdict(dict))
with open("metrics/table_ptbxl_d.csv") as f:
    for row in csv.DictReader(f):
        if row["cls"] == "QAb" or row["arch"] != "resnet":
            continue
        v = row["spec_paper"]
        d[row["tag"]][row["cls"]][int(row["fold"])] = \
            float(v) if v not in ("", "nan") else np.nan

print(f"{'tag':<16}" + "".join(f"{c:>9}" for c in CLASSES))
print("-" * 70)
for t in ("ptbxl-real", "ptbxl-pdept"):
    if t not in d or not d[t]:
        print(f"{t:<16}(missing)")
        continue
    print(f"{t:<16}" + "".join(
        f"{np.nanmean([d[t][c].get(f, np.nan) for f in range(5)]):9.3f}" for c in CLASSES))

print("\n=== paired t-test (vs ptbxl-real) ===")
for c in CLASSES:
    a = np.array([d["ptbxl-pdept"][c].get(f, np.nan) for f in range(5)])
    b = np.array([d["ptbxl-real"][c].get(f, np.nan) for f in range(5)])
    delta = a - b
    ok = ~np.isnan(delta)
    if ok.sum() < 3:
        continue
    t, p = stats.ttest_1samp(delta[ok], 0)
    if p < 0.05 and np.nanmean(delta) > 0:
        mark = "  <- significant positive"
    elif p < 0.05:
        mark = "  <- significant negative"
    elif p < 0.10:
        mark = "  (marginal)"
    else:
        mark = ""
    print(f"  {c:<8} Δ={np.nanmean(delta):+.4f}  {delta.round(3)}  "
          f"pos {(delta > 0).sum()}/5  p={p:.3f}{mark}")
