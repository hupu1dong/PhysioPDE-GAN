"""PTB-XL comparison: cross-domain pretraining vs same-domain pretraining vs same-domain mixing."""
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

CLASSES = ["IAVB", "RBBB", "LBBB", "NSIVCB", "LAnFB", "AFL"]

GROUPS = [
    ("real (baseline)", "ptbxl-real"),
    ("+pde 1:1 (G12EC samples)", "ptbxl-real+pde"),
    ("D: G12EC pretraining", "ptbxl-pdept"),
    ("D: PTB-XL pretraining", "ptbxl-pdept2"),
    ("1:1 (PTB-XL samples)", "ptbxl-pde2"),
]

d = defaultdict(lambda: defaultdict(dict))
with open("metrics/table_ptbxl_final.csv") as f:
    for row in csv.DictReader(f):
        if row["cls"] == "QAb" or row["arch"] != "resnet":
            continue
        v = row["spec_paper"]
        d[row["tag"]][row["cls"]][int(row["fold"])] = \
            float(v) if v not in ("", "nan") else np.nan


def m(tag, c):
    return np.array([d[tag][c].get(f, np.nan) for f in range(5)])


print(f"{'setting':<26}" + "".join(f"{c:>9}" for c in CLASSES))
print("-" * 82)
for name, tag in GROUPS:
    if tag not in d or not d[tag]:
        print(f"{name:<26}(missing)")
        continue
    print(f"{name:<26}" + "".join(f"{np.nanmean(m(tag, c)):9.3f}" for c in CLASSES))

print("\n=== paired t-test (vs ptbxl-real) ===")
for name, tag in GROUPS:
    if tag == "ptbxl-real" or tag not in d or not d[tag]:
        continue
    print(f"\n{name}:")
    for c in CLASSES:
        a, b = m(tag, c), m("ptbxl-real", c)
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
        print(f"  {c:<8} Δ={np.nanmean(delta):+.4f}  "
              f"pos {(delta > 0).sum()}/5  p={p:.3f}{mark}")
