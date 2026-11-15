import csv
from collections import defaultdict

import numpy as np
from scipy import stats

CLASSES = ["IAVB", "RBBB", "LBBB", "NSIVCB", "LAnFB", "AFL"]

SPECS = [
    ("real（resnet）", "metrics/table_exp_resnet.csv", "g12-real", "resnet"),
    ("+pde 1:1（resnet）", "metrics/table_exp_resnet.csv", "g12-real+pdeal", "resnet"),
    ("+pde r=0.5", "metrics/table_exp_resnet.csv", "g12-pdealr05", "resnet"),
    ("+pde r=2.0", "metrics/table_exp_resnet.csv", "g12-pdealr20", "resnet"),
    ("real（attn）", "metrics/table_exp_attn.csv", "g12-real", "attn"),
    ("+pde 1:1（attn）", "metrics/table_exp_attn.csv", "g12-pdeal", "attn"),
]

CACHE = {}


def load(path):
    if path in CACHE:
        return CACHE[path]
    d = defaultdict(lambda: defaultdict(dict))
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["cls"] == "QAb":
                    continue
                v = row["spec_paper"]
                d[(row["tag"], row["arch"])][row["cls"]][int(row["fold"])] = \
                    float(v) if v and v != "nan" else np.nan
    except FileNotFoundError:
        pass
    CACHE[path] = d
    return d


def mat(d, key, cls):
    return np.array([d[key][cls].get(f, np.nan) for f in range(5)])


rows = {}
print("-" * 78)
for name, path, tag, arch in SPECS:
    d = load(path)
    key = (tag, arch)
    if key not in d or not d[key]:
        continue
    m = {c: mat(d, key, c) for c in CLASSES}
    rows[name] = (m, arch)
    print(f"{name:<22}" + "".join(f"{np.nanmean(m[c]):9.3f}" for c in CLASSES))

for name, (m, arch) in rows.items():
    if name.startswith("real"):
        continue
    base_name = "real（attn）" if arch == "attn" else "real（resnet）"
    if base_name not in rows:
        continue
    ref = rows[base_name][0]
    print(f"\n{name}:")
    for c in CLASSES:
        delta = m[c] - ref[c]
        ok = ~np.isnan(delta)
        if ok.sum() < 3:
            continue
        t, p = stats.ttest_1samp(delta[ok], 0)
        print(f"  {c:<8} Δ={np.nanmean(delta):+.4f}  p={p:.3f}{mark}")
