import csv
from collections import defaultdict

import numpy as np
from scipy import stats

CLASSES = ["IAVB", "RBBB", "LBBB", "NSIVCB", "LAnFB", "AFL"]
ARCH = "resnet"


def load(path):
    """tag -> cls -> fold -> spec_paper"""
    d = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["arch"] != ARCH:
                continue
            if row["cls"] == "QAb":
                continue
            t, c, fo = row["tag"], row["cls"], int(row["fold"])
            v = row["spec_paper"]
            d[t][c][fo] = float(v) if v and v != "nan" else np.nan
    return d


def mat(d, tag, cls):
    return np.array([d[tag][cls].get(f, np.nan) for f in range(5)])


def main():
    d = load("metrics/table_exp_resnet.csv")
    tags = ["g12-real", "g12-pdeal", "g12-pdealr05", "g12-pdealr20",
            "g12-pdealweak", "g12-pdept"]
    tags = [t for t in tags if t in d]

    print(f"{'tag':<18}" + "".join(f"{c:>9}" for c in CLASSES))
    print("-" * 80)
    for t in tags:
        vals = [np.nanmean(mat(d, t, c)) for c in CLASSES]
        print(f"{t:<18}" + "".join(f"{v:9.3f}" for v in vals))

    for t in tags:
        if t == "g12-real":
            continue
        print(f"\n{t}:")
        for c in CLASSES:
            a, b = mat(d, t, c), mat(d, "g12-real", c)
            delta = a - b
            ok = ~np.isnan(delta)
            if ok.sum() < 3:
                continue
            tt, p = stats.ttest_1samp(delta[ok], 0)
            print(f"  {c:<8} Δ={np.nanmean(delta):+.4f}  "
                  f"[{', '.join(f'{x:+.3f}' for x in delta)}]  p={p:.3f}{star}")


if __name__ == "__main__":
    main()
