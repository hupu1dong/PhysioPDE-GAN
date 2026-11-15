import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve

CLASSES = ["IAVB", "RBBB", "LBBB", "NSIVCB", "LAnFB", "QAb", "AFL"]


def spec_at_sens(y, p, sens=0.94):
    if y.min() == y.max():
        return np.nan
    fpr, tpr, _ = roc_curve(y, p)
    idx = min(np.searchsorted(tpr, sens), len(fpr) - 1)
    return 1 - fpr[idx]


def load(tag, arch="resnet"):
    out = {}
    for f in range(5):
        try:
            out[f] = np.load(f"metrics/probs_{tag}_f{f}_{arch}.npz")
        except FileNotFoundError:
            return None
    return out


def spec_mat(probs):
    rows = []
    for f in range(5):
        d = probs[f]
        y, p = d["y_true"], d["y_prob"]
        rows.append([spec_at_sens(y[:, i], p[:, i]) for i in range(y.shape[1])])
    return np.array(rows)


GROUPS = [
    ("real (resnet)", "g12-real", "resnet", False),
    ("pdeal (resnet)", "g12-pdeal", "resnet", False),
    ("real (attn)", "g12-real", "attn", True),
    ("pdeal (attn)", "g12-pdeal", "attn", True),
    ("pdeal ratio=0.5", "g12-pdealr05", "resnet", False),
    ("pdeal ratio=2.0", "g12-pdealr20", "resnet", False),
    ("pdeal weak-only", "g12-pdealweak", "resnet", False),
    ("pdeal pretrain", "g12-pdept", "resnet", False),
]

data, is_attn = {}, {}
print("-" * 82)
for name, tag, arch, attn in GROUPS:
    pr = load(tag, arch)
    if pr is None:
        continue
    m = spec_mat(pr)
    data[name] = m
    is_attn[name] = attn
    print(f"{name:<18}" + "".join(f"{np.nanmean(m[:, i]):8.3f}" for i in range(len(CLASSES))))

for name, m in data.items():
    if name.startswith("real"):
        continue
    ref = data.get("real (attn)") if is_attn[name] else data.get("real (resnet)")
    if ref is None:
        continue
    print(f"\n{name}:")
    for i, c in enumerate(CLASSES):
        d = m[:, i] - ref[:, i]
        ok = ~np.isnan(d)
        if ok.sum() < 3:
            continue
        t, p = stats.ttest_1samp(d[ok], 0)
        print(f"  {c:<8} Δ={np.nanmean(d):+.4f}  t={t:+6.2f}  p={p:.3f}{flag}")
