import numpy as np
import torch


def kurtosis(x):
    m = x.mean(-1, keepdims=True)
    sd = x.std(-1, keepdims=True) + 1e-6
    return float((((x - m) / sd) ** 4).mean())


def hf_ratio(x, sr=500, lo=10, hi=40):
    f = np.fft.rfft(x, axis=-1)
    p = np.abs(f) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    return float((p[..., mask].sum(-1) / (p.sum(-1) + 1e-8)).mean())


def lead_corr_dist(x):
    x = x.reshape(-1, 12, x.shape[-1])
    c = np.corrcoef(x.reshape(12, -1))
    off = c - np.eye(12)
    return float(np.abs(off).mean())


print("-" * 58)

z = np.load("data/processed/ptbxl_beats.npz")
real = z["beats"][:2000].astype(np.float32)

for f in range(5):
    s = np.load(f"data/synthetic/pde_ptbxl_f{f}_aligned.npz")["beats"][:2000].astype(np.float32)

try:
    g = np.load("data/synthetic/pde_g12_f0_aligned.npz")["beats"][:2000].astype(np.float32)
except FileNotFoundError:
    pass

try:
    zg = np.load("data/processed/g12ec_beats.npz")
    r = zg["beats"][:2000].astype(np.float32)
except FileNotFoundError:
    pass
