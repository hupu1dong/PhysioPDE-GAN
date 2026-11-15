"""Generation quality diagnostics: distribution statistics of real vs synthetic beats.

Metrics:
    std       standard deviation of per-lead amplitude (mean over leads)
    kurt      kurtosis (sharp QRS complexes produce high kurtosis)
    hf_ratio  10-40 Hz energy fraction (QRS band)
    lf_ratio  0.5-5 Hz energy fraction (P/T waves and baseline drift)
    lead_corr Frobenius distance between the 12x12 lead correlation matrix and the real one
    sep       centroid distance of real vs synthetic in the first 8 PCA dimensions, normalised by real spread
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SR = 500


def band_ratio(x: np.ndarray, lo: float, hi: float) -> float:
    """x: (N,12,T) -> energy fraction in the given band (averaged over samples)."""
    f = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(x.shape[-1], 1 / SR)
    p = np.abs(f) ** 2
    band = p[..., (freqs >= lo) & (freqs < hi)].sum(-1)
    total = p.sum(-1) + 1e-12
    return float((band / total).mean())


def kurtosis(x: np.ndarray) -> float:
    from scipy.stats import kurtosis as kt
    return float(np.nanmean(kt(x, axis=-1)))


def lead_corr(x: np.ndarray) -> np.ndarray:
    flat = x.transpose(0, 2, 1).reshape(-1, x.shape[1])   # (N*T, 12)
    return np.corrcoef(flat.T)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(ROOT / "data" / "processed" / "ptbxl_beats.npz"))
    ap.add_argument("--syn", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tag", default="pde")
    ap.add_argument("--out", default=str(ROOT / "metrics" / "diagnostics.csv"))
    args = ap.parse_args()

    r = np.load(args.real)
    s = np.load(args.syn)
    fold = r["fold"]
    real = r["beats"][np.where(fold < 0)[0][: args.n]].astype(np.float32)
    syn = s["beats"][: args.n].astype(np.float32)

    row = {"tag": args.tag, "n_real": len(real), "n_syn": len(syn)}
    row["std_real"] = float(real.std()); row["std_syn"] = float(syn.std())
    row["absmax_real"] = float(np.abs(real).max()); row["absmax_syn"] = float(np.abs(syn).max())
    row["kurt_real"] = kurtosis(real); row["kurt_syn"] = kurtosis(syn)
    row["hf_real"] = band_ratio(real, 10, 40); row["hf_syn"] = band_ratio(syn, 10, 40)
    row["lf_real"] = band_ratio(real, 0.5, 5); row["lf_syn"] = band_ratio(syn, 0.5, 5)
    row["leadcorr_dist"] = float(np.linalg.norm(lead_corr(real) - lead_corr(syn)))

    # PCA centroid distance normalised by real spread
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import euclidean_distances
    fr = real.reshape(len(real), -1); fs = syn.reshape(len(syn), -1)
    mu, sd = fr.mean(0, keepdims=True), fr.std(0, keepdims=True) + 1e-6
    pca = PCA(n_components=8, random_state=0).fit((fr - mu) / sd)
    zr = pca.transform((fr - mu) / sd); zs = pca.transform((fs - mu) / sd)
    within = float(np.linalg.norm(zr - zr.mean(0), axis=1).mean())
    row["pca_sep"] = float(np.linalg.norm(zs.mean(0) - zr.mean(0)) / (within + 1e-9))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    print(f"[diag] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
