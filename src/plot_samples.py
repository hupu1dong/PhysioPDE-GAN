"""Reproduce Figure 2 of the paper: qualitative real vs generated comparison (Lead II / V1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", required=True, help="synthetic beats npz")
    ap.add_argument("--real", default=str(ROOT / "data" / "processed" / "ptbxl_beats.npz"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--lead", default="II,V1")
    ap.add_argument("--title", default="PhysioPDE-GAN (ours)")
    ap.add_argument("--out", default=str(ROOT / "samples" / "fig2_real_vs_generated.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.load(args.real)
    s = np.load(args.syn)
    fold = r["fold"]
    test_idx = np.where(fold < 0)[0][: args.n]
    real = r["beats"][test_idx].astype(np.float32)
    gen = s["beats"][: args.n].astype(np.float32)

    leads = [LEAD_NAMES.index(x.strip()) for x in args.lead.split(",")]
    t = (np.arange(real.shape[-1]) - 140) / 500.0

    fig, axes = plt.subplots(len(leads), 2, figsize=(10, 2.6 * len(leads)), sharex=True)
    if len(leads) == 1:
        axes = np.atleast_2d(axes)
    for j, li in enumerate(leads):
        axl, axr = axes[j, 0], axes[j, 1]
        for i in range(args.n):
            axl.plot(t, real[i, li] + i * 1.2, lw=0.9, color="black")
            axr.plot(t, gen[i, li] + i * 1.2, lw=0.9, color="tab:orange", ls="--")
        axl.set_ylabel(f"Lead {LEAD_NAMES[li]}\nVoltage (norm)")
        axl.set_title("Real ECG" if j == 0 else "")
        axr.set_title(args.title if j == 0 else "")
        axl.grid(alpha=0.25); axr.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Time relative to R peak (s)")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"[plot] saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
