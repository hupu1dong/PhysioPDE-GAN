"""Check whether conditional generation works: fix z, vary the class label, inspect the output difference."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.gan import GenConfig, PhysioPDEGANGenerator  # noqa: E402


def main() -> int:
    ck_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "checkpoints" / "g12_cond_f0" / "physiopde_gan_last.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    kw = {k: v for k, v in ck["cfg"].items() if k in GenConfig.__dataclass_fields__}
    kw["n_classes"] = max(int(ck["gen"]["fc.weight"].shape[1]) - kw.get("z_dim", 128), 0)
    G = PhysioPDEGANGenerator(GenConfig(**kw)).eval()
    G.load_state_dict(ck["gen"])
    cls = ["IAVB", "RBBB", "LBBB", "NSIVCB", "LAnFB", "QAb", "AFL", "NORM"]

    torch.manual_seed(0)
    z = torch.randn(8, 128)
    outs = []
    with torch.no_grad():
        for i in range(8):
            y = torch.zeros(8, 8)
            y[:, i] = 1
            u, _ = G(z, labels=y)
            outs.append(u.numpy())
    outs = np.stack(outs)   # (8 classes, 8 samples, 12, 400)

    print("Mean waveform difference across classes for the same z (L2/sqrt(N)):")
    diffs = []
    for i in range(8):
        for j in range(i + 1, 8):
            d = float(np.linalg.norm(outs[i].mean(0) - outs[j].mean(0)) / np.sqrt(outs[i][0].size))
            diffs.append(d)
    print(f"  between-class: mean={np.mean(diffs):.4f} min={np.min(diffs):.4f} max={np.max(diffs):.4f}")

    within = []
    for i in range(8):
        for k in range(1, 8):
            within.append(float(np.linalg.norm(outs[i][0] - outs[i][k]) / np.sqrt(outs[i][0].size)))
    print(f"  within-class (different z): mean={np.mean(within):.4f}")
    print(f"  -> between/within ratio = {np.mean(diffs)/max(np.mean(within),1e-9):.3f} (>1 means conditioning works)")

    for i in range(8):
        per_lead_std = outs[i].std(axis=(0, 2))   # (12,)
        print(f"  {cls[i]:7s} mean std={outs[i].std():.4f}  per-lead std range=[{per_lead_std.min():.3f},{per_lead_std.max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
