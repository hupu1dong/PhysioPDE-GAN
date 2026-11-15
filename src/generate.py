"""Generate synthetic beats with a trained PhysioPDE-GAN for downstream augmentation.

Label strategy (assumption A4): the generator is unconditional, so synthetic labels are resampled
from the real class distribution of the training set, leaving the class prior unchanged.

Usage:
  python src/generate.py --ckpt checkpoints/physiopde_gan_best.pt \
      --npz data/processed/ptbxl_beats.npz --fold 0 --n-per-class 500 \
      --out data/synthetic/pde_f0.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.gan import GenConfig, PhysioPDEGANGenerator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", default=str(ROOT / "data" / "processed" / "ptbxl_beats.npz"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="total samples; 0 means match the training set size")
    ap.add_argument("--n-per-class", type=int, default=0, help="per-class quota (overrides --n when > 0)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    if "model" in ck:                      # DCGAN / WaveGAN baseline
        from src.models.baselines import DCGANGenerator1d, WaveGANGenerator1d
        z_dim = ck["args"].get("z_dim", 128)
        G = (DCGANGenerator1d(z_dim) if ck["model"] == "dcgan"
             else WaveGANGenerator1d(z_dim)).to(dev).eval()
        G.load_state_dict(ck["gen"])
        cfg = GenConfig(z_dim=z_dim)
    else:                                   # PhysioPDE-GAN (also accepts older unconditional weights)
        kw = {k: v for k, v in ck["cfg"].items() if k in GenConfig.__dataclass_fields__}
        fc_w = ck["gen"]["fc.weight"]
        kw["n_classes"] = max(int(fc_w.shape[1]) - kw.get("z_dim", 128), 0)
        cfg = GenConfig(**kw)
        G = PhysioPDEGANGenerator(cfg).to(dev).eval()
        G.load_state_dict(ck["gen"])
        print(f"[gen] n_classes={cfg.n_classes} (0 = unconditional)", flush=True)

    d = np.load(args.npz)
    labels = d["labels"]
    fold = d["fold"]
    split = d["split"] if "split" in d else np.where(fold < 0, 1, 0)
    tr = (split == 0) & (fold != args.fold)

    if args.n_per_class > 0:
        idx = []
        for c in range(labels.shape[1]):
            pool = np.where(tr & (labels[:, c] == 1))[0]
            if len(pool) == 0:
                continue
            idx += np.random.choice(pool, size=args.n_per_class, replace=True).tolist()
        total = len(idx)
        src_labels = labels[idx]
    else:
        n = args.n if args.n > 0 else int(tr.sum())
        idx = np.random.choice(np.where(tr)[0], size=n, replace=True)
        total = n
        src_labels = labels[idx]
    print(f"[gen] generated {total} synthetic beats (labels resampled from the training distribution)", flush=True)

    outs = []
    with torch.no_grad():
        for s in range(0, total, args.batch):
            nb = min(args.batch, total - s)
            z = torch.randn(nb, cfg.z_dim, device=dev)
            if cfg.n_classes > 0:
                y = torch.from_numpy(src_labels[s:s + nb].astype(np.float32)).to(dev)
                out = G(z, labels=y)
            else:
                out = G(z)
            u = out[0] if isinstance(out, (tuple, list)) else out
            outs.append(u.cpu().numpy().astype(np.float16))
    beats = np.concatenate(outs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, beats=beats, labels=src_labels.astype(np.uint8),
                        classes=d["classes"], src_ckpt=str(args.ckpt))
    print(f"[gen] saved {out} beats={beats.shape} "
          f"range=[{beats.min():.2f},{beats.max():.2f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
