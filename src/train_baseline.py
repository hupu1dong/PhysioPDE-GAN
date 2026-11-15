"""Train DCGAN / WaveGAN baselines (adversarial loss only); outputs use the same format as PDE-GAN."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.baselines import DCGANDiscriminator1d, DCGANGenerator1d, WaveGANGenerator1d  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(ROOT / "data" / "processed" / "ptbxl_beats.npz"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", choices=["dcgan", "wavegan"], default="dcgan")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load(args.npz)
    beats, fold = d["beats"], d["fold"]
    tr = np.where(fold != args.fold)[0]
    if len(tr) == 0:
        print("training set is empty", file=sys.stderr)
        return 1
    print(f"[{args.model}] train={len(tr)} dev={dev}", flush=True)

    G = (DCGANGenerator1d(args.z_dim) if args.model == "dcgan"
         else WaveGANGenerator1d(args.z_dim)).to(dev)
    D = DCGANDiscriminator1d().to(dev)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{args.model}_best.pt"
    log_path = Path(args.log) if args.log else ROOT / "logs" / f"train_{args.model}_f{args.fold}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a")
    log_f.write(json.dumps(vars(args)) + "\n")

    spe = max(len(tr) // args.batch, 1)
    for ep in range(1, args.epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        ad = ag = 0.0
        for _ in range(spe):
            idx = np.random.choice(tr, size=args.batch, replace=len(tr) < args.batch)
            real = torch.from_numpy(beats[idx].astype(np.float32)).to(dev)
            z = torch.randn(real.shape[0], args.z_dim, device=dev)
            with torch.no_grad():
                fake = G(z)
            ld = F.binary_cross_entropy_with_logits(D(real), torch.ones(real.shape[0], device=dev)) \
                + F.binary_cross_entropy_with_logits(D(fake), torch.zeros(real.shape[0], device=dev))
            opt_d.zero_grad(set_to_none=True); ld.backward(); opt_d.step()

            z = torch.randn(real.shape[0], args.z_dim, device=dev)
            fake = G(z)
            lg = F.binary_cross_entropy_with_logits(D(fake), torch.ones(real.shape[0], device=dev))
            opt_g.zero_grad(set_to_none=True); lg.backward(); opt_g.step()
            ad += ld.item(); ag += lg.item()
        msg = f"ep{ep:03d} d={ad/spe:.3f} g={ag/spe:.3f} {time.time()-t0:.1f}s"
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()
        torch.save({"gen": G.state_dict(), "disc": D.state_dict(), "model": args.model,
                    "args": vars(args), "epoch": ep}, best_path)
    log_f.close()
    print(f"[{args.model}] saved {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
