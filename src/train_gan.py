"""Train PhysioPDE-GAN (paper §3.4).

Usage:
  python src/train_gan.py --npz data/processed/ptbxl_beats.npz --fold 0 --epochs 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.gan import ECGCritic, GenConfig, PhysioPDEGANGenerator  # noqa: E402

PRE = 140
SR = 500
W = 400


def hf_ratio(x, lo=10.0, hi=40.0, sr=500.0):
    """Fraction of energy in the 10-40 Hz band (QRS band)."""
    f = torch.fft.rfft(x.float(), dim=-1)
    p = f.abs().pow(2)
    freqs = torch.fft.rfftfreq(x.shape[-1], 1.0 / sr, device=x.device)
    mask = (freqs >= lo) & (freqs < hi)
    return (p[..., mask].sum(-1) / (p.sum(-1) + 1e-8)).mean()


def kurtosis_t(x):
    m = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True) + 1e-6
    return ((x - m) / sd).pow(4).mean()


def align_loss(u, r_idx: int = 140, alpha: float = 20.0):
    """Require the Lead II amplitude at r_idx (R peak) to approach the global maximum (differentiable soft-peak constraint)."""
    absl = u[:, 1, :].abs()
    peak = absl[:, r_idx]
    lse = torch.logsumexp(alpha * absl, dim=-1) / alpha
    return torch.relu(lse - peak).mean()


def build_masks(peaks: np.ndarray, seq_len: int = W, pre: int = PRE) -> torch.Tensor:
    """Build physiological window masks for the three subsystems from P/Q/S/T positions (seconds).

    Args:
        peaks: (N,4) [P,Q,S,T] in seconds relative to the R peak; NaN falls back to the cohort median
    Returns:
        (N,3,1,seq_len) 0/1 masks in SA/AV/HP order
    """
    p = np.nanmedian(peaks[:, 0]) if np.isfinite(peaks[:, 0]).any() else -0.16
    q = np.nanmedian(peaks[:, 1]) if np.isfinite(peaks[:, 1]).any() else -0.03
    t = np.nanmedian(peaks[:, 3]) if np.isfinite(peaks[:, 3]).any() else 0.26

    t_axis = (np.arange(seq_len) - pre) / SR          # time relative to the R peak (seconds)
    masks = np.zeros((3, seq_len), np.float32)
    sa = (t_axis >= p - 0.055) & (t_axis <= p + 0.055)
    av = (t_axis > p + 0.055) & (t_axis <= q)
    hp = (t_axis > q - 0.01) & (t_axis <= t + 0.13)
    masks[0, sa] = 1.0
    masks[1, av] = 1.0
    masks[2, hp] = 1.0
    return torch.from_numpy(masks)                      # (3,W)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(ROOT / "data" / "processed" / "ptbxl_beats.npz"))
    ap.add_argument("--fold", type=int, default=0, help="CV fold used for validation; the rest are used for training")
    ap.add_argument("--resume", action="store_true", help="back up *_best.pt before resuming")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=1e-4, help="discriminator lr (lower than G to keep D in check)")
    ap.add_argument("--label-smooth", type=float, default=0.9)
    ap.add_argument("--loss", choices=["bce", "hinge"], default="hinge")
    ap.add_argument("--r1", type=float, default=10.0, help="R1 gradient penalty coefficient (0 = off)")
    ap.add_argument("--r1-every", type=int, default=16)
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--lambda-pde", type=float, default=0.1)
    ap.add_argument("--lambda-temporal", type=float, default=0.2)
    ap.add_argument("--lambda-stat", type=float, default=1.0, help="amplitude statistics matching (suppresses trivial constant solutions)")
    ap.add_argument("--lambda-spec", type=float, default=2.0, help="QRS-band energy fraction matching")
    ap.add_argument("--lambda-kurt", type=float, default=0.05, help="kurtosis matching (peak sharpness)")
    ap.add_argument("--lambda-align", type=float, default=5.0, help="R-peak alignment loss (in-model)")
    ap.add_argument("--spatial", choices=["conv", "global", "attn", "both"], default="conv",
                    help="inter-lead spatial modelling")
    ap.add_argument("--no-coupling", action="store_true", help="ablation: disable SA->AV->HP coupling")
    ap.add_argument("--pde-mode", choices=["cn", "euler"], default="cn")
    ap.add_argument("--pde-last-n", type=int, default=8, help="compute the PDE residual over the last N steps only")
    ap.add_argument("--steps-per-epoch", type=int, default=200, help="optimisation steps per epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-beats", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load(args.npz)
    beats = d["beats"]
    Y_all = d["labels"].astype(np.float32)
    fold = d["fold"]
    peaks = d["peaks"]
    split = d["split"] if "split" in d else np.where(fold < 0, 1, 0)
    # fit the generator on the CV training part only (split==0) to avoid test leakage
    tr = np.where((split == 0) & (fold != args.fold))[0]
    va = np.where((split == 0) & (fold == args.fold))[0]
    if args.max_beats:
        tr = tr[: args.max_beats]
    print(f"[gan] train={len(tr)} val={len(va)} beats, dev={dev}", flush=True)

    masks = build_masks(peaks).to(dev)                 # (1,3,1,W)

    cfg = GenConfig(seq_len=args.steps, z_dim=args.z_dim, hidden=args.hidden, dt=args.dt,
                    spatial=args.spatial, coupling=not args.no_coupling)
    G = PhysioPDEGANGenerator(cfg).to(dev)
    D = ECGCritic().to(dev)
    n_g = sum(p.numel() for p in G.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"[gan] G params={n_g/1e3:.1f}K  D params={n_d/1e3:.1f}K", flush=True)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "physiopde_gan_best.pt"
    if args.resume and best_path.exists():
        best_path.rename(out_dir / f"physiopde_gan_best_backup_{int(time.time())}.pt")

    log_path = Path(args.log) if args.log else ROOT / "logs" / f"train_gan_f{args.fold}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a")
    log_f.write(json.dumps(vars(args)) + "\n")

    def sample_batch(idx: np.ndarray, n: int):
        take = np.random.choice(idx, size=min(n, len(idx)), replace=len(idx) < n)
        x = torch.from_numpy(beats[take].astype(np.float32))
        y = torch.from_numpy(Y_all[take])
        return x.to(dev), y.to(dev)

    real_std = float(beats[tr[:2000]].astype(np.float32).std())
    print(f"[gan] real_std={real_std:.4f}", flush=True)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else max(len(tr) // args.batch, 1)
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        agg = {"d": 0.0, "g": 0.0, "adv": 0.0, "pde": 0.0, "tmp": 0.0, "stat": 0.0,
               "spec": 0.0, "kurt": 0.0, "align": 0.0}
        for step_i in range(steps_per_epoch):
            # ---- Discriminator ----
            real, y_real = sample_batch(tr, args.batch)
            z = torch.randn(real.shape[0], args.z_dim, device=dev)
            with torch.no_grad():
                fake, _ = G(z, labels=y_real)
            d_real = D(real + 0.02 * torch.randn_like(real))
            d_fake = D(fake)
            if args.loss == "hinge":
                loss_d = torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()
            else:
                loss_d = F.binary_cross_entropy_with_logits(
                    d_real, torch.full_like(d_real, args.label_smooth)) \
                    + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            # R1 regularisation (every r1_every steps): constrains the gradient of D on real samples
            if args.r1 > 0 and step_i % args.r1_every == 0:
                real_r1 = real.detach().clone().requires_grad_(True)
                dr = D(real_r1)
                grad = torch.autograd.grad(dr.sum(), real_r1, create_graph=True)[0]
                loss_d = loss_d + 0.5 * args.r1 * grad.pow(2).flatten(1).sum(1).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---- Generator ----
            z = torch.randn(real.shape[0], args.z_dim, device=dev)
            fake, comps, trace = G(z, return_trace=True, labels=y_real)
            d_out = D(fake)
            adv = (-d_out).mean() if args.loss == "hinge" else \
                F.binary_cross_entropy_with_logits(d_out, torch.ones_like(d_out))
            pde = G.pde_residual(trace, mode=args.pde_mode, last_n=args.pde_last_n, labels=y_real)
            tmp = G.temporal_penalty(comps, (masks[0], masks[1], masks[2]))
            l_align = align_loss(fake, r_idx=cfg.r_offset)
            l_stat = (fake.std() - real_std).abs()
            l_spec = (hf_ratio(fake) - hf_ratio(real)).abs()
            l_kurt = (kurtosis_t(fake) - kurtosis_t(real)).abs()
            loss_g = (adv + args.lambda_pde * pde + args.lambda_temporal * tmp
                      + args.lambda_stat * l_stat + args.lambda_spec * l_spec
                      + args.lambda_kurt * l_kurt + args.lambda_align * l_align)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_g.step()

            agg["d"] += loss_d.item(); agg["g"] += loss_g.item()
            agg["adv"] += adv.item(); agg["pde"] += pde.item(); agg["tmp"] += tmp.item()
            agg["stat"] = agg.get("stat", 0.0) + l_stat.item()
            agg["spec"] = agg.get("spec", 0.0) + l_spec.item()
            agg["kurt"] = agg.get("kurt", 0.0) + l_kurt.item()
            agg["align"] = agg.get("align", 0.0) + l_align.item()

        # ---- validation: discriminator loss + PDE residual ----
        G.eval()
        with torch.no_grad():
            vb = torch.from_numpy(beats[va[:256]].astype(np.float32)).to(dev)
            vy = torch.from_numpy(Y_all[va[:256]]).to(dev)
            vz = torch.randn(vb.shape[0], args.z_dim, device=dev)
            vf, vc, vt = G(vz, return_trace=True, labels=vy)
            vd = D(vf)
            v_adv = (-vd).mean().item() if args.loss == "hinge" else \
                F.binary_cross_entropy_with_logits(vd, torch.ones_like(vd)).item()
            v_pde = G.pde_residual(vt, mode=args.pde_mode, last_n=args.pde_last_n, labels=vy).item()
            v_rec = F.mse_loss(vf, vb).item()      # monitoring only (not a training target)
        msg = (f"ep{ep:03d} d={agg['d']/steps_per_epoch:.3f} g={agg['g']/steps_per_epoch:.3f} "
               f"adv={agg['adv']/steps_per_epoch:.3f} pde={agg['pde']/steps_per_epoch:.5f} "
               f"tmp={agg['tmp']/steps_per_epoch:.3f} stat={agg['stat']/steps_per_epoch:.3f} "
               f"spec={agg['spec']/steps_per_epoch:.3f} kurt={agg['kurt']/steps_per_epoch:.3f} "
               f"align={agg['align']/steps_per_epoch:.3f} | val adv={v_adv:.3f} pde={v_pde:.5f} "
               f"mse={v_rec:.4f} | {time.time()-t0:.1f}s")
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

        torch.save({"gen": G.state_dict(), "disc": D.state_dict(), "cfg": cfg.__dict__,
                    "args": vars(args), "epoch": ep}, out_dir / "physiopde_gan_last.pt")
        # proxy metric for best selection: val adversarial term + PDE residual (lower is better)
        score = v_adv + args.lambda_pde * v_pde
        if score < best:
            best = score
            torch.save({"gen": G.state_dict(), "disc": D.state_dict(), "cfg": cfg.__dict__,
                        "args": vars(args), "epoch": ep, "score": best}, best_path)
    log_f.close()
    print(f"[gan] done, best={best:.4f} -> {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
