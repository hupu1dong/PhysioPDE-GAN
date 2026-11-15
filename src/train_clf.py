"""Train the downstream 12-lead classifier (paper §4.2 / §4.3).

Three settings (paper §4.3):
    real          real data only
    real+pde      real + PhysioPDE-GAN synthetic
    real+baseline real + other generators (passed via --syn)

Output (metrics/probs_<tag>.npz):
  y_true (N,8), y_prob (N,8), rec_ids, fold
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.classifier import ResNet1d  # noqa: E402


def auc_safe(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(ROOT / "data" / "processed" / "ptbxl_beats.npz"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--syn", default="", help="synthetic beats npz (output of generate.py)")
    ap.add_argument("--syn-ratio", type=float, default=1.0, help="synthetic samples / real training samples")
    ap.add_argument("--arch", choices=["resnet", "attn"], default="resnet")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-frac", type=float, default=1.0,
                    help="use only a random subset of real training data (data-scale ablation, strategy C)")
    ap.add_argument("--syn-classes", default="", help="keep only synthetic samples of these classes (comma separated, strategy B)")
    ap.add_argument("--syn-for-pretrain", action="store_true",
                    help="synthetic data used for pretraining only, not mixed into the main training set")
    ap.add_argument("--pretrain-epochs", type=int, default=0,
                    help="strategy D: pretrain on synthetic data for N epochs, then finetune on the main data")
    ap.add_argument("--tag", default="real")
    ap.add_argument("--out-dir", default=str(ROOT / "metrics"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    syn_beats = syn_labels = None
    d = np.load(args.npz)
    X = d["beats"]
    Y = d["labels"].astype(np.float32)
    fold = d["fold"]
    split = d["split"] if "split" in d else np.where(fold < 0, 1, 0)
    rec_ids = d["rec_ids"]

    tr = (split == 0) & (fold != args.fold)
    va = (split == 0) & (fold == args.fold)
    te = split == 1
    print(f"[clf] real train={tr.sum()} val={va.sum()} test={te.sum()}", flush=True)

    Xtr, Ytr = X[tr].astype(np.float32), Y[tr]
    if args.real_frac < 1.0:
        n_keep = max(int(len(Xtr) * args.real_frac), 64)
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(len(Xtr), size=n_keep, replace=False)
        Xtr, Ytr = Xtr[sel], Ytr[sel]
        print(f"[clf] real-frac={args.real_frac} -> train={n_keep}", flush=True)
    if args.syn:
        sd = np.load(args.syn)
        syn_beats = sd["beats"]
        syn_labels = sd["labels"]
        if args.syn_classes:
            names = [c.strip() for c in args.syn_classes.split(",") if c.strip()]
            cls_list = [str(c) for c in sd["classes"]]
            cols = [cls_list.index(n) for n in names]
            mask = syn_labels[:, cols].sum(1) > 0
            syn_beats, syn_labels = syn_beats[mask], syn_labels[mask]
            print(f"[clf] syn-classes={names} -> {mask.sum()} samples", flush=True)
        n_syn = int(len(Xtr) * args.syn_ratio)
        take = np.random.choice(len(syn_beats), size=min(n_syn, len(syn_beats)), replace=len(syn_beats) < n_syn)
        Xs = syn_beats[take].astype(np.float32)
        Ys = syn_labels[take].astype(np.float32)
        if not args.syn_for_pretrain:
            Xtr = np.concatenate([Xtr, Xs])
            Ytr = np.concatenate([Ytr, Ys])
        else:
            print("[clf] synthetic used for pretraining only, not mixed into training", flush=True)
        print(f"[clf] + synthetic {Xs.shape} -> total {Xtr.shape[0]}", flush=True)

    pos = Ytr.sum(0)
    neg = len(Ytr) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 20.0), dtype=torch.float32, device=dev)

    model = ResNet1d(n_leads=12, n_classes=Y.shape[1], width=args.width,
                     attention=(args.arch == "attn")).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[clf] arch={args.arch} params={n_param/1e6:.2f}M", flush=True)

    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xva = torch.from_numpy(X[va].astype(np.float32)).to(dev)
    Yva = Y[va]
    Xte = torch.from_numpy(X[te].astype(np.float32)).to(dev)

    log_path = ROOT / "logs" / f"train_clf_{args.tag}_f{args.fold}_{args.arch}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a")
    log_f.write(json.dumps(vars(args)) + "\n")

    # ---------- strategy D: synthetic-only pretraining ----------
    if args.pretrain_epochs > 0 and syn_beats is not None:
        print(f"[clf] pretraining: synthetic only, {len(syn_beats)} samples x {args.pretrain_epochs} ep", flush=True)
        ds_pre = torch.utils.data.TensorDataset(
            torch.from_numpy(syn_beats.astype(np.float32)),
            torch.from_numpy(syn_labels.astype(np.float32)))
        dl_pre = torch.utils.data.DataLoader(ds_pre, batch_size=args.batch, shuffle=True,
                                             num_workers=2, pin_memory=True, drop_last=True)
        opt_pre = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        for ep in range(1, args.pretrain_epochs + 1):
            model.train()
            tot = 0.0
            for xb, yb in dl_pre:
                xb, yb = xb.to(dev), yb.to(dev)
                loss = lossf(model(xb), yb)
                opt_pre.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt_pre.step()
                tot += loss.item()
            print(f"[pre] ep{ep:02d} loss={tot/max(len(dl_pre),1):.4f}", flush=True)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb)
            loss = lossf(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva)).cpu().numpy()
        aucs = [auc_safe(Yva[:, i], pv[:, i]) for i in range(Yva.shape[1])]
        m_auc = float(np.nanmean(aucs))
        msg = (f"ep{ep:03d} loss={tot/max(len(dl),1):.4f} val_auc={m_auc:.4f} "
               f"({time.time()-t0:.1f}s) " + " ".join(f"{a:.3f}" for a in aucs))
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()
        if m_auc > best_auc:
            best_auc, bad = m_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                print("[clf] early stop", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = torch.sigmoid(model(Xte)).cpu().numpy()
        pva = torch.sigmoid(model(Xva)).cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"probs_{args.tag}_f{args.fold}_{args.arch}.npz"
    np.savez_compressed(out, y_true=Y[te], y_prob=pte, y_true_val=Yva, y_prob_val=pva,
                        rec_ids=rec_ids[te], classes=d["classes"], best_auc=best_auc)
    print(f"[clf] saved {out} best_val_auc={best_auc:.4f}", flush=True)
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
