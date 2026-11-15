"""PhysioPDE-GAN generator (a PDE numerical solver) and discriminator.

Generator (paper §3.2 / §3.3):
    u_SA(0) = FC(z) ∈ R^12,   u_AV(0) = u_HP(0) = 0
    for t = 0..T-1:
        u_SA(t+1) = u_SA(t) + Δt · F_SA(u_SA, Dx u_SA, Dxx u_SA)
        u_AV(t+1) = u_AV(t) + Δt · F_AV(..., S(u_SA(t)))     # AV node triggered by SA
        u_HP(t+1) = u_HP(t) + Δt · F_HP(..., S(u_AV(t)))     # His-Purkinje triggered by AV
        u(t+1)    = u_SA(t+1) + u_AV(t+1) + u_HP(t+1)        # linear superposition (Eq. 4)
    → u ∈ R^{12×T}

The same PDE parameters are integrated repeatedly over time (numerical-solver semantics); Dx/Dxx act along the 12 leads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from .pde import PDEBlock


@dataclass
class GenConfig:
    n_leads: int = 12
    seq_len: int = 400
    z_dim: int = 128
    hidden: int = 32
    kernel_size: int = 5
    dt: float = 1.0
    trace_steps: int = 32       # keep intermediates for the last N steps only (needed by the PDE residual)
    n_blocks: int = 8           # piecewise PDE parameters along time (temporal programme of cardiac conduction)
    n_harmonics: int = 16       # number of phase harmonics (must reach 10-40 Hz to represent QRS)
    r_offset: int = 140         # index of the R peak in the window (phase zero point)
    n_classes: int = 8          # number of classes for conditional generation (0 = unconditional)
    spatial: str = "conv"       # conv | global | attn | both (inter-lead spatial modelling)
    coupling: bool = True       # False = disable SA->AV->HP stimulus coupling (for ablation)


class PhysioPDEGANGenerator(nn.Module):
    def __init__(self, cfg: GenConfig):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.z_dim + cfg.n_classes, cfg.n_leads)
        self.fc_scale = nn.Parameter(torch.tensor(0.05))
        # Lead identity: breaks the permutation symmetry of the 12 leads
        self.lead_init = nn.Parameter(torch.zeros(1, cfg.n_leads))
        mk = lambda: nn.ModuleList([                                  # noqa: E731
            PDEBlock(cfg.n_leads, cfg.hidden, cfg.kernel_size, cfg.dt,
                     phase_dim=2 * cfg.n_harmonics, label_dim=cfg.n_classes,
                     spatial=cfg.spatial, coupling=cfg.coupling)
            for _ in range(cfg.n_blocks)])
        self.sa, self.av, self.hp = mk(), mk(), mk()
        # reaction gain with small initialisation: suppresses divergence from 400 accumulated steps
        self.gain = nn.Parameter(torch.full((1, cfg.n_leads), 0.15))
        # precomputed phase table (T, 2H) to avoid building small tensors every step
        import math
        T, H = cfg.seq_len, cfg.n_harmonics
        tab = torch.zeros(T, 2 * H)
        for t in range(T):
            w0 = 2.0 * math.pi * (t - cfg.r_offset) / T
            for k in range(1, H + 1):
                tab[t, 2 * (k - 1)] = math.sin(k * w0)
                tab[t, 2 * (k - 1) + 1] = math.cos(k * w0)
        self.register_buffer("phase_table", tab)
        self.u_clip = 5.0

    def _phase(self, t: int, b: int, device) -> torch.Tensor:
        """Periodic phase features (lookup from the precomputed table)."""
        return self.phase_table[t].to(device).unsqueeze(0).expand(b, -1)

    def _step(self, blk: PDEBlock, u: torch.Tensor, parent: torch.Tensor | None,
              phase: torch.Tensor | None = None, label: torch.Tensor | None = None):
        f = blk.reaction(u, parent, phase, label) * self.gain
        u1 = torch.clamp(u + self.cfg.dt * f, -self.u_clip, self.u_clip)
        return u1, f

    def forward(self, z: torch.Tensor, return_trace: bool = False,
                labels: torch.Tensor | None = None):
        b = z.shape[0]
        c = self.cfg
        if c.n_classes > 0:
            if labels is None:
                labels = torch.zeros(b, c.n_classes, device=z.device)
            z = torch.cat([z, labels], dim=-1)
        u_sa = self.fc(z) * self.fc_scale + self.lead_init
        u_av = torch.zeros_like(u_sa)
        u_hp = torch.zeros_like(u_sa)

        seq_sa, seq_av, seq_hp = [], [], []
        trace: List[dict] = []
        keep = max(c.trace_steps, 1)
        for t in range(c.seq_len):
            ph = self._phase(t, b, u_sa.device)
            bi = min(t * c.n_blocks // c.seq_len, c.n_blocks - 1)
            sa_next, f_sa = self._step(self.sa[bi], u_sa, None, ph, labels)
            av_next, f_av = self._step(self.av[bi], u_av, u_sa, ph, labels)
            hp_next, f_hp = self._step(self.hp[bi], u_hp, u_av, ph, labels)
            if return_trace and t >= c.seq_len - keep:
                trace.append({
                    "sa": (u_sa, sa_next, f_sa, None, ph, bi),
                    "av": (u_av, av_next, f_av, u_sa, ph, bi),
                    "hp": (u_hp, hp_next, f_hp, u_av, ph, bi),
                })
            seq_sa.append(sa_next)
            seq_av.append(av_next)
            seq_hp.append(hp_next)
            u_sa, u_av, u_hp = sa_next, av_next, hp_next

        u = torch.stack([a + b_ + c_ for a, b_, c_ in zip(seq_sa, seq_av, seq_hp)], dim=-1)
        comps = (torch.stack(seq_sa, -1), torch.stack(seq_av, -1), torch.stack(seq_hp, -1))
        if return_trace:
            return u, comps, trace
        return u, comps

    def pde_residual(self, trace: List[dict], mode: str = "cn", last_n: int = 8,
                     labels: torch.Tensor | None = None) -> torch.Tensor:
        """PDE residual (paper Eq. 6).

        Assumption A1: the Euler update makes the residual identically zero, so a Crank-Nicolson midpoint residual is used,
        normalised by mean(F(u0)^2) so the term carries a non-trivial gradient.
        """
        blocks = {"sa": self.sa, "av": self.av, "hp": self.hp}
        total = 0.0
        idxs = list(range(max(len(trace) - max(last_n, 1), 0), len(trace)))
        for i in idxs:
            for key, (u0, u1, f0, parent, ph, bi) in trace[i].items():
                if mode == "euler":
                    res = (u1 - u0) / self.cfg.dt - f0
                else:
                    f1 = blocks[key][bi].reaction(u1, parent, ph, labels) * self.gain
                    res = (u1 - u0) / self.cfg.dt - 0.5 * (f0 + f1)
                scale = f0.pow(2).mean().detach() + 1e-8
                total = total + res.pow(2).mean() / scale
        return total / max(len(idxs), 1)

    def temporal_penalty(self, comps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                         masks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """L_temporal (paper Eq. 8): penalises component energy outside its physiological window. masks: (3,T)."""
        loss = 0.0
        for u, m in zip(comps, masks):
            loss = loss + (u * (1.0 - m)).pow(2).mean()
        return loss / 3.0


class ECGCritic(nn.Module):
    """1D-CNN discriminator (the paper omits details; layer count and channels are chosen here)."""

    def __init__(self, n_leads: int = 12, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, base, 7, 2, 3), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base, base * 2, 7, 2, 3), nn.BatchNorm1d(base * 2), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base * 2, base * 4, 5, 2, 2), nn.BatchNorm1d(base * 4), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base * 4, base * 4, 3, 2, 1), nn.BatchNorm1d(base * 4), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Linear(base * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h.mean(dim=-1)
        return self.head(h).squeeze(1)
