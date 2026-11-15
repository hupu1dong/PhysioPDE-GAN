"""Learnable finite-difference operators (PDE-Net style moment constraints) and one PDE evolution block.

Note: u(t) here is the 12-lead voltage vector at a single instant, u(t) in R^12; t advances point by point.
(paper §3: "simulates the system's evolution over K discrete time steps to generate a full
heartbeat sequence"). Spatial derivatives Dx/Dxx are therefore taken along the lead index (12 points),
while the temporal dimension is carried by the Euler step itself.

Moment constraints (paper §3.3):
  first order:  sum_k q[k] = 0,  sum_k k*q[k] = 1
  second order: sum_k q[k] = 0,  sum_k k*q[k] = 0,  sum_k k^2*q[k] = 2

Implementation: q = q0 + P w, with q0 the minimum-norm particular solution and P the null-space projector,
so the constraints hold exactly and remain differentiable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentConstrainedFilter(nn.Module):
    """Learnable first/second-order difference operator along the lead dimension (12 points).

    Args:
        order: 1 -> Dx, 2 -> Dxx
        kernel_size: odd kernel width N
    """

    def __init__(self, order: int, kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1
        self.order = order
        self.k = kernel_size
        self.half = kernel_size // 2

        idx = torch.arange(-self.half, self.half + 1, dtype=torch.float64)
        rows, mom = [], []
        rows.append(torch.ones_like(idx))
        mom.append(torch.tensor(0.0, dtype=torch.float64))
        rows.append(idx)
        mom.append(torch.tensor(1.0 if order == 1 else 0.0, dtype=torch.float64))
        if order == 2:
            rows.append(idx ** 2)
            mom.append(torch.tensor(2.0, dtype=torch.float64))

        A = torch.stack(rows)                      # (m, N)
        m = torch.stack(mom)                       # (m,)
        AtA_inv = torch.linalg.inv(A @ A.t())
        q0 = A.t() @ (AtA_inv @ m)
        P = torch.eye(self.k, dtype=torch.float64) - A.t() @ (AtA_inv @ A)

        self.register_buffer("q0", q0.float())
        self.register_buffer("P", P.float())
        self.w = nn.Parameter(torch.zeros(self.k))  # w=0 gives the classical minimum-norm difference

    def kernel(self) -> torch.Tensor:
        return self.q0 + self.P @ self.w

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, C, L) or (B, L); 1D convolution along the last dimension (leads)."""
        squeeze = u.dim() == 2
        if squeeze:
            u = u.unsqueeze(1)                       # (B,1,12)
        b, c, l = u.shape
        q = self.kernel().view(1, 1, self.k)
        x = F.pad(u, (self.half, self.half), mode="replicate")
        y = F.conv1d(x, q.expand(c, 1, self.k), groups=c)
        return y.squeeze(1) if squeeze else y


class GlobalSpatialOp(nn.Module):
    """Global spatial operator: replaces the 5-point Dxx convolution with a learnable 12x12 dense matrix.

    Initialised as the 5-point Dxx kernel under circular boundaries, so training starts from the original model.
    """

    def __init__(self, n_leads: int = 12, kernel_size: int = 5):
        super().__init__()
        q = MomentConstrainedFilter(2, kernel_size).kernel().detach()
        half = kernel_size // 2
        W = torch.zeros(n_leads, n_leads)
        for i in range(n_leads):
            for j in range(kernel_size):
                W[i, (i + j - half) % n_leads] = q[j]
        self.W = nn.Parameter(W)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u @ self.W          # (B,12) @ (12,12) -> (B,12)


class LeadAttention(nn.Module):
    """Multi-head self-attention over leads: the 12 leads are 12 tokens, learning which lead affects which."""

    def __init__(self, d: int = 8, heads: int = 2):
        super().__init__()
        self.proj_in = nn.Linear(1, d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj_out = nn.Linear(d, 1)
        self.heads, self.d = heads, d

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(u.unsqueeze(-1))                  # (B,12,d)
        b, n, d = h.shape
        hd = d // self.heads
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, hd).transpose(1, 2)
        k = k.view(b, n, self.heads, hd).transpose(1, 2)
        v = v.view(b, n, self.heads, hd).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / (hd ** 0.5), dim=-1)
        o = (a @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj_out(h + o).squeeze(-1)             # (B,12)


class PDEBlock(nn.Module):
    """One reaction-diffusion step for a single subsystem (SA / AV / HP).

    u(t+dt) = u(t) + dt * F(u, Dx u, Dxx u; S(u_parent))
    Here u is in R^12 and F is a two-hidden-layer ReLU MLP.
    """

    def __init__(self, n_leads: int = 12, hidden: int = 32, kernel_size: int = 5,
                 dt: float = 1.0, phase_dim: int = 4, label_dim: int = 0,
                 spatial: str = "conv", coupling: bool = True):
        super().__init__()
        self.n_leads = n_leads
        self.phase_dim = phase_dim
        self.label_dim = label_dim
        self.spatial = spatial
        self.coupling = coupling
        self.dx = MomentConstrainedFilter(1, kernel_size)
        self.dxx = MomentConstrainedFilter(2, kernel_size)
        if spatial in ("global", "both"):
            self.global_op = GlobalSpatialOp(n_leads, kernel_size)
        if spatial in ("attn", "both"):
            self.lead_attn = LeadAttention()
        self.stim = nn.Sequential(
            nn.Linear(2 * n_leads, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_leads),
        )
        self.F = nn.Sequential(
            nn.Linear(4 * n_leads + phase_dim + label_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_leads),
        )
        self.dt = dt

    def reaction(self, u: torch.Tensor, u_parent: torch.Tensor | None,
                 phase: torch.Tensor | None = None,
                 label: torch.Tensor | None = None) -> torch.Tensor:
        """phase: (B, phase_dim) periodic phase; label: (B, label_dim) class condition (cGAN)."""
        ux = self.dx(u)
        if self.spatial in ("attn", "both"):
            ux = ux + self.lead_attn(u)        # inter-lead relations as an extra first-order channel
        uxx = self.global_op(u) if self.spatial in ("global", "both") else self.dxx(u)
        if u_parent is None or not self.coupling:
            s = torch.zeros_like(u)
        else:
            s = self.stim(torch.cat([u_parent, self.dx(u_parent)], dim=-1))
        feats = [u, ux, uxx, s]
        if phase is not None:
            feats.append(phase)
        if label is not None:
            feats.append(label)
        return self.F(torch.cat(feats, dim=-1))             # (B,12)

    def forward(self, u: torch.Tensor, u_parent: torch.Tensor | None = None,
                phase: torch.Tensor | None = None, label: torch.Tensor | None = None):
        f = self.reaction(u, u_parent, phase, label)
        return u + self.dt * f, f
