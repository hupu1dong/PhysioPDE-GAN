"""Classical generative baselines (1D adaptations of DCGAN and WaveGAN).

ME-GAN / MultiODE-GAN / SSSD-ECG require third-party implementations and are not included here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DCGANGenerator1d(nn.Module):
    """z -> Linear -> (256,25) -> 4x ConvTranspose1d -> (12,400), tanh output."""

    def __init__(self, z_dim: int = 128, n_leads: int = 12, seq_len: int = 400,
                 base: int = 256, kernel: int = 4):
        super().__init__()
        assert seq_len == 400
        self.seq_len = seq_len
        self.fc = nn.Linear(z_dim, base * 25)
        self.base = base
        self.net = nn.Sequential(
            nn.BatchNorm1d(base), nn.ReLU(True),
            nn.ConvTranspose1d(base, base // 2, kernel, 2, 1), nn.BatchNorm1d(base // 2), nn.ReLU(True),
            nn.ConvTranspose1d(base // 2, base // 4, kernel, 2, 1), nn.BatchNorm1d(base // 4), nn.ReLU(True),
            nn.ConvTranspose1d(base // 4, base // 8, kernel, 2, 1), nn.BatchNorm1d(base // 8), nn.ReLU(True),
            nn.ConvTranspose1d(base // 8, n_leads, kernel, 2, 1), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.base, 25)
        return self.net(h)


class WaveGANGenerator1d(nn.Module):
    """WaveGAN style: large kernel (25) with the same upsampling backbone."""

    def __init__(self, z_dim: int = 128, n_leads: int = 12, seq_len: int = 400, base: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(z_dim, base * 25)
        self.base = base
        k = 25
        self.net = nn.Sequential(
            nn.BatchNorm1d(base), nn.ReLU(True),
            nn.ConvTranspose1d(base, base // 2, k, 2, k // 4), nn.BatchNorm1d(base // 2), nn.ReLU(True),
            nn.ConvTranspose1d(base // 2, base // 4, k, 2, k // 4), nn.BatchNorm1d(base // 4), nn.ReLU(True),
            nn.ConvTranspose1d(base // 4, base // 8, k, 2, k // 4), nn.BatchNorm1d(base // 8), nn.ReLU(True),
            nn.ConvTranspose1d(base // 8, n_leads, k, 2, k // 4), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.base, 25)
        out = self.net(h)
        return out[..., : self.seq_len]


class DCGANDiscriminator1d(nn.Module):
    def __init__(self, n_leads: int = 12, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, base, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base, base * 2, 4, 2, 1), nn.BatchNorm1d(base * 2), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base * 2, base * 4, 4, 2, 1), nn.BatchNorm1d(base * 4), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base * 4, base * 8, 4, 2, 1), nn.BatchNorm1d(base * 8), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Linear(base * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h.mean(dim=-1)
        return self.head(h).squeeze(1)
