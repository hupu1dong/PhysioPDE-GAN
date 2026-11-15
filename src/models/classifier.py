"""Downstream classifier: ResNet1d (Ribeiro et al. 2020) and ResNet1d with multi-head self-attention (Nejedly et al. 2021).

Paper §4.2 gives only a high-level description: 1 conv + 5 residual blocks, each with 3 conv + BN + ReLU + dropout + skip.
This implementation follows the 1D adaptation of Ribeiro et al. 2020.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 16, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv3 = nn.Conv1d(out_ch, out_ch, kernel, 1, pad, bias=False)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = torch.relu(self.bn2(self.conv2(h)))
        h = self.drop(h)
        h = self.bn3(self.conv3(h))
        res = self.shortcut(x)
        # align a possible +/-1 length mismatch from conv padding/stride
        if h.shape[-1] != res.shape[-1]:
            m = min(h.shape[-1], res.shape[-1])
            h, res = h[..., :m], res[..., :m]
        return torch.relu(h + res)


class ResNet1d(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 8, width: int = 64,
                 n_blocks: int = 5, dropout: float = 0.2, attention: bool = False,
                 n_heads: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, width, 16, 2, 8, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True),
        )
        blocks = []
        ch = width
        for i in range(n_blocks):
            out = min(width * (2 ** (i // 2)), width * 4)
            stride = 2 if i < 2 else 1
            blocks.append(ResidualBlock(ch, out, stride=stride, dropout=dropout))
            ch = out
        self.blocks = nn.Sequential(*blocks)
        self.attention = None
        if attention:
            self.attention = nn.MultiheadAttention(ch, n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(ch)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))            # (B,C,T)
        if self.attention is not None:
            s = h.transpose(1, 2)                # (B,T,C)
            a, _ = self.attention(s, s, s, need_weights=False)
            h = self.norm(s + a).transpose(1, 2)
        h = h.mean(dim=-1)
        return self.head(h)
