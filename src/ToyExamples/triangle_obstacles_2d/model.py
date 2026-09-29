"""Unconditional 1D-UNet noise predictor for 2D trajectories.

The trajectory ``(B, T, 2)`` is denoised conditioned only on the diffusion
timestep ``t`` — no maze image, no start/end FiLM (the start is a fixed
constant enforced via inpainting in the sampler, and the goal is a *line*
rather than a point so we don't condition on it).

Building blocks copied from ``src/maze_fkc/model.py``; the top-level
``UNet1D`` is slimmed accordingly.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        ang = t[:, None].float() * freqs[None]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * ch)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gb = self.proj(c)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma[..., None]) + beta[..., None]


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, padding=2)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.film = FiLM(cond_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, c):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, c)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    def __init__(self, ch: int, heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).permute(0, 2, 1)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.permute(0, 2, 1)


class Down1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNet1D(nn.Module):
    """Unconditional symmetric 1D U-Net acting on (B, channels=2, T) traj."""

    def __init__(self,
                 in_ch: int = 2,
                 base_ch: int = 64,
                 channel_mult: tuple[int, ...] = (1, 2, 4),
                 cond_dim: int = 256,
                 t_emb_dim: int = 128):
        super().__init__()
        self.depth = len(channel_mult)

        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(t_emb_dim),
            nn.Linear(t_emb_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        chans = [base_ch * m for m in channel_mult]
        self.in_conv = nn.Conv1d(in_ch, chans[0], 5, padding=2)
        self.downs = nn.ModuleList()
        prev = chans[0]
        for c in chans:
            self.downs.append(nn.ModuleList([
                ResBlock1D(prev, c, cond_dim),
                ResBlock1D(c, c, cond_dim),
                Down1D(c),
            ]))
            prev = c

        self.mid1 = ResBlock1D(prev, prev * 2, cond_dim)
        self.mid_attn = SelfAttention1D(prev * 2,
                                          heads=max(1, min(8, prev * 2 // 64)))
        self.mid2 = ResBlock1D(prev * 2, prev, cond_dim)

        self.ups = nn.ModuleList()
        for c in reversed(chans):
            self.ups.append(nn.ModuleList([
                Up1D(prev),
                ResBlock1D(prev + c, c, cond_dim),
                ResBlock1D(c, c, cond_dim),
            ]))
            prev = c

        self.out_norm = nn.GroupNorm(min(8, chans[0]), chans[0])
        self.out_conv = nn.Conv1d(chans[0], in_ch, 5, padding=2)

    def forward(self, traj_btc: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """traj_btc: (B, T, 2) -> noise pred (B, T, 2)."""
        c = self.t_emb(t)                          # (B, cond_dim)
        x = traj_btc.transpose(1, 2)                # (B, 2, T)
        h = self.in_conv(x)
        skips = []
        for r1, r2, dn in self.downs:
            h = r1(h, c)
            h = r2(h, c)
            skips.append(h)
            h = dn(h)
        h = self.mid1(h, c)
        h = self.mid_attn(h)
        h = self.mid2(h, c)
        for (up, r1, r2), skip in zip(self.ups, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = r1(h, c)
            h = r2(h, c)
        h = F.silu(self.out_norm(h))
        out = self.out_conv(h)                     # (B, 2, T)
        return out.transpose(1, 2)
