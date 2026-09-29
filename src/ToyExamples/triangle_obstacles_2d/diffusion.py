"""DDPM scheduler (cosine variance schedule) and helpers.

Direct port of ``src/maze_fkc/diffusion.py``.
"""
from __future__ import annotations

import math
import torch


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps) / num_steps
    f = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-6, 0.999)


def linear_beta_schedule(num_steps: int,
                         beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


class DDPM:
    """Variance-preserving diffusion in discrete time."""

    def __init__(self, num_steps: int = 100, schedule: str = "cosine"):
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_steps)
        elif schedule == "linear":
            betas = linear_beta_schedule(num_steps)
        else:
            raise ValueError(schedule)
        self.num_steps = num_steps
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = torch.sqrt(self.alpha_bar)
        self.sqrt_1m_ab = torch.sqrt(1.0 - self.alpha_bar)
        self.alpha_bar_prev = torch.cat(
            [torch.tensor([1.0]), self.alpha_bar[:-1]], dim=0
        )
        self.posterior_var = (
            betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )

    def to(self, device):
        for n in ["betas", "alphas", "alpha_bar", "sqrt_ab", "sqrt_1m_ab",
                  "alpha_bar_prev", "posterior_var"]:
            setattr(self, n, getattr(self, n).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, *([1] * (x0.ndim - 1)))
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor,
                            eps: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, *([1] * (x_t.ndim - 1)))
        return (x_t - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)

    def posterior_mean(self, x0: torch.Tensor, x_t: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, *([1] * (x_t.ndim - 1)))
        ab_prev = self.alpha_bar_prev[t].view(-1, *([1] * (x_t.ndim - 1)))
        beta = self.betas[t].view(-1, *([1] * (x_t.ndim - 1)))
        a = self.alphas[t].view(-1, *([1] * (x_t.ndim - 1)))
        coef_x0 = beta * torch.sqrt(ab_prev) / (1.0 - ab)
        coef_xt = (1.0 - ab_prev) * torch.sqrt(a) / (1.0 - ab)
        return coef_x0 * x0 + coef_xt * x_t

    def posterior_std(self, t: torch.Tensor, x_shape) -> torch.Tensor:
        return torch.sqrt(self.posterior_var[t]).view(
            -1, *([1] * (len(x_shape) - 1))
        )
