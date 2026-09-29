"""Inference-time samplers: vanilla / linear_combo / fkc.

Port of ``src/maze_fkc/samplers.py`` adapted for this toy:

  * The denoiser is unconditional — ``eps = model(x, t)`` (no maze image, no
    start/end FiLM).
  * Only the *start* point is inpainted at every reverse step. The goal is a
    *line* segment; we let the model discover its endpoint distribution.
  * The cost is supplied by ``cost.constraint_cost`` over the env's
    inference-time constraints (cones + circles + bottom-row halos).

The FKC math (cost-tilted target marginals via Feynman-Kac PDE Eq. 23 of
``docs/Feynman_Kac_PDEs_for_Constrained_and_Cost_Guided_Diffusion.pdf``) is
unchanged from the maze_fkc reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .cost import constraint_cost, constraint_cost_and_grad
from .diffusion import DDPM
from .env import TriangleObstacleEnv


@dataclass
class FKCConfig:
    guidance_weight: float = 200.0
    beta_anneal: bool = True
    beta_pow: float = 3.0
    num_particles: int = 32
    resample_ess_frac: float = 0.5
    weight_mode: str = "girsanov"      # "energy" or "girsanov"
    active_window: tuple[float, float] = (0.3, 0.95)
    clearance: float = 0.0
    return_best: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inpaint_start(traj: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """Pin only the first waypoint (start). Goal is a line, not a point."""
    out = traj.clone()
    out[..., 0, :] = start
    return out


def _systematic_resample(log_w: torch.Tensor) -> torch.Tensor:
    K = log_w.numel()
    w = torch.softmax(log_w, dim=0)
    cdf = torch.cumsum(w, dim=0)
    u0 = torch.rand((), device=log_w.device) / K
    points = u0 + torch.arange(K, device=log_w.device) / K
    idx = torch.searchsorted(cdf, points)
    return idx.clamp(max=K - 1)


def _ess(log_w: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(log_w, dim=0)
    return 1.0 / (w * w).sum()


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_vanilla(model, ddpm: DDPM, *, traj_len: int,
                   env: TriangleObstacleEnv,
                   num_samples: int = 1,
                   device: torch.device | str = "cuda") -> torch.Tensor:
    """Plain DDPM ancestral sampling, no guidance. Returns (num_samples, T, 2)."""
    B = num_samples
    start = torch.tensor(env.start, device=device, dtype=torch.float32)
    start_b = start.expand(B, 2)

    x = torch.randn(B, traj_len, 2, device=device)
    x = _inpaint_start(x, start_b)
    for t_idx in reversed(range(ddpm.num_steps)):
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)
        eps = model(x, t)
        x0_pred = ddpm.predict_x0_from_eps(x, t, eps)
        x0_pred = _inpaint_start(x0_pred, start_b)
        mean = ddpm.posterior_mean(x0_pred, x, t)
        if t_idx > 0:
            std = ddpm.posterior_std(t, x.shape)
            x = mean + std * torch.randn_like(x)
        else:
            x = mean
        x = _inpaint_start(x, start_b)
    return x


def sample_linear_combo(model, ddpm: DDPM, *, traj_len: int,
                        env: TriangleObstacleEnv,
                        guidance_weight: float = 200.0,
                        beta_anneal: bool = True, beta_pow: float = 3.0,
                        clearance: float = 0.0,
                        num_samples: int = 1,
                        device: torch.device | str = "cuda") -> torch.Tensor:
    """Classifier-style guidance: eps_g = eps + sqrt(1 - alpha_bar) * w * grad J."""
    B = num_samples
    start = torch.tensor(env.start, device=device, dtype=torch.float32)
    start_b = start.expand(B, 2)

    x = torch.randn(B, traj_len, 2, device=device)
    x = _inpaint_start(x, start_b)

    for t_idx in reversed(range(ddpm.num_steps)):
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, t)

        x0_pred = ddpm.predict_x0_from_eps(x, t, eps)
        x0_pred = _inpaint_start(x0_pred, start_b)
        _, gJ = constraint_cost_and_grad(x0_pred, env, clearance=clearance)
        # Normalize per-sample so guidance_weight is a strength knob.
        gJ = gJ / gJ.norm(p=2, dim=(-2, -1), keepdim=True).clamp(min=1e-8)

        if beta_anneal:
            frac = (ddpm.num_steps - 1 - t_idx) / max(ddpm.num_steps - 1, 1)
            w_t = guidance_weight * (frac ** beta_pow)
        else:
            w_t = guidance_weight

        sqrt_1m_ab = ddpm.sqrt_1m_ab[t].view(-1, 1, 1)
        eps_guided = eps + sqrt_1m_ab * w_t * gJ

        with torch.no_grad():
            x0_pred = ddpm.predict_x0_from_eps(x, t, eps_guided)
            x0_pred = _inpaint_start(x0_pred, start_b)
            mean = ddpm.posterior_mean(x0_pred, x, t)
            if t_idx > 0:
                std = ddpm.posterior_std(t, x.shape)
                x = mean + std * torch.randn_like(x)
            else:
                x = mean
            x = _inpaint_start(x, start_b)
    return x


def sample_fkc(model, ddpm: DDPM, *, traj_len: int,
               env: TriangleObstacleEnv, cfg: FKCConfig,
               device: torch.device | str = "cuda"
               ) -> tuple[torch.Tensor, dict]:
    """Feynman-Kac corrector sampling with K particles + ESS resampling."""
    K = cfg.num_particles
    start = torch.tensor(env.start, device=device, dtype=torch.float32)
    start_b = start.expand(K, 2)

    x = torch.randn(K, traj_len, 2, device=device)
    x = _inpaint_start(x, start_b)
    log_w = torch.zeros(K, device=device)

    with torch.no_grad():
        prev_J = constraint_cost(x, env, clearance=cfg.clearance)

    info = {"resample_steps": [], "ess_log": [], "weight_log": []}

    for t_idx in reversed(range(ddpm.num_steps)):
        t = torch.full((K,), t_idx, device=device, dtype=torch.long)
        with torch.no_grad():
            eps = model(x, t)

        x0_pred = ddpm.predict_x0_from_eps(x, t, eps)
        x0_pred = _inpaint_start(x0_pred, start_b)
        J_at_x0, gJ = constraint_cost_and_grad(x0_pred, env,
                                                clearance=cfg.clearance)
        gJ = gJ / gJ.norm(p=2, dim=(-2, -1), keepdim=True).clamp(min=1e-8)

        if cfg.beta_anneal:
            frac = (ddpm.num_steps - 1 - t_idx) / max(ddpm.num_steps - 1, 1)
            beta_t = cfg.guidance_weight * (frac ** cfg.beta_pow)
            beta_prev = cfg.guidance_weight * (
                ((ddpm.num_steps - t_idx) / max(ddpm.num_steps - 1, 1)) ** cfg.beta_pow
            )
        else:
            beta_t = cfg.guidance_weight
            beta_prev = cfg.guidance_weight

        sqrt_1m_ab = ddpm.sqrt_1m_ab[t].view(-1, 1, 1)
        eps_guided = eps + sqrt_1m_ab * beta_t * gJ

        with torch.no_grad():
            x0_g = ddpm.predict_x0_from_eps(x, t, eps_guided)
            x0_g = _inpaint_start(x0_g, start_b)
            mean = ddpm.posterior_mean(x0_g, x, t)
            if t_idx > 0:
                std = ddpm.posterior_std(t, x.shape)
                x_next = mean + std * torch.randn_like(x)
            else:
                x_next = mean
            x_next = _inpaint_start(x_next, start_b)
            J_next = constraint_cost(x_next, env, clearance=cfg.clearance)

        # ----- log-weight increment -----
        if cfg.weight_mode == "energy":
            dlogw = -(beta_t * J_next - beta_prev * prev_J)
        elif cfg.weight_mode == "girsanov":
            with torch.no_grad():
                ab = ddpm.alpha_bar[t].view(-1, 1, 1)
                a = ddpm.alphas[t].view(-1, 1, 1)
                score = -eps / torch.sqrt(1.0 - ab)
                f_dt = -(1.0 - torch.sqrt(a)) * x
                drift_diff = 0.5 * (1.0 - a) * score - f_dt
                inner = (gJ * drift_diff).flatten(1).sum(dim=-1)
                dlogw_anneal = -(beta_prev - beta_t) * prev_J
                dlogw_drift = -beta_t * inner
                dlogw = dlogw_anneal + dlogw_drift
        else:
            raise ValueError(cfg.weight_mode)

        log_w = log_w + dlogw
        prev_J = J_next
        x = x_next

        info["weight_log"].append(log_w.detach().cpu().clone())

        progress = (ddpm.num_steps - t_idx) / ddpm.num_steps
        in_window = cfg.active_window[0] <= progress <= cfg.active_window[1]
        ess_val = _ess(log_w).item()
        info["ess_log"].append(ess_val)
        if in_window and ess_val < cfg.resample_ess_frac * K and t_idx > 0:
            anc = _systematic_resample(log_w)
            x = x[anc]
            prev_J = prev_J[anc]
            log_w = torch.zeros_like(log_w)
            info["resample_steps"].append(t_idx)

    if cfg.return_best:
        with torch.no_grad():
            J_final = constraint_cost(x, env, clearance=cfg.clearance)
            order = torch.argsort(J_final)
            return x[order], {**info, "final_J": J_final.detach().cpu()}
    return x, info
