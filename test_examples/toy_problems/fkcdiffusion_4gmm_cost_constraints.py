#!/usr/bin/env python3
"""
fkcdiffusion_4gmm_cost_constraints_PAPER_FAITHFUL_constbeta_GROUPED_FKC.py

Self-contained PyTorch demo:
- 2D 4-mode Gaussian mixture "data"
- VP-SDE score model training (denoising score matching)
- Baseline reverse-SDE sampling
- Baseline: linear combo of score and grad J (no FKC weights/resampling)
- Feynman–Kac (FKC) Gibbs-tilted sampling with SMC resampling

IMPORTANT CORRECTION (per user request / paper intent for single-sample metrics):
- To produce K final FKC samples, we run an *inner* batch of M particles per sample,
  reweight/resample within that batch, then pick ONE particle.
- This is repeated K times in parallel by simulating K groups.
- Total simulated diffusions for FKC = K * M.

Constraints (as requested):
  - inequality: y >= 0
  - equality:   x^2 + y^2 = 4  (implemented as (sqrt(x^2+y^2) - 2)^2)

Cost:
  - cost(x) = 1 (constant; gradient is 0)

Guidance:
  - Gibbs tilting with constant beta_guid (beta_final is constant in time).

Usage:
  # Train
  python fkcdiffusion_4gmm_cost_constraints_PAPER_FAITHFUL_constbeta_GROUPED_FKC.py --train --plot

  # Sample + plot (loads ckpt if present)
  python fkcdiffusion_4gmm_cost_constraints_PAPER_FAITHFUL_constbeta_GROUPED_FKC.py --sample --plot
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

R = 3.0

# ---------------------------
# 1) 4-Gaussian mixture "data"
# ---------------------------

@dataclass
class GMM4:
    centers: torch.Tensor  # (4, 2)
    std: float = 0.25

    @staticmethod
    def default(device: torch.device) -> "GMM4":
        centers = torch.tensor(
            [[-2.0, -2.0],
             [-2.0,  2.0],
             [ 2.0, -2.0],
             [ 2.0,  2.0]],
            device=device,
            dtype=torch.float32,
        )
        return GMM4(centers=centers, std=0.25)

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, 4, (n,), device=self.centers.device)
        means = self.centers[idx]
        return means + self.std * torch.randn(n, 2, device=self.centers.device, dtype=torch.float32)


# ---------------------------
# 2) VP-SDE utilities (constant beta(t))
# ---------------------------

@dataclass
class VPSDE:
    """
    VP SDE (noise-time t in [0,1]) with constant beta:
      dx = f(x,t) dt + g(t) dW
      f(x,t) = -0.5 * beta * x
      g(t)   = sqrt(beta)
    """
    beta_const: float = 1.0

    # kept for compatibility with original constructor calls
    beta_min: float = 0.1
    beta_max: float = 20.0

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, self.beta_const)

    def int_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_const * t

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.int_beta(t))

    def drift_f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.beta(t) * x

    def diffusion_g(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta(t))


# ---------------------------
# 3) Score network (tiny MLP)
# ---------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64, max_freq: float = 1000.0):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(max_freq), half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.size(1) == 1:
            t = t[:, 0]
        args = t[:, None] * self.freqs[None, :].to(t.device)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ScoreMLP(nn.Module):
    def __init__(self, hidden: int = 128, time_dim: int = 64):
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time(t)
        h = torch.cat([x, te], dim=-1)
        return self.net(h)


# ---------------------------
# 4) Systematic resampling (batched / group-wise)
# ---------------------------

@torch.no_grad()
def systematic_resample_batched(logw: torch.Tensor) -> torch.Tensor:
    """
    Batched systematic resampling.
    logw: (B, K)
    returns idx: (B, K), each row is resampled indices in [0, K-1]
    """
    B, K = logw.shape
    logw = logw - logw.max(dim=1, keepdim=True).values
    w = torch.softmax(logw, dim=1)

    # numerical guard: if any row becomes invalid, replace with uniform
    bad = (~torch.isfinite(w)).any(dim=1) | (w.sum(dim=1) <= 0)
    if bad.any():
        w[bad] = 1.0 / K

    cdf = torch.cumsum(w, dim=1)
    cdf[:, -1] = 1.0  # ensure last exactly 1

    u0 = torch.rand(B, 1, device=logw.device, dtype=logw.dtype) / K
    u = u0 + (torch.arange(K, device=logw.device, dtype=logw.dtype)[None, :] / K)
    u = torch.clamp(u, 0.0, 1.0 - 1e-7)

    idx = torch.searchsorted(cdf, u)
    return torch.clamp(idx, 0, K - 1).long()


@torch.no_grad()
def sample_one_index_from_logw(logw: torch.Tensor) -> torch.Tensor:
    """
    Draw ONE categorical sample per row from weights proportional to exp(logw).
    Uses Gumbel-max trick for vectorized sampling.
    logw: (B, K)
    returns idx: (B,)
    """
    # stabilize
    logw = logw - logw.max(dim=1, keepdim=True).values
    g = -torch.log(-torch.log(torch.rand_like(logw)))
    return (logw + g).argmax(dim=1)


# ---------------------------
# 5) Cost + constraints => J(x) and ∇J(x)
# ---------------------------

@dataclass
class ConstrainedObjective:
    """
    J(x) = cost_weight * cost(x)
         + c_eq   * (sqrt(x^2+y^2) - 2)^2
         + c_ineq * softplus(-y)^2   (enforces y >= 0)

    cost(x) default is constant 1 (requested).
    Constraint gradients are analytic for stability/speed.
    """
    cost_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    cost_weight: float = 1000.0

    c_eq: float = 50.0
    c_ineq: float = 50.0
    softplus_beta: float = 10.0
    eps_r: float = 1e-8

    def cost(self, x: torch.Tensor) -> torch.Tensor:
        if self.cost_fn is None:
            return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        return self.cost_fn(x)

    def J_and_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Autograd for BOTH constraints and cost (no analytic gradients).
        x_req = x.detach().clone().requires_grad_(True)

        x0 = x_req[:, 0]
        y0 = x_req[:, 1]

        # equality: x^2 + y^2 = 4  <=>  sqrt(x^2+y^2) = 2
        r = torch.sqrt(x0 * x0 + y0 * y0 + self.eps_r)
        h = r - R
        eq_pen = self.c_eq * (h * h)

        # equality removed; now inside-circle inequality
        """r = torch.sqrt(x0 * x0 + y0 * y0 + self.eps_r)
        h = r - R
        v = F.softplus(self.softplus_beta * h) / self.softplus_beta
        eq_pen = self.c_eq * (v * v)"""

        # inequality: y >= 0 via smooth relu(-y)
        a = self.softplus_beta * (-y0 + 1)
        v = F.softplus(a) / self.softplus_beta
        ineq_pen = self.c_ineq * (v * v)

        # cost (constant 1 if cost_fn is None)
        L = self.cost(x_req)

        J_total = 5.0 * (self.cost_weight * L + eq_pen + ineq_pen)
        grad_total = torch.autograd.grad(J_total.sum(), x_req, create_graph=False)[0]

        return J_total.detach(), grad_total.detach()

    def J_only(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0]
        y0 = x[:, 1]

        r = torch.sqrt(x0 * x0 + y0 * y0 + self.eps_r)
        h = r - R
        eq_pen = self.c_eq * (h * h)

        """r = torch.sqrt(x0 * x0 + y0 * y0 + self.eps_r)
        h = r - R
        v = F.softplus(self.softplus_beta * h) / self.softplus_beta
        eq_pen = self.c_eq * (v * v)"""

        a = self.softplus_beta * (-y0)
        v = F.softplus(a) / self.softplus_beta
        ineq_pen = self.c_ineq * (v * v)

        if self.cost_fn is None:
            L = torch.ones_like(eq_pen)
        else:
            L = self.cost_fn(x)

        return self.cost_weight * L + eq_pen + ineq_pen


# ---------------------------
# 6) Training (denoising score matching)
# ---------------------------

def dsm_loss(model: ScoreMLP, sde: VPSDE, x0: torch.Tensor) -> torch.Tensor:
    B = x0.shape[0]
    eps = 1e-4
    t = torch.rand(B, 1, device=x0.device, dtype=x0.dtype) * (1.0 - eps) + eps

    ab = sde.alpha_bar(t)
    std = torch.sqrt(torch.clamp(1.0 - ab, min=1e-6))
    z = torch.randn_like(x0)
    xt = torch.sqrt(ab) * x0 + std * z

    target = -(xt - torch.sqrt(ab) * x0) / (std ** 2)
    pred = model(xt, t)
    return ((pred - target) ** 2).sum(dim=-1).mean()


def train(
    model: ScoreMLP,
    sde: VPSDE,
    gmm: GMM4,
    steps: int,
    batch: int,
    lr: float,
    save_path: str,
):
    model.train()
    opt = Adam(model.parameters(), lr=lr)

    for it in range(1, steps + 1):
        x0 = gmm.sample(batch)
        loss = dsm_loss(model, sde, x0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if it % 1000 == 0:
            print(f"[train] step {it:6d} | loss {loss.item():.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"[train] saved: {save_path}")


# ---------------------------
# 7) Sampling helpers
# ---------------------------

def guidance_schedule(t_noise: torch.Tensor, beta_final: float):
    """
    Constant Gibbs tilting strength beta_guid = beta_final.
    => derivative term is 0.
    """
    beta_guid = torch.full_like(t_noise, beta_final)
    dbeta_ds = torch.zeros_like(t_noise)
    return beta_guid, dbeta_ds


@torch.no_grad()
def sample_baseline(model: ScoreMLP, sde: VPSDE, n: int, steps: int, eps: float) -> torch.Tensor:
    """
    Reverse-SDE Euler-Maruyama with dt<0:
      dx = ( f(x,t) - g(t)^2 * score(x,t) ) dt + g(t) dW̄
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    x = torch.randn(n, 2, device=device, dtype=dtype)
    dt = -(1.0 - eps) / steps

    for k in range(steps):
        tval = 1.0 + k * dt
        t = torch.full((n, 1), tval, device=device, dtype=dtype)

        beta_t = sde.beta(t)[:, 0]
        f_t = sde.drift_f(x, t)
        score = model(x, t)

        drift = f_t - beta_t[:, None] * score
        x = x + drift * dt + torch.sqrt(beta_t)[:, None] * math.sqrt(-dt) * torch.randn_like(x)

    return x.detach()


def sample_linear_score_plus_gradJ(
    model: ScoreMLP,
    sde: VPSDE,
    obj: ConstrainedObjective,
    n: int,
    steps: int,
    eps: float,
    beta_final: float,
    clip_J: float,
    clip_gradJ: float,
) -> torch.Tensor:
    """
    Baseline heuristic (NOT FKC):
      score_tilde = score + (beta_guid/2) * ∇J
      drift = f - beta(t) * score_tilde
    """
    _ = clip_J, clip_gradJ

    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    x = torch.randn(n, 2, device=device, dtype=dtype)
    dt = -(1.0 - eps) / steps

    for k in range(steps):
        tval = 1.0 + k * dt
        t = torch.full((n, 1), tval, device=device, dtype=dtype)

        beta_t = sde.beta(t)[:, 0]
        f_t = sde.drift_f(x, t)

        with torch.no_grad():
            score = model(x, t)

        _, gradJ = obj.J_and_grad(x)

        beta_guid, _ = guidance_schedule(t_noise=t, beta_final=beta_final)
        beta_guid = beta_guid[:, 0]

        score_tilde = score + (beta_guid[:, None] / 2.0) * gradJ
        drift = f_t - beta_t[:, None] * score_tilde

        x = x + 1.0 * (drift * dt + torch.sqrt(beta_t)[:, None] * math.sqrt(-dt) * torch.randn_like(x))

    return x.detach()


def fkc_guided_sample_grouped(
    model: ScoreMLP,
    sde: VPSDE,
    obj: ConstrainedObjective,
    K_out: int,                 # number of final samples returned
    K_inner: int,               # particles simulated per final sample
    steps: int,
    beta_final: float,
    eps: float,
    tmin: float,
    tmax: float,
    resample_every: int,
    center_dw: bool,
    clip_J: float,
    clip_gradJ: float,
    clip_dw: float,
) -> torch.Tensor:
    """
    Grouped FKC:
      - Simulate K_out independent groups, each with K_inner particles.
      - Within each group: reweight/resample (SMC) during the active interval.
      - At the end: pick ONE particle per group according to weights since last resampling.
      - Return K_out picked samples.

    Total simulated diffusions = K_out * K_inner.
    """
    _ = clip_J, clip_gradJ, clip_dw

    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # state: (B, K, 2)
    x = torch.randn(K_out, K_inner, 2, device=device, dtype=dtype)
    logw = torch.zeros(K_out, K_inner, device=device, dtype=dtype)

    dt = -(1.0 - eps) / steps
    ds = -dt  # positive

    for k in range(steps):
        tval = 1.0 + k * dt
        active = (tmin <= tval <= tmax)

        # flatten for model/objective
        x_flat = x.reshape(K_out * K_inner, 2)
        t_flat = torch.full((K_out * K_inner, 1), tval, device=device, dtype=dtype)

        beta_t = sde.beta(t_flat)[:, 0]              # (B*K,)
        f_t = sde.drift_f(x_flat, t_flat)            # (B*K,2)

        with torch.no_grad():
            score = model(x_flat, t_flat)            # (B*K,2)

        J, gradJ = obj.J_and_grad(x_flat)            # (B*K,), (B*K,2)

        beta_guid, dbeta_ds = guidance_schedule(t_noise=t_flat, beta_final=beta_final)
        beta_guid = beta_guid[:, 0]                  # (B*K,)
        dbeta_ds = dbeta_ds[:, 0]                    # zeros

        # ---- weights (paper practice: zero outside active interval) ----
        if not active:
            logw.zero_()
        else:
            term = (beta_t[:, None] / 2.0) * score - f_t
            inner = (gradJ * term).sum(dim=-1)        # (B*K,)
            g = dbeta_ds * J + beta_guid * inner      # (B*K,)
            dw = (g * ds).reshape(K_out, K_inner)     # (B,K)

            if center_dw:
                dw = dw - dw.mean(dim=1, keepdim=True)

            logw = logw + dw

            if (k % max(1, resample_every)) == 0:
                idx = systematic_resample_batched(logw)  # (B,K)
                idx_exp = idx[..., None].expand(-1, -1, 2)
                x = torch.gather(x, dim=1, index=idx_exp)
                logw.zero_()

        # ---- guided drift propagation (reverse-time EM with dt<0) ----
        drift = f_t - beta_t[:, None] * score - (beta_guid[:, None] * beta_t[:, None] / 2.0) * gradJ
        noise = torch.randn_like(x_flat)
        x_flat = x_flat + drift * dt + torch.sqrt(beta_t)[:, None] * math.sqrt(-dt) * noise
        x = x_flat.reshape(K_out, K_inner, 2)

    # Final selection: pick ONE per group according to weights since last resampling.
    # If weights are all zero (e.g. outside active interval), this reduces to uniform selection.
    idx_pick = sample_one_index_from_logw(logw)  # (B,)
    picked = x[torch.arange(K_out, device=device), idx_pick]  # (B,2)
    return picked.detach()


# ---------------------------
# 8) Plotting + stats
# ---------------------------


def draw_constraint_circle(ax, radius: float = 2.0, **kwargs):
    import numpy as np
    th = np.linspace(0.0, np.pi, 512)  # upper semicircle: y >= 0
    x = radius * np.cos(th)
    y = radius * np.sin(th)
    #ax.plot(x, y, **kwargs)
    ax.plot(
        x, y,
        color=kwargs.pop("color", "green"),
        linewidth=kwargs.pop("linewidth", 1.5),
        linestyle=kwargs.pop("linestyle", "--"),
        zorder=kwargs.pop("zorder", 5),
        **kwargs
    )

    #ax.axhline(0.0, color="green", linewidth=1.5, linestyle="--", zorder=5)
    ax.plot([-radius, radius], [0.0, 0.0],
            color="green", linewidth=1.5, linestyle="--", zorder=5)


def scatter(ax, x: torch.Tensor, title: str):
    x = x.detach().float().cpu()
    ax.scatter(x[:, 0].numpy(), x[:, 1].numpy(), s=3, alpha=0.6)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def scatter_underlay(ax, x: torch.Tensor, s: float = 1.0, alpha: float = 0.08):
    x = x.detach().float().cpu()
    ax.scatter(x[:, 0].numpy(), x[:, 1].numpy(), s=s, alpha=alpha, zorder=0)


def print_J_stats(tag: str, J: torch.Tensor):
    J = J.detach()
    print(
        f"[{tag}] J: mean={J.mean().item():.6f} | "
        f"std={J.std(unbiased=False).item():.6f} | "
        f"min={J.min().item():.6f} | "
        f"max={J.max().item():.6f}"
    )


# ---------------------------
# 9) Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train", action="store_true")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--plot", action="store_true")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    # training
    p.add_argument("--train_steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)

    # sampling
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--n_plot", type=int, default=4096)

    # guidance
    p.add_argument("--beta_final", type=float, default=-2.0)

    # number of returned samples (same meaning as before)
    p.add_argument("--K", type=int, default=256)

    # NEW: particles-per-sample for FKC grouped corrector
    p.add_argument(
        "--fkc_particles_per_sample",
        type=int,
        default=8,
        help="Inner particle batch size per returned FKC sample. Total simulated diffusions = K * fkc_particles_per_sample."
    )

    # FKC active interval + resampling
    p.add_argument("--tmin", type=float, default=0.05)
    p.add_argument("--tmax", type=float, default=0.95)
    p.add_argument("--resample_every", type=int, default=1)
    p.add_argument("--no_center_dw", action="store_true")

    # objective weights
    p.add_argument("--c_eq", type=float, default=50.0)
    p.add_argument("--c_ineq", type=float, default=50.0)
    p.add_argument("--cost_weight", type=float, default=1.0)

    # kept for interface compatibility (not used for paper-faithful core)
    p.add_argument("--clip_J", type=float, default=1e4)
    p.add_argument("--clip_gradJ", type=float, default=5e2)
    p.add_argument("--clip_dw", type=float, default=200.0)

    p.add_argument("--ckpt", type=str, default="score_mlp.pt")
    return p.parse_args()


def pick_device(which: str) -> torch.device:
    if which == "cpu":
        return torch.device("cpu")
    if which == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    device = pick_device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    sde = VPSDE(beta_min=0.1, beta_max=20.0)  # beta_const is used (default 1.0)
    gmm = GMM4.default(device)
    model = ScoreMLP(hidden=128, time_dim=64).to(device=device, dtype=torch.float32)

    if args.train:
        train(
            model=model,
            sde=sde,
            gmm=gmm,
            steps=args.train_steps,
            batch=args.batch,
            lr=args.lr,
            save_path=args.ckpt,
        )

    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"[load] loaded: {args.ckpt}")
    else:
        print(f"[warn] checkpoint not found: {args.ckpt} (run --train first)")

    if args.sample:
        x_data = gmm.sample(args.n_plot)
        x_base = sample_baseline(model, sde, n=args.n_plot, steps=args.steps, eps=args.eps)

        # cost is constant 1 (as requested)
        obj = ConstrainedObjective(
            #cost_fn=lambda x: -x[:, 1],
            cost_fn=None,
            cost_weight=args.cost_weight,
            c_eq=args.c_eq,
            c_ineq=args.c_ineq,
        )

        # linear baseline returns K samples (same as before)
        x_lin = sample_linear_score_plus_gradJ(
            model=model,
            sde=sde,
            obj=obj,
            n=args.K,
            steps=args.steps,
            eps=args.eps,
            beta_final=args.beta_final,
            clip_J=args.clip_J,
            clip_gradJ=args.clip_gradJ,
        )

        # FKC now returns K *picked* samples, each produced from fkc_particles_per_sample inner diffusions
        x_fkc = fkc_guided_sample_grouped(
            model=model,
            sde=sde,
            obj=obj,
            K_out=args.K,
            K_inner=max(1, args.fkc_particles_per_sample),
            steps=args.steps,
            beta_final=args.beta_final,
            eps=args.eps,
            tmin=args.tmin,
            tmax=args.tmax,
            resample_every=max(1, args.resample_every),
            center_dw=(not args.no_center_dw),
            clip_J=args.clip_J,
            clip_gradJ=args.clip_gradJ,
            clip_dw=args.clip_dw,
        )

        J_lin = obj.J_only(x_lin)
        J_fkc = obj.J_only(x_fkc)

        print_J_stats("linear_score_plus_gradJ", J_lin)
        print_J_stats("FKC", J_fkc)

        if args.plot:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 4, figsize=(20, 4))
            scatter(axs[0], x_data, "True data: 4-Gaussian mixture")
            scatter(axs[1], x_base, "Baseline diffusion samples")
            scatter_underlay(axs[2], x_base)
            scatter(axs[2], x_lin, f"Linear score+∇J (beta_final={args.beta_final})")
            draw_constraint_circle(axs[2], radius=R, color="black", linewidth=1.5)
            scatter_underlay(axs[3], x_base)
            scatter(
                axs[3],
                x_fkc,
                f"FKC grouped (beta_final={args.beta_final}, inner={args.fkc_particles_per_sample})"
            )
            draw_constraint_circle(axs[3], radius=R, color="black", linewidth=1.5)
            plt.tight_layout()
            plt.savefig("fkcdiffusion_results.png", dpi=200, bbox_inches="tight")
            print("[plot] saved: fkcdiffusion_results.png")


if __name__ == "__main__":
    main()