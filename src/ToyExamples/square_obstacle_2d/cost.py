"""Inference-time cost J(x) and gradient for FKC / linear_combo guidance.

Computed analytically from the inverted-U boxes — no SDF approximation:

    J(x) = sum_box  mean_t  max(0, h_box(x_t))^2

with the inside-distance for an axis-aligned box

    h_box(x, y) = min( w/2 - |x - cx|,  h/2 - |y - cy| )

(positive inside the box, negative outside). ``relu(h_box)`` is the
violation amount that the squared hinge penalises. Differentiable in
``(x, y)``, so ``torch.autograd.grad`` returns ``∇J`` for FKC.

The training-time square obstacle is **not** part of J — only the
inference-time scenario constraints contribute. The square is used only
by the dataset to shape the demonstrations (and as an evaluation metric
for collision rate).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .env import (AnnulusSegmentRegion, BoxRegion, ShadedRegion,
                   SquareObstacleEnv)


# ---------------------------------------------------------------------------
# Per-region analytical violation
# ---------------------------------------------------------------------------


def _box_violation(pts: torch.Tensor, region: BoxRegion) -> torch.Tensor:
    """relu(min(half_w - |x-cx|, half_h - |y-cy|)) — positive inside the box."""
    dx = pts[..., 0] - region.cx
    dy = pts[..., 1] - region.cy
    pen_x = region.half_w - dx.abs()
    pen_y = region.half_h - dy.abs()
    inside_dist = torch.minimum(pen_x, pen_y)
    return F.relu(inside_dist)


def _annulus_segment_violation(pts: torch.Tensor,
                                region: AnnulusSegmentRegion) -> torch.Tensor:
    """relu( min( r_outer - d,  d - r_inner,  y - cy ) ) — positive inside the
    upper-half annulus.
    """
    dx = pts[..., 0] - region.cx
    dy = pts[..., 1] - region.cy
    d = torch.sqrt(dx * dx + dy * dy + 1e-12)
    pen_outer = region.r_outer - d                 # inside outer disk
    pen_inner = d - region.r_inner                 # outside inner disk
    pen_upper = dy                                  # upper half-plane
    inside_dist = torch.minimum(
        torch.minimum(pen_outer, pen_inner), pen_upper
    )
    return F.relu(inside_dist)


def _region_violation(pts: torch.Tensor,
                       region: ShadedRegion) -> torch.Tensor:
    if isinstance(region, BoxRegion):
        return _box_violation(pts, region)
    if isinstance(region, AnnulusSegmentRegion):
        return _annulus_segment_violation(pts, region)
    raise TypeError(type(region))


# ---------------------------------------------------------------------------
# Cost J(x) and gradient
# ---------------------------------------------------------------------------


def constraint_cost(traj: torch.Tensor, env: SquareObstacleEnv,
                    clearance: float = 0.0) -> torch.Tensor:
    """J(x): (B, T, 2) -> (B,).

    For each inference-time box, sums  mean_t [ relu(h_box(x_t) + clearance)^2 ].
    """
    if not env.inference_constraints:
        return traj.new_zeros(traj.shape[0])
    cost = traj.new_zeros(traj.shape[0])
    for region in env.inference_constraints:
        v = _region_violation(traj, region)        # (B, T)
        if clearance > 0.0:
            v = F.relu(v + clearance)
        cost = cost + (v ** 2).mean(dim=-1)
    return cost


def constraint_cost_and_grad(traj: torch.Tensor, env: SquareObstacleEnv,
                              clearance: float = 0.0
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """(J_sum, dJ/dtraj) via autograd."""
    traj_g = traj.detach().clone().requires_grad_(True)
    j = constraint_cost(traj_g, env, clearance=clearance).sum()
    g = torch.autograd.grad(j, traj_g)[0]
    return j.detach(), g.detach()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def square_collision_rate(traj: torch.Tensor,
                           env: SquareObstacleEnv) -> torch.Tensor:
    """Fraction of trajectory points inside the training square.
    (B, T, 2) -> (B,).
    """
    v = _box_violation(traj, env.training_box)     # (B, T)
    return (v > 0).float().mean(dim=-1)


def constraint_violation_rate(traj: torch.Tensor,
                               env: SquareObstacleEnv) -> torch.Tensor:
    """Fraction of trajectory points inside ANY inference-time box.
    (B, T, 2) -> (B,).
    """
    if not env.inference_constraints:
        return traj.new_zeros(traj.shape[0])
    inside = torch.zeros(traj.shape[:-1], dtype=torch.bool, device=traj.device)
    for region in env.inference_constraints:
        v = _region_violation(traj, region)
        inside = inside | (v > 0)
    return inside.float().mean(dim=-1)


def goal_reach(traj: torch.Tensor, env: SquareObstacleEnv,
               tol: float = 0.05) -> torch.Tensor:
    """Did the trajectory's final point land near the goal point?
    (B, T, 2) -> (B,) bool.
    """
    last = traj[:, -1]
    goal = traj.new_tensor(env.goal)
    return (torch.linalg.vector_norm(last - goal, dim=-1) < tol).float()
