"""Inference-time cost J(x) and gradient for FKC / linear_combo guidance.

Computed from the **ground-truth analytical constraints** defined in
``env.py`` — no SDF approximation. Each constraint contributes a
squared-hinge penalty on its analytical violation function, mirroring
``src/diffusion_policy/diffusion_policy/policy/cost_constraint_objective.py``:

    J(x) = sum_regions  c_ineq * mean_t  max(0, h_region(x_t))^2

where ``h_region(x)`` is the analytical inside-distance function for the
region (positive when ``x`` violates the constraint, negative otherwise).

For a forbidden disk centred at ``c`` with radius ``r``:

    h_circle(x) = r - || x - c ||

For a forbidden triangle (CCW polygon with outward normals ``n_i`` and
vertices ``v_i``):

    h_triangle(x) = -max_i  n_i . (x - v_i)
                  =  min_i (-n_i . (x - v_i))         # negative outside

Both formulas are differentiable in ``x``, so ``torch.autograd.grad`` gives
``∇J`` for FKC / linear_combo without any precomputed grid.

The *training-time* obstacles (the 6 disks the diffusion model learned to
avoid) are NOT part of J — only the inference-time scenario constraints
contribute. They appear here only as evaluation metrics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .env import CircleRegion, ShadedRegion, TriangleObstacleEnv, TriangleRegion


# ---------------------------------------------------------------------------
# Per-region analytical violation
# ---------------------------------------------------------------------------


def _circle_violation(pts: torch.Tensor, region: CircleRegion) -> torch.Tensor:
    """h(x) = r - ||x - c||, returns relu(h) — positive only inside the disk."""
    c = pts.new_tensor([region.cx, region.cy])
    inside_dist = region.r - torch.linalg.vector_norm(pts - c, dim=-1)
    return F.relu(inside_dist)


def _triangle_violation(pts: torch.Tensor,
                         region: TriangleRegion) -> torch.Tensor:
    """h(x) = -max_i n_i . (x - v_i), returns relu(h) — positive inside.

    For a CCW polygon with outward-facing edge normals ``n_i``, the quantity
    ``max_i n_i . (x - v_i)`` is negative for points strictly inside the
    polygon (so its negation is positive — that's the violation).
    """
    verts = pts.new_tensor(region._verts)        # (3, 2)
    normals = pts.new_tensor(region._normals)    # (3, 2)
    diff = pts.unsqueeze(-2) - verts              # (..., 3, 2)
    halfplane = (diff * normals).sum(dim=-1)      # (..., 3)
    inside_dist = -halfplane.max(dim=-1).values   # (...,)  positive inside
    return F.relu(inside_dist)


def _region_violation(pts: torch.Tensor,
                       region: ShadedRegion) -> torch.Tensor:
    if isinstance(region, CircleRegion):
        return _circle_violation(pts, region)
    if isinstance(region, TriangleRegion):
        return _triangle_violation(pts, region)
    raise TypeError(type(region))


# ---------------------------------------------------------------------------
# Cost J(x) and gradient
# ---------------------------------------------------------------------------


def constraint_cost(traj: torch.Tensor, env: TriangleObstacleEnv,
                    clearance: float = 0.0) -> torch.Tensor:
    """J(x): (B, T, 2) -> (B,).

    For each inference-time constraint, sums  mean_t [ relu(h(x_t) + clearance)^2 ]
    where h(x) is the analytical inside-distance. ``clearance >= 0`` shifts
    the violation boundary outward by that margin.
    """
    if not env.inference_constraints:
        return traj.new_zeros(traj.shape[0])
    cost = traj.new_zeros(traj.shape[0])
    for region in env.inference_constraints:
        v = _region_violation(traj, region)               # (B, T)
        if clearance > 0.0:
            v = v + clearance
            v = F.relu(v)
        cost = cost + (v ** 2).mean(dim=-1)
    return cost


def constraint_cost_and_grad(traj: torch.Tensor, env: TriangleObstacleEnv,
                              clearance: float = 0.0
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """(J, dJ/dtraj) via autograd. Returns (sum_J: scalar, grad_J: (B, T, 2))."""
    traj_g = traj.detach().clone().requires_grad_(True)
    j = constraint_cost(traj_g, env, clearance=clearance).sum()
    g = torch.autograd.grad(j, traj_g)[0]
    return j.detach(), g.detach()


# ---------------------------------------------------------------------------
# Evaluation metrics (NOT used as gradient signal — pure indicators)
# ---------------------------------------------------------------------------


def obstacle_collision_rate(traj: torch.Tensor,
                             env: TriangleObstacleEnv) -> torch.Tensor:
    """Fraction of trajectory points inside ANY of the 6 training-time disks.
    (B, T, 2) -> (B,).
    """
    centres = traj.new_tensor(env.obstacle_centers)
    dist = torch.linalg.vector_norm(
        traj.unsqueeze(-2) - centres, dim=-1
    )                                                      # (B, T, K)
    inside_any = (dist < env.obstacle_radius).any(dim=-1)   # (B, T)
    return inside_any.float().mean(dim=-1)


def constraint_violation_rate(traj: torch.Tensor,
                               env: TriangleObstacleEnv) -> torch.Tensor:
    """Fraction of trajectory points inside ANY inference-time constraint.
    (B, T, 2) -> (B,).
    """
    if not env.inference_constraints:
        return traj.new_zeros(traj.shape[0])
    inside = torch.zeros(traj.shape[:-1], dtype=torch.bool, device=traj.device)
    for region in env.inference_constraints:
        v = _region_violation(traj, region)
        inside = inside | (v > 0)
    return inside.float().mean(dim=-1)


def goal_reach(traj: torch.Tensor, env: TriangleObstacleEnv,
               tol: float = 0.04) -> torch.Tensor:
    """Did the trajectory's final point land on the goal line?
    (B, T, 2) -> (B,) bool.
    """
    last = traj[:, -1]
    on_y = (last[:, 1] - env.goal_y).abs() < tol
    on_x = (last[:, 0] >= env.goal_x_min - tol) & (last[:, 0] <= env.goal_x_max + tol)
    return (on_y & on_x).float()
