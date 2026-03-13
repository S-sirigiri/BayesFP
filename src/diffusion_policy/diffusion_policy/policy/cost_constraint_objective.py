"""
Cost and constraint objective for cost-guided diffusion inference.

Edit only the cost(), equality_penalty(), and inequality_penalty() methods.
All gradients are computed automatically via autograd in J_and_grad().
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F


class CostConstraintObjective:
    """
    Augmented-Lagrangian style objective for cost-guided diffusion.

    J(x) = cost_weight * cost(x)
         + equality_penalty(x)
         + inequality_penalty(x)

    - x is typically a batch of trajectories: shape (B, T, D),
      but code only assumes the last dimension is the feature dim.

    To customize behavior, edit:
      - cost(self, x)
      - equality_penalty(self, x)
      - inequality_penalty(self, x)

    All gradients are obtained via autograd; no hardcoded derivatives.
    """

    # -------- Hyperparameters (single place to edit) --------
    # Overall weight multiplying the cost term
    cost_weight: float = 1.0

    # Penalty weights for equality and inequality constraints
    c_eq: float = 50.0
    c_ineq: float = 50.0

    # Smoothness for soft inequality constraints (softplus)
    softplus_beta: float = 10.0

    # Numerical epsilon
    eps_numerical: float = 1e-8

    # Optional target terminal state for equality constraint
    # (e.g. desired final state). Shape: (D,) or broadcastable to (B, D).
    target_state: Optional[torch.Tensor] = None

    def __init__(
        self,
        cost_weight: Optional[float] = None,
        c_eq: Optional[float] = None,
        c_ineq: Optional[float] = None,
        softplus_beta: Optional[float] = None,
        target_state: Optional[torch.Tensor] = None,
    ):
        if cost_weight is not None:
            self.cost_weight = cost_weight
        if c_eq is not None:
            self.c_eq = c_eq
        if c_ineq is not None:
            self.c_ineq = c_ineq
        if softplus_beta is not None:
            self.softplus_beta = softplus_beta
        if target_state is not None:
            self.target_state = target_state

    # ------------- EDIT THESE THREE METHODS ONLY -------------

    def cost(self, x: torch.Tensor) -> torch.Tensor:
        """
        Example cost function.

        x: (B, T, D) or (B, ..., D)
        Returns: (B,) cost per sample.

        Example here:
        - Quadratic energy over the entire trajectory:
          sum over all non-batch dimensions of x^2.
        """
        B = x.shape[0]
        cost_val = (x ** 2).view(B, -1).sum(dim=1)
        return cost_val * 0.0

    def equality_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Example equality-constraint penalty.

        Returns: (B,) penalty per sample.

        Example here:
        - Terminal state x_T should be close to target_state (if provided):
          c_eq * ||x_T - target_state||^2.
        - If no target_state is provided, the penalty is zero.
        """
        B = x.shape[0]
        if self.target_state is None:
            return torch.zeros(B, device=x.device, dtype=x.dtype)

        # Ensure target_state is broadcastable to (B, D_flat)
        if self.target_state.dim() == 1:
            target = self.target_state.unsqueeze(0)  # (1, D)
        else:
            target = self.target_state

        # Assume last time step is the terminal state: x[:, -1, :]
        x_T = x[:, -1, ...]  # (B, D or ...)
        diff = x_T.view(B, -1) - target.view(1, -1)
        eq_pen = self.c_eq * (diff ** 2).sum(dim=1)
        return eq_pen * 0.0

    def inequality_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Example inequality-constraint penalty.

        Returns: (B,) penalty per sample.

        Example here:
        - Enforce that the second coordinate of the terminal state
          (e.g. a "height" or y-position) is >= 0:
          c_ineq * softplus(-y_T)^2   (smooth hinge).
        - If feature dimension < 2, penalty is zero.
        """
        """B = x.shape[0]
        if x.size(-1) < 2:
            return torch.zeros(B, device=x.device, dtype=x.dtype)

        # Terminal state
        x_T = x[:, -1, ...]      # (B, D)
        y_T = x_T[..., 1]        # (B,)

        # Positive part of violation: y_T < 0  ->  softplus(-y_T)
        v = F.softplus(self.softplus_beta * (-y_T)) / self.softplus_beta
        ineq_pen = self.c_ineq * (v ** 2)
        return ineq_pen
        
        B = x.shape[0]
        if x.size(-1) < 2:
            return torch.zeros(B, device=x.device, dtype=x.dtype)

        # If x is already normalized and you want the normalized threshold:
        y_threshold_norm = (384.0 / 512.0) * 2.0 - 1.0

        # Use all time steps
        y = x[..., 1]  # shape (B, T, ...)

        # Penalize when y < threshold
        v = F.softplus(self.softplus_beta * (y - y_threshold_norm)) / self.softplus_beta

        # Aggregate over all non-batch dimensions
        ineq_pen = self.c_ineq * (v ** 2).view(B, -1).sum(dim=1)
        return ineq_pen * 100"""

        B = x.shape[0]
        if x.size(-1) < 2:
            return torch.zeros(B, device=x.device, dtype=x.dtype)

        # Use all time steps, not just the last one
        x_coord = x[..., 0]  # shape (B, T, ...)
        y_coord = x[..., 1]  # shape (B, T, ...)

        # Violation is positive when y > -x
        violation = x_coord + y_coord

        # Smooth hinge
        v = F.softplus(self.softplus_beta * violation) / self.softplus_beta

        # Aggregate over all non-batch dimensions
        ineq_pen = self.c_ineq * (v ** 2).view(B, -1).sum(dim=1)
        return ineq_pen * 100

        # ------------- Derived helpers (do NOT edit) -------------

    def J(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augmented Lagrangian:
        J(x) = cost_weight * cost(x)
             + equality_penalty(x)
             + inequality_penalty(x)
        """
        return (
            self.cost_weight * self.cost(x)
            + self.equality_penalty(x)
            + self.inequality_penalty(x)
        )

    def J_and_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (J(x), grad_J(x)) using autograd.

        - J(x): (B,)
        - grad_J(x): same shape as x
        """
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            J_total = self.J(x_req)  # (B,)
            grad_J = torch.autograd.grad(
                J_total.sum(), x_req, create_graph=False
            )[0]
        return J_total.detach(), grad_J.detach()

    def J_only(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: J(x) without computing gradients."""
        return self.J(x)