"""
Inference with optional cost-guided guidance: linear combo (score + grad J)
or Feynman-Kac correctors (SMC). All guidance uses autograd via the
CostConstraintObjective class.

Notes on the FKC branch
-----------------------
This implementation keeps the original three guidance modes but fixes the
FKC discretization so that the weight update matches the *effective* discrete
VP/DDPM step that the scheduler actually applies.

For a skipped-step DDPM chain, the relevant per-step variance increment is
    current_beta_t = 1 - alpha_bar_t / alpha_bar_prev,
not the raw training beta stored in scheduler.betas[t]. In the continuous VP
SDE, this quantity corresponds to beta(t) * dt for the current reverse step.
Therefore the FKC log-weight increment should use `current_beta_t` directly,
with no extra normalized `dt` multiplier.
"""

from typing import Optional, Tuple

import torch

from .cost_constraint_objective import CostConstraintObjective


# ========== Hyperparameters (single place to edit) ==========
# Which guidance to use: "none" | "linear_combo" | "fkc"
GUIDANCE_MODE = "none"

# Linear combination (score + (beta_guid/2) * grad_J)
BETA_GUID = 1.0

# FKC (Feynman-Kac correctors with SMC)
FKC_NUM_PARTICLES = 8
FKC_RESAMPLE_EVERY = 1
FKC_TMIN = 0.05
FKC_TMAX = 0.95
FKC_CENTER_DW = True


def _get_alpha_sigma(scheduler, t: torch.Tensor, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get alpha_bar_t and sigma_t for DDPM-style scheduler (for score <-> epsilon)."""
    t_long = t.long()
    if t_long.dim() == 0:
        t_long = t_long.unsqueeze(0)
    alpha_bar = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    alpha_prod_t = alpha_bar[t_long].flatten()
    while alpha_prod_t.dim() < 2:
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    sigma_t = (1.0 - alpha_prod_t).clamp(min=1e-8).sqrt()
    return alpha_prod_t, sigma_t


def _prediction_type(scheduler) -> str:
    pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if pred_type not in ("epsilon", "sample"):
        raise ValueError(f"Unsupported prediction type {pred_type}")
    return pred_type


def _broadcast_noise_terms(alpha_prod_t: torch.Tensor, sigma_t: torch.Tensor, ref: torch.Tensor):
    while alpha_prod_t.dim() < ref.dim():
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    while sigma_t.dim() < ref.dim():
        sigma_t = sigma_t.unsqueeze(-1)
    return alpha_prod_t, sigma_t


def _model_output_to_eps_and_score(
    scheduler,
    t: torch.Tensor,
    xt: torch.Tensor,
    model_output: torch.Tensor,
    device,
    dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha_prod_t, sigma_t = _get_alpha_sigma(scheduler, t, device, dtype)
    alpha_prod_t, sigma_t = _broadcast_noise_terms(alpha_prod_t, sigma_t, xt)
    pred_type = _prediction_type(scheduler)

    if pred_type == "epsilon":
        eps = model_output
    else:  # pred_type == "sample"
        sqrt_alpha_prod_t = alpha_prod_t.clamp(min=1e-8).sqrt()
        eps = (xt - sqrt_alpha_prod_t * model_output) / sigma_t

    score = -eps / sigma_t
    return eps, score


def _eps_to_model_output(
    scheduler,
    t: torch.Tensor,
    xt: torch.Tensor,
    eps: torch.Tensor,
    device,
    dtype,
) -> torch.Tensor:
    pred_type = _prediction_type(scheduler)
    if pred_type == "epsilon":
        return eps

    alpha_prod_t, sigma_t = _get_alpha_sigma(scheduler, t, device, dtype)
    alpha_prod_t, sigma_t = _broadcast_noise_terms(alpha_prod_t, sigma_t, xt)
    sqrt_alpha_prod_t = alpha_prod_t.clamp(min=1e-8).sqrt()
    return (xt - sigma_t * eps) / sqrt_alpha_prod_t


def _apply_cost_guidance(
    scheduler,
    t: torch.Tensor,
    xt: torch.Tensor,
    model_output: torch.Tensor,
    grad_J: torch.Tensor,
    device,
    dtype,
) -> torch.Tensor:
    """
    Modify the model output so that the implied score becomes
        score_tilde = score + (beta_cost / 2) * grad_J,
    where beta_cost = -BETA_GUID for a cost objective J.

    Since score = -eps / sigma_score, this is achieved by
        eps_tilde = eps + (BETA_GUID / 2) * sigma_score * grad_J.
    """
    _, sigma_t = _get_alpha_sigma(scheduler, t, device, dtype)
    while sigma_t.dim() < xt.dim():
        sigma_t = sigma_t.unsqueeze(-1)
    eps, _ = _model_output_to_eps_and_score(
        scheduler, t, xt, model_output, device, dtype
    )
    eps_tilde = eps + (BETA_GUID / 2.0) * sigma_t * grad_J
    return _eps_to_model_output(
        scheduler, t, xt, eps_tilde, device, dtype
    )


def _get_prev_train_timestep(timesteps, i: int) -> int:
    """
    Previous discrete training-time index used by the current inference step.

    This mirrors the DDPMScheduler notion of `prev_timestep`: for the last
    inference step it is -1, corresponding to alpha_bar_prev = 1.
    """
    if i + 1 >= len(timesteps):
        return -1
    t_prev = timesteps[i + 1]
    return int(t_prev.item() if hasattr(t_prev, "item") else t_prev)



def _get_current_step_beta(
    scheduler,
    timesteps,
    i: int,
    device,
    dtype,
) -> torch.Tensor:
    """
    Effective VP variance increment for the current reverse step.

    For a skipped-step DDPM chain this is
        current_beta_t = 1 - alpha_bar_t / alpha_bar_prev,
    exactly the coefficient used internally by DDPMScheduler.step(...).

    Important: this is already a *discrete step increment* (continuous beta(t)
    multiplied by the step size), so downstream Euler/FKC formulas should NOT
    multiply by an additional normalized `dt`.
    """
    t_cur = timesteps[i]
    t_cur_long = t_cur.long()
    if t_cur_long.dim() == 0:
        t_cur_long = t_cur_long.unsqueeze(0)

    alpha_bar = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    alpha_prod_t = alpha_bar[t_cur_long].flatten()

    prev_t = _get_prev_train_timestep(timesteps, i)
    if prev_t >= 0:
        alpha_prod_t_prev = alpha_bar[prev_t].expand_as(alpha_prod_t)
    else:
        alpha_prod_t_prev = torch.ones_like(alpha_prod_t)

    current_alpha_t = alpha_prod_t / alpha_prod_t_prev.clamp(min=1e-12)
    current_beta_t = 1.0 - current_alpha_t

    while current_beta_t.dim() < 2:
        current_beta_t = current_beta_t.unsqueeze(-1)
    return current_beta_t



def _get_step_forward_drift_increment(
    scheduler,
    timesteps,
    i: int,
    xt: torch.Tensor,
    device,
    dtype,
) -> torch.Tensor:
    """
    Discrete forward-drift increment f_t(x_t) * dt for the current VP step.

    For the VP/DDPM SDE, f_t(x) = -(beta(t)/2) x. Using the effective skipped-step
    coefficient current_beta_t = beta(t) * dt gives
        f_t(x) * dt = -(current_beta_t / 2) * x.
    """
    current_beta_t = _get_current_step_beta(scheduler, timesteps, i, device, dtype)
    while current_beta_t.dim() < xt.dim():
        current_beta_t = current_beta_t.unsqueeze(-1)
    return -(current_beta_t / 2.0) * xt


@torch.no_grad()
def _systematic_resample_batched(logw: torch.Tensor) -> torch.Tensor:
    """Batched systematic resampling. logw: (B, K) -> idx (B, K)."""
    B, K = logw.shape
    logw = logw - logw.max(dim=1, keepdim=True).values
    w = torch.softmax(logw, dim=1)
    bad = (~torch.isfinite(w)).any(dim=1) | (w.sum(dim=1) <= 0)
    if bad.any():
        w[bad] = 1.0 / K
    cdf = torch.cumsum(w, dim=1)
    cdf[:, -1] = 1.0
    u0 = torch.rand(B, 1, device=logw.device, dtype=logw.dtype) / K
    u = u0 + (torch.arange(K, device=logw.device, dtype=logw.dtype)[None, :] / K)
    u = torch.clamp(u, 0.0, 1.0 - 1e-7)
    idx = torch.searchsorted(cdf, u)
    return torch.clamp(idx, 0, K - 1).long()


@torch.no_grad()
def _sample_one_index_from_logw(logw: torch.Tensor) -> torch.Tensor:
    """One categorical sample per row from logw. logw: (B, K) -> idx (B,)."""
    logw = logw - logw.max(dim=1, keepdim=True).values
    g = -torch.log(-torch.log(torch.rand_like(logw)))
    return (logw + g).argmax(dim=1)


class Inference:
    """
    Shared conditional_sample implementation for diffusion policies.
    Supports optional cost/constraint guidance via linear combo or FKC (SMC).
    Cost/constraint objective is held on the class; edit CostConstraintObjective
    to change cost and constraints.
    """

    _COND_UNSET = object()
    cost_constraint_objective = CostConstraintObjective()

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        cond=_COND_UNSET,
        generator=None,
        *,
        guidance_mode: Optional[str] = None,
        **kwargs
    ):
        model = self.model
        scheduler = self.noise_scheduler
        device = condition_data.device
        dtype = condition_data.dtype
        B, T, D = condition_data.shape

        mode = guidance_mode if guidance_mode is not None else GUIDANCE_MODE
        use_linear = mode == "linear_combo"
        use_fkc = mode == "fkc"

        if use_fkc:
            return self._conditional_sample_fkc(
                condition_data,
                condition_mask,
                local_cond,
                global_cond,
                cond,
                generator,
                self.cost_constraint_objective,
                B, T, D,
                kwargs,
            )

        # Single trajectory (no FKC) or baseline.
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]

            if cond is self._COND_UNSET:
                model_output = model(
                    trajectory, t,
                    local_cond=local_cond,
                    global_cond=global_cond,
                )
            else:
                model_output = model(trajectory, t, cond)

            if use_linear:
                _, grad_J = self.cost_constraint_objective.J_and_grad(trajectory)
                model_output = _apply_cost_guidance(
                    scheduler, t, trajectory, model_output, grad_J, device, dtype
                )

            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _conditional_sample_fkc(
        self,
        condition_data,
        condition_mask,
        local_cond,
        global_cond,
        cond,
        generator,
        cost_objective: "CostConstraintObjective",
        B, T, D,
        step_kwargs,
    ):
        """
        Cost-guided FKC with SMC: K particles per batch index, resample, then
        pick one per batch.

        The proposal dynamics still use the scheduler's reverse step, but the
        FKC weights now use the *matching skipped-step VP coefficient*
            current_beta_t = 1 - alpha_bar_t / alpha_bar_prev,
        and therefore do not multiply by any extra normalized `dt`.
        """
        model = self.model
        scheduler = self.noise_scheduler
        device = condition_data.device
        dtype = condition_data.dtype
        K = max(1, FKC_NUM_PARTICLES)

        # Expand batch: (B, T, D) -> (B*K, T, D)
        cond_exp = condition_data.unsqueeze(1).expand(B, K, T, D).reshape(B * K, T, D)
        mask_exp = condition_mask.unsqueeze(1).expand(B, K, T, D).reshape(B * K, T, D)
        if local_cond is not None:
            local_cond = local_cond.unsqueeze(1).expand(B, K, *local_cond.shape[1:]).reshape(B * K, *local_cond.shape[1:])
        if global_cond is not None:
            global_cond = global_cond.unsqueeze(1).expand(B, K, *global_cond.shape[1:]).reshape(B * K, *global_cond.shape[1:])

        trajectory = torch.randn(B * K, T, D, dtype=dtype, device=device, generator=generator)
        logw = torch.zeros(B, K, device=device, dtype=dtype)

        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps
        t_denom = max(1, int(scheduler.config.num_train_timesteps) - 1)

        for i, t in enumerate(timesteps):
            t_val = int(t.item() if hasattr(t, "item") else t)
            t_frac = float(t_val) / float(t_denom)
            active = FKC_TMIN <= t_frac <= FKC_TMAX

            trajectory[mask_exp] = cond_exp[mask_exp]

            if cond is self._COND_UNSET:
                model_output = model(
                    trajectory, t,
                    local_cond=local_cond,
                    global_cond=global_cond,
                )
            else:
                model_output = model(trajectory, t, cond)

            J_val, grad_J = cost_objective.J_and_grad(trajectory)

            # Guided proposal corresponding to the cost-tilted reverse SDE with
            # beta_cost = -BETA_GUID. In score form this adds
            #   (beta_cost / 2) * grad_J
            # to the base score, implemented here through the epsilon parameterization.
            model_output_guided = _apply_cost_guidance(
                scheduler, t, trajectory, model_output, grad_J, device, dtype
            )

            if active:
                # Constant-beta cost-guided FKC weight update.
                #
                # Continuous weighted SDE (notes Eq. 23 / 36 for constant beta):
                #   dw_t = <beta_cost * grad_J,
                #            (sigma_sde^2 / 2) * nabla log q_t - f_t> dt.
                #
                # For the discrete VP/DDPM sampler, the per-step increment used by
                # DDPMScheduler.step is
                #   current_beta_t = 1 - alpha_bar_t / alpha_bar_prev,
                # which already equals sigma_sde^2 * dt for the current skipped step.
                # Therefore the discrete Euler/FKC increment is
                #   Delta w = <beta_cost * grad_J,
                #              (current_beta_t / 2) * nabla log q_t - f_t * dt>.
                #
                # Since f_t(x) * dt = -(current_beta_t / 2) * x for VP/DDPM, we use
                # the matching discrete forward-drift increment below. No extra dt.
                beta_cost = -BETA_GUID

                current_beta_t = _get_current_step_beta(
                    scheduler, timesteps, i, device, dtype
                )
                while current_beta_t.dim() < trajectory.dim():
                    current_beta_t = current_beta_t.unsqueeze(-1)

                f_dt_xt = _get_step_forward_drift_increment(
                    scheduler, timesteps, i, trajectory, device, dtype
                )
                _, score_t = _model_output_to_eps_and_score(
                    scheduler, t, trajectory, model_output, device, dtype
                )

                combo = (current_beta_t / 2.0) * score_t - f_dt_xt

                g_flat = grad_J.reshape(B * K, -1)
                combo_flat = combo.reshape(B * K, -1)
                inner = (g_flat * combo_flat).sum(dim=-1)

                dw = beta_cost * inner
                dw = dw.reshape(B, K)

                # Optional: time-dependent cost tilt beta_cost(t).
                # If you add one, include the discrete counterpart of
                #   (d beta_cost / dt) * J * dt = Delta beta_cost * J.
                # For constant beta (the intended setup here), that term is zero.
                _ = J_val  # kept for readability / future extension

                if FKC_CENTER_DW:
                    dw = dw - dw.mean(dim=1, keepdim=True)
                logw = logw + dw

                if (i % max(1, FKC_RESAMPLE_EVERY)) == 0:
                    idx = _systematic_resample_batched(logw)
                    idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(B, K, T, D)
                    traj_res = trajectory.reshape(B, K, T, D)
                    traj_res = torch.gather(traj_res, 1, idx_exp)
                    trajectory = traj_res.reshape(B * K, T, D)
                    logw.zero_()

            trajectory = scheduler.step(
                model_output_guided, t, trajectory,
                generator=generator,
                **step_kwargs
            ).prev_sample

        # Pick one particle per batch by the weights accumulated since the last resampling.
        idx_pick = _sample_one_index_from_logw(logw)
        trajectory = trajectory.reshape(B, K, T, D)
        out = trajectory[torch.arange(B, device=device), idx_pick]

        sel_mask = mask_exp.reshape(B, K, T, D)[torch.arange(B, device=device), idx_pick]
        out[sel_mask] = condition_data[sel_mask]
        return out
