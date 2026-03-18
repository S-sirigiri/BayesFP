# make_pusht_diffusion_traces.py
"""
Usage examples:

1) Freeze a start state sampled once from a seed:
    python make_pusht_diffusion_traces.py \
        --checkpoint /path/to/your_checkpoint.ckpt \
        --out diffusion_policy_traces.png \
        --device cuda:0 \
        --init_seed 1 \
        --n_rollouts 25

2) Use an explicit hard-coded initial state:
    python make_pusht_diffusion_traces.py \
        --checkpoint /path/to/your_checkpoint.ckpt \
        --out diffusion_policy_traces.png \
        --device cuda:0 \
        --init_state 404.0 410.0 250.0 285.0 -0.82 \
        --n_rollouts 25

State format:
    --init_state agent_x agent_y block_x block_y block_angle

Notes:
- If --init_state is given, it overrides --init_seed.
- If only --init_seed is given, the env samples once from that seed, then freezes
  that sampled state for all rollouts.
- Diffusion sampling randomness still changes across rollouts via torch.manual_seed(k).
"""

import os
import re
import inspect
import dill
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from collections import deque
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv


def load_policy(checkpoint_path, device="cuda:0"):
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=os.getcwd())
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def get_policy_device(policy):
    if hasattr(policy, "device"):
        return policy.device
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def stack_obs(buf):
    # buf holds dict obs from PushTImageEnv: {'image': (3,H,W), 'agent_pos': (2,)}
    imgs = np.stack([o["image"] for o in buf], axis=0).astype(np.float32)      # (T,3,H,W)
    apos = np.stack([o["agent_pos"] for o in buf], axis=0).astype(np.float32)  # (T,2)
    return {"image": imgs[None], "agent_pos": apos[None]}  # add batch dim


def freeze_init_state(env, init_state=None, init_seed=None):
    """
    Freeze PushT to a specific initial state.

    Priority:
    1) If init_state is provided, use it directly.
    2) Else sample once using init_seed, then store the sampled state in
       env.reset_to_state so all future env.reset() calls reuse it.

    State format:
        [agent_x, agent_y, block_x, block_y, block_angle]
    """
    if init_state is not None:
        frozen = np.asarray(init_state, dtype=np.float64).copy()
        if frozen.shape != (5,):
            raise ValueError(
                f"init_state must have shape (5,), got shape {frozen.shape}. "
                "Expected: [agent_x, agent_y, block_x, block_y, block_angle]"
            )
        env.reset_to_state = frozen
        return frozen.copy()

    if init_seed is None:
        raise ValueError("Either init_state or init_seed must be provided.")

    # Sample once from the seed, then freeze that exact sampled state
    env.seed(init_seed)
    _ = env.reset()
    frozen = np.array([
        env.agent.position[0],
        env.agent.position[1],
        env.block.position[0],
        env.block.position[1],
        env.block.angle,
    ], dtype=np.float64)

    env.reset_to_state = frozen
    return frozen.copy()


@torch.no_grad()
def rollout(policy, env, n_obs_steps, n_action_steps, max_steps):
    # env.reset() will use env.reset_to_state if it has been set
    obs = env.reset()

    # init obs buffer (pad by repeating first obs, same idea as MultiStepWrapper)
    buf = deque([obs], maxlen=n_obs_steps)
    while len(buf) < n_obs_steps:
        buf.append(obs)

    traj = []
    steps = 0
    done = False
    policy_device = get_policy_device(policy)

    while (not done) and (steps < max_steps):
        obs_np = stack_obs(list(buf))
        obs_t = {k: torch.from_numpy(v).to(policy_device) for k, v in obs_np.items()}

        # policy.predict_action is what PushTImageRunner calls
        action_dict = policy.predict_action(obs_t)
        actions = action_dict["action"].detach().cpu().numpy()[0]  # (n_action_steps, 2)

        for a in actions[:n_action_steps]:
            obs, reward, done, info = env.step(a)
            # PushTEnv puts agent position in info['pos_agent']
            traj.append(np.array(info["pos_agent"], dtype=np.float32))
            buf.append(obs)
            steps += 1
            if done or steps >= max_steps:
                break

    return np.stack(traj, axis=0) if len(traj) else np.zeros((0, 2), dtype=np.float32)


def _world_to_pixel(world_coords, render_size, world_scale=512.0):
    """Convert world coords (0..world_scale) to pixel coords (0..render_size)."""
    return world_coords / world_scale * render_size


def _get_constraint_classes():
    """Discover CostConstraintObjective and all its subclasses from the constraint module."""
    from diffusion_policy.policy import cost_constraint_objective as constraint_module
    from diffusion_policy.policy.cost_constraint_objective import CostConstraintObjective

    classes = [CostConstraintObjective]
    for _name, obj in inspect.getmembers(constraint_module, inspect.isclass):
        if obj is not CostConstraintObjective and issubclass(obj, CostConstraintObjective):
            classes.append(obj)
    return classes


def _extract_inequality_y_thresholds(constraint_instance):
    """
    Extract y-thresholds from inequality_penalty source (e.g. (40.0-y_T) -> y >= 40.0).
    Returns list of floats; empty if not found or no second dimension.
    """
    try:
        source = inspect.getsource(constraint_instance.inequality_penalty)
    except (TypeError, OSError):
        return []

    pattern = re.compile(r"\(\s*(\d+\.?\d*)\s*-\s*y_T\s*\)")
    return [float(m.group(1)) for m in pattern.finditer(source)]


def _plot_constraints(ax, render_size, world_scale=512.0):
    """
    Dynamically read constraint classes from cost_constraint_objective and plot
    equality (target_state) and inequality (y-threshold lines) for each.
    """
    constraint_classes = _get_constraint_classes()

    for cls in constraint_classes:
        try:
            obj = cls()
        except Exception:
            continue

        # ----- Equality: target_state (point in state space) -----
        if getattr(obj, "target_state", None) is not None:
            target = obj.target_state
            if isinstance(target, torch.Tensor):
                target = target.detach().cpu().numpy()
            target = np.atleast_2d(target)

            for i in range(target.shape[0]):
                pt = target[i]
                if pt.size >= 2:
                    x_w, y_w = float(pt.flat[0]), float(pt.flat[1])
                    x_px = _world_to_pixel(x_w, render_size, world_scale)
                    y_px = _world_to_pixel(y_w, render_size, world_scale)
                    ax.scatter(
                        [x_px], [y_px],
                        s=80,
                        c="lime",
                        edgecolors="darkgreen",
                        linewidths=2,
                        zorder=5,
                        label="target (eq)" if i == 0 else None
                    )

        # ----- Inequality: y >= threshold (horizontal line) -----
        thresholds = _extract_inequality_y_thresholds(obj)
        for j, th in enumerate(thresholds):
            y_px = _world_to_pixel(th, render_size, world_scale)
            ax.axhline(
                y=y_px,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label="y ≥ threshold (ineq)" if j == 0 else None
            )

        # Optional fallback: support explicit class attribute
        if not thresholds and hasattr(obj, "inequality_y_threshold"):
            th = getattr(obj, "inequality_y_threshold")
            y_px = _world_to_pixel(float(th), render_size, world_scale)
            ax.axhline(
                y=y_px,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label="y ≥ threshold (ineq)"
            )


def _plot_custom_constraints(ax, render_size, world_scale=512.0):
    """
    Placeholder for custom user-defined constraints.
    Add arbitrary equations / curves here.

    For now: draw a circle centered at a placeholder world coordinate
    with a placeholder radius.
    """
    """center_x_w = 256.0
    center_y_w = 256.0
    radius_w = 148.0

    center_x_px = _world_to_pixel(center_x_w, render_size, world_scale)
    center_y_px = _world_to_pixel(center_y_w, render_size, world_scale)
    radius_px = _world_to_pixel(radius_w, render_size, world_scale)

    circle = plt.Circle(
        (center_x_px, center_y_px),
        radius_px,
        fill=False,
        color="cyan",
        linestyle="-.",
        linewidth=2.0,
        alpha=0.9,
        zorder=6,
        label="custom constraint",
    )
    ax.add_patch(circle)"""

    # -------- Add constraint: y = -x (shifted to workspace: y = 512 - x) --------
    x_world = np.array([0.0, world_scale])
    y_world = world_scale - x_world  # y = 512 - x

    x_px = _world_to_pixel(x_world, render_size, world_scale)
    y_px = _world_to_pixel(y_world, render_size, world_scale)

    ax.plot(
        x_px,
        y_px,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.9,
        zorder=6,
        label="y = -x (custom)"
    )


def main(
    checkpoint,
    out_png="diffusion_policy_traces.png",
    device="cuda:0",
    init_seed=100000,
    init_state=None,
    n_rollouts=25,
    grid=False,
    plot_constraints=False,
    plot_custom_constraints=False,
):
    policy, cfg = load_policy(checkpoint, device=device)

    # pull rollout params from the checkpoint config (same ones eval uses)
    er = cfg.task.env_runner
    n_obs_steps = int(er.n_obs_steps)
    n_action_steps = int(er.n_action_steps)
    max_steps = int(er.max_steps)
    render_size = int(getattr(er, "render_size", 96))
    legacy_test = bool(getattr(er, "legacy_test", False))

    env = PushTImageEnv(legacy=legacy_test, render_size=render_size)

    # Freeze the initial state exactly
    frozen_state = freeze_init_state(
        env,
        init_state=init_state,
        init_seed=init_seed
    )
    print("Using frozen init state:", frozen_state.tolist())

    # Background render from the exact same frozen state
    _ = env.reset()
    bg = env.render(mode="rgb_array")  # HxWx3

    all_trajs_px = []
    for k in tqdm(
        range(n_rollouts),
        desc="Rollouts",
        unit="rollout",
        dynamic_ncols=True,
        leave=True
    ):
        # Keep env initial state fixed; only vary diffusion sampling
        torch.manual_seed(k)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(k)

        traj = rollout(policy, env, n_obs_steps, n_action_steps, max_steps)

        # map world coords (0..512) -> pixels (0..render_size)
        traj_px = traj / 512.0 * float(render_size)
        all_trajs_px.append(traj_px)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=200)
    ax.imshow(bg)

    for traj_px in all_trajs_px:
        if len(traj_px) >= 2:
            ax.plot(traj_px[:, 0], traj_px[:, 1], linewidth=1.5, alpha=0.8)

    if plot_constraints:
        _plot_constraints(ax, render_size, world_scale=512.0)

    if plot_custom_constraints:
        _plot_custom_constraints(ax, render_size, world_scale=512.0)

    if grid:
        n_ticks = 5
        tick_positions = np.linspace(0, render_size - 1, n_ticks)
        # Tick labels in world coordinates (PushT uses 0–512 for both x and y)
        world_labels = (tick_positions / (render_size - 1)) * 512.0
        tick_labels = [str(int(round(l))) for l in world_labels]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.grid(True, alpha=0.4, color="gray", linestyle="-")
        for spine in ax.spines.values():
            spine.set_visible(True)
    else:
        ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    print("Wrote:", out_png)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="diffusion_policy_traces.png")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--init_seed",
        type=int,
        default=100000,
        help="Seed used only once to sample an initial state, which is then frozen."
    )
    p.add_argument(
        "--init_state",
        type=float,
        nargs=5,
        default=None,
        metavar=("AGENT_X", "AGENT_Y", "BLOCK_X", "BLOCK_Y", "BLOCK_ANGLE"),
        help="Explicit initial state: agent_x agent_y block_x block_y block_angle. "
             "Overrides --init_seed."
    )
    p.add_argument("--n_rollouts", type=int, default=25)
    p.add_argument("--grid", action="store_true", help="Draw grid lines on the plot.")
    p.add_argument(
        "--plot_constraints",
        action="store_true",
        help="Plot constraints from cost_constraint_objective (equality target, inequality lines)."
    )
    p.add_argument(
        "--plot_custom_constraints",
        action="store_true",
        help="Plot custom constraints from _plot_custom_constraints."
    )

    args = p.parse_args()

    main(
        checkpoint=args.checkpoint,
        out_png=args.out,
        device=args.device,
        init_seed=args.init_seed,
        init_state=args.init_state,
        n_rollouts=args.n_rollouts,
        grid=args.grid,
        plot_constraints=args.plot_constraints,
        plot_custom_constraints=args.plot_custom_constraints,
    )