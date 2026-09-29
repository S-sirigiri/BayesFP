import argparse
import importlib.util
import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAAC_GR00T_ROOT = REPO_ROOT / "src" / "Isaac-GR00T"
COST_OBJECTIVE_PATH = (
    ISAAC_GR00T_ROOT / "gr00t" / "model" / "gr00t_n1d6" / "cost_constraint_objective.py"
)
INFERENCE_CONFIG_PATH = (
    ISAAC_GR00T_ROOT / "gr00t" / "model" / "gr00t_n1d6" / "cost_guided_inference.yaml"
)


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CostConstraintObjective = _load_module(
    "cost_constraint_objective_local",
    COST_OBJECTIVE_PATH,
).CostConstraintObjective


def _load_rollout(npz_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(npz_path, allow_pickle=False) as data:
        if "states" not in data or "actions" not in data:
            raise KeyError(f"{npz_path} must contain both 'states' and 'actions'. Found {data.files}.")
        metadata = {}
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"]))
        return np.asarray(data["states"]), np.asarray(data["actions"]), metadata


def _build_objective() -> CostConstraintObjective:
    with INFERENCE_CONFIG_PATH.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}
    config["embodiments"] = {
        int(embodiment_id): spec
        for embodiment_id, spec in config.get("embodiments", {}).items()
    }
    return CostConstraintObjective(
        objective_config=config.get("objective", {}),
        embodiments=config.get("embodiments", {}),
    )


def _reconstruct_proxy_positions(
    objective: CostConstraintObjective,
    states: np.ndarray,
    actions: np.ndarray,
    embodiment_id: int,
    n_action_steps: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            f"Expected unbatched arrays with shapes [T+1, D] and [T, D], got {states.shape} and {actions.shape}."
        )
    if states.shape[0] < 2:
        raise ValueError("Rollout must contain at least an initial state and one action.")

    num_steps = min(actions.shape[0], states.shape[0] - 1)
    proxy_positions = np.full((num_steps, 3), np.nan, dtype=np.float32)
    chunk_ranges: list[tuple[int, int]] = []

    for start in range(0, num_steps, n_action_steps):
        end = min(start + n_action_steps, num_steps)
        current_position = torch.as_tensor(states[start, :3], dtype=torch.float32)
        controller_actions = torch.as_tensor(actions[start:end], dtype=torch.float32)
        proxy_chunk = objective.reconstruct_libero_proxy_positions_from_controller_actions(
            controller_actions=controller_actions,
            embodiment_id=embodiment_id,
            current_position=current_position,
        )
        proxy_positions[start:end] = proxy_chunk.detach().cpu().numpy()
        chunk_ranges.append((start, end))

    return proxy_positions, chunk_ranges


def _compute_summary(actual_positions: np.ndarray, proxy_positions: np.ndarray) -> dict[str, float]:
    errors = proxy_positions - actual_positions
    xy_errors = errors[:, :2]
    return {
        "rmse_x": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "rmse_y": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "rmse_z": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
        "rmse_xy": float(np.sqrt(np.mean(np.sum(xy_errors**2, axis=1)))),
        "max_xy_error": float(np.max(np.sqrt(np.sum(xy_errors**2, axis=1)))),
    }


def plot_actual_vs_proxy(
    npz_path: Path,
    output_path: Path,
    embodiment_id: int,
    n_action_steps: int,
    show: bool,
) -> None:
    states, actions, metadata = _load_rollout(npz_path)
    objective = _build_objective()

    num_steps = min(actions.shape[0], states.shape[0] - 1)
    actual_positions = states[1 : num_steps + 1, :3].astype(np.float32, copy=False)
    proxy_positions, chunk_ranges = _reconstruct_proxy_positions(
        objective=objective,
        states=states,
        actions=actions,
        embodiment_id=embodiment_id,
        n_action_steps=n_action_steps,
    )
    summary = _compute_summary(actual_positions, proxy_positions)

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    ax.plot(
        actual_positions[:, 0],
        actual_positions[:, 1],
        color="#1f77b4",
        linewidth=2.2,
        label="Actual executed EE",
    )

    first_proxy = True
    for start, end in chunk_ranges:
        label = "Proxy rollout from cost_constraint_objective" if first_proxy else None
        first_proxy = False
        chunk_proxy = proxy_positions[start:end]
        ax.plot(
            chunk_proxy[:, 0],
            chunk_proxy[:, 1],
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.6,
            alpha=0.95,
            label=label,
        )
        ax.scatter(
            chunk_proxy[0, 0],
            chunk_proxy[0, 1],
            color="#ff7f0e",
            s=18,
            alpha=0.7,
        )

    ax.scatter(actual_positions[0, 0], actual_positions[0, 1], color="#1f77b4", s=28, zorder=4)
    ax.scatter(
        actual_positions[-1, 0],
        actual_positions[-1, 1],
        color="#1f77b4",
        marker="x",
        s=48,
        linewidths=1.8,
        zorder=4,
    )

    all_xy = np.concatenate([actual_positions[:, :2], proxy_positions[:, :2]], axis=0)
    ax.set_xlim(all_xy[:, 0].min() - 0.02, all_xy[:, 0].max() + 0.02)
    ax.set_ylim(all_xy[:, 1].min() - 0.02, all_xy[:, 1].max() + 0.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    env_name = metadata.get("env_name", npz_path.stem)
    ax.set_title(
        "LIBERO Actual vs Proxy EE Trajectory\n"
        f"{env_name}\n"
        f"steps={num_steps}, n_action_steps={n_action_steps}, "
        f"RMSE_xy={summary['rmse_xy']:.5f}, max_xy_error={summary['max_xy_error']:.5f}",
        fontsize=12,
        pad=8,
    )
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    print(f"Input rollout: {npz_path}")
    print("Comparison uses only the executed portion of each policy query.")
    print(
        "GR00T may predict a longer chunk, but only the first n_action_steps controller inputs "
        "have matching executed states before replanning."
    )
    print(f"Saved plot: {output_path}")
    print(
        f"RMSE xyz = ({summary['rmse_x']:.6f}, {summary['rmse_y']:.6f}, {summary['rmse_z']:.6f})"
    )
    print(
        f"RMSE_xy = {summary['rmse_xy']:.6f}, max_xy_error = {summary['max_xy_error']:.6f}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_npz", help="Path to a rollout NPZ containing 'states' and 'actions'.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to <rollout_stem>_actual_vs_proxy.png next to the NPZ.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Number of actions executed per policy query in rollout_policy.py.",
    )
    parser.add_argument(
        "--embodiment-id",
        type=int,
        default=2,
        help="Embodiment id used by cost_guided_inference.yaml. LIBERO_PANDA is 2 by default.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot after saving.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rollout_npz = Path(args.rollout_npz).expanduser().resolve()
    if args.output is None:
        output_path = rollout_npz.with_name(f"{rollout_npz.stem}_actual_vs_proxy.png")
    else:
        output_path = Path(args.output).expanduser().resolve()

    plot_actual_vs_proxy(
        npz_path=rollout_npz,
        output_path=output_path,
        embodiment_id=args.embodiment_id,
        n_action_steps=args.n_action_steps,
        show=args.show,
    )


if __name__ == "__main__":
    main()
