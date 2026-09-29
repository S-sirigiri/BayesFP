import argparse
import logging
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch


TITLE = "End-Effector XY Trajectories With Scene Overlay"
TASK_SUITE_NAME = "libero_object_cylinder"
TASK_NAME = "pick_up_the_butter_and_place_it_in_the_basket"
ENV_NAME = "libero_sim/libero_object_cylinder_pick_up_the_butter_and_place_it_in_the_basket"

CYLINDER_CENTER_XY = (-0.01145, 0.02955)
CYLINDER_RADIUS = 0.02
PLOT_EPSILON = 0.01
KEEP_OUT_RADIUS = CYLINDER_RADIUS + PLOT_EPSILON

TARGET_OBJECT_REGION = (-0.145, -0.265, -0.095, -0.215)
BIN_REGION = (-0.01, 0.25, 0.01, 0.27)

SUCCESS_TRAJECTORY_COLOR = "#39b54a"
FAILURE_TRAJECTORY_COLOR = "#d62728"
CYLINDER_COLOR = "#1f77b4"
TARGET_REGION_COLOR = "#f2c94c"
BASKET_REGION_COLOR = SUCCESS_TRAJECTORY_COLOR

SUPPORTED_SUFFIXES = (".npz", ".npy", ".pt", ".pth")


def _load_trajectory_array(
    trajectory_path: Path,
    array_key: str | None,
) -> np.ndarray:
    suffix = trajectory_path.suffix.lower()

    if suffix == ".npy":
        array = np.load(trajectory_path)
    elif suffix == ".npz":
        npz = np.load(trajectory_path)
        if array_key is None:
            if "states" in npz.files:
                array = npz["states"]
            elif "actions" in npz.files:
                array = npz["actions"]
            elif len(npz.files) == 1:
                array = npz[npz.files[0]]
            else:
                raise ValueError(
                    f"{trajectory_path} contains multiple arrays {npz.files}; pass --array-key."
                )
        else:
            array = npz[array_key]
    elif suffix in {".pt", ".pth"}:
        loaded = torch.load(trajectory_path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            array = loaded.detach().cpu().numpy()
        elif isinstance(loaded, np.ndarray):
            array = loaded
        elif isinstance(loaded, dict):
            if array_key is None:
                if "states" in loaded and isinstance(loaded["states"], (torch.Tensor, np.ndarray)):
                    array = loaded["states"]
                elif "actions" in loaded and isinstance(loaded["actions"], (torch.Tensor, np.ndarray)):
                    array = loaded["actions"]
                else:
                    tensor_keys = [
                        key
                        for key, value in loaded.items()
                        if isinstance(value, (torch.Tensor, np.ndarray))
                    ]
                    if len(tensor_keys) != 1:
                        raise ValueError(
                            f"{trajectory_path} contains multiple candidate arrays {tensor_keys}; pass --array-key."
                        )
                    array = loaded[tensor_keys[0]]
            else:
                array = loaded[array_key]

            if isinstance(array, torch.Tensor):
                array = array.detach().cpu().numpy()
            elif not isinstance(array, np.ndarray):
                raise TypeError(
                    f"Loaded key '{array_key}' is type {type(array)}, expected tensor/ndarray."
                )
        else:
            raise TypeError(
                f"Unsupported object in {trajectory_path}: {type(loaded)}. Expected tensor, ndarray, or dict."
            )
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Use {SUPPORTED_SUFFIXES}."
        )

    return np.asarray(array)


def _load_success_status(trajectory_path: Path) -> bool | None:
    suffix = trajectory_path.suffix.lower()

    if suffix == ".npz":
        npz = np.load(trajectory_path)
        if "success_flags" in npz.files:
            success_flags = np.asarray(npz["success_flags"])
            if success_flags.size > 0:
                return bool(success_flags.reshape(-1)[-1])
        return None

    if suffix in {".pt", ".pth"}:
        loaded = torch.load(trajectory_path, map_location="cpu")
        if isinstance(loaded, dict) and "success_flags" in loaded:
            success_flags = loaded["success_flags"]
            if isinstance(success_flags, torch.Tensor):
                success_flags = success_flags.detach().cpu().numpy()
            success_flags = np.asarray(success_flags)
            if success_flags.size > 0:
                return bool(success_flags.reshape(-1)[-1])

    return None


def _select_trajectory(
    trajectory_array: np.ndarray,
    batch_index: int,
) -> np.ndarray:
    if trajectory_array.ndim == 2:
        selected = trajectory_array
    elif trajectory_array.ndim == 3:
        if batch_index < 0 or batch_index >= trajectory_array.shape[0]:
            raise IndexError(
                f"batch_index {batch_index} is out of range for shape {trajectory_array.shape}."
            )
        selected = trajectory_array[batch_index]
    else:
        raise ValueError(
            f"Expected trajectory array shape [T, D] or [B, T, D], received {trajectory_array.shape}."
        )

    if selected.shape[-1] < 2:
        raise ValueError(
            f"Trajectory must have at least 2 action dimensions for XY plotting, received {selected.shape}."
        )
    return selected


def _collect_trajectory_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported input file type: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    npz_files = [path for path in files if path.suffix.lower() == ".npz"]
    if npz_files:
        return npz_files
    if files:
        return files
    raise FileNotFoundError(f"No trajectory files with suffix {SUPPORTED_SUFFIXES} found in {input_path}")


def _load_xy_trajectories(
    input_path: Path,
    array_key: str | None,
    batch_index: int,
    xy_dims: tuple[int, int],
) -> tuple[list[np.ndarray], list[bool | None], list[Path]]:
    x_dim, y_dim = xy_dims
    trajectories = []
    statuses = []
    files = _collect_trajectory_files(input_path)

    for path in files:
        trajectory_array = _load_trajectory_array(path, array_key)
        trajectory = _select_trajectory(trajectory_array, batch_index)
        if x_dim >= trajectory.shape[-1] or y_dim >= trajectory.shape[-1]:
            raise IndexError(
                f"xy_dims {xy_dims} exceed trajectory dimension {trajectory.shape[-1]} for {path.name}."
            )
        trajectories.append(trajectory[:, [x_dim, y_dim]])
        statuses.append(_load_success_status(path))

    return trajectories, statuses, files


def _region_width_height(region: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = region
    return x1 - x0, y1 - y0


def _make_title_label(input_path: Path) -> str:
    suffix = input_path.stem if input_path.is_file() else input_path.name
    return f"{ENV_NAME}_{suffix}"


def plot_scene_overlay(
    trajectories_xy: list[np.ndarray],
    trajectory_successes: list[bool | None],
    input_label: str,
    output_path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 10.0))

    for index, trajectory_xy in enumerate(trajectories_xy):
        is_success = trajectory_successes[index]
        trajectory_color = SUCCESS_TRAJECTORY_COLOR if is_success is not False else FAILURE_TRAJECTORY_COLOR
        ax.plot(
            trajectory_xy[:, 0],
            trajectory_xy[:, 1],
            color=trajectory_color,
            linewidth=1.6,
            alpha=0.95,
            zorder=3 if is_success is False else 2,
        )
        ax.scatter(
            trajectory_xy[0, 0],
            trajectory_xy[0, 1],
            color=trajectory_color,
            s=18,
            zorder=3,
        )
        ax.scatter(
            trajectory_xy[-1, 0],
            trajectory_xy[-1, 1],
            color=trajectory_color,
            marker="x",
            s=36,
            linewidths=1.4,
            zorder=4,
        )

    target_width, target_height = _region_width_height(TARGET_OBJECT_REGION)
    target_patch = Rectangle(
        (TARGET_OBJECT_REGION[0], TARGET_OBJECT_REGION[1]),
        target_width,
        target_height,
        facecolor=TARGET_REGION_COLOR,
        edgecolor=TARGET_REGION_COLOR,
        linewidth=2.0,
        alpha=0.25,
    )
    ax.add_patch(target_patch)

    bin_width, bin_height = _region_width_height(BIN_REGION)
    basket_patch = Rectangle(
        (BIN_REGION[0], BIN_REGION[1]),
        bin_width,
        bin_height,
        facecolor=BASKET_REGION_COLOR,
        edgecolor=BASKET_REGION_COLOR,
        linewidth=1.5,
        alpha=0.10,
    )
    ax.add_patch(basket_patch)

    keep_out = Circle(
        CYLINDER_CENTER_XY,
        KEEP_OUT_RADIUS,
        facecolor="none",
        edgecolor=CYLINDER_COLOR,
        linestyle="--",
        linewidth=2.0,
    )
    cylinder = Circle(
        CYLINDER_CENTER_XY,
        CYLINDER_RADIUS,
        facecolor=CYLINDER_COLOR,
        edgecolor=CYLINDER_COLOR,
        alpha=0.25,
    )
    ax.add_patch(keep_out)
    ax.add_patch(cylinder)

    ax.text(
        CYLINDER_CENTER_XY[0] + 0.004,
        CYLINDER_CENTER_XY[1] + 0.002,
        "Cylinder",
        color=CYLINDER_COLOR,
        fontsize=10,
    )
    ax.text(
        BIN_REGION[0] + 0.002,
        BIN_REGION[1] + 0.008,
        "Basket Region",
        color=BASKET_REGION_COLOR,
        fontsize=10,
    )
    ax.text(
        TARGET_OBJECT_REGION[0] + 0.003,
        TARGET_OBJECT_REGION[1] + 0.02,
        "Target Object Region",
        color="#d4a017",
        fontsize=10,
    )

    all_x = np.concatenate(
        [
            np.concatenate([trajectory[:, 0] for trajectory in trajectories_xy]),
            np.array(
                [
                    TARGET_OBJECT_REGION[0],
                    TARGET_OBJECT_REGION[2],
                    BIN_REGION[0],
                    BIN_REGION[2],
                    CYLINDER_CENTER_XY[0] - KEEP_OUT_RADIUS,
                    CYLINDER_CENTER_XY[0] + KEEP_OUT_RADIUS,
                ]
            ),
        ]
    )
    all_y = np.concatenate(
        [
            np.concatenate([trajectory[:, 1] for trajectory in trajectories_xy]),
            np.array(
                [
                    TARGET_OBJECT_REGION[1],
                    TARGET_OBJECT_REGION[3],
                    BIN_REGION[1],
                    BIN_REGION[3],
                    CYLINDER_CENTER_XY[1] - KEEP_OUT_RADIUS,
                    CYLINDER_CENTER_XY[1] + KEEP_OUT_RADIUS,
                ]
            ),
        ]
    )

    ax.set_xlim(all_x.min() - 0.01, all_x.max() + 0.01)
    ax.set_ylim(all_y.min() - 0.02, all_y.max() + 0.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"{TITLE}\n"
        f"{input_label}\n"
        f"cylinder center=({CYLINDER_CENTER_XY[0]:.5f}, {CYLINDER_CENTER_XY[1]:.5f}), "
        f"radius={CYLINDER_RADIUS:.2f}, epsilon={PLOT_EPSILON:.02f}",
        fontsize=14,
        pad=4,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Trajectory file or directory containing multiple trajectory files.")
    parser.add_argument("--output", required=True, help="Output image path, e.g. /tmp/vanilla_o.png")
    parser.add_argument("--array-key", default=None, help="Optional key for .npz/.pt dict inputs.")
    parser.add_argument("--batch-index", type=int, default=0, help="Batch index for [B, T, D] arrays.")
    parser.add_argument("--xy-dims", type=int, nargs=2, default=(0, 1), metavar=("X_DIM", "Y_DIM"))
    parser.add_argument("--show", action="store_true", help="Display the figure after saving.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_path = Path(args.input_path)
    output_path = Path(args.output)

    trajectories_xy, trajectory_successes, files = _load_xy_trajectories(
        input_path=input_path,
        array_key=args.array_key,
        batch_index=args.batch_index,
        xy_dims=tuple(args.xy_dims),
    )

    input_label = _make_title_label(input_path)
    logging.info(f"Task suite: {TASK_SUITE_NAME}")
    logging.info(f"Task name: {TASK_NAME}")
    logging.info(f"Env label: {ENV_NAME}")
    logging.info(f"Input path: {input_path}")
    logging.info(f"Found {len(files)} trajectory files")
    logging.info(f"Using xy_dims: {tuple(args.xy_dims)}")
    logging.info(f"Output path: {output_path}")

    plot_scene_overlay(
        trajectories_xy=trajectories_xy,
        trajectory_successes=trajectory_successes,
        input_label=input_label,
        output_path=output_path,
        show=args.show,
    )
    logging.info(f"Saved XY plot: {output_path}")


if __name__ == "__main__":
    main()
