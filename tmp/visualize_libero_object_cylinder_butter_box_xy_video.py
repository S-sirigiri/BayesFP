from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

from visualize_libero_object_cylinder_butter_xy import (
    BIN_REGION,
    BASKET_REGION_COLOR,
    CYLINDER_CENTER_XY,
    CYLINDER_COLOR,
    CYLINDER_RADIUS,
    ENV_NAME,
    FAILURE_TRAJECTORY_COLOR,
    KEEP_OUT_RADIUS,
    PLOT_EPSILON,
    SUCCESS_TRAJECTORY_COLOR,
    SUPPORTED_SUFFIXES,
    TARGET_OBJECT_REGION,
    TARGET_REGION_COLOR,
    TASK_NAME,
    TASK_SUITE_NAME,
    _collect_trajectory_files,
    _load_trajectory_array,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GR00T_SRC_ROOT = REPO_ROOT / "src" / "Isaac-GR00T"
if str(GR00T_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(GR00T_SRC_ROOT))

from gr00t.model.gr00t_n1d6.cost_constraint_objective import CostConstraintObjective


TITLE = "Panda Palm-Box XY Trajectory Video"
UNKNOWN_TRAJECTORY_COLOR = "#7f7f7f"
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
DEFAULT_FIGSIZE = (7.5, 10.0)
DEFAULT_DPI = 120
DEFAULT_FPS = 10


def _region_width_height(region: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = region
    return x1 - x0, y1 - y0


def _load_success_statuses(
    trajectory_path: Path,
    num_trajectories: int,
) -> list[bool | None]:
    suffix = trajectory_path.suffix.lower()

    if suffix == ".npz":
        with np.load(trajectory_path) as npz:
            success_flags = npz.get("success_flags")
            if success_flags is None:
                return [None] * num_trajectories
            return _reshape_success_statuses(np.asarray(success_flags), num_trajectories)

    if suffix in {".pt", ".pth"}:
        import torch

        loaded = torch.load(trajectory_path, map_location="cpu")
        if isinstance(loaded, dict) and "success_flags" in loaded:
            success_flags = loaded["success_flags"]
            if isinstance(success_flags, torch.Tensor):
                success_flags = success_flags.detach().cpu().numpy()
            return _reshape_success_statuses(np.asarray(success_flags), num_trajectories)

    return [None] * num_trajectories


def _reshape_success_statuses(success_flags: np.ndarray, num_trajectories: int) -> list[bool | None]:
    if success_flags.size == 0:
        return [None] * num_trajectories

    if num_trajectories == 1:
        return [bool(success_flags.reshape(-1)[-1])]

    if success_flags.ndim >= 1 and success_flags.shape[0] == num_trajectories:
        return [bool(np.asarray(success_flags[index]).reshape(-1)[-1]) for index in range(num_trajectories)]

    if success_flags.ndim >= 2 and success_flags.shape[-1] == num_trajectories:
        return [bool(np.asarray(success_flags[..., index]).reshape(-1)[-1]) for index in range(num_trajectories)]

    if success_flags.size == num_trajectories:
        return [bool(value) for value in success_flags.reshape(-1)]

    logging.warning(
        "Ambiguous success_flags shape %s for %d trajectories; defaulting to unknown status.",
        success_flags.shape,
        num_trajectories,
    )
    return [None] * num_trajectories


def _axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(axis_angle, dtype=np.float64).reshape(3, 1))
    return rotation_matrix.astype(np.float32)


def _palm_box_local_corners() -> np.ndarray:
    offset = np.asarray(CostConstraintObjective.LIBERO_PANDA_PALM_BOX_OFFSET, dtype=np.float32)
    half_extents = np.asarray(
        CostConstraintObjective.LIBERO_PANDA_PALM_BOX_HALF_EXTENTS, dtype=np.float32
    )

    corners = []
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            for z_sign in (-1.0, 1.0):
                corners.append(offset + np.array([x_sign, y_sign, z_sign], dtype=np.float32) * half_extents)
    return np.stack(corners, axis=0)


PALM_BOX_LOCAL_CORNERS = _palm_box_local_corners()
PALM_BOX_LOCAL_OFFSET = np.asarray(CostConstraintObjective.LIBERO_PANDA_PALM_BOX_OFFSET, dtype=np.float32)


def _projected_palm_box_sequence(states: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    centers_xy = []
    footprint_hulls = []

    for state in states:
        position = np.asarray(state[:3], dtype=np.float32)
        rotation_vector = np.asarray(state[3:6], dtype=np.float32)
        rotation_matrix = _axis_angle_to_rotation_matrix(rotation_vector)

        center_xy = position[:2] + (rotation_matrix @ PALM_BOX_LOCAL_OFFSET)[:2]
        world_corners = position[None, :] + (rotation_matrix @ PALM_BOX_LOCAL_CORNERS.T).T
        projected_xy = np.ascontiguousarray(world_corners[:, :2].astype(np.float32))
        hull = cv2.convexHull(projected_xy).reshape(-1, 2)

        centers_xy.append(center_xy)
        footprint_hulls.append(hull)

    return np.stack(centers_xy, axis=0), footprint_hulls


def _split_state_trajectories(
    trajectory_array: np.ndarray,
) -> tuple[list[np.ndarray], bool]:
    if trajectory_array.ndim == 2:
        trajectories = [np.asarray(trajectory_array)]
        is_batched_source = False
    elif trajectory_array.ndim == 3:
        trajectories = [np.asarray(trajectory_array[index]) for index in range(trajectory_array.shape[0])]
        is_batched_source = True
    else:
        raise ValueError(
            f"Expected state array shape [T, D] or [B, T, D], received {trajectory_array.shape}."
        )

    for trajectory in trajectories:
        if trajectory.shape[-1] < 6:
            raise ValueError(
                "State trajectory must contain at least 6 dims: "
                f"[x, y, z, roll, pitch, yaw, ...]. Received shape {trajectory.shape}."
            )
    return trajectories, is_batched_source


def _trajectory_color(success_flag: bool | None) -> str:
    if success_flag is True:
        return SUCCESS_TRAJECTORY_COLOR
    if success_flag is False:
        return FAILURE_TRAJECTORY_COLOR
    return UNKNOWN_TRAJECTORY_COLOR


def _compute_axis_limits(footprint_hulls: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    all_hull_points = np.concatenate(footprint_hulls, axis=0)
    scene_points = np.array(
        [
            [TARGET_OBJECT_REGION[0], TARGET_OBJECT_REGION[1]],
            [TARGET_OBJECT_REGION[2], TARGET_OBJECT_REGION[3]],
            [BIN_REGION[0], BIN_REGION[1]],
            [BIN_REGION[2], BIN_REGION[3]],
            [CYLINDER_CENTER_XY[0] - KEEP_OUT_RADIUS, CYLINDER_CENTER_XY[1] - KEEP_OUT_RADIUS],
            [CYLINDER_CENTER_XY[0] + KEEP_OUT_RADIUS, CYLINDER_CENTER_XY[1] + KEEP_OUT_RADIUS],
        ],
        dtype=np.float32,
    )
    all_points = np.concatenate([all_hull_points, scene_points], axis=0)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    return (float(x_min - 0.01), float(x_max + 0.01)), (float(y_min - 0.02), float(y_max + 0.02))


def _configure_scene_axes(
    ax,
    trajectory_label: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    target_width, target_height = _region_width_height(TARGET_OBJECT_REGION)
    ax.add_patch(
        Rectangle(
            (TARGET_OBJECT_REGION[0], TARGET_OBJECT_REGION[1]),
            target_width,
            target_height,
            facecolor=TARGET_REGION_COLOR,
            edgecolor=TARGET_REGION_COLOR,
            linewidth=2.0,
            alpha=0.25,
        )
    )

    bin_width, bin_height = _region_width_height(BIN_REGION)
    ax.add_patch(
        Rectangle(
            (BIN_REGION[0], BIN_REGION[1]),
            bin_width,
            bin_height,
            facecolor=BASKET_REGION_COLOR,
            edgecolor=BASKET_REGION_COLOR,
            linewidth=1.5,
            alpha=0.10,
        )
    )

    ax.add_patch(
        Circle(
            CYLINDER_CENTER_XY,
            KEEP_OUT_RADIUS,
            facecolor="none",
            edgecolor=CYLINDER_COLOR,
            linestyle="--",
            linewidth=2.0,
        )
    )
    ax.add_patch(
        Circle(
            CYLINDER_CENTER_XY,
            CYLINDER_RADIUS,
            facecolor=CYLINDER_COLOR,
            edgecolor=CYLINDER_COLOR,
            alpha=0.25,
        )
    )

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

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"{TITLE}\n"
        f"{trajectory_label}\n"
        f"{ENV_NAME} | cylinder radius={CYLINDER_RADIUS:.2f}, epsilon={PLOT_EPSILON:.02f}",
        fontsize=13,
        pad=4,
    )


def _render_trajectory_video(
    states: np.ndarray,
    success_flag: bool | None,
    trajectory_label: str,
    output_path: Path,
    fps: int,
    figsize: tuple[float, float],
    dpi: int,
    show_frame_index: bool,
) -> None:
    color = _trajectory_color(success_flag)
    centers_xy, footprint_hulls = _projected_palm_box_sequence(states)
    x_limits, y_limits = _compute_axis_limits(footprint_hulls)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _configure_scene_axes(ax, trajectory_label=trajectory_label, x_limits=x_limits, y_limits=y_limits)

    trail_line, = ax.plot([], [], color=color, linewidth=1.8, alpha=0.95, zorder=3)
    start_marker, = ax.plot([], [], marker="o", markersize=5, color=color, linestyle="None", zorder=4)
    current_marker, = ax.plot([], [], marker="o", markersize=6, color=color, linestyle="None", zorder=5)
    end_marker, = ax.plot([], [], marker="x", markersize=8, markeredgewidth=1.6, color=color, linestyle="None", zorder=6)
    footprint_patch = Polygon(
        footprint_hulls[0],
        closed=True,
        facecolor=color,
        edgecolor=color,
        linewidth=1.8,
        alpha=0.25,
        zorder=4,
    )
    ax.add_patch(footprint_patch)

    frame_text = None
    if show_frame_index:
        frame_text = ax.text(
            0.015,
            0.985,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 3.0},
        )

    fig.tight_layout()
    fig.canvas.draw()
    frame_width, frame_height = fig.canvas.get_width_height()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), FOURCC, float(fps), (frame_width, frame_height))
    if not writer.isOpened():
        plt.close(fig)
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        start_marker.set_data([centers_xy[0, 0]], [centers_xy[0, 1]])
        for frame_index, hull_xy in enumerate(footprint_hulls):
            trail_line.set_data(centers_xy[: frame_index + 1, 0], centers_xy[: frame_index + 1, 1])
            current_marker.set_data([centers_xy[frame_index, 0]], [centers_xy[frame_index, 1]])
            footprint_patch.set_xy(hull_xy)

            if frame_index == len(footprint_hulls) - 1:
                end_marker.set_data([centers_xy[-1, 0]], [centers_xy[-1, 1]])
            else:
                end_marker.set_data([], [])

            if frame_text is not None:
                frame_text.set_text(f"frame {frame_index + 1}/{len(footprint_hulls)}")

            fig.canvas.draw()
            frame_rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()
        plt.close(fig)


def _video_output_path(
    output_dir: Path,
    source_path: Path,
    batch_index: int | None,
) -> Path:
    if batch_index is None:
        return output_dir / f"{source_path.stem}.mp4"
    return output_dir / f"{source_path.stem}_b{batch_index:03d}.mp4"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Trajectory file or directory containing saved rollout state files.")
    parser.add_argument("--output-dir", required=True, help="Directory where per-trajectory .mp4 videos will be stored.")
    parser.add_argument("--array-key", default="states", help="Array key for .npz/.pt dict inputs. Defaults to 'states'.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Output video frames per second.")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=DEFAULT_FIGSIZE,
        metavar=("WIDTH", "HEIGHT"),
        help="Matplotlib figure size in inches.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Matplotlib figure DPI.")
    parser.add_argument("--show-frame-index", action="store_true", help="Overlay the frame index on each video frame.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    figsize = tuple(args.figsize)

    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, received {args.fps}.")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be positive, received {args.dpi}.")

    files = _collect_trajectory_files(input_path)
    logging.info("Task suite: %s", TASK_SUITE_NAME)
    logging.info("Task name: %s", TASK_NAME)
    logging.info("Env label: %s", ENV_NAME)
    logging.info("Input path: %s", input_path)
    logging.info("Found %d trajectory files with suffixes %s", len(files), SUPPORTED_SUFFIXES)
    logging.info("Array key: %s", args.array_key)
    logging.info("Output dir: %s", output_dir)

    videos_written = 0
    for path in files:
        trajectory_array = _load_trajectory_array(path, args.array_key)
        trajectories, is_batched_source = _split_state_trajectories(trajectory_array)
        success_statuses = _load_success_statuses(path, len(trajectories))

        for index, states in enumerate(trajectories):
            batch_index = index if is_batched_source else None
            output_path = _video_output_path(output_dir, path, batch_index)
            trajectory_label = path.stem if batch_index is None else f"{path.stem} | batch {index}"
            _render_trajectory_video(
                states=states,
                success_flag=success_statuses[index],
                trajectory_label=trajectory_label,
                output_path=output_path,
                fps=args.fps,
                figsize=(float(figsize[0]), float(figsize[1])),
                dpi=args.dpi,
                show_frame_index=args.show_frame_index,
            )
            videos_written += 1
            logging.info("Saved video: %s", output_path)

    logging.info("Wrote %d trajectory videos to %s", videos_written, output_dir)


if __name__ == "__main__":
    main()
