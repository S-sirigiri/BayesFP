"""Offline validation for the calibrated LIBERO constraint proxy.

Replays every 8-step chunk from a real rollout through three trajectory
models and checks that the calibrated proxy flags the same cylinder
violations as the recorded real trajectory:

    naive:      p_k = p_0 + cumsum(alpha * action_delta_k)
    calibrated: p_k = p_0 + cumsum(gain * alpha * action_delta_k)
    real:       states[k+1:k+1+N, :3]

where alpha = 0.05 (OSC_POSE output scale) and
gain = [0.166, 0.257, 0.233] (per-axis first-order tracking gains).

For each 8-step window we compare the minimum XY distance to the
cylinder for naive, calibrated, and real; the success criterion is that
the calibrated proxy and the real trajectory agree on which chunks
penetrate the cylinder, with sub-centimeter endpoint residual.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CYLINDER_CENTER = np.array([-0.01145, 0.02955], dtype=np.float64)
CYLINDER_RADIUS = 0.02
ALPHA = 0.05                                        # output_max per pos axis
TRACKING_GAIN = np.array([0.166, 0.257, 0.233])     # calibrated per-axis gain
N_ACTION_STEPS = 8                                  # executed chunk length


def _chunk_starts(num_steps: int, chunk: int) -> np.ndarray:
    return np.arange(0, num_steps - chunk + 1, chunk)


def _proxy_from_chunk(
    current_position: np.ndarray,
    action_chunk: np.ndarray,
    gain: np.ndarray | float,
) -> np.ndarray:
    scaled_delta = action_chunk[:, :3] * ALPHA * gain
    return current_position[None, :] + np.cumsum(scaled_delta, axis=0)


def _min_xy_distance(trajectory: np.ndarray) -> float:
    delta = trajectory[:, :2] - CYLINDER_CENTER
    return float(np.sqrt((delta ** 2).sum(axis=1)).min())


def _any_inside(trajectory: np.ndarray) -> bool:
    delta = trajectory[:, :2] - CYLINDER_CENTER
    return bool((np.sqrt((delta ** 2).sum(axis=1)) < CYLINDER_RADIUS).any())


def validate_one(path: Path) -> dict:
    data = np.load(path)
    actions = np.asarray(data["actions"], dtype=np.float64)
    states = np.asarray(data["states"], dtype=np.float64)
    num_steps = min(actions.shape[0], states.shape[0] - 1)

    chunk_starts = _chunk_starts(num_steps, N_ACTION_STEPS)
    per_chunk = []
    endpoint_residuals = []
    for start in chunk_starts:
        end = start + N_ACTION_STEPS
        real_traj = states[start + 1 : end + 1, :3]
        if real_traj.shape[0] < N_ACTION_STEPS:
            break
        current_position = states[start, :3]
        action_chunk = actions[start:end]

        naive_traj = _proxy_from_chunk(current_position, action_chunk, gain=1.0)
        calibrated_traj = _proxy_from_chunk(current_position, action_chunk, gain=TRACKING_GAIN)

        real_min = _min_xy_distance(real_traj)
        naive_min = _min_xy_distance(naive_traj)
        calibrated_min = _min_xy_distance(calibrated_traj)

        endpoint_residuals.append(np.linalg.norm(calibrated_traj[-1] - real_traj[-1]))

        per_chunk.append({
            "start": int(start),
            "real_min": real_min,
            "naive_min": naive_min,
            "calibrated_min": calibrated_min,
            "real_inside": _any_inside(real_traj),
            "naive_inside": _any_inside(naive_traj),
            "calibrated_inside": _any_inside(calibrated_traj),
        })

    return {
        "path": path,
        "per_chunk": per_chunk,
        "endpoint_residual_mean": float(np.mean(endpoint_residuals)) if endpoint_residuals else float("nan"),
        "endpoint_residual_max": float(np.max(endpoint_residuals)) if endpoint_residuals else float("nan"),
    }


def _summarise(result: dict) -> None:
    chunks_inside_real = [c for c in result["per_chunk"] if c["real_inside"]]
    chunks_inside_naive = [c for c in result["per_chunk"] if c["naive_inside"]]
    chunks_inside_calib = [c for c in result["per_chunk"] if c["calibrated_inside"]]

    real_set = {c["start"] for c in chunks_inside_real}
    naive_set = {c["start"] for c in chunks_inside_naive}
    calib_set = {c["start"] for c in chunks_inside_calib}

    missed_by_naive = sorted(real_set - naive_set)
    caught_by_calib = sorted(real_set & calib_set)
    false_alarms_calib = sorted(calib_set - real_set)

    print(f"file: {result['path']}")
    print(f"  chunks evaluated:      {len(result['per_chunk'])}")
    print(f"  chunks inside (real):  {len(real_set):3d}  starts={sorted(real_set)}")
    print(f"  chunks inside (naive): {len(naive_set):3d}  starts={sorted(naive_set)}")
    print(f"  chunks inside (calib): {len(calib_set):3d}  starts={sorted(calib_set)}")
    print(f"  missed by naive:       {len(missed_by_naive):3d}  starts={missed_by_naive}")
    print(f"  caught by calibrated:  {len(caught_by_calib):3d}  starts={caught_by_calib}")
    print(f"  calib false alarms:    {len(false_alarms_calib):3d}  starts={false_alarms_calib}")
    print(
        f"  endpoint residual:     mean={result['endpoint_residual_mean']:.4f} m  "
        f"max={result['endpoint_residual_max']:.4f} m"
    )

    if chunks_inside_real:
        print("  chunks where real trajectory was inside cylinder:")
        for chunk in chunks_inside_real:
            print(
                f"    start={chunk['start']:4d}  real_min={chunk['real_min']:.4f}"
                f"  naive_min={chunk['naive_min']:.4f}"
                f"  calib_min={chunk['calibrated_min']:.4f}"
                f"  calib_inside={chunk['calibrated_inside']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="NPZ rollout files to validate against.")
    args = parser.parse_args()

    overall_real = 0
    overall_naive_caught = 0
    overall_calib_caught = 0
    max_residual = 0.0

    for path_str in args.paths:
        path = Path(path_str)
        result = validate_one(path)
        _summarise(result)
        print()

        for chunk in result["per_chunk"]:
            if chunk["real_inside"]:
                overall_real += 1
                if chunk["naive_inside"]:
                    overall_naive_caught += 1
                if chunk["calibrated_inside"]:
                    overall_calib_caught += 1
        if result["endpoint_residual_max"] == result["endpoint_residual_max"]:
            max_residual = max(max_residual, result["endpoint_residual_max"])

    print("=" * 60)
    print(f"overall chunks inside (real): {overall_real}")
    print(f"  caught by naive proxy:      {overall_naive_caught}")
    print(f"  caught by calibrated proxy: {overall_calib_caught}")
    print(f"  max endpoint residual:      {max_residual:.4f} m")


if __name__ == "__main__":
    main()
