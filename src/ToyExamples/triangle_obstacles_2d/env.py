"""Environment for the 2D triangle-obstacles path-planning toy example.

Reproduces the geometry of Figure 11 (Avoiding-Cluttered) of the JM2D paper
(arxiv 2509.08775), using the layout the user supplied as the reference
mockup. The diffusion model lives in normalised coordinates
[-1, 1] x [-1, 1]; the y axis points up.

Geometry (matches the user's reference image):

  * The yellow **wedge** is an inverted triangle: narrow point (apex) at the
    BOTTOM (where the start sits), and a wide horizontal base at the TOP
    (where the goal line sits). Trajectories travel upward through this
    expanding corridor.

  * The **6 training-time obstacles** form a downward-pointing triangle
    (3 + 2 + 1) inside the corridor. The widest row sits near the goal,
    one disk sits closest to the start. Demonstrations *avoid* these, so
    the diffusion model implicitly learns to route around them. We draw
    them BLUE (paper draws them red).

  * The **inference-time constraints** (added at sampling time only, never
    seen during training) are: two large triangular cones outside the
    wedge on the left/right, one small circle in the middle of the
    corridor, and three small halos around the **bottom three obstacles**
    (the row of 1 + the row of 2 — i.e. the obstacles closest to the
    start). We draw them RED (paper draws them light blue).

Four named scenario presets mirror Figure 10(b): ``top_left``, ``top_right``,
``both_hard``, ``cluttered`` (the default).

Each ``ShadedRegion`` stores the analytical pieces needed to evaluate its
constraint function ``h(x)`` directly (no SDF approximation):
  * ``CircleRegion`` keeps ``(cx, cy, r)``.
  * ``TriangleRegion`` keeps the three vertices and outward-facing edge
    normals.
``cost.py`` reads these fields and computes the squared-hinge violation
``relu(h(x))^2`` analytically per constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Shaded region primitives (used for inference-time constraints)
# ---------------------------------------------------------------------------


@dataclass
class CircleRegion:
    """A filled disk: forbidden interior. Constraint h(x) = r - ||x - c||."""
    cx: float
    cy: float
    r: float


@dataclass
class TriangleRegion:
    """A filled triangle, given by 3 vertices in any order.

    ``__post_init__`` orients the polygon CCW and pre-computes outward edge
    normals, so ``cost.py`` can evaluate the inside-distance analytically:

        h(x) = -max_i  n_i . (x - v_i)        # positive inside the triangle
    """
    v0: tuple[float, float]
    v1: tuple[float, float]
    v2: tuple[float, float]

    def __post_init__(self):
        v = np.array([self.v0, self.v1, self.v2], dtype=np.float32)
        if _signed_area(v) < 0:
            v = np.ascontiguousarray(v[::-1])
        edges = np.roll(v, -1, axis=0) - v
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
        self._verts = np.ascontiguousarray(v)
        self._normals = np.ascontiguousarray(normals)


def _signed_area(v: np.ndarray) -> float:
    x = v[:, 0]
    y = v[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# Type alias.
ShadedRegion = CircleRegion | TriangleRegion


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


# Obstacle layout: a downward-pointing triangle of 3 + 2 + 1 disks.
# Listed top-to-bottom so it's easy to slice "the bottom 3" later.
_OBSTACLE_CENTERS = np.array([
    # Top row (closest to the goal line) — 3 disks
    [-0.50,  0.55], [0.00,  0.55], [0.50,  0.55],
    # Middle row — 2 disks
    [-0.27,  0.10], [0.27,  0.10],
    # Bottom row (closest to the start) — 1 disk (apex of obstacle triangle)
    [0.00, -0.40],
], dtype=np.float32)


@dataclass
class TriangleObstacleEnv:
    """Fixed 2D environment with obstacles + scenario-specific constraints."""

    obstacle_centers: np.ndarray = field(
        default_factory=lambda: _OBSTACLE_CENTERS.copy()
    )
    obstacle_radius: float = 0.07

    # Wedge corridor: apex at the bottom, wide base at the top.
    wedge_apex: tuple[float, float] = (0.0, -0.95)
    wedge_base_left: tuple[float, float] = (-0.85, 0.95)
    wedge_base_right: tuple[float, float] = (0.85, 0.95)

    # Fixed start point (apex of the wedge).
    start: tuple[float, float] = (0.0, -0.92)
    # Goal is a horizontal line segment that spans the FULL workspace width
    # — end-to-end, not just the wedge corridor. The diffusion model is
    # trained on demos whose endpoints are anywhere along this line, so the
    # learned distribution covers the whole top edge. At inference time the
    # cone constraints (when present) pull samples back into the wedge.
    goal_y: float = 0.92
    goal_x_min: float = -0.95
    goal_x_max: float = 0.95

    # Inference-time scenario name.
    scenario: str = "cluttered"

    def __post_init__(self):
        self.inference_constraints: list[ShadedRegion] = self._build_scenario(
            self.scenario
        )

    # --------------------------- scenario presets -------------------------

    def _build_scenario(self, name: str) -> list[ShadedRegion]:
        # Two triangular cones in the upper-left and upper-right corners of
        # the workspace. They are NARROW at the bottom corner and WIDEN
        # upward — the original maze_fkc-style definition.
        cone_left = TriangleRegion(
            v0=(-1.05, -1.05),
            v1=(-1.05,  1.05),
            v2=(-0.10,  1.05),
        )
        cone_right = TriangleRegion(
            v0=( 1.05, -1.05),
            v1=( 0.10,  1.05),
            v2=( 1.05,  1.05),
        )
        # Small middle inference circle, just below the top row of obstacles
        # (acts as an extra hazard mid-corridor).
        middle_circle = CircleRegion(cx=0.0, cy=0.32, r=0.08)
        # Halos around the BOTTOM THREE obstacles (the row of 2 + the apex).
        # These are the disks closest to the start.
        bottom_three_idx = [3, 4, 5]   # rows of 2 + 1 in our layout above
        bottom_halos = [
            CircleRegion(cx=float(self.obstacle_centers[i, 0]),
                         cy=float(self.obstacle_centers[i, 1]),
                         r=self.obstacle_radius + 0.13)
            for i in bottom_three_idx
        ]
        if name == "top_left":
            return [cone_left]
        if name == "top_right":
            return [cone_right]
        if name == "both_hard":
            return [cone_left, cone_right]
        if name == "cluttered":
            return [cone_left, cone_right, middle_circle, *bottom_halos]
        raise ValueError(f"Unknown scenario {name!r}")

    @classmethod
    def with_scenario(cls, name: str) -> "TriangleObstacleEnv":
        return cls(scenario=name)

    # --------------------------- demo planning utilities ------------------

    def occupancy_grid(self, resolution: int = 96,
                       padding: float = 0.0) -> np.ndarray:
        ys = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        occ = np.zeros((resolution, resolution), dtype=bool)
        r2 = (self.obstacle_radius + padding) ** 2
        for cx, cy in self.obstacle_centers:
            occ |= (X - cx) ** 2 + (Y - cy) ** 2 <= r2
        return occ

    def grid_xy(self, resolution: int = 96) -> tuple[np.ndarray, np.ndarray]:
        ys = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
        return xs, ys

    def sample_goal_point(self, rng: np.random.Generator) -> np.ndarray:
        x = rng.uniform(self.goal_x_min, self.goal_x_max)
        return np.array([x, self.goal_y], dtype=np.float32)
