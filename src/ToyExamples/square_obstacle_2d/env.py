"""Environment for the square-obstacle / inverted-C (or U) toy example.

Setup:

  * **Workspace**: [-1, 1] x [-1, 1].
  * **Start** is a fixed point at the bottom of the workspace.
  * **Goal** is a fixed point at the top of the workspace, directly above
    the start. The straight line connecting them passes through the origin.
  * **Training-time obstacle**: a single small **square** centred on the
    midpoint of the start-goal line. The diffusion model is trained on
    demonstrations that go from start to goal while avoiding this square.
  * **Inference-time constraint**: a much wider curved obstacle replaces
    the square at sampling time. Two shapes are supported:
        - ``inverted_c`` (default): an **upper-half annulus** (semicircular
          ring) — analytically defined.
        - ``inverted_u``: three boxes — top cap + two side arms.

Both constraint types are analytical and differentiable, so ``cost.py``
evaluates the squared-hinge violation directly with autograd — no SDF
approximation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Box (axis-aligned rectangle) primitive
# ---------------------------------------------------------------------------


@dataclass
class BoxRegion:
    """Axis-aligned rectangle: forbidden interior.

    A point ``(x, y)`` is inside the box iff
        |x - cx| <= w / 2  AND  |y - cy| <= h / 2.

    The inside-distance (penetration depth) used by ``cost.py`` is
        h_box(x, y) = min(w/2 - |x - cx|, h/2 - |y - cy|),
    positive inside the box and negative outside. ``relu(h_box)`` is the
    violation that the FKC / linear_combo guidance hinge-penalises.
    """
    cx: float
    cy: float
    w: float    # full width (x-extent)
    h: float    # full height (y-extent)

    @property
    def half_w(self) -> float:
        return self.w * 0.5

    @property
    def half_h(self) -> float:
        return self.h * 0.5

    @property
    def x_min(self) -> float:
        return self.cx - self.half_w

    @property
    def x_max(self) -> float:
        return self.cx + self.half_w

    @property
    def y_min(self) -> float:
        return self.cy - self.half_h

    @property
    def y_max(self) -> float:
        return self.cy + self.half_h


@dataclass
class AnnulusSegmentRegion:
    """Upper-half annulus (inverted-C / semicircular ring): forbidden interior.

    A point ``(x, y)`` is inside iff
        r_inner  <=  || (x, y) - (cx, cy) ||  <=  r_outer    AND    y >= cy.

    The inside-distance (intersection of three half-spaces in distance/y) is
        h(x, y) = min( r_outer - d,  d - r_inner,  y - cy ),
    where ``d = || (x, y) - (cx, cy) ||``. Positive inside, negative outside.
    ``relu(h)`` is the violation that the squared-hinge cost penalises.
    All three terms are differentiable (sub-differentiable on boundaries),
    so autograd flows through cleanly.
    """
    cx: float
    cy: float
    r_inner: float
    r_outer: float


# Type alias for inference-time constraints.
ShadedRegion = BoxRegion | AnnulusSegmentRegion


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class SquareObstacleEnv:
    """Fixed start/goal + scenario-specific box constraints."""

    # Fixed endpoints; a straight line between them runs through (0, 0).
    start: tuple[float, float] = (0.0, -0.70)
    goal: tuple[float, float] = (0.0,  0.70)

    # Training-time square obstacle (centred at the midpoint of start-goal).
    square_center: tuple[float, float] = (0.0, 0.0)
    square_side: float = 0.20

    # Inverted-U (box-decomposition inference scenario) outer dimensions.
    u_outer_half_w: float = 0.18    # outer x-extent: [-w, +w]
    u_top_y: float = 0.18           # top of the cap
    u_bottom_y: float = -0.18       # bottom of the arms
    u_thickness: float = 0.08       # wall thickness (cap + arm width)

    # Inverted-C / upper-half-annulus inference scenario.
    c_center: tuple[float, float] = (0.0, 0.0)
    c_r_outer: float = 0.30
    c_r_inner: float = 0.20

    # Active scenario: "none", "square", "inverted_u" or "inverted_c".
    scenario: str = "inverted_c"

    def __post_init__(self):
        # The training obstacle is always a single box, regardless of which
        # scenario is loaded — it's only used by the dataset, never by the
        # inference cost.
        self.training_box: BoxRegion = BoxRegion(
            cx=self.square_center[0], cy=self.square_center[1],
            w=self.square_side, h=self.square_side,
        )
        self.inference_constraints: list[ShadedRegion] = self._build_scenario(
            self.scenario
        )

    # --------------------------- scenario presets -------------------------

    def _build_scenario(self, name: str) -> list[ShadedRegion]:
        if name == "none":
            return []
        if name == "square":
            # Same as training — useful for sanity-checking guidance.
            return [BoxRegion(
                cx=self.square_center[0], cy=self.square_center[1],
                w=self.square_side, h=self.square_side,
            )]
        if name == "inverted_u":
            # Decompose the U into three axis-aligned boxes.
            tw = self.u_thickness
            top = BoxRegion(
                cx=0.0,
                cy=self.u_top_y - tw * 0.5,
                w=2.0 * self.u_outer_half_w,
                h=tw,
            )
            arm_h = (self.u_top_y - tw) - self.u_bottom_y
            arm_cy = self.u_bottom_y + 0.5 * arm_h
            left_arm = BoxRegion(
                cx=-(self.u_outer_half_w - tw * 0.5),
                cy=arm_cy,
                w=tw,
                h=arm_h,
            )
            right_arm = BoxRegion(
                cx=+(self.u_outer_half_w - tw * 0.5),
                cy=arm_cy,
                w=tw,
                h=arm_h,
            )
            return [top, left_arm, right_arm]
        if name == "inverted_c":
            return [AnnulusSegmentRegion(
                cx=self.c_center[0],
                cy=self.c_center[1],
                r_inner=self.c_r_inner,
                r_outer=self.c_r_outer,
            )]
        raise ValueError(f"Unknown scenario {name!r}")

    @classmethod
    def with_scenario(cls, name: str) -> "SquareObstacleEnv":
        return cls(scenario=name)
