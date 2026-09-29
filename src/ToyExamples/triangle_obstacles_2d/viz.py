"""Matplotlib helpers for plotting the environment and trajectories.

Color convention (swap of the paper, by user request):
    - Training-time obstacles (6 disks) ........ BLUE filled circles
    - Inference-time constraints (cones, etc.) . RED shaded regions
    - Goal line ................................ GREEN (end-to-end)
    - Trajectories ............................. dark blue lines
    - Start point .............................. black square marker
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from .env import (CircleRegion, ShadedRegion, TriangleObstacleEnv,
                   TriangleRegion)


# ---------------------------------------------------------------------------
# Global font: Nimbus Roman No9 L (with sane fallbacks). TrueType embedding
# (fonttype=42) keeps PDF text selectable / copy-pasteable.
# ---------------------------------------------------------------------------
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = [
    "Nimbus Roman No9 L",
    "Nimbus Roman",
    "Times New Roman",
    "Times",
    "DejaVu Serif",
]
mpl.rcParams["mathtext.fontset"] = "stix"     # serif math glyphs
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _draw_obstacles(ax, env: TriangleObstacleEnv) -> None:
    for cx, cy in env.obstacle_centers:
        ax.add_patch(Circle((cx, cy), env.obstacle_radius,
                             facecolor="#1f4ec5", edgecolor="#0a2a82",
                             linewidth=0.8, zorder=3))


def _draw_constraint(ax, region: ShadedRegion, *, alpha: float = 0.45) -> None:
    if isinstance(region, CircleRegion):
        ax.add_patch(Circle((region.cx, region.cy), region.r,
                             facecolor="#d33b3b", edgecolor="#7a1414",
                             alpha=alpha, linewidth=0.6, zorder=2))
    elif isinstance(region, TriangleRegion):
        ax.add_patch(Polygon(region._verts, closed=True,
                              facecolor="#d33b3b", edgecolor="#7a1414",
                              alpha=alpha, linewidth=0.6, zorder=2))
    else:
        raise TypeError(type(region))


def _draw_goal_line(ax, env: TriangleObstacleEnv) -> None:
    # End-to-end horizontal line across the full workspace width.
    ax.plot([-1.05, 1.05],
            [env.goal_y, env.goal_y],
            color="#2c8a2c", linewidth=4.0, zorder=4, solid_capstyle="round")


def _draw_start(ax, env: TriangleObstacleEnv) -> None:
    sx, sy = env.start
    ax.plot([sx], [sy], marker="s", color="black",
            markersize=8, markeredgecolor="black", zorder=5)


def _draw_trajectories(ax, trajs: np.ndarray, *,
                        color: str = "#0a2353", lw: float = 1.1,
                        alpha: float = 0.85) -> None:
    """trajs: (N, T, 2) in (x, y)."""
    for tr in trajs:
        ax.plot(tr[:, 0], tr[:, 1], color=color, linewidth=lw,
                alpha=alpha, zorder=6)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def plot_env(ax, env: TriangleObstacleEnv,
             *, show_constraints: bool = True) -> None:
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    if show_constraints:
        for r in env.inference_constraints:
            _draw_constraint(ax, r)
    _draw_obstacles(ax, env)
    _draw_goal_line(ax, env)
    _draw_start(ax, env)


def plot_panels(env: TriangleObstacleEnv,
                panels: Sequence[tuple[str, np.ndarray]],
                out_path: str,
                suptitle: str | None = None,
                show_constraints: bool = True,
                font_size: float = 11.0,
                panel_title_size: float | None = None,
                suptitle_size: float | None = None) -> None:
    """Draw a row of panels, each with title + trajectories.

    ``font_size`` is the base for everything (ticks, default text). The two
    overrides take precedence on the titles when given; otherwise they default
    to ``font_size`` and ``font_size + 1`` respectively.
    """
    n = len(panels)
    pt_size = font_size if panel_title_size is None else panel_title_size
    st_size = (font_size + 1.0) if suptitle_size is None else suptitle_size
    # Apply the base size to anything the user didn't explicitly override.
    rc_overrides = {
        "font.size": font_size,
        "axes.titlesize": pt_size,
        "figure.titlesize": st_size,
    }
    with mpl.rc_context(rc_overrides):
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.4), squeeze=False)
        for ax, (title, trajs) in zip(axes[0], panels):
            plot_env(ax, env, show_constraints=show_constraints)
            _draw_trajectories(ax, trajs)
            ax.set_title(title, fontsize=pt_size)
        if suptitle:
            fig.suptitle(suptitle, fontsize=st_size)
        fig.tight_layout()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Always write both .png (for quick previews) and .pdf (selectable
        # text / vector graphics) regardless of the extension the caller
        # asked for.
        stem = out.with_suffix("")
        fig.savefig(str(stem) + ".png", dpi=180, bbox_inches="tight")
        fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
        plt.close(fig)
