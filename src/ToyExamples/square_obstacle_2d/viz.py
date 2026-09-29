"""Matplotlib helpers for the square / inverted-U toy."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Wedge

from .env import AnnulusSegmentRegion, BoxRegion, SquareObstacleEnv


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


def _draw_box(ax, box: BoxRegion, *, facecolor: str, edgecolor: str = "black",
               alpha: float = 1.0, lw: float = 1.5, zorder: int = 3) -> None:
    rect = Rectangle((box.x_min, box.y_min), box.w, box.h,
                      facecolor=facecolor, edgecolor=edgecolor,
                      linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(rect)


def _draw_annulus_segment(ax, region: AnnulusSegmentRegion, *,
                            facecolor: str, edgecolor: str = "black",
                            alpha: float = 1.0, lw: float = 1.5,
                            zorder: int = 3) -> None:
    """Draw the upper-half annulus (theta from 0 to 180 deg)."""
    width = region.r_outer - region.r_inner
    wedge = Wedge((region.cx, region.cy), region.r_outer,
                   theta1=0.0, theta2=180.0, width=width,
                   facecolor=facecolor, edgecolor=edgecolor,
                   linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(wedge)


def _draw_training_square(ax, env: SquareObstacleEnv) -> None:
    _draw_box(ax, env.training_box, facecolor="#d33b3b",
              edgecolor="black", alpha=1.0, lw=1.6, zorder=3)


def _draw_inference_constraints(ax, env: SquareObstacleEnv) -> None:
    for region in env.inference_constraints:
        if isinstance(region, BoxRegion):
            _draw_box(ax, region, facecolor="#d33b3b",
                      edgecolor="black", alpha=1.0, lw=1.6, zorder=3)
        elif isinstance(region, AnnulusSegmentRegion):
            _draw_annulus_segment(ax, region, facecolor="#d33b3b",
                                    edgecolor="black", alpha=1.0,
                                    lw=1.6, zorder=3)
        else:
            raise TypeError(type(region))


def _draw_start(ax, env: SquareObstacleEnv) -> None:
    sx, sy = env.start
    ax.plot([sx], [sy], marker="s", color="black",
            markersize=8, zorder=5)


def _draw_goal(ax, env: SquareObstacleEnv) -> None:
    gx, gy = env.goal
    ax.plot([gx], [gy], marker="*", color="#2c8a2c",
            markersize=18, markeredgecolor="black", markeredgewidth=0.8,
            zorder=5)


def _draw_trajectories(ax, trajs: np.ndarray, *,
                        color: str = "#0a2353", lw: float = 1.1,
                        alpha: float = 0.85) -> None:
    for tr in trajs:
        ax.plot(tr[:, 0], tr[:, 1], color=color, linewidth=lw,
                alpha=alpha, zorder=6)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def plot_env(ax, env: SquareObstacleEnv,
             *, show_training_obstacle: bool = False,
             show_constraints: bool = True) -> None:
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    if show_training_obstacle:
        _draw_training_square(ax, env)
    if show_constraints:
        _draw_inference_constraints(ax, env)
    _draw_start(ax, env)
    _draw_goal(ax, env)


def plot_panels(env: SquareObstacleEnv,
                panels: Sequence[tuple[str, np.ndarray]],
                out_path: str,
                suptitle: str | None = None,
                show_training_obstacle: bool = False,
                show_constraints: bool = True,
                font_size: float = 11.0,
                panel_title_size: float | None = None,
                suptitle_size: float | None = None) -> None:
    """Draw a row of panels, each with title + trajectories.

    ``font_size`` is the base for everything (ticks, default text); the two
    overrides take precedence on the titles when given; otherwise they default
    to ``font_size`` and ``font_size + 1`` respectively.
    """
    n = len(panels)
    pt_size = font_size if panel_title_size is None else panel_title_size
    st_size = (font_size + 1.0) if suptitle_size is None else suptitle_size
    rc_overrides = {
        "font.size": font_size,
        "axes.titlesize": pt_size,
        "figure.titlesize": st_size,
    }
    with mpl.rc_context(rc_overrides):
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.4), squeeze=False)
        for ax, (title, trajs) in zip(axes[0], panels):
            plot_env(ax, env,
                     show_training_obstacle=show_training_obstacle,
                     show_constraints=show_constraints)
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
