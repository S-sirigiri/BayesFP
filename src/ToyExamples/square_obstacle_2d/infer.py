"""Reproduce the inverted-U scenario: 3-panel comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .cost import (constraint_violation_rate, goal_reach,
                    square_collision_rate)
from .diffusion import DDPM
from .env import SquareObstacleEnv
from .model import UNet1D
from .samplers import FKCConfig, sample_fkc, sample_linear_combo, sample_vanilla
from .viz import plot_panels


def _load_model(cfg, ckpt_path: str, device: torch.device) -> UNet1D:
    model = UNet1D(
        in_ch=2,
        base_ch=cfg.model.base_ch,
        channel_mult=tuple(cfg.model.channel_mult),
        cond_dim=cfg.model.cond_dim,
        t_emb_dim=cfg.model.t_emb_dim,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(state)
    model.eval()
    return model


def _metrics(name: str, trajs: torch.Tensor,
             env: SquareObstacleEnv) -> dict:
    return {
        "sampler": name,
        "square_collision_rate": float(square_collision_rate(trajs, env).mean()),
        "constraint_violation_rate": float(constraint_violation_rate(trajs, env).mean()),
        "goal_reach_rate": float(goal_reach(trajs, env).mean()),
    }


def _print_table(rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    widths = {k: max(len(k), max(len(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])) for r in rows)) for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    print("  ".join("-" * widths[k] for k in keys))
    for r in rows:
        cells = []
        for k in keys:
            v = r[k]
            cells.append(f"{v:.4f}".ljust(widths[k]) if isinstance(v, float) else str(v).ljust(widths[k]))
        print("  ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=str(Path(__file__).parent / "configs/default.yaml"))
    parser.add_argument("--scenario", type=str, default="inverted_c",
                         choices=["none", "square", "inverted_u", "inverted_c"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--traj_len", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[infer] device={device} scenario={args.scenario}")

    env = SquareObstacleEnv(scenario=args.scenario)

    ckpt_path = args.ckpt or str(Path(cfg.train.ckpt_dir) / "latest.pt")
    print(f"[infer] loading {ckpt_path}")
    model = _load_model(cfg, ckpt_path, device)

    ddpm = DDPM(num_steps=cfg.diffusion.num_steps,
                 schedule=cfg.diffusion.schedule).to(device)

    traj_len = args.traj_len or cfg.infer.traj_len

    print("[infer] sampling vanilla ...")
    vanilla = sample_vanilla(model, ddpm, traj_len=traj_len, env=env,
                              num_samples=args.num_samples, device=device)

    print("[infer] sampling linear_combo ...")
    lc = sample_linear_combo(
        model, ddpm, traj_len=traj_len, env=env,
        guidance_weight=cfg.fkc.guidance_weight,
        beta_anneal=cfg.fkc.beta_anneal,
        beta_pow=cfg.fkc.beta_pow,
        clearance=cfg.fkc.clearance,
        num_samples=args.num_samples, device=device,
    )

    fkc_cfg = FKCConfig(
        guidance_weight=cfg.fkc.guidance_weight,
        beta_anneal=cfg.fkc.beta_anneal,
        beta_pow=cfg.fkc.beta_pow,
        num_particles=cfg.fkc.num_particles,
        resample_ess_frac=cfg.fkc.resample_ess_frac,
        weight_mode=cfg.fkc.weight_mode,
        active_window=tuple(cfg.fkc.active_window),
        clearance=cfg.fkc.clearance,
        return_best=cfg.fkc.return_best,
    )
    print(f"[infer] sampling fkc ({args.num_samples} runs of "
          f"K={fkc_cfg.num_particles} particles) ...")
    fkc_runs = []
    for _ in range(args.num_samples):
        x_pop, _ = sample_fkc(model, ddpm, traj_len=traj_len, env=env,
                                cfg=fkc_cfg, device=device)
        fkc_runs.append(x_pop[0])
    fkc = torch.stack(fkc_runs, dim=0)

    rows = [
        _metrics("vanilla", vanilla, env),
        _metrics("linear_combo", lc, env),
        _metrics("fkc", fkc, env),
    ]
    _print_table(rows)

    out_path = args.out or f"src/ToyExamples/square_obstacle_2d/results/{args.scenario}.png"

    # Read the optional viz block from config (titles + font sizes).
    viz_cfg = cfg.get("viz") or {}
    panel_titles = viz_cfg.get("panel_titles") or {}
    v_title = panel_titles.get("vanilla", "vanilla")
    lc_title = panel_titles.get("linear_combo", "linear_combo")
    fkc_title = panel_titles.get("fkc", "fkc")
    suptitle_fmt = viz_cfg.get(
        "suptitle", "square_obstacle_2d / scenario={scenario}"
    )
    suptitle = suptitle_fmt.format(scenario=args.scenario)
    font_size = float(viz_cfg.get("font_size", 11.0))
    panel_title_size = viz_cfg.get("panel_title_size", None)
    suptitle_size = viz_cfg.get("suptitle_size", None)

    plot_panels(
        env=env,
        panels=[
            (v_title,   vanilla.detach().cpu().numpy()),
            (lc_title,  lc.detach().cpu().numpy()),
            (fkc_title, fkc.detach().cpu().numpy()),
        ],
        out_path=out_path,
        suptitle=suptitle,
        # At inference we hide the training-time square (it's not a constraint
        # any more) and show only the inference-time constraint.
        show_training_obstacle=False,
        show_constraints=True,
        font_size=font_size,
        panel_title_size=(None if panel_title_size is None else float(panel_title_size)),
        suptitle_size=(None if suptitle_size is None else float(suptitle_size)),
    )
    out_stem = str(Path(out_path).with_suffix(""))
    print(f"[infer] wrote {out_stem}.png and {out_stem}.pdf")


if __name__ == "__main__":
    main()
