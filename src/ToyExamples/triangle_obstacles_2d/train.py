"""Training script for the unconditional 2D trajectory diffusion model."""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .data import (LengthBucketBatchSampler, TriangleObstacleTrajectoryDataset,
                    collate_fixed_len)
from .diffusion import DDPM
from .env import TriangleObstacleEnv
from .model import UNet1D


def _ema_update(ema_model, model, decay):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for b_ema, b in zip(ema_model.buffers(), model.buffers()):
            b_ema.copy_(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "configs/default.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.out is not None:
        cfg.train.ckpt_dir = args.out

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # Use the training scenario for the env (the obstacle disks are scenario-
    # independent; the inference_constraints aren't read by the dataset).
    env = TriangleObstacleEnv(scenario="cluttered")

    ds = TriangleObstacleTrajectoryDataset(
        env=env,
        num_demos=cfg.data.num_demos,
        trajs_per_demo=cfg.data.trajs_per_demo,
        traj_lens=list(cfg.data.traj_lens),
        base_traj_len=cfg.data.base_traj_len,
        clearance=cfg.data.clearance,
        opt_iters=cfg.data.opt_iters,
        opt_lr=cfg.data.opt_lr,
        w_obs=cfg.data.w_obs,
        w_smooth=cfg.data.w_smooth,
        init_perturb=cfg.data.init_perturb,
        cost_filter_quantile=cfg.data.cost_filter_quantile,
        device=str(device),
        seed=cfg.seed,
        cache_path=str(Path(cfg.data.cache_path)),
    )
    print(f"[train] dataset: {ds.meta.num_demos} demos, "
          f"{len(ds)} samples (trajs_per_demo={cfg.data.trajs_per_demo})")

    sampler = LengthBucketBatchSampler(ds, cfg.train.batch_size, shuffle=True,
                                         seed=cfg.seed)
    loader = DataLoader(ds, batch_sampler=sampler,
                         num_workers=cfg.train.num_workers,
                         collate_fn=collate_fixed_len,
                         persistent_workers=cfg.train.num_workers > 0)

    model = UNet1D(
        in_ch=2,
        base_ch=cfg.model.base_ch,
        channel_mult=tuple(cfg.model.channel_mult),
        cond_dim=cfg.model.cond_dim,
        t_emb_dim=cfg.model.t_emb_dim,
    ).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                               weight_decay=cfg.train.weight_decay)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ddpm = DDPM(num_steps=cfg.diffusion.num_steps,
                 schedule=cfg.diffusion.schedule).to(device)

    ckpt_dir = Path(cfg.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    t0 = time.time()
    for epoch in range(cfg.train.epochs):
        for batch in loader:
            traj = batch["traj"].to(device, non_blocking=True)
            B = traj.shape[0]
            t = torch.randint(0, cfg.diffusion.num_steps, (B,), device=device)
            noise = torch.randn_like(traj)
            x_t = ddpm.q_sample(traj, t, noise)

            with torch.amp.autocast("cuda", enabled=use_amp):
                eps_pred = model(x_t, t)
                # Don't penalise the start point — it's inpainted at sample time.
                mask = torch.ones_like(eps_pred)
                mask[:, 0] = 0.0
                loss = ((eps_pred - noise) ** 2 * mask).sum() / mask.sum()

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            _ema_update(ema, model, cfg.train.ema_decay)

            if step % cfg.train.log_every == 0:
                el = time.time() - t0
                print(f"[train] ep={epoch} step={step} loss={loss.item():.4f} "
                      f"len={traj.shape[1]} elapsed={el:.1f}s")
            step += 1

        if (epoch + 1) % cfg.train.ckpt_every_epochs == 0 or epoch == cfg.train.epochs - 1:
            payload = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "config": OmegaConf.to_container(cfg),
            }
            torch.save(payload, ckpt_dir / f"ep{epoch+1:04d}.pt")
            torch.save(payload, ckpt_dir / "latest.pt")
            print(f"[train] saved checkpoint -> {ckpt_dir / f'ep{epoch+1:04d}.pt'}")

    print(f"[train] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
