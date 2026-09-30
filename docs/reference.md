# Configuration and API reference

[Project overview](../README.md) · [Usage guide](getting-started.md)

## Command-line interface

Both examples expose `data`, `train`, and `infer` modules. Use `python -m src.ToyExamples.<example>.<module> --help` to inspect the parser after installing dependencies.

| Training option | Behavior |
| --- | --- |
| `--config PATH` | YAML configuration; defaults to the example's `configs/default.yaml` |
| `--epochs N` | Override training epochs |
| `--out DIRECTORY` | Override checkpoint directory |

| Inference option | Behavior |
| --- | --- |
| `--config PATH` | YAML configuration |
| `--ckpt PATH` | Defaults to `train.ckpt_dir/latest.pt` |
| `--scenario NAME` | Triangle: `cluttered` (default), `top_left`, `top_right`, `both_hard`; square: `inverted_c` (default), `inverted_u`, `square`, `none` |
| `--num_samples N` | Outputs per sampler; default 16 |
| `--traj_len N` | Override `infer.traj_len` |
| `--out PATH` | Figure path; PNG and PDF share its stem |
| `--seed N` | NumPy and PyTorch inference seed; default 0 |

The square `none` scenario is parser-supported but currently incompatible with cost-gradient computation; see [troubleshooting](getting-started.md#square-and-non-convex-obstacles).

## YAML settings

The source of truth is each example's default YAML: [triangle](../src/ToyExamples/triangle_obstacles_2d/configs/default.yaml), [square](../src/ToyExamples/square_obstacle_2d/configs/default.yaml). Copy a file before tuning. The CLI does not support arbitrary Hydra-style overrides.

| Setting | Triangle / square defaults | Meaning |
| --- | --- | --- |
| `seed`, `device` | `0`, `cuda` | Training seed and preferred device |
| `data.num_demos` | `4096` / `8192` | Initial demonstration count before filtering |
| `data.trajs_per_demo` | `4` | Length-resampling multiplicity |
| `data.traj_lens` | `[32, 48, 64, 80, 96]` | Training trajectory lengths |
| `data.opt_iters`, `data.opt_lr` | `600`, `0.01` | Demonstration optimization |
| `data.cache_path` | Example-specific | Existing caches bypass generation |
| `model.base_ch`, `model.channel_mult` | `96`, `[1, 2, 4]` | U-Net width and scale multipliers |
| `diffusion.num_steps`, `diffusion.schedule` | `100`, `cosine` | DDPM schedule |
| `train.epochs`, `train.batch_size` | `250`, `128` | Training duration and batch size |
| `train.lr`, `train.ema_decay` | `2e-4`, `0.999` | AdamW learning rate and EMA decay |
| `train.ckpt_every_epochs` | `25` | Also saves on the last epoch |
| `train.amp` | `true` | Mixed precision enabled only on CUDA |
| `infer.traj_len` | `64` | Inference waypoint count |
| `viz` | Example-specific | Panel titles, overall title, and font sizes |

### FKC settings

These values are from the YAML files, which override the dataclass defaults.

| `fkc` key | Triangle / square defaults | Meaning |
| --- | --- | --- |
| `guidance_weight` | `200.0` / `0.7` | Positive guidance strength; also used by linear-combination sampling |
| `beta_anneal` | `true` / `false` | Ramp guidance with denoising progress |
| `beta_pow` | `3.0` | Annealing power |
| `num_particles` | `128` | Population size per FKC output |
| `resample_ess_frac` | `0.5` | ESS threshold as a fraction of population size |
| `weight_mode` | `girsanov` | `girsanov` or `energy` weight update |
| `active_window` | `[0.3, 0.95]` | Denoising-progress interval allowing resampling |
| `clearance` | `0.0` | Offset applied to already-clipped region violations |
| `return_best` | `true` | Sort final population by increasing constraint cost |

Positive `clearance` currently adds an offset **after** the region violation has been clipped to zero. It does not expand obstacle geometry or create a repulsive gradient outside an obstacle. Keep that distinction in mind when designing margins.

## Python interfaces

Each example defines its own `UNet1D`, `DDPM`, environment class, and sampler functions. Import all of them from the same example. Given an initialized model, DDPM, and environment:

```python
from src.ToyExamples.triangle_obstacles_2d.samplers import FKCConfig, sample_fkc

# model, ddpm, and env must already be initialized;
# model and ddpm must be on the requested device.
particles, info = sample_fkc(
    model, ddpm, traj_len=64, env=env,
    cfg=FKCConfig(num_particles=32), device="cpu",
)
best_trajectory = particles[0]  # return_best defaults to True
```

| Interface | Return |
| --- | --- |
| `model(traj, t)` | Noise prediction `(B, T, 2)`; integer diffusion steps `t` have shape `(B,)` |
| `constraint_cost(traj, env, clearance=0.0)` | Per-trajectory costs `(B,)` |
| `constraint_cost_and_grad(traj, env, clearance=0.0)` | Detached scalar **sum** of costs and gradient `(B, T, 2)` |
| `sample_vanilla(...)`, `sample_linear_combo(...)` | Trajectories `(num_samples, T, 2)` |
| `sample_fkc(...)` | Population `(K, T, 2)` and diagnostics dictionary |

Diagnostics contain `resample_steps`, `ess_log`, and `weight_log`. With `return_best=True`, they also contain `final_J`. Those final costs retain the original population order even though returned trajectories are sorted; recompute costs on the returned trajectories if aligned values are needed. Logged weights precede any resampling at that step and are not reordered with the final population.

## Extending the toys

Add geometry and scenario construction in `env.py`, analytical penalties in `cost.py`, and rendering in `viz.py`. Keep costs batched and differentiable with respect to trajectories. A cost with no differentiable dependence on its input needs special handling before calling `torch.autograd.grad`.

Training obstacles and inference constraints are separate. A new inference constraint can reuse the prior; changes to training geometry require a new demonstration cache and retraining. For a new policy family, use its submodule's model interface and normalization conventions rather than assuming the toy `(T, 2)` representation applies.
