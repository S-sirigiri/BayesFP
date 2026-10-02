# 2D toy examples

[Back to the main README](../README.md)

These examples train unconditional trajectory diffusion models, then compare `vanilla`, `linear_combo`, and `fkc` sampling under new inference-time obstacles. Both examples use analytic obstacle costs and run without a robot simulator.

## Environment and working directory

Follow the existing [toy setup instructions](../src/ToyExamples/triangle_obstacles_2d/README.md#setup). Both examples share that environment; its dependencies are recorded in [environment.yml](../src/ToyExamples/triangle_obstacles_2d/environment.yml).

Run every command below from the **BayesFP repository root** with that environment active. The paths in the default configurations are relative to this directory. Training and inference fall back to CPU when CUDA is unavailable; the default training runs and 128-particle inference are more practical on a GPU. The optional dataset preview commands accept `--device cpu` explicitly.

## Cluttered environment

The learned demonstrations avoid fixed blue circular obstacles. At inference, red triangular/circular regions add constraints that were absent from training.

Optionally inspect generated demonstrations:

```bash
python -m src.ToyExamples.triangle_obstacles_2d.data \
  --num_demos 256 --plot_samples 16
```

Train the model and run the three samplers:

```bash
python -m src.ToyExamples.triangle_obstacles_2d.train \
  --config src/ToyExamples/triangle_obstacles_2d/configs/default.yaml

python -m src.ToyExamples.triangle_obstacles_2d.infer \
  --scenario cluttered --num_samples 16 --seed 0
```

Inference loads `src/ToyExamples/triangle_obstacles_2d/checkpoints/latest.pt` by default. It writes `results/cluttered.png` and `results/cluttered.pdf` inside that toy directory.

Other supported scenarios use the same checkpoint:

```bash
for scenario in top_left top_right both_hard; do
  python -m src.ToyExamples.triangle_obstacles_2d.infer \
    --scenario "$scenario" --num_samples 16 --seed 0
done
```

## Non-convex obstacle environment

The demonstrations detour around a small square, with varied lateral arcs. At inference, a larger non-convex obstacle changes the feasible routes. The current inference default is `inverted_c`; specify the scenario explicitly to select the intended figure.

Optionally inspect demonstrations:

```bash
python -m src.ToyExamples.square_obstacle_2d.data \
  --scenario inverted_c --num_demos 256 --plot_samples 16
```

Train a separate model and run inference:

```bash
python -m src.ToyExamples.square_obstacle_2d.train \
  --config src/ToyExamples/square_obstacle_2d/configs/default.yaml

python -m src.ToyExamples.square_obstacle_2d.infer \
  --scenario inverted_c --num_samples 16 --seed 0
```

Inference loads `src/ToyExamples/square_obstacle_2d/checkpoints/latest.pt` and writes `results/inverted_c.png` and `results/inverted_c.pdf` inside that directory. To use the inverted-U obstacle instead:

```bash
python -m src.ToyExamples.square_obstacle_2d.infer \
  --scenario inverted_u --num_samples 16 --seed 0
```

Use `--scenario square` to evaluate the training obstacle as the inference constraint. The parser also accepts `none`, but the current three-sampler comparison does not handle the empty cost's gradient in its guided modes, so that scenario fails during guided inference.

## Configuration and checkpoints

The defaults are [triangle YAML](../src/ToyExamples/triangle_obstacles_2d/configs/default.yaml) and [square YAML](../src/ToyExamples/square_obstacle_2d/configs/default.yaml). Training generates demonstrations if the configured cache is missing, then saves periodic `epXXXX.pt` checkpoints and `latest.pt`. The shipped settings use 250 epochs; the triangle dataset requests 4096 demonstrations and the square dataset requests 8192. An existing cache is reused, so changing data-generation settings requires a new `data.cache_path` to generate a new dataset.

For a short training check with a separate output directory:

```bash
python -m src.ToyExamples.triangle_obstacles_2d.train \
  --epochs 1 --out /tmp/bayesfp-toy-checkpoints
```

This checks the training path; a one-epoch model is not a reproduction checkpoint. Select any compatible trained checkpoint and output location explicitly:

```bash
python -m src.ToyExamples.triangle_obstacles_2d.infer \
  --ckpt /absolute/path/to/latest.pt \
  --scenario cluttered --num_samples 4 --seed 0 \
  --out /tmp/bayesfp-cluttered.png
```

Use the model architecture/configuration associated with the checkpoint via `--config` when it differs from the default. Inference prefers EMA weights stored in the checkpoint. It does not automatically replace the supplied YAML with the checkpoint's saved configuration.

| YAML setting | Purpose |
| --- | --- |
| `device` | Preferred device, with CPU fallback if CUDA is unavailable |
| `data.cache_path` | Cached training demonstrations |
| `train.ckpt_dir` | Training checkpoint directory |
| `diffusion.num_steps` | DDPM denoising steps |
| `fkc.guidance_weight` | Positive cost-guidance strength; larger values push harder toward lower cost |
| `fkc.beta_anneal`, `fkc.beta_pow` | Guidance-strength schedule |
| `fkc.num_particles` | Particles in each FKC population, default 128 |
| `fkc.resample_ess_frac` | Effective-sample-size threshold for resampling |
| `fkc.active_window` | Interval of reverse-sampling progress in which resampling is allowed |
| `fkc.weight_mode` | `girsanov` or `energy` reweighting |
| `fkc.return_best` | Sort final particles by cost; enabled in the default configurations |
| `viz` | Plot titles and font sizes |

`--num_samples` controls the number of displayed trajectories per sampler. For FKC, each trajectory is obtained from a separate population of `fkc.num_particles` particles. With the default `return_best: true`, the figure uses the lowest-cost final particle from each population. This toy visualization selects a best particle; it should not be interpreted as a weighted posterior draw. Reducing `--num_samples` reduces the number of populations, while reducing `fkc.num_particles` reduces each population's memory/computation cost.

The scripts use `--out` for figure paths; without it they write a scenario-named figure under the toy's `results/` directory. The current inference scripts do not read `infer.out_dir` when choosing that path. Repeating a scenario with the same output path overwrites its PNG/PDF.

## Metrics and outputs

Both inference scripts print one row per sampler. Rates are fractions between 0 and 1:

- `obstacle_collision_rate` (triangle) or `square_collision_rate` (square): mean fraction of trajectory waypoints inside the training obstacle geometry.
- `constraint_violation_rate`: mean fraction of waypoints violating at least one inference-time constraint.
- `goal_reach_rate`: fraction of trajectories that reach the goal according to the environment's endpoint criterion.

Lower collision/violation values and higher goal-reaching values are preferable. These are toy waypoint metrics, not the per-rollout collision rates used in some robot experiments. The square example inpaints both start and goal; the triangle example inpaints only the start, so goal-reaching has different interpretation between the examples.

Dataset previews write `results/dataset_sanity.png` and its PDF counterpart. Training caches live in `data_cache/`, checkpoints in `checkpoints/`, and inference figures in `results/` unless overridden. Checkpoints generated on one machine are not a dependency supplied automatically to a fresh clone; train before inference or provide `--ckpt`.

## Common issues

- **Module not found / missing file:** run from the BayesFP root and activate the environment linked above.
- **Missing `latest.pt`:** complete training first or provide `--ckpt` pointing to a compatible checkpoint.
- **Dataset settings appear unchanged:** choose a new cache path when changing demonstration generation settings.
- **CUDA memory pressure:** reduce `fkc.num_particles` in a copied YAML, and pass that file with `--config`.
- **Running without a display:** set `MPLBACKEND=Agg` before launching the plotting commands.

Use `python -m src.ToyExamples.triangle_obstacles_2d.infer --help` or the corresponding square module to inspect the available CLI options.
