# Getting started

[Project overview](../README.md) · [Configuration reference](reference.md)

## Environment

Run all module commands from the repository root. No editable package installation is needed.

```bash
conda env create -f src/ToyExamples/triangle_obstacles_2d/environment.yml
conda activate triangle_obstacles_2d
```

The environment specifies Python 3.11 and installs PyTorch, torchvision, NumPy, SciPy, Matplotlib, tqdm, einops, and OmegaConf. It includes the PyTorch CUDA 12.8 extra package index; it is not a dependency lockfile. Use a PyTorch build compatible with your hardware. Both toys share this environment. Their training and inference entry points use `cfg.device` when CUDA is available and otherwise use CPU.

## Triangle obstacles

The environment has six training-time disks arranged in rows of three, two, and one. Demonstrations run from a fixed start toward a goal line. Additional constraints are introduced only at inference.

Optionally inspect a small demonstration set first:

```bash
python -m src.ToyExamples.triangle_obstacles_2d.data \
  --num_demos 256 --plot_samples 16 --device cpu
```

This writes a dataset preview under `src/ToyExamples/triangle_obstacles_2d/results/`. The standalone preview does not populate the training cache. Unlike training and inference, its device argument defaults to `cuda`; explicitly use `--device cpu` on CPU machines.

Train and compare the samplers:

```bash
python -m src.ToyExamples.triangle_obstacles_2d.train \
  --config src/ToyExamples/triangle_obstacles_2d/configs/default.yaml
python -m src.ToyExamples.triangle_obstacles_2d.infer \
  --scenario cluttered --num_samples 16
```

Other scenarios reuse the same checkpoint:

```bash
for scenario in top_left top_right both_hard; do
  python -m src.ToyExamples.triangle_obstacles_2d.infer --scenario "$scenario"
done
```

## Square and non-convex obstacles

This model learns demonstrations around a small square, with both start and goal fixed. It needs its own checkpoint:

```bash
python -m src.ToyExamples.square_obstacle_2d.data \
  --num_demos 256 --plot_samples 16 --device cpu
python -m src.ToyExamples.square_obstacle_2d.train
python -m src.ToyExamples.square_obstacle_2d.infer \
  --scenario inverted_c --num_samples 16
python -m src.ToyExamples.square_obstacle_2d.infer \
  --scenario inverted_u --num_samples 16
```

`inverted_c` is the inference default and uses an upper-half annulus. `inverted_u` uses boxes. `square` reuses the training obstacle as a constraint. Although `none` is accepted by the parser, the guided samplers currently fail when differentiating its constant zero cost; use `sample_vanilla` directly for a constraint-free comparison.

## Outputs and checkpoints

Each example writes under its own directory:

| Output | Purpose |
| --- | --- |
| `data_cache/demos.npz` (triangle) or `data_cache/demos_wide.npz` (square) | Cached optimized training demonstrations |
| `checkpoints/epNNNN.pt` and `checkpoints/latest.pt` | Model and EMA weights, epoch, and configuration |
| `results/<scenario>.png` and `.pdf` | Three-panel sampler comparison |
| Standard output | Training progress or inference metrics |

Inference prefers EMA weights. Its model and diffusion settings come from the selected YAML, not automatically from the checkpoint's stored configuration. Keep those settings consistent. Checkpoints do not include optimizer state, and the training CLI has no resume option.

Use an explicit checkpoint and output path when needed:

```bash
python -m src.ToyExamples.triangle_obstacles_2d.infer \
  --ckpt src/ToyExamples/triangle_obstacles_2d/checkpoints/latest.pt \
  --scenario cluttered --num_samples 4 --seed 0 \
  --out /tmp/bayesfp-cluttered.png
```

Collision and constraint-violation rates are fractions of **waypoints**, averaged across trajectories. They do not measure the fraction of entirely collision-free paths or check segments between waypoints. Triangle goal success checks the final point against the goal line with tolerance `0.04`; square goal success uses distance `< 0.05` from the fixed goal. Square sampling pins that goal, so goal success largely reflects endpoint enforcement.

## Short checks and troubleshooting

Use `train --epochs 1 --out /tmp/bayesfp-checkpoints` for a short training check, then pass its `latest.pt` to inference with `--num_samples 1`. This still generates the full default dataset on a cache miss and is not a trained reproduction. To reduce that cost, copy the YAML and lower `data.num_demos`, `data.opt_iters`, and `fkc.num_particles`; assign a separate cache path and pass the same `--config` to training and inference.

| Symptom | Action |
| --- | --- |
| Missing `omegaconf` or other imports | Activate the environment and complete its dependency installation. |
| Module or relative-path errors | Run from the repository root with `python -m src.ToyExamples...`. |
| Missing `latest.pt` | Train that example or supply `--ckpt`. |
| CUDA unavailable in the dataset preview | Pass `--device cpu`. |
| Out of memory | Reduce training batch size or FKC particle count in a copied YAML. |
| Changed data settings appear ineffective | Use a new cache path; existing caches are loaded without checking generation settings. |
| Figure output directory ignores YAML changes | Use `infer --out`; `infer.out_dir` is not read by the entry point. |
| Sample count ignores YAML changes | Use `--num_samples`; the CLI default is 16, independently of `infer.num_samples`. |

For headless plotting, prefix a command with `MPLBACKEND=Agg`. Record the commit, configuration, checkpoint, and seed when comparing runs; hardware and library versions can affect results.
