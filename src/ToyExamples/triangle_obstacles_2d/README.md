# Triangle obstacles: 2D diffusion planning

A self-contained BayesFP toy example with six training-time disks arranged in rows of three, two, and one. Trajectories start at a fixed point and end near a goal line. Inference adds scenario-specific constraints without retraining the denoiser.

The environment is adapted from the Avoiding-Cluttered setup in [Joint Model-based Model-free Diffusion for Planning with Constraints](https://arxiv.org/abs/2509.08775). Training disks are blue and inference constraints are red in these plots.

From the repository root:

```bash
conda env create -f src/ToyExamples/triangle_obstacles_2d/environment.yml
conda activate triangle_obstacles_2d
python -m src.ToyExamples.triangle_obstacles_2d.data --num_demos 256 --plot_samples 16 --device cpu
python -m src.ToyExamples.triangle_obstacles_2d.train
python -m src.ToyExamples.triangle_obstacles_2d.infer --scenario cluttered --num_samples 16
```

Inference loads `checkpoints/latest.pt` under this example and writes PNG/PDF comparisons under `results/`. Available scenarios are `cluttered`, `top_left`, `top_right`, and `both_hard`. The samplers compare vanilla DDPM, linear-combination guidance, and FKC with particle resampling.

See the [project README](../../../README.md), [usage guide](../../../docs/getting-started.md), and [configuration/API reference](../../../docs/reference.md) for setup, outputs, metrics, and implementation details.
