# BayesFP

Code for **BayesFP: Posterior Estimation for Flow-Based Policies via Feynman-Kac Sampling**, by Sreevardhan Sirigiri, Weiming Zhi, and Fabio Ramos.

[Paper](https://arxiv.org/abs/2606.21014) · [Getting started](docs/getting-started.md) · [Method and implementation](docs/method.md) · [Configuration and API](docs/reference.md) · [Robot integrations](docs/integrations.md)

BayesFP guides pretrained generative policies toward inference-time objectives and constraints without retraining the policy. The paper treats the policy as a prior and uses a cost-derived likelihood to define a posterior, sampled using Feynman–Kac corrections. It covers diffusion policies and an extension to flow-matching policies. See the [paper](https://arxiv.org/abs/2606.21014) for the derivation and experiments.

## What is in this repository?

| Path | Contents |
| --- | --- |
| [`src/ToyExamples/triangle_obstacles_2d`](src/ToyExamples/triangle_obstacles_2d) | 2D diffusion planning through disks with additional inference-time constraints |
| [`src/ToyExamples/square_obstacle_2d`](src/ToyExamples/square_obstacle_2d) | 2D diffusion planning around a square, with non-convex inference-time obstacles |
| `src/diffusion_policy`, `src/Isaac-GR00T`, `src/openpi`, `src/RoboLab` | Policy and robotics repositories pinned as Git submodules |
| [`scripts/run_all_libero_10.sh`](scripts/run_all_libero_10.sh) | LIBERO evaluation client for an already-running GR00T policy server |
| [`test_examples/toy_problems`](test_examples/toy_problems) | Additional Gaussian-mixture experiment and FKC reference code |
| [`papers`](papers), [`preliminary_results`](preliminary_results) | Supporting references and exploratory results |

The two main toy examples are self-contained and do not require the robot submodules. This is a research repository with separate environments, not a single installable Python package. Robot checkpoints, simulator assets, and policy environments must be set up separately.

## Quick start: 2D planning

Run these commands from the repository root after cloning. Both examples share the supplied Python 3.11 Conda environment:

```bash
conda env create -f src/ToyExamples/triangle_obstacles_2d/environment.yml
conda activate triangle_obstacles_2d

# Generate demonstrations and train the base diffusion model.
python -m src.ToyExamples.triangle_obstacles_2d.train

# Compare no guidance, linear-combination guidance, and BayesFP/FKC.
python -m src.ToyExamples.triangle_obstacles_2d.infer \
  --scenario cluttered --num_samples 16
```

Training writes `src/ToyExamples/triangle_obstacles_2d/checkpoints/latest.pt`. Inference writes `cluttered.png` and `cluttered.pdf` under that example's `results/` directory and prints collision, constraint-violation, and goal-reaching metrics. A trained checkpoint is required; none is tracked for these examples.

Training and inference fall back to CPU when CUDA is unavailable. Default training runs for 250 epochs, and each displayed FKC trajectory uses a separate population of 128 particles, so a full run can be substantial. See [getting started](docs/getting-started.md) for dataset previews, shorter checks, and the square example.

## Documentation

- [Getting started](docs/getting-started.md): installation, training, inference, outputs, and troubleshooting.
- [Method and implementation](docs/method.md): posterior formulation and how the toy samplers implement guidance and resampling.
- [Configuration and API](docs/reference.md): CLI options, YAML settings, tensor shapes, and extension points.
- [Robot integrations](docs/integrations.md): submodule setup and LIBERO prerequisites.

## Citation

```bibtex
@misc{sirigiri2026bayesfp,
  title         = {BayesFP: Posterior Estimation for Flow-Based Policies via Feynman-Kac Sampling},
  author        = {Sreevardhan Sirigiri and Weiming Zhi and Fabio Ramos},
  year          = {2026},
  eprint        = {2606.21014},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2606.21014},
  url           = {https://arxiv.org/abs/2606.21014}
}
```
