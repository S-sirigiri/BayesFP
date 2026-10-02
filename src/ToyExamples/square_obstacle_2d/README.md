# square_obstacle_2d

Second toy example: vertical 2D path planning from a fixed **start** to a
fixed **goal** point directly above. The diffusion model is trained on
demonstrations that detour around a small **square** at the midpoint. At
inference time the square is replaced by a much wider **inverted-U** as
the FKC / linear_combo constraint, forcing samples to take a much larger
lateral detour than anything seen during training.

Uses the same conda env as `triangle_obstacles_2d` (`triangle_obstacles_2d`).

## 1. Sanity-check the dataset

```bash
python -m src.ToyExamples.square_obstacle_2d.data --num_demos 256 --plot_samples 16
```

Writes `results/dataset_sanity.png`. Demos go from start (black square) to
goal (green star) and detour around the small red square at the centre.

## 2. Train

```bash
python -m src.ToyExamples.square_obstacle_2d.train
```

First time: runs ~4096 batched trajectory optimisations, then trains a
~22M-parameter unconditional 1D U-Net for 250 epochs (~10 min on GPU).

## 3. Run inference (replace square with inverted-U)

```bash
python -m src.ToyExamples.square_obstacle_2d.infer --num_samples 16
```

Writes `results/inverted_u.png` with three panels (vanilla, linear_combo,
fkc) and prints metrics (`square_collision_rate`, `constraint_violation_rate`,
`goal_reach_rate`).

## Layout

```
src/ToyExamples/square_obstacle_2d/
├── env.py            # BoxRegion + SquareObstacleEnv (square, inverted-U)
├── cost.py           # analytical box-violation cost
├── data.py           # batched-traj-opt demo generator
├── model.py          # unconditional 1D UNet
├── diffusion.py      # cosine-schedule DDPM
├── samplers.py       # vanilla / linear_combo / fkc (start+goal inpainting)
├── viz.py            # square + inverted-U + start/goal markers
├── train.py
├── infer.py
├── configs/default.yaml
├── checkpoints/      # populated by train.py
└── results/          # populated by data.py / infer.py
```
