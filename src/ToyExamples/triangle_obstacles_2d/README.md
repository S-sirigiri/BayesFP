# triangle_obstacles_2d

A self-contained 2D toy reproduction of **Figure 11 (Avoiding-Cluttered)** of
the JM2D paper *"Joint Model-based Model-free Diffusion for Planning with
Constraints"* (Jung, Mishra, Arachchige, Chen, Xu, Kousik — CoRL 2025,
[arxiv 2509.08775](https://arxiv.org/abs/2509.08775)).

The diffusion model only sees demonstrations that avoid the **9 fixed obstacle
disks** in a 3×3 wedge pattern (drawn **blue** here). At inference time we add
the *Cluttered* scenario constraints (drawn **red** here): two triangular
cones along the wedge edges, one shaded circle in the middle, and three
shaded halos around the bottom row of obstacles. Inference-time guidance via
**FKC** (Feynman-Kac corrector) and **linear_combo** (classifier-style guided
diffusion) — both ported from `src/diffusion_policy` and `src/maze_fkc` —
steers samples around the new constraints without retraining.

## Color convention (swap of the paper)

| Element | Paper | This repo |
|---|---|---|
| Training-time obstacles (9 disks) | red | **blue** filled circles |
| Inference-time constraints | light blue | **red** shaded regions |
| Goal line | green | green |

## Setup

```bash
conda env create -f src/ToyExamples/triangle_obstacles_2d/environment.yml
conda activate triangle_obstacles_2d
```

(CPU-only is fine; the toy is small.)

## 1. Sanity-check the dataset

```bash
python -m src.ToyExamples.triangle_obstacles_2d.data \
       --num_demos 256 --plot_samples 16
```

Generates a few hundred demos via batched trajectory optimisation and writes
`results/dataset_sanity.png`. Demonstrations should clearly avoid the 9 blue
disks; the red shaded regions are drawn for reference but the demos do *not*
respect them (those constraints are inference-only).

## 2. Train

```bash
python -m src.ToyExamples.triangle_obstacles_2d.train \
       --config src/ToyExamples/triangle_obstacles_2d/configs/default.yaml
```

First time: runs ~4096 batched trajectory optimisations (a few minutes on
GPU, ~10–15 min on CPU) and caches the demos to
`data_cache/demos.npz`. Then trains the unconditional diffusion U-Net.

## 3. Reproduce Figure 11

```bash
python -m src.ToyExamples.triangle_obstacles_2d.infer --num_samples 16
```

Writes `results/cluttered.png` with three side-by-side panels (vanilla,
linear_combo, fkc) and prints a metrics table.

To produce the other Figure 10(b) scenarios reusing the same checkpoint:

```bash
for s in top_left top_right both_hard; do
  python -m src.ToyExamples.triangle_obstacles_2d.infer --scenario "$s"
done
```

## Layout

```
src/ToyExamples/triangle_obstacles_2d/
├── README.md
├── environment.yml
├── configs/default.yaml
├── env.py            # geometry: wedge, 9 obstacles, scenario presets, SDFs
├── cost.py           # inference-time J(x) over scenario constraints
├── data.py           # batched-traj-opt demo generator + Dataset
├── model.py          # unconditional 1D UNet
├── diffusion.py      # cosine-schedule DDPM
├── samplers.py       # vanilla / linear_combo / fkc (start-only inpaint)
├── viz.py            # matplotlib panels
├── train.py
├── infer.py
├── checkpoints/      # populated by train.py
└── results/          # populated by data.py / infer.py
```
