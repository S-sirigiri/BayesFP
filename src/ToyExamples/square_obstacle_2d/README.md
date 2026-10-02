# Square obstacles: non-convex 2D planning

A self-contained BayesFP toy example trained on demonstrations around a small square. Start and goal are fixed and enforced during sampling. Inference can replace the constraint with an upper-half annulus (`inverted_c`, the default) or a box-based inverted U (`inverted_u`).

Use the shared `triangle_obstacles_2d` Conda environment. From the repository root:

```bash
conda env create -f src/ToyExamples/triangle_obstacles_2d/environment.yml
conda activate triangle_obstacles_2d
python -m src.ToyExamples.square_obstacle_2d.data --num_demos 256 --plot_samples 16 --device cpu
python -m src.ToyExamples.square_obstacle_2d.train
python -m src.ToyExamples.square_obstacle_2d.infer --scenario inverted_c --num_samples 16
python -m src.ToyExamples.square_obstacle_2d.infer --scenario inverted_u --num_samples 16
```

Training generates demonstrations and writes this example's `checkpoints/latest.pt`. Inference writes PNG/PDF comparisons under `results/` and prints waypoint collision/violation rates and endpoint goal success. It compares vanilla DDPM, linear-combination guidance, and FKC.

The `square` scenario uses the training square as an inference constraint. Use `--scenario none` to disable inference constraints for all three samplers; see the [usage guide](../../../docs/getting-started.md).

See the [project README](../../../README.md) and [configuration/API reference](../../../docs/reference.md) for shared setup and sampler details.
