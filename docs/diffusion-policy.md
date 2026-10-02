# Diffusion Policy: RoboMimic obstacle avoidance

[Back to the main README](../README.md)

This experiment applies inference-time guidance to pretrained image-conditioned Diffusion Policy checkpoints on **Can** and **Transport**. The evaluation environments add a cylindrical obstacle to the original RoboMimic task. Policies predict absolute Cartesian poses; guidance penalizes sampled gripper and held-object points inside the obstacle.

## Environment and assets

Follow the **Installation → Simulation** section of the checked-out [Diffusion Policy README](../src/diffusion_policy/README.md). Its [environment specification](../src/diffusion_policy/conda_environment.yaml) pins the simulator and model dependencies. Use that environment for every command below. The README's **Reproducing Simulation Benchmark Results** and **Evaluate Pre-trained Checkpoints** sections cover dataset download and checkpoint selection.

The upstream [training-data index](https://diffusion-policy.cs.columbia.edu/data/training/) provides the RoboMimic image archive. Pretrained image CNN runs are listed under [Can PH](https://diffusion-policy.cs.columbia.edu/data/experiments/image/can_ph/diffusion_policy_cnn/) and [Transport PH](https://diffusion-policy.cs.columbia.edu/data/experiments/image/transport_ph/diffusion_policy_cnn/). `ph` means proficient-human demonstrations. Download a compatible checkpoint and place it at the path below, or supply its actual path to `eval.py`.

All paths below are relative to `src/diffusion_policy`:

| Task | Dataset required even for evaluation | Checkpoint path expected by the comparison script |
| --- | --- | --- |
| Can | `data/robomimic/datasets/can/ph/image_abs.hdf5` | `data/pretrained_ckpts/image/can_ph/diffusion_cnn/best.ckpt` |
| Transport | `data/robomimic/datasets/transport/ph/image_abs.hdf5` | `data/pretrained_ckpts/image/transport_ph/diffusion_cnn/checkpoints/best.ckpt` |

The evaluator reads environment metadata and demonstration initial states from the HDF5 dataset, so a checkpoint alone is insufficient. Use the **absolute-action image** datasets and matching CNN checkpoints: Can has a 10-dimensional action representation and Transport has 20 dimensions, including 6D rotations. Changing `--task` does not convert a relative-action checkpoint or a checkpoint trained for another observation layout. `data/` is ignored by Git; these assets must be obtained separately after cloning.

## Compare unguided, linear guidance, and FKC

Run from the repository root, with the Diffusion Policy environment active:

```bash
cd src/diffusion_policy
python scripts/compare_guidance_modes.py \
  --device cuda:0 \
  --tasks can transport \
  --modes none linear_combo fkc \
  --output data/guidance_mode_compare
```

This launches six evaluations using the checkpoint paths above. To run only Can with FKC, use `--tasks can --modes fkc` and a fresh `--output` directory. The script copies each task's base guidance YAML and changes only `guidance_mode`; it uses `default.yaml` for Can and `transport_can_style_v2.yaml` for Transport.

The available modes are:

| `guidance_mode` | Behavior |
| --- | --- |
| `none` | Original diffusion sampling without the collision objective. |
| `linear_combo` | A single trajectory with a cost-gradient correction during denoising. |
| `fkc` | Cost-guided proposals plus Feynman–Kac particle weighting and resampling. |

The comparison writes generated YAML files to `configs/`, each evaluation to `<task>_<mode>/`, and aggregate results to `summary.csv` and `summary.txt`. It reuses any existing `stdout.log` containing a test summary, without checking whether parameters or checkpoints changed. Use a new output directory for a changed experiment.

## Evaluate a single checkpoint

Both `default.yaml` and `transport_can_style_v2.yaml` currently set `guidance_mode: none`. Create explicit FKC copies before running guided evaluation:

```bash
python - <<'PY'
from pathlib import Path
import yaml

root = Path("diffusion_policy/config/guidance")
out = Path("data/guidance_configs")
out.mkdir(parents=True, exist_ok=True)
for task, source in (
    ("can", "default.yaml"),
    ("transport", "transport_can_style_v2.yaml"),
):
    config = yaml.safe_load((root / source).read_text())
    config["guidance_mode"] = "fkc"
    (out / f"{task}_fkc.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

Evaluate Can:

```bash
python eval.py \
  --checkpoint data/pretrained_ckpts/image/can_ph/diffusion_cnn/best.ckpt \
  --output_dir data/can_fkc_eval \
  --device cuda:0 \
  --task can_with_obs_image_abs \
  --guidance-config data/guidance_configs/can_fkc.yaml
```

Evaluate Transport:

```bash
python eval.py \
  --checkpoint data/pretrained_ckpts/image/transport_ph/diffusion_cnn/checkpoints/best.ckpt \
  --output_dir data/transport_fkc_eval \
  --device cuda:0 \
  --task transport_with_obs_image_abs \
  --guidance-config data/guidance_configs/transport_fkc.yaml
```

For an unguided evaluation, pass the corresponding original YAML instead. For linear guidance, change the copied YAML's `guidance_mode` to `linear_combo`. The `--task` flag chooses the simulator task; the guidance YAML's `task` chooses the cost objective. Keep them matched (`can` or `transport`). The evaluator prompts before reusing an existing output directory.

## Configuration and results

The task configurations are [Can](../src/diffusion_policy/diffusion_policy/config/task/can_with_obs_image_abs.yaml) and [Transport](../src/diffusion_policy/diffusion_policy/config/task/transport_with_obs_image_abs.yaml). Each runs 6 demonstration initial states and 50 test episodes, with test seeds starting at `100000`. Can allows 400 steps and Transport 700 steps. Both configurations launch 28 parallel environments; their comments estimate roughly 1 GB per environment and recommend a machine with 16 cores and 64 GB RAM. FKC also multiplies the denoising batch by its particle count, increasing GPU memory use. To reduce simulator memory, lower `env_runner.n_envs` in the task YAML; `eval.py` does not accept Hydra overrides for that value.

Tune inference using a copied guidance YAML:

| Parameter | Purpose |
| --- | --- |
| `beta_guid` | Strength of the cost tilt. |
| `c_ineq` | Weight of the collision penalty. |
| `safety_margin` | Obstacle padding in meters. |
| `include_object` | Include held-object sample points in the collision objective. |
| `fkc_num_particles` | Number of FKC particles per environment; the provided configurations use 32. |
| `fkc_resample_every` | Resampling interval in denoising steps. |
| `fkc_tmin`, `fkc_tmax` | Normalized diffusion-time window for FKC weight updates and resampling. |
| `hammer_holder_gate`, `gripper_box` | Transport-specific held-object gating and gripper geometry. |

Use the actual YAML values when recording an experiment. Some comments in `transport_can_style_v2.yaml` describe older settings; the current values include `hammer_holder_gate: ungated` and `gripper_box: can_style`. `transport_default.yaml` is a separate configuration with different geometry settings, so it is not interchangeable with the comparison script's base config.

Each evaluation produces:

- `eval_log.json`: aggregate and per-episode metrics, plus video paths.
- `media/`: rollout videos for the configured visualization episodes.
- `eef_paths.png`: simulated end-effector paths projected into the XY plane.

Use `test/mean_success` for task success and `test/obstacle_contact_rate` for the fraction of episodes with at least one obstacle contact. The runner also logs `test/n_obstacle_contacts`, `test/n_episodes`, and per-seed values. Success and contact are independent: an episode can complete the task after touching the obstacle. The contact rate is an episode-level statistic, not the fraction of individual timesteps in collision. `test/mean_score` averages maximum rewards and should not be assumed to equal success for other tasks.

## Extending and reproducing experiments

The sampler is implemented in [inference.py](../src/diffusion_policy/diffusion_policy/policy/inference.py). The [Can objective](../src/diffusion_policy/diffusion_policy/policy/cost_constraint_objective.py) and [Transport objective](../src/diffusion_policy/diffusion_policy/policy/cost_constraint_objective_transport.py) define the collision penalties. Transport obtains its cylinder geometry from the task YAML. Can's objective currently hardcodes the cylinder geometry in world coordinates; if you move or resize the Can obstacle, update both the task YAML and objective so the simulator and guidance agree.

For training the underlying policy, follow **Running for a single seed** in the [Diffusion Policy README](../src/diffusion_policy/README.md). Guidance is applied at inference time and does not require policy retraining.

The maintained comparison script covers `none`, `linear_combo`, and `fkc`; it does not reproduce every baseline in the paper's table. The older `sweep_guidance.py` expects a `Cylinder contacts` summary while the current runner prints `Obstacle contacts`, so its summary parser needs updating before use. Record the checkpoint filename, submodule commit, generated guidance YAML, task YAML, and output metrics with any reported result. The examples above document the current code configuration; they do not establish exact reproduction of a particular paper row without the corresponding checkpoint and experiment settings.
