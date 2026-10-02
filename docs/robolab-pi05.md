# RoboLab with π0.5

[Back to the main README](../README.md)

This experiment runs the DROID joint-position π0.5 policy in RoboLab, with optional Feynman–Kac collision guidance. The policy server runs in the OpenPI environment; the simulator and evaluation client run in the RoboLab environment. Guided runs also start a separate `nvblox_torch` process to build an ESDF from scene-object bounding boxes.

## Environment setup and checkpoint

Follow the existing repositories' instructions:

- [RoboLab installation and requirements](../src/RoboLab/README.md#getting-started), including Isaac Sim, Isaac Lab, and assets.
- [OpenPI installation](../src/openpi/README.md#installation) for the model server.
- [RoboLab OpenPI client setup](../src/RoboLab/docs/inference.md#openpi-pi0--pi0-fast--pi05). Use the sibling `src/openpi` checkout for the server and client package so this repository's guidance changes are available.
- For guided runs, follow the [NVIDIA nvblox installation documentation](https://nvidia-isaac.github.io/nvblox/) for `nvblox_torch`. The local [sidecar smoke test](../src/RoboLab/scripts/test_nvblox_sidecar.py) references [nvblox v0.0.9](https://github.com/nvidia-isaac/nvblox/releases/tag/v0.0.9). Keep its interpreter separate from the RoboLab and OpenPI environments; its default location is `src/RoboLab/.venv_nvblox_sidecar/bin/python`. Both the RoboLab and sidecar interpreters must provide `numpy`, `msgpack`, and `msgpack_numpy`; the sidecar also needs CUDA-enabled `torch` and `nvblox_torch`.

The commands below assume these environments are already configured. Activate each repository's environment in its terminal before running its commands.

Use `pi05_droid_jointpos` with `gs://openpi-assets-simeval/pi05_droid_jointpos`, as specified by RoboLab's inference guide. This configuration converts the first seven predicted joint deltas into absolute joint-position commands. The ordinary `pi05_droid` configuration has a different output transform and should not be substituted for this example. A downloaded local checkpoint can replace the cloud URI; it must include the JAX `params/` directory and `assets/droid/norm_stats.json`.

OpenPI downloads cloud checkpoints on first use. Guided inference currently supports the **JAX** path and Franka Panda kinematics. A checkpoint containing `model.safetensors` selects PyTorch, which does not support this guidance implementation.

## Start the vanilla policy server

In terminal 1, starting at the BayesFP root:

```bash
cd src/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

The memory fraction leaves room for Isaac Sim when the server and simulator share a GPU. Adjust it and the number of simulation environments to fit your hardware.

## Run one evaluation

In terminal 2, starting at the BayesFP root, use the RoboLab environment:

```bash
cd src/RoboLab
python examples/policy/run_eval.py \
  --policy pi05 \
  --task FoodPacking2CansTask \
  --headless \
  --num-envs 1 \
  --seed 0 \
  --inference-seed 0 \
  --report-collisions \
  --sdf-ooi-exclusion-mode static \
  --sdf-ooi-object-names tomato_soup_can tuna_can \
  --output-folder-name pi05_vanilla_foodpacking_seed0
```

`--report-collisions` works independently of guidance. The exclusion options above keep the manipulation targets out of the collision checks. Use the same exclusions for both vanilla and guided comparisons.

The default connection is `localhost:8000`. For a server on another machine, add `--remote-host HOST --remote-port PORT`, or `--remote-uri wss://HOST` for a full WebSocket URI. π0.5's default open-loop horizon is 15 actions, matching this model configuration; `--open-loop-horizon` overrides it.

## Enable Feynman–Kac guidance

First check the sidecar from the RoboLab directory:

```bash
python scripts/test_nvblox_sidecar.py
```

For a different sidecar interpreter, this check accepts `--sidecar-python /absolute/path/to/python`; evaluation accepts `--nvblox-sidecar-python /absolute/path/to/python`.

Stop the vanilla server and restart terminal 1 from `src/openpi` with the guidance config:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos \
  --policy.fkc-config=configs/fkc/placeholder.yaml
```

In terminal 2, from `src/RoboLab`:

```bash
python examples/policy/run_eval.py \
  --policy pi05 \
  --task FoodPacking2CansTask \
  --headless \
  --num-envs 1 \
  --seed 0 \
  --inference-seed 0 \
  --enable-sdf-guidance \
  --report-collisions \
  --sdf-ooi-exclusion-mode static \
  --sdf-ooi-object-names tomato_soup_can tuna_can \
  --output-folder-name pi05_fkc_foodpacking_seed0
```

An active guided server requires the ESDF fields attached by `--enable-sdf-guidance`. The client launches and closes the sidecar automatically.

The checked-in [example YAML](../src/openpi/configs/fkc/placeholder.yaml) uses `mode: fkc`, four particles, and a squared-hinge collision penalty. It is a starting configuration, not a complete specification for reproducing every paper result. Copy it to a separate file when tuning, pass that path to `--policy.fkc-config`, and restart the server after changes.

| YAML option | Effect |
| --- | --- |
| `mode: vanilla` | Original unguided sampler; ESDF fields are optional. Omitting the YAML also selects vanilla. |
| `mode: linear_combo` | A single particle with a gradient-augmented stochastic drift. |
| `mode: fkc` / `num_particles` | Weighted particles with resampling and a final particle selection. |
| `num_steps` | Denoising steps; `null` resolves to 10 in the current policy wrapper. |
| `w_ineq_value`, `w_ineq_grad`, `beta_strength` | Constraint weighting and guidance strength. |
| `sigma_schedule`, `sigma_scale` | Noise schedule and scale. Use a nonzero schedule and scale for the gradient drift to contribute. |
| `collision.*` | Robot sample points, penalty type, and clearance margin in metres. |
| `fk.*`, `dynamics.*` | Robot transforms and the actuator rollout used to predict the physical trajectory. |

The terminal cost is currently zero; changing `cost.target_xyz` does not add a reaching objective. See the [configuration schema](../src/openpi/src/openpi/fkc/config.py) and [sampler](../src/openpi/src/openpi/fkc/sampling.py) for the implemented options.

## Paper tasks and larger runs

The paper evaluates the pretrained DROID policy zero-shot on these four tasks. For the paper's non-target-obstacle convention, set `--sdf-ooi-exclusion-mode static` and pass the corresponding names as separate arguments:

| `--task` | `--sdf-ooi-object-names` |
| --- | --- |
| `FoodPacking2CansTask` | `tomato_soup_can tuna_can` |
| `BBQSauceInBinTask` | `bbq_sauce_bottle bbq_sauce_bottle_01` |
| `HammersInLeftBinTask` | `red_hammer husky_hammer` |
| `TakeMugsOffOfShelfTask` | `ceramic_mug mug` |

The default `dynamic` exclusion treats targets as obstacles until contact indicates a grasp. These four task definitions do not declare target names automatically, so use the explicit overrides above for static exclusion. Avoid commas between names.

Increase `--num-envs` for parallel episodes. Total episodes per task equal `--num-envs × --num-runs`; reduce the parallel count if GPU memory is exhausted. For repeated seeds, rerun the command with explicit `--seed` and `--inference-seed` values and a unique output folder for each seed and method. A repeated output folder resumes completed episodes, so use a new name when changing a checkpoint or guidance settings.

`--task` accepts multiple tasks, and omitting it evaluates all registered benchmark tasks. When using static target exclusions, run the four paper tasks separately with their own names from the table. The existing [seed sweep script](../src/RoboLab/scripts/sweep_seeds.sh) is a task-specific example; review its hard-coded task and target list before adapting it.

## Results and inspection

Results are saved under `src/RoboLab/output/<output-folder-name>/`:

- `episode_results.jsonl`: task success and per-episode metrics.
- Per-environment subdirectories: `run_<index>.hdf5`, environment configuration, and videos according to `--video-mode` (`all` by default, or `viewport`, `sensor`, `none`).
- `collisions_run_<index>.npz`: per-step collision records when `--report-collisions` is enabled.

From `src/RoboLab`, summarize the two example runs:

```bash
python analysis/read_results.py pi05_vanilla_foodpacking_seed0 pi05_fkc_foodpacking_seed0 --timing
python scripts/analyze_collisions.py output/pi05_vanilla_foodpacking_seed0
python scripts/analyze_collisions.py output/pi05_fkc_foodpacking_seed0
```

`read_results.py` aggregates the folders passed to it; summarize each folder separately when you want one table per method. The collision analyzer reports the fraction of logged steps in collision. It uses robot body centres with a configurable radius against obstacle bounding boxes; this is a geometric proxy and differs from the paper's sampled-point ESDF penetration metric. Keep the radius, padding, body filter, exclusions, and voxel size fixed across comparisons.

See RoboLab's [analysis guide](../src/RoboLab/docs/analysis.md), [data format](../src/RoboLab/docs/data.md), and [debugging guide](../src/RoboLab/docs/debug.md) for further inspection. Keep `--headless` for multi-task runs, verify the sidecar before guided evaluation, and retain the YAML, checkpoint identifier, seeds, and full command with each result.
