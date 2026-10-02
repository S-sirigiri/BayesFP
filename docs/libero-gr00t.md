# LIBERO with GR00T-N1.6

[Back to the main README](../README.md)

This experiment runs a GR00T-N1.6 policy fine-tuned on LIBERO-Object and adds an obstacle at inference time. The paper evaluates the butter pick-and-place task with a cylindrical obstacle and a non-convex obstacle. The latter is named `l_obstacle` in the implementation, although the paper describes it as V-shaped.

## Environment and checkpoint prerequisites

Follow the instructions in the checked-out repositories:

- [GR00T installation guide](../src/Isaac-GR00T/README.md#installation-guide) for the model environment and hardware requirements.
- [LIBERO evaluation setup](../src/Isaac-GR00T/examples/LIBERO/README.md#evaluate-checkpoint) for the separate simulator environment. This creates `gr00t/eval/sim/LIBERO/libero_uv/.venv`.
- [LIBERO-Object data and fine-tuning guide](../src/Isaac-GR00T/examples/LIBERO/README.md#fine-tune-libero-object) to obtain a compatible checkpoint. The supplied fine-tuning script saves checkpoints under `/tmp/libero_object`; adjust its GPU count and output location for your machine.

Use the `src/Isaac-GR00T` fork included in this repository. It contains the guidance implementation and a vendored LIBERO copy with the custom obstacle tasks. Installing an unmodified upstream LIBERO copy will not register these tasks. Keep the model and simulator environments separate as described in the setup guide.

You need a complete **LIBERO-Object fine-tuned GR00T-N1.6 checkpoint**, including its processor and embodiment metadata. A base checkpoint or a DROID checkpoint does not substitute for this fine-tuning. The paper's fine-tuned checkpoint is not distributed by this repository; supply your own compatible checkpoint or train one using the linked guide.

## Select the sampler and obstacle

All paths and commands below assume your terminal starts at the BayesFP repository root:

```bash
cd src/Isaac-GR00T
```

Inference reads [cost_guided_inference.yaml](../src/Isaac-GR00T/gr00t/model/gr00t_n1d6/cost_guided_inference.yaml). Edit the existing file, retaining its full `embodiments` mapping. There is no server CLI flag to select another guidance YAML.

| Setting | Meaning / example |
| --- | --- |
| `mode: vanilla` | Original policy sampler; use for the unguided baseline. |
| `mode: linear_combo` | Gradient guidance baseline. |
| `mode: fkc` | BayesFP / FK particle sampler. |
| `objective.constraint_task: cylinder_abs` | Cylindrical clearance penalty. |
| `objective.constraint_task: l_obstacle_box` | Box-based penalty for the non-convex obstacle. |
| `fkc.num_particles: 8` | Particle count used for GR00T in the paper. |
| `time_grid.num_steps: 4` | Denoising steps for guided modes; vanilla uses the checkpoint's inference-step setting. |
| `sigma_schedule.kind: zero_ends`, `alpha: 0.25` | Flow-policy noise schedule. |
| `beta_schedule.kind: constant` | Constant inverse-temperature schedule. |
| `beta_schedule.guidance_strength` | Positive guidance strength; the implementation applies its negative value as beta. |
| `objective.weights.inequality` | Weight of the obstacle penalty. |
| `embodiments.2.constraint_trajectory.executed_steps: 8` | Number of action steps considered by the LIBERO trajectory proxy; match the client's `--n_action_steps`. |

For the first guided cylinder run, set `mode: fkc`, `objective.constraint_task: cylinder_abs`, and `fkc.num_particles: 8`. Keep `objective.weights.cost` and `objective.weights.equality` at `0.0` for obstacle-only guidance. The checked-in guidance strength and inequality weight are starting values to tune for your checkpoint and scene, rather than a guarantee of the paper's reported results.

The `cylinder_ellipse` option invokes an elliptical penalty. Choose `cylinder_abs` explicitly for the cylinder example; the filename or existing default alone does not determine the obstacle geometry. `cylinder_squared_hinge` is another implemented cylinder penalty and changes the cost scale.

The inference YAML is reloaded when it changes. Finish one evaluation before changing it, and save the configuration alongside the results. Concurrent servers using the same checkout share this file.

## Launch the policy server

In terminal 1, from `src/Isaac-GR00T`, replace the checkpoint path and start the server:

```bash
export BAYESFP_GR00T_CHECKPOINT=/absolute/path/to/libero_object/checkpoint-20000

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$BAYESFP_GR00T_CHECKPOINT" \
  --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper \
  --host 127.0.0.1 \
  --port 5555
```

Wait for `Server is ready and listening` before starting the client. The simulation policy wrapper is required to adapt the client's observations and actions to the model interface.

## Run the simulator client

In terminal 2, also from `src/Isaac-GR00T`, start with one episode and one environment:

```bash
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
  gr00t/eval/rollout_policy.py \
  --policy_client_host 127.0.0.1 \
  --policy_client_port 5555 \
  --env_name libero_sim/libero_object_cylinder_pick_up_the_butter_and_place_it_in_the_basket \
  --n_episodes 1 \
  --n_envs 1 \
  --max_episode_steps 720 \
  --n_action_steps 8
```

For a larger evaluation, change `--n_episodes` to `100`, matching the paper's LIBERO evaluation count, and increase `--n_envs` as memory permits. FK memory use increases with both the number of environments and the number of particles. The vectorized runner can finish more than the requested number of episodes on its final batch; use `--n_envs 1` if an exact episode count is essential. The current CLI does not expose a random-seed flag.

Use the following task and constraint pairs:

| Scene | Client `--env_name` | Guidance `objective.constraint_task` |
| --- | --- | --- |
| Original butter task | `libero_sim/pick_up_the_butter_and_place_it_in_the_basket` | Use `mode: vanilla` for a no-obstacle reference. |
| Cylinder | `libero_sim/libero_object_cylinder_pick_up_the_butter_and_place_it_in_the_basket` | `cylinder_abs` |
| Non-convex obstacle | `libero_sim/libero_object_l_obstacle_pick_up_the_butter_and_place_it_in_the_basket` | `l_obstacle_box` |

To compare FK with the unguided and linear-combination baselines, repeat the same client command with `mode` set to `vanilla`, `linear_combo`, and `fkc`, respectively. Retain the checkpoint, obstacle geometry, action execution horizon, and evaluation budget across comparisons. These three modes do not cover every baseline reported in the paper.

## Outputs and experiment records

The client prints `results:`, `success rate:`, and `Video saved to:`. Outputs go to a unique directory under `/tmp/sim_eval_videos_...`; the printed path is the authoritative location. Copy it to persistent storage after the run.

Completed episodes produce MP4 videos and compressed NPZ trajectories. The NPZ files contain `states`, `actions`, `rewards`, `dones`, `success_flags`, and `metadata_json`. For LIBERO, the first three state columns are the measured end-effector world position; actions are the commands actually sent to the simulator. Success labels report task completion. The main rollout command does not automatically report the paper's constraint-violation metric; that requires a separate geometric analysis of the saved trajectories.

Record the checkpoint path, submodule revision, guidance YAML, obstacle geometry, episode count, and command line for each run. Store separate output directories for each mode and scene.

## Geometry and common issues

- The non-convex obstacle's scene and penalty read [gr00t/configs/l_obstacle/config.yaml](../src/Isaac-GR00T/gr00t/configs/l_obstacle/config.yaml). Environment registration generates the corresponding MJCF and BDDL files. Set geometry before launching the client, and recreate environments after changing it so the scene agrees with the penalty.
- The cylinder scene and analytic cost are configured separately: inspect the [cylinder MJCF](../src/Isaac-GR00T/external_dependencies/LIBERO/libero/libero/assets/custom_objects/cylinder/cylinder.xml), [task BDDL](../src/Isaac-GR00T/external_dependencies/LIBERO/libero/libero/bddl_files/libero_object_cylinder/pick_up_the_butter_and_place_it_in_the_basket.bddl), and [constraint objective](../src/Isaac-GR00T/gr00t/model/gr00t_n1d6/cost_constraint_objective.py). Synchronize their center and radius when reproducing a specified scene; the analytic safety margin is additional to the obstacle radius.
- `No module named gr00t` usually indicates the editable installation is missing or still references a checkout that has moved. Follow the linked setup guides again for the current checkout in each environment.
- A missing obstacle task usually means the simulator imported a different LIBERO installation. Use the vendored copy installed by the fork's setup guide.
- For EGL or offscreen rendering failures, consult the graphics-library requirements in the linked LIBERO evaluation setup. The environment selects EGL by default.
- If GPU memory is exhausted, reduce `--n_envs` first. Reducing `fkc.num_particles` also lowers memory use but changes the sampler configuration.

The research comparison and sweep scripts under `src/Isaac-GR00T/tmp/` contain machine-specific paths and environment names. Adapt those scripts before using them for automation; the server/client commands above are the portable evaluation entry points.
