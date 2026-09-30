# Robot integrations and repository structure

[Project overview](../README.md)

## Initialize the policy repositories

The root repository pins four Git submodules in [`.gitmodules`](../.gitmodules):

| Directory | Repository |
| --- | --- |
| `src/diffusion_policy` | [S-sirigiri/diffusion-policy](https://github.com/S-sirigiri/diffusion-policy) |
| `src/Isaac-GR00T` | [S-sirigiri/Isaac-GR00T](https://github.com/S-sirigiri/Isaac-GR00T) |
| `src/openpi` | [S-sirigiri/openpi](https://github.com/S-sirigiri/openpi) |
| `src/RoboLab` | [S-sirigiri/RoboLab](https://github.com/S-sirigiri/RoboLab) |

From the repository root:

```bash
git submodule update --init --recursive
git submodule status
```

Or initialize only the integration you need:

```bash
git submodule update --init --recursive src/Isaac-GR00T
```

A leading `-` in `git submodule status` means the submodule is uninitialized. Three URLs use SSH and require GitHub SSH access; `openpi` uses HTTPS. If SSH access is unavailable, a local HTTPS override can be set before initialization, for example:

```bash
git config submodule.src/Isaac-GR00T.url https://github.com/S-sirigiri/Isaac-GR00T.git
git submodule update --init --recursive src/Isaac-GR00T
```

Repository access is still required. Keep the pinned revisions for reproducibility. Each integration has its own dependency, checkpoint, and simulator requirements; follow its README after initialization instead of installing everything into the toy environment. The submodules were uninitialized when these docs were prepared, so integration-specific server commands and checkpoint compatibility have not been verified here.

## Existing LIBERO launcher

[`scripts/run_all_libero_10.sh`](../scripts/run_all_libero_10.sh) runs ten named LIBERO tasks. It assumes:

- The current directory is the initialized `src/Isaac-GR00T` repository.
- Its LIBERO environment exists at `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python`.
- `gr00t/eval/rollout_policy.py` and simulator assets are available.
- A compatible policy server is already listening on `127.0.0.1:5555`.

After completing those prerequisites:

```bash
cd src/Isaac-GR00T
bash ../../scripts/run_all_libero_10.sh
```

The launcher requests 10 episodes, 10 parallel environments, 720 maximum episode steps, and 8 action steps per task. It stops on the first failed command. It neither starts the policy server nor selects FKC settings; those belong to the server configuration. It is therefore not a complete paper-reproduction command by itself.

## Additional experiments and artifacts

`test_examples/toy_problems/fkcdiffusion_4gmm_cost_constraints.py` is a standalone Gaussian-mixture score-model experiment. Inspect its `--help` for its separate training and sampling controls. The adjacent `fkc-diffusion/` tree contains reference applications with their own setup instructions; its tests are not a unified test suite for the root project.

`papers/` holds supporting PDFs. `preliminary_results/` contains exploratory plots and trajectories. `tmp/` contains experiment-specific scripts, configurations, and logs, including paths tied to particular environments. These artifacts are useful context but do not replace a documented, configured evaluation run.
