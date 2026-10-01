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

## Set up the LIBERO client

Use a complete GR00T checkout and `uv`. The following client setup was tested with the pinned submodule revisions. It uses Python 3.10 and MuJoCo 2.3.7; newer MuJoCo versions can fail inside the pinned robosuite joint lookup.

From the BayesFP root, set `GR00T_ROOT` to the checkout you want to use:

```bash
BAYESFP_ROOT="$PWD"
GR00T_ROOT="$BAYESFP_ROOT/src/Isaac-GR00T"
LIBERO_ENV="$GR00T_ROOT/gr00t/eval/sim/LIBERO/libero_uv/.venv"
git -C "$GR00T_ROOT" submodule update --init external_dependencies/LIBERO
uv venv "$LIBERO_ENV" --python 3.10
uv pip install --python "$LIBERO_ENV/bin/python" \
  -r "$GR00T_ROOT/external_dependencies/LIBERO/requirements.txt"
uv pip install --python "$LIBERO_ENV/bin/python" \
  -e "$GR00T_ROOT/external_dependencies/LIBERO" --config-settings editable_mode=compat
uv pip install --python "$LIBERO_ENV/bin/python" -e "$GR00T_ROOT" --no-deps
uv pip install --python "$LIBERO_ENV/bin/python" \
  torch==2.5.1 torchvision==0.20.1 pydantic av tianshou==0.5.1 tyro pandas \
  dm_tree einops==0.8.1 albumentations==1.4.18 pyzmq transformers==4.51.3 \
  msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1 numpy==1.26.4 mujoco==2.3.7
```

The pinned GR00T code references two custom obstacle suites absent from its upstream LIBERO dependency. Apply the supplied [compatibility patch](../patches/isaac-gr00t-libero-optional-suites.patch) once to skip those optional suites when unavailable; standard LIBERO tasks remain registered:

```bash
git -C "$GR00T_ROOT" apply --check "$BAYESFP_ROOT/patches/isaac-gr00t-libero-optional-suites.patch"
git -C "$GR00T_ROOT" apply "$BAYESFP_ROOT/patches/isaac-gr00t-libero-optional-suites.patch"
```

Initialize an isolated LIBERO path configuration interactively. Answer `n` to use its default dataset location. Keep `LIBERO_CONFIG_PATH` exported when running the client:

```bash
export LIBERO_CONFIG_PATH="$GR00T_ROOT/gr00t/eval/sim/LIBERO/libero_uv/config"
"$LIBERO_ENV/bin/python" -c 'import libero.libero'
"$LIBERO_ENV/bin/python" "$GR00T_ROOT/gr00t/eval/rollout_policy.py" --help
```

These commands prepare the simulation client, not the model server. The server needs its own compatible model environment and checkpoint. EGL/MuJoCo rendering support must be available on the host. A simulator reset and step were validated with this setup; full policy evaluation requires a running server. The wrapper may emit observation-space/dtype warnings, which this setup does not address.

## Existing LIBERO launcher

[`scripts/run_all_libero_10.sh`](../scripts/run_all_libero_10.sh) runs ten named LIBERO tasks. It assumes:

- A complete `src/Isaac-GR00T` checkout is available; paths are resolved relative to the launcher, so it can run from any directory.
- Its LIBERO environment exists at `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python`.
- `gr00t/eval/rollout_policy.py` and simulator assets are available.
- A compatible policy server is already listening on `127.0.0.1:5555`.

After completing those prerequisites:

```bash
bash scripts/run_all_libero_10.sh
```

If the checkout, client environment, or server is elsewhere, override them explicitly:

```bash
GR00T_ROOT=/path/to/Isaac-GR00T \
PYTHON_BIN=/path/to/libero-env/bin/python \
POLICY_CLIENT_HOST=127.0.0.1 POLICY_CLIENT_PORT=5555 \
  bash scripts/run_all_libero_10.sh
```

The launcher reports missing checkout files or a missing interpreter before starting tasks. A cloned Git repository can still have missing working-tree files after a failed Git LFS checkout; check `git -C src/Isaac-GR00T status` as well as submodule status. Preserve intentional local changes before repairing a checkout.

The launcher requests 10 episodes, 10 parallel environments, 720 maximum episode steps, and 8 action steps per task. It stops on the first failed command. It neither starts the policy server nor selects FKC settings; those belong to the server configuration. It is therefore not a complete paper-reproduction command by itself.

## Additional experiments and artifacts

`test_examples/toy_problems/fkcdiffusion_4gmm_cost_constraints.py` is a standalone Gaussian-mixture score-model experiment. Inspect its `--help` for its separate training and sampling controls. The adjacent `fkc-diffusion/` tree contains reference applications with their own setup instructions; its tests are not a unified test suite for the root project.

`papers/` holds supporting PDFs. `preliminary_results/` contains exploratory plots and trajectories. `tmp/` contains experiment-specific scripts, configurations, and logs, including paths tied to particular environments. These artifacts are useful context but do not replace a documented, configured evaluation run.
