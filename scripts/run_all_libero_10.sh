#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="${GR00T_ROOT:-$SCRIPT_DIR/../src/Isaac-GR00T}"
PYTHON_BIN="${PYTHON_BIN:-$GR00T_ROOT/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}"
ROLLOUT_SCRIPT="$GR00T_ROOT/gr00t/eval/rollout_policy.py"
POLICY_CLIENT_HOST="${POLICY_CLIENT_HOST:-127.0.0.1}"
POLICY_CLIENT_PORT="${POLICY_CLIENT_PORT:-5555}"

if [[ ! -f "$ROLLOUT_SCRIPT" ]]; then
  echo "Missing GR00T rollout script: $ROLLOUT_SCRIPT" >&2
  echo "Initialize src/Isaac-GR00T, or set GR00T_ROOT to a complete checkout." >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing LIBERO Python environment: $PYTHON_BIN" >&2
  echo "Follow docs/integrations.md to set up LIBERO, or set PYTHON_BIN to its Python executable." >&2
  exit 1
fi

# Resolve paths before changing directory, including caller-supplied relative paths.
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
PYTHON_BIN="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd)/$(basename -- "$PYTHON_BIN")"
cd -- "$GR00T_ROOT"
ROLLOUT_SCRIPT="$PWD/gr00t/eval/rollout_policy.py"

TASKS=(
  "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
  "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
  "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
  "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
  "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
  "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
  "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
  "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
  "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
  "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
)

for ENV_NAME in "${TASKS[@]}"; do
  echo "=================================================="
  echo "Running env: $ENV_NAME"
  echo "=================================================="

  "$PYTHON_BIN" "$ROLLOUT_SCRIPT" \
    --n_episodes 10 \
    --policy_client_host "$POLICY_CLIENT_HOST" \
    --policy_client_port "$POLICY_CLIENT_PORT" \
    --max_episode_steps 720 \
    --env_name "$ENV_NAME" \
    --n_action_steps 8 \
    --n_envs 10
done
