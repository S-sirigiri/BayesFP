#!/bin/bash
# Usage: bash run_one.sh <run_name> <yaml_path> <num_envs>
# - copies <yaml_path> over src/openpi/configs/fkc/placeholder.yaml
# - starts serve_policy in background
# - waits for websocket port 8000 to open
# - runs run_eval.py with --output-folder-name fkc_sweep_<run_name>
# - kills server, runs analyze_collisions
# Logs land in tmp/fkc_sweep/<run_name>.{server,eval,analyze}.log

set -euo pipefail

NAME="${1:?run name}"
YAML="${2:?yaml path}"
NUM_ENVS="${3:-4}"

ROOT=/home/vardhan/mnt/Saturn/Projects/FKC_concost_diffusion
PLACEHOLDER=$ROOT/src/openpi/configs/fkc/placeholder.yaml
SWEEP=$ROOT/tmp/fkc_sweep

# 1) Stage the YAML
cp "$YAML" "$PLACEHOLDER"

# 2) Make sure no stale server holds port 8000
if ss -tln | grep -q ":8000 "; then
  echo "[harness] Port 8000 is busy, attempting to free it..."
  pkill -f "scripts/serve_policy.py" || true
  sleep 3
fi

# 3) Start server
SERVER_LOG="$SWEEP/${NAME}.server.log"
EVAL_LOG="$SWEEP/${NAME}.eval.log"
ANALYZE_LOG="$SWEEP/${NAME}.analyze.log"

cd "$ROOT/src/openpi"
echo "[harness] Starting serve_policy for run=$NAME ..."
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 nohup uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos \
  --policy.fkc_config=configs/fkc/placeholder.yaml \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[harness] server pid=$SERVER_PID, log=$SERVER_LOG"

# 4) Wait for port 8000 to be listening (max 15 min for first compile)
WAITED=0
MAX_WAIT=900
while ! ss -tln | grep -q ":8000 "; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[harness] server died before listening; tail of log:"
    tail -n 60 "$SERVER_LOG"
    exit 1
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    echo "[harness] server didn't start within $MAX_WAIT s"
    kill "$SERVER_PID" 2>/dev/null || true
    tail -n 60 "$SERVER_LOG"
    exit 1
  fi
done
echo "[harness] server listening (waited ${WAITED}s)"

# 5) Run eval
cd "$ROOT/src/RoboLab"
OUTPUT_NAME="fkc_sweep_${NAME}"
echo "[harness] Starting run_eval (num_envs=$NUM_ENVS, output=$OUTPUT_NAME) ..."
set +e
.venv/bin/python examples/policy/run_eval.py \
  --policy pi05 \
  --task FruitsGreenLimesOnPlateTask \
  --num-envs "$NUM_ENVS" \
  --headless \
  --enable-sdf-guidance \
  --report-collisions \
  --sdf-ooi-exclusion-mode static \
  --sdf-ooi-object-names lime01 lime01_01 lemon_01 lemon_02 clay_plates \
  --output-folder-name "$OUTPUT_NAME" \
  > "$EVAL_LOG" 2>&1
EVAL_RC=$?
set -e

# 6) Kill server
echo "[harness] Killing server pid=$SERVER_PID ..."
kill "$SERVER_PID" 2>/dev/null || true
sleep 2
kill -9 "$SERVER_PID" 2>/dev/null || true
pkill -f "scripts/serve_policy.py" 2>/dev/null || true

# 7) Analyze collisions
OUTDIR="$ROOT/src/RoboLab/output/$OUTPUT_NAME"
if [ -d "$OUTDIR" ]; then
  echo "[harness] Analyzing $OUTDIR ..."
  "$ROOT/src/RoboLab/.venv/bin/python" \
    "$ROOT/src/RoboLab/scripts/analyze_collisions.py" "$OUTDIR" \
    > "$ANALYZE_LOG" 2>&1 || true
fi

echo "[harness] DONE name=$NAME eval_rc=$EVAL_RC"
exit $EVAL_RC
