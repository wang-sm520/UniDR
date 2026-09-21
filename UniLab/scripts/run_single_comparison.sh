#!/usr/bin/env bash
# Orchestration only: native PPO, runtime audit, then the fixed MuJoCo holdout.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 NEW_EXPERIMENT_ROOT [NUM_ENVS=1024] [ITERATIONS=20000]" >&2
  exit 2
fi
experiment_root=$(realpath -m -- "$1")
num_envs=${2:-1024}
iterations=${3:-20000}
[[ $num_envs =~ ^[1-9][0-9]*$ && $iterations =~ ^[1-9][0-9]*$ ]] || exit 2
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -- "$experiment_root"
exec 9>"$experiment_root/queue.lock"
flock -n 9
printf '%s\n' "$$" > "$experiment_root/queue.pid"
source_name=none
stage=initializing
child=
completed=false

publish_state() {
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$source_name" "$stage" > "$experiment_root/status.tmp"
  mv -- "$experiment_root/status.tmp" "$experiment_root/status.tsv"
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n $child ]] && kill -0 -- "-$child" 2>/dev/null; then
    kill -INT -- "-$child" 2>/dev/null || true
    for ((attempt=0; attempt<30; attempt++)); do
      kill -0 -- "-$child" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-$child" 2>/dev/null || true
    for ((attempt=0; attempt<5; attempt++)); do
      kill -0 -- "-$child" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  if [[ $completed != true ]]; then
    publish_state failed
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_stage() {
  stage=$1
  local log=$2
  shift 2
  publish_state running
  setsid env --default-signal=INT "$@" > "$log" 2>&1 &
  child=$!
  printf '%s\n' "$child" > "$experiment_root/stage.pid"
  wait "$child"
  if kill -0 -- "-$child" 2>/dev/null; then
    echo "Stage exited with remaining owned processes: $source_name/$stage" >&2
    return 1
  fi
  child=
  printf '0\n' > "$experiment_root/stage.pid"
}

for source_name in motrix isaacsim isaacgym genesis; do
  run_dir="$experiment_root/$source_name"
  run_stage prepare "$experiment_root/$source_name-prepare.log" \
    uv run --no-sync python -m unilab.training.single_comparison "$source_name" "$run_dir" \
    --num-envs "$num_envs" --iterations "$iterations"
  run_stage train "$run_dir/train.log" \
    uv run --no-sync train --algo ppo --task g1_flip_tracking --sim "$source_name" --profile comparison \
    "algo.num_envs=$num_envs" "algo.max_iterations=$iterations" \
    training.device=cuda:0 "training.log_dir=$run_dir"
  run_stage audit "$run_dir/single_audit.json" \
    uv run --no-sync python -m uni_rl.logging.single_run_audit "$run_dir" \
    --expected-iterations "$iterations" --num-envs "$num_envs"
  run_stage sim2sim "$run_dir/sim2sim.log" \
    env MUJOCO_GL=egl uv run --no-sync python scripts/play_single_reference.py \
    "$run_dir/model_$((iterations-1)).pt" --output "$run_dir/mujoco-front-reference" \
    --expected-iterations "$iterations" --num-envs "$num_envs"
  publish_state source_completed
done
completed=true
publish_state completed
