#!/usr/bin/env bash
# Follow-on orchestration; PPO, source timing and budget audit remain in uni_rl.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "Usage: $0 SINGLE_ROOT NEW_JOINT_ROOT SINGLE_UNIT INVOCATION_ID" >&2
  exit 2
fi
single_root=$(realpath -e -- "$1")
joint_root=$(realpath -m -- "$2")
single_unit=$3
invocation=$4
[[ $single_unit == *.service && $invocation =~ ^[0-9a-f]{32}$ ]] || exit 2
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
runtime_dir=vendor/unilab_rl
[[ -d $runtime_dir ]] || runtime_dir=../unilab_rl
mkdir -- "$joint_root"
exec 9>"$joint_root/queue.lock"
flock -n 9
printf '%s\n' "$$" > "$joint_root/queue.pid"
stage=waiting_for_singles
child=
completed=false

publish_state() {
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$stage" > "$joint_root/status.tmp"
  mv -- "$joint_root/status.tmp" "$joint_root/status.tsv"
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
  if [[ $completed != true ]]; then publish_state failed; fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

publish_state waiting
while true; do
  properties=$(systemctl --user show "$single_unit" -p LoadState -p InvocationID \
    -p ActiveState -p SubState -p MainPID -p Result -p ExecMainCode -p ExecMainStatus \
    -p ExecMainExitTimestampMonotonic)
  declare -A unit=()
  while IFS='=' read -r key value; do unit[$key]=$value; done <<< "$properties"
  [[ ${unit[LoadState]:-} == loaded && ${unit[InvocationID]:-} == "$invocation" ]] || {
    echo "Single queue disappeared or invocation changed; refusing training" >&2; exit 1;
  }
  case ${unit[ActiveState]:-} in
    active|activating|deactivating) sleep 30 ;;
    inactive)
      [[ ${unit[MainPID]:-} == 0 && ${unit[Result]:-} == success && \
         ${unit[SubState]:-} == dead && ${unit[ExecMainCode]:-} == 1 && \
         ${unit[ExecMainStatus]:-} == 0 && \
         ${unit[ExecMainExitTimestampMonotonic]:-0} -gt 0 ]] || exit 1
      break ;;
    *) echo "Single queue failed; refusing joint training" >&2; exit 1 ;;
  esac
done
printf '%s\n' "$properties" > "$joint_root/single-unit-completed.txt"

run_stage() {
  stage=$1
  local log=$2
  shift 2
  publish_state running
  setsid env --default-signal=INT "$@" > "$log" 2>&1 &
  child=$!
  printf '%s\n' "$child" > "$joint_root/stage.pid"
  wait "$child"
  if kill -0 -- "-$child" 2>/dev/null; then
    echo "Stage exited with remaining owned processes: $stage" >&2
    return 1
  fi
  child=
  printf '0\n' > "$joint_root/stage.pid"
}

run_stage preflight "$joint_root/preflight.json" \
  uv run --no-sync python -m unilab.training.joint_comparison "$single_root"
run_stage train "$joint_root/train.log" \
  uv run --no-sync python scripts/train_unidr.py task=g1_flip_tracking/unidr_comparison \
  "training.log_dir=$joint_root/train"
run_stage audit_report "$joint_root/report.log" \
  uv run --no-sync python "$runtime_dir/examples/report_synchronous.py" \
  "$joint_root/train" "$joint_root/report" --expected-iterations 5000 --num-envs 1024
completed=true
publish_state completed
