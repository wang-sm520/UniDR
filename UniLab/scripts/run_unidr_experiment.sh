#!/usr/bin/env bash
# Orchestration only: the runtime owns training and audit; UniLab owns holdout playback.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 RUN_DIR REPORT_DIR HOLDOUT_DIR [single|four]" >&2
  exit 2
fi
run_dir=$1
report_dir=$2
holdout_dir=$3
layout=${4:-single}
case "$layout" in single|four) ;; *) echo "Expected single or four GPU layout" >&2; exit 2 ;; esac
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
uv run --no-sync python scripts/train_unidr.py \
  "task=g1_flip_tracking/unidr_${layout}_gpu" \
  algo.num_envs=1024 algo.max_iterations=10000 \
  "training.log_dir=$run_dir"
uv run --no-sync python ../unilab_rl/examples/report_synchronous.py "$run_dir" "$report_dir"
MUJOCO_GL=egl uv run --no-sync python scripts/play_unidr_holdout.py \
  "$run_dir/model_9999.pt" --output "$holdout_dir"
