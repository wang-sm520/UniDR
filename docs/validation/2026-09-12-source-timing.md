# Per-simulator sampling timing, 2026-09-12

## Status

Implemented and CPU-validated, **not activated in the running GPU job**.
No training processes were signalled or restarted, and no GPU tests were run.
Checkpoint continuation requires user confirmation. Historical source timings
cannot be reconstructed from the existing aggregate TensorBoard records.
This report does not claim physics, capacity, stability, or convergence acceptance.

Architecture scope: ADR-0010, synchronous multi-source training observability.
There are no IPC, backend, quota, policy-I/O, DR, or PPO-loop changes here.

## Measured Baseline

Run directory:
`/home/wsm/wang-sm/UniLab/logs/rsl_rl_ppo/G1WalkFlat/20260912_205814_multisim_10000`.

Read TensorBoard scalar events for zero-based iterations 482 through 581, inclusive:

| Scalar | Samples | Mean |
| --- | ---: | ---: |
| `Perf/collection_time` | 100 | 5.817350735664368 s/update |
| `Perf/learning_time` | 100 | 0.2673628759384155 s/update |
| `Perf/total_fps` | 100 | 31570.84 transitions/s |

Collection accounts for approximately 95.6% of collection plus learning time.
The combined 24-step rollout averages about 242.4 ms/control step, including
learner inference and collection overhead. This is not any individual simulator's
time. Four sources run 2000 environments each, yielding 192000 transitions/update.
No `source/*/sampling/*` scalar tags exist in this old job. The slowest simulator
and physics/reset/transport contributions remain unmeasured.

These baseline iterations precede this change's full CPU test runs. CPU tests can
compete for host resources despite CUDA being hidden, so no measurements made
during them are presented as an uncontended training performance baseline.

## Changed Files In This Timing Task

UniLab:

- `src/unilab/training/multi_source.py`: thin opt-in logger context assembly.
- `src/unilab/scripts/train_rsl_rl.py`: wrap ordinary/bounded training calls.
- `src/unilab/conf/ppo/config.yaml`: timing disabled globally by default.
- `src/unilab/conf/ppo/task/g1_walk_flat/multisim.yaml`: enable for multisim.
- `tests/training/test_multi_source.py`: owner defaults and callback wiring.
- `docs/multisim_training.md`: output schema, units, limitations, activation.
- `docs/validation/2026-09-12-source-timing.md`: this evidence record.

unilab_rl:

- `src/uni_rl/algos/rsl_rl_source_timing.py`: reusable logger adapter using public
  source statistics, without importing UniLab or unisim.
- `tests/algos/test_rsl_rl_source_timing.py`: deterministic aggregation, writer
  delegation, signed sub-timers, missing steps, failure restoration, no overwrite.
- `tests/ipc/test_multi_source_ppo.py`: real spawn and stock PPO integration,
  including bounded validation, timeout handling, epoch quotas and checkpoints.

unisim: no changes in this timing task. Pre-existing dirty changes in all three
repositories are preserved and are not part of this change list.

## Output

For newly started multisim runs, `source_timing.jsonl` is exclusively created
inside the run directory and flushed after every full PPO update. Every row
contains the 24 individual `step_seconds` values for each source, sum, mean,
linear-interpolated P50/P95, maximum, samples, endpoint PID/RSS and deltas of
existing environment sub-timer totals. Partial rollouts are not reported complete.

The terminal prints a compact source summary; the configured PPO writer receives
`source/<name>/sampling/*` and `source/<name>/timing/<field>/rollout_sum` scalars.
Step durations include internal resets and exclude outer transport. Nested native
timers preserve their original units and are not additive. No extra CUDA
synchronization is introduced. GPU logging overhead is not yet measured.

The sum of per-step maximum source durations is a comparison statistic, not a
direct barrier measurement. Its difference from collection time includes policy
inference, transfer, copying, scheduling and in-collection logging. Post-update
JSONL/TensorBoard/console logging is outside RSL-RL collection/learning timers.
Do not sum all four source durations or label the residual as pure IPC time.

## Commands And Results

All commands used the existing editable development environment without syncing:

```bash
source /home/wsm/wang-sm/UniLab/.tmp/multisim-gpu.env
```

From `/home/wsm/wang-sm/UniLab`:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q tests/training/test_multi_source.py tests/training/test_multisim_validation.py tests/test_multisim_cli.py
UV_NO_SYNC=1 make check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 UV_NO_SYNC=1 make test
git diff --check
```

- Focused tests: 43 passed.
- Initial `make check`: one new `no-any-return` error at the ignored editable
  library boundary. Fixed with an explicit context-manager cast; rerun passed.
- Final `make check`: passed; existing optional `drake_uni.runtime` import warning.
- Full non-slow tests: 1703 passed, 30 skipped, 850 deselected, 1 xfailed.
- Diff whitespace check: passed.

From `/home/wsm/wang-sm/unilab_rl`:

```bash
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy src/uni_rl
uv run --no-sync pyright
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --no-sync pytest -q
git diff --check
```

- Ruff, format, mypy, Pyright and diff checks: passed. Pyright notes the sibling
  repo has no local `.venv`; commands use the configured shared environment.
- Full default CPU suite: 459 passed, 34 skipped, 3 deselected.
- The four real-spawn PPO integration combinations passed, with 24 timing samples
  and 48 transitions/source in their small CPU fixtures. These are not simulator
  speed measurements or full-scale GPU acceptance.

Read-only review found no new timing integration blockers. It noted a pre-existing
bounded-validation cleanup limitation: failure before writer initialization may
be masked by `stop_logging_writer()`. This is outside this timing change.

No `make test-all`, PR, commit, release, physical acceptance or GPU benchmark was
performed in this task. Before activation, checkpoint continuation must preserve
optimizer/cumulative counters and account for RSL-RL's zero-based saved iteration
and adaptive learning-rate restoration. Environments and partial rollouts are not
checkpointed and will reset; the 10000-update target must not become 10000 extra
updates. The current run remains active while confirmation is pending.
