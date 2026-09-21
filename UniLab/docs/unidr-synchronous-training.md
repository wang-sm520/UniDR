# Four-source G1 flip training

On 2026-09-16 the user shortened the active experiment to the checkpoint numbered
5000 and requested the final audit and MuJoCo video. The existing periodic save
uses zero-based filenames: `model_5000.pt` contains 5001 completed updates,
491618304 transitions and 100020 Adam steps. This count was explicitly reported
before stopping. The selection is recorded before MuJoCo evaluation in
`logs/unidr/stop-at5000-20260916/selection.json`; the original 10000-iteration
`run_config.json` remains untouched. Use the explicit checkpoint-prefix audit
and `--expected-iterations 5001` holdout commands below for this experiment.
The checkpoint was saved and training stopped at 11:49 CST. Its independent audit
passed; fixed-policy MuJoCo playback produced a verified 20-second 720p/50 FPS
video with four backward flips followed by stable upright recovery before native
reference resets, zero early terminations and two normal episode timeouts. See
[the completed experiment record](../logs/unidr/stop-at5000-20260916/delivery.md).

This delivery uses the successful Motrix run's native MuJoCo-aligned task and PPO
profile. It starts one fresh shared learner; the historical per-backend baseline
models are not loaded. See [configuration evidence](unidr-four-source-config.md),
[runtime children and ADR](../../unilab_rl/docs/unidr-central-ppo.md), and
[fixed-final holdout](unidr-holdout.md). MuJoCo is excluded from training and scoring.

## Local provenance

| Owner | Actual base commit | Local checkout |
| --- | --- | --- |
| UniLab, Apache-2.0 | `b4e6b58fe0861a435fd19c0f0206bd84f4427a9c` | `/home/wsm/wang-sm/UniLab-unidr-backends` |
| uni_rl, Apache-2.0 | `79418e0cbff7b95fe6e49709b194454701ba20fa` | `/home/wsm/wang-sm/unilab_rl` |
| UniSim, Apache-2.0 | `4270aa81d868744980db90dac6dd959d3f542f50` | `/home/wsm/wang-sm/unisim-unidr-backends` |

The original UniLab checkout and existing dirty baseline/C1/backend changes were
preserved. These are local diffs, not published releases. AGENTS, licenses and locks
were rechecked. UniLab locks `unilab-rl==1.2.0` and RSL-RL 5.0.1; only the former
was replaced by the local editable checkout using `uv pip install --no-deps`.
The runtime lock also has RSL-RL 5.5.0; both dependency profiles are tested.
No vendor environment or system driver was reinstalled.

Local hardware: RTX 3090 24576 MiB, driver 580.178.04, i9-12900K, 64 GiB RAM.
Four-GPU mapping is tested in configuration only; this machine has one GPU.
In the real single-GPU process tree, central PPO, Genesis, Isaac Gym's Python 3.8
worker and Isaac Sim's Python 3.11 worker each hold GPU contexts; Motrix physics is
an independent CPU service. The four service processes contain no policy copies.

## Run

From the UniLab experiment checkout, keep the local dependency profile explicit:

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export UNILAB_LOCAL_UNISIM=/home/wsm/wang-sm/unisim-unidr-backends
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
uv pip install --python .venv/bin/python --no-deps -e /home/wsm/wang-sm/unilab_rl
uv run --no-sync python scripts/train_unidr.py task=g1_flip_tracking/unidr_single_gpu algo.num_envs=2 algo.max_iterations=2 training.log_dir=/absolute/new/smoke
uv run --no-sync python scripts/train_unidr.py task=g1_flip_tracking/unidr_single_gpu algo.num_envs=1024 algo.max_iterations=100 training.log_dir=/absolute/new/capacity
```

The full authorized pipeline uses native owner defaults, starts fresh, audits the
fixed final checkpoint, generates curves, then records MuJoCo:

```bash
bash scripts/run_unidr_experiment.sh /absolute/new/formal /absolute/new/report /absolute/new/holdout single
```

Use `four` for the implemented four-GPU mapping (Sim 0, Gym 1, Genesis 2, Motrix CPU,
central learner 3). The equivalent direct training owner is
`task=g1_flip_tracking/unidr_four_gpu`. Do not use `training.sim_backend` to select
the aggregate sources. Source task/reward/action/PPO behavior is checked against
the aligned owner; source overrides permit physical device binding only.

To resume a complete boundary into a new directory:

```bash
uv run --no-sync python scripts/train_unidr.py task=g1_flip_tracking/unidr_single_gpu algo.resume=true algo.resume_path=/absolute/parent/model_500.pt training.log_dir=/absolute/new/recovered
```

For an unchanged full-run request, the target remains 10000 completed iterations.
The learner restores model,
optimizer, LR, normalizers, counters and RNG; all environments reset into new
episodes/generation. The task/assets/budget fingerprint must agree. No physical
simulation state is claimed restored, and partial windows are never reused.

For the user-selected earlier stopping point, after the complete checkpoint and
resource cleanup have been verified:

```bash
uv run --no-sync python ../unilab_rl/examples/report_synchronous.py logs/unidr/single-formal-1024-10000-20260916 logs/unidr/formal-report-5000-20260916 --expected-iterations 5001 --num-envs 1024 --final-checkpoint logs/unidr/single-formal-1024-10000-20260916/model_5000.pt
MUJOCO_GL=egl uv run --no-sync python scripts/play_unidr_holdout.py logs/unidr/single-formal-1024-10000-20260916/model_5000.pt --expected-iterations 5001 --output logs/unidr/mujoco-final-5000-20260916
```

Prefix mode reports the original configured maximum, selected checkpoint hash,
exact committed budget and any excluded log tail. It does not rewrite the
training history or reuse an interrupted next window. The default full-run audit
still requires the actual iteration count to equal the original intention.
The shortening change passed the full UniLab non-slow coverage gate (1666 passed,
27 skipped, 845 deselected, 71% coverage, 96.61 seconds). Runtime RSL-RL 5.0.1 and
5.5.0 each passed 497 tests (36 skipped, 3 deselected, 65% coverage). Runtime Ruff,
Mypy and Pyright passed; UniLab retained its same ten pre-existing Pyright errors.

## Executed evidence

- Four individual 2-env reference-phase checks passed, covering frames
  0/100/124/180/224, exact initial state/IMU consistency and finite physics steps.
- Concurrent 4×2, two iterations passed in
  `logs/unidr/single-smoke-20260916c`: 384 transitions, 40 Adam steps, both int64
  normalizer counts 384. Earlier attempts exposed/fixed real init-state and flat
  observation-space differences; step validation was not weakened.
- Concurrent 4×1024, 100 iterations passed in
  `logs/unidr/single-capacity-1024-100-20260916a`: 9,830,400 transitions, 2,000 Adam
  steps, 500 full epochs, 2,457,600 samples/source. Both MLPs changed; all model
  tensors are finite, and both normalizer counts equal the transition budget.
  Training time 628.531 seconds; average 15,640 transitions/s. GPU memory was
  approximately 11 GiB. Clean exit released all four services/vendor processes.
  Checkpoint SHA256:
  `60f38ddc6334ad455ce1b69ea6a9e9450023541e01774f2348739ad102ecd738`.
- Capacity curves and independent audit are in
  `logs/unidr/capacity-report-20260916/{audit.json,curves.png,curves.pdf}`.
- Formal fresh run started 2026-09-16 01:56:22 CST under
  `unidr-four-source-20260916.service`; outputs go to
  `logs/unidr/single-formal-1024-10000-20260916`, `formal-report-20260916`, and
  `mujoco-final-20260916`. Console log: `logs/unidr/formal-console-20260916.log`.
  Starting this service is not evidence that its 10000 iterations or holdout have
  completed. Final audit and video must exist and be checked before claiming that.

Gate commands executed with `UV_NO_SYNC=1`:

```bash
PATH=/home/wsm/.vscode-server/cli/servers/Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1/server:$PATH UV_NO_SYNC=1 make check
UNILAB_LOCAL_UNISIM=/home/wsm/wang-sm/unisim-unidr-backends OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 UV_NO_SYNC=1 make test
UNILAB_LOCAL_UNISIM=/home/wsm/wang-sm/unisim-unidr-backends OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 UV_NO_SYNC=1 COVERAGE_FILE=/tmp/unidr-unilab-delivery-coverage make test-cov
UV_NO_SYNC=1 make check-tests
UV_NO_SYNC=1 make test-benchmark-smoke
PATH=/home/wsm/.vscode-server/cli/servers/Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1/server:$PATH UV_NO_SYNC=1 make test-all
```

Final UniLab `make test-cov` passed 1655 tests, with 27 skipped, 845 deselected,
14 warnings and 71% repository coverage in 91.52 seconds. The holdout owner has
94% coverage. Ruff (404 files) and Mypy (140 source files) pass.
`make check`/`make test-all` stop at the same ten pre-existing
Pyright errors in `base/reset_state.py`, `managers/_buffers/circular_buffer.py`,
and `tasks/motion_tracking/common/manager_terms.py`; no strict setting was disabled.
Benchmark module/script import smoke passed 34/35 each (one Apple-only MLX skip).
The final runtime suites passed 482 tests under each of RSL-RL 5.0.1 and 5.5.0;
see the linked runtime record for exact commands and the two final review guards.
Runtime tests, actual backend checks, capacity, formal training and final transfer
are distinct evidence levels. Adaptive source shares and independent probes remain
outside this delivery.

## Fixed final-model MuJoCo holdout

`scripts/play_unidr_holdout.py` records the final four-source policy through
`unilab.visualization.unidr_holdout`. By default it accepts only `model_9999.pt`
from the completed 10000-update formal run. An explicit `--expected-iterations N`
requires exactly `model_{N-1}.pt` and a source plan covering all N updates.
It does not search for checkpoints or choose
models using MuJoCo outcomes. Existing task/config ownership follows
[ADR-0003](sphinx/source/adr/ADR-0003-task-owner-and-config-compose-contract.md).

Required inputs are the checkpoint and its sibling `run_config.json` and
`sources_manifest.json`. Preflight validates the formal four-source order,
1024-environment source slices, exactly N×98304 transitions, complete central
update/version counters and every Adam parameter's finite moments at step N×20.
It recomputes the source/algorithm/asset manifest digest, checks checkpoint
algorithm equality, training asset hashes and the complete strict Sim2Sim snapshot. It composes
the native `g1_flip_tracking/mujoco` owner and requires identical environment and
reward semantics. Per-joint action scales, motion clip, normalization, native
termination thresholds and clip wrapping remain unchanged.

Both actor and critic load strictly against native 160/286-input, 29-action MLP
shapes before environment construction. Missing models or normalizer state and
non-finite tensors fail closed. The actor then runs deterministically on CPU
with frozen parameters and normalization; no optimizer or learner is created.
Evaluation seed 1 is applied before construction and startup randomization, then
recorded as `env.seed` in `mujoco_config.json` and reused at the initial reset.

After the fixed formal checkpoint exists, run from
`/home/wsm/wang-sm/UniLab-unidr-backends` with the completed run directory:

```bash
export UNILAB_LOCAL_UNISIM=/home/wsm/wang-sm/unisim-unidr-backends
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MUJOCO_GL=egl
uv run --no-sync python scripts/play_unidr_holdout.py logs/unidr/single-formal-1024-10000-20260916/model_9999.pt --output logs/unidr/mujoco-final-20260916
```

The output directory must be new. It contains `mujoco_config.json`,
`telemetry.jsonl`, `holdout.mp4` and, only after successful video validation,
`holdout.json`. The recording is 1000 actual policy steps at 0.02 seconds each:
20 seconds, 1280×720, 50 fps. `SnapshotPlaybackSession` uses a plain root-tracking
side camera at 3.2 m distance, elevation -12°, azimuth 0°, with one robot. The bundled
ffmpeg fully decodes the finished stream and checks every frame and its format;
missing rendering or decode errors fail instead of reporting a completed video.
Review the final video's start, inversion, landing and reset frames for framing.

Each telemetry row records the reference frame used for the action, the next
reference frame, post-step root/foot kinematics, inversion, airborne state,
native termination terms and explicit reset/clip-wrap boundaries. Terminal
physics frames are captured before manual resets; native termination behavior
is preserved. `landing_candidate` is conservative kinematic evidence: the same
uninterrupted attempt must have airborne feet above 0.12 m and an inverted root,
then sustain 10 frames with upright-axis Z > 0.8, both feet below 0.12 m,
absolute root vertical speed < 0.5 m/s and reference frame ≥ 150. A done state or
clip wrap clears that history and cannot be a landing. This is not a contact
measurement or an automatic full-flip success verdict; inspect the actual video.

Engineering validation uses synthetic checkpoints, stub environments and a
synthetic encoding fixture. It never evaluates a trained policy on MuJoCo:

```bash
uv run --no-sync pytest -q tests/visualization/test_unidr_holdout.py
uv run --no-sync mypy src/unilab/visualization/unidr_holdout.py
uv run --no-sync ruff check src/unilab/visualization/unidr_holdout.py scripts/play_unidr_holdout.py tests/visualization/test_unidr_holdout.py
uv run --no-sync ruff format --check src/unilab/visualization/unidr_holdout.py scripts/play_unidr_holdout.py tests/visualization/test_unidr_holdout.py
```

On 2026-09-16, the new suite plus `tests/visualization/test_playback_session.py`
passed 41 tests in 7.10 seconds, including constructor/reset seed consistency.
Mypy, Ruff lint/format, CLI `--help` and
`git diff --check` passed. The later user-selected `model_5000.pt` holdout completed;
its artifact record above distinguishes reference/physics resets from landings.
