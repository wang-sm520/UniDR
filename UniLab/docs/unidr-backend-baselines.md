# G1 Flip Single-Backend Baselines

Local experiments establish the four training sources separately before connecting
the C1 unified rollout. Each experiment has its own PPO model. This is validation
work, not the final architecture with one shared learner. MuJoCo is excluded from
these training and scoring entrypoints. No C1 runtime files were changed.

## Provenance and Ownership

- UniLab base: `b4e6b58fe0861a435fd19c0f0206bd84f4427a9c`, Apache-2.0;
  local branch `codex/unidr-backend-baselines` in `UniLab-unidr-backends`.
- UniSim base: v1.3.0, `4270aa81d868744980db90dac6dd959d3f542f50`, Apache-2.0;
  local branch `codex/unidr-motion-body-ids` in `unisim-unidr-backends`.
  Both deliveries are uncommitted diffs against these SHAs.
- Runtime: published `unilab-rl==1.2.0`, RSL-RL `5.0.1`, TensorDict `0.11.0`,
  Torch `2.8.0+cu128`; these experiments do not use the C1 fake runner.
- Local hardware: one RTX 3090 (24 GiB), driver `580.178.04`, i9-12900K.
  GPU physics jobs run serially; each baseline uses a CPU PPO learner. Motrix
  physics also runs on CPU. This is not the planned four-GPU topology.
- Isaac Sim: existing Python 3.11 / IsaacSim 5.1 / IsaacLab checkout
  `37ddf626871758333d6ed89cf64ad702aef127d0` (BSD-3-Clause), worker Torch 2.7+cu128.
  Isaac Gym: existing Python 3.8.20 worker and Preview 4 SDK, NumPy 1.21.6.
  Genesis: locked 1.3.3. Motrix: locked motrixsim-core 0.8.2.
- Robot hub revision: `9d489de88a0111533dac3d2fd01dab6feda18301`.
  Motion hub revision: `5b002ecc17ecb490af5924a779ecc063673c72df`.
  Flip NPZ SHA256: `bd6785de42e569deb0f680057c0ff62449d3d99058e3c0c1ce437c373e8b604d`.
  G1 XML SHA256: `bb6089243f8fe1c97a6410ccde8754eaa58e5d8e476aca69d790ce501cb4a48e`.

UniLab owns the five baseline YAML files, G1 registration, task diagnostics and
tests. `scripts/validate_g1_flip.py` only composes existing factories, wrappers,
checkpoint loading and diagnostics. UniSim owns the prerequisite motion-body ID
mapping, Gym reset corrections and offset-IMU velocity correction. Its existing
public interfaces are unchanged.
See `../unisim-unidr-backends/docs/isaacgym-reset-validation.md` from the checkout
parent for the backend validation record. Existing owner decisions are described
in `sphinx/source/adr/ADR-0003-task-owner-and-config-compose-contract.md`.

## Baseline Contract

All four `--profile baseline` owners use the actual `flip_360_001__A304.npz`:
225 frames at 50 Hz, G1 29 DoF, actor observation 160, critic observation 286,
29 actions, control dt 0.02 s and simulation dt 0.005 s. They preserve the original
Motrix reward/action profile explicitly in `base.yaml` (action scale 0.25,
fixed identity normalizer), independently of the native Motrix defaults.
No G1 flat task is substituted. Default pool capacity is 2048; the bounded
experiments below use 64 environments, 24 steps, 5 epochs and 4 minibatches.

Clip end is an explicit timeout. Reference-relative anchor/limb tests permit the
motion's inversion and flight; the absolute body-height contact proxy is removed.
Unused geometry bindings are removed because tracking uses named body states.
Finite checks and strict checkpoint contract validation remain enabled.

Evaluation reloads the exact source run config in a fresh process, freezes the
actor/normalizer, and counts one first-episode opportunity per environment.
Finished rows are manually reset to satisfy NpEnv but never counted again.
Unobserved steps after failure receive the fixed 0.5 m body-error cap; return is
the raw sum up to first done with zero padding. Reported diagnostics are not the
future adaptive scheduler's normalized E/S/R contract or checkpoint selector.

## Reproduce

Run from `/home/wsm/wang-sm/UniLab-unidr-backends`:

```bash
uv sync --locked --python 3.10 --extra motrix --extra genesis --extra mujoco
uv pip install --python .venv/bin/python --no-deps -e ../unisim-unidr-backends
export UNILAB_LOCAL_UNISIM=/home/wsm/wang-sm/unisim-unidr-backends
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
UNIDR_REAL_BACKEND=motrix uv run --no-sync pytest -q -m '' tests/tasks/test_g1_flip_baselines.py tests/tasks/test_g1_flip_reference_reset.py
uv run --no-sync train --algo ppo --task g1_flip_tracking --sim motrix --profile baseline algo.num_envs=64 algo.max_iterations=1000 algo.save_interval=250 training.device=cpu training.log_dir=/home/wsm/wang-sm/UniLab-unidr-backends/logs/baselines/motrix-1000
uv run --no-sync python scripts/validate_g1_flip.py logs/baselines/motrix-1000/model_999.pt --output logs/baselines/motrix-1000/validation.json
```

Replace `motrix` in the backend and run paths with `isaacsim`, `isaacgym` or
`genesis` for independent experiments. The valid Isaac Sim run directory is
`isaacsim-1000-corrected`. Preserve `--no-sync` after the editable
UniSim installation; the declared local dependency profile checks its exact path.
Do not rerun SDK setup scripts against the existing Isaac worker symlink trees.
MuJoCo is installed for repository checks/import utilities, never as a baseline
training or evaluation source. Meshes, clips, logs and checkpoints stay outside git.

## Native Motrix Alignment

The native PPO `g1_flip_tracking/motrix` owner now inherits MuJoCo's per-joint
action scales, complete reward profile and empirical actor/critic observation
normalization. Action-rate weight changes from -0.05 to -0.005; root-position
weight from 1 to 0.5; body-position/orientation weights from 1/1 to 2/1.5.
End-effector height tracking (weight 2, std 0.3) and undesired-contact reward
(weight -0.1) are restored. Terminations, the reference clip, 1024 environments
and Motrix's 30000-iteration budget are unchanged. The 23-DoF and APPO owners
and all four historical `--profile baseline` configurations are unchanged.

Start a new native run without `--profile baseline`; old action scales and
identity-normalizer checkpoints are incompatible with these new defaults.
Strict contract validation remains enabled. Preserve old run snapshots for replay.

```bash
uv run --no-sync train --algo ppo --task g1_flip_tracking --sim motrix algo.num_envs=1024 algo.max_iterations=30000 algo.resume=false algo.load_run=-1 training.device=cuda:0 training.no_play=true training.log_dir=/home/wsm/wang-sm/UniLab-unidr-backends/logs/baselines/motrix-aligned-1024-30000
```

This configuration alignment does not establish full-flip learning success;
a fresh training run and behavioral validation are still required.

Validation on 2026-09-15 used the UniLab SHA and published runtime versions above:

```bash
export UNILAB_LOCAL_UNISIM=/home/wsm/wang-sm/unisim-unidr-backends
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
uv run --no-sync train --algo ppo --task g1_flip_tracking --sim motrix algo.num_envs=2 algo.max_iterations=2 algo.save_interval=1 algo.resume=false algo.load_run=-1 training.device=cpu training.no_play=true training.log_dir=/home/wsm/wang-sm/UniLab-unidr-backends/logs/baselines/motrix-aligned-smoke-20260915
uv run --no-sync pytest -q -m '' tests/tasks/test_g1_flip_baselines.py tests/config/test_locomotion_params.py tests/envs/test_motion_profiles.py tests/algos/test_rsl_rl_runner.py::test_normalize_ppo_train_cfg_maps_empirical_normalization_to_models -k 'not representative_motion_profiles_reset_and_step'
PATH=/home/wsm/.vscode-server/cli/servers/Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1/server:$PATH UV_NO_SYNC=1 make check
UV_NO_SYNC=1 make check-tests
UV_NO_SYNC=1 make test
git diff --check
```

The native Motrix CPU smoke completed two PPO updates: 96 transitions, 40 Adam
steps, finite changed actor/critic MLP parameters, and saved normalizer statistics
with count 96 for each model. The restored end-effector reward appears in its
training log. The config suite passed 138 tests (1 opt-in test skipped, 6 unrelated
native reset/step cases deselected); an earlier broader attempt was interrupted
while waiting in socket I/O on those unrelated cases. The stale registry test
was updated to include the previously registered flip baseline backends.
The full non-slow suite passed 1604 tests (27 skipped, 844 deselected).
Ruff, Mypy and test lint passed; `make check` still fails on the same 10 pre-existing
Pyright errors recorded below. `git diff --check` passed. Logs are
`logs/baselines/motrix-aligned-{smoke,config,check,test}-20260915.log`.

Resolved config hashes for all four historical baselines match their pre-edit
values. Strict preflight rejects the completed original native run's snapshot
for both action and normalizer differences before environment construction.
No long retraining run was launched; use a fresh directory for subsequent runs.

## Verification Record

Each completed run uses 1000 iterations; the stock runner stores the final
zero-based index as 999. Checkpoint optimizer state, TensorBoard steps 0..999 and
run summaries are cross-checked: 1,536,000 transitions and 20,000 Adam steps.
All actor/critic tensors are finite and changed relative to `model_0.pt` (the
two-iteration native test separately compares against the untrained model).

| Source | Updates | Mean evaluation steps / 225 | Full-clip completion | First failure |
| --- | ---: | ---: | ---: | --- |
| Motrix | 1000 | 99 | 0/8 | reference torso height |
| Isaac Gym | 1000 | 84 | 0/8 | reference end-effector height |
| Genesis | 1000 | 101 | 0/8 | reference end-effector height |
| Isaac Sim, corrected | 1000 | 81 | 0/8 | torso (1), end-effector (7) reference height |

These are fixed-budget final checkpoints, not best-checkpoint selections. Each
`logs/baselines/<source>-1000/validation.json` records the checkpoint SHA256,
seed, capped error, raw return and termination counts. Evaluation uses seed 1
and identical starts; eight opportunities are not eight independent training
seeds. The policies have not learned a full flip. Mean evaluation lengths end
before the inverted reference frames 116..132; `anchor_ori` was never triggered.

The original `isaacsim-1000` experiment was interrupted because the Isaac sensor
omitted the rotational contribution to velocity at the offset IMU site. Its
`INVALIDATED.md` excludes those weights. The corrected run starts from scratch.
Isaac Sim's corrected native PPO/checkpoint/reset suite passed 8 tests in 45.70 s;
Gym's suite passed 8 in 25.45 s. Genesis passed 8 in 61.29 s, followed by the
new IMU reference assertion (1 passed in 6.51 s); Motrix also passed that assertion.
The final CPU baseline tests passed 10, with 1 native test deselected.

Commands for the repository gates, from this checkout:

```bash
PATH=/home/wsm/.vscode-server/cli/servers/Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1/server:$PATH UV_NO_SYNC=1 make check
UNILAB_LOCAL_UNISIM=/home/wsm/wang-sm/unisim-unidr-backends OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 UV_NO_SYNC=1 make test
uv run --no-sync pytest -q tests/tasks/test_g1_flip_baselines.py
git diff --check
```

`make check` ran: Ruff and Mypy passed; Pyright reports the same 10 baseline errors
in `base/reset_state.py`, `managers/_buffers/circular_buffer.py`, and
`tasks/motion_tracking/common/manager_terms.py`. No strict checks were weakened.
The first full test run exposed missing Go2/A2/Allegro assets, the undeclared local
UniSim install profile, and the stale generated support matrix. Registered asset
pulls, `UNILAB_LOCAL_UNISIM`, and the normal generator resolved those failures;
the final full run passed 1601 tests with 27 skips and 844 deselected in 45.84 s.
The baseline regressions cover asynchronous episode ends, output overwriting
training inputs, and malformed/incomplete contracts failing before env creation.
Gate logs are `logs/baselines/check-final.log` and `test-final.log`.

Checkpoint audit (CPU, after all four runs complete):

```bash
uv run --no-sync python -c 'import json, torch
from pathlib import Path
for name in ("motrix-1000", "isaacgym-1000", "genesis-1000", "isaacsim-1000-corrected"):
    p = Path("logs/baselines") / name
    first = torch.load(p / "model_0.pt", map_location="cpu", weights_only=False)
    last = torch.load(p / "model_999.pt", map_location="cpu", weights_only=False)
    assert last["iter"] == 999
    for key in ("actor_state_dict", "critic_state_dict"):
        assert all(torch.isfinite(v).all() for v in last[key].values())
        assert any(not torch.equal(first[key][k], v) for k, v in last[key].items())
    assert {int(s["step"]) for s in last["optimizer_state_dict"]["state"].values()} == {20000}
    summary = json.loads((p / "run_summary.json").read_text())
    assert summary["total_env_steps"] == last["unilab_logger_state"]["tot_timesteps"] == 1536000
    print(name, "PASS: changed finite actor/critic; 20000 updates; 1536000 transitions")'
```

From the UniSim checkout, `make check` passed Ruff and 290 tests (16 skipped);
`make package` built the source archive and wheel. The opt-in command
`UNISIM_TEST_ISAACGYM=1 uv run --no-sync pytest -q tests/test_isaacgym_reset.py`
passed 14 native tests. The Genesis inverted/moving reset test also passed.

## Remaining Work

Fixed-start evaluation has failed before inversion; learning-rate and
phase-coverage diagnostics need further training investigation. Longer
training alone is not evidence that the current setup will learn the full clip.
Full-clip behavioral success, multiple training seeds, target-size 2048 pools,
four-GPU operation and final MuJoCo sim2sim are separate milestones. Unified
rollout integration remains deferred until all source validations are accepted.
No normalizer aggregation, scheduler, probe, learner IPC or recovery contract was
added. No PR was created and `make test-all` has not been claimed as passing.
