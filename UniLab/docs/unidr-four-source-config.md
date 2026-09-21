# Aligned G1 Flip Four-Source Configuration

The four `g1_flip_tracking/unidr_<backend>` task owners use the successful native
Motrix flip profile. They inherit `unidr.yaml`, then native `motrix.yaml` and
`mujoco.yaml`. The historical `base.yaml` and `*_baseline.yaml` owners are separate
experiments with incompatible action scales and observation normalization.
The inheritance name `mujoco.yaml` supplies task semantics; MuJoCo is excluded
from the four source lists and from the validation commands below.

UniLab owns these task/source configs and their factory adaptation. `uni_rl`
owns central PPO, collectors, IPC and training state. This follows
[ADR-0003](sphinx/source/adr/ADR-0003-task-owner-and-config-compose-contract.md).
The underlying UniLab base is `b4e6b58fe0861a435fd19c0f0206bd84f4427a9c`.

## Profile and Provenance

The reference is
`logs/baselines/motrix-aligned-1024-10000-20260915-2108/run_config.json`,
SHA256 `98f4e7ecdfe1d2e3da1a5c7519394078b8f683cdc8504a9ec940121aefa3ea53`.
All four sources preserve its task, reward and native PPO settings. The user
subsequently requested disabling self-collision in all four training sources;
this is an explicit physics change from the recorded experiment, not a change
to its historical snapshots or checkpoints. The task-owner/factory boundary
continues to follow ADR-0003 above. The common owner now sets
`env.motrix_disable_self_collision: true` and
`env.genesis_enable_self_collision: false`; Isaac Sim and Isaac Gym already
disable robot self-collision in their adapters. Robot-ground contacts remain
enabled. The native single-source `motrix` owner retains its original collision
settings. Both single-source and central PPO wrappers give true termination
precedence over simultaneous timeout, with no bootstrap for that transition.
The MuJoCo holdout keeps its native collision profile: its local task comparison
excludes these two source-only adapter fields while retaining strict policy and
remaining task checks. Both historical and new manifests remain immutable.

The unchanged training profile is:

- 1024 environments, 10000 iterations, 24 steps, 5 epochs, 4 minibatches,
  checkpoint interval 500, adaptive learning rate and empirical actor/critic
  observation normalization.
- G1 29 DoF, actor observation 160, critic observation 286, 29 actions;
  per-joint action scales, simulation dt 0.005 s, control dt 0.02 s.
- `motions/g1/flip_360_001__A304.npz`, 225 frames at 50 Hz, fixed start sampling,
  zero pose/velocity/joint-position reset noise and native clip wrapping.
- Native rewards and terminations, including action-rate weight -0.005,
  end-effector height reward, undesired-contact reward/termination and
  anchor-orientation threshold `1.0e9`.

Declared backend materialization settings include device IDs, Isaac Sim
worker timeout, Genesis integrator and the explicit self-collision controls above.
Isaac Sim, Isaac Gym and Genesis set
`geom_names: null`: their adapters do not expose geometry IDs, while tracking
uses the same named bodies. Isaac Sim's first native-profile reset exposed this
binding gap before physics construction. The
robot model, joint/body ordering and clip are unchanged. Strict Sim2Sim remains
enabled; the older fixed-normalizer/scalar-action checkpoints are rejected.

`tests/tasks/test_g1_flip_unidr.py` pins the reference semantic SHA256 to
`558789835ea6a0025b5b2508303b27d4d4f0dc72efaf96f860aa2094e2ebabde`.
It hashes resolved `env`, `reward` and `algo` as sorted compact JSON, excluding
the declared backend settings (including the separately asserted collision
controls) and geometry binding and converting `algo.load_run`
to a string (the saved CLI value is numeric `-1`). The check therefore detects
future shared-owner drift as well as backend-specific changes. Separate checks
compare complete Sim2Sim snapshots and preserve Motrix's geometry bindings.

## Source and Device Layouts

Select `task=g1_flip_tracking/unidr_four_gpu` or
`task=g1_flip_tracking/unidr_single_gpu` for the aggregate config.
`unidr.sources` is ordered and contains `name`, source-owner `task`, `device` and
`env_overrides`. `unidr.learner_device` also resolves `training.device`.

| Source | Owner suffix | Four-GPU device | Single-GPU device |
| --- | --- | --- | --- |
| Isaac Sim | `unidr_isaacsim` | `cuda:0` | `cuda:0` |
| Isaac Gym | `unidr_isaacgym` | `cuda:1` | `cuda:0` |
| Genesis | `unidr_genesis` | `cuda:2` | `cuda:0` |
| Motrix | `unidr_motrix` | `cpu` | `cpu` |
| Central learner | — | `cuda:3` | `cuda:0` |

Motrix physics stays on CPU. Per-source device IDs belong to each environment
override; setting only the aggregate `training.sim_backend` is insufficient to
select the source owners. Each source has capacity `algo.num_envs`; the central
collector chooses the active rows. The source/rollout timeout is 180 seconds and
the startup timeout is 600 seconds.

## Focused Validation

From `/home/wsm/wang-sm/UniLab-unidr-backends`:

```bash
export UNILAB_LOCAL_UNISIM=/home/wsm/wang-sm/unisim-unidr-backends
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
uv run --no-sync pytest -q tests/tasks/test_g1_flip_unidr.py
uv run --no-sync ruff check tests/tasks/test_g1_flip_unidr.py
uv run --no-sync ruff format --check tests/tasks/test_g1_flip_unidr.py
UNIDR_REAL_BACKEND=motrix uv run --no-sync pytest -q -s -m '' tests/tasks/test_g1_flip_unidr.py::test_real_aligned_flip_reference_phases
UNIDR_REAL_BACKEND=isaacsim uv run --no-sync pytest -q -s -m '' tests/tasks/test_g1_flip_unidr.py::test_real_aligned_flip_reference_phases
UNIDR_REAL_BACKEND=isaacgym uv run --no-sync pytest -q -s -m '' tests/tasks/test_g1_flip_unidr.py::test_real_aligned_flip_reference_phases
UNIDR_REAL_BACKEND=genesis uv run --no-sync pytest -q -s -m '' tests/tasks/test_g1_flip_unidr.py::test_real_aligned_flip_reference_phases
```

The opt-in test constructs two real environments and injects reference frames
0, 100, 124, 180 and 224 through the normal reset sampler. It checks joint/body
state, offset-IMU velocity, policy dimensions and finite observations/rewards
after one zero-action step per phase. Native termination results are reported;
the test does not change thresholds. This is state/contract validation, without
PPO updates or claims about learned flip success.

Validation on 2026-09-16 passed 10 new config checks, or 23 tests with the existing
`tests/tasks/test_g1_flip_baselines.py` suite (2 opt-in native tests deselected).
Ruff lint/format and `git diff --check` passed. Direct comparison with the saved
successful `run_config.json` also matched all four complete semantic profiles
and passed the strict Sim2Sim resolver.

All four real phase smokes passed: Motrix 0.65 s, Isaac Sim 11.75 s, Isaac Gym
6.81 s and Genesis 6.57 s. Each reported no termination after its one step at
each reference phase. Genesis emitted existing mesh convex-hull fallback warnings
and one xacro deprecation warning. These checks ran serially on the local RTX 3090
with CPU thread limits of 2. Four-GPU hardware placement, concurrent source
collection and central training are separate integration checks.
