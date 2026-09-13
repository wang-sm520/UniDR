# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `uni_rl.algos.rsl_rl_validation.run_bounded_ppo` runs the unchanged stock
  learning loop to an update or post-warmup wall-clock budget, checks finite
  updates and source sample counts, and saves a final checkpoint plus JSONL
  evidence at a complete-update boundary. Numeric environment substep timings
  are accumulated per source, including autoresets inside `step`.
- Generic synchronous multi-source environments through
  `uni_rl.ipc.multi_source_env`: `EnvSourceSpec`, `MultiSourceOptions`,
  `make_multi_source_env`, and source-attributed `MultiSourceEnvError`.
  Non-daemon spawn workers use fixed source slices, preallocated double-buffered
  shared memory, strict descriptor/data validation and sequence barriers, with
  all-source failure poisoning and Linux nested process-group cleanup. Isolated
  per-source CPython resource trackers also reclaim source/child Python-tracked
  SHM after native crashes, without scanning or unlinking unrelated learner SHM.
- Source-namespaced logs, immutable source counter/timing/PID/RSS statistics,
  global-ID reset scattering, and per-source episode-length/NaN-guard forwarding.
  Optional `SupportsEpisodeLengthBufferProtocol` and
  `SupportsSourceStatisticsProtocol` describe these additive capabilities;
  `NanGuard.cfg` exposes a copy for process-local guard reconstruction.
- Real-spawn CPU fault/cleanup tests and stock PPO 24-step update/checkpoint
  integration with full-epoch source-label coverage. Small correctness fixtures
  do not establish simulator-scale or GPU acceptance.

### Changed

- Accept infinite bounds for unbounded action spaces in multi-source startup
  descriptors, while retaining NaN/order checks and finite transition data.
- Existing `uni_rl.ipc` and `uni_rl.utils` exports load lazily so numpy-only
  multi-source transport does not import torch-based training primitives or
  device helpers at module import time.

## [1.2.0] - 2026-09-10

### Changed

- No code changes since 1.1.3. This minor bump re-anchors the public-contract
  changes shipped in 1.1.3 (the new off-policy actor adapter API and the
  removal of the `uni_rl.algos.hora` namespace) under a minor version, per the
  semver discipline that public-contract changes require at least a minor
  bump. Consumers pinning `~=1.1` should review the 1.1.3 changelog entries
  before upgrading.

## [1.1.3] - 2026-09-09

### Added

- Generic off-policy actor adapter registry
  (`uni_rl.offpolicy.actor_adapter.OffPolicyActorAdapter`,
  `register_offpolicy_actor_adapter`, `get_offpolicy_actor_adapter`) so external
  packages can plug custom actor construction, exploration sampling, privileged
  context extraction, and inference-context slicing into the generic off-policy
  runtime. Spawn-safety is provided by the new optional
  `algo.actor_adapter_modules` config key (also on `OffPolicyRuntime`), whose
  dotted modules are imported in both the learner process and the spawn
  collector subprocess.

### Removed

- The HORA implementation (`uni_rl.algos.hora`) and its hardcoded `hora_sac`
  branches in the generic off-policy runtime moved to the standalone
  `sharpa_rl_unilab` repository. The old import namespace is removed without a
  forwarding shim; consumers register an `OffPolicyActorAdapter` instead.

## [1.1.1] - 2026-09-08

### Added

- Optional PPO runtime runner selection and `TrainingStateOnPolicyRunner`, with
  an explicit versioned checkpoint envelope for downstream-owned curriculum
  progress. Existing PPO runners are unchanged; requested state restoration
  rejects missing or incompatible envelopes rather than restarting a curriculum
  ([#16](https://github.com/unilabsim/unilab_rl/issues/16)).

### Removed

- The HIM-PPO implementation and tests moved to
  [legged-manipulation_unilab](https://github.com/unilabsim/legged-manipulation_unilab)
  under [UniLab #1528](https://github.com/unilabsim/UniLab/issues/1528).
  The old import namespace is removed without a forwarding shim.

## [1.1.0] - 2026-09-06

### Fixed

- `uni_rl.utils.device.resolve_backend_process_device` now treats `newton`
  like `mjwarp`: both backends require an explicit CUDA process device shared
  with the learner, so off-policy collectors invoke the injected
  `backend_device_binder` for `newton` runs instead of silently skipping the
  binding (previously the spawned collector built the backend without a bound
  device).
- `DpRankSupervisor` now re-runs the downstream owner's original
  `sys.argv[0]` entry script for spawned off-policy ranks instead of redirecting
  it to the nonexistent `uni_rl/scripts/` directory, restoring multi-GPU
  SAC/TD3 launches from installed consumers such as UniLab (#12).

## [1.0.0] - 2026-09-04

First stable release. The public contract (`uni_rl.env_contract` protocols and
factory signature, runner / `runtime_resolver` conventions, algorithm config
keys) is now covered by semantic versioning.

### Added

- `README_zh.md`（简体中文 README）and a Citation section (UniLab paper,
  `jia2026unilab`) in both READMEs.

### Changed

- Rewrote the README: documents the relationship with UniLab, PyPI
  installation, env-contract usage, and development commands. PyPI is the
  release channel; TestPyPI instructions were removed.

## [0.3.0] - 2026-09-04

### Added

- Optional env algo-capabilities extension point in `uni_rl.env_contract`:
  `EnvAlgoCapabilitiesProtocol` (per-dimension `action_low` / `action_high`
  bounds and `joint_names`, all fields optional), the
  `SupportsAlgoCapabilitiesProtocol` provider protocol, the frozen
  `EnvAlgoCapabilities` default carrier, and the `get_algo_capabilities(env)`
  helper that falls back to an all-`None` default for envs that do not provide
  capabilities. Intended for algorithm-side features such as per-joint action
  scaling and symmetry augmentation; cold-path reads only (runner init, dim
  probe). (UniLab issue #1487)

## [0.2.0] - 2026-09-04

### Changed

- Grouped algorithm packages under `uni_rl.algos` (`appo`, `fast_sac`,
  `fast_td3`, `flash_sac`, `him_ppo`, `hora`, `rsl_rl` wrappers, `common`).
- Added CI (ruff / mypy / pyright / pytest+coverage) and release workflows,
  plus `AGENTS.md` contributor guidance.

## [0.1.0] - 2026-09-04

### Added

- Migrated the RL algorithm and async runtime layer from UniLab into the
  standalone `uni_rl` package: PPO/APPO/SAC/TD3/FlashSAC/HIM-PPO/HORA runners,
  learners, collectors, IPC, and training logging.
- Decoupled `uni_rl` from `unilab` via the injected env contract
  (`uni_rl.env_contract.EnvFactory` / `EnvProtocol`) and dependency injection;
  `uni_rl` never imports `unilab` / `unisim`.
- Forwarded `backend_device_binder` through runner builders.
