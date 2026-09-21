# Central four-source PPO delivery

Accepted architecture: [ADR 0002](adr/0002-central-synchronous-ppo.md).
Runtime base `79418e0cbff7b95fe6e49709b194454701ba20fa`, Apache-2.0;
the local implementation is an uncommitted diff on `codex/unidr-c1-rollout`.
The existing nine-file C1 delivery remains intact; only its common validated
GAE/batch preparation was extracted behind its original fixed-normalizer checks.

The approved umbrella is split into independent local children:

| Child | Result | Owned changes |
| --- | --- | --- |
| U1 | Successful task profile on four backends | UniLab `unidr*.yaml`, reference/config tests and record |
| U2 | Native central inference/normalization and source-local GAE | `algos/synchronous_ppo.py`, its tests, shared C1 helper, CONTEXT/ADR |
| U3 | Synchronous environment services and cleanup | `ipc/synchronous_env.py`, its tests |
| U4 | Learner lifecycle, logs and checkpoint recovery | `algos/synchronous_runner.py`, its tests and record |
| U5 | Hydra/factory/device/asset wiring and integration | UniLab training owner, thin CLI, wiring tests and experiment record |
| U6 | Fixed-final MuJoCo strict playback/video | UniLab holdout owner, thin CLI, tests and record |
| U7 | Independent budget audit and curves | `logging/synchronous_report.py`, tests and thin report CLI |

Each child has one main result, fewer than 15 files and at most 800 net hand-written
lines. No PR or release was created. Public additions belong to this local runtime
checkout; the UniLab consumer keeps RSL-RL 5.0.1 installed. The runtime's lock also
uses 5.5.0; both versions are tested independently.

`CentralCollector` retains raw dictionaries (including original group names),
action/value/log-prob/Gaussian parameters, unmodified rewards, both done flags,
final observations, bootstrap, source/env/episode/segment identities, window and
policy versions, per-step normalizer versions, and recovery generation. Four
services receive one action slice each; all responses must arrive before the next
control step. No policy runs in an environment service.

Policy weights remain fixed for 24 steps. Native normalizer updates consume only
the complete post-step observation batch, including autoresets, once per step;
timeout values use final observations with the updated statistics. True termination
overrides timeout. Ordinary bootstrap comes from the next sampled old value; the
last step uses the window-end critic. Native PPO computes GAE inside each source's
continuous columns, then a separate full batch normalizes advantages globally.
Optimization freezes normalizers explicitly and keeps all old sampling quantities.
Every native update is checked for five complete epochs and 20 Adam steps.

Raw data remains available through `runner.latest_window` during a run; it is not
written for all 983 million transitions. Persistent JSONL/TensorBoard records
source budgets, metrics, version stamps and optimizer/sample counters. Checkpoints
are complete boundaries only; resume resets physics into a new generation and
restores learner/normalizer/optimizer/LR/RNG/config identity, not simulation state.

PolySim reference: `EmboMaster/PolySim` commit
`0fdb349785479b5c0c30d3c676fae67025b8d34a` (root MIT license; setup metadata retains
ASAP/BSD wording; no root lock or AGENTS found during audit). Its `env_client.py`
source slices and concurrent scatter/gather, environment-only `hydra_server.py`,
and environment-column storage guided the organization. None of its dual-optimizer,
pre-step timeout, retry, missing-field fallback or remainder-dropping PPO paths
was transplanted. The PPO and GAE implementation remains the installed RSL runtime.

Run from this checkout:

```bash
UV_PROJECT_ENVIRONMENT=.venv-rsl501 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 COVERAGE_FILE=/tmp/unidr-rsl501-final-coverage uv run --no-sync pytest --cov=src/uni_rl
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 COVERAGE_FILE=/tmp/unidr-rsl55-final-coverage uv run --no-sync pytest --cov=src/uni_rl
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy src/uni_rl
PATH=/home/wsm/.vscode-server/cli/servers/Stable-520fb30b2d3d324b4cb2342f6e88e2cd93751de1/server:$PATH uv run --no-sync pyright
```

The focused tests include a real four-spawn-service fake-data update, a stepwise
native PPO oracle (statistics, old values/log-probs, returns, losses and parameters),
hand-computed C1 GAE, every-epoch sample identities, int64 normalization above
`2^24`, rejected stale revisions/payloads, and delayed/shared-memory cleanup.
These validate fake-driven real PPO; robot support, physical capacity, full training
and final transfer require the separate UniLab experiment evidence.

Final runtime gates on 2026-09-16: **482 passed, 35 skipped, 3 deselected** under
both RSL-RL 5.0.1 and 5.5.0, with 65% repository coverage (separate coverage files).
The final runs took 41.89 and 44.27 seconds respectively.
Ruff lint/format, Mypy (80 source files), and Pyright (zero errors/warnings) passed.
Matplotlib is optional in the runtime environment; all 29 report tests, including
PNG/PDF rendering, passed in the actual UniLab consumer environment. The independent
capacity report verified all 100 completed iterations and the actual Adam/model
state. Reported recovery histories follow explicit checkpoint ancestry, trim any
abandoned parent tail, and require complete, contiguous committed iterations.

Final independent review also added duplicate optimizer parameter-ID rejection
before recovery and a clone of retained tail observations. The regression tests
cover corrupt same-shaped parameter aliases and bit-exact preservation across
environment buffer reuse. These two guards were added after the formal service
started; that process keeps its already-loaded code. Its fresh run does not use
recovery, and its synchronous services allocate new result arrays at each step.

On 2026-09-16 the user selected the earlier periodic `model_5000.pt`, which contains
5001 complete updates. Offline `report(..., expected_iterations=5001,
num_envs=1024, final_checkpoint=run_dir / "model_5000.pt")` now audits that explicit
committed prefix, while retaining `configured_max_iterations=10000` and recording
excluded tail bytes. Default full-run audit behavior stays strict. The original
configuration, manifest, metrics and checkpoint are never rewritten by the audit.
The CLI exposes the same explicit flags; this is offline selection by the user,
not selection based on MuJoCo performance.
