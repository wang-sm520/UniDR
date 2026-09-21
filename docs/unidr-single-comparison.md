# Four independent G1 flip comparisons

The first sections preserve the original 1024-environment / 20000-update
comparison design. [Run](#run) also records the later 4096-environment /
5000-update variant. Dated results describe those specific experiments, not a
fresh validation of this source bundle. Current commands run from the UniDR
repository root after the [README setup](../README.md); historical `logs/`
artifacts are not bundled.

The requested comparison trains four independent policies sequentially, in the
order Motrix, Isaac Sim, Isaac Gym, Genesis. Each native PPO run starts fresh,
uses 1024 environments and 20000 updates, and evaluates only its fixed final
`model_19999.pt` in MuJoCo after training. The user repeated Isaac Gym in the
requested order; the sequence includes all four distinct requested sources.

The `<backend>_comparison` task owners inherit the current `unidr_<backend>`
profiles. Actions, rewards, native terminations, reference motion, assets,
observation dimensions, network, normalization and adaptive PPO settings are
shared with the joint-training profile. Self-collision is disabled on all four;
pure timeout bootstraps from final observation, while true termination takes
precedence over simultaneous timeout. No parameter DR is enabled. Environment
seeds match the joint services: Motrix 5, Isaac Sim 2, Isaac Gym 3, Genesis 4;
the learner seed is 1. Iterations increase from the joint owner's 10000 to 20000.

Each single-source update contains 24576 transitions, with 5 epochs and 4
minibatches (6144 samples each). Expected completed budget per source is
491520000 transitions and 400000 Adam updates. Four sources require 1966080000
transitions and 1600000 Adam updates in total. The joint update batch is four
times larger. Historical joint checkpoints preceded the self-collision change;
they are not an identical-physics control for this new experiment.

Ownership follows [ADR-0003](sphinx/source/adr/ADR-0003-task-owner-and-config-compose-contract.md):
task/config/asset fingerprints and MuJoCo playback are UniLab-owned; native PPO
and saved-budget audit are `uni_rl`-owned. The shell only sequences commands.
No new rollout, IPC or learner implementation is introduced.

## Run

The 2026-09-20 comparison uses **Genesis → Motrix → Isaac Gym → Isaac Sim**,
4096 environments and 5000 iterations per independent fresh policy:

```bash
bash scripts/run_single_comparison.sh /absolute/new/experiment-4096-5000 \
  4096 5000 genesis,motrix,isaacgym,isaacsim --train-only
```

`--train-only` retains preparation and final-budget audit, omitting playback.
Only a successfully completed and audited run advances the queue. Each run has
98304 transitions per iteration, 491520000 total transitions and 100000 Adam
updates. The fixed final is `model_4999.pt`; periodic saves remain every500
iterations using native zero-based filenames. All other comparison parameters
remain unchanged, including fresh initialization, environment/learner seeds and
native PPO (no adaptive multi-source scheduling/probes). The queue records its
requested source order, budget and playback mode in `queue_config.json`.
Existing calls without the extra arguments retain the historical order and
post-training playback described below.

From the UniDR repository root, the original default queue is:

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
bash scripts/run_single_comparison.sh /absolute/new/experiment
```

The queue refuses an existing experiment directory. For each source it snapshots
the effective config and asset hashes, runs native training to process exit,
verifies completed summary and final model/Adam/normalizer budgets, then records
1000 real MuJoCo steps. Only after successful recording and full video decoding
does it advance. A failed stage stops the queue and records `failed` in
`status.tsv`; it does not blindly restart or load an earlier checkpoint.
`queue.pid` and `stage.pid` identify live process handles, not completion proof.

Single-task entrypoint:

```bash
uv run --no-sync train --algo ppo --task g1_flip_tracking --sim isaacsim --profile comparison training.device=cuda:0 training.log_dir=/absolute/new/run
```

For auditable playback, prepare that run with
`uv run --no-sync python -m unilab.training.single_comparison isaacsim /absolute/new/run`
before training. Do not override the string `algo.load_run="-1"` with numeric -1.

The native run summary stores final iteration index 19999. The audit checks that
this means 20000 completed updates; file existence alone is never sufficient.
The final video is `<source>/mujoco-front-reference/front-reference.mp4`, with
classic MuJoCo background, frontal camera, silver actual robot, cyan reference,
720p/50 FPS/20 seconds and no text overlays. Telemetry distinguishes early
termination, timeout, reference reset and landing candidates. Evaluation never
selects a checkpoint or feeds training configuration changes.

## Validation

Configuration/queue tests cover owner equivalence, fresh initialization, strict
Sim2Sim, exact command/materialized-config equivalence, stage order, failure
stop, overwrite rejection and owned-process interruption. Native 2-env/2-update
training and final-budget audit passed for all four backends. A real 20-second
MuJoCo reference video from the Motrix smoke checkpoint decoded all 1000 frames;
its 20 early failures are expected for a two-update engineering check.

Executed artifacts and logs are in `logs/comparison-20260916-validation`.
UniLab non-slow suite: 1707 passed, 27 skipped, 849 deselected. Actual RSL 5.0.1
runtime suite: 553 passed, 8 skipped, 3 deselected. Ruff/format/mypy passed.
Full UniLab Pyright still reports 10 pre-existing errors in unmodified files;
focused checks for the new owner modules pass using the installed VSCode Node.
No SDK/system dependencies were installed or changed.

Base SHAs (plus preserved local changes): UniLab
`b4e6b58fe0861a435fd19c0f0206bd84f4427a9c`, runtime
`79418e0cbff7b95fe6e49709b194454701ba20fa`, UniSim
`4270aa81d868744980db90dac6dd959d3f542f50`.
This implementation/smoke evidence does not establish completion or task success
of the four requested 20000-update runs; those require their actual final audit,
videos and interpretation of the recorded policy behavior.
