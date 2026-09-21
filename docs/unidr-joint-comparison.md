# Joint comparison after the four single-source runs

This is the original September 2026 follower design for the 1024-environment /
20000-update single-source queue. Its sample and optimizer comparisons below
refer to that design. The later 4096-environment / 5000-update singles have the
same batch and optimizer budgets as the joint run; see [single-source comparison](unidr-single-comparison.md).
For current setup and direct joint training, start with the [README](../README.md).
Commands use the bundled UniDR layout; referenced `logs/` records remain external
experiment artifacts.

This follow-on was requested after the independent comparison queue started.
It does not change or restart that queue. After all four 1024-environment,
20000-iteration runs and their fixed MuJoCo reference videos finish, a new
`unidr_comparison` owner trains one shared PPO policy from scratch for 5000
iterations. Each source contributes 1024 environments × 24 steps per iteration.
The final checkpoint is fixed at `model_4999.pt`: 491520000 transitions and
100000 optimizer updates, with 5 complete epochs and 4 minibatches per epoch.

Actions, rewards, motion, assets, terminations, normalization and adaptive PPO
settings inherit the current single-GPU joint owner. All sources disable
self-collision. True termination takes precedence over simultaneous timeout;
pure timeout still bootstraps. No parameter DR is enabled. MuJoCo remains a
holdout, and its measured policy performance never decides admission or model
selection. The four-source batch is four times the single-source batch.

The owner boundary follows UniLab ADR-0003 and the runtime
[central PPO ADR](../vendor/unilab_rl/docs/adr/0002-central-synchronous-ppo.md): UniLab validates task/config/assets;
`uni_rl` measures services, aggregates iteration logs, updates PPO and audits
saved budgets. The shell only waits and sequences existing entrypoints.

## Gate and launch

```bash
bash scripts/run_joint_comparison.sh \
  /absolute/completed-or-running-single-root /absolute/new-joint-root \
  existing-single-queue.service EXACT_32_HEX_INVOCATION_ID
```

Run the follower in its own user systemd unit with an ordering/reference
dependency on the already active single queue. The follower polls completion;
`After=` alone is not a completion wait. On this host `Wants=` plus `After=`
keeps the transient source unit loaded after its successful exit. Verify the
source is already active before registering the follower; never reissue the
single queue's launch command. The follower has no automatic restart policy.

The shell requires the same invocation, inactive/dead state, MainPID=0,
successful exited status and a nonzero exit timestamp. Missing units, changed
invocations and failures stop it. It then locks the completed queue and checks
all four native final-budget audits, manifest/config/asset equivalence, fixed
video hashes and full video decode. It rejects an existing joint output root.
Cancelling the follower only stops its own active stage, not the single queue.

Each stage writes `status.tsv` and a log in the new joint root. Training writes
to `train/`; the final runtime report writes to `report/`. No checkpoint from
the single runs is loaded. Task configuration, source manifests and saved
optimizer/normalizer counts remain the authoritative budget evidence.

## Final joint reference video

After the joint run completes, its fixed final can use the same front/reference
renderer as the four single-source videos:

```bash
MUJOCO_GL=egl uv run --no-sync python scripts/play_unidr_holdout.py \
  /absolute/completed-joint-root/train/model_4999.pt \
  --expected-iterations 5000 --front-reference \
  --output /absolute/completed-joint-root/mujoco-front-reference
```

This option re-audits the full fresh joint budget and strict task/model/asset
contract before creating the environment. It records 1000 real MuJoCo steps,
classic background, front camera, cyan phase-matched reference and no text.
The shared renderer produces `front-reference.mp4`, a preview and telemetry,
with 720p/50 FPS/20-second full-decode verification. The existing follower does
not automatically invoke this post-training command. Holdout behavior is
reported without changing configurations or selecting a different checkpoint.

## Source timing

The runtime JSONL and TensorBoard record these sums over all 24 steps of each
iteration, separately for Isaac Sim, Isaac Gym, Genesis and Motrix:

| Metric | Measurement |
|---|---|
| `env_step_seconds` | Worker wall time inside `env.step`, including task/physics/observations |
| `request_response_seconds` | Parent request thread starts until full response arrives; includes IPC, excludes thread-pool queueing |
| `barrier_wait_seconds` | A complete response waits until the last source response arrives |
| `timing_steps` | Exactly 24 successful source steps |

`collect_seconds` remains total global collection/batch preparation time;
`learn_seconds` measures the one shared PPO update. Source durations overlap
and must not be added as total iteration time. No additional GPU synchronization
is inserted: worker time is observable call wall time, not pure GPU kernel time.
The report produces source performance curves plus `timing.png`/`timing.pdf`,
and timing coverage, totals, means, minima, maxima and p95 values in its audit.
Historical logs without timings remain readable; malformed partial timings
are rejected. An invalid collection window never produces a new iteration log.

## Final comparison axes and metrics

Align policies by global transitions: native TensorBoard iteration `i` maps to
`(i + 1) * 24576`, while joint JSONL already records `total_transitions`.
Each policy receives 491520000 transitions, but the joint policy sees only
122880000 from each source. Single-source policies perform 400000 optimizer
steps each; the joint policy performs 100000 with minibatches four times larger.
This is a fixed-sample comparison with deliberately different update budgets.

Compare native `Train/mean_reward` and `Train/mean_episode_length` with the
corresponding joint `Source/<backend>/episode_return` and `episode_length`.
Both are rolling means of the last 100 completed episodes. Joint `reward` is
mean reward per transition, so it must not be plotted against native episodic
return as the same quantity. The joint global episode average follows episode
completion frequency and is not an equal-weight average of four source means.

Use the sum of `Perf/collection_time` and `Perf/learning_time` for native loop
timing, and `collect_seconds + learn_seconds` for joint loop timing. Native RSL
includes GAE in learning time; the joint runner includes GAE in collection/batch
preparation. Separate components therefore have different boundaries and cannot
establish a learner speedup across runners. Native collection time also includes
central inference and is not equivalent to joint worker-only step time.
TensorBoard `Train/*/time` uses integer
elapsed seconds as its step, not iteration; it must not be converted to samples.
The native reward-term logs are rates before multiplication by control dt;
`Episode_Reward/*` and `Episode_Termination/*` use reset-group aggregation.
`Metrics/motion/error_*` describes states at reset, not full-trajectory error.
These auxiliary logs alone cannot establish complete flip or landing success.

## Validation and limitations

Run focused tests through `uv run --no-sync pytest`, using paths relative to the
UniDR repository root and its bundled runtime:

- UniLab: `tests/training/test_joint_comparison.py`,
  `tests/scripts/test_joint_comparison_queue.py`, `tests/tasks/test_g1_flip_comparison.py`.
- Runtime: `vendor/unilab_rl/tests/ipc/test_synchronous_env.py`,
  `vendor/unilab_rl/tests/algos/test_synchronous_runner.py`,
  `vendor/unilab_rl/tests/logging/test_synchronous_report.py`.

The queue tests use stub commands; runtime integration uses four spawned fake
sources and real PPO. Neither proves completion of the requested real training.
The active single runs and the later joint run require their own completed
audits. Exact launch details and executed validation are recorded under
`logs/comparison-20260916-validation` and the experiment roots. Training-code
provenance for the follower is recorded separately from the earlier singles.
