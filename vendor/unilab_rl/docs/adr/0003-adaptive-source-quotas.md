---
status: accepted
---

# Optional adaptive source quotas

On 2026-09-20 the user authorized an adaptive interface, then explicitly deferred
training: this delivery develops and tests the interface only, with no training
launch. The intended later scale is 4 × 1024 and 5000 iterations. This extends ADR-0002;
the existing fixed synchronous path remains available. No MuJoCo feedback is used.

Resident pools remain fixed. Allocate 96 whole-pool steps per iteration with
source-local contiguous storage, then reuse existing GAE and the single PPO
learner (5 epochs × 4 minibatches). A wave steps only sources with remaining
quota. Inactive sources pause physics and contribute no samples or normalizer
statistics. Normalizers update once per active post-step batch. Timeout final
observations use that updated snapshot; source tails use the window-end snapshot.
True termination takes precedence over timeout. Version counts track actual waves.

Every 100 completed iterations, frozen-policy probes reuse the four training
pools with identical frame-zero reset, zero extra DR, seed and 224-step horizon.
The task owner supplies fresh per-row tracking error. Native early termination
ends that row's evaluation opportunity: remaining errors are padded to 0.5 m and
remaining rewards to zero. Tracking error is capped at 0.5 m; reward normalization
uses fixed common scales recorded in configuration. Survival means no native
termination/timeout during the horizon, not the MuJoCo five-second holdout test.
Probe samples never enter training storage or normalizer statistics. All pools
reset and training episode IDs advance afterward; no physics state restoration
is claimed. Factories expose an explicit optional probe capability over IPC.

Runtime owns EMA, bounded ratio projection, integer quotas, probe orchestration,
logging and checkpoint state. UniLab owns task metrics, matching initial states,
configuration and factories. Ratios start equal; E/S/R weights start .5/.25/.25,
EMA alpha=.2, reference weights are fixed equal. Missing or nonfinite metrics hold
the complete scheduler state. Target changes are at most .02, bounds [.1,.7].
Integer actual shares also obey these bounds: at budget96 one source can change
at most one step per update (two steps would exceed .02). Stable-all-source
metric-weight adaptation is separately configured, with positive failure weight.

Checkpoint only after PPO and any due probe/scheduler transaction finish. Save
schedule state, scales, version/sample counters and RNG; recovery resets pools
into new episodes. A failed request/window/probe aborts, without stateful retries.

Reviewable children (local delivery, no PR/publication requested): A1 scheduler
and integer solver; A2 selective IPC; A3 quota collection; A4 independent probes;
A5 runner/config/checkpoint integration. Each targets <=15 files /800 net handwritten
lines, with independent tests. Real backend training smoke and 4×1024 capacity,
including an actual quota change, remain future gates before the deferred fresh
5000 iterations (491,520,000 training transitions, 100,000 optimizer steps).
Hardware here is one RTX3090; four-GPU
configuration does not imply four-GPU execution evidence.
