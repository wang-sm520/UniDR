---
status: accepted
---

# Central inference across synchronous environment services

The user authorized the four-source plan on 2026-09-16: one actor/critic sends actions to isolated environment services and waits for every source before advancing the control step, borrowing PolySim's environment-axis scatter/gather organization. Native PPO normalization updates exactly once from the complete post-step observation batch; timeout final observations use those new statistics without contributing to them, while true terminations override timeout flags. This deliberately supersedes C1's frozen-normalizer restriction for the new synchronous runner, preserving the successful native Motrix training semantics and existing PPO optimizer/GAE.

Each request has an acknowledged source/sequence identity and one in-flight step per source; a failure aborts the entire window without retry. Complete iteration checkpoints restore learner state and counters but reset all environment services into a new recovery generation, since physics snapshots are not part of this protocol. UniLab injects factories and task configuration, `uni_rl` owns services/collection/learning, and UniSim owns backend physics; MuJoCo remains a final holdout.
