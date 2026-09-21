# Central synchronous PPO runner

`uni_rl.algos.synchronous_runner.SynchronousPPORunner` uses the native RSL-RL
constructor, adaptive PPO optimizer, and logger. Its injected `CentralVecEnv`
contains four ordered source slices. Every iteration collects 24 complete
control steps, prepares GAE within each source, normalizes advantages globally,
and performs five epochs with four minibatches each. Runtime audits require all
samples in each epoch and exactly 20 Adam steps per iteration.

Pass a nonempty `manifest_digest` describing the task configuration and assets.
The runner also records the complete training configuration and ordered source
topology. `learn(n)` runs `n` additional iterations and closes environment
services and the logging writer on success or failure. `latest_window` retains
the latest raw source trajectories and recovery generation in memory. Optional
`init_at_random_ep_len=True` preserves native initial episode staggering. JSONL metrics contain iteration,
policy/normalizer versions, sample counts, losses, learning rate, timing, source
rewards, terminal counts, and rolling source episode statistics; native
TensorBoard curves provide the global training metrics and source curves.

With a log directory, checkpoints are written atomically at iteration indices
0, 500, 1000, ... and at the final iteration (9999 for a 10,000-iteration run).
They retain native actor/critic/optimizer keys for inference consumers, plus
completed-iteration counters, adaptive learning rate, empirical normalization
statistics with exact integer counts, Python/NumPy/Torch/CUDA RNG states,
rolling logger metrics, and the configuration/assets/topology contract.
Partial windows and partial optimizer updates cannot be checkpointed.

Resume with a fresh runner and `load(path)`, then `learn(remaining_iterations)`.
Resume requires an exact contract match and coherent counters, including all
Adam parameter steps and empirical normalizer sample counts. It starts at
the saved iteration plus one, resets every environment, clears incomplete
episode accounting, and increments the recovery generation. Physics state is
not restored, so resumed trajectories are not identical to uninterrupted
training. No rollout archive is written by default.

This implements [ADR-0002](adr/0002-central-synchronous-ppo.md). Fake-source
lifecycle tests exercise the real native PPO on both RSL-RL 5.0.1 and 5.5;
they make no robot-backend training support claim.
