---
status: accepted
---

# Preserve trajectories before preparing PPO batches

The user authorized uniDR C1's in-memory window contract and PPO bridge on 2026-09-14. `uni_rl.algos.multi_source_ppo` retains source segments and uses the installed PPO's GAE before flattening into its existing feedforward storage, keeping learning in the RL runtime and preserving UniLab's injected env contract. Raw rewards and terminal data remain separate from the timeout-corrected optimization view, preventing cross-source GAE and source-local advantage normalization from changing the intended sample mix.

Whole-pool segment lengths vary while capacity and total transitions stay fixed; this also varies the GAE horizon and must be recorded. C1 does not publish snapshots, merge normalizer statistics, introduce IPC or a persistent runner, or claim robot backend support. Those contracts require separate child approval. See [C1 delivery and validation](../unidr-c1.md) and UniLab ADR-0005 (obs/critic contract).
