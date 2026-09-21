# Multi-source PPO Windows

## Language

**Window**: A complete, fixed transition budget collected under one policy weight version before a PPO update. Its normalization protocol identifies either one frozen version or the consecutive versions used at each control step.

**Source**: An independently identified producer of environment transitions.

**Segment**: Consecutive vector steps from one source, retaining separate environment and episode identities.

**Quota**: The number of transitions requested from a source in a window, independent of its resident environment capacity.

**Control step**: One centrally inferred action batch followed by one acknowledged environment step from every source.

**Environment service**: An independently identified environment pool that executes actions without owning a policy or optimizer.

**Recovery generation**: A new set of episodes beginning when all environment services reset after loading a complete training boundary.
