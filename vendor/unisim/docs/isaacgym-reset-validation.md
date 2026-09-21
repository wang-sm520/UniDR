# Isaac Gym Reset Validation

Local validation on 2026-09-14, branch `codex/unidr-motion-body-ids`, based on
UniSim v1.3.0 (`4270aa81d868744980db90dac6dd959d3f542f50`, Apache-2.0).
The repository AGENTS.md and uv.lock were inspected; dependencies were not changed.
Hardware: RTX 3090, driver 580.178.04. Existing vendor worker: Python 3.8.20,
NumPy 1.21.6, Isaac Gym Preview 4. Standalone helper import passed on that interpreter.

Reset now publishes cached MJCF FK immediately and coalesces native tensor writes
until the next STEP. Each root/DoF setter runs once before simulate. Native COM
velocities are converted to/from the public link-origin velocity convention.
Only selected reset rows change; their stale contact forces are cleared until the
next physics solve. No physics step or XML parse occurs during reset.

From the UniSim checkout:

```sh
make check
UNISIM_TEST_ISAACGYM=1 uv run --no-sync pytest -q tests/test_isaacgym_reset.py
git diff --check
make package
```

The ordinary gate passed: Ruff clean, 288 passed and 16 skipped. The opt-in suite
passed all 14 tests in 4.58s. Packaging passed and the source archive contains the
standalone FK helper. The SDK is not a base dependency. Its test checks repeated
resets, native link velocities with nonzero COM offsets, and subsequent stepping.
CPU tests cover anchored joints, refs/angle units, fixed/hinge/slide joints,
velocity derivatives, cold includes, row isolation, staging/REFRESH order, and Python 3.8 syntax.

Additional local commands, run from `/home/wsm/wang-sm/UniLab-unidr-backends`:
`uv run --no-sync python /tmp/unidr_gym_reset_native.py` and
`uv run --no-sync python /tmp/unidr_gym_g1_reset.py` both passed. The latter used
two environments and the registered G1 flip NPZ frames 0, 80, and 160. Maximum
reset errors were position 1.19e-7, linear velocity 4.77e-7, angular velocity 9.54e-7.
After a native STEP, FK discrepancies were at most 8.98e-7, 5.69e-6, and 3.11e-6.
G1, Go1, and Go2 flat-scene XMLs all passed cold compilation; other MJCF structures
outside the documented FK subset fail with an initialization diagnostic.
These are reset correctness checks, not evidence of flip learning or multi-GPU training.
