# Isaac selected-reset and contact contract (M1)

This implements the IsaacGym/IsaacSim portion of the approved cross-repository
multisim plan. It changes no `SimBackend` signatures and adds no UniLab or
`uni_rl` dependency. Runtime imports remain in worker initialization. The
package version is unchanged; **wire protocol version 2 is intentionally
incompatible with old workers**.

Implementation support is not physics-effect acceptance. The initial
September 12, 2026 NVML mismatch was resolved before the subsequent CUDA
validation: kernel and library are now `580.178.04`. CPU fake/readback tests
cannot certify native force response, gains, or contact physics. The separate four-backend
G1 gate is owned by UniLab's `tests/backends/test_g1_multisim_physics.py`;
explicitly requested gates must fail rather than accept skipped tests.

## Public behavior

- `get_dr_capabilities()` lazily materializes and validates the worker before
  advertising exactly `RESET_TERM_BODY_MASS`, `RESET_TERM_KP`, and
  `RESET_TERM_KD`. Other reset and interval terms remain unsupported.
- `get_body_mass()` returns a detached, one-dimensional **nominal** native
  mass table in public body order. It never returns the last randomized row.
  `get_actuator_gains()` returns nominal joint-order arrays (MJCF metadata
  before materialization, native metadata afterward).
- `set_state(env_indices, qpos, qvel, randomization)` consumes dense tables of
  shape `(selected_count, num_bodies)` for mass and
  `(selected_count, num_dof)` for gains. Rows follow `env_indices`, not sorted
  environment order. Body/joint columns follow the public name/ID contract.
  These are **absolute values, not multipliers**. Applying the same payload
  twice does not compound its effect. Omitted terms retain their current
  values, including on selected rows.
- Mass must be positive; gains may be zero but not negative. All values must
  be real and finite after float32 conversion. State shapes, nonzero root
  quaternions, integer unique in-range environment IDs, and every requested
  term are checked before writing shared memory or calling native setters.
  An empty selection is a validated no-op.
- A native setter failure may occur after an earlier selected write succeeded.
  There is no rollback promise. The host invalidates that worker and rejects
  subsequent state/step access rather than continuing partially mutated
  physics. Construct a new backend after such a failure.

## Version 2 wire layout

The canonical, path-loadable module is
`src/unisim/backend/subprocess_ipc/protocol.py`. It imports only the standard
library and NumPy and retains pickle protocol 4 and the existing framing.
The legacy `isaacgym.protocol` re-export remains valid.

INIT adds:

```json
{"protocol_version": 2, "required_reset_terms": ["body_mass", "kp", "kd"]}
```

META retains all existing body/joint/render/origin fields and requires:

| Field | Meaning |
| --- | --- |
| `protocol_version` | Integer `2` |
| `supported_reset_terms` | All three of `body_mass`, `kp`, `kd` |
| `nominal_body_mass` | Positive float32-compatible table, length `num_bodies` |
| `nominal_kp`, `nominal_kd` | Nonnegative nominal tables, length `num_dof` |
| `contact_reporter` | `isaacgym_net_contact_force`, `isaaclab_contact_sensor`, or `null` |

A reporter must match the requested adapter. `null` is valid for a compatible
worker without contacts but keeps contact sensor declarations unsupported.
Missing version, reset support, nominal data, or reporter declarations fail
materialization. Nominal tables are snapshots of native properties, not
host estimates. Importer remappings are resolved once by names.

ATTACH adds `protocol_version: 2` and all of these new float32 slots:

| Slot | Shape |
| --- | --- |
| `reset_body_mass` | `(num_envs, num_bodies)` |
| `reset_kp` | `(num_envs, num_dof)` |
| `reset_kd` | `(num_envs, num_dof)` |

The worker validates the complete slot name/shape/dtype layout before opening
any segment. The first `count` rows are a packed transaction, just like
`reset_env_ids`, `reset_qpos`, and `reset_qvel`. SET_STATE sends only:

```json
{"count": 2, "randomization_terms": ["body_mass", "kd", "kp"]}
```

State-only resets must send `randomization_terms: []`. Unlisted slots are not
read, so old slot contents cannot accidentally reapply a previous reset.
Both sides validate the transaction; numeric payloads remain in shared memory.

## Native implementation

IsaacGym caches per-actor rigid-body and DOF properties after actor creation.
Mass writes call `set_actor_rigid_body_properties` with `recomputeInertia=False`;
gain writes call `set_actor_dof_properties`. Only selected actors are touched.
Native DOF/body order is remapped to public order for metadata, control, reset,
state, and contact publication. Native setters returning false are errors.

Root/DOF resets update the logical wrapped state immediately, but consecutive
reset row selections are unioned and submitted through each indexed tensor setter
only once, before the next STEP. The index tensor remains owned until the
simulation consumes it. Native refreshes occur after simulation, not after a
pending reset setter; this follows IsaacGym's GPU tensor API ordering rules.

The Preview 4 GPU tensor pipeline also discards root resets after runtime
rigid-body mass writes, even when those writes preserve nominal mass. Real
G1 diagnostics distinguish this from contact reporting: the native robot
starts below the plane and is launched by a large contact impulse. The
adapter therefore uses CPU state tensors (`use_gpu_pipeline=False`) while
keeping `physx.use_gpu=True` for nonnegative device IDs. This is an SDK-supported
transport mode, not CPU physics or an extra simulation step. Original G1
mass/gain effects and grounded/airborne per-foot contacts pass with this
configuration. All IsaacGym users now use this transport; full-scale capacity
and its throughput cost require measurement before training readiness.

IsaacSim uses the IsaacLab 2.3.2 / IsaacSim 5.1 public articulation APIs:
`root_physx_view.set_masses(full_cpu_table, indices=selected_cpu_ids)`,
`write_joint_stiffness_to_sim`, and `write_joint_damping_to_sim`. A cached
full mass table satisfies the PhysX indexed setter layout; only the selected
indices are committed. Mass writes never call an inertia setter. Implicit
actuator gain buffers are updated alongside native gains so their effort
estimates do not retain obsolete gains.

Shutdown clears IsaacLab callbacks and the SimulationContext singleton before
closing Kit's stage. Otherwise the standalone STOP callback can re-enter its
render loop while stage closure waits, even after native assertions succeed.
This cleanup uses the public SDK lifecycle methods and does not skip cleanup.

Contacts come from a real `ContactSensor` constructed before simulation reset
with asset `activate_contact_sensors=True`. Discovery of the imported rigid
link parent and name permutations is cold-path work. The current contact
profile requires all contract links under one imported parent; missing,
duplicated, or differently nested links fail closed rather than receiving
zero placeholders. Reporter initialization, complete name coverage, array
shape, and finite values are checked before publishing data.

`contact_force` holds per-link world-frame net normal contact forces. Its
columns are mapped by sensor body names independently of articulation order.
The existing scene `contact data="found"` surface tests the vector magnitude;
it does not promise pair filtering, tangential forces, or contact counts.
The host does not expose a new vector-force getter under M1.

Contact rows are zero immediately after initialization or selected reset,
because there is no new physics sample for those states. Repeated refreshes
cannot resurrect the previous state's contacts. Unselected rows retain their
last sample; the next successful physics step makes contact data valid again.
IsaacGym still has no kinematics-only forward: body poses after reset retain
the pre-existing next-STEP refresh limitation. No asset parsing occurs during
step/reset/randomization.

## Validation and downstream fake workers

CPU tests run actual host pipe/SHM barriers and actual worker reset/state
methods against NumPy SDK doubles, including reordered names and unordered
selected rows:

```bash
PYTHONPATH=src uv run --no-project --with numpy --with pytest pytest -q \
  tests/test_isaac_reset_protocol.py tests/test_isaac_native_readback.py
```

Native **property readback only**, one backend/process, on a working GPU.
The probe resolves the same configured/cache runtime and library environment
as the adapter; optional interpreter overrides retain their usual meaning:

```bash
UNISIM_ISAAC_READBACK=1 \
PYTHONPATH=src uv run --no-project --with numpy --with pytest pytest -q \
  'tests/test_isaac_native_readback.py::test_native_selected_property_readback[isaacgym]'

UNISIM_ISAAC_READBACK=1 \
PYTHONPATH=src uv run --no-project --with numpy --with pytest pytest -q \
  'tests/test_isaac_native_readback.py::test_native_selected_property_readback[isaacsim]'
```

Without opt-in, these two tests skip. With opt-in, failed GPU preflight,
missing SDKs, and native failures are test failures, not passing skips. A
successful readback explicitly reports `physical_effects_validated: false`:
it proves native table writes, selected-row isolation, repeated absolute
payloads, preserved inertia, and reset clearing, **not force-response physics**.
The parent G1 gate remains necessary before training acceptance.

Downstream compatibility work is confined to fake tests, not new env-private
calls: update UniLab's `tests/base/isaacgym_mock_worker.py` META/INIT/ATTACH and
SET_STATE handling; extend `test_protocol_slot_layout` and
`test_dr_and_pre_step_control_fail_closed` in
`tests/base/test_isaacgym_backend.py`; replace unconditional contact rejection
in `tests/base/test_isaacsim_backend.py` with reporter-present/absent cases.
Nominal arrays must match the fixture's names, dimensions, and actuator gains.

## Initial local validation record: September 12, 2026

All commands ran from the unisim repository, without using or modifying the
parent UniLab virtual environment. No commits, branches, version bumps, or
releases were created. The base remains
`1ef3bb6b04a2cd76f20be72c49cd4d15f528e51b`.

Focused CPU suite: **85 passed, 2 skipped** (the two native readback cases):

```bash
PYTHONPATH=src uv run --no-project --python 3.11 --with numpy --with pytest \
  pytest -q tests/test_isaac_reset_protocol.py tests/test_isaac_native_readback.py
```

Complete `make check` gate in an isolated, supported Python 3.11.16 environment:
**Ruff passed; 314 tests passed, 21 skipped** (unavailable optional runtimes).
The explicit environment selection prevents nested Makefile commands from
syncing any project environment:

```bash
PYTHONPATH=src uv run --no-project --python 3.11 \
  --with numpy --with pytest --with ruff \
  --with 'mujoco==3.11.0' --with 'mujoco-uni-runtime==0.5.0' \
  python -c 'import os, subprocess, sys; print(sys.version, flush=True); env = dict(os.environ, UV_PROJECT_ENVIRONMENT=sys.prefix, UV_NO_SYNC="1"); sys.exit(subprocess.call(["make", "check"], env=env))'
```

Actual Python 3.8.20 protocol round trip and SDK-free worker import: **passed**:

```bash
uv run --no-project --python /home/wsm/miniconda3/envs/homierl/bin/python python - <<'PY'
import io
import runpy
import sys

wire = runpy.run_path("src/unisim/backend/subprocess_ipc/protocol.py")
runpy.run_path("src/unisim/backend/isaacgym/worker.py")
assert sys.version_info[:2] == (3, 8)
stream = io.BytesIO()
wire["send_message"](stream, "INIT", {"protocol_version": 2})
stream.seek(0)
assert wire["recv_message"](stream)["payload"]["protocol_version"] == 2
assert not any(name in sys.modules for name in ("torch", "isaacgym", "isaacsim", "unilab", "uni_rl"))
print(sys.version)
print("Python 3.8 lazy imports and protocol v2 round trip: PASS")
PY
```

Explicitly opted-in native preflight: **2 failed/blocked**, not skipped or
accepted. Both failed at `nvidia-smi` (exit 18, NVML library `580.178` mismatch),
before either vendor interpreter was launched:

```bash
UNISIM_ISAAC_READBACK=1 PYTHONPATH=src uv run --no-project --with numpy --with pytest \
  pytest -q -m slow tests/test_isaac_native_readback.py
```

Additional checks all passed: `uv lock --check`, `make package`,
`git diff --check`, and:

```bash
uv run --no-project --python 3.11 --with ruff ruff format --check \
  tests/isaac_worker_fakes.py tests/isaac_protocol_worker.py \
  tests/test_isaac_reset_protocol.py tests/isaac_native_readback_worker.py \
  tests/test_isaac_native_readback.py
```

Packaging produced the unchanged-version `1.2.0` sdist and wheel under ignored
`dist/`; only the pre-existing license-classifier deprecation warning remains.
The parent project's compatibility tests and G1 effect gate are not part of
these results and were not claimed as passed.
