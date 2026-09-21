# Isaac Sim shared ground repair

The MJCF converter includes the world floor in the USD asset. Referencing that
entire asset below every robot replicated an infinite plane and its imported
kinematic articulation. At 4096 G1 environments PhysX exhausted GPU pair buffers
and continued stepping with missing interactions. Collision groups alone did
not prevent that broadphase capacity demand.

`backend/isaacsim/worker.py` now extracts static horizontal collision planes on
the cold path. It retains the original USD reference, transforms and material
bindings, removes the imported ground-only rigid/articulation wrappers, and
shares ground through the cloner's global collision group. Robots retain their
environment groups and existing origins. The decorative playback ground has
collision disabled. Tilted, dynamic and animated planes fail closed rather than
being silently shared. Complex terrain and moving platforms are outside this
flat-ground repair.

The subprocess owner persists stderr under
`${XDG_CACHE_HOME:-~/.cache}/unisim/worker-logs/` and prints its unique path.
An independent incremental reader checks before requests and after responses.
Explicit lost-interaction, allocation, contact/patch-buffer and collision-stack
errors reject the response and close the worker/shared memory. Ordinary warnings
remain visible without automatically becoming fatal errors. Logs survive cleanup;
they are diagnostic artifacts and are not rotated by this change.

## Validation

The original real-engine regression used the successful G1 flip configuration,
seed 2, 32 control steps and identical zero actions (default-pose PD targets).
No policy or optimizer was involved. The faulty 4096 run terminated 3763/4096
environments, while 1024 terminated none. Increasing two buffers still failed;
the repair therefore does not change PhysX capacities.

With the repair, real 1024 and 4096 runs both completed 32 steps without early
termination or native errors. The largest difference between per-step median
body coordinates was 0.058 mm. The old successful 1024 reference and repaired
1024 reset states were identical; their short trajectories differed by at most
1.84 mm at an individual body coordinate. This is numerical agreement, not
bitwise trajectory equivalence or proof that a trained policy completes a flip.

Additional real tests passed: sinusoidal actions with a half-pool reset at step
16 (the other rows remained unchanged), reference phases 0/100/124/180/224, and
render/no-render equality for two environments. A fresh 4096-environment PPO
capacity run completed 100 updates with 9,830,400 transitions and 2,000 Adam
steps. Actor and critic parameters changed, both normalizer counts matched the
budget, and the saved native worker log contained no errors. Its final mean
return was 13.49 and mean episode length 114.02 steps. These observations verify
recovery from the immediate physical collapse, not final policy performance.

`tests/test_isaacsim_ground.py` runs against actual USD composition, including
single/multiple environment collision counts, transforms, direct and inherited
physics materials, robot articulation preservation, and rejection of unsupported
planes. It skips when USD is absent and must additionally run in vendor Python.
`tests/test_worker_physics_errors.py` uses real subprocesses and shared memory to
test error propagation, split writes, warning handling and cleanup.

Base gate: `UV_NO_SYNC=1 make check`. Vendor check: `uv run --no-project --python
<isaacsim-python> -m pytest tests/test_isaacsim_ground.py -q`, with the installed
vendor USD library directories on `PYTHONPATH` and `LD_LIBRARY_PATH`. No SDK
installation or dependency-lock changes are required.

The application experiment artifacts under
`UniLab-unidr-backends/logs/isaacsim-4096-diagnosis-20260921/` retain original and
repaired states, native logs, exact commands and further capacity validation.
Long training must start fresh only after real capacity checks; a contaminated
checkpoint is not a valid repair-validation starting point. This repair does not
change task rewards, action scaling, termination or PPO parameters.
