# Synchronous multi-source environments

`uni_rl.ipc.multi_source_env` composes caller-injected environments into one
numpy `EnvProtocol`. It does not select simulators, tasks, devices, seeds or
environment quotas. Algorithm runners, wrappers, rollout storage, GAE and PPO
updates do not change.

## Public interface

```python
from uni_rl.ipc.multi_source_env import (
    EnvSourceSpec,
    MultiSourceEnvError,
    MultiSourceOptions,
    make_multi_source_env,
)
```

- `EnvSourceSpec(name: str, factory: EnvFactory, num_envs: int,
  env_cfg_override: Mapping | None = None, seed: int | None = None)` is frozen.
  Factories retain the existing `(num_envs, env_cfg_override)` signature.
- `MultiSourceOptions(startup_timeout_s=600, operation_timeout_s=120,
  shutdown_timeout_s=10)` is frozen. Timeouts must be finite and positive.
- `make_multi_source_env(sources, *, options=MultiSourceOptions()) -> EnvProtocol`
  accepts a nonempty sequence, starts every source, and validates compatibility
  before returning. The returned object also implements
  `SupportsEpisodeLengthBufferProtocol` and `SupportsSourceStatisticsProtocol`.
- `MultiSourceEnvError(source_name, operation, sequence_id, remote_traceback)`
  exposes all four arguments as attributes. A worker exception includes its
  formatted traceback; timeouts, schema errors, transport failures and abrupt
  exits include their source and operation identity instead.

Call the constructor from a spawn-safe main entrypoint, protected by
`if __name__ == "__main__":`. Factories must be top-level functions or picklable
callable objects, not closures or lambdas. Config overrides are opaque to this
runtime and must also be picklable. Source names are unique identifiers matching
`[A-Za-z0-9][A-Za-z0-9_.-]*`; counts are positive integers. Optional seeds are
integers in `[0, 2**32)`. A seed initializes Python and numpy RNGs, and torch when
installed, before the factory runs. It does not rewrite the override mapping;
factories own any independently seeded generators and device binding.

For example, an owner can construct source specifications without teaching the
runtime anything about its environment implementations:

```python
from uni_rl.env_contract import EnvFactory, EnvProtocol
from uni_rl.ipc.multi_source_env import EnvSourceSpec, make_multi_source_env


def compose_training_env(
    first_factory: EnvFactory,
    second_factory: EnvFactory,
    first_count: int,
    second_count: int,
) -> EnvProtocol:
    return make_multi_source_env(
        [
            EnvSourceSpec("first", first_factory, first_count),
            EnvSourceSpec("second", second_factory, second_count),
        ]
    )
```

Always close the returned environment in `finally`. It also supports a context
manager. A daemon process cannot host this composition: workers themselves are
non-daemon spawn processes so their factories can create nested workers.
This implementation requires POSIX CPython; Linux is the validated platform.

## Initialization and compatibility

Each source calls `init_state()` once during startup to discover and validate
actual array schemas. The facade's `state` remains `None` until its public
`init_state()`, `step()` or `reset()` is called. Public `init_state()` is
idempotent: it exposes the cached initial state without resetting sources again.

All sources must agree exactly on:

- Observation-group key order, dimensions and actual numpy dtypes, including
  the required actor group `obs` and any optional `critic` group.
- Flat observation/action space shapes and dtypes. Supplied action bounds and
  joint names must agree, including whether metadata is present. Bounds from
  `algo_capabilities` must agree with bounds on `action_space` when both exist.
  Unbounded dimensions may use negative/positive infinity in these cold-path
  bounds; NaN and unordered bounds remain invalid. This does not permit
  non-finite actions or other transition arrays.
- Reward and per-environment step-counter dtypes.
- Finite, positive `cfg.ctrl_dt` and `cfg.max_episode_seconds`.

The facade exposes a common `cfg` view containing those two timing fields,
space views containing `shape`/`dtype` (and action `low`/`high`),
`obs_groups_spec`, `algo_capabilities`, and `play_capabilities`. It does not
pickle a source's full config or gym space, synthesize backend-specific fields,
or promise gym methods outside `EnvProtocol`. Whole-composition physics-state
playback is not exposed, so the facade reports
`supports_physics_state_playback=False`. Each worker retains its real source
capability when reconstructing a NaN guard.

## Data and synchronization

Source order defines permanent contiguous global environment slices. Every
global step submits actions to **all** sources before waiting. Completion order
does not change sample order or source proportions. A global result is published
only after every source acknowledges the exact operation and sequence number.

Actions, grouped observations, rewards, terminated/truncated flags, final
observations, final-observation masks, step counters, reset indices and episode
length inputs travel in owner-allocated shared-memory arrays. Each source has
two preallocated slots. A process-shared lock atomically publishes the completed
slot and sequence; the corresponding pipe ACK is checked against that
publication. A worker never publishes an incomplete transition. Public results
are independent numpy snapshots, not SHM views that later operations overwrite.

Pipes contain only commands, schema descriptors, bounded scalar metadata,
statistics and error diagnostics. Messages are limited to 64 KiB. The runtime
checks available Linux `/dev/shm` space for the full transport budget before any
allocation, and again before allocating each segment. Linux transport allocation
reserves segment storage with `posix_fallocate` before touching mapped arrays,
so concurrent SHM use can produce a handled allocation error rather than a
learner-side SIGBUS on first write.

The supported hot-path schema is intentionally explicit:

| Field | Required schema |
| --- | --- |
| `actions` | Finite floating ndarray `(total_num_envs, action_dim)`; converted to the declared action dtype with an overflow check; never clipped by the facade |
| Each observation group | Finite native float16/float32/float64 ndarray `(source_num_envs, group_dim)` with its startup dtype |
| `reward` | Finite floating ndarray `(source_num_envs,)` with its startup dtype |
| `terminated`, `truncated` | Boolean ndarrays `(source_num_envs,)` |
| `info['steps']` | Non-negative integer ndarray `(source_num_envs,)` with its startup dtype |
| Final observations | Full source-sized dict with exactly the observation schema; all entries finite |
| `info['_final_observation']` | Boolean ndarray `(source_num_envs,)`; when omitted, inferred from done flags only if final observations exist |
| `info['log']` | String-keyed mapping of finite numeric scalar metrics |
| Other `info` entries | Finite scalar values, strings, booleans, `None`, or string-keyed mappings of these, up to eight nesting levels |

Timeouts must have valid final observations and a true mask. Masks cannot mark
non-done rows in a step. `state.final_observation` and the compatible
`info['final_observation']` must agree when both exist. The facade preserves
global step counters and masks and supplies both final-observation forms when
the mask is nonempty. Terminal-only transitions may omit final observations.

Unknown bulk arrays, object arrays, ragged observations, changing group order
or dtype, vector-valued metrics, and non-finite data fail clearly rather than
being silently dropped, averaged or serialized through pipes.

## Reset, logging and guards

`reset(global_ids)` rejects duplicate, negative, out-of-range, non-integer or
non-vector IDs before dispatch. IDs are mapped to each source's local indices;
returned observations and `info['steps']` follow the original request order.
Unselected sources are not reset. Empty IDs return empty observations without a
worker reset. Source reset observations must match the selected rows of its
current state; reset-only final-observation arrays are unsupported and should
instead be exposed in the source state.

Source metrics retain their meaning:

- `state.info['log']['source/<name>/<metric>']` contains that source's scalar
  metric unchanged. There is no cross-source averaging.
- Other metadata appears as `state.info['source/<name>/<key>']`.
- `steps`, `_final_observation`, and `final_observation` retain their global
  per-environment semantics rather than becoming source-level scalar metrics.

`set_episode_length_buf(values)` scatters a global integer vector. All sources
must implement the public optional setter and update `state.info['steps']`.
Support is checked across every source before sending any setter command. This
works with the existing PPO wrapper's randomized initial episode lengths.

`set_nan_guard(guard)` accepts `NanGuard` or `None`. `NanGuard.cfg` returns a
configuration copy. Only this configuration is sent: captured physics arrays
and global-sized buffers are never pickled. Each worker creates a guard with
its local environment count and playback capability, using
`<output_dir>/source/<name>` for dumps. An unset output directory uses
`<tempdir>/uni_rl/nan_dumps/source/<name>`.

## Source statistics

`source_statistics` is a read-only mapping of source name to read-only scalar
mapping. Each access returns the latest complete-barrier snapshot; an earlier
snapshot does not change. No observation, action or GPU array is included.

| Key | Meaning |
| --- | --- |
| `num_envs` | Fixed source environment count |
| `step_calls` | Completed explicit source `step` calls since construction |
| `transitions` | `step_calls * num_envs` |
| `reset_calls` | Completed explicit, nonempty source `reset` calls |
| `step_seconds`, `reset_seconds` | Cumulative wall time inside those source methods |
| `last_step_seconds`, `last_reset_seconds` | Duration of the most recent corresponding source call, initially zero |
| `pid` | Source worker PID |
| `rss_bytes` | Current Linux worker RSS sampled at acknowledgement, excluding descendants; zero when `/proc` is unavailable |

These counts exclude initialization and internal autoresets. Timings exclude
transport, publication, validation and the global barrier; they are not
end-to-end training throughput. RSS is current resident memory, not peak RSS
and not GPU memory. Acceptance drivers can compare per-source transition deltas
between completed updates without changing storage or minibatch sampling.

## Failure and shutdown

There is no retry, automatic source removal or partial success for step/reset.
Worker exceptions, crashes, timeouts, invalid data, unsupported source schemas,
stale/duplicate ACKs and stale SHM publications poison the facade. Subsequent
operations and property reads raise the stored `MultiSourceEnvError`; only
idempotent `close()` remains usable. Caller input errors caught before dispatch
raise `ValueError` without advancing or poisoning an otherwise healthy env.

On POSIX each source enters its own session/process group before constructing
the environment. Shutdown terminates whole source groups, including nested
children even if the worker already exited, escalates to SIGKILL, joins workers,
closes pipes, and unlinks owner SHM. Normal close first allows bounded cooperative
`env.close()`; failure skips cooperative shutdown. Forced reaping has a short,
shared grace period after the configured shutdown budget, including up to one
second for resource-tracker completion. Factories must not
detach descendants into unrelated sessions/groups or launch externally managed
services and expect this runtime to own them. Linux group cleanup is tested;
equivalent descendant cleanup on other operating systems is not claimed.

Each source also has a coordinator-owned CPython resource tracker outside its
simulation process group. Before executing the factory, the worker replaces
its inherited tracker FD with this source-local one; subsequently spawned
children inherit that tracker. This is a small, isolated CPython-private adapter,
not a simulator cleanup protocol. The coordinator closes its copy of the writer
FD after spawn, kills the source group on failure, then waits for tracker EOF
cleanup. Thus Python-tracked source and child SHM/semaphores are reclaimed even
after a native crash skips every Python `finally`. The coordinator-owned
transport attachment is explicitly excluded from the source tracker. There is
no `/dev/shm` name scan and no unlinking of unrelated learner allocations.

Untracked native allocations, deliberately detached children, borrowed external
SHM registered as if source-owned, and simultaneous loss of the coordinator and
cleanup tracker are outside this guarantee. Owners must retain responsibility
for such resources. A tracker that cannot finish cleanup is reported as a
cleanup failure, not silently treated as success.

## Validation boundary

`tests/ipc/test_multi_source_env.py` uses real CPU spawn workers and deterministic
factories to test ordering, concurrent submission, reset scattering, finite data,
schema errors, wire faults, timeouts, crashes, nested-process cleanup, SHM
cleanup (including source/child SHM after SIGSEGV, with core dumps disabled),
guard reconstruction, seeds and source statistics. It can run without
the training dependencies:

```bash
PYTHONPATH="$PWD/src:$PWD/tests/ipc" uv run --no-project --python 3.11 --with numpy --with pytest \
  pytest --confcutdir=tests/ipc tests/ipc/test_multi_source_env.py -q
```

`tests/ipc/test_multi_source_ppo.py` uses the actual stock CPU `OnPolicyRunner`
and PPO (also the existing final-observation-aware variant) for a 24-step
rollout, full update, finite-loss checks, checkpoint save/load, and inspection
of the unchanged stock storage minibatch iterator. Every complete epoch must
consume every source's labels in the configured proportions; individual
minibatches need not be balanced. It requires the normal training dependencies.
These small CPU fixtures are correctness tests, not simulator-scale, GPU,
long-duration, throughput or experiment acceptance evidence.

## Bounded PPO validation

`uni_rl.algos.rsl_rl_validation.run_bounded_ppo` accepts an existing runner and
exactly one of `max_updates` or `duration_seconds`. Wall-clock budgets exclude
`warmup_updates`. A temporary logger adapter observes the stock runner only
after rollout, GAE and a full policy update. It never interrupts `step` or
copies the training loop. On success it saves `validation_final.pt`,
`validation_report.json` and per-update `updates.jsonl` in a fresh directory.
Failures propagate without an acceptance checkpoint; the caller owns env close.

For a composite env, pass `source_statistics=lambda: env.source_statistics`.
The helper verifies exact transition deltas for each source and rejects PPO
minibatch counts that would discard rollout samples. Tests use four sources,
both supported PPO variants, and ordinary as well as bounded runner execution.

Numeric `state.info["timing"]` values also appear in source statistics under
`timing/<name>/last` and `timing/<name>/total`, preserving the original units.
This records env-owned internal reset timings, not just explicit facade reset
RPCs, without introducing any task or simulator names into the transport.
