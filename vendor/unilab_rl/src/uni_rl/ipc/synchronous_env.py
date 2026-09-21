"""Synchronous physics services for a centrally evaluated policy.

One spawn worker owns each injected environment, including its nested backend
workers. A bounded pipe permits one request per source; a failed request poisons
the entire service because physics calls cannot safely be retried.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing.connection import Connection
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from uni_rl.env_contract import EnvFactory, EnvProtocol, EnvStateProtocol
from uni_rl.utils.seed import apply_training_seed


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    factory: EnvFactory
    num_envs: int
    env_cfg_override: Mapping[str, Any] | None = None
    process_env: Mapping[str, str] | None = None


@dataclass
class _State:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any]
    final_observation: dict[str, np.ndarray] | None
    timing: dict[str, float] | None = None


@runtime_checkable
class _EpisodeLengthSetter(Protocol):
    def set_episode_length_buf(self, values: np.ndarray) -> None: ...


@runtime_checkable
class _ProbeEnv(Protocol):
    def reset_probe(self, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...

    def finish_probe(self) -> object: ...  # The coordinator validates the None acknowledgement.


def _metadata(env: EnvProtocol) -> dict[str, Any]:
    return dict(
        num_envs=env.num_envs,
        obs_groups_spec=dict(env.obs_groups_spec),
        observation_shape=tuple(env.observation_space.shape),
        action_shape=tuple(env.action_space.shape),
        max_episode_seconds=float(env.cfg.max_episode_seconds),
        ctrl_dt=float(env.cfg.ctrl_dt),
    )


def _state_payload(state: EnvStateProtocol) -> _State:
    return _State(
        state.obs,
        state.reward,
        state.terminated,
        state.truncated,
        state.info,
        state.final_observation,
    )


def _worker(conn: Connection, source: SourceSpec) -> None:
    env: EnvProtocol | None = None
    seq = 0
    try:
        if sys.platform == "linux":
            os.setsid()
        os.environ.update(source.process_env or {})
        override = dict(source.env_cfg_override or {})
        apply_training_seed(int(override.get("seed", 0)), cuda=False)
        env = source.factory(source.num_envs, override or None)
        conn.send((source.source_id, seq, "ok", _metadata(env)))
        while True:
            incoming, operation, payload = conn.recv()
            if incoming != seq + 1:
                raise RuntimeError(f"expected request {seq + 1}, got {incoming}")
            seq = incoming
            if operation == "close":
                break
            if operation == "init_state":
                result: Any = _state_payload(env.init_state())
            elif operation == "step":
                started = time.perf_counter()
                state = env.step(payload)
                elapsed = time.perf_counter() - started
                result = _state_payload(state)
                result.timing = {"env_step_seconds": elapsed}
            elif operation == "reset":
                result = (
                    env.reset(payload)
                    if len(payload)
                    else (
                        {k: np.empty((0, d), np.float32) for k, d in env.obs_groups_spec.items()},
                        {},
                    )
                )
            elif operation == "set_episode_length_buf":
                if not isinstance(env, _EpisodeLengthSetter):
                    raise TypeError("source does not support set_episode_length_buf")
                env.set_episode_length_buf(payload)
                result = None
            elif operation in {"reset_probe", "finish_probe"}:
                if not isinstance(env, _ProbeEnv):
                    raise TypeError("source does not support the independent probe protocol")
                result = (
                    env.reset_probe(payload) if operation == "reset_probe" else env.finish_probe()
                )
            else:
                raise RuntimeError(f"unknown operation {operation!r}")
            conn.send((source.source_id, seq, "ok", result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            conn.send((source.source_id, seq, "error", traceback.format_exc()))
        except (EOFError, OSError):
            pass
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            conn.close()


def _receive(conn: Connection, request: tuple | None) -> Any:
    if request is not None:
        if conn.poll():
            raise RuntimeError("unsolicited or duplicate source response")
        conn.send(request)
    result = conn.recv()
    if result[2] == "ok" and conn.poll():
        raise RuntimeError("duplicate source response or worker exit")
    return result


def _timed_receive(conn: Connection, request: tuple | None) -> tuple[Any, float, float]:
    # Parent-local timestamps; include serialization/IPC but not thread-pool queueing.
    started = time.perf_counter()
    response = _receive(conn, request)
    return response, started, time.perf_counter()


def _group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


class SynchronousEnv:
    """Aggregate injected sources in declared order, with no child inference.

    Arrays are float32 (flags bool), with full source batches for terminal
    observations. ``info['source_logs']`` preserves each source's complete info.
    This service is single-caller; overlapping calls are rejected.

    Successful steps expose ``source_timings`` in seconds: worker ``env.step``
    call wall time, parent request/response time (without thread-pool queueing),
    and time a fully received response waits for the last response. The latter
    excludes validation/merge and learner time. No GPU synchronization is added.
    """

    def __init__(
        self,
        sources: Sequence[SourceSpec],
        *,
        timeout_s: float = 30.0,
        startup_timeout_s: float = 600.0,
    ) -> None:
        self.sources = tuple(sources)
        if not self.sources or len({s.source_id for s in sources}) != len(sources):
            raise ValueError("sources must be nonempty with unique source IDs")
        if any(not s.source_id or s.num_envs <= 0 for s in sources):
            raise ValueError("each source needs an ID and a positive num_envs")
        if not all(np.isfinite(t) and t > 0 for t in (timeout_s, startup_timeout_s)):
            raise ValueError("timeouts must be finite and positive")
        self.timeout_s = timeout_s
        self.source_slices: dict[str, slice] = {}
        self.num_envs = 0
        for source in self.sources:
            self.source_slices[source.source_id] = slice(
                self.num_envs, self.num_envs + source.num_envs
            )
            self.num_envs += source.num_envs
        self.state: _State | None = None
        self._sequences = [0] * len(self.sources)
        self._probe_active = False
        self._closed = self._inflight = False
        self._connections: list[Connection] = []
        self._processes: list[Any] = []
        self._requests: dict[int, Future] = {}
        self._pool = ThreadPoolExecutor(max_workers=len(sources))
        self.play_capabilities = SimpleNamespace(supports_physics_state_playback=False)
        try:
            context = mp.get_context("spawn")
            for source in self.sources:
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(child, source),
                    name=f"env-{source.source_id}",
                    daemon=False,
                )
                self._connections.append(parent)
                try:
                    process.start()
                    self._processes.append(process)
                finally:
                    child.close()
            metadata = self._exchange(None, [None] * len(sources), startup_timeout_s)
            reference = {k: v for k, v in metadata[0].items() if k != "num_envs"}
            for source, current in zip(self.sources, metadata, strict=True):
                if current["num_envs"] != source.num_envs or any(
                    current[key] != value for key, value in reference.items()
                ):
                    raise ValueError(f"source {source.source_id}: environment metadata mismatch")
            self.obs_groups_spec: dict[str, int] = reference["obs_groups_spec"]
            if "obs" not in self.obs_groups_spec or any(
                not isinstance(d, int) or d <= 0 for d in self.obs_groups_spec.values()
            ):
                raise ValueError("observation groups require positive dimensions and an obs group")
            self.observation_space = SimpleNamespace(shape=reference["observation_shape"])
            self.action_space = SimpleNamespace(shape=reference["action_shape"])
            observation_shapes = {
                (self.obs_groups_spec["obs"],),
                (sum(self.obs_groups_spec.values()),),
            }
            if self.observation_space.shape not in observation_shapes or (
                len(self.action_space.shape) != 1 or self.action_space.shape[0] <= 0
            ):
                raise ValueError("invalid observation or action space")
            self.cfg = SimpleNamespace(
                max_episode_seconds=reference["max_episode_seconds"], ctrl_dt=reference["ctrl_dt"]
            )
            if not all(
                np.isfinite(v) and v > 0 for v in (self.cfg.max_episode_seconds, self.cfg.ctrl_dt)
            ):
                raise ValueError(
                    "episode duration and control timestep must be finite and positive"
                )
        except BaseException:
            self.close()
            raise

    def _exchange(
        self,
        operation: str | None,
        payloads: list[Any],
        timeout: float,
        indices: Sequence[int] | None = None,
    ) -> list[Any]:
        if self._closed:
            raise RuntimeError("synchronous environment is closed")
        if self._inflight:
            raise RuntimeError("a synchronous environment request is already in flight")
        selected = tuple(range(len(self.sources))) if indices is None else tuple(indices)
        if len(payloads) != len(selected):
            raise ValueError("request payload count mismatch")
        self._inflight = True
        try:
            self._requests = {}
            for index, payload in zip(selected, payloads, strict=True):
                if operation is not None:
                    self._sequences[index] += 1
                self._requests[index] = self._pool.submit(
                    _timed_receive,
                    self._connections[index],
                    None if operation is None else (self._sequences[index], operation, payload),
                )
            futures = list(self._requests.values())
            _, pending = wait(futures, timeout=timeout)
            if pending:
                missing = [
                    self.sources[index].source_id
                    for index, future in self._requests.items()
                    if future in pending
                ]
                raise TimeoutError(f"{operation or 'startup'} request: missing {missing}")
            values = []
            response_times = []
            for index, future in self._requests.items():
                source = self.sources[index]
                try:
                    response, started, ready = future.result()
                    source_id, seq, status, value = response
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f"source {source.source_id} worker crashed") from exc
                if source_id != source.source_id or seq != self._sequences[index]:
                    raise RuntimeError(f"source {source.source_id}: stale or mismatched response")
                if status != "ok":
                    raise RuntimeError(f"source {source_id} request {seq} failed:\n{value}")
                values.append(value)
                response_times.append((started, ready))
            if operation == "step":
                barrier = max(ready for _, ready in response_times)
                for value, (started, ready) in zip(values, response_times, strict=True):
                    if not isinstance(value.timing, dict) or set(value.timing) != {
                        "env_step_seconds"
                    }:
                        raise ValueError("missing worker step timing")
                    value.timing.update(
                        request_response_seconds=ready - started,
                        barrier_wait_seconds=barrier - ready,
                    )
            return values
        except BaseException:
            self.close()
            raise
        finally:
            self._inflight = False

    @staticmethod
    def _array(value: Any, shape: tuple[int, ...], dtype: Any, label: str) -> None:
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{label}: expected finite {np.dtype(dtype)} array of shape {shape}")

    def _observations(self, obs: Any, count: int, label: str) -> None:
        if not isinstance(obs, dict) or obs.keys() != self.obs_groups_spec.keys():
            raise ValueError(f"{label}: observation groups mismatch")
        for key, dim in self.obs_groups_spec.items():
            self._array(obs[key], (count, dim), np.float32, f"{label}/{key}")

    def _validate_state(self, source: SourceSpec, state: _State, *, transition: bool) -> None:
        label, count = source.source_id, source.num_envs
        self._observations(state.obs, count, label)
        self._array(state.reward, (count,), np.float32, f"{label}/reward")
        for name in ("terminated", "truncated"):
            self._array(getattr(state, name), (count,), np.bool_, f"{label}/{name}")
        if not isinstance(state.info, dict):
            raise ValueError(f"{label}: info must be a dict")
        if transition and (
            not isinstance(state.timing, dict)
            or set(state.timing)
            != {"env_step_seconds", "request_response_seconds", "barrier_wait_seconds"}
            or any(
                type(value) is not float or not np.isfinite(value) or value < 0
                for value in state.timing.values()
            )
        ):
            raise ValueError(f"{label}: invalid source timing")
        if state.final_observation is not None:
            self._observations(state.final_observation, count, f"{label}/final_observation")
        elif transition and np.any(state.terminated | state.truncated):
            raise ValueError(f"{label}: done requires final_observation")

    def _aggregate(self, states: list[_State], *, transition: bool) -> _State:
        try:
            for source, state in zip(self.sources, states, strict=True):
                self._validate_state(source, state, transition=transition)
            self.state = _State(
                {key: np.concatenate([s.obs[key] for s in states]) for key in self.obs_groups_spec},
                np.concatenate([s.reward for s in states]),
                np.concatenate([s.terminated for s in states]),
                np.concatenate([s.truncated for s in states]),
                {
                    "source_logs": {
                        s.source_id: v.info for s, v in zip(self.sources, states, strict=True)
                    }
                },
                {
                    key: np.concatenate([(s.final_observation or s.obs)[key] for s in states])
                    for key in self.obs_groups_spec
                }
                if any(s.final_observation is not None for s in states)
                else None,
            )
            if transition:
                # Publish timing only after every source response and transition passed validation.
                self.state.info["source_timings"] = {
                    source.source_id: dict(state.timing or {})
                    for source, state in zip(self.sources, states, strict=True)
                }
            return self.state
        except BaseException:
            self.close()
            raise

    def init_state(self) -> _State:
        if self.state is not None:
            if self._closed:
                raise RuntimeError("synchronous environment is closed")
            return self.state
        return self._aggregate(
            self._exchange("init_state", [None] * len(self.sources), self.timeout_s),
            transition=False,  # Initialization flags may be reset sentinels, not terminal steps.
        )

    def step(self, actions: np.ndarray) -> _State:
        self._array(actions, (self.num_envs, *self.action_space.shape), np.float32, "actions")
        self.step_selected(
            {
                source.source_id: actions[self.source_slices[source.source_id]]
                for source in self.sources
            }
        )
        assert self.state is not None
        return self.state

    def step_selected(self, actions: Mapping[str, np.ndarray]) -> dict[str, _State]:
        """Advance only the declared full source pools, in canonical source order.

        Returned states are the sole new transitions. The aggregate state caches
        all observations but has zero reward/flags outside ``active_sources``;
        those inactive slots are not samples and must never enter storage.
        """
        if self._closed:
            raise RuntimeError("synchronous environment is closed")
        if self.state is None:
            raise RuntimeError("call init_state or reset before step")
        if not isinstance(actions, Mapping):
            raise ValueError("selected sources must be a nonempty ordered subset")
        indices = [
            index for index, source in enumerate(self.sources) if source.source_id in actions
        ]
        if not indices or tuple(actions) != tuple(
            self.sources[index].source_id for index in indices
        ):
            raise ValueError("selected sources must be a nonempty ordered subset")
        for index in indices:
            source = self.sources[index]
            self._array(
                actions[source.source_id],
                (source.num_envs, *self.action_space.shape),
                np.float32,
                f"{source.source_id}/actions",
            )
        values = self._exchange("step", list(actions.values()), self.timeout_s, indices=indices)
        try:
            for index, state in zip(indices, values, strict=True):
                self._validate_state(self.sources[index], state, transition=True)
            result = dict(zip(actions, values, strict=True))
            obs = {key: value.copy() for key, value in self.state.obs.items()}
            rewards = np.zeros(self.num_envs, np.float32)
            terminated, truncated = np.zeros(self.num_envs, bool), np.zeros(self.num_envs, bool)
            final = (
                {key: value.copy() for key, value in obs.items()}
                if any(state.final_observation is not None for state in values)
                else None
            )
            for key, state in result.items():
                span = self.source_slices[key]
                for group in obs:
                    obs[group][span] = state.obs[group]
                    if final is not None:
                        final[group][span] = (state.final_observation or state.obs)[group]
                rewards[span] = state.reward
                terminated[span], truncated[span] = state.terminated, state.truncated
            self.state = _State(
                obs,
                rewards,
                terminated,
                truncated,
                {
                    "active_sources": tuple(result),
                    "source_logs": {key: state.info for key, state in result.items()},
                    "source_timings": {
                        key: dict(state.timing or {}) for key, state in result.items()
                    },
                },
                final,
            )
            return result
        except BaseException:
            self.close()
            raise

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        indices = np.asarray(env_indices)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= self.num_envs)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("reset requires unique valid environment indices")
        self.init_state()
        selections = [
            np.flatnonzero((indices >= span.start) & (indices < span.stop))
            for span in self.source_slices.values()
        ]
        payloads = [
            indices[selection] - span.start
            for selection, span in zip(selections, self.source_slices.values(), strict=True)
        ]
        results = self._exchange("reset", payloads, self.timeout_s)
        return self._apply_reset_results(indices, selections, results)

    def _apply_reset_results(
        self, indices: np.ndarray, selections: list[np.ndarray], results: list[Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        try:
            assert self.state is not None
            state = self.state
            obs = {
                key: np.empty((len(indices), dim), np.float32)
                for key, dim in self.obs_groups_spec.items()
            }
            logs = {}
            for source, selection, (part, info) in zip(
                self.sources, selections, results, strict=True
            ):
                self._observations(part, len(selection), f"{source.source_id}/reset")
                if not isinstance(info, dict):
                    raise ValueError("reset info must be a dict")
                logs[source.source_id] = info
                for key in obs:
                    obs[key][selection] = part[key]
            for key in obs:
                state.obs[key][indices] = obs[key]
            state.reward[indices] = 0
            state.terminated[indices] = state.truncated[indices] = False
            state.final_observation = None
            state.info = {"source_logs": logs}
            return obs, state.info
        except BaseException:
            self.close()
            raise

    def reset_probe(self, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start an isolated evaluation episode in every source, without a rollout."""
        if type(seed) is not int or seed < 0 or self._probe_active:
            raise ValueError("probe reset needs a nonnegative seed and no active probe")
        self.init_state()
        results = self._exchange("reset_probe", [seed] * len(self.sources), self.timeout_s)
        indices = np.arange(self.num_envs)
        selections = [indices[span] for span in self.source_slices.values()]
        result = self._apply_reset_results(indices, selections, results)
        self._probe_active = True
        return result

    def finish_probe(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Disable probe instrumentation and reset every pool for new training episodes."""
        if not self._probe_active:
            raise ValueError("no active probe")
        results = self._exchange("finish_probe", [None] * len(self.sources), self.timeout_s)
        if any(value is not None for value in results):
            self.close()
            raise ValueError("finish_probe must return None")
        result = self.reset(np.arange(self.num_envs))
        self._probe_active = False
        return result

    def set_episode_length_buf(self, values: np.ndarray) -> None:
        if (
            not isinstance(values, np.ndarray)
            or values.shape != (self.num_envs,)
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
        ):
            raise ValueError("episode lengths must be nonnegative integer values, one per env")
        self._exchange(
            "set_episode_length_buf",
            [values[self.source_slices[s.source_id]] for s in self.sources],
            self.timeout_s,
        )

    def set_nan_guard(self, guard: Any) -> None:
        if guard is not None:
            raise ValueError("synchronous physics services do not support playback NaN guards")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closing = []
        for index, conn in enumerate(self._connections):
            request = self._requests.get(index)
            if not self._inflight or request is None or request.done():
                self._pool.submit(conn.send, (self._sequences[index] + 1, "close", None))
                closing.append(index)
        # Vendor services release nested pools and shared memory in env.close().
        # A timed-out request still executing cannot acknowledge close safely.
        deadline = time.monotonic() + 15.0
        for index in closing:
            if index < len(self._processes):
                self._processes[index].join(max(0.0, deadline - time.monotonic()))
        # A leader may already have exited while its vendor workers remain alive.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for process in self._processes:
                if sys.platform == "linux":
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                if process.is_alive():
                    process.kill() if sig == signal.SIGKILL else process.terminate()
            deadline = time.monotonic() + 0.5
            for process in self._processes:
                process.join(max(0.0, deadline - time.monotonic()))
            if sys.platform == "linux":
                while time.monotonic() < deadline and any(
                    _group_exists(process.pid) for process in self._processes
                ):
                    time.sleep(0.01)
        for conn in self._connections:
            conn.close()
        self._pool.shutdown(wait=True, cancel_futures=True)
        for process in self._processes:
            process.close()

    def __enter__(self) -> SynchronousEnv:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
