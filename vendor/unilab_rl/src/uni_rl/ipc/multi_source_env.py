"""Synchronous, simulator-agnostic composition of injected environment factories.

Each non-daemon spawn worker owns a fixed slice. Bulk numpy data uses
preallocated, double-buffered shared memory; pipes carry only control,
descriptors, bounded scalar metadata and error tracebacks. A failed source
poisons the entire facade, with no retry, removal or partial publication.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import re
import signal
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from multiprocessing.connection import wait
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np

from uni_rl.env_contract import EnvAlgoCapabilities, EnvFactory, EnvProtocol, EnvStateProtocol
from uni_rl.ipc._multi_source_resources import SourceResources
from uni_rl.ipc._multi_source_shared import (
    Descriptor,
    SharedArrays,
    array,
    ensure_memory_budget,
    metadata,
    receive_message,
    send_message,
    transport_nbytes,
)
from uni_rl.ipc._multi_source_worker import run_source
from uni_rl.utils.nan_guard import NanGuard


@dataclass(frozen=True)
class EnvSourceSpec:
    """One named source; the caller owns factory, configuration, count and seed."""

    name: str
    factory: EnvFactory
    num_envs: int
    env_cfg_override: Mapping[str, Any] | None = None
    seed: int | None = None


@dataclass(frozen=True)
class MultiSourceOptions:
    """Wall-clock budgets shared across sources, not multiplied by source count."""

    startup_timeout_s: float = 600
    operation_timeout_s: float = 120
    shutdown_timeout_s: float = 10


class MultiSourceEnvError(RuntimeError):
    """Fatal source failure with operation identity and remote diagnostic context."""

    def __init__(self, source_name: str, operation: str, sequence_id: int, remote_traceback: str):
        self.source_name = source_name
        self.operation = operation
        self.sequence_id = sequence_id
        self.remote_traceback = remote_traceback
        super().__init__(source_name, operation, sequence_id, remote_traceback)

    def __str__(self) -> str:
        return f"source {self.source_name!r} failed during {self.operation} (sequence {self.sequence_id})\n{self.remote_traceback}"


@dataclass
class _State:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any]
    final_observation: dict[str, np.ndarray] | None


@dataclass
class _Source:
    spec: EnvSourceSpec
    selection: slice
    connection: Any
    process: Any
    lock: Any
    resources: SourceResources
    descriptor: Descriptor | None = None
    shared: SharedArrays | None = None
    started: bool = False


class _MultiSourceEnv:
    def __init__(self, sources: Sequence[EnvSourceSpec], options: MultiSourceOptions):
        self._mutex = threading.RLock()
        self._sources: list[_Source] = []
        self._options = options
        self._failure: MultiSourceEnvError | None = None
        self._closed = False
        self._state: _State | None = None
        self._sequence_id = 0
        self._descriptor: Descriptor
        self._num_envs = sum(source.num_envs for source in sources)
        deadline = time.monotonic() + options.startup_timeout_s
        context = mp.get_context("spawn")
        offset = 0
        current_name = sources[0].name
        try:
            for spec in sources:
                current_name = spec.name
                parent_connection, child_connection = context.Pipe()
                lock = context.Lock()
                try:
                    resources = SourceResources()
                except BaseException:
                    parent_connection.close()
                    child_connection.close()
                    raise
                process = context.Process(
                    target=run_source,
                    args=(spec, options, child_connection, lock, resources.writer),
                    name=f"env-source-{spec.name}",
                    daemon=False,
                )
                source = _Source(
                    spec,
                    slice(offset, offset + spec.num_envs),
                    parent_connection,
                    process,
                    lock,
                    resources,
                )
                self._sources.append(source)
                try:
                    process.start()
                finally:
                    child_connection.close()
                    resources.release_parent_writer()
                offset += spec.num_envs
            messages = self._barrier("startup", 0, "descriptor", deadline)
            for source in self._sources:
                current_name = source.spec.name
                descriptor = messages[current_name].get("descriptor")
                if not isinstance(descriptor, Descriptor):
                    raise ValueError("worker did not supply an environment descriptor")
                source.descriptor = descriptor
                if source is not self._sources[0]:
                    self._descriptor.compatible_with(descriptor)
                else:
                    self._descriptor = descriptor
            ensure_memory_budget(
                sum(
                    transport_nbytes(self._descriptor, source.spec.num_envs)
                    for source in self._sources
                )
            )
            for source in self._sources:
                current_name = source.spec.name
                source.shared = SharedArrays(self._descriptor, source.spec.num_envs, source.lock)
                send_message(
                    source.connection,
                    {
                        "operation": "attach",
                        "sequence_id": 0,
                        "shm_name": source.shared.memory.name,
                    },
                )
            messages = self._barrier("startup", 0, "ack", deadline)
            self._initial_state = self._assemble(messages, 0, deadline)
            descriptor = self._descriptor
            self._cfg = SimpleNamespace(
                ctrl_dt=descriptor.ctrl_dt, max_episode_seconds=descriptor.max_episode_seconds
            )
            low = (
                None
                if descriptor.action_low is None
                else np.asarray(descriptor.action_low, dtype=descriptor.action_dtype)
            )
            high = (
                None
                if descriptor.action_high is None
                else np.asarray(descriptor.action_high, dtype=descriptor.action_dtype)
            )
            self._action_space = SimpleNamespace(
                shape=descriptor.action_shape,
                dtype=np.dtype(descriptor.action_dtype),
                low=low,
                high=high,
            )
            self._observation_space = SimpleNamespace(
                shape=descriptor.observation_shape, dtype=np.dtype(descriptor.observation_dtype)
            )
            self._algo_capabilities = EnvAlgoCapabilities(
                action_low=low, action_high=high, joint_names=descriptor.joint_names
            )
            self._play_capabilities = SimpleNamespace(supports_physics_state_playback=False)
        except BaseException as error:
            self._poison(error, current_name, "startup", 0)
            raise

    def _check(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RuntimeError("multi-source environment is closed")
        for source in self._sources:
            if source.process.exitcode is not None:
                error = MultiSourceEnvError(
                    source.spec.name,
                    "health",
                    self._sequence_id,
                    f"source exited with code {source.process.exitcode}",
                )
                self._poison(error, source.spec.name, error.operation, error.sequence_id)

    def _poison(
        self, error: BaseException, source_name: str, operation: str, sequence_id: int
    ) -> None:
        failure = (
            error
            if isinstance(error, MultiSourceEnvError)
            else MultiSourceEnvError(
                source_name,
                operation,
                sequence_id,
                "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            )
        )
        self._failure = failure
        self._cleanup(graceful=False)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise error
        raise failure from (None if error is failure else error)

    def _barrier(
        self, operation: str, sequence_id: int, kind: str, deadline: float
    ) -> dict[str, dict[str, Any]]:
        pending = {source.spec.name: source for source in self._sources}
        messages: dict[str, dict[str, Any]] = {}
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MultiSourceEnvError(
                    next(iter(pending)), operation, sequence_id, f"timeout waiting for {kind}"
                )
            objects = [source.connection for source in self._sources]
            objects.extend(source.process.sentinel for source in self._sources)
            ready = wait(objects, timeout=remaining)
            for source in self._sources:
                name = source.spec.name
                if source.connection in ready:
                    try:
                        message = receive_message(
                            source.connection, deadline=deadline, sentinel=source.process.sentinel
                        )
                    except (EOFError, OSError, ValueError) as error:
                        raise MultiSourceEnvError(
                            name,
                            operation,
                            sequence_id,
                            f"source control connection failed: {error}",
                        ) from error
                    if (
                        message.get("sequence_id") != sequence_id
                        or message.get("operation") != operation
                    ):
                        raise MultiSourceEnvError(
                            name, operation, sequence_id, f"stale or out-of-order ACK: {message}"
                        )
                    if message.get("kind") == "error":
                        raise MultiSourceEnvError(
                            name,
                            operation,
                            sequence_id,
                            str(message.get("remote_traceback", "missing remote traceback")),
                        )
                    if (
                        message.get("kind") == "started"
                        and kind == "descriptor"
                        and not source.started
                    ):
                        source.started = True
                    elif message.get("kind") == kind and name in pending:
                        messages[name] = message
                        del pending[name]
                    else:
                        raise MultiSourceEnvError(
                            name, operation, sequence_id, f"unexpected or duplicate ACK: {message}"
                        )
                if source.process.sentinel in ready and not source.connection.poll():
                    raise MultiSourceEnvError(
                        name,
                        operation,
                        sequence_id,
                        f"source exited with code {source.process.exitcode}",
                    )
        return messages

    def _assemble(
        self, messages: dict[str, dict[str, Any]], sequence_id: int, deadline: float
    ) -> _State:
        arrays: dict[str, list[np.ndarray]] = {}
        info: dict[str, Any] = {"log": {}}
        statistics = {}
        for source in self._sources:
            shared = source.shared
            assert shared is not None
            try:
                shared.check_publication(1, sequence_id, max(0, deadline - time.monotonic()))
                for name, value in shared.arrays.items():
                    if name in {
                        "reward",
                        "terminated",
                        "truncated",
                        "steps",
                        "mask",
                    } or name.startswith(("obs/", "final/")):
                        snapshot = value[sequence_id % 2].copy()
                        array(snapshot, snapshot.shape, name, value.dtype.str)
                        arrays.setdefault(name, []).append(snapshot)
                self._merge_info(info, source.spec.name, messages[source.spec.name]["info"])
                statistics[source.spec.name] = MappingProxyType(
                    metadata(messages[source.spec.name]["statistics"], "source_statistics")
                )
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise MultiSourceEnvError(
                    source.spec.name,
                    messages[source.spec.name]["operation"],
                    sequence_id,
                    str(error),
                ) from error
        combined = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
        observations = {group: combined[f"obs/{group}"] for group, _, _ in self._descriptor.groups}
        final_observation = None
        info["steps"] = combined["steps"]
        info["_final_observation"] = combined["mask"]
        if np.any(combined["mask"]):
            final_observation = {
                group: combined[f"final/{group}"] for group, _, _ in self._descriptor.groups
            }
            info["final_observation"] = final_observation
        self._source_statistics = MappingProxyType(statistics)
        return _State(
            observations,
            combined["reward"],
            combined["terminated"],
            combined["truncated"],
            info,
            final_observation,
        )

    @staticmethod
    def _merge_info(target: dict[str, Any], source_name: str, info: dict[str, Any]) -> None:
        for key, value in info.items():
            if key == "log":
                target["log"].update(
                    {f"source/{source_name}/{metric}": scalar for metric, scalar in value.items()}
                )
            else:
                target[f"source/{source_name}/{key}"] = value

    def _operate(
        self, operation: str, payloads: dict[str, dict[str, Any]] | None = None
    ) -> dict[str, dict[str, Any]]:
        self._sequence_id += 1
        sequence_id = self._sequence_id
        deadline = time.monotonic() + self._options.operation_timeout_s
        current_name = self._sources[0].spec.name
        try:
            for source in self._sources:
                current_name = source.spec.name
                assert source.shared is not None
                source.shared.publish(0, sequence_id, max(0, deadline - time.monotonic()))
                message = {"operation": operation, "sequence_id": sequence_id}
                if payloads is not None:
                    message.update(payloads[current_name])
                send_message(source.connection, message)
            messages = self._barrier(operation, sequence_id, "ack", deadline)
            state = self._assemble(messages, sequence_id, deadline)
            if self._state is not None:
                self._state = state
            else:
                self._initial_state = state
            return messages
        except BaseException as error:
            self._poison(error, current_name, operation, sequence_id)
            raise

    @property
    def num_envs(self) -> int:
        self._check()
        return self._num_envs

    @property
    def cfg(self) -> Any:
        self._check()
        return self._cfg

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        self._check()
        return {group: dimension for group, dimension, _ in self._descriptor.groups}

    @property
    def observation_space(self) -> Any:
        self._check()
        return self._observation_space

    @property
    def action_space(self) -> Any:
        self._check()
        return self._action_space

    @property
    def algo_capabilities(self) -> EnvAlgoCapabilities:
        self._check()
        return self._algo_capabilities

    @property
    def play_capabilities(self) -> Any:
        self._check()
        return self._play_capabilities

    @property
    def state(self) -> EnvStateProtocol | None:
        self._check()
        return self._state

    @property
    def source_statistics(self) -> Mapping[str, Mapping[str, int | float]]:
        """Return an immutable barrier-consistent snapshot of source counters."""
        self._check()
        return self._source_statistics

    def init_state(self) -> EnvStateProtocol:
        with self._mutex:
            self._check()
            if self._state is None:
                self._state = self._initial_state
            return self._state

    def step(self, actions: np.ndarray) -> EnvStateProtocol:
        with self._mutex:
            self._check()
            actions = array(actions, (self._num_envs, *self._descriptor.action_shape), "actions")
            if actions.dtype.kind != "f":
                raise ValueError("actions must have a floating dtype")
            with np.errstate(over="ignore", invalid="ignore"):
                converted = actions.astype(self._descriptor.action_dtype, copy=False)
            array(converted, actions.shape, "actions after dtype conversion")
            self.init_state()
            slot = (self._sequence_id + 1) % 2
            for source in self._sources:
                assert source.shared is not None
                np.copyto(source.shared.arrays["actions"][slot], converted[source.selection])
            self._operate("step")
            assert self._state is not None
            return self._state

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        with self._mutex:
            self._check()
            if (
                not isinstance(env_indices, np.ndarray)
                or env_indices.ndim != 1
                or env_indices.dtype.kind not in "iu"
            ):
                raise ValueError("reset indices must be a one-dimensional integer ndarray")
            if np.any(env_indices < 0) or np.any(env_indices >= self._num_envs):
                raise ValueError("reset indices are out of range")
            if len(np.unique(env_indices)) != len(env_indices):
                raise ValueError("reset indices must not contain duplicates")
            self.init_state()
            positions = {}
            payloads = {}
            slot = (self._sequence_id + 1) % 2
            for source in self._sources:
                selection = source.selection
                selected = np.flatnonzero(
                    (env_indices >= selection.start) & (env_indices < selection.stop)
                )
                positions[source.spec.name] = selected
                payloads[source.spec.name] = {"count": len(selected)}
                assert source.shared is not None
                source.shared.arrays["indices"][slot, : len(selected)] = (
                    env_indices[selected] - selection.start
                )
            observations = {
                group: np.empty((len(env_indices), dimension), dtype=dtype)
                for group, dimension, dtype in self._descriptor.groups
            }
            info: dict[str, Any] = {"log": {}}
            if len(env_indices):
                messages = self._operate("reset", payloads)
                for source in self._sources:
                    selected = positions[source.spec.name]
                    assert source.shared is not None
                    for group in observations:
                        observations[group][selected] = source.shared.arrays[f"reset/{group}"][
                            slot, : len(selected)
                        ]
                    self._merge_info(
                        info, source.spec.name, messages[source.spec.name]["reset_info"]
                    )
            assert self._state is not None
            info["steps"] = self._state.info["steps"][env_indices].copy()
            return observations, info

    def set_episode_length_buf(self, values: np.ndarray) -> None:
        """Scatter global timeout counters through each source's public setter."""
        with self._mutex:
            self._check()
            values = array(values, (self._num_envs,), "episode lengths")
            upper = min(
                np.iinfo(np.int64).max, np.iinfo(np.dtype(self._descriptor.steps_dtype)).max
            )
            if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values > upper):
                raise ValueError("episode lengths must be non-negative representable integers")
            slot = (self._sequence_id + 1) % 2
            for source in self._sources:
                assert source.descriptor is not None and source.shared is not None
                if not source.descriptor.episode_length_setter:
                    error = MultiSourceEnvError(
                        source.spec.name,
                        "set_episode_length_buf",
                        self._sequence_id + 1,
                        "source does not support set_episode_length_buf",
                    )
                    self._poison(error, source.spec.name, error.operation, error.sequence_id)
                np.copyto(
                    source.shared.arrays["lengths"][slot],
                    values[source.selection],
                    casting="unsafe",
                )
            self._operate("set_episode_length_buf")

    def set_nan_guard(self, guard: Any) -> None:
        """Rebuild guards in workers; never pickle captured physics buffers."""
        with self._mutex:
            self._check()
            if guard is not None and not isinstance(guard, NanGuard):
                raise ValueError("set_nan_guard expects NanGuard or None")
            config = None if guard is None else asdict(guard.cfg)
            self._operate(
                "set_nan_guard",
                {source.spec.name: {"guard_config": config} for source in self._sources},
            )

    def _signal_sources(self, signum: int) -> None:
        for source in self._sources:
            process = source.process
            if process.pid is None:
                continue
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signum)
            if process.is_alive():
                if signum == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()

    def _cleanup(self, *, graceful: bool) -> None:
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + self._options.shutdown_timeout_s
        if graceful:
            for source in self._sources:
                with suppress(BrokenPipeError, EOFError, OSError, ValueError):
                    send_message(
                        source.connection,
                        {"operation": "close", "sequence_id": self._sequence_id + 1},
                    )
            pending = [source for source in self._sources if source.process.pid is not None]
            while pending and time.monotonic() < deadline:
                ready = wait(
                    [source.connection for source in pending]
                    + [source.process.sentinel for source in pending],
                    max(0, deadline - time.monotonic()),
                )
                for source in pending[:]:
                    if source.connection in ready or source.process.sentinel in ready:
                        pending.remove(source)
        self._signal_sources(signal.SIGTERM)
        term_deadline = min(deadline, time.monotonic() + 0.2)
        for source in self._sources:
            if source.process.pid is not None:
                source.process.join(max(0, term_deadline - time.monotonic()))
        self._signal_sources(signal.SIGKILL)
        reap_deadline = max(deadline, time.monotonic() + 0.2)
        cleanup_errors = []
        for source in self._sources:
            if source.process.pid is not None:
                source.process.join(max(0, reap_deadline - time.monotonic()))
                if not source.process.is_alive():
                    source.process.close()
            source.connection.close()
            if source.shared is not None:
                try:
                    source.shared.close(unlink=True)
                except (OSError, BufferError) as error:
                    cleanup_errors.append(f"{source.spec.name}: transport cleanup failed: {error}")
        tracker_deadline = max(deadline, time.monotonic() + 1)
        for source in self._sources:
            try:
                source.resources.finish(max(0, tracker_deadline - time.monotonic()))
            except RuntimeError as error:
                cleanup_errors.append(f"{source.spec.name}: {error}")
        if cleanup_errors:
            details = "\n".join(cleanup_errors)
            if self._failure is not None:
                self._failure.remote_traceback += f"\nCleanup errors:\n{details}"
            else:
                self._failure = MultiSourceEnvError("cleanup", "close", self._sequence_id, details)
                raise self._failure

    def close(self) -> None:
        with self._mutex:
            self._cleanup(graceful=True)

    def __enter__(self) -> _MultiSourceEnv:
        self._check()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_mutex"):
            with suppress(BaseException):
                self.close()


def make_multi_source_env(
    sources: Sequence[EnvSourceSpec], *, options: MultiSourceOptions = MultiSourceOptions()
) -> EnvProtocol:
    """Spawn all sources, validate their contracts and return one synchronous env.

    Use from a spawn-safe main entrypoint. Factories must be pickleable;
    source names must be unique path-safe identifiers. No backend selection,
    environment quotas, task configuration or device placement is inferred.
    """
    sources = tuple(sources)
    if not sources:
        raise ValueError("at least one environment source is required")
    if not isinstance(options, MultiSourceOptions):
        raise TypeError("options must be MultiSourceOptions")
    for name, value in asdict(options).items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    names = set()
    for source in sources:
        if not isinstance(source, EnvSourceSpec):
            raise TypeError("sources must contain EnvSourceSpec instances")
        if not isinstance(source.name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", source.name
        ):
            raise ValueError("source names must be nonempty path-safe identifiers")
        if source.name in names:
            raise ValueError(f"duplicate source name: {source.name}")
        names.add(source.name)
        if type(source.num_envs) is not int or source.num_envs <= 0:
            raise ValueError("source num_envs must be a positive integer")
        if source.seed is not None and (
            type(source.seed) is not int or not 0 <= source.seed < 2**32
        ):
            raise ValueError("source seed must be an integer in [0, 2**32)")
        if source.env_cfg_override is not None and not isinstance(source.env_cfg_override, Mapping):
            raise TypeError("env_cfg_override must be a Mapping or None")
        if not callable(source.factory):
            raise TypeError("source factory must be callable")
        try:
            pickle.dumps(source)
        except Exception as error:
            raise ValueError(
                f"source {source.name!r} factory/config must be pickleable for spawn"
            ) from error
    return _MultiSourceEnv(sources, options)


__all__ = ["EnvSourceSpec", "MultiSourceOptions", "make_multi_source_env", "MultiSourceEnvError"]
