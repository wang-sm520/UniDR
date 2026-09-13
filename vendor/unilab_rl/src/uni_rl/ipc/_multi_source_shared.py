"""Validated, double-buffered numpy transport for synchronous environment sources."""

from __future__ import annotations

import os
import pickle
import shutil
import struct
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.connection import wait
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from uni_rl.env_contract import (
    EnvProtocol,
    EnvStateProtocol,
    SupportsEpisodeLengthBufferProtocol,
    get_algo_capabilities,
)
from uni_rl.ipc._multi_source_resources import reserve_transport_memory

MAX_MESSAGE_BYTES = 65536


@dataclass(frozen=True)
class Descriptor:
    """Cold-path schema; environment count is intentionally source-local."""

    groups: tuple[tuple[str, int, str], ...]
    observation_shape: tuple[int, ...]
    observation_dtype: str
    action_shape: tuple[int, ...]
    action_dtype: str
    action_low: tuple[float, ...] | None
    action_high: tuple[float, ...] | None
    joint_names: tuple[str, ...] | None
    reward_dtype: str
    steps_dtype: str
    ctrl_dt: float
    max_episode_seconds: float
    episode_length_setter: bool
    physics_playback: bool

    def compatible_with(self, other: Descriptor) -> None:
        """Reject differences that change policy I/O or transition semantics."""
        for field in self.__dataclass_fields__:
            if field not in {"episode_length_setter", "physics_playback"}:
                if getattr(self, field) != getattr(other, field):
                    raise ValueError(f"source descriptor mismatch: {field}")


def array(
    value: Any,
    shape: tuple[int, ...],
    label: str,
    dtype: str | None = None,
    *,
    allow_infinite: bool = False,
) -> np.ndarray:
    """Validate numeric arrays; only space-bound metadata may allow infinity."""
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{label} must be an ndarray with shape {shape}")
    if value.dtype.kind not in "biuf" or not value.dtype.isnative:
        raise ValueError(f"{label} has unsupported dtype {value.dtype}")
    if dtype is not None and value.dtype.str != dtype:
        raise ValueError(f"{label} dtype must be {dtype}, got {value.dtype.str}")
    if np.isnan(value).any() if allow_infinite else not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite data")
    return value


def _floating_dtype(value: Any, label: str) -> str:
    dtype = np.dtype(value)
    if dtype.kind != "f" or not dtype.isnative or dtype.itemsize not in (2, 4, 8):
        raise ValueError(f"{label} must use native float16, float32 or float64")
    return str(dtype.str)


def _space_shape(space: Any, label: str) -> tuple[int, ...]:
    shape = tuple(space.shape)
    if len(shape) != 1 or not isinstance(shape[0], (int, np.integer)) or shape[0] <= 0:
        raise ValueError(f"{label} must be a nonempty flat space")
    return shape


def describe(env: EnvProtocol, state: EnvStateProtocol, num_envs: int) -> Descriptor:
    """Describe actual initialized arrays rather than trusting dimension claims."""
    if env.num_envs != num_envs:
        raise ValueError(f"factory returned num_envs={env.num_envs}, expected {num_envs}")
    spec = env.obs_groups_spec
    if not isinstance(spec, dict) or "obs" not in spec:
        raise ValueError("obs_groups_spec must be an ordered dict containing 'obs'")
    if not isinstance(state.obs, dict) or tuple(state.obs) != tuple(spec):
        raise ValueError("observation key order must match obs_groups_spec")
    groups = []
    for name, dimension in spec.items():
        if not isinstance(name, str) or not name or type(dimension) is not int or dimension <= 0:
            raise ValueError("observation groups require nonempty names and positive integer dims")
        observation = array(state.obs[name], (num_envs, dimension), f"obs/{name}")
        groups.append((name, dimension, _floating_dtype(observation.dtype, f"obs/{name}")))
    reward = array(state.reward, (num_envs,), "reward")
    if not isinstance(state.info, dict) or "steps" not in state.info:
        raise ValueError("state.info must contain per-env integer 'steps'")
    steps = array(state.info["steps"], (num_envs,), "steps")
    if steps.dtype.kind not in "iu":
        raise ValueError("steps must have an integer dtype")
    action_shape = _space_shape(env.action_space, "action_space")
    capabilities = get_algo_capabilities(env)
    bounds: list[tuple[float, ...] | None] = []
    for name in ("low", "high"):
        capability = getattr(capabilities, f"action_{name}")
        space_bound = getattr(env.action_space, name, None)
        bound = capability if capability is not None else space_bound
        if bound is not None:
            bound = array(bound, action_shape, f"action_{name}", allow_infinite=True)
            if capability is not None and space_bound is not None:
                space_bound = array(
                    space_bound, action_shape, f"action_space.{name}", allow_infinite=True
                )
                if not np.array_equal(bound, space_bound):
                    raise ValueError(f"action_{name} disagrees with action_space.{name}")
            bounds.append(tuple(float(value) for value in bound))
        else:
            bounds.append(None)
    if (bounds[0] is None) != (bounds[1] is None):
        raise ValueError("action bounds must supply both low and high")
    if bounds[0] is not None and bounds[1] is not None:
        if np.any(np.asarray(bounds[0]) >= np.asarray(bounds[1])):
            raise ValueError("action_low must be strictly less than action_high")
    joint_names = capabilities.joint_names
    if joint_names is not None:
        if (
            not isinstance(joint_names, tuple)
            or len(joint_names) != action_shape[0]
            or any(not isinstance(name, str) or not name for name in joint_names)
            or len(set(joint_names)) != len(joint_names)
        ):
            raise ValueError("joint_names must be unique strings in action order")
    ctrl_dt = float(env.cfg.ctrl_dt)
    max_episode_seconds = float(env.cfg.max_episode_seconds)
    if not all(np.isfinite(value) and value > 0 for value in (ctrl_dt, max_episode_seconds)):
        raise ValueError("ctrl_dt and max_episode_seconds must be finite and positive")
    return Descriptor(
        groups=tuple(groups),
        observation_shape=_space_shape(env.observation_space, "observation_space"),
        observation_dtype=_floating_dtype(
            getattr(env.observation_space, "dtype", state.obs["obs"].dtype), "observation_space"
        ),
        action_shape=action_shape,
        action_dtype=_floating_dtype(
            getattr(env.action_space, "dtype", np.float32), "action_space"
        ),
        action_low=bounds[0],
        action_high=bounds[1],
        joint_names=joint_names,
        reward_dtype=_floating_dtype(reward.dtype, "reward"),
        steps_dtype=steps.dtype.str,
        ctrl_dt=ctrl_dt,
        max_episode_seconds=max_episode_seconds,
        episode_length_setter=isinstance(env, SupportsEpisodeLengthBufferProtocol),
        physics_playback=bool(env.play_capabilities.supports_physics_state_playback),
    )


def metadata(value: Any, label: str = "info", depth: int = 0) -> Any:
    """Copy bounded scalar metadata without smuggling bulk arrays through pipes."""
    if depth > 8:
        raise ValueError(f"{label}: metadata nesting exceeds eight levels")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label}: metadata keys must be strings")
        return {key: metadata(item, f"{label}/{key}", depth + 1) for key, item in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError(f"{label}: unsupported metadata; expected finite scalars or string-keyed maps")


def info_metadata(info: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(info, dict) or any(not isinstance(key, str) for key in info):
        raise ValueError("info must be a string-keyed dict")
    result = {
        key: metadata(value, f"info/{key}")
        for key, value in info.items()
        if key not in {"steps", "final_observation", "_final_observation"}
    }
    if "log" in result:
        if not isinstance(result["log"], dict) or any(
            not isinstance(value, (int, float)) for value in result["log"].values()
        ):
            raise ValueError("info/log must map metric names to finite numeric scalars")
    return result


def send_message(connection: Any, message: dict[str, Any]) -> None:
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("source control/metadata message exceeds 64 KiB")
    connection.send_bytes(data)


def receive_message(
    connection: Any, *, deadline: float | None = None, sentinel: Any = None
) -> dict[str, Any]:
    if deadline is None:
        payload = connection.recv_bytes(MAX_MESSAGE_BYTES)
    else:

        def read_exact(size: int) -> bytes:
            chunks = []
            remaining = size
            while remaining:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise TimeoutError("timeout reading source control frame")
                handles = [connection] if sentinel is None else [connection, sentinel]
                ready = wait(handles, timeout)
                if connection not in ready:
                    if sentinel is not None and sentinel in ready:
                        raise EOFError("source exited during a partial control frame")
                    raise TimeoutError("timeout reading source control frame")
                chunk = os.read(connection.fileno(), remaining)
                if not chunk:
                    raise EOFError("source closed a partial control frame")
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        length = struct.unpack("!i", read_exact(4))[0]
        if not 0 <= length <= MAX_MESSAGE_BYTES:
            raise ValueError("source control frame exceeds 64 KiB or has an invalid length")
        payload = read_exact(length)
    message = pickle.loads(payload)
    if not isinstance(message, dict):
        raise ValueError("source sent a malformed control message")
    return message


class SharedArrays:
    """One owner-allocated segment with two slots and lock-protected publications."""

    @staticmethod
    def fields_for(descriptor: Descriptor, num_envs: int) -> list[tuple[str, tuple[int, ...], str]]:
        fields: list[tuple[str, tuple[int, ...], str]] = [
            ("publication", (2, 2), np.dtype(np.int64).str),
            ("actions", (2, num_envs, *descriptor.action_shape), descriptor.action_dtype),
            ("reward", (2, num_envs), descriptor.reward_dtype),
            ("terminated", (2, num_envs), np.dtype(bool).str),
            ("truncated", (2, num_envs), np.dtype(bool).str),
            ("mask", (2, num_envs), np.dtype(bool).str),
            ("steps", (2, num_envs), descriptor.steps_dtype),
            ("indices", (2, num_envs), np.dtype(np.int64).str),
            ("lengths", (2, num_envs), np.dtype(np.int64).str),
        ]
        for group, dimension, dtype in descriptor.groups:
            for prefix in ("obs", "final", "reset"):
                fields.append((f"{prefix}/{group}", (2, num_envs, dimension), dtype))
        return fields

    def __init__(self, descriptor: Descriptor, num_envs: int, lock: Any, name: str | None = None):
        self.descriptor = descriptor
        self.num_envs = num_envs
        self.lock = lock
        fields = self.fields_for(descriptor, num_envs)
        size = transport_nbytes(descriptor, num_envs)
        if name is None:
            ensure_memory_budget(size)
        self.memory = SharedMemory(name=name, create=name is None, size=size if name is None else 0)
        self.arrays: dict[str, np.ndarray] = {}
        offset = 0
        try:
            if name is None:
                reserve_transport_memory(self.memory, size)
            for field, shape, dtype in fields:
                view = np.ndarray(shape, dtype=dtype, buffer=self.memory.buf, offset=offset)
                self.arrays[field] = view
                offset += ((view.nbytes + 63) // 64) * 64
            if name is None:
                self.arrays["publication"].fill(-1)
        except BaseException:
            self.close(unlink=name is None)
            raise

    def publish(self, direction: int, sequence_id: int, timeout: float) -> None:
        if not self.lock.acquire(timeout=timeout):
            raise TimeoutError("timed out acquiring SHM publication lock")
        try:
            self.arrays["publication"][direction] = (sequence_id, sequence_id % 2)
        finally:
            self.lock.release()

    def check_publication(self, direction: int, sequence_id: int, timeout: float) -> None:
        if not self.lock.acquire(timeout=timeout):
            raise TimeoutError("timed out acquiring SHM publication lock")
        try:
            publication = tuple(self.arrays["publication"][direction])
            if publication != (sequence_id, sequence_id % 2):
                raise ValueError(
                    f"stale SHM publication {publication}, expected sequence {sequence_id}"
                )
        finally:
            self.lock.release()

    def write_observations(self, obs: Any, prefix: str, slot: int, count: int) -> None:
        if not isinstance(obs, dict) or tuple(obs) != tuple(
            group for group, _, _ in self.descriptor.groups
        ):
            raise ValueError(f"{prefix}: observation key order changed")
        for group, dimension, dtype in self.descriptor.groups:
            value = array(obs[group], (count, dimension), f"{prefix}/{group}", dtype)
            np.copyto(self.arrays[f"{prefix}/{group}"][slot, :count], value)

    def write_state(
        self, state: EnvStateProtocol, sequence_id: int, operation: str
    ) -> dict[str, Any]:
        if not isinstance(state.info, dict):
            raise ValueError("state.info must be a dict")
        slot = sequence_id % 2
        self.write_observations(state.obs, "obs", slot, self.num_envs)
        for field in ("reward", "terminated", "truncated"):
            target = self.arrays[field][slot]
            np.copyto(target, array(getattr(state, field), target.shape, field, target.dtype.str))
        target = self.arrays["steps"][slot]
        steps = array(state.info.get("steps"), target.shape, "steps", target.dtype.str)
        if np.any(steps < 0):
            raise ValueError("steps must be non-negative")
        np.copyto(target, steps)
        final_obs = state.final_observation
        compat_final = state.info.get("final_observation")
        if final_obs is None:
            final_obs = compat_final
        elif compat_final is not None:
            self.write_observations(compat_final, "final", slot, self.num_envs)
            if any(
                not np.array_equal(final_obs[group], compat_final[group])
                for group, _, _ in self.descriptor.groups
            ):
                raise ValueError("final_observation disagrees with info/final_observation")
        done = state.terminated | state.truncated
        mask = state.info.get("_final_observation")
        if mask is None:
            mask = done if final_obs is not None else np.zeros(self.num_envs, dtype=bool)
        mask = array(mask, (self.num_envs,), "_final_observation", np.dtype(bool).str)
        if np.any(mask) and final_obs is None:
            raise ValueError("final-observation mask requires final_observation")
        if operation == "step" and (np.any(mask & ~done) or np.any(state.truncated & ~mask)):
            raise ValueError("final-observation mask must cover timeouts and only mark done envs")
        np.copyto(self.arrays["mask"][slot], mask)
        if final_obs is not None:
            self.write_observations(final_obs, "final", slot, self.num_envs)
        else:
            for group, _, _ in self.descriptor.groups:
                self.arrays[f"final/{group}"][slot].fill(0)
        return info_metadata(state.info)

    def close(self, *, unlink: bool = False) -> None:
        self.arrays.clear()
        try:
            self.memory.close()
        finally:
            if unlink:
                try:
                    self.memory.unlink()
                except FileNotFoundError:
                    pass


def transport_nbytes(descriptor: Descriptor, num_envs: int) -> int:
    return sum(
        ((int(np.prod(shape)) * np.dtype(dtype).itemsize + 63) // 64) * 64
        for _, shape, dtype in SharedArrays.fields_for(descriptor, num_envs)
    )


def ensure_memory_budget(size: int) -> None:
    if sys.platform == "linux" and size > shutil.disk_usage("/dev/shm").free:
        raise MemoryError(f"insufficient /dev/shm space for source transport ({size} bytes)")
