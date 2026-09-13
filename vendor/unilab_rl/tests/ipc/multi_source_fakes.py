"""Deterministic, importable factories exercising real spawn and nested children."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import struct
import time
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from uni_rl.env_contract import EnvAlgoCapabilities
from uni_rl.ipc._multi_source_shared import receive_message, send_message


@dataclass
class FakeState:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any]
    final_observation: dict[str, np.ndarray] | None = None


def nested_child(pid_file: str, shm_file: str | None = None) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if shm_file is not None:
        memory = SharedMemory(create=True, size=128)
        Path(shm_file).write_text(memory.name)
    Path(pid_file).write_text(str(os.getpid()))
    while True:
        time.sleep(1)


class DeterministicEnv:
    def __init__(self, num_envs: int, override: Mapping[str, Any] | None):
        self.num_envs = num_envs
        self.override = dict(override or {})
        self.marker = self.override.get("marker", 10)
        self.dtype = np.dtype(self.override.get("dtype", "float32"))
        self.obs_groups_spec = {"obs": 3, "critic": 4}
        if self.override.get("reverse_keys"):
            self.obs_groups_spec = dict(reversed(tuple(self.obs_groups_spec.items())))
        if self.override.get("dimension"):
            self.obs_groups_spec["obs"] = self.override["dimension"]
        low = np.full(2, self.override.get("low", -1), dtype=np.float32)
        high = np.full(2, self.override.get("high", 1), dtype=np.float32)
        self.action_space = SimpleNamespace(
            shape=(2,), dtype=np.dtype("float32"), low=low, high=high
        )
        self.observation_space = SimpleNamespace(
            shape=(sum(self.obs_groups_spec.values()),), dtype=self.dtype
        )
        self.algo_capabilities = EnvAlgoCapabilities(
            None if self.override.get("space_bounds_only") else low,
            None if self.override.get("space_bounds_only") else high,
            tuple(self.override.get("joints", ("joint_a", "joint_b"))),
        )
        self.play_capabilities = SimpleNamespace(
            supports_physics_state_playback=self.override.get("playback", False)
        )
        self.cfg = SimpleNamespace(
            ctrl_dt=self.override.get("ctrl_dt", 0.02),
            max_episode_seconds=self.override.get("max_episode_seconds", 0.14),
        )
        self.state: FakeState | None = None
        self.step_count = 0
        self.init_count = 0
        self.random_value = float(np.random.random())
        self.guard = None
        self.child = None
        self.memory = None
        if self.override.get("pid_file"):
            Path(self.override["pid_file"]).write_text(str(os.getpid()))
        if self.override.get("child_file"):
            if self.override.get("shm_file"):
                self.memory = SharedMemory(create=True, size=256)
                Path(self.override["shm_file"]).write_text(self.memory.name)
            self.child = mp.get_context("spawn").Process(
                target=nested_child,
                args=(self.override["child_file"], self.override.get("child_shm_file")),
            )
            self.child.start()
            deadline = time.monotonic() + 10
            while not Path(self.override["child_file"]).exists() and time.monotonic() < deadline:
                time.sleep(0.01)
        if self.override.get("startup_error"):
            raise RuntimeError("controlled factory failure")
        if self.override.get("startup_hang"):
            time.sleep(60)
        if self.override.get("unsupported_setter"):
            delattr(self.__class__, "set_episode_length_buf")

    def _observations(self) -> dict[str, np.ndarray]:
        observations = {
            name: np.zeros((self.num_envs, dim), dtype=self.dtype)
            for name, dim in self.obs_groups_spec.items()
        }
        for values in observations.values():
            values[:, 0] = self.marker + np.arange(self.num_envs)
            values[:, 1] = self.step_count
            values[:, 2] = self.random_value
        return observations

    def init_state(self) -> FakeState:
        self.init_count += 1
        self.state = FakeState(
            self._observations(),
            np.zeros(self.num_envs, dtype=self.dtype),
            np.zeros(self.num_envs, dtype=bool),
            np.zeros(self.num_envs, dtype=bool),
            {"steps": np.zeros(self.num_envs, dtype=np.uint32), "log": {}},
        )
        if self.override.get("init_nonfinite"):
            self.state.obs["obs"][0, 0] = np.nan
        return self.state

    def step(self, actions: np.ndarray) -> FakeState:
        assert self.state is not None
        self.step_count += 1
        if self.override.get("rendezvous"):
            root = Path(self.override["rendezvous"])
            (root / f"{self.marker}-{self.step_count}").touch()
            deadline = time.monotonic() + 5
            while len(list(root.glob(f"*-{self.step_count}"))) != self.override["source_count"]:
                if time.monotonic() >= deadline:
                    raise RuntimeError("sources were not submitted together")
                time.sleep(0.01)
        time.sleep(self.override.get("delay", 0))
        fault = self.override.get("fault")
        if fault == "hang":
            time.sleep(60)
        if fault == "crash":
            os._exit(23)
        if fault == "native_crash":
            import resource

            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            os.kill(os.getpid(), signal.SIGSEGV)
        if fault == "raise":
            raise RuntimeError("controlled step failure")
        self.state.obs = self._observations()
        self.state.obs["obs"][:, 2] = actions[:, 0]
        self.state.reward = (self.marker + actions[:, 1]).astype(self.dtype)
        self.state.info["steps"] += 1
        horizon = self.override.get("horizon", 7)
        self.state.truncated = self.state.info["steps"] >= horizon
        self.state.terminated = np.zeros(self.num_envs, dtype=bool)
        final = {name: values.copy() for name, values in self.state.obs.items()}
        self.state.final_observation = final if self.state.truncated.any() else None
        self.state.info["_final_observation"] = self.state.truncated.copy()
        self.state.info.pop("final_observation", None)
        if self.state.final_observation is not None:
            self.state.info["final_observation"] = self.state.final_observation
        self.state.info["steps"][self.state.truncated] = 0
        for values in self.state.obs.values():
            values[self.state.truncated, 1] = 0
        self.state.info["log"] = {
            "reward/raw": float(self.marker),
            "steps_taken": self.step_count,
            "init_calls": self.init_count,
        }
        self.state.info["timing"] = {"local": float(self.marker)}
        if self.guard is not None:
            self.state.info["guard"] = {
                "output_dir": self.guard.cfg.output_dir,
                "num_envs": self.guard._num_envs,
                "playback": self.guard._supports_state_playback,
                "buffer_count": len(self.guard._buffer),
            }
        if fault == "nan_obs":
            self.state.obs["obs"][0, 0] = np.nan
        if fault == "inf_reward":
            self.state.reward[0] = np.inf
        if fault == "dtype":
            self.state.obs["obs"] = self.state.obs["obs"].astype(np.float64)
        if fault == "shape":
            self.state.obs["obs"] = self.state.obs["obs"][:, :2]
        if fault == "keys":
            self.state.obs = dict(reversed(tuple(self.state.obs.items())))
        if fault == "flags":
            self.state.terminated = self.state.terminated.astype(np.int32)
        if fault == "steps":
            self.state.info["steps"] = np.full(self.num_envs, np.nan)
        if fault == "metadata":
            self.state.info["other_array"] = np.zeros((self.num_envs, 2))
        if fault == "log":
            self.state.info["log"]["invalid"] = np.nan
        if fault == "missing_final":
            self.state.truncated[:] = True
            self.state.final_observation = None
        if fault in {"nan_final", "mask"}:
            self.state.truncated[:] = True
            self.state.final_observation = final
            self.state.info["_final_observation"][:] = True
            if fault == "nan_final":
                final["obs"][0, 0] = np.nan
            else:
                self.state.info["_final_observation"] = np.ones(self.num_envs, dtype=np.int32)
        return self.state

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        assert self.state is not None
        if self.override.get("reset_fault") == "raise":
            raise RuntimeError("controlled reset failure")
        if self.override.get("reset_fault") == "hang":
            time.sleep(60)
        self.state.info["steps"][env_indices] = 0
        for values in self.state.obs.values():
            values[env_indices, 1] = -1
        self.state.terminated[env_indices] = False
        self.state.truncated[env_indices] = False
        self.state.final_observation = None
        self.state.info.pop("final_observation", None)
        self.state.info["_final_observation"] = np.zeros(self.num_envs, dtype=bool)
        return {name: values[env_indices].copy() for name, values in self.state.obs.items()}, {
            "log": {"reset_count": len(env_indices)}
        }

    def set_episode_length_buf(self, values: np.ndarray) -> None:
        assert self.state is not None
        self.state.info["steps"][:] = values

    def set_nan_guard(self, guard: Any) -> None:
        self.guard = guard

    def close(self) -> None:
        if self.override.get("close_file"):
            Path(self.override["close_file"]).touch()
        if self.override.get("close_hang"):
            time.sleep(60)
        if self.memory is not None:
            self.memory.close()
            self.memory.unlink()


def make_deterministic_env(
    num_envs: int, env_cfg_override: Mapping[str, Any] | None = None
) -> DeterministicEnv:
    return DeterministicEnv(num_envs, env_cfg_override)


def corrupting_worker(
    source: Any, options: Any, connection: Any, lock: Any, tracker_connection: Any
) -> None:
    """Inject wire faults while delegating execution to the actual source worker."""
    import uni_rl.ipc._multi_source_worker as worker

    def corrupted_send(target: Any, message: dict[str, Any]) -> None:
        if message.get("operation") == "step" and message.get("kind") == "ack":
            fault = source.env_cfg_override["wire_fault"]
            if fault == "stale":
                message = {**message, "sequence_id": message["sequence_id"] - 1}
            if fault == "duplicate":
                send_message(target, message)
            if fault == "partial":
                os.write(target.fileno(), struct.pack("!i", 128) + b"partial")
                time.sleep(60)
            if fault == "unpublished":
                from uni_rl.ipc._multi_source_resources import exclude_transport_attachment

                memory = SharedMemory(name=shared_name[0])
                exclude_transport_attachment(memory)
                publication = np.ndarray((2, 2), dtype=np.int64, buffer=memory.buf)
                with lock:
                    publication[1] = (-1, -1)
                del publication
                memory.close()
        send_message(target, message)

    shared_name: list[str] = []

    def capture_receive(target: Any) -> dict[str, Any]:
        message = receive_message(target)
        if message.get("operation") == "attach":
            shared_name.append(message["shm_name"])
        return message

    worker.send_message = corrupted_send
    worker.receive_message = capture_receive
    worker.run_source(source, options, connection, lock, tracker_connection)
