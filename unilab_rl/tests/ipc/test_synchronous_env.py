from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import time
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uni_rl.env_contract import EnvProtocol
from uni_rl.ipc import synchronous_env
from uni_rl.ipc.synchronous_env import SourceSpec, SynchronousEnv


class PhysicsEnv:
    obs_groups_spec = {"obs": 2, "critic": 3}
    observation_space = SimpleNamespace(shape=(2,))
    action_space = SimpleNamespace(shape=(1,))
    cfg = SimpleNamespace(max_episode_seconds=10.0, ctrl_dt=0.1)
    play_capabilities = SimpleNamespace(supports_physics_state_playback=False)

    def __init__(self, count, override, source=0):
        self.num_envs, self.source = count, source
        self.override = override or {}
        self.mode = self.override.get("mode")
        self.value = np.full(count, source, np.float32)
        self.age = np.zeros(count, np.int64)
        self.rng_sample = float(np.random.random())
        self.binding = os.environ.get("SYNC_BINDING")
        self.state = None
        if self.override.get("shared_memory"):
            self.shared_memory = SharedMemory(create=True, size=8)
            Path(self.override["shared_memory"]).write_text(self.shared_memory.name)
        if self.override.get("nested"):
            self.nested = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            Path(self.override["pids"]).write_text(f"{os.getpid()} {self.nested.pid}")
        if self.mode == "constructor":
            raise RuntimeError("constructor failed")
        if self.mode == "startup_hang":
            time.sleep(60)
        if self.mode == "metadata":
            self.cfg = SimpleNamespace(max_episode_seconds=10.0, ctrl_dt=0.2)

    def observations(self):
        obs = np.column_stack((self.value, self.age)).astype(np.float32)
        return {"obs": obs, "critic": np.column_stack((obs, self.value + 10)).astype(np.float32)}

    def init_state(self):
        self.state = SimpleNamespace(
            obs=self.observations(),
            reward=self.value.copy(),
            terminated=np.zeros(self.num_envs, bool),
            truncated=np.zeros(self.num_envs, bool),
            final_observation=None,
            info={
                "pid": os.getpid(),
                "binding": self.binding,
                "rng": self.rng_sample,
            },
        )
        if self.override.get("initial_done"):
            self.state.terminated[:] = True
        return self.state

    def step(self, actions):
        if self.override.get("barrier"):
            barrier = Path(self.override["barrier"])
            (barrier / str(self.source)).touch()
            deadline = time.monotonic() + 3
            while len(list(barrier.iterdir())) != 4:
                if time.monotonic() > deadline:
                    raise RuntimeError("not all sources received actions before waiting")
                time.sleep(0.01)
        if self.mode == "hang":
            time.sleep(60)
        if self.mode == "crash":
            os._exit(7)
        self.value += actions[:, 0]
        self.age += 1
        final = self.observations()
        done = self.age >= 2
        self.value[done] = self.source
        state = self.init_state()
        state.terminated = done.copy()
        state.truncated = done.copy()
        state.final_observation = final
        if self.mode == "groups":
            state.obs["extra"] = state.obs["obs"]
        elif self.mode == "shape":
            state.obs["obs"] = state.obs["obs"][:, :1]
        elif self.mode == "dtype":
            state.obs["obs"] = state.obs["obs"].astype(np.float64)
        elif self.mode == "nan":
            state.reward[0] = np.nan
        elif self.mode == "flags":
            state.terminated = done.astype(np.int8)
        elif self.mode == "final":
            state.terminated[:] = True
            state.final_observation = None
        return state

    def reset(self, indices):
        self.value[indices] = self.source
        self.age[indices] = 0
        return {k: v[indices] for k, v in self.observations().items()}, {"reset": indices.tolist()}

    def set_episode_length_buf(self, values):
        self.age[:] = values

    def set_nan_guard(self, guard):
        pass

    def close(self):
        time.sleep(self.override.get("close_delay", 0))
        if self.override.get("shared_memory"):
            self.shared_memory.close()
            self.shared_memory.unlink()
        if self.override.get("closed"):
            Path(self.override["closed"]).write_text("closed")


class FlatObservationEnv(PhysicsEnv):
    obs_groups_spec = {"obs": 160, "critic": 286}
    observation_space = SimpleNamespace(shape=(446,))

    def observations(self):
        return {
            key: np.zeros((self.num_envs, dim), np.float32)
            for key, dim in self.obs_groups_spec.items()
        }


def specs(count=4, **override):
    return [
        SourceSpec(str(i), partial(PhysicsEnv, source=i), i + 1, override, {"SYNC_BINDING": str(i)})
        for i in range(count)
    ]


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split(")", 1)[1].split()[0] != "Z"


def test_four_source_order_actions_reset_final_observations_and_seed():
    with SynchronousEnv(specs(seed=17)) as env:
        assert isinstance(env, EnvProtocol)
        assert env.state is None and env.num_envs == 10
        # The ordinary PPO wrapper calls reset before reading state.
        obs, _ = env.reset(np.arange(10))
        expected = np.repeat(np.arange(4), np.arange(1, 5)).astype(np.float32)
        np.testing.assert_array_equal(obs["obs"][:, 0], expected)
        state = env.init_state()
        actions = np.arange(10, dtype=np.float32)[:, None]
        state = env.step(actions)
        pids = {value["pid"] for value in state.info["source_logs"].values()}
        assert len(pids) == 4 and os.getpid() not in pids
        assert [value["binding"] for value in state.info["source_logs"].values()] == list("0123")
        assert len({value["rng"] for value in state.info["source_logs"].values()}) == 1
        assert state.info["source_logs"]["0"]["rng"] == np.random.RandomState(17).random()
        np.testing.assert_array_equal(state.obs["obs"][:, 0], expected + actions[:, 0])
        state = env.step(actions)
        assert state.terminated.all() and state.truncated.all()
        np.testing.assert_array_equal(
            state.final_observation["obs"][:, 0], expected + 2 * actions[:, 0]
        )
        np.testing.assert_array_equal(state.obs["obs"][:, 0], expected)
        env.set_episode_length_buf(np.arange(10, dtype=np.int64))
        obs, info = env.reset(np.array([8, 0, 3]))
        np.testing.assert_array_equal(obs["obs"][:, 0], expected[[8, 0, 3]])
        assert info["source_logs"]["3"]["reset"] == [2]
        assert not env.state.terminated[[8, 0, 3]].any()
        assert env.source_slices["2"] == slice(3, 6)
    assert all(not alive(pid) for pid in pids)
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(actions)


def test_dispatches_all_sources_before_waiting_for_step(tmp_path):
    with SynchronousEnv(specs(barrier=str(tmp_path))) as env:
        env.init_state()
        state = env.step(np.ones((10, 1), np.float32))
        assert state.obs["obs"].shape == (10, 2)


@pytest.mark.parametrize("mode", ["groups", "shape", "dtype", "nan", "flags", "final"])
def test_malformed_source_poisons_service(mode):
    env = SynchronousEnv(specs(1, mode=mode))
    env.init_state()
    with pytest.raises(ValueError):
        env.step(np.zeros((1, 1), np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        env.init_state()


def test_flat_observation_space_includes_critic_dimensions():
    with SynchronousEnv([SourceSpec("flat", FlatObservationEnv, 2)]) as env:
        assert env.observation_space.shape == (446,)
        assert env.init_state().obs["obs"].shape == (2, 160)
        assert env.state.obs["critic"].shape == (2, 286)


def test_initialization_done_sentinels_do_not_require_terminal_observations():
    with SynchronousEnv(specs(1, initial_done=True)) as env:
        state = env.init_state()
        assert state.terminated.all() and state.final_observation is None
        env.reset(np.array([0]))
        assert not env.state.terminated.any()
        state = env.step(np.ones((1, 1), np.float32))
        assert not (state.terminated | state.truncated).any()
        state = env.step(np.ones((1, 1), np.float32))
        assert state.terminated.all() and state.final_observation is not None
        np.testing.assert_array_equal(state.final_observation["obs"][:, 0], [2])


@pytest.mark.parametrize("mode", ["hang", "crash"])
def test_failed_request_cleans_worker_and_nested_processes(tmp_path, mode):
    marker = tmp_path / "pids"
    env = SynchronousEnv(specs(1, mode=mode, nested=True, pids=str(marker)))
    env.init_state()
    pids = [int(pid) for pid in marker.read_text().split()]
    env.timeout_s = 0.2
    before = time.monotonic()
    with pytest.raises((TimeoutError, RuntimeError)):
        env.step(np.zeros((1, 1), np.float32))
    assert time.monotonic() - before < 5
    assert all(not alive(pid) for pid in pids)
    assert all(conn.closed for conn in env._connections)


def test_graceful_close_still_cleans_unclosed_nested_workers(tmp_path):
    marker, closed = tmp_path / "pids", tmp_path / "closed"
    env = SynchronousEnv(specs(1, nested=True, pids=str(marker), closed=str(closed)))
    pids = [int(pid) for pid in marker.read_text().split()]
    env.close()
    assert closed.read_text() == "closed"
    assert all(not alive(pid) for pid in pids)


def test_graceful_close_waits_for_shared_memory_release(tmp_path):
    marker, closed = tmp_path / "shared_memory", tmp_path / "closed"
    env = SynchronousEnv(specs(1, shared_memory=str(marker), close_delay=1.5, closed=str(closed)))
    allocation = Path("/dev/shm") / marker.read_text()
    assert allocation.exists()
    before = time.monotonic()
    env.close()
    assert 1.5 <= time.monotonic() - before < 5
    assert closed.read_text() == "closed"
    assert not allocation.exists()


def test_metadata_mismatch_and_constructor_failure_close_all(tmp_path):
    closed, marker = tmp_path / "closed", tmp_path / "shared_memory"
    for mode in ("metadata", "constructor"):
        sources = specs(2, closed=str(closed), close_delay=1.5, shared_memory=str(marker))
        sources[1] = SourceSpec("1", PhysicsEnv, 2, {"mode": mode})
        with pytest.raises((ValueError, RuntimeError)):
            SynchronousEnv(sources)
        assert closed.exists()
        assert not (Path("/dev/shm") / marker.read_text()).exists()
        closed.unlink()


def test_startup_timeout_closes_partial_construction(tmp_path):
    marker = tmp_path / "pids"
    with pytest.raises(TimeoutError, match="startup"):
        SynchronousEnv(
            specs(1, mode="startup_hang", nested=True, pids=str(marker)), startup_timeout_s=5
        )
    if marker.exists():
        assert all(not alive(int(pid)) for pid in marker.read_text().split())


def test_stale_source_response_closes_service(monkeypatch):
    env = SynchronousEnv(specs(1))
    original = synchronous_env._receive

    def stale(conn, request):
        source, seq, status, result = original(conn, request)
        return source, seq - 1, status, result

    monkeypatch.setattr(synchronous_env, "_receive", stale)
    with pytest.raises(RuntimeError, match="stale"):
        env.init_state()
    assert all(conn.closed for conn in env._connections)


def test_duplicate_response_rejected():
    parent, child = mp.Pipe()
    try:
        child.send(("a", 0, "ok", None))
        child.send(("a", 0, "ok", None))
        with pytest.raises(RuntimeError, match="duplicate"):
            synchronous_env._receive(parent, None)
    finally:
        parent.close()
        child.close()
