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
        time.sleep(self.override.get("step_delay", 0))
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


class ProbePhysicsEnv(PhysicsEnv):
    probing = False

    def reset_probe(self, seed):
        self.probing = True
        obs, info = self.reset(np.arange(self.num_envs))
        info["probe_seed"] = seed
        return obs, info

    def step(self, actions):
        state = super().step(actions)
        if self.probing:
            state.info["probe"] = {"error": np.abs(self.value).astype(np.float32)}
        return state

    def finish_probe(self):
        self.probing = False
        if self.mode == "bad_finish":
            return True


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


def test_source_timing_maps_readiness_to_common_barrier(monkeypatch):
    with SynchronousEnv(specs()) as env:
        env.init_state()
        receive = synchronous_env._timed_receive

        def measured(conn, request):
            response, _, _ = receive(conn, request)
            source = int(response[0])
            return response, 100.0, 100.0 + (source + 1) * 0.1

        monkeypatch.setattr(synchronous_env, "_timed_receive", measured)
        state = env.step(np.zeros((10, 1), np.float32))
        timings = state.info["source_timings"]
        assert list(timings) == list("0123")
        for index, timing in enumerate(timings.values()):
            assert timing["request_response_seconds"] == pytest.approx((index + 1) * 0.1)
            assert timing["barrier_wait_seconds"] == pytest.approx((3 - index) * 0.1)
            assert all(
                type(value) is float and np.isfinite(value) and value >= 0
                for value in timing.values()
            )
        assert all("env_step_seconds" not in info for info in state.info["source_logs"].values())


def test_worker_timing_includes_source_step_delay():
    with SynchronousEnv(specs(1, step_delay=0.05)) as env:
        env.init_state()
        timing = env.step(np.zeros((1, 1), np.float32)).info["source_timings"]["0"]
        assert 0.05 <= timing["env_step_seconds"] <= timing["request_response_seconds"]
        assert timing["barrier_wait_seconds"] == 0


def test_failed_exchange_does_not_publish_partial_timing(monkeypatch):
    env = SynchronousEnv(specs(2))
    initial = env.init_state()
    receive = synchronous_env._receive

    def corrupt(conn, request):
        source, seq, status, value = receive(conn, request)
        return source, seq - int(source == "1"), status, value

    monkeypatch.setattr(synchronous_env, "_receive", corrupt)
    with pytest.raises(RuntimeError, match="stale"):
        env.step(np.zeros((3, 1), np.float32))
    assert env.state is initial and "source_timings" not in initial.info


@pytest.mark.parametrize("value", [float("nan"), -1.0, None])
def test_invalid_worker_timing_poisons_exchange(monkeypatch, value):
    env = SynchronousEnv(specs(1))
    initial, receive = env.init_state(), synchronous_env._receive

    def corrupt(conn, request):
        response = receive(conn, request)
        response[3].timing["env_step_seconds"] = value
        return response

    monkeypatch.setattr(synchronous_env, "_receive", corrupt)
    with pytest.raises(ValueError, match="timing"):
        env.step(np.zeros((1, 1), np.float32))
    assert env.state is initial and "source_timings" not in initial.info
    assert all(conn.closed for conn in env._connections)


@pytest.mark.parametrize("mode", ["groups", "shape", "dtype", "nan", "flags", "final"])
def test_malformed_source_poisons_service(mode):
    env = SynchronousEnv(specs(1, mode=mode))
    initial = env.init_state()
    with pytest.raises(ValueError):
        env.step(np.zeros((1, 1), np.float32))
    assert env.state is initial and "source_timings" not in initial.info
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


def test_selected_sources_do_not_advance_inactive_pools_and_preserve_full_steps():
    with SynchronousEnv(specs()) as env:
        initial = env.init_state().obs["obs"].copy()
        selected = {"1": np.ones((2, 1), np.float32), "3": np.ones((4, 1), np.float32)}
        states = env.step_selected(selected)
        assert tuple(states) == ("1", "3")
        assert env.state.info["active_sources"] == ("1", "3")
        assert tuple(env.state.info["source_timings"]) == ("1", "3")
        inactive = np.r_[0, 3:6]
        np.testing.assert_array_equal(env.state.obs["obs"][inactive], initial[inactive])
        assert not env.state.reward[inactive].any()
        for source in states:
            assert (states[source].obs["obs"][:, 1] == 1).all()
        env.step_selected({"3": selected["3"]})
        np.testing.assert_array_equal(env.state.obs["obs"][:, 1], [0, 1, 1, 0, 0, 0, 2, 2, 2, 2])
        state = env.step(np.zeros((10, 1), np.float32))
        np.testing.assert_array_equal(state.obs["obs"][:, 1], [1, 2, 2, 1, 1, 1, 3, 3, 3, 3])
        assert env._sequences == [2, 3, 2, 4]
        env.reset(np.arange(10))
        assert not env.state.obs["obs"][:, 1].any()


@pytest.mark.parametrize("selection", [(), ("4",), ("3", "1")])
def test_invalid_selected_mapping_rejected_before_physics(selection):
    with SynchronousEnv(specs()) as env:
        initial = env.init_state()
        with pytest.raises(ValueError, match="ordered subset"):
            env.step_selected({key: np.zeros((int(key) + 1, 1), np.float32) for key in selection})
        assert env.state is initial and env._sequences == [1, 1, 1, 1]


@pytest.mark.parametrize("mode", ["groups", "shape", "dtype", "nan", "flags", "final"])
def test_selected_source_validation_is_strict_and_atomic(mode):
    sources = specs()
    sources[2] = SourceSpec("2", partial(PhysicsEnv, source=2), 3, {"mode": mode})
    env = SynchronousEnv(sources)
    initial = env.init_state()
    with pytest.raises(ValueError):
        env.step_selected({"2": np.zeros((3, 1), np.float32)})
    assert env.state is initial and all(conn.closed for conn in env._connections)


@pytest.mark.parametrize("mode", ["hang", "crash"])
def test_selected_failure_cleans_active_and_idle_workers(tmp_path, mode):
    sources = specs()
    marker = tmp_path / "selected-pids"
    sources[2] = SourceSpec(
        "2",
        partial(PhysicsEnv, source=2),
        3,
        {"mode": mode, "nested": True, "pids": str(marker)},
    )
    env = SynchronousEnv(sources)
    initial = env.init_state()
    pids = [state["pid"] for state in initial.info["source_logs"].values()]
    pids += [int(pid) for pid in marker.read_text().split()]
    env.timeout_s = 0.2
    with pytest.raises((TimeoutError, RuntimeError)):
        env.step_selected({"2": np.zeros((3, 1), np.float32)})
    assert all(not alive(pid) for pid in pids)
    assert all(conn.closed for conn in env._connections)


def test_selected_stale_response_rejects_without_publishing(monkeypatch):
    env = SynchronousEnv(specs())
    initial = env.init_state()
    original = synchronous_env._receive

    def stale(conn, request):
        source, seq, status, result = original(conn, request)
        return source, seq - 1, status, result

    monkeypatch.setattr(synchronous_env, "_receive", stale)
    with pytest.raises(RuntimeError, match="stale"):
        env.step_selected({"3": np.zeros((4, 1), np.float32)})
    assert env.state is initial and all(conn.closed for conn in env._connections)


def test_probe_protocol_uses_equal_seed_and_resets_to_new_training_episodes():
    sources = [SourceSpec(str(i), partial(ProbePhysicsEnv, source=i), 2) for i in range(4)]
    with SynchronousEnv(sources) as env:
        env.init_state()
        env.step_selected({"3": np.ones((2, 1), np.float32)})
        obs, info = env.reset_probe(19)
        assert not obs["obs"][:, 1].any()
        assert [value["probe_seed"] for value in info["source_logs"].values()] == [19] * 4
        with pytest.raises(ValueError, match="active probe"):
            env.reset_probe(19)
        state = env.step(np.ones((8, 1), np.float32))
        assert all("probe" in value for value in state.info["source_logs"].values())
        obs, _ = env.finish_probe()
        assert not obs["obs"][:, 1].any()
        state = env.step(np.zeros((8, 1), np.float32))
        assert all("probe" not in value for value in state.info["source_logs"].values())
        assert (state.obs["obs"][:, 1] == 1).all()
        with pytest.raises(ValueError, match="no active probe"):
            env.finish_probe()


def test_probe_missing_capability_closes_all_workers():
    env = SynchronousEnv(specs())
    env.init_state()
    with pytest.raises(RuntimeError, match="independent probe protocol"):
        env.reset_probe(1)
    assert all(conn.closed for conn in env._connections)


def test_invalid_probe_finish_cannot_resume_training():
    env = SynchronousEnv([SourceSpec("0", ProbePhysicsEnv, 2, {"mode": "bad_finish"})])
    env.reset_probe(1)
    with pytest.raises(ValueError, match="finish_probe must return None"):
        env.finish_probe()
    assert all(conn.closed for conn in env._connections)
