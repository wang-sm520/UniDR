"""CPU-only integration tests using real spawn processes, pipes and shared memory."""

from __future__ import annotations

import os
import signal
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from multi_source_fakes import corrupting_worker, make_deterministic_env

from uni_rl.env_contract import EnvProtocol, SupportsEpisodeLengthBufferProtocol
from uni_rl.ipc.multi_source_env import (
    EnvSourceSpec,
    MultiSourceEnvError,
    MultiSourceOptions,
    make_multi_source_env,
)
from uni_rl.utils.nan_guard import NanGuard, NanGuardCfg

OPTIONS = MultiSourceOptions(startup_timeout_s=15, operation_timeout_s=4, shutdown_timeout_s=0.5)


def source(name: str, count: int = 2, **config: Any) -> EnvSourceSpec:
    return EnvSourceSpec(name, make_deterministic_env, count, config, seed=17)


def resources(env: Any) -> tuple[list[Any], list[str]]:
    return [item.process for item in env._sources], [
        item.shared.memory.name for item in env._sources
    ]


def assert_released(processes: list[Any], names: list[str]) -> None:
    assert all(process._closed or not process.is_alive() for process in processes)
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


def assert_poisoned(env: Any, failure: MultiSourceEnvError) -> None:
    for operation in (
        env.init_state,
        lambda: env.step(np.zeros((2, 2), dtype=np.float32)),
        lambda: env.reset(np.array([0])),
        lambda: env.set_episode_length_buf(np.zeros(2, dtype=np.int64)),
        lambda: env.set_nan_guard(None),
        lambda: env.state,
        lambda: env.cfg,
        lambda: env.obs_groups_spec,
        lambda: env.source_statistics,
    ):
        with pytest.raises(MultiSourceEnvError) as caught:
            operation()
        assert caught.value is failure
    env.close()
    env.close()


def test_public_contract_fixed_slices_and_parallel_barrier(tmp_path: Path) -> None:
    env: Any = make_multi_source_env(
        [
            source("first", 1, marker=10, delay=0.12, rendezvous=str(tmp_path), source_count=3),
            source("second", 2, marker=20, rendezvous=str(tmp_path), source_count=3),
            source("third", 3, marker=30, delay=0.04, rendezvous=str(tmp_path), source_count=3),
        ],
        options=OPTIONS,
    )
    processes, names = resources(env)
    try:
        assert isinstance(env, EnvProtocol)
        assert isinstance(env, SupportsEpisodeLengthBufferProtocol)
        assert env.num_envs == 6
        assert env.state is None
        assert env.obs_groups_spec == {"obs": 3, "critic": 4}
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (7,)
        assert env.cfg.ctrl_dt == 0.02
        assert env.cfg.max_episode_seconds == 0.14
        assert env.algo_capabilities.joint_names == ("joint_a", "joint_b")
        np.testing.assert_array_equal(env.algo_capabilities.action_low, [-1, -1])
        assert not env.play_capabilities.supports_physics_state_playback
        assert all(not process.daemon and process.pid != os.getpid() for process in processes)
        initial = env.init_state()
        assert env.init_state() is initial
        expected_ids = np.array([10, 20, 21, 30, 31, 32])
        np.testing.assert_array_equal(initial.obs["obs"][:, 0], expected_ids)
        actions = np.arange(12, dtype=np.float32).reshape(6, 2)
        state = env.step(actions)
        np.testing.assert_array_equal(state.obs["obs"][:, 0], expected_ids)
        np.testing.assert_array_equal(state.obs["obs"][:, 2], actions[:, 0])
        np.testing.assert_array_equal(state.reward, [11, 23, 25, 37, 39, 41])
        assert state.info["log"]["source/first/reward/raw"] == 10
        assert state.info["log"]["source/third/reward/raw"] == 30
        assert state.info["log"]["source/second/init_calls"] == 1
        assert state.info["source/second/timing"] == {"local": 20.0}
        saved = state.obs["obs"].copy()
        for _ in range(3):
            env.step(actions + 1)
        np.testing.assert_array_equal(state.obs["obs"], saved)
        np.testing.assert_array_equal(initial.obs["obs"][:, 1], 0)
    finally:
        env.close()
    assert_released(processes, names)


@pytest.mark.parametrize("low,high", [(-np.inf, np.inf), (-np.inf, 1), (-1, np.inf)])
@pytest.mark.parametrize("space_bounds_only", [False, True])
def test_unbounded_action_spaces_preserve_bounds_and_finite_actions(
    low: float, high: float, space_bounds_only: bool
) -> None:
    env: Any = make_multi_source_env(
        [
            source(name, low=low, high=high, space_bounds_only=space_bounds_only)
            for name in ("first", "second")
        ],
        options=OPTIONS,
    )
    try:
        np.testing.assert_array_equal(env.action_space.low, [low, low])
        np.testing.assert_array_equal(env.action_space.high, [high, high])
        np.testing.assert_array_equal(env.algo_capabilities.action_low, [low, low])
        np.testing.assert_array_equal(env.algo_capabilities.action_high, [high, high])
        for invalid in (np.inf, -np.inf, np.nan):
            with pytest.raises(ValueError, match="non-finite"):
                env.step(np.full((4, 2), invalid, dtype=np.float32))
        state = env.step(np.zeros((4, 2), dtype=np.float32))
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


@pytest.mark.parametrize("low,high", [(np.nan, 1), (-1, np.nan), (np.inf, np.inf), (-1, -np.inf)])
def test_invalid_action_bounds_still_fail_startup(low: float, high: float) -> None:
    with pytest.raises(MultiSourceEnvError, match="action_"):
        make_multi_source_env([source("invalid", low=low, high=high)], options=OPTIONS)


def test_reset_maps_local_ids_and_restores_request_order() -> None:
    env: Any = make_multi_source_env(
        [source("one", 1, marker=10), source("two", 2, marker=20), source("three", 3, marker=30)],
        options=OPTIONS,
    )
    try:
        env.step(np.zeros((6, 2), dtype=np.float32))
        observations, info = env.reset(np.array([5, 0, 3, 2], dtype=np.int32))
        np.testing.assert_array_equal(observations["obs"][:, 0], [32, 10, 30, 21])
        np.testing.assert_array_equal(observations["obs"][:, 1], -1)
        np.testing.assert_array_equal(env.state.info["steps"], [0, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(info["steps"], [0, 0, 0, 0])
        assert info["log"]["source/three/reset_count"] == 2
        empty, empty_info = env.reset(np.array([], dtype=np.int64))
        assert empty["obs"].shape == (0, 3)
        assert empty_info["steps"].size == 0
        for indices in (
            np.array([1, 1]),
            np.array([-1]),
            np.array([6]),
            np.array([0.0]),
            np.array([[0]]),
            np.array([True]),
        ):
            with pytest.raises(ValueError):
                env.reset(indices)
        env.step(np.zeros((6, 2), dtype=np.float32))
    finally:
        env.close()


def test_final_observations_masks_steps_and_episode_length_setter() -> None:
    env: Any = make_multi_source_env(
        [source("short", 1, horizon=2), source("long", 2, horizon=5)], options=OPTIONS
    )
    try:
        env.init_state()
        env.set_episode_length_buf(np.array([1, 0, 4], dtype=np.int64))
        np.testing.assert_array_equal(env.state.info["steps"], [1, 0, 4])
        state = env.step(np.zeros((3, 2), dtype=np.float32))
        np.testing.assert_array_equal(state.truncated, [True, False, True])
        np.testing.assert_array_equal(state.info["_final_observation"], state.truncated)
        np.testing.assert_array_equal(state.info["steps"], [0, 1, 0])
        assert state.final_observation is state.info["final_observation"]
        np.testing.assert_array_equal(state.final_observation["obs"][[0, 2], 1], 1)
        np.testing.assert_array_equal(state.obs["obs"][[0, 2], 1], 0)
        next_state = env.step(np.zeros((3, 2), dtype=np.float32))
        assert next_state.final_observation is None
        assert not next_state.info["_final_observation"].any()
    finally:
        env.close()


def test_guard_is_reconstructed_with_local_count_capability_and_path(tmp_path: Path) -> None:
    env: Any = make_multi_source_env(
        [source("alpha", 1, playback=True), source("beta", 3)], options=OPTIONS
    )
    try:
        guard = NanGuard(NanGuardCfg(enabled=True, output_dir=str(tmp_path)), 4, False)
        guard.capture(np.ones((4, 100)))
        env.set_nan_guard(guard)
        assert env.state is None
        state = env.step(np.zeros((4, 2), dtype=np.float32))
        assert state.info["source/alpha/guard"] == {
            "output_dir": str(tmp_path / "source" / "alpha"),
            "num_envs": 1,
            "playback": True,
            "buffer_count": 0,
        }
        assert state.info["source/beta/guard"]["num_envs"] == 3
        assert not state.info["source/beta/guard"]["playback"]
        assert guard.cfg.output_dir == str(tmp_path)
        env.set_nan_guard(None)
    finally:
        env.close()


@pytest.mark.parametrize(
    "difference,match",
    [
        ({"reverse_keys": True}, "groups"),
        ({"dtype": "float64"}, "groups"),
        ({"dimension": 4}, "groups"),
        ({"high": 2}, "action_high"),
        ({"joints": ("joint_b", "joint_a")}, "joint_names"),
        ({"ctrl_dt": 0.01}, "ctrl_dt"),
        ({"max_episode_seconds": 1}, "max_episode_seconds"),
        ({"init_nonfinite": True}, "non-finite"),
    ],
)
def test_descriptor_mismatches_fail_before_training(difference: dict[str, Any], match: str) -> None:
    with pytest.raises(MultiSourceEnvError, match=match) as caught:
        make_multi_source_env([source("good"), source("bad", **difference)], options=OPTIONS)
    assert caught.value.source_name == "bad"
    assert caught.value.operation == "startup"


@pytest.mark.parametrize(
    "fault",
    [
        "raise",
        "crash",
        "hang",
        "nan_obs",
        "inf_reward",
        "dtype",
        "shape",
        "keys",
        "flags",
        "steps",
        "metadata",
        "log",
        "missing_final",
        "nan_final",
        "mask",
    ],
)
def test_source_faults_poison_all_operations_and_unlink_shm(fault: str) -> None:
    env: Any = make_multi_source_env(
        [source("good", 1), source("bad", 1, fault=fault)], options=MultiSourceOptions(15, 0.4, 0.5)
    )
    processes, names = resources(env)
    initial = env.init_state()
    saved = initial.obs["obs"].copy()
    with pytest.raises(MultiSourceEnvError) as caught:
        env.step(np.zeros((2, 2), dtype=np.float32))
    assert caught.value.source_name == "bad"
    assert caught.value.operation == "step"
    assert caught.value.sequence_id == 1
    if fault == "raise":
        assert "controlled step failure" in caught.value.remote_traceback
        assert "Traceback" in caught.value.remote_traceback
    np.testing.assert_array_equal(initial.obs["obs"], saved)
    assert_poisoned(env, caught.value)
    assert_released(processes, names)


@pytest.mark.parametrize("fault", ["raise", "hang"])
def test_reset_failure_is_not_retried(fault: str) -> None:
    env: Any = make_multi_source_env(
        [source("bad", reset_fault=fault)], options=MultiSourceOptions(15, 0.3, 0.5)
    )
    processes, names = resources(env)
    with pytest.raises(MultiSourceEnvError) as caught:
        env.reset(np.array([1]))
    assert caught.value.operation == "reset"
    assert_poisoned(env, caught.value)
    assert_released(processes, names)


@pytest.mark.parametrize("fault", ["stale", "duplicate", "unpublished", "partial"])
def test_wire_sequence_and_publication_faults(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    monkeypatch.setattr("uni_rl.ipc.multi_source_env.run_source", corrupting_worker)
    env: Any = make_multi_source_env(
        [source("bad", 1, wire_fault=fault), source("slow", 1, wire_fault="none", delay=0.15)],
        options=MultiSourceOptions(15, 0.4, 0.5),
    )
    processes, names = resources(env)
    with pytest.raises(MultiSourceEnvError, match="stale|duplicate|timeout") as caught:
        env.step(np.zeros((2, 2), dtype=np.float32))
    assert caught.value.source_name == "bad"
    assert_poisoned(env, caught.value)
    assert_released(processes, names)


def process_running(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().split(") ", 1)[1][0] != "Z"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
@pytest.mark.parametrize(
    "mode",
    ["close", "close_hang", "crash", "native_crash", "hang", "startup_error", "startup_hang"],
)
def test_nested_process_groups_are_cleaned(tmp_path: Path, mode: str) -> None:
    pid_file = tmp_path / "worker.pid"
    child_file = tmp_path / "child.pid"
    shm_file = tmp_path / "source.shm"
    child_shm_file = tmp_path / "child.shm"
    options = MultiSourceOptions(8 if mode == "startup_hang" else 15, 0.3, 0.4)
    config: dict[str, Any] = {
        "pid_file": str(pid_file),
        "child_file": str(child_file),
        "shm_file": str(shm_file),
        "child_shm_file": str(child_shm_file),
    }
    if mode in {"crash", "native_crash", "hang"}:
        config["fault"] = mode
    elif mode != "close":
        config[mode] = True
    if mode.startswith("startup"):
        with pytest.raises(MultiSourceEnvError):
            make_multi_source_env([source("nested", **config)], options=options)
    else:
        env: Any = make_multi_source_env([source("nested", **config)], options=options)
        processes, names = resources(env)
        if mode in {"crash", "native_crash", "hang"}:
            with pytest.raises(MultiSourceEnvError):
                env.step(np.zeros((2, 2), dtype=np.float32))
        env.close()
        env.close()
        assert_released(processes, names)
    assert pid_file.exists() and child_file.exists()
    deadline = time.monotonic() + 3
    while process_running(int(child_file.read_text())) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not process_running(int(pid_file.read_text()))
    assert not process_running(int(child_file.read_text()))
    assert_released([], [shm_file.read_text(), child_shm_file.read_text()])


def test_source_tracker_never_unlinks_unrelated_learner_shm() -> None:
    unrelated = SharedMemory(create=True, size=64)
    unrelated.buf[0] = 37
    try:
        env: Any = make_multi_source_env([source("crash", fault="crash")], options=OPTIONS)
        trackers = [item.resources.process for item in env._sources]
        with pytest.raises(MultiSourceEnvError):
            env.step(np.zeros((2, 2), dtype=np.float32))
        env.close()
        assert all(tracker.poll() is not None for tracker in trackers)
        attached = SharedMemory(name=unrelated.name)
        try:
            assert attached.buf[0] == 37
        finally:
            attached.close()
    finally:
        unrelated.close()
        unrelated.unlink()


def test_bad_arguments_are_rejected_before_spawning() -> None:
    for sources in (
        [],
        [source("same"), source("same")],
        [source("../escape")],
        [source("empty", 0)],
        [EnvSourceSpec("lambda", lambda count, cfg: None, 1)],
    ):
        with pytest.raises(ValueError):
            make_multi_source_env(sources, options=OPTIONS)
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            make_multi_source_env(
                [source("valid")], options=MultiSourceOptions(operation_timeout_s=value)
            )


def test_actions_are_finite_and_shape_checked_without_dispatch() -> None:
    env: Any = make_multi_source_env([source("valid")], options=OPTIONS)
    try:
        for actions in (
            np.zeros((1, 2)),
            np.ones((2, 2), dtype=np.int32),
            np.full((2, 2), np.nan),
            np.full((2, 2), np.inf),
            np.full((2, 2), 1e300),
        ):
            with pytest.raises(ValueError):
                env.step(actions)
        assert env.state is None
        assert (
            env.step(np.zeros((2, 2), dtype=np.float32)).info["log"]["source/valid/steps_taken"]
            == 1
        )
    finally:
        env.close()


def test_source_seed_is_reproducible() -> None:
    snapshots = []
    for _ in range(2):
        env = make_multi_source_env([source("repeatable", 1)], options=OPTIONS)
        try:
            snapshots.append(env.init_state().obs["obs"].copy())
        finally:
            env.close()
    np.testing.assert_array_equal(*snapshots)


def test_total_transport_budget_is_checked_before_any_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import uni_rl.ipc.multi_source_env as runtime
    from uni_rl.ipc._multi_source_shared import transport_nbytes

    env: Any = make_multi_source_env([source("probe")], options=OPTIONS)
    required_per_source = transport_nbytes(env._descriptor, 2)
    env.close()

    def unexpected_allocation(*args: Any) -> Any:
        raise AssertionError("transport was allocated before checking the complete budget")

    monkeypatch.setattr(
        "uni_rl.ipc._multi_source_shared.shutil.disk_usage",
        lambda _: SimpleNamespace(free=required_per_source + 1),
    )
    monkeypatch.setattr(runtime, "SharedArrays", unexpected_allocation)
    with pytest.raises(MultiSourceEnvError, match="insufficient /dev/shm"):
        make_multi_source_env([source("first"), source("second")], options=OPTIONS)


def test_source_statistics_are_exact_immutable_barrier_snapshots() -> None:
    env: Any = make_multi_source_env(
        [source("one", 1, delay=0.03), source("two", 2)], options=OPTIONS
    )
    try:
        initial = env.source_statistics
        assert initial["one"]["step_calls"] == 0
        assert initial["two"]["transitions"] == 0
        for _ in range(3):
            env.step(np.zeros((3, 2), dtype=np.float32))
        env.reset(np.array([2]))
        current = env.source_statistics
        assert initial["one"]["step_calls"] == 0
        assert current["one"]["num_envs"] == 1
        assert current["one"]["transitions"] == 3
        assert current["two"]["transitions"] == 6
        assert current["one"]["reset_calls"] == 0
        assert current["two"]["reset_calls"] == 1
        assert current["one"]["step_seconds"] >= 0.09
        assert current["one"]["last_step_seconds"] >= 0.03
        assert current["two"]["reset_seconds"] > 0
        assert current["one"]["pid"] != current["two"]["pid"]
        if Path("/proc/self/statm").exists():
            assert current["one"]["rss_bytes"] > 0
        with pytest.raises(TypeError):
            current["one"]["step_calls"] = 0
        with pytest.raises(TypeError):
            current["other"] = {}
    finally:
        env.close()


def test_episode_length_support_is_preflighted_for_every_source() -> None:
    env: Any = make_multi_source_env(
        [source("yes", 1), source("no", 1, unsupported_setter=True)], options=OPTIONS
    )
    processes, names = resources(env)
    with pytest.raises(MultiSourceEnvError, match="does not support") as caught:
        env.set_episode_length_buf(np.array([1, 2]))
    assert caught.value.source_name == "no"
    assert_poisoned(env, caught.value)
    assert_released(processes, names)


def test_idle_worker_death_poisons_cached_operations() -> None:
    env: Any = make_multi_source_env([source("idle")], options=OPTIONS)
    processes, names = resources(env)
    processes[0].kill()
    processes[0].join(3)
    with pytest.raises(MultiSourceEnvError) as caught:
        env.init_state()
    assert caught.value.source_name == "idle"
    assert_poisoned(env, caught.value)
    assert_released(processes, names)


def test_shm_allocation_failure_cleans_already_created_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uni_rl.ipc.multi_source_env as runtime

    original_shared = runtime.SharedArrays
    names = []

    def allocate(descriptor: Any, count: int, lock: Any) -> Any:
        if names:
            raise MemoryError("controlled allocation failure")
        shared = original_shared(descriptor, count, lock)
        names.append(shared.memory.name)
        return shared

    monkeypatch.setattr(runtime, "SharedArrays", allocate)
    with pytest.raises(MultiSourceEnvError, match="allocation failure"):
        make_multi_source_env([source("first"), source("second")], options=OPTIONS)
    assert names
    assert_released([], names)


def test_pipes_never_carry_bulk_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    import uni_rl.ipc.multi_source_env as runtime

    original_send = runtime.send_message
    original_receive = runtime.receive_message
    observed = []

    def inspect(value: Any) -> None:
        assert not isinstance(value, np.ndarray)
        if dataclasses.is_dataclass(value):
            inspect(dataclasses.asdict(value))
        elif isinstance(value, dict):
            for item in value.values():
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)

    def checked_send(connection: Any, message: dict[str, Any]) -> None:
        inspect(message)
        observed.append(message["operation"])
        original_send(connection, message)

    def checked_receive(connection: Any, **kwargs: Any) -> dict[str, Any]:
        message = original_receive(connection, **kwargs)
        inspect(message)
        return message

    monkeypatch.setattr(runtime, "send_message", checked_send)
    monkeypatch.setattr(runtime, "receive_message", checked_receive)
    env: Any = make_multi_source_env([source("only")], options=OPTIONS)
    try:
        env.step(np.zeros((2, 2), dtype=np.float32))
        env.reset(np.array([1, 0]))
        env.set_episode_length_buf(np.array([0, 1]))
        env.set_nan_guard(NanGuard(NanGuardCfg(), 2, False))
    finally:
        env.close()
    assert {"attach", "step", "reset", "set_episode_length_buf", "set_nan_guard", "close"} <= set(
        observed
    )
