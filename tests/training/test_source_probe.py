"""Task probe lifecycle, pre-reset telemetry, and preservation of training state."""

import pickle
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base.np_env import NpEnvState
from unilab.training.source_probe import SourceProbeEnv, probe_env_factory


class FakeEnv:
    def __init__(self):
        self.num_envs = 2
        self.cfg = SimpleNamespace(
            ctrl_dt=0.02,
            auto_reset=True,
            events={},
            curriculum={},
            observations={},
            seed=7,
        )
        params = SimpleNamespace(
            sampling_mode="start",
            pose_range={"x": (0, 0)},
            velocity_range={},
            joint_position_range=(0, 0),
            joint_default_position_range=(0, 0),
        )
        self.reference = np.array([[[0, 0, 1]], [[10, 20, 1]]], dtype=np.float32)
        self.motion = SimpleNamespace(
            cfg=SimpleNamespace(params=params),
            motion=SimpleNamespace(num_clips=1, num_frames=225, fps=50),
            time_steps=np.zeros(2, dtype=np.int32),
            body_pos_w=self.reference.copy(),
            robot_body_pos_w=self.reference.copy(),
        )
        self.command_manager = SimpleNamespace(get_term=lambda _: self.motion)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.counter = 30
        self.closed = False
        self.autoreset = True
        self.reset_calls = 0
        self.state = NpEnvState(
            {"obs": np.zeros((2, 3), np.float32), "critic": np.zeros((2, 4), np.float32)},
            np.zeros(2, np.float32),
            np.zeros(2, bool),
            np.zeros(2, bool),
            {"log": {}},
        )
        self.done_flags = np.array([True, False])
        self.bad_error = False
        self.bad_reward = False
        self.bad_observation = False
        self.bad_reset = False

    def set_autoreset(self, enabled):
        self.autoreset = enabled

    def export_training_state(self):
        return {"counter": self.counter}

    def import_training_state(self, state):
        self.counter = state["counter"]

    def reset(self, ids=None, *, seed=None):
        if seed is not None:
            self.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
            self.cfg.seed = seed
        ids = np.arange(2) if ids is None else ids
        self.reset_calls += 1
        self.rng.random()
        self.motion.time_steps[ids] = int(self.bad_reset)
        self.motion.robot_body_pos_w[ids] = self.reference[ids]
        for value in self.state.obs.values():
            value[ids] = 0
        self.state.reward[ids] = 0
        self.state.terminated[ids] = self.state.truncated[ids] = False
        self.state.info["log"] = {"reset": True}
        return {key: value[ids].copy() for key, value in self.state.obs.items()}, {}

    def step(self, actions):
        self.counter += 1
        self.motion.time_steps += 1
        self.motion.robot_body_pos_w[:, :, 0] += 0.1
        self.state.reward[:] = [1, 2]
        self.state.terminated[:] = self.state.truncated[:] = self.done_flags
        for value in self.state.obs.values():
            value[:] += 11
        self.state.info["log"] = {"reward/test": 1.5}
        if self.bad_error:
            self.motion.robot_body_pos_w[:] = np.nan
        if self.bad_reward:
            self.state.reward[:] = np.inf
        if self.bad_observation:
            self.state.obs["critic"][:] = np.nan
        return self.state

    def close(self):
        self.closed = True


def test_training_is_exact_passthrough():
    env = FakeEnv()
    wrapper = SourceProbeEnv(env)
    state = wrapper.step(np.zeros((2, 29), np.float32))
    assert state is env.state and wrapper.state is state
    assert "probe" not in state.info
    assert env.autoreset and env.reset_calls == 0


def test_probe_retains_terminal_error_final_obs_rewards_and_dual_flags_before_reset():
    env = FakeEnv()
    wrapper = SourceProbeEnv(env)
    wrapper.reset_probe(1)
    state = wrapper.step(np.zeros((2, 29), np.float32))
    np.testing.assert_allclose(state.info["probe"]["error"], [0.1, 0.1], atol=1e-6)
    assert state.info["probe"]["error"].dtype == np.float32
    np.testing.assert_array_equal(state.reward, [1, 2])
    np.testing.assert_array_equal(state.terminated, [True, False])
    np.testing.assert_array_equal(state.truncated, [True, False])
    assert np.all(state.final_observation["obs"] == 11)
    assert np.all(state.obs["obs"][0] == 0)
    assert np.all(state.obs["obs"][1] == 11)
    assert state.info["log"] == {"reward/test": 1.5}
    assert not env.state.terminated.any()
    assert state is wrapper.state
    # Copies survive later steps and manual resets.
    wrapper.step(np.zeros((2, 29), np.float32))
    np.testing.assert_array_equal(state.reward, [1, 2])
    assert np.all(state.final_observation["obs"] == 11)


def test_finish_restores_rng_counters_and_allows_fresh_training_episode():
    env = FakeEnv()
    wrapper = SourceProbeEnv(env)
    saved_rng = deepcopy(env.rng.bit_generator.state)
    saved_np, saved_py = np.random.get_state(), random.getstate()
    wrapper.reset_probe(123)
    for _ in range(3):
        wrapper.step(np.zeros((2, 29), np.float32))
    np.random.random()
    random.random()
    wrapper.finish_probe()
    assert env.autoreset and env.cfg.seed == 7 and env.counter == 30
    assert env.rng.bit_generator.state == saved_rng
    assert pickle.dumps(np.random.get_state()) == pickle.dumps(saved_np)
    assert random.getstate() == saved_py
    # Runtime subsequently resets all training rows; probe data does not persist.
    wrapper.reset(np.arange(2))
    assert "probe" not in wrapper.state.info
    state = wrapper.step(np.zeros((2, 29), np.float32))
    assert state is env.state and env.counter == 31


def test_common_seed_reproduces_initial_observation_across_different_training_states():
    left, right = FakeEnv(), FakeEnv()
    right.rng.random(17)
    right.motion.robot_body_pos_w[:] += 100
    a, b = SourceProbeEnv(left), SourceProbeEnv(right)
    oa, _ = a.reset_probe(42)
    ob, _ = b.reset_probe(42)
    for key in oa:
        np.testing.assert_array_equal(oa[key], ob[key])
    assert left.rng.bit_generator.state == right.rng.bit_generator.state


def test_full_horizon_and_no_extra_reference_loop():
    env = FakeEnv()
    env.done_flags[:] = False
    wrapper = SourceProbeEnv(env)
    wrapper.reset_probe(1)
    for _ in range(224):
        state = wrapper.step(np.zeros((2, 29), np.float32))
        assert state.final_observation is None
    np.testing.assert_array_equal(env.motion.time_steps, [224, 224])
    with pytest.raises(RuntimeError, match="beyond"):
        wrapper.step(np.zeros((2, 29), np.float32))
    assert env.counter == 254


@pytest.mark.parametrize("field", ["bad_error", "bad_reward", "bad_observation"])
def test_nonfinite_probe_telemetry_is_rejected(field):
    env = FakeEnv()
    wrapper = SourceProbeEnv(env)
    wrapper.reset_probe(1)
    setattr(env, field, True)
    with pytest.raises(ValueError, match="non-finite"):
        wrapper.step(np.zeros((2, 29), np.float32))


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_bad_probe_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="integer seed"):
        SourceProbeEnv(FakeEnv()).reset_probe(seed)


def test_invalid_lifecycle_and_failed_reset_restore_mode():
    env = FakeEnv()
    wrapper = SourceProbeEnv(env)
    with pytest.raises(RuntimeError, match="No probe"):
        wrapper.finish_probe()
    env.bad_reset = True
    with pytest.raises(ValueError, match="frame zero"):
        wrapper.reset_probe(1)
    assert env.autoreset and env.cfg.seed == 7
    env.bad_reset = False
    wrapper.reset_probe(1)
    with pytest.raises(ValueError, match="inactive"):
        wrapper.reset_probe(1)


@pytest.mark.parametrize(
    "path,value",
    [
        ("cfg.ctrl_dt", 0.03),
        ("cfg.auto_reset", False),
        ("cfg.events", {"push": object()}),
        ("cfg.curriculum", {"schedule": object()}),
        ("cfg.observations", {"actor": SimpleNamespace(enable_corruption=True)}),
        ("motion.cfg.params.sampling_mode", "adaptive"),
        ("motion.cfg.params.pose_range", {"pitch": (-0.1, 0.1)}),
        ("motion.cfg.params.velocity_range", {"x": (-1, 1)}),
        ("motion.cfg.params.joint_position_range", (-0.1, 0.1)),
        ("motion.cfg.params.joint_default_position_range", (0, 0.1)),
        ("motion.motion.num_clips", 2),
        ("motion.motion.num_frames", 224),
        ("motion.motion.fps", 25),
    ],
)
def test_unsupported_probe_configuration_fails_closed(path, value):
    env = FakeEnv()
    parent = env
    for name in path.split(".")[:-1]:
        parent = getattr(parent, name)
    setattr(parent, path.split(".")[-1], value)
    with pytest.raises(ValueError, match="nominal"):
        SourceProbeEnv(env)


def test_factory_is_picklable_excludes_holdout_and_closes_invalid_env(monkeypatch):
    with pytest.raises(ValueError, match="four training"):
        probe_env_factory("mujoco")
    env = FakeEnv()
    env.cfg.auto_reset = False
    calls = []

    def registry(*args):
        calls.append(args)
        return env

    monkeypatch.setattr("unilab.training.source_probe.make_registry_env", registry)
    factory = pickle.loads(pickle.dumps(probe_env_factory("motrix")))
    with pytest.raises(ValueError, match="nominal"):
        factory(1024, {"seed": 99})
    assert calls == [("G1FlipTracking", "motrix", 1024, {"seed": 99})]
    assert env.closed
