"""Diagnostic continuation preserves native failure evidence and physical motion."""

import os
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.managers.termination_manager import TerminationManager, TerminationTermCfg
from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand
from unilab.visualization.failure_tail import FailureTailTerminations, _HoldLastMotion


class _NativeFailure:
    def __init__(self, cfg, env):
        self.env = env

    def __call__(self, env):
        assert self.env is env, "native term must observe the live environment"
        return env.native_failure

    def reset(self, env_ids):
        self.env.term_resets += 1


def _timeout(env):
    return env.native_timeout


@pytest.fixture
def gate():
    terms = {
        "failure": TerminationTermCfg(func=_NativeFailure),
        "timeout": TerminationTermCfg(func=_timeout, time_out=True),
    }
    env = SimpleNamespace(
        num_envs=1,
        cfg=SimpleNamespace(terminations=terms),
        native_failure=np.zeros(1, bool),
        native_timeout=np.zeros(1, bool),
        term_resets=0,
    )
    return env, FailureTailTerminations(env, tail_steps=3)


@pytest.mark.parametrize("first", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("persistent", [False, True])
def test_native_evidence_and_exact_delayed_reset(gate, first, persistent):
    env, delayed = gate
    assert not delayed.compute()[0] and delayed.first_done is None
    assert not delayed.compute()[0] and delayed.steps_after_failure is None
    env.native_failure[0], env.native_timeout[0] = first
    assert not delayed.compute()[0]
    assert (delayed.native_terminated, delayed.native_truncated) == first
    assert delayed.get_term("failure")[0] == first[0]
    assert delayed.get_term("timeout")[0] == first[1]
    assert delayed.first_done == 2 and delayed.steps_after_failure == 0
    if not persistent:
        env.native_failure.fill(False)
        env.native_timeout.fill(False)
    for elapsed in (1, 2):
        assert not delayed.compute()[0]
        assert delayed.steps_after_failure == elapsed and delayed.first_done == 2
    assert delayed.compute()[0] and delayed.steps_after_failure == 3
    assert bool(delayed.terminated[0]) is first[0]
    assert bool(delayed.time_outs[0]) is (not first[0])
    assert (delayed.native_terminated, delayed.native_truncated) == (
        first if persistent else (False, False)
    )

    delayed.reset(np.array([0], np.int32))
    assert env.term_resets == 1
    assert delayed.tick == 0 and delayed.first_done is None
    assert delayed.steps_after_failure is None
    assert not delayed.native_terminated and not delayed.native_truncated
    assert not delayed.seen_termination
    env.native_failure.fill(False)
    env.native_timeout.fill(False)
    for _ in range(5):
        assert not delayed.compute()[0]
    assert delayed.first_done is None


def test_true_failure_during_timeout_tail_wins_without_restarting_delay(gate):
    env, delayed = gate
    env.native_timeout[0] = True
    assert not delayed.compute()[0]
    env.native_timeout[0], env.native_failure[0] = False, True
    assert not delayed.compute()[0]
    env.native_failure[0] = False
    assert not delayed.compute()[0]
    assert delayed.compute()[0]
    assert delayed.first_done == 0 and delayed.steps_after_failure == 3
    assert delayed.terminated[0] and not delayed.time_outs[0]


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_tail_rejected(gate, steps):
    env, _ = gate
    with pytest.raises(ValueError, match="positive steps"):
        FailureTailTerminations(env, tail_steps=steps)


@pytest.mark.parametrize("first_done", [None, 0, 250])
def test_clip_end_preserves_native_loop_until_failure_is_latched(gate, monkeypatch, first_done):
    env, delayed = gate
    env.termination_manager = delayed
    delayed.first_done = first_done
    motion = object.__new__(_HoldLastMotion)
    motion._env = env
    motion.time_steps = np.array([224])
    motion.sampler = SimpleNamespace(current_clip_end_frames=np.array([224]))
    calls = []
    monkeypatch.setattr(MotionCommand, "_update_command", lambda self, ids: calls.append(ids))
    motion._update_command(None)
    assert len(calls) == 1
    if first_done is None:
        assert calls[0] is None, "The next native flip must remain reachable before failure"
    else:
        np.testing.assert_array_equal(calls[0], [0])
    delayed.reset()
    motion._update_command(None)
    assert calls[-1] is None, "After manual reset, native clip loops must resume"


def _scripted_failure(env):
    return np.array([env.common_step_counter >= 251])


@pytest.mark.slow
def test_real_clip_loop_matches_native_before_late_failure_then_holds_reference():
    if os.environ.get("UNIDR_REAL_FAILURE_TAIL") != "1":
        pytest.skip("set UNIDR_REAL_FAILURE_TAIL=1 for real MuJoCo physics")
    root = Path(__file__).parents[2]
    trajectories, phases = [], []
    for diagnostic in (False, True):
        with initialize_config_dir(
            config_dir=str(root / "src/unilab/conf/ppo"), version_base="1.3"
        ):
            cfg = compose("config", overrides=["task=g1_flip_tracking/mujoco"])
        override = BackendAdapter(cfg, root_dir=root, algo_name="ppo").build_task_env_cfg_override()
        if diagnostic:
            override["commands"]["motion"]["_target_"] = (
                "unilab.visualization.failure_tail.HoldLastMotionCfg"
            )
        with closing(registry_env_factory("G1FlipTracking", "mujoco")(1, override)) as env:
            # Deterministic late failure exercises the second reference cycle.
            # This synthetic term is only a regression fixture, never playback.
            env.cfg.terminations = {"scripted": TerminationTermCfg(func=_scripted_failure)}
            env.termination_manager = (
                FailureTailTerminations(env, 400)
                if diagnostic
                else TerminationManager(env.cfg.terminations, env)
            )
            env.set_autoreset(False)
            env.reset(seed=1)
            motion = env.command_manager.get_term("motion")
            states, frames = [], []
            for _ in range(470 if diagnostic else 251):
                env.step(np.zeros((1, 29), np.float32))
                states.append(env.get_physics_state_snapshot())
                frames.append(int(motion.time_steps[0]))
            trajectories.append(np.stack(states))
            phases.append(np.array(frames))
            if diagnostic:
                assert env.termination_manager.first_done == 250
    np.testing.assert_array_equal(trajectories[0], trajectories[1][:251])
    np.testing.assert_array_equal(phases[0][:250], phases[1][:250])
    assert phases[1][223] == 224 and phases[1][224] == 0
    assert np.all(phases[1][448:] == 224)


@pytest.mark.slow
def test_real_mujoco_continues_through_failure_and_holds_last_reference():
    if os.environ.get("UNIDR_REAL_FAILURE_TAIL") != "1":
        pytest.skip("set UNIDR_REAL_FAILURE_TAIL=1 for real MuJoCo physics")
    root = Path(__file__).parents[2]
    with initialize_config_dir(config_dir=str(root / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_flip_tracking/mujoco"])
    override = BackendAdapter(cfg, root_dir=root, algo_name="ppo").build_task_env_cfg_override()
    override["commands"]["motion"]["_target_"] = (
        "unilab.visualization.failure_tail.HoldLastMotionCfg"
    )
    with closing(registry_env_factory("G1FlipTracking", "mujoco")(1, override)) as env:
        env.termination_manager = FailureTailTerminations(env, 400)
        env.set_autoreset(False)
        env.reset(seed=1)
        motion = env.command_manager.get_term("motion")
        frames, times = [], []
        for _ in range(240):
            state = env.step(np.zeros((1, 29), np.float32))
            assert not state.terminated[0] and not state.truncated[0]
            assert all(np.isfinite(value).all() for value in state.obs.values())
            snapshot = env.get_physics_state_snapshot()
            assert np.isfinite(snapshot).all()
            frames.append(int(motion.time_steps[0]))
            times.append(float(snapshot[0, 0]))
        assert env.termination_manager.first_done is not None
        np.testing.assert_array_equal(frames, np.minimum(np.arange(1, 241), 224))
        np.testing.assert_allclose(np.diff(times), env.step_dt, atol=1e-6)
        reference = motion.motion.get_motion_at_frame(np.array([224]))
        np.testing.assert_array_equal(motion.command[0, :29], reference.joint_pos[0])
        # A resample would restore the initial state and increment this counter.
        assert int(motion.command_counter[0]) == 1
