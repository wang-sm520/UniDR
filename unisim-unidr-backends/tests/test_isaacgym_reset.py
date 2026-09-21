"""Reset kinematics and deferred-tensor handling, without importing the Gym SDK."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from unisim.backend.isaacgym import kinematics, worker
from unisim.backend.isaacgym.backend import IsaacGymBackend
from unisim.backend.subprocess_ipc import protocol
from unisim.scene import SceneCfg


@pytest.fixture
def chain(tmp_path):
    model = tmp_path / "chain.xml"
    model.write_text(
        """<mujoco><default><default class="hinge">
          <joint axis="0 0 2" pos="1 0 0" ref="30"/>
        </default></default><worldbody><body name="root"><freejoint/>
          <body name="arm" pos="1 0 0"><joint class="hinge" name="turn"/>
            <body name="tip" pos="2 0 0"/>
            <body name="slider" pos="0 1 0"><joint name="slide" type="slide"
              axis="0 1 0" ref="0.1"/></body>
          </body></body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco><include file="chain.xml"/></mujoco>', encoding="utf-8")
    return scene


def _root_state():
    root = np.zeros((1, 13))
    root[:, :3] = [1, 2, 3]
    root[:, 3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    return root


def test_fk_joint_anchor_ref_units_native_order_and_cold_include(chain, monkeypatch):
    fk = kinematics.ResetKinematics(chain, ["tip", "root", "slider", "arm"], ["slide", "turn"])
    monkeypatch.setattr(kinematics.ET, "parse", Mock(side_effect=AssertionError("hot XML parse")))
    states = fk.evaluate(_root_state(), np.array([[[0.3, 0], [2 * np.pi / 3, 0]]]))
    np.testing.assert_allclose(
        states[0, :, :3], [[0, 4, 3], [1, 2, 3], [2, 2.8, 3], [2, 4, 3]], atol=1e-14
    )
    np.testing.assert_allclose(states[0, [0, 2, 3], 3:7], [[0, 0, 0, 1]] * 3, atol=1e-15)
    np.testing.assert_array_equal(states[:, :, 7:], 0)


def test_fk_world_velocities_match_pose_derivatives(chain):
    fk = kinematics.ResetKinematics(chain, ["root", "arm", "tip", "slider"], ["turn", "slide"])
    root = _root_state()
    root[:, 7:13] = [0.2, -0.3, 0.1, 0.4, -0.5, 0.6]
    dof = np.array([[[0.7, 0.8], [0.3, -0.2]]])
    states = fk.evaluate(root, dof)
    epsilon = 1e-6
    moved_root = root.copy()
    moved_root[:, :3] += epsilon * root[:, 7:10]
    # Integrate the free root with a world-axis rotation.
    omega = root[0, 10:13]
    angle = np.linalg.norm(omega) * epsilon
    delta = np.r_[np.cos(angle / 2), omega / np.linalg.norm(omega) * np.sin(angle / 2)]
    moved_root[:, 3:7] = kinematics._multiply(delta, root[:, 3:7])
    moved_dof = dof.copy()
    moved_dof[:, :, 0] += epsilon * dof[:, :, 1]
    moved = fk.evaluate(moved_root, moved_dof)
    np.testing.assert_allclose(
        (moved[:, :, :3] - states[:, :, :3]) / epsilon, states[:, :, 7:10], atol=2e-6
    )
    inverse = states[:, :, 3:7].copy()
    inverse[:, :, 1:] *= -1
    world_delta = kinematics._multiply(moved[:, :, 3:7], inverse)
    np.testing.assert_allclose(2 * world_delta[:, :, 1:] / epsilon, states[:, :, 10:13], atol=1e-6)


@pytest.mark.parametrize(
    "replace, replacement, message",
    [
        ("<freejoint/>", "", "free root"),
        ('name="arm" pos="1 0 0"', 'name="arm" euler="0 0 30"', "pos/quat"),
        ('name="turn"', 'name="turn" type="ball"', "joint"),
        ('name="turn"', 'name="turn" axis="0 0 0"', "zero joint axis"),
        ('name="turn"', 'name="turn" ref="nan"', "joint ref"),
        ('<body name="tip" pos="2 0 0"/>', "<frame/>", "body construction"),
    ],
)
def test_unsupported_xml_fails_closed(chain, replace, replacement, message):
    included = chain.parent / "chain.xml"
    included.write_text(included.read_text().replace(replace, replacement))
    with pytest.raises(ValueError, match=message):
        kinematics.ResetKinematics(chain, ["root", "arm", "tip", "slider"], ["turn", "slide"])


class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def view(self, *shape):
        return _Tensor(self.array.reshape(shape))

    reshape = view

    def contiguous(self):
        return self

    def to(self, _device):
        return self

    def long(self):
        return self.array.astype(np.int64)

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def __getitem__(self, key):
        return _Tensor(self.array[key])

    def __setitem__(self, key, value):
        self.array[key] = value.array


def _context(chain):
    ctx = worker._WorkerContext(protocol)
    ctx.num_envs, ctx.num_dof, ctx.num_bodies = 3, 2, 4
    ctx._pending_reset_rows = np.zeros(3, dtype=bool)
    ctx._body_com = np.zeros((4, 3))
    ctx.dof_names = ["turn", "slide"]
    ctx.torch = SimpleNamespace(
        from_numpy=_Tensor,
        int32=np.int32,
        arange=lambda n, **kwargs: _Tensor(np.arange(n, dtype=kwargs["dtype"])),
    )
    ctx.gymtorch = SimpleNamespace(unwrap_tensor=lambda tensor: tensor.array)
    ctx.gym = SimpleNamespace(
        set_actor_root_state_tensor_indexed=Mock(),
        set_dof_state_tensor_indexed=Mock(),
    )
    ctx._root_state = _Tensor(np.zeros((3, 13)))
    ctx._dof_state = _Tensor(np.zeros((3, 2, 2)))
    ctx._reset_kinematics = kinematics.ResetKinematics(
        chain,
        ["root", "arm", "tip", "slider"],
        ctx.dof_names,
    )
    ctx.slots = {
        name: np.full(shape, 123, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.slot_shapes(3, 2, 4).items()
    }
    return ctx


def test_reset_publishes_selected_rows_without_simulate_or_refresh(chain, monkeypatch):
    ctx = _context(chain)
    monkeypatch.setattr(kinematics.ET, "parse", Mock(side_effect=AssertionError("hot XML parse")))
    ctx.slots["reset_env_ids"][:2] = [2, 0]
    ctx.slots["reset_qpos"][:2] = [
        [1, 2, 3, np.sqrt(0.5), 0, 0, np.sqrt(0.5), 0.7, 0.3],
        [3, 2, 1, 1, 0, 0, 0, 0.8, 0.4],
    ]
    ctx.slots["reset_qvel"][:2] = [[0.2, 0.3, 0.4, 1, 0, 0, 0.8, -0.2], [0] * 8]
    ctx.set_state({"count": 2})
    for name in ("root_state", "dof_state", "body_state", "contact_force"):
        np.testing.assert_array_equal(ctx.slots[name][1], 123)
    np.testing.assert_allclose(ctx.slots["root_state"][2, 10:13], [0, 1, 0], atol=1e-7)
    np.testing.assert_array_equal(
        ctx.slots["root_state"][[2, 0], :7], ctx.slots["reset_qpos"][:2, :7]
    )
    np.testing.assert_array_equal(
        ctx.slots["dof_state"][[2, 0], :, 0], ctx.slots["reset_qpos"][:2, 7:]
    )
    np.testing.assert_array_equal(ctx.slots["contact_force"][[2, 0]], 0)
    assert np.isfinite(ctx.slots["body_state"][[2, 0]]).all()
    assert ctx.gym.set_actor_root_state_tensor_indexed.call_count == 0
    assert ctx.gym.set_dof_state_tensor_indexed.call_count == 0
    np.testing.assert_array_equal(ctx._pending_reset_rows, [True, False, True])
    ctx.slots["reset_env_ids"][:1] = [0]
    ctx.slots["reset_qpos"][0, 0] = 42
    ctx.set_state({"count": 1})
    worker._dispatch(ctx, protocol, protocol.CMD_REFRESH, None)
    assert ctx.slots["root_state"][0, 0] == 42
    ctx._flush_reset_writes()
    ctx._flush_reset_writes()
    assert ctx.gym.set_actor_root_state_tensor_indexed.call_count == 1
    assert ctx.gym.set_dof_state_tensor_indexed.call_count == 1
    arguments = ctx.gym.set_actor_root_state_tensor_indexed.call_args.args
    np.testing.assert_array_equal(arguments[2], [0, 2])
    assert arguments[1][0, 0] == 42
    assert arguments[3] == 2
    # Any simulate or refresh would raise: the SDK double only exposes setters.


def test_initial_keyframe_has_current_fk_and_no_refresh(chain):
    ctx = _context(chain)
    qpos = np.r_[_root_state()[0, :7], 0.3, 2 * np.pi / 3]
    ctx._apply_initial_keyframe(qpos, ["slide", "turn"])
    ctx.attach_slots({"slots": {}})
    assert ctx.gym.set_actor_root_state_tensor_indexed.call_count == 0
    assert ctx._initial_reset is None
    np.testing.assert_allclose(ctx.slots["body_state"][:, 2, :3], [[0, 4, 3]] * 3, atol=1e-6)
    np.testing.assert_array_equal(ctx.slots["body_state"][:, :, 7:], 0)
    np.testing.assert_array_equal(ctx.slots["contact_force"], 0)


def test_native_com_velocity_conversion_preserves_link_contract(chain):
    ctx = _context(chain)
    ctx._body_com[:, 0] = 0.1
    ctx.slots["reset_env_ids"][:] = np.arange(3)
    ctx.slots["reset_qpos"][:] = np.r_[_root_state()[0, :7], 0.7, 0.3]
    ctx.slots["reset_qvel"][:] = [1, 2, 3, 0, 0, 2, 1, 0]
    ctx.set_state({"count": 3})
    np.testing.assert_allclose(ctx._root_state.array[:, 7:10], [[0.8, 2, 3]] * 3, atol=1e-7)
    expected_root = ctx.slots["root_state"].copy()
    expected_body = ctx.slots["body_state"].copy()
    native_body = expected_body.copy()
    native_body[:, :, 7:10] += np.cross(
        native_body[:, :, 10:13], protocol.quat_rotate(native_body[:, :, 3:7], ctx._body_com)
    )
    native_body[:, :, 3:7] = protocol.wxyz_to_xyzw(native_body[:, :, 3:7])
    ctx._body_state = _Tensor(native_body)
    ctx._contact_force = _Tensor(np.zeros((3, 4, 3)))
    ctx._refresh_tensors = lambda: None
    ctx._pending_reset_rows[:] = False
    ctx.refresh_state_slots()
    np.testing.assert_allclose(ctx.slots["root_state"], expected_root, atol=1e-7)
    np.testing.assert_allclose(ctx.slots["body_state"], expected_body, atol=1e-6)


def test_step_commits_pending_rows_once_before_simulate(chain):
    ctx = _context(chain)
    ctx._pending_reset_rows[:] = True
    calls = Mock()
    ctx.gym = calls
    ctx.refresh_state_slots = lambda: calls.refresh()
    ctx.step({"nsteps": 2})
    ctx.step({"nsteps": 1})
    assert [entry[0] for entry in calls.mock_calls] == [
        "set_actor_root_state_tensor_indexed",
        "set_dof_state_tensor_indexed",
        "set_dof_position_target_tensor",
        "simulate",
        "fetch_results",
        "simulate",
        "fetch_results",
        "refresh",
        "set_dof_position_target_tensor",
        "simulate",
        "fetch_results",
        "refresh",
    ]


@pytest.mark.skipif(os.getenv("UNISIM_TEST_ISAACGYM") != "1", reason="opt-in vendor SDK test")
def test_native_gym_reset_and_link_velocities(tmp_path):
    model = tmp_path / "native.xml"
    model.write_text("""<mujoco><compiler angle="radian"/><worldbody>
      <body name="root" pos="0 0 3"><freejoint/>
        <inertial pos="0.1 0 0" mass="1" diaginertia="0.1 0.1 0.1"/><geom size="0.1"/>
        <body name="arm" pos="0.5 0 0"><joint name="turn" axis="0 0 1"/>
          <inertial pos="0.2 0 0" mass="1" diaginertia="0.1 0.1 0.1"/><geom size="0.05"/>
        </body></body></worldbody><actuator>
      <position joint="turn" kp="0" kv="0" forcerange="-100 100"/></actuator></mujoco>""")
    backend = IsaacGymBackend(SceneCfg(model_file=str(model)), 2, 0.0001, base_name="root")
    try:
        backend.materialize()
        qpos = np.array([[1, 2, 3, 1, 0, 0, 0, 0.7], [2, 3, 4, 1, 0, 0, 0, 0.8]])
        qvel = np.array([[0.2, 0.3, 0.4, 0, 0, 2, 1], [0.4, 0.5, 0.6, 0, 0, 3, -1]])
        backend.set_state(np.array([0, 1]), qpos, qvel)
        before = backend.get_body_pos_w(np.array([0, 1])).copy()
        backend.set_state(np.array([1]), qpos[1:], qvel[1:])
        np.testing.assert_array_equal(backend.get_body_pos_w(np.array([0, 1])), before)
        backend.step(np.zeros((2, 1)))
        np.testing.assert_allclose(
            (backend.get_base_pos() - qpos[:, :3]) / 0.0001, qvel[:, :3], atol=0.02
        )
        root = np.concatenate(
            (
                backend.get_base_pos(),
                backend.get_base_quat(),
                backend.get_base_lin_vel(),
                backend.get_base_ang_vel(),
            ),
            axis=-1,
        )
        dof = np.stack((backend.get_dof_pos(), backend.get_dof_vel()), axis=-1)
        expected = kinematics.ResetKinematics(model, ["root", "arm"], ["turn"]).evaluate(root, dof)
        np.testing.assert_allclose(
            backend.get_body_pos_w(np.array([0, 1])), expected[:, :, :3], atol=1e-5
        )
        np.testing.assert_allclose(
            backend.get_body_lin_vel_w(np.array([0, 1])), expected[:, :, 7:10], atol=1e-5
        )
    finally:
        backend.close()


def test_standalone_worker_dependencies_keep_python38_syntax():
    for module in (kinematics, worker):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 8))
    loaded = worker._load_module(kinematics.__file__, "standalone_reset_kinematics")
    assert loaded.ResetKinematics.__name__ == "ResetKinematics"
