"""Optional native regression for non-upright, moving reset states."""

from __future__ import annotations

import numpy as np
import pytest

from unisim.backend.genesis.backend import GenesisBackend
from unisim.scene import SceneCfg
from unisim.utils.rotation import (
    np_quat_apply_batched,
    np_quat_apply_inverse_batched,
    np_quat_mul_batched,
)


def test_genesis_reset_refreshes_body_and_sensor_velocity_without_step(tmp_path, monkeypatch):
    pytest.importorskip("genesis")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Genesis numerical reset regression requires a CUDA device")
    model = tmp_path / "moving_reset.xml"
    model.write_text(
        """<mujoco><worldbody>
          <geom name="floor" type="plane" size="0 0 0.05"/>
          <body name="base" pos="0 0 1"><freejoint/>
            <geom type="sphere" size="0.05" mass="1"/>
            <body name="rotor" pos="0.2 0 0"><joint name="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.03" mass="0.2"/>
              <site name="imu" pos="0.1 0 0"/>
            </body>
          </body>
        </worldbody><actuator><position name="hinge" joint="hinge" kp="10" kv="1"/>
        </actuator><sensor><velocimeter name="velocity" site="imu"/>
          <gyro name="gyro" site="imu"/></sensor></mujoco>""",
        encoding="utf-8",
    )
    backend = GenesisBackend(SceneCfg(model_file=str(model)), 2, 0.005, base_name="base")
    try:
        backend.materialize()

        def no_step(*args, **kwargs):
            pytest.fail("reset must not advance physics to refresh kinematics")

        monkeypatch.setattr(backend._scene, "step", no_step)
        half = np.sqrt(0.5)
        qpos = np.array([[1, 2, 3, half, half, 0, 0, 0.4], [3, 2, 1, 0, 1, 0, 0, -0.5]], np.float32)
        qvel = np.array(
            [[1, 2, 3, 0.3, -0.4, 1.2, 2.3], [-2, 1, 3, -0.5, 0.7, -1, -1.4]], np.float32
        )
        ids = backend.get_body_ids(["base", "rotor"])
        for rows in (np.array([0, 1], np.int32), np.array([1], np.int32)):
            if rows.size == 1:
                qpos[1, 3:7] = [half, 0, half, 0]
                qpos[1, -1] = 0.9
                qvel[1] *= -0.6
            backend.set_state(rows, qpos[rows], qvel[rows])
            root_quat = qpos[:, 3:7]
            joint_quat = np.zeros((2, 4), np.float32)
            joint_quat[:, 0] = np.cos(qpos[:, -1] / 2)
            joint_quat[:, 3] = np.sin(qpos[:, -1] / 2)
            rotor_quat = np_quat_mul_batched(root_quat, joint_quat)
            offset = np_quat_apply_batched(root_quat, np.tile([0.2, 0, 0], (2, 1)))
            root_ang = np_quat_apply_batched(root_quat, qvel[:, 3:6])
            joint_ang = np.column_stack((np.zeros((2, 2)), qvel[:, -1]))
            rotor_ang = root_ang + np_quat_apply_batched(root_quat, joint_ang)
            rotor_vel = qvel[:, :3] + np.cross(root_ang, offset)
            expected = (
                np.stack((qpos[:, :3], qpos[:, :3] + offset), axis=1),
                np.stack((root_quat, rotor_quat), axis=1),
                np.stack((qvel[:, :3], rotor_vel), axis=1),
                np.stack((root_ang, rotor_ang), axis=1),
            )
            actual = backend.get_body_state_w(ids)
            for field, wanted in zip(actual, expected, strict=True):
                np.testing.assert_allclose(field, wanted, atol=2e-5)
            sensor_offset = np_quat_apply_batched(rotor_quat, np.tile([0.1, 0, 0], (2, 1)))
            sensor_vel = np_quat_apply_inverse_batched(
                rotor_quat, rotor_vel + np.cross(rotor_ang, sensor_offset)
            )
            np.testing.assert_allclose(backend.get_sensor_data("velocity"), sensor_vel, atol=2e-5)
            np.testing.assert_allclose(
                backend.get_sensor_data("gyro"),
                np_quat_apply_inverse_batched(rotor_quat, rotor_ang),
                atol=2e-5,
            )
    finally:
        backend.close()
