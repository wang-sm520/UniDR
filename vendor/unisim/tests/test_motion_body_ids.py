"""Motion datasets preserve MJCF body order, including worldbody at index zero."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.backend.genesis.backend import GenesisBackend
from unisim.backend.genesis.materialization import scan_genesis_model_metadata
from unisim.backend.isaacgym.backend import IsaacGymBackend
from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.backend.subprocess_ipc.sensors import KIND_LOCAL_LINVEL, SceneSensorSpec
from unisim.scene import SceneCfg


@pytest.fixture
def motion_scene(tmp_path: Path) -> SceneCfg:
    path = tmp_path / "motion.xml"
    path.write_text(
        """<mujoco><worldbody>
          <body name="pelvis"><freejoint/><geom size="0.1"/>
            <body name="torso"><joint name="waist"/><geom size="0.1"/></body>
            <body name="foot"><joint name="ankle"/><geom size="0.1"/></body>
          </body>
        </worldbody><actuator>
          <position name="waist" joint="waist" kp="10"/>
          <position name="ankle" joint="ankle" kp="10"/>
        </actuator></mujoco>""",
        encoding="utf-8",
    )
    return SceneCfg(model_file=str(path))


@pytest.mark.parametrize("backend_cls", [IsaacGymBackend, IsaacSimBackend])
def test_worker_motion_ids_keep_world_slot_and_requested_order(backend_cls, motion_scene):
    backend = backend_cls(motion_scene, 2, 0.005, base_name="pelvis")
    try:
        names = ["foot", "pelvis", "torso", "pelvis"]
        np.testing.assert_array_equal(backend.get_body_ids(names), [2, 0, 1, 0])
        ids = backend.get_motion_body_ids(names)
        np.testing.assert_array_equal(ids, [3, 1, 2, 1])
        assert ids.dtype == np.int32
        dataset = np.array(["world", "pelvis", "torso", "foot"])
        np.testing.assert_array_equal(dataset[ids], names)
        assert backend.get_motion_body_ids([]).shape == (0,)
        with pytest.raises(ValueError, match="missing"):
            backend.get_motion_body_ids(["missing"])
    finally:
        backend.close()


def test_genesis_motion_ids_keep_existing_world_slot(motion_scene):
    mujoco = pytest.importorskip("mujoco")
    metadata = scan_genesis_model_metadata(mujoco, motion_scene)
    assert metadata.body_names == ("world", "pelvis", "torso", "foot")
    # Only test cold metadata mapping; no Genesis session or GPU is needed.
    backend = GenesisBackend.__new__(GenesisBackend)
    backend._body_ids = {name: index for index, name in enumerate(metadata.body_names)}
    names = ["foot", "pelvis", "torso", "pelvis"]
    ids = backend.get_motion_body_ids(names)
    np.testing.assert_array_equal(ids, [3, 1, 2, 1])
    np.testing.assert_array_equal(np.asarray(metadata.body_names)[ids], names)
    assert ids.dtype == np.int32
    assert backend.get_motion_body_ids([]).shape == (0,)
    with pytest.raises(ValueError, match="missing"):
        backend.get_motion_body_ids(["missing"])


@pytest.mark.parametrize("backend_cls", [IsaacGymBackend, IsaacSimBackend])
def test_offset_velocimeter_rotates_body_and_site_frames(backend_cls, motion_scene, monkeypatch):
    backend = backend_cls(motion_scene, 2, 0.005, base_name="pelvis")
    try:
        monkeypatch.setattr(backend, "_require_state", lambda _: None)
        half = np.sqrt(0.5)
        spec = SceneSensorSpec(
            name="speed",
            kind=KIND_LOCAL_LINVEL,
            body_name="pelvis",
            local_pos=(0.5, 0, 0),
            local_quat=(half, half, 0, 0),
        )
        backend._sensor_map = {"speed": (spec, 0)}
        state = np.zeros((2, 1, 13), dtype=np.float32)
        state[:, 0, 3:7] = (half, 0, 0, half)
        state[:, 0, 7:10] = (1, 2, 3)
        state[:, 0, 10:13] = [(0, 0, 2), (2, 0, 0)]
        backend._slots["body_state"] = state
        # World site velocities are (0, 2, 3) and (1, 2, 4), then rotate
        # through inverse body Rz(90) and inverse local-site Rx(90).
        np.testing.assert_allclose(
            backend.get_sensor_data("speed"), [[2, 3, 0], [2, 4, 1]], atol=1e-6
        )
    finally:
        backend.close()
