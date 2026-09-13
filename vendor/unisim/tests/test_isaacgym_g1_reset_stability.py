"""Native G1 regressions for mass resets with the IsaacGym GPU PhysX solver."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from unisim import create_backend
from unisim.dr.types import ResetRandomizationPayload
from unisim.scene import SceneCfg

pytestmark = [
    pytest.mark.slow,
    pytest.mark.optional,
    pytest.mark.skipif(
        os.environ.get("UNILAB_RUN_MULTISIM_PHYSICS") != "1",
        reason="native G1 regression requires UNILAB_RUN_MULTISIM_PHYSICS=1",
    ),
]


@pytest.mark.parametrize("reapply_gains", (False, True), ids=("mass-only", "mass-and-gains"))
def test_g1_repeated_mass_reset_is_finite(reapply_gains: bool) -> None:
    scene = Path(
        os.environ.get(
            "UNILAB_G1_SCENE",
            str(
                Path(__file__).resolve().parents[2]
                / "UniLab/src/unilab/assets/robots/g1/scene_flat.xml"
            ),
        )
    )
    assert scene.is_absolute() and scene.is_file()
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=str(scene), default_keyframe_name="stand"),
        4,
        1.0 / 150.0,
        base_name="pelvis",
        isaacgym_device_id=0,
        isaacgym_worker_timeout_s=120.0,
    )
    try:
        backend.materialize()
        rows = np.arange(4, dtype=np.int32)
        selected = np.array([1, 3], dtype=np.int32)
        joints = backend.get_actuator_joint_names()
        assert len(joints) == 29
        qpos = np.tile(np.asarray(backend.get_keyframe_qpos("stand"), dtype=np.float32), (4, 1))
        qpos[:, 0] += np.array([0.0, 0.03, -0.02, 0.05])
        qpos[:, 2] += 2.0
        qvel = np.tile(np.asarray(backend.get_init_qvel(), dtype=np.float32), (4, 1))
        ctrl = qpos[:, backend.get_joint_state_qpos_indices(joints)].copy()
        ctrl[:, joints.index("left_hip_pitch_joint")] += 0.18
        ctrl[:, joints.index("right_hip_pitch_joint")] += 0.09
        mass = np.tile(backend.get_body_mass(), (4, 1))
        nominal_mass = mass.copy()
        kp, kd = (np.tile(values, (4, 1)) for values in backend.get_actuator_gains())
        pelvis_id = backend.get_body_id("pelvis")
        for label, multiplier in (("nominal", 1.0), ("repeat", 1.0), ("low", 0.8), ("high", 1.2)):
            backend.set_state(
                rows,
                qpos.copy(),
                qvel.copy(),
                randomization=ResetRandomizationPayload(
                    body_mass=mass.copy(), kp=kp.copy(), kd=kd.copy()
                ),
            )
            mass[selected, pelvis_id] = nominal_mass[selected, pelvis_id] * multiplier
            reset_rows = rows if multiplier == 1.0 else selected
            write_gains = reapply_gains or multiplier == 1.0
            backend.set_state(
                reset_rows,
                qpos[reset_rows].copy(),
                qvel[reset_rows].copy(),
                randomization=ResetRandomizationPayload(
                    body_mass=mass[reset_rows].copy(),
                    kp=kp[reset_rows].copy() if write_gains else None,
                    kd=kd[reset_rows].copy() if write_gains else None,
                ),
            )
            for step in range(12):
                backend.step(ctrl.copy(), nsteps=1)
                state = backend.get_state(("qpos", "qvel"))
                if step == 0:
                    np.testing.assert_allclose(
                        backend.get_base_pos()[:, 2],
                        qpos[:, 2],
                        rtol=0.0,
                        atol=0.02,
                        err_msg=f"{label}: mass reset discarded the requested airborne root pose",
                    )
                for name, values in state.items():
                    assert np.isfinite(values).all(), (
                        f"{label}, step {step}, {name}: non-finite native G1 state; "
                        f"per-row max magnitude={np.max(np.abs(values), axis=1).tolist()}"
                    )
    finally:
        backend.cleanup_scene_assets()


def test_g1_mass_reset_preserves_grounded_and_airborne_contacts() -> None:
    scene = Path(
        os.environ.get(
            "UNILAB_G1_SCENE",
            str(
                Path(__file__).resolve().parents[2]
                / "UniLab/src/unilab/assets/robots/g1/scene_flat.xml"
            ),
        )
    )
    assert scene.is_absolute() and scene.is_file()
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=str(scene), default_keyframe_name="stand"),
        4,
        1.0 / 150.0,
        base_name="pelvis",
        isaacgym_device_id=0,
        isaacgym_worker_timeout_s=120.0,
    )
    try:
        backend.materialize()
        rows = np.arange(4, dtype=np.int32)
        stand = np.tile(np.asarray(backend.get_keyframe_qpos("stand"), dtype=np.float32), (4, 1))
        qvel = np.tile(backend.get_init_qvel(), (4, 1))
        joints = backend.get_actuator_joint_names()
        foot_ids = backend.get_body_ids(["left_ankle_roll_link", "right_ankle_roll_link"])
        sensor_names = tuple(
            f"{side}_foot_contact_{corner}" for side in ("left", "right") for corner in range(4)
        )
        mass = np.tile(backend.get_body_mass(), (4, 1))
        mass[:, backend.get_body_id("pelvis")] *= np.array([0.8, 1.2, 1.2, 0.8])
        kp, kd = (np.tile(values, (4, 1)) for values in backend.get_actuator_gains())
        for airborne in (
            np.array([True, False, True, False]),
            np.array([False, True, False, True]),
        ):
            qpos = stand.copy()
            qpos[airborne, 2] += 2.0
            backend.set_state(
                rows,
                qpos.copy(),
                qvel.copy(),
                randomization=ResetRandomizationPayload(body_mass=mass, kp=kp, kd=kd),
            )
            ctrl = qpos[:, backend.get_joint_state_qpos_indices(joints)].copy()
            hits = np.zeros((4, 2), dtype=np.int32)
            for step in range(40):
                backend.step(ctrl.copy(), nsteps=1)
                root = backend.get_base_pos().copy()
                feet = backend.get_body_pos_w(foot_ids).copy()
                contacts = backend.get_sensor_data_batch(sensor_names).reshape(4, 2, 4)
                assert np.isfinite(root).all() and np.isfinite(feet).all()
                assert np.isfinite(contacts).all()
                if step == 0:
                    np.testing.assert_allclose(
                        root[:, 2],
                        qpos[:, 2],
                        rtol=0.0,
                        atol=0.02,
                        err_msg="mass reset must preserve root height before the first integration",
                    )
                assert (feet[airborne, :, 2] > 1.0).all()
                assert not contacts[airborne].any()
                if step >= 2:
                    hits += contacts.any(axis=2)
            assert (hits[~airborne] >= 2).all(), hits.tolist()
    finally:
        backend.cleanup_scene_assets()
