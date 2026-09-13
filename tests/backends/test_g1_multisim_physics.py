"""Opt-in G1Flat4 physical acceptance, with one JUnit case per real backend.

The validation driver must run each parameter in a separate process, including
Motrix and Genesis. Set UNILAB_RUN_MULTISIM_PHYSICS=1 to opt in; missing assets,
SDKs, devices, or public capabilities then fail rather than skip. No fake worker,
replacement robot, private SDK read, or dependency installation is used here.

Reset payloads exercise deterministic endpoints of the task's pelvis mass
0.8..1.2 and all 29 actuator KP/KD 0.9..1.1 ranges. Sampling/config composition
belongs to the owner tests. Public mass/gain getters currently expose nominal
tables, not live per-environment parameters. Physical trajectories therefore
must demonstrate each override, selected-row isolation, and nominal restoration.
The payload never requests inertia changes; SimBackend has no live inertia
getter, so exact native inertia preservation cannot be certified by this gate.
These limits are also written to JUnit properties, not represented as skips.
The damping probe uses 1.2 rad/s initial joint speed: at 0.6 rad/s PhysX can
stop a joint in the first step under both nominal and higher damping, masking
their difference. This diagnostic excitation does not change training resets.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from unisim.backend.base import SimBackend
    from unisim.dr.types import ResetRandomizationPayload

pytestmark = pytest.mark.skipif(
    os.environ.get("UNILAB_RUN_MULTISIM_PHYSICS") != "1",
    reason="real G1 physics gate requires UNILAB_RUN_MULTISIM_PHYSICS=1",
)

_BACKENDS = ("isaacgym", "isaacsim", "motrix", "genesis")
_NUM_ENVS = 4
_NUM_JOINTS = 29
_SIM_DT = 1.0 / 150.0
_ROWS = np.arange(_NUM_ENVS, dtype=np.int32)
_DR_ROWS = np.array([1, 3], dtype=np.int32)
_CONTROL_ROWS = np.array([0, 2], dtype=np.int32)
_STATE_ATOL = 2e-5
_STATE_RTOL = 5e-4
_EFFECT_ATOL = 5e-6
_KD_PROBE_SPEED = 1.2
_FOOT_SENSORS = tuple(
    f"{side}_foot_contact_{corner}" for side in ("left", "right") for corner in range(4)
)


def _state(backend: SimBackend) -> np.ndarray:
    snapshot = backend.get_state(("qpos", "qvel"))
    state = np.concatenate((snapshot["qpos"], snapshot["qvel"]), axis=1).copy()
    assert state.shape == (_NUM_ENVS, 7 + _NUM_JOINTS + 6 + _NUM_JOINTS)
    assert np.isfinite(state).all(), "G1 physics state contains non-finite values"
    return state


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    np.testing.assert_allclose(actual, expected, rtol=_STATE_RTOL, atol=_STATE_ATOL, err_msg=label)


@dataclass
class _G1Probe:
    backend: SimBackend
    body_mass: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    pelvis_id: int
    joint_names: tuple[str, ...]
    joint_qpos: np.ndarray
    joint_qvel: np.ndarray
    root_qpos: tuple[int, ...]
    stand_qpos: np.ndarray
    zero_qvel: np.ndarray
    active_levels: np.ndarray = field(default_factory=lambda: np.zeros((_NUM_ENVS, 3)))

    def payload(self, channel: str, levels: np.ndarray) -> ResetRandomizationPayload:
        from unisim.dr.types import ResetRandomizationPayload

        fields = {}
        if channel in ("mass", "combined"):
            mass = np.tile(self.body_mass, (len(levels), 1))
            mass[:, self.pelvis_id] = self.body_mass[self.pelvis_id] * (1.0 + 0.2 * levels)
            fields["body_mass"] = mass
        for name, nominal in (("kp", self.kp), ("kd", self.kd)):
            if channel in (name, "combined"):
                fields[name] = nominal[None, :] * (1.0 + 0.1 * levels[:, None])
        assert fields, f"unknown physics probe channel: {channel}"
        payload = ResetRandomizationPayload(**fields)
        assert payload.body_inertia is None and payload.body_iquat is None
        assert payload.body_ipos is None
        return payload

    def assert_nominal_readback(self) -> None:
        np.testing.assert_array_equal(self.backend.get_body_mass(), self.body_mass)
        actual_kp, actual_kd = self.backend.get_actuator_gains()
        np.testing.assert_array_equal(actual_kp, self.kp)
        np.testing.assert_array_equal(actual_kd, self.kd)

    def run(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        steps: int,
        rows: np.ndarray,
        channel: str,
        levels: np.ndarray,
    ) -> np.ndarray:
        """Replay absolute parameters without assuming omitted reset fields persist."""
        from unisim.dr.types import ResetRandomizationPayload

        active = ResetRandomizationPayload(
            body_mass=self.payload("mass", self.active_levels[:, 0]).body_mass,
            kp=self.payload("kp", self.active_levels[:, 1]).kp,
            kd=self.payload("kd", self.active_levels[:, 2]).kd,
        )
        self.backend.set_state(_ROWS, qpos.copy(), qvel.copy(), randomization=active)
        before = _state(self.backend)
        self.backend.set_state(
            rows,
            qpos[rows].copy(),
            qvel[rows].copy(),
            randomization=self.payload(channel, levels),
        )
        for column, name in enumerate(("mass", "kp", "kd")):
            if channel in (name, "combined"):
                self.active_levels[rows, column] = levels
        after = _state(self.backend)
        untouched = np.setdiff1d(_ROWS, rows)
        _assert_close(after[untouched], before[untouched], f"{channel}: reset row isolation")
        _assert_close(after[rows, : qpos.shape[1]], qpos[rows], f"{channel}: selected qpos")
        self.assert_nominal_readback()
        trajectory = []
        for _ in range(steps):
            self.backend.step(ctrl.copy(), nsteps=1)
            trajectory.append(_state(self.backend))
        return np.stack(trajectory)


def _prepare_probe(backend: SimBackend, robot_file: Path) -> _G1Probe:
    assert backend.num_envs == _NUM_ENVS
    assert backend.num_actuators == _NUM_JOINTS
    assert backend.num_dof_vel == _NUM_JOINTS
    required = frozenset({"body_mass", "kp", "kd"})
    unsupported = backend.get_dr_capabilities().get_unsupported_reset_terms(required)
    assert not unsupported, f"{backend.backend_type}: required reset terms missing: {unsupported}"

    robot = ET.parse(robot_file).getroot()
    positions = {element.attrib["joint"]: element for element in robot.findall("actuator/position")}
    joint_names = backend.get_actuator_joint_names()
    assert len(set(joint_names)) == _NUM_JOINTS
    assert set(joint_names) == set(positions), "runtime must use the actual 29-joint G1 asset"
    kp, kd = (np.asarray(values).copy() for values in backend.get_actuator_gains())
    assert kp.shape == kd.shape == (_NUM_JOINTS,)
    for values, attribute in ((kp, "kp"), (kd, "kv")):
        expected = np.array([float(positions[name].attrib[attribute]) for name in joint_names])
        np.testing.assert_allclose(values, expected, rtol=2e-5, atol=1e-6)
        assert np.isfinite(values).all() and (values > 0).all()

    pelvis_id = backend.get_body_id("pelvis")
    body_mass = np.asarray(backend.get_body_mass()).copy()
    assert body_mass.ndim == 1 and 0 <= pelvis_id < body_mass.size
    assert np.isfinite(body_mass).all() and (body_mass >= 0).all()
    pelvis = robot.find("worldbody/body[@name='pelvis']/inertial")
    assert pelvis is not None
    np.testing.assert_allclose(body_mass[pelvis_id], float(pelvis.attrib["mass"]), rtol=2e-5)
    root_layout = backend.get_root_state_layout("pelvis")
    stand = np.asarray(backend.get_keyframe_qpos("stand"), dtype=np.float32)
    zero_qvel = np.asarray(backend.get_init_qvel(), dtype=np.float32)
    assert stand.shape == (7 + _NUM_JOINTS,) and zero_qvel.shape == (6 + _NUM_JOINTS,)
    np.testing.assert_array_equal(zero_qvel, np.zeros_like(zero_qvel))
    joint_qpos = backend.get_joint_state_qpos_indices(joint_names)
    joint_qvel = backend.get_joint_state_qvel_indices(joint_names)
    return _G1Probe(
        backend=backend,
        body_mass=body_mass,
        kp=kp,
        kd=kd,
        pelvis_id=pelvis_id,
        joint_names=joint_names,
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        root_qpos=root_layout.qpos_indices,
        stand_qpos=stand,
        zero_qvel=zero_qvel,
    )


def _require_effect(
    probe: _G1Probe,
    channel: str,
    actual: np.ndarray,
    reference: np.ndarray,
    nominal_noise: np.ndarray,
) -> float:
    if channel == "mass":
        components = slice(probe.stand_qpos.size, probe.stand_qpos.size + 2)
    else:
        components = slice(-_NUM_JOINTS, None)
    delta = np.abs(actual[:, _DR_ROWS, components] - reference[:, _DR_ROWS, components])
    noise = nominal_noise[:, _DR_ROWS, components]
    if channel in ("kp", "kd"):
        signal = delta.max(axis=0)
        threshold = np.maximum(_EFFECT_ATOL, 10.0 * noise.max(axis=0))
        missing = np.argwhere(signal <= threshold)
        assert not missing.size, (
            f"{channel}: no physical effect above repeated-nominal noise for "
            f"{[(_DR_ROWS[row], probe.joint_names[joint]) for row, joint in missing]}; "
            f"signal={signal.tolist()}, threshold={threshold.tolist()}"
        )
    else:
        signal = delta.max(axis=(0, 2))
        threshold = np.maximum(_EFFECT_ATOL, 10.0 * noise.max(axis=(0, 2)))
        assert (signal > threshold).all(), (
            f"{channel}: missing physical effect in rows {_DR_ROWS.tolist()}; "
            f"signal={signal.tolist()}, threshold={threshold.tolist()}"
        )
    return float(delta.max())


def _check_randomization(probe: _G1Probe, channel: str) -> dict[str, float]:
    qpos = np.tile(probe.stand_qpos, (_NUM_ENVS, 1))
    qpos[:, probe.root_qpos[0]] += np.array([0.0, 0.03, -0.02, 0.05])
    qpos[:, probe.root_qpos[2]] += 2.0
    qvel = np.tile(probe.zero_qvel, (_NUM_ENVS, 1))
    ctrl = qpos[:, probe.joint_qpos].copy()
    direction = np.where(np.arange(_NUM_JOINTS) % 2 == 0, 1.0, -1.0)
    if channel == "kd":
        qvel[:, probe.joint_qvel] = _KD_PROBE_SPEED * direction
    elif channel == "mass":
        ctrl[:, probe.joint_names.index("left_hip_pitch_joint")] += 0.18
        ctrl[:, probe.joint_names.index("right_hip_pitch_joint")] += 0.09
    else:
        ctrl += 0.1 * direction
    steps = 12 if channel == "mass" else 6

    def run(rows, requested_channel, levels):
        return probe.run(
            qpos, qvel, ctrl, steps, rows, requested_channel, np.asarray(levels, dtype=np.float32)
        )

    nominal = run(_ROWS, "combined", np.zeros(_NUM_ENVS))
    nominal_repeat = run(_ROWS, "combined", np.zeros(_NUM_ENVS))
    _assert_close(nominal_repeat, nominal, f"{channel}: repeated nominal trajectory")
    noise = np.abs(nominal_repeat - nominal)
    low = run(_DR_ROWS, channel, [-1.0, -1.0])
    high = run(_DR_ROWS, channel, [1.0, 1.0])
    effects = {
        "low_vs_nominal": _require_effect(probe, channel, low, nominal, noise),
        "high_vs_nominal": _require_effect(probe, channel, high, nominal, noise),
        "high_vs_low": _require_effect(probe, channel, high, low, noise),
        "nominal_repeat_max_error": float(noise.max()),
    }
    for label, trajectory in (("low", low), ("high", high)):
        _assert_close(
            trajectory[:, _CONTROL_ROWS],
            nominal[:, _CONTROL_ROWS],
            f"{channel}/{label}: untouched rows must remain physically nominal",
        )

    for levels in ([-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]):
        mixed = run(_DR_ROWS, channel, levels)
        expected = nominal.copy()
        for row, level in zip(_DR_ROWS, levels):
            expected[:, row] = (low if level < 0 else high)[:, row]
        _assert_close(mixed, expected, f"{channel}: sparse-row assignment and no compounding")

    expected = nominal.copy()
    expected[:, _DR_ROWS[1]] = low[:, _DR_ROWS[1]]
    for _ in range(2):
        partial_restore = run(_DR_ROWS[:1], "combined", [0.0])
        _assert_close(
            partial_restore, expected, f"{channel}: restoring one row preserves the other's DR"
        )
    for _ in range(2):
        restored = run(_DR_ROWS, "combined", [0.0, 0.0])
        _assert_close(restored, nominal, f"{channel}: repeated explicit nominal restoration")
    return effects


def _contact_trial(
    probe: _G1Probe, qpos: np.ndarray, lifted: np.ndarray, steps: int
) -> list[list[int]]:
    probe.backend.set_state(
        _ROWS,
        qpos.copy(),
        np.tile(probe.zero_qvel, (_NUM_ENVS, 1)),
        randomization=probe.payload("combined", np.zeros(_NUM_ENVS)),
    )
    foot_ids = probe.backend.get_body_ids(["left_ankle_roll_link", "right_ankle_roll_link"])
    ctrl = qpos[:, probe.joint_qpos].copy()
    hits = np.zeros((_NUM_ENVS, 2), dtype=np.int32)
    for step in range(steps):
        probe.backend.step(ctrl.copy(), nsteps=1)
        _state(probe.backend)
        contacts = np.asarray(probe.backend.get_sensor_data_batch(_FOOT_SENSORS))
        assert contacts.shape == (_NUM_ENVS, 8)
        assert np.isfinite(contacts).all()
        assert ((contacts >= 0.0) & (contacts <= 1.0)).all(), "contact-found must be boolean"
        if step < 2:
            continue
        touching = (contacts.reshape(_NUM_ENVS, 2, 4) > 0.5).any(axis=2)
        positions = probe.backend.get_body_pos_w(foot_ids)
        assert positions.shape == (_NUM_ENVS, 2, 3) and np.isfinite(positions).all()
        assert (positions[:, :, 2][lifted] > 0.05).all(), "lifted foot lost physical clearance"
        assert not touching[lifted].any(), (
            "airborne foot reports stale, global, or synthetic contact"
        )
        hits += touching
    assert (hits[~lifted] >= 2).all(), (
        f"grounded feet must produce real contact in multiple samples; hits={hits.tolist()}"
    )
    return hits.tolist()


def _check_contacts(probe: _G1Probe) -> dict[str, list[list[int]]]:
    results = {}
    for label, rows in (("mixed_height", _CONTROL_ROWS), ("swapped_height", _DR_ROWS)):
        qpos = np.tile(probe.stand_qpos, (_NUM_ENVS, 1))
        qpos[rows, probe.root_qpos[2]] += 2.0
        lifted = np.zeros((_NUM_ENVS, 2), dtype=bool)
        lifted[rows] = True
        results[label] = _contact_trial(probe, qpos, lifted, steps=40)

    qpos = np.tile(probe.stand_qpos, (_NUM_ENVS, 1))
    lifted = np.zeros((_NUM_ENVS, 2), dtype=bool)
    for side_index, side in enumerate(("left", "right")):
        rows = _CONTROL_ROWS if side_index == 0 else _DR_ROWS
        lifted[rows, side_index] = True
        for joint, offset in (("hip_pitch", -0.35), ("knee", 0.70), ("ankle_pitch", -0.35)):
            column = probe.joint_qpos[probe.joint_names.index(f"{side}_{joint}_joint")]
            qpos[rows, column] += offset
    results["single_foot_same_root_height"] = _contact_trial(probe, qpos, lifted, steps=10)
    return results


@pytest.mark.parametrize("backend_name", _BACKENDS, ids=_BACKENDS)
def test_g1_multisim_physics(backend_name: str, request: pytest.FixtureRequest) -> None:
    from unilab.assets import ASSETS_ROOT_PATH
    from unilab.base.backend_factory import create_backend
    from unilab.base.scene import SceneCfg

    def evidence(name: str, value) -> None:
        request.node.user_properties.append((name, json.dumps(value, sort_keys=True)))

    evidence("backend", backend_name)
    evidence("live_parameter_readback", "unavailable; public mass/gain tables are nominal only")
    evidence(
        "inertia_limitation", "no public live inertia getter; unchanged native inertia unverified"
    )
    scene_file = ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"
    evidence("scene", str(scene_file))
    options = {
        "isaacgym": {"isaacgym_device_id": 0, "isaacgym_worker_timeout_s": 120.0},
        "isaacsim": {
            "isaacsim_device_id": 0,
            "isaacsim_worker_timeout_s": 120.0,
            "isaacsim_render_mode": "none",
        },
        "motrix": {},
        "genesis": {"genesis_device_id": 0, "genesis_integrator": "implicitfast"},
    }
    backend = None
    try:
        backend = create_backend(
            backend_name,
            SceneCfg(model_file=str(scene_file), default_keyframe_name="stand"),
            _NUM_ENVS,
            _SIM_DT,
            base_name="pelvis",
            **options[backend_name],
        )
        backend.materialize()
        probe = _prepare_probe(backend, scene_file.with_name("g1.xml"))
        for channel in ("mass", "kp", "kd", "combined"):
            evidence(channel, _check_randomization(probe, channel))
        evidence("contacts", _check_contacts(probe))
        evidence("public_physics_checks_complete", True)
    finally:
        if backend is not None:
            backend.cleanup_scene_assets()
