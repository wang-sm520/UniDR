"""Required native G1 parameter integrity checks, separate from physical effects.

Opt in with UNILAB_RUN_MULTISIM_PHYSICS=1. UNILAB_G1_SCENE may point to an
absolute, already materialized G1 scene; otherwise the sibling UniLab G1 scene
is used without importing UniLab, downloading assets, or replacing the robot.
Each case launches this file in a fresh process, like the Isaac readback lane.

Genesis initializes the fixed world link at gs.EPS mass rather than canonical
zero. That initial value is checked explicitly; selected resets are still
checked against the requested zero. Only canonical massive robot links require
nonzero inertia, while inertia invariance is checked for every native link.

Assertion-only readers inspect native state; all writes use SimBackend.set_state.
Genesis 1.3.3 exposes live inertia through solver.dyn_info.links.inertial_i and
genesis.utils.misc.qd_to_numpy. Motrix 0.8.2 exposes live mass/KP/KD overrides,
but its low.pyi only exposes inertia on LowSceneModel, not LowData. That model
table cannot certify per-environment inertia preservation: the Motrix case
fails its required inertia assertion until an actual live reader is available.
No missing native capability is a passing property, fallback, skip, or xfail.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

_BACKENDS = ("motrix", "genesis")
_NUM_ENVS = 4
_NUM_JOINTS = 29
_ROWS = np.arange(_NUM_ENVS, dtype=np.int32)
_SELECTED = np.array([1, 3], dtype=np.int32)
_RESULT_PREFIX = "UNISIM_MULTISIM_READBACK_RESULT "
_MOTRIX_INERTIA_ERROR = (
    "Required Motrix live per-environment inertia readback is unavailable: "
    "motrixsim 0.8.2 Link and SceneData.low/LowData expose no inertia getter. "
    "LowSceneModel.link_inertias is model-only and is NOT evidence that reset "
    "mass overrides leave native per-environment inertia unchanged. An SDK "
    "live inertia reader and an assertion-only binding are required before "
    "this case can pass; nominal tables or force-response effects cannot replace it."
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.optional,
    pytest.mark.skipif(
        os.environ.get("UNILAB_RUN_MULTISIM_PHYSICS") != "1",
        reason="native G1 readback requires UNILAB_RUN_MULTISIM_PHYSICS=1",
    ),
]


@dataclass(frozen=True)
class _NativeTables:
    mass: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    inertia: np.ndarray | None
    inertia_error: str = "Required live native inertia readback is unavailable"


def _scene_path() -> Path:
    configured = os.environ.get("UNILAB_G1_SCENE")
    scene = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[2]
        / "UniLab/src/unilab/assets/robots/g1/scene_flat.xml"
    )
    assert scene.is_absolute(), "UNILAB_G1_SCENE must be an absolute materialized G1 scene path"
    assert scene.is_file(), f"G1 scene is missing: {scene}; configure UNILAB_G1_SCENE"
    return scene.resolve()


def _column(value, label: str) -> np.ndarray:
    assert value is not None, f"Required native {label} readback returned None"
    values = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    assert values.shape == (_NUM_ENVS,), f"{label} must expose every environment, not nominal data"
    assert np.isfinite(values).all(), f"Native {label} contains non-finite values"
    return values


def _motrix_tables(backend) -> _NativeTables:
    """Read installed _ffi.pyi Link/PositionActuator override APIs, not owner caches."""
    model = backend.model
    data = backend._data
    links = sorted(model.links, key=lambda link: int(link.index))
    np.testing.assert_array_equal([link.index for link in links], np.arange(len(links)))
    actuators = [model.get_actuator(name) for name in backend.get_actuator_names()]
    assert len(actuators) == _NUM_JOINTS and all(actuator is not None for actuator in actuators)
    assert all(actuator.typ == "position" for actuator in actuators)
    assert (
        tuple(actuator.target_name for actuator in actuators) == backend.get_actuator_joint_names()
    )
    return _NativeTables(
        mass=np.stack(
            [_column(link.get_mass_override(data), f"{link.name} mass") for link in links], axis=1
        ),
        kp=np.stack(
            [_column(actuator.get_kp_override(data), "KP") for actuator in actuators], axis=1
        ),
        kd=np.stack(
            [_column(actuator.get_kd_override(data), "KD") for actuator in actuators], axis=1
        ),
        inertia=None,
        inertia_error=_MOTRIX_INERTIA_ERROR,
    )


def _genesis_tables(backend) -> _NativeTables:
    """Read the live batched solver tensor used by rigid/abd/accessor.py kernels."""
    from genesis.utils.misc import qd_to_numpy

    entity = backend._entity
    links = entity.links
    np.testing.assert_array_equal([link.idx_local for link in links], np.arange(len(links)))
    contract_ids = backend.get_body_ids([link.name for link in links])
    np.testing.assert_array_equal(np.sort(contract_ids), np.arange(len(links)))
    native_order = np.argsort(contract_ids)
    dofs = []
    for name in backend.get_actuator_joint_names():
        joint = entity.get_joint(name)
        assert joint.n_dofs == 1, f"G1 actuator target {name} is not single-DoF"
        dofs.append(int(joint.dofs_idx_local[0]))
    solver = entity.solver
    assert hasattr(solver, "dyn_info") and hasattr(solver.dyn_info.links, "inertial_i"), (
        "Required Genesis live inertia tensor dyn_info.links.inertial_i is unavailable; "
        "link metadata or owner nominal caches are not acceptable substitutes"
    )
    inertia = qd_to_numpy(
        solver.dyn_info.links.inertial_i,
        col_mask=(entity.link_start + native_order).tolist(),
        transpose=True,
        copy=True,
    )
    return _NativeTables(
        mass=entity.get_links_inertial_mass().detach().cpu().numpy()[:, native_order].copy(),
        kp=entity.get_dofs_kp(dofs_idx_local=dofs).detach().cpu().numpy().copy(),
        kd=entity.get_dofs_kv(dofs_idx_local=dofs).detach().cpu().numpy().copy(),
        inertia=np.asarray(inertia).copy(),
    )


def _assert_parameters(actual: _NativeTables, expected: dict[str, np.ndarray], label: str) -> None:
    for name in ("mass", "kp", "kd"):
        values = getattr(actual, name)
        assert values.shape == expected[name].shape, f"{label}: invalid native {name} shape"
        assert np.isfinite(values).all(), f"{label}: non-finite native {name}"
        np.testing.assert_allclose(
            values, expected[name], rtol=1e-5, atol=1e-7, err_msg=f"{label}: live {name}"
        )


def _assert_inertia(
    actual: _NativeTables,
    nominal: _NativeTables,
    label: str,
    massive_bodies: np.ndarray,
) -> None:
    assert nominal.inertia is not None, nominal.inertia_error
    assert actual.inertia is not None, actual.inertia_error
    shape = (*nominal.mass.shape, 3, 3)
    assert nominal.inertia.shape == actual.inertia.shape == shape, (
        f"{label}: inertia must be live, batched, full per-link tensors with shape {shape}"
    )
    assert np.isfinite(nominal.inertia).all() and np.isfinite(actual.inertia).all()
    assert (np.linalg.norm(nominal.inertia, axis=(-2, -1))[:, massive_bodies] > 0).all(), (
        "Massive G1 links must have nonzero native inertia, not placeholder zero tables"
    )
    np.testing.assert_allclose(
        actual.inertia,
        nominal.inertia,
        rtol=1e-6,
        atol=1e-9,
        err_msg=f"{label}: native inertia changed, including selected or untouched rows/links",
    )


def _assert_no_unilab_import() -> None:
    assert not any(name == "unilab" or name.startswith("unilab.") for name in sys.modules), (
        "The unisim native probe must not import UniLab"
    )


def _exercise_readback(backend, kind: str) -> None:
    from unisim.dr.types import ResetRandomizationPayload

    assert backend.num_envs == _NUM_ENVS
    assert backend.num_actuators == backend.num_dof_vel == _NUM_JOINTS
    missing = backend.get_dr_capabilities().get_unsupported_reset_terms(
        frozenset({"body_mass", "kp", "kd"})
    )
    assert not missing, f"Required G1 reset capabilities are missing: {sorted(missing)}"
    joint_names = backend.get_actuator_joint_names()
    assert len(joint_names) == len(set(joint_names)) == _NUM_JOINTS
    assert {"left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint"} <= set(joint_names)
    pelvis_id = backend.get_body_id("pelvis")
    mass = np.asarray(backend.get_body_mass()).copy()
    kp, kd = (np.asarray(value).copy() for value in backend.get_actuator_gains())
    assert mass.ndim == 1 and kp.shape == kd.shape == (_NUM_JOINTS,)
    assert (kp > 0).all() and (kd > 0).all()
    np.testing.assert_allclose(mass[pelvis_id], 3.813, rtol=1e-5, err_msg="G1 nominal pelvis mass")
    expected = {
        name: np.tile(values, (_NUM_ENVS, 1))
        for name, values in (("mass", mass), ("kp", kp), ("kd", kd))
    }
    nominal_values = {name: values.copy() for name, values in expected.items()}
    reader = _motrix_tables if kind == "motrix" else _genesis_tables
    nominal = reader(backend)
    if kind == "genesis":
        import genesis as gs

        world_id = backend.get_body_id("world")
        assert mass[world_id] == 0 and backend._entity.get_link("world").is_fixed
        expected["mass"][:, world_id] = float(gs.EPS)
    _assert_parameters(nominal, expected, "materialized G1 nominal parameters")

    stand = np.asarray(backend.get_keyframe_qpos("stand"), dtype=np.float32)
    zero = np.asarray(backend.get_init_qvel(), dtype=np.float32)
    assert stand.shape == (7 + _NUM_JOINTS,) and zero.shape == (6 + _NUM_JOINTS,)
    qpos = np.tile(stand, (_NUM_ENVS, 1))
    root = backend.get_root_state_layout("pelvis")
    qpos[:, root.qpos_indices[0]] += np.array([0.0, 0.03, -0.02, 0.05])
    qpos[:, root.qpos_indices[2]] += 2.0
    qvel = np.tile(zero, (_NUM_ENVS, 1))
    ctrl = qpos[:, backend.get_joint_state_qpos_indices(joint_names)].copy()

    def reset(rows: np.ndarray, values: dict[str, np.ndarray], label: str) -> None:
        before = {name: value.copy() for name, value in backend.get_state().items()}
        for name in expected:
            expected[name][rows] = values[name]
        payload = ResetRandomizationPayload(
            body_mass=values["mass"].copy(), kp=values["kp"].copy(), kd=values["kd"].copy()
        )
        assert payload.body_inertia is payload.body_iquat is payload.body_ipos is None
        backend.set_state(rows, qpos[rows].copy(), qvel[rows].copy(), randomization=payload)
        after = backend.get_state()
        untouched = np.setdiff1d(_ROWS, rows)
        for name in before:
            np.testing.assert_allclose(
                after[name][untouched], before[name][untouched], rtol=1e-5, atol=2e-5
            )
        current = reader(backend)
        _assert_parameters(current, expected, label)
        _assert_inertia(current, nominal, label, mass > 0)
        np.testing.assert_array_equal(backend.get_body_mass(), mass)
        for current_gains, original in zip(backend.get_actuator_gains(), (kp, kd)):
            np.testing.assert_array_equal(current_gains, original)
        backend.step(ctrl.copy(), nsteps=1)
        stepped = reader(backend)
        _assert_parameters(stepped, expected, f"{label}, after physics substep")
        _assert_inertia(stepped, nominal, f"{label}, after physics substep", mass > 0)

    backend.set_state(_ROWS, qpos.copy(), qvel.copy())
    changed = {name: values[_SELECTED].copy() for name, values in nominal_values.items()}
    changed["mass"][:, pelvis_id] = mass[pelvis_id] * np.array([0.8, 1.2])
    low_scales = np.linspace(0.9, 0.99, _NUM_JOINTS, dtype=np.float32)
    high_scales = 2.0 - low_scales
    changed["kp"] *= np.stack((low_scales, high_scales))
    changed["kd"] *= np.stack((high_scales, low_scales))
    for repetition in range(2):
        reset(_SELECTED, changed, f"selected G1 mass/KP/KD endpoints, repetition {repetition}")
    reversed_values = {name: values[::-1].copy() for name, values in changed.items()}
    reset(_SELECTED, reversed_values, "reversed selected-row values relative to nominal")
    for rows in (_SELECTED[:1], _SELECTED[1:]):
        for repetition in range(2):
            reset(
                rows,
                {name: values[rows].copy() for name, values in nominal_values.items()},
                f"restore row {rows.tolist()} to nominal, repetition {repetition}",
            )
    _assert_parameters(reader(backend), expected, "restored robot parameters and untouched world")
    _assert_no_unilab_import()


def _require_genesis_gpu() -> None:
    utility = shutil.which("nvidia-smi")
    assert utility is not None, "Genesis native readback requires nvidia-smi and a working CUDA GPU"
    result = subprocess.run(
        [utility, "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        "Genesis native readback BLOCKED: " + result.stdout + result.stderr
    )
    import torch

    assert torch.cuda.is_available(), (
        "Genesis native readback requires CUDA Torch; CPU fallback is forbidden"
    )


def _run_native(kind: str, scene: Path) -> dict:
    assert os.environ.get("UNILAB_RUN_MULTISIM_PHYSICS") == "1", (
        "Explicit native readback opt-in required"
    )
    assert scene.is_absolute() and scene.is_file(), f"Materialized G1 scene is missing: {scene}"
    _assert_no_unilab_import()
    if kind == "genesis":
        _require_genesis_gpu()
    from unisim import create_backend
    from unisim.scene import SceneCfg

    options = (
        {"genesis_integrator": "implicitfast", "genesis_device_id": 0} if kind == "genesis" else {}
    )
    backend = None
    try:
        backend = create_backend(
            kind,
            SceneCfg(model_file=str(scene), default_keyframe_name="stand"),
            _NUM_ENVS,
            1.0 / 150.0,
            base_name="pelvis",
            **options,
        )
        backend.materialize()
        _exercise_readback(backend, kind)
    finally:
        if backend is not None:
            if kind == "genesis":
                backend.close()
            backend.cleanup_scene_assets()
    return {
        "backend": kind,
        "sdk_version": version("motrixsim-core" if kind == "motrix" else "genesis-world"),
        "scene": str(scene),
        "num_envs": _NUM_ENVS,
        "num_joints": _NUM_JOINTS,
        "status": "passed",
        "native_inertia_verified": True,
        "physical_effects_validated": False,
    }


@pytest.mark.parametrize("kind", _BACKENDS, ids=_BACKENDS)
def test_native_selected_property_readback(kind: str, request: pytest.FixtureRequest) -> None:
    scene = _scene_path()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--backend", kind, "--scene", str(scene)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-16000:]
    records = [
        json.loads(line[len(_RESULT_PREFIX) :])
        for line in result.stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    assert len(records) == 1, "Native child must emit exactly one completed readback result"
    record = records[0]
    assert record["backend"] == kind and record["scene"] == str(scene)
    assert record["num_envs"] == _NUM_ENVS and record["num_joints"] == _NUM_JOINTS
    assert record["status"] == "passed" and record["native_inertia_verified"] is True
    assert record["physical_effects_validated"] is False
    request.node.user_properties.append(("native_readback", json.dumps(record, sort_keys=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=_BACKENDS, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    arguments = parser.parse_args()
    print(_RESULT_PREFIX + json.dumps(_run_native(arguments.backend, arguments.scene)), flush=True)
