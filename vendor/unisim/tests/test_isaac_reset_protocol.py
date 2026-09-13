"""CPU coverage of the versioned host boundary and selected native worker writes."""

from __future__ import annotations

import ast
import copy
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from isaac_worker_fakes import (
    fill_reset,
    make_worker,
    nominal_metadata,
    physical_properties,
)

from unisim.backend.isaacgym.backend import IsaacGymBackend
from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.backend.isaacsim.worker import _contact_sensor_path
from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.backend import SubprocessWorkerError
from unisim.dr.types import (
    RESET_TERM_BODY_MASS,
    RESET_TERM_KD,
    RESET_TERM_KP,
    ResetRandomizationPayload,
)
from unisim.scene import SceneCfg

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(params=("isaacgym", "isaacsim"))
def worker_case(request):
    return request.param, make_worker(request.param)


def test_shared_protocol_v2_layout_and_framing():
    assert set(protocol.RESET_TERMS) == {RESET_TERM_BODY_MASS, RESET_TERM_KP, RESET_TERM_KD}
    assert protocol.PROTOCOL_VERSION == 2
    shapes = protocol.slot_shapes(3, 2, 4)
    assert shapes["reset_body_mass"] == (3, 4)
    assert shapes["reset_kp"] == shapes["reset_kd"] == (3, 2)
    for name in ("reset_body_mass", "reset_kp", "reset_kd"):
        assert protocol.slot_dtype(name) == np.float32
    stream = io.BytesIO()
    protocol.send_message(
        stream,
        protocol.CMD_SET_STATE,
        {
            "count": 2,
            "randomization_terms": list(protocol.RESET_TERMS),
        },
    )
    assert stream.getvalue()[8:10] == b"\x80\x04"
    stream.seek(0)
    assert protocol.recv_message(stream)["payload"]["randomization_terms"] == list(
        protocol.RESET_TERMS
    )


@pytest.mark.parametrize("version", (None, 1, 3, "2", True, 2.0))
def test_version_fails_closed_before_sdk_import(worker_case, version):
    _, worker = worker_case
    with pytest.raises(ValueError, match="incompatible subprocess protocol"):
        worker.init_sim({"protocol_version": version})


def test_worker_and_protocol_remain_python38_parseable_and_lazy():
    root = Path(protocol.__file__).parents[1]
    for path in (
        Path(protocol.__file__),
        root / "isaacgym" / "worker.py",
        Path(__file__).with_name("isaac_native_readback_worker.py"),
    ):
        ast.parse(path.read_text(), feature_version=(3, 8))
    code = (
        "import sys, unisim; import unisim.backend.isaacgym.worker; "
        "import unisim.backend.isaacsim.worker; "
        "assert not any(name.split('.')[0] in "
        "{'torch','isaacgym','isaacsim','isaaclab','omni','pxr','unilab','uni_rl'} "
        "for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_selected_absolute_writes_mapping_inertia_and_nominal_cache(worker_case):
    kind, worker = worker_case
    before = physical_properties(worker, kind)
    nominal = nominal_metadata(worker)
    contacts = worker.slots["contact_force"].copy()
    state = worker.slots["root_state"].copy()
    request = fill_reset(worker)
    worker.set_state(request)
    after = physical_properties(worker, kind)
    for term, field, mapping in (
        ("body_mass", "mass", worker.native_body_for_contract),
        ("kp", "kp", worker.native_joint_for_contract),
        ("kd", "kd", worker.native_joint_for_contract),
    ):
        np.testing.assert_array_equal(after[field][1], before[field][1])
        np.testing.assert_array_equal(
            after[field][[2, 0]][:, mapping], worker.slots["reset_" + term][:2]
        )
    np.testing.assert_array_equal(after["inertia"], before["inertia"])
    np.testing.assert_array_equal(worker.slots["root_state"][1], state[1])
    np.testing.assert_allclose(worker.slots["root_state"][[2, 0], :3], [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_allclose(worker.slots["dof_state"][[2, 0], :, 0], [[0.4, 0.8], [0.4, 0.8]])
    np.testing.assert_array_equal(worker.slots["contact_force"][1], contacts[1])
    assert not worker.slots["contact_force"][[2, 0]].any()
    worker.refresh_state_slots()
    assert not worker.slots["contact_force"][[2, 0]].any()
    worker.set_state(request)
    for field, value in after.items():
        np.testing.assert_array_equal(physical_properties(worker, kind)[field], value)
    assert worker.get_meta()["nominal_body_mass"] == [5, 2, 3]
    assert worker._reset_metadata == nominal
    assert worker.get_meta()["nominal_kp"] == [20, 10]
    if kind == "isaacgym":
        assert worker.gym.reads == 6
        assert {call[1] for call in worker.gym.calls if call[0] == "mass"} == {0, 2}
        np.testing.assert_array_equal(worker.gym.properties["effort"], [[70, 80]] * 3)
    else:
        assert worker.robot.root_physx_view.reads == 1
        for rows in worker.robot.root_physx_view.calls:
            np.testing.assert_array_equal(rows, [2, 0])
        np.testing.assert_array_equal(worker.robot.actuators["all"].stiffness, after["kp"])
        np.testing.assert_array_equal(worker.robot.actuators["all"].damping, after["kd"])


def test_omitted_terms_and_empty_reset_do_not_reapply_stale_slots(worker_case):
    kind, worker = worker_case
    worker.set_state(fill_reset(worker))
    before = physical_properties(worker, kind)
    request = fill_reset(worker, rows=(1,), terms=("kd",))
    worker.slots["reset_body_mass"][:] = np.nan
    worker.slots["reset_kp"][:] = np.nan
    worker.set_state(request)
    after = physical_properties(worker, kind)
    for field in ("mass", "kp", "inertia"):
        np.testing.assert_array_equal(after[field], before[field])
    worker.set_state({"count": 0, "randomization_terms": []})
    for field, value in after.items():
        np.testing.assert_array_equal(physical_properties(worker, kind)[field], value)


def test_zero_gains_disable_selected_drives_without_restoring_other_rows(worker_case):
    kind, worker = worker_case
    before = physical_properties(worker, kind)
    request = fill_reset(worker, rows=(2,), terms=("kp", "kd"))
    worker.slots["reset_kp"][:1] = 0
    worker.slots["reset_kd"][:1] = 0
    worker.set_state(request)
    after = physical_properties(worker, kind)
    for field in ("kp", "kd"):
        np.testing.assert_array_equal(after[field][:2], before[field][:2])
        assert not after[field][2].any()
    np.testing.assert_array_equal(after["mass"], before["mass"])


def test_zero_step_cannot_revalidate_stale_reset_contacts(worker_case):
    _, worker = worker_case
    worker.set_state(fill_reset(worker))
    with pytest.raises(ValueError, match="nsteps must be positive"):
        worker.step({"nsteps": 0})
    worker.refresh_state_slots()
    assert not worker.slots["contact_force"][[2, 0]].any()


@pytest.mark.parametrize(
    "fault",
    (
        "negative_mass",
        "zero_mass",
        "negative_kp",
        "nan_kd",
        "bad_qpos",
        "bad_quat",
        "duplicate_rows",
        "out_of_range",
        "negative_count",
        "float_count",
        "unsupported",
        "missing_terms",
    ),
)
def test_complete_transaction_validated_before_native_mutation(worker_case, fault):
    kind, worker = worker_case
    request = fill_reset(worker)
    if fault == "negative_mass":
        worker.slots["reset_body_mass"][1, 2] = -1
    elif fault == "zero_mass":
        worker.slots["reset_body_mass"][1, 2] = 0
    elif fault == "negative_kp":
        worker.slots["reset_kp"][1, 0] = -1
    elif fault == "nan_kd":
        worker.slots["reset_kd"][1, 1] = np.nan
    elif fault == "bad_qpos":
        worker.slots["reset_qpos"][1, 0] = np.inf
    elif fault == "bad_quat":
        worker.slots["reset_qpos"][1, 3:7] = 0
    elif fault == "duplicate_rows":
        worker.slots["reset_env_ids"][:2] = 1
    elif fault == "out_of_range":
        worker.slots["reset_env_ids"][1] = 3
    elif fault == "negative_count":
        request["count"] = -1
    elif fault == "float_count":
        request["count"] = 1.5
    elif fault == "unsupported":
        request["randomization_terms"].append("gravity")
    else:
        del request["randomization_terms"]
    before = physical_properties(worker, kind)
    with pytest.raises(ValueError):
        worker.set_state(request)
    for field, value in before.items():
        np.testing.assert_array_equal(physical_properties(worker, kind)[field], value)
    if kind == "isaacgym":
        assert worker.gym.calls == []
    else:
        assert worker.robot.calls == []
        assert worker.robot.root_physx_view.calls == []


@pytest.mark.parametrize("term,value", (("body_mass", 0), ("kp", -1), ("kd", np.nan)))
def test_reset_value_validation_rejects_bad_shape_and_values(term, value):
    with pytest.raises(ValueError):
        protocol.reset_values([[value]], (1, 1), term)
    for data in ([1], [[1, 2]], [[1j]], [["1"]], [[True]], [[1e300]]):
        with pytest.raises(ValueError):
            protocol.reset_values(data, (1, 1), term)


@pytest.mark.parametrize("rows", ([0.5], [True], [[0]], [3], [-1], [0, 0]))
def test_reset_row_validation(rows):
    with pytest.raises(ValueError):
        protocol.validate_reset_state(
            rows, np.zeros((len(rows), 9)), np.zeros((len(rows), 8)), 3, 2
        )


@pytest.mark.parametrize("field", ("mass", "gains"))
def test_gym_native_rejection_is_not_acknowledged(field):
    worker = make_worker("isaacgym")
    worker.gym.reject = field
    with pytest.raises(RuntimeError, match="IsaacGym rejected"):
        worker.set_state(fill_reset(worker))
    assert not any(call[0] == "root" for call in worker.gym.calls)


def test_contacts_and_controls_use_independent_cached_name_maps(worker_case):
    kind, worker = worker_case
    worker.slots["ctrl"][:] = [0.5, 0.9]
    worker.step({"nsteps": 2})
    if kind == "isaacgym":
        targets = worker.gym.targets
        forces = np.asarray(worker._contact_force)[:, worker.native_body_for_contract]
    else:
        targets = worker.robot.targets
        forces = worker.contact_sensor.forces[:, worker.contact_body_for_contract]
    np.testing.assert_allclose(targets, [[0.9, 0.5]] * 3)
    np.testing.assert_array_equal(worker.slots["contact_force"], forces)


@pytest.mark.parametrize("fault", ("missing", "uninitialized", "names", "shape", "nan"))
def test_isaacsim_contact_reporter_fails_closed(fault):
    worker = make_worker("isaacsim")
    if fault == "missing":
        worker.contact_sensor = None
    elif fault == "uninitialized":
        worker.contact_sensor.is_initialized = False
    elif fault == "names":
        worker.contact_sensor.body_names = ["base", "leg", "leg"]
    elif fault == "shape":
        worker.contact_sensor.data.net_forces_w.array = np.zeros((3, 2, 3))
    else:
        worker.contact_sensor.data.net_forces_w.array[0, 0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError)):
        if fault in ("shape", "nan"):
            worker.refresh_state_slots()
        else:
            worker._bind_contact_reporter()


def test_contact_path_is_discovered_cold_and_not_assumed(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "pxr", SimpleNamespace(UsdPhysics=SimpleNamespace(RigidBodyAPI=1))
    )
    paths = ["/World/envs/env_0/Robot/nested/links/" + name for name in ("foot", "base", "leg")]
    utilities = SimpleNamespace(
        get_all_matching_child_prims=lambda *args, **kwargs: [
            SimpleNamespace(GetPath=lambda path=path: path) for path in paths
        ]
    )
    actual = _contact_sensor_path(utilities, "/World/envs/env_0/Robot", ["base", "leg", "foot"])
    assert actual == "/World/envs/env_.*/Robot/nested/links/(base|leg|foot)"
    paths[0] = "/World/envs/env_0/Robot/elsewhere/foot"
    with pytest.raises(RuntimeError, match="one imported parent"):
        _contact_sensor_path(utilities, "/World/envs/env_0/Robot", ["base", "leg", "foot"])


def test_attach_rejects_incompatible_slots_before_opening_shm(worker_case):
    _, worker = worker_case
    specs = {
        name: {"shape": shape, "dtype": str(protocol.slot_dtype(name)), "shm": "not-opened"}
        for name, shape in protocol.slot_shapes(3, 2, 3).items()
    }
    payload = {"protocol_version": 2, "slots": specs}
    protocol.validate_slot_specs(payload, 3, 2, 3)
    for fault in ("missing", "shape", "dtype"):
        malformed = copy.deepcopy(payload)
        if fault == "missing":
            del malformed["slots"]["reset_kp"]
        else:
            malformed["slots"]["reset_kp"][fault] = (3, 5) if fault == "shape" else "float64"
        with pytest.raises(ValueError, match="shared-memory"):
            worker.attach_slots(malformed)


def make_host(kind, mode="normal"):
    adapter = IsaacGymBackend if kind == "isaacgym" else IsaacSimBackend
    kwargs = {"render_mode": "none"} if kind == "isaacsim" else {}
    return adapter(
        scene=SceneCfg(model_file=str(ASSETS / "isaac_reset.xml")),
        num_envs=3,
        sim_dt=0.002,
        base_name="base",
        worker_timeout_s=15,
        worker_command=[
            sys.executable,
            str(Path(__file__).with_name("isaac_protocol_worker.py")),
            "--kind",
            kind,
            "--mode",
            mode,
        ],
        **kwargs,
    )


@pytest.mark.parametrize("kind", ("isaacgym", "isaacsim"))
def test_real_pipe_shm_public_reset_and_nominal_metadata(kind, monkeypatch):
    backend = make_host(kind)
    try:
        assert backend.get_dr_capabilities().supported_reset_terms == frozenset(
            protocol.RESET_TERMS
        )
        masses = backend.get_body_mass()
        np.testing.assert_array_equal(masses, [5, 2, 3])
        masses[:] = 999
        np.testing.assert_array_equal(backend.get_body_mass(), [5, 2, 3])
        monkeypatch.setattr(
            "unisim.backend.subprocess_ipc.backend.scan_scene_metadata",
            lambda *args, **kwargs: pytest.fail("hot-path XML scan"),
        )
        qpos = np.tile(backend.get_default_qpos(), (2, 1))
        qpos[:, 0] = [1, 2]
        qvel = np.zeros((2, 8), dtype=np.float32)
        payload = ResetRandomizationPayload(
            body_mass=np.array([[11, 12, 13], [14, 15, 16]]),
            kp=np.array([[30, 40], [50, 60]]),
            kd=np.array([[3, 4], [5, 6]]),
        )
        for _ in range(2):
            backend.set_state(np.array([2, 0]), qpos, qvel, payload)
            np.testing.assert_array_equal(backend.get_base_pos()[[2, 0], 0], [1, 2])
            np.testing.assert_array_equal(backend.get_body_mass(), [5, 2, 3])
            assert not backend.get_sensor_data("foot_contact")[[2, 0]].any()
        backend.step(np.zeros((3, 2)))
        assert backend.get_sensor_data("foot_contact").all()
        backend.set_state(np.array([1]), qpos[:1], qvel[:1])
        np.testing.assert_array_equal(
            backend.get_sensor_data("foot_contact").reshape(-1), [1, 0, 1]
        )
        before = {name: array.copy() for name, array in backend._slots.items()}
        with pytest.raises(ValueError, match="nonnegative"):
            backend.set_state(
                np.array([2, 0]), qpos, qvel, ResetRandomizationPayload(kd=-payload.kd)
            )
        with pytest.raises(NotImplementedError, match="gravity"):
            backend.set_state(
                np.array([2, 0]), qpos, qvel, ResetRandomizationPayload(gravity=np.zeros((2, 3)))
            )
        for name, value in before.items():
            np.testing.assert_array_equal(backend._slots[name], value)
    finally:
        backend.close()
    assert not backend._slots


@pytest.mark.parametrize("kind", ("isaacgym", "isaacsim"))
@pytest.mark.parametrize(
    "mode,match",
    (
        ("old", "incompatible subprocess protocol"),
        ("unsupported", "required reset terms"),
        ("bad_nominal", "invalid worker nominal metadata"),
        ("wrong_reporter", "contact_reporter does not match"),
    ),
)
def test_incompatible_worker_fails_materialization_and_cleans_up(kind, mode, match):
    backend = make_host(kind, mode)
    with pytest.raises(SubprocessWorkerError, match=match):
        backend.materialize()
    assert backend._closed
    assert not backend._slots
    assert backend._proc is None
    with pytest.raises(SubprocessWorkerError, match="closed"):
        backend.get_dr_capabilities()


@pytest.mark.parametrize("kind", ("isaacgym", "isaacsim"))
def test_host_contact_declarations_require_reporter_handshake(kind):
    backend = make_host(kind, "no_contact")
    try:
        backend.materialize()
        with pytest.raises(NotImplementedError, match="did not declare a contact reporter"):
            backend.get_sensor_data("foot_contact")
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ("isaacgym", "isaacsim"))
def test_native_reset_error_poisons_host_state_until_close(kind):
    backend = make_host(kind, "reject_mass")
    try:
        backend.materialize()
        with pytest.raises(SubprocessWorkerError, match="rejected"):
            backend.set_state(
                np.array([1]),
                backend.get_default_qpos()[None],
                np.zeros((1, 8)),
                ResetRandomizationPayload(body_mass=np.ones((1, 3))),
            )
        with pytest.raises(SubprocessWorkerError, match="earlier failure"):
            backend.step(np.zeros((3, 2)))
        with pytest.raises(SubprocessWorkerError, match="earlier failure"):
            backend.get_base_pos()
    finally:
        backend.close()


@pytest.mark.parametrize(
    "native,contract",
    (
        (["one", "two"], ["one", "one"]),
        (["one", "one"], ["one", "two"]),
        (["one", "two"], ["one", "three"]),
        (["one"], ["one", "two"]),
    ),
)
def test_name_mapping_fails_closed(native, contract):
    with pytest.raises(ValueError, match="mapping mismatch"):
        protocol.name_permutation(native, contract, "body")
