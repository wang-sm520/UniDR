"""Genesis reset must refresh mass-dependent caches after absolute DR writes.

The SDK double models Genesis 1.3.3's setter semantics: mass writes do not
invalidate forward caches, while set_qpos recomputes spatial inertia. The
opt-in native counterpart verifies these semantics against the real solver.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.genesis.backend import GenesisBackend
from unisim.dr.types import ResetRandomizationPayload


class _Entity:
    def __init__(self, mass: np.ndarray) -> None:
        self.mass = mass.copy()
        self.forward_mass = mass.copy()
        self.velocity_mass = mass.copy()
        self.qpos = np.zeros((4, 8), dtype=np.float32)
        self.qvel = np.zeros((4, 7), dtype=np.float32)
        self.kp = np.full((4, 1), 20.0, dtype=np.float32)
        self.kd = np.full((4, 1), 2.0, dtype=np.float32)
        self.calls: list[str] = []

    def set_links_inertial_mass(self, mass, *, envs_idx):
        self.calls.append("mass")
        self.mass[envs_idx] = mass

    def set_qpos(self, qpos, *, envs_idx, zero_velocity):
        assert zero_velocity is False
        self.calls.append("qpos")
        self.qpos[envs_idx] = qpos
        self.forward_mass[envs_idx] = self.mass[envs_idx]

    def set_dofs_velocity(self, qvel, *, envs_idx):
        self.calls.append("qvel")
        self.qvel[envs_idx] = qvel
        self.velocity_mass[envs_idx] = self.forward_mass[envs_idx]

    def set_dofs_kp(self, gains, *, dofs_idx_local, envs_idx):
        assert dofs_idx_local == [6]
        self.calls.append("kp")
        self.kp[envs_idx] = gains

    def set_dofs_kv(self, gains, *, dofs_idx_local, envs_idx):
        assert dofs_idx_local == [6]
        self.calls.append("kd")
        self.kd[envs_idx] = gains


@pytest.fixture
def backend() -> GenesisBackend:
    backend = GenesisBackend.__new__(GenesisBackend)
    nominal = np.array([0.0, 5.0, 2.0], dtype=np.float32)
    backend._metadata = SimpleNamespace(
        nq=8,
        nv=7,
        nbody=3,
        body_mass=nominal,
        actuator_names=("hinge_drive",),
        actuator_kp=np.array([20.0], dtype=np.float32),
        actuator_kv=np.array([2.0], dtype=np.float32),
    )
    backend._entity = _Entity(np.tile(nominal, (4, 1)))
    backend._base_link_idx = 1
    backend._actuated_dofs = [6]
    backend._num_envs = 4
    backend._closed = False
    backend._materialized = True
    backend._time_cache = np.ones(4, dtype=np.float32)
    backend._to_device = lambda values: values.copy()
    backend._refresh_host_cache = lambda: backend._entity.calls.append("refresh")
    return backend


@pytest.mark.parametrize("rows", (np.array([1, 3]), np.array([3, 1])))
@pytest.mark.parametrize("mass_mode", ("body_mass", "base_mass_delta", "both"))
def test_sparse_mass_reset_refreshes_dynamics_without_compounding(backend, rows, mass_mode):
    entity = backend.model
    nominal = np.tile(backend.get_body_mass(), (4, 1))
    expected = nominal.copy()
    original_qpos = entity.qpos.copy()
    original_qvel = entity.qvel.copy()
    qpos = np.arange(16, dtype=np.float32).reshape(2, 8)
    qvel = np.arange(14, dtype=np.float32).reshape(2, 7)
    untouched = np.setdiff1d(np.arange(4), rows)
    for factors in ([0.8, 1.2], [0.8, 1.2], [1.2, 0.8], [1.0, 1.0], [1.0, 1.0]):
        expected[rows, 1] = nominal[rows, 1] * factors
        if mass_mode == "body_mass":
            payload = ResetRandomizationPayload(body_mass=expected[rows].copy())
        else:
            payload = ResetRandomizationPayload(
                body_mass=nominal[rows].copy() if mass_mode == "both" else None,
                base_mass_delta=expected[rows, 1] - nominal[rows, 1],
            )
        backend.set_state(rows, qpos, qvel, payload)
        for values in (entity.mass, entity.forward_mass, entity.velocity_mass):
            np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(entity.qpos[rows], qpos)
        np.testing.assert_array_equal(entity.qvel[rows], qvel)
        np.testing.assert_array_equal(entity.qpos[untouched], original_qpos[untouched])
        np.testing.assert_array_equal(entity.qvel[untouched], original_qvel[untouched])
        np.testing.assert_array_equal(backend.get_body_mass(), nominal[0])
        np.testing.assert_array_equal(backend._time_cache[rows], 0.0)
        np.testing.assert_array_equal(backend._time_cache[untouched], 1.0)


def test_combined_reset_keeps_absolute_gains_and_nominal_tables(backend):
    entity = backend.model
    rows = np.array([3, 1])
    nominal_kp, nominal_kd = backend.get_actuator_gains()
    payload = ResetRandomizationPayload(
        body_mass=entity.mass[rows] * [[1.0, 0.8, 1.0], [1.0, 1.2, 1.0]],
        kp=nominal_kp[None, :] * [[0.9], [1.1]],
        kd=nominal_kd[None, :] * [[1.1], [0.9]],
    )
    for _ in range(2):
        backend.set_state(rows, entity.qpos[rows], entity.qvel[rows], payload)
        np.testing.assert_array_equal(entity.forward_mass, entity.mass)
        np.testing.assert_allclose(entity.kp[rows], payload.kp)
        np.testing.assert_allclose(entity.kd[rows], payload.kd)
        np.testing.assert_array_equal(entity.kp[[0, 2]], np.tile(nominal_kp, (2, 1)))
        np.testing.assert_array_equal(entity.kd[[0, 2]], np.tile(nominal_kd, (2, 1)))
        np.testing.assert_array_equal(backend.get_actuator_gains(), (nominal_kp, nominal_kd))


@pytest.mark.parametrize("payload", (None, ResetRandomizationPayload()))
def test_plain_reset_preserves_existing_parameters(backend, payload):
    entity = backend.model
    entity.mass[1, 1] = 4.0
    backend.set_state(np.array([1]), entity.qpos[[1]], entity.qvel[[1]], payload)
    np.testing.assert_array_equal(entity.forward_mass[1], entity.mass[1])
    assert entity.calls == ["qpos", "qvel", "refresh"]


def test_empty_reset_does_not_write_or_refresh(backend):
    backend.set_state(
        np.array([], dtype=np.int32),
        np.empty((0, 8)),
        np.empty((0, 7)),
        ResetRandomizationPayload(body_mass=np.empty((0, 3))),
    )
    assert backend.model.calls == []
