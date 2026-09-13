"""Opt-in Genesis 1.3.3 native reset replay; run in a fresh process.

UNISIM_RUN_GENESIS_RESET=1 enables this case. Set CUDA_VISIBLE_DEVICES='' for
the CPU diagnostic lane; the same test runs on CUDA when explicitly scheduled.
All writes use SimBackend; native tables are assertion-only evidence.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.genesis.backend import GenesisBackend
from unisim.dr.types import ResetRandomizationPayload
from unisim.scene import SceneCfg

pytestmark = pytest.mark.skipif(
    os.environ.get("UNISIM_RUN_GENESIS_RESET") != "1",
    reason="native Genesis reset replay requires UNISIM_RUN_GENESIS_RESET=1",
)


def test_native_mass_replay_has_fresh_dynamics_and_unchanged_inertia() -> None:
    from genesis.utils.misc import qd_to_numpy

    backend = GenesisBackend(
        SceneCfg(
            model_file=str(Path(__file__).parent / "assets/genesis_mass_replay.xml"),
            default_keyframe_name="init",
        ),
        4,
        1.0 / 150.0,
        base_name="base",
        integrator="implicitfast",
    )
    try:
        backend.materialize()
        entity = backend.model
        rows = np.arange(4, dtype=np.int32)
        selected = np.array([1, 3], dtype=np.int32)
        base = backend.get_body_id("base")
        nominal = np.tile(backend.get_body_mass(), (4, 1))
        active = nominal.copy()
        qpos = np.tile(backend.get_keyframe_qpos("init"), (4, 1)).astype(np.float32)
        qpos[:, 0] = [0.0, 0.03, -0.02, 0.05]
        qvel = np.tile(backend.get_init_qvel(), (4, 1)).astype(np.float32)
        control = np.full((4, 1), 0.18, dtype=np.float32)
        inertia = qd_to_numpy(entity.solver.dyn_info.links.inertial_i, transpose=True, copy=True)

        def state() -> np.ndarray:
            values = backend.get_state(("qpos", "qvel"))
            return np.concatenate((values["qpos"], values["qvel"]), axis=1).copy()

        def assert_parameters() -> None:
            np.testing.assert_allclose(
                entity.get_links_inertial_mass().cpu().numpy(), active, rtol=1e-6, atol=1e-7
            )
            np.testing.assert_array_equal(
                qd_to_numpy(entity.solver.dyn_info.links.inertial_i, transpose=True, copy=True),
                inertia,
            )
            cached = qd_to_numpy(entity.solver.dyn_state.links.cinr_mass, transpose=True, copy=True)
            np.testing.assert_allclose(
                cached,
                active,
                rtol=1e-6,
                atol=1e-7,
                err_msg="mass reset must refresh spatial inertia before the next physics substep",
            )

        def replay(reset_rows: np.ndarray, factors: list[float]) -> np.ndarray:
            backend.set_state(rows, qpos, qvel, ResetRandomizationPayload(body_mass=active.copy()))
            before = state()
            active[reset_rows, base] = nominal[reset_rows, base] * factors
            backend.set_state(
                reset_rows,
                qpos[reset_rows],
                qvel[reset_rows],
                ResetRandomizationPayload(body_mass=active[reset_rows].copy()),
            )
            untouched = np.setdiff1d(rows, reset_rows)
            np.testing.assert_array_equal(state()[untouched], before[untouched])
            assert_parameters()
            trajectory = []
            for _ in range(12):
                backend.step(control)
                trajectory.append(state())
                assert_parameters()
            return np.stack(trajectory)

        reference = replay(rows, [1.0] * 4)
        repeat = replay(rows, [1.0] * 4)
        np.testing.assert_allclose(repeat, reference, rtol=5e-4, atol=2e-5)
        low = replay(selected, [0.8, 0.8])
        high = replay(selected, [1.2, 1.2])
        assert np.abs(low[:, selected] - high[:, selected]).max() > 5e-6
        for factors in ([0.8, 1.2], [0.8, 1.2], [1.2, 0.8]):
            actual = replay(selected, factors)
            expected = reference.copy()
            for row, factor in zip(selected, factors, strict=True):
                expected[:, row] = (low if factor < 1.0 else high)[:, row]
            np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=2e-5)
        expected = reference.copy()
        expected[:, selected[1]] = low[:, selected[1]]
        for _ in range(2):
            actual = replay(selected[:1], [1.0])
            np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=2e-5)
        for _ in range(2):
            actual = replay(selected, [1.0, 1.0])
            np.testing.assert_allclose(actual, reference, rtol=5e-4, atol=2e-5)
    finally:
        backend.cleanup_scene_assets()
