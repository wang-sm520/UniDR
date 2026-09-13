"""CPU regression for the IsaacGym GPU tensor submission lifecycle."""

from __future__ import annotations

import weakref

import numpy as np
import pytest
from isaac_worker_fakes import fill_reset, make_worker

_REFRESH = (
    "refresh_actor_root_state_tensor",
    "refresh_dof_state_tensor",
    "refresh_rigid_body_state_tensor",
    "refresh_net_contact_force_tensor",
)
_SETTERS = ("set_actor_root_state_tensor_indexed", "set_dof_state_tensor_indexed")


@pytest.mark.parametrize("initial_keyframe", (False, True), ids=("reset", "init-and-reset"))
def test_resets_coalesce_and_refresh_only_after_simulation(monkeypatch, initial_keyframe):
    worker = make_worker("isaacgym")
    events = []
    submitted = []
    index_refs = []

    def trace(name):
        original = getattr(worker.gym, name)

        def call(*args):
            events.append(name)
            if name in _SETTERS:
                _, values, indices, count = args
                index_refs.append(weakref.ref(indices))
                submitted.append((name, np.asarray(values).copy(), np.asarray(indices).copy()))
                assert np.asarray(indices).dtype == np.int32
                assert len(np.asarray(indices)) == count
            if name == "simulate" and index_refs:
                assert all(reference() is not None for reference in index_refs)
                assert index_refs[0]() is index_refs[1]() is worker._reset_actor_indices
            return original(*args)

        monkeypatch.setattr(worker.gym, name, call)

    for name in (
        *_REFRESH,
        *_SETTERS,
        "set_dof_position_target_tensor",
        "simulate",
        "fetch_results",
    ):
        trace(name)

    worker._tensors_need_refresh = True
    worker.refresh_state_slots()
    worker.refresh_state_slots()
    assert events == list(_REFRESH)
    events.clear()

    if initial_keyframe:
        worker.dof_names = ["knee", "hip"]
        worker._apply_initial_keyframe([1, 2, 3, 1, 0, 0, 0, 0.2, 0.3], worker.contract_joint_names)
        worker.refresh_state_slots()
    before = worker.slots["root_state"].copy()
    for env_id, position in ((2, 20.0), (0, 10.0), (2, 22.0)):
        request = fill_reset(worker, rows=(env_id,))
        worker.slots["reset_qpos"][0, 0] = position
        worker.set_state(request)
        worker.refresh_state_slots()
        assert worker.slots["root_state"][env_id, 0] == position
        np.testing.assert_array_equal(worker.slots["root_state"][1], before[1])
        assert events == []

    expected_root = np.asarray(worker._root_state).copy()
    expected_dof = np.asarray(worker._dof_state).copy()
    worker.step({"nsteps": 1})
    assert events == [
        *_SETTERS,
        "set_dof_position_target_tensor",
        "simulate",
        "fetch_results",
        *_REFRESH,
    ]
    assert not worker._pending_reset_env_ids
    expected_indices = [0, 1, 2] if initial_keyframe else [0, 2]
    for (_, values, indices), expected in zip(submitted, (expected_root, expected_dof)):
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(expected_root[[0, 2], 0], [10.0, 22.0])

    events.clear()
    worker.refresh_state_slots()
    worker.refresh_state_slots()
    assert events == []
    worker.step({"nsteps": 2})
    assert events == [
        "set_dof_position_target_tensor",
        "simulate",
        "fetch_results",
        "simulate",
        "fetch_results",
        *_REFRESH,
    ]
