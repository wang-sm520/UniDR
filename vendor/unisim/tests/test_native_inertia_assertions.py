"""Static world representation must not weaken robot or invariance assertions."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_multisim_native_readback import _assert_inertia, _NativeTables


def _nominal_tables():
    return _NativeTables(
        mass=np.array([[np.finfo(np.float32).eps, 3.813]]),
        kp=np.ones((1, 29)),
        kd=np.ones((1, 29)),
        inertia=np.array([[np.zeros((3, 3)), np.eye(3)]]),
    )


def test_static_world_may_have_zero_inertia():
    nominal = _nominal_tables()
    _assert_inertia(nominal, nominal, "baseline", np.array([False, True]))


def test_zero_robot_inertia_is_rejected():
    nominal = replace(_nominal_tables(), inertia=np.zeros((1, 2, 3, 3)))
    with pytest.raises(AssertionError, match="Massive G1 links"):
        _assert_inertia(nominal, nominal, "baseline", np.array([False, True]))


def test_static_world_inertia_changes_are_still_rejected():
    nominal = _nominal_tables()
    changed = nominal.inertia.copy()
    changed[:, 0] += 0.001
    with pytest.raises(AssertionError, match="native inertia changed"):
        _assert_inertia(
            replace(nominal, inertia=changed), nominal, "reset", np.array([False, True])
        )


def test_missing_live_reader_is_still_rejected():
    nominal = replace(_nominal_tables(), inertia=None)
    with pytest.raises(AssertionError, match="Required live native inertia"):
        _assert_inertia(nominal, nominal, "reset", np.array([False, True]))
