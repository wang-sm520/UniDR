"""IsaacLab stop callbacks must not re-enter rendering during worker shutdown."""

from __future__ import annotations

from types import SimpleNamespace

from unisim.backend.isaacsim.worker import _WorkerContext
from unisim.backend.subprocess_ipc import protocol


def test_shutdown_clears_simulation_callbacks_before_closing_kit():
    calls = []
    worker = _WorkerContext(protocol)
    worker.sim = SimpleNamespace(
        clear_all_callbacks=lambda: calls.append("clear_callbacks"),
        clear_instance=lambda: calls.append("clear_instance"),
    )
    worker.simulation_app = SimpleNamespace(close=lambda: calls.append("close_kit"))
    worker.robot = object()
    worker._gain_actuators = [object()]
    worker._mass_table = object()

    worker.shutdown()
    worker.shutdown()

    assert calls == ["clear_callbacks", "clear_instance", "close_kit"]
    assert worker.sim is None and worker.simulation_app is None
    assert worker.robot is None and worker._mass_table is None
    assert not worker._gain_actuators


def test_shutdown_before_simulation_construction_still_closes_kit():
    calls = []
    worker = _WorkerContext(protocol)
    worker.simulation_app = SimpleNamespace(close=lambda: calls.append("close_kit"))
    worker.shutdown()
    worker.shutdown()
    assert calls == ["close_kit"]
