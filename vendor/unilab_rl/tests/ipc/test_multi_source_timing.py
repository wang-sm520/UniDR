from __future__ import annotations

from uni_rl.ipc._multi_source_worker import _accumulate_environment_timings


def test_internal_reset_timings_are_accumulated_without_task_knowledge() -> None:
    statistics = {}
    _accumulate_environment_timings(
        statistics, {"timing": {"reset_done_reset_call_ms": 2.5, "reset_done_count": 2}}
    )
    _accumulate_environment_timings(
        statistics, {"timing": {"reset_done_reset_call_ms": 1.5, "reset_done_count": 1}}
    )
    assert statistics["timing/reset_done_reset_call_ms/total"] == 4.0
    assert statistics["timing/reset_done_count/total"] == 3.0
    assert statistics["timing/reset_done_count/last"] == 1.0
