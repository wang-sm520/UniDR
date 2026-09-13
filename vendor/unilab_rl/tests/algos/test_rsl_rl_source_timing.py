from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from uni_rl.algos.rsl_rl_source_timing import record_source_timing


@pytest.fixture
def timing_run():
    stats = {
        name: {
            "num_envs": count,
            "step_calls": 10,
            "transitions": count * 10,
            "last_step_seconds": 0.0,
            "timing/reset_done_count/total": 100.0,
            "timing/reset_done_reset_call_ms/total": 1000.0,
            "pid": 100 + count,
            "rss_bytes": 4096,
        }
        for name, count in (("left", 2), ("right", 3))
    }
    logger = SimpleNamespace(writer=Mock(), log=Mock(), process_env_step=Mock())
    runner = SimpleNamespace(
        logger=logger, cfg={"num_steps_per_env": 2}, env=SimpleNamespace(num_envs=5)
    )

    def step(left_seconds, right_seconds):
        for name, seconds in (("left", left_seconds), ("right", right_seconds)):
            stats[name]["step_calls"] += 1
            stats[name]["transitions"] += stats[name]["num_envs"]
            stats[name]["last_step_seconds"] = seconds
            stats[name]["timing/reset_done_count/total"] += 1.0
            stats[name]["timing/reset_done_reset_call_ms/total"] += 3.0
        runner.logger.process_env_step("rewards", "dones", {"log": {}}, None)

    return runner, logger, stats, step


def test_records_actual_step_samples_and_rollout_deltas(tmp_path, timing_run, capsys):
    runner, logger, stats, step = timing_run
    output = tmp_path / "source_timing.jsonl"
    with record_source_timing(runner, output_path=output, source_statistics=lambda: stats):
        assert runner.logger.writer is logger.writer
        for iteration in (73, 74):
            step(0.010, 0.030)
            step(0.040, 0.020)
            runner.logger.log(it=iteration, collect_time=0.080, learn_time=0.005)
            assert len(output.read_text().splitlines()) == iteration - 72
    assert runner.logger is logger
    assert logger.process_env_step.call_count == 4
    assert logger.log.call_count == 2
    records = [json.loads(line) for line in output.read_text().splitlines()]
    for record in records:
        assert record["source_order"] == ["left", "right"]
        assert record["max_source_step_seconds_sum"] == pytest.approx(0.070)
        assert record["collection_minus_max_source_step_seconds"] == pytest.approx(0.010)
        for name, count, samples in (("left", 2, [0.010, 0.040]), ("right", 3, [0.030, 0.020])):
            source = record["sources"][name]
            assert source["transitions"] == count * 2
            assert source["step_calls"] == 2
            assert source["step_seconds"] == samples
            assert source["step_seconds_sum"] == pytest.approx(0.050)
            assert source["step_seconds_mean"] == pytest.approx(0.025)
            assert source["step_seconds_p50"] == pytest.approx(0.025)
            assert source["slowest_steps"] == 1
            assert source["environment_timing_totals"] == {
                "reset_done_count": 2.0,
                "reset_done_reset_call_ms": 6.0,
            }
        assert record["sources"]["left"]["step_seconds_p95"] == pytest.approx(0.0385)
    logger.writer.add_scalar.assert_any_call("source/left/sampling/step_seconds_mean", 0.025, 73)
    logger.writer.add_scalar.assert_any_call(
        "source/left/timing/reset_done_count/rollout_sum", 2.0, 74
    )
    assert "left: 25.0 ms/step (p95 38.5), 0.050 s/rollout" in capsys.readouterr().out


def test_no_writer_still_records_json(tmp_path, timing_run):
    runner, logger, stats, step = timing_run
    logger.writer = None
    output = tmp_path / "timing.jsonl"
    with record_source_timing(runner, output_path=output, source_statistics=lambda: stats):
        step(0.01, 0.02)
        step(0.01, 0.02)
        runner.logger.log(it=0, collect_time=0.1, learn_time=0.01)
    assert len(output.read_text().splitlines()) == 1


def test_partial_rollout_on_error_is_not_reported(tmp_path, timing_run):
    runner, logger, stats, step = timing_run
    output = tmp_path / "timing.jsonl"
    with pytest.raises(RuntimeError, match="source failed"):
        with record_source_timing(runner, output_path=output, source_statistics=lambda: stats):
            step(0.01, 0.02)
            raise RuntimeError("source failed")
    assert runner.logger is logger
    assert output.read_text() == ""


def test_incomplete_update_is_rejected(tmp_path, timing_run):
    runner, logger, stats, step = timing_run
    with pytest.raises(RuntimeError, match="incomplete PPO rollout"):
        with record_source_timing(
            runner, output_path=tmp_path / "timing.jsonl", source_statistics=lambda: stats
        ):
            step(0.01, 0.02)
            runner.logger.log(it=0, collect_time=0.1, learn_time=0.01)
    assert runner.logger is logger
    logger.log.assert_not_called()


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf")])
def test_invalid_step_duration_is_rejected(tmp_path, timing_run, seconds):
    runner, logger, stats, step = timing_run
    with pytest.raises(ValueError, match="invalid step duration"):
        with record_source_timing(
            runner, output_path=tmp_path / "timing.jsonl", source_statistics=lambda: stats
        ):
            step(seconds, 0.02)
    assert runner.logger is logger


@pytest.mark.parametrize("field", ["step_calls", "transitions", "num_envs"])
def test_stale_or_missing_source_step_is_rejected(tmp_path, timing_run, field):
    runner, logger, stats, step = timing_run
    with pytest.raises(RuntimeError, match="exactly one full step"):
        with record_source_timing(
            runner, output_path=tmp_path / "timing.jsonl", source_statistics=lambda: stats
        ):
            stats["left"][field] += 1
            step(0.01, 0.02)
    assert runner.logger is logger


def test_file_is_never_overwritten(tmp_path, timing_run):
    runner, logger, stats, _step = timing_run
    output = tmp_path / "timing.jsonl"
    output.write_text("previous evidence\n")
    with pytest.raises(FileExistsError):
        with record_source_timing(runner, output_path=output, source_statistics=lambda: stats):
            pytest.fail("Must not enter")
    assert runner.logger is logger
    assert output.read_text() == "previous evidence\n"


def test_signed_environment_timer_is_preserved(tmp_path, timing_run):
    runner, _logger, stats, step = timing_run
    output = tmp_path / "timing.jsonl"
    with record_source_timing(runner, output_path=output, source_statistics=lambda: stats):
        stats["left"]["timing/internal_gap_ms/total"] = -0.1
        step(0.01, 0.02)
        step(0.01, 0.02)
        runner.logger.log(it=0, collect_time=0.1, learn_time=0.01)
    record = json.loads(output.read_text())
    assert record["sources"]["left"]["environment_timing_totals"]["internal_gap_ms"] == -0.1


def test_distributed_runner_rejected(tmp_path, timing_run):
    runner, _logger, stats, _step = timing_run
    runner.is_distributed = True
    with pytest.raises(ValueError, match="single learner"):
        with record_source_timing(
            runner, output_path=tmp_path / "timing.jsonl", source_statistics=lambda: stats
        ):
            pytest.fail("Must not enter")
