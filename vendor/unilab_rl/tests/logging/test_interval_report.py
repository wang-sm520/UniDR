"""Interval boundaries, missing metrics, shared clocks, and raw scalar fidelity."""

import csv
import json

import pytest
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from uni_rl.logging import interval_report as interval

PLOTS = interval._plots


def _events(path, iterations=8, extra=()):
    rows = [
        (tag, i, value)
        for i in range(iterations)
        for tag, value in (
            ("Perf/collection_time", 1.0),
            ("Perf/learning_time", 0.25),
            *((tag, 0.1) for tag in interval.PPO),
            ("reward/motion_body_pos", 2.5),
        )
    ]
    # Seconds-axis records have duplicate steps and must not enter interval indexing.
    rows.extend(("Train/mean_reward/time", 1, float(i)) for i in range(iterations))
    rows.extend(("Train/mean_reward", i, float(i)) for i in range(1, iterations))
    rows.extend(("Train/mean_episode_length", i, 500.0) for i in range(1, iterations))
    rows.extend(("Metrics/motion/error_body_pos", i, 0.2) for i in range(1, iterations, 2))
    rows.extend(extra)
    writer = EventFileWriter(str(path))
    for tag, step, value in rows:
        writer.add_event(
            Event(
                step=step,
                wall_time=1000 + 2 * step,
                summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]),
            )
        )
    writer.close()


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root, joint = tmp_path / "single", tmp_path / "joint"
    for source in interval.SOURCE_ORDER:
        _events(root / source)
        for i in (0, 2, 4, 6, 7):
            (root / source / f"model_{i}.pt").write_bytes(b"fixture")
    _events(joint, 2)
    rows = [
        dict(
            iteration=i,
            collect_seconds=4.0,
            learn_seconds=0.5,
            sources={
                source: dict(
                    episode_return=None if i == 0 else 10.0,
                    episode_length=None if i == 0 else 500.0,
                    reward=0.25,
                    terminated=2,
                    truncated=3,
                    env_step_seconds=2.0,
                    request_response_seconds=3.0,
                    barrier_wait_seconds=0.5,
                )
                for source in interval.SOURCE_ORDER
            },
        )
        for i in range(2)
    ]
    (joint / "synchronous_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (joint / "model_1.pt").write_bytes(b"fixture")
    monkeypatch.setattr(
        interval, "audit_single_run", lambda run, **kw: dict(source=run.name, total_transitions=384)
    )
    monkeypatch.setattr(
        interval,
        "_read_audited",
        lambda *args: (dict(total_transitions=384, timing=dict(missing_iterations=0)), rows),
    )
    monkeypatch.setattr(interval, "_plots", lambda *args: None)
    return root, joint, tmp_path / "output"


def test_complete_intervals_and_original_metric_semantics(experiment):
    root, joint, output = experiment
    result = interval.report(
        root, joint, output, interval=2, single_iterations=8, joint_iterations=2, num_envs=2
    )
    assert len(result["intervals"]) == 17
    assert len(result["source_intervals"]) == 20
    native = [r for r in result["intervals"] if r["policy"] == "motrix"]
    assert [(r["first_completed_update"], r["last_completed_update"]) for r in native] == [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
    ]
    assert [r["loop_seconds"] for r in native] == [2.5] * 4
    assert native[-1]["cumulative_loop_seconds"] == 10
    assert native[-1]["cumulative_transitions"] == 384
    assert native[-1]["cumulative_optimizer_steps"] == 160
    assert native[0]["event_boundary_elapsed_seconds"] is None
    assert native[1]["event_boundary_elapsed_seconds"] == 4
    assert native[0]["episode_return_mean"] == 1.0
    assert native[0]["episode_return_observed_updates"] == 1
    shared = [r for r in result["intervals"] if r["policy"] == "joint"]
    assert len(shared) == 1 and shared[0]["loop_seconds"] == 9
    assert shared[0]["cumulative_transitions"] == 384
    source = next(r for r in result["source_intervals"] if r["policy"] == "joint")
    assert source["interval_source_transitions"] == 96
    assert source["terminated_sum"] == 4 and source["truncated_sum"] == 6
    assert source["true_termination_flags_per_transition"] == 4 / 96
    assert source["env_step_seconds_sum"] == 4
    assert source["error_body_pos_mean"] is None
    assert source["error_body_pos_observed_updates"] == 0
    clocks = [r for r in result["checkpoint_clocks"] if r["policy"] == "motrix"]
    assert clocks[1]["iteration"] == 2 and clocks[1]["completed_updates"] == 3
    assert clocks[1]["cumulative_loop_seconds"] == 3.75
    with (output / "scalar_intervals.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(not r["tag"].endswith("/time") for r in rows)
    motion = next(r for r in rows if r["tag"] == "Metrics/motion/error_body_pos")
    assert motion["observed_updates"] == "1" and motion["missing_updates"] == "1"
    assert float(motion["mean"]) == pytest.approx(0.2)
    assert result["metric_inventory"]["motrix"]["Train/mean_reward/time"]["included"] is False
    assert json.loads((output / "intervals.json").read_text())["interval_updates"] == 2


def test_partial_tail_and_missing_values_keep_true_denominator():
    bounds = list(interval._bounds(5, 2))
    assert bounds[-1] == (
        4,
        5,
        dict(
            first_completed_update=5,
            last_completed_update=5,
            first_iteration=4,
            last_iteration=4,
            updates=1,
        ),
    )
    stat = interval._stats([None, 2.0, None, 4.0], 0, 4)
    assert stat == dict(
        observed_updates=2,
        missing_updates=2,
        mean=3.0,
        std=1.0,
        min=2.0,
        p95=3.9,
        max=4.0,
        last=4.0,
    )
    assert interval._stats([None], 0, 1)["mean"] is None


@pytest.mark.parametrize("bad", [0, -2, True, 1.5])
def test_invalid_interval_rejected_without_output(experiment, bad):
    root, joint, output = experiment
    with pytest.raises(ValueError, match="interval"):
        interval.report(root, joint, output, interval=bad)
    assert not output.exists()


def test_existing_output_preserved(experiment):
    root, joint, output = experiment
    output.mkdir()
    (output / "user.txt").write_text("preserve")
    with pytest.raises(ValueError, match="must be new"):
        interval.report(root, joint, output)
    assert (output / "user.txt").read_text() == "preserve"


@pytest.mark.parametrize(
    "extra", [[("Loss/value", 0, 1.0)], [("extra", 9, 1.0)], [("extra", 0, float("nan"))]]
)
def test_event_corruption_rejected(tmp_path, extra):
    _events(tmp_path, extra=extra)
    with pytest.raises(ValueError):
        interval._read_events(tmp_path, 8)


def test_full_plot_generation(experiment, monkeypatch):
    pytest.importorskip("matplotlib")
    root, joint, output = experiment
    result = interval.report(
        root, joint, output, interval=2, single_iterations=8, joint_iterations=2, num_envs=2
    )
    PLOTS(output, result["intervals"], result["source_intervals"])
    for name in ("timing_ppo", "episodes", "tracking"):
        assert (output / f"{name}.png").stat().st_size > 100
        assert (output / f"{name}.pdf").stat().st_size > 100
