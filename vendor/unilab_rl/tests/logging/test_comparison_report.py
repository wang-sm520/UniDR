"""Report semantics use real event files; checkpoint audits have separate tensor tests."""

import csv
import json
import math
from pathlib import Path

import pytest
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from uni_rl.logging import comparison_report as comparison


def write_events(path, records):
    writer = EventFileWriter(str(path))
    for tag, step, value in records:
        writer.add_event(
            Event(step=step, summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]))
        )
    writer.close()


def records(iterations=8):
    return [
        (tag, i, value)
        for i in range(iterations)
        for tag, value in (
            ("Perf/collection_time", 1.0),
            ("Perf/learning_time", 0.25),
            *(
                (("Train/mean_reward", float(i)), ("Train/mean_episode_length", 500.0))
                if i > 0
                else ()
            ),
        )
    ]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root, joint = tmp_path / "singles", tmp_path / "joint"
    root.mkdir()
    joint.mkdir()
    calls = []
    for source in comparison.SOURCE_ORDER:
        write_events(root / source, records())
    rows = [
        {
            "iteration": i,
            "collect_seconds": 4.0,
            "learn_seconds": 0.5,
            "sources": {
                name: {
                    "episode_return": None if i == 0 else 10.0 + n,
                    "episode_length": None if i == 0 else 400.0 + n,
                }
                for n, name in enumerate(comparison.SOURCE_ORDER)
            },
        }
        for i in range(2)
    ]
    (joint / "synchronous_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (joint / "run_config.json").write_text(
        json.dumps(
            {
                "config": {
                    "algo": {
                        "resume": False,
                        "resume_path": None,
                        "load_run": "-1",
                    }
                }
            }
        )
    )
    joint_audit = {
        "generations": [0],
        "history": [{"resume_checkpoint": None}],
        "total_transitions": 384,
        "recorded_training_seconds": 9.0,
        "timing": {"timed_iterations": 2, "missing_iterations": 0},
    }

    def audit_native(run, *, expected_iterations, num_envs):
        calls.append(("single", run.name, expected_iterations, num_envs))
        return {"source": run.name, "total_transitions": 384}

    def audit_joint(path, iterations, num_envs, final):
        calls.append(("joint", path.name, iterations, num_envs, final))
        return joint_audit, rows

    monkeypatch.setattr(comparison, "audit_single_run", audit_native)
    monkeypatch.setattr(comparison, "_read_audited", audit_joint)
    return root, joint, tmp_path / "output", calls, joint_audit


def test_equal_samples_missing_metrics_and_one_shared_clock(experiment):
    pytest.importorskip("matplotlib")
    root, joint, output, calls, _ = experiment
    result = comparison.report(
        root, joint, output, single_iterations=8, joint_iterations=2, num_envs=2
    )
    assert calls == [("single", s, 8, 2) for s in comparison.SOURCE_ORDER] + [
        ("joint", "joint", 2, 2, None)
    ]
    assert result["loop_seconds"] == {**{s: 10.0 for s in comparison.SOURCE_ORDER}, "joint": 9.0}
    assert result["recent_training_metrics"]["isaacsim"]["single"]["episode_return"] == {
        "observed_updates": 7,
        "mean": 4.0,
    }
    assert result["recent_training_metrics"]["isaacsim"]["joint"]["episode_return"]["mean"] == 10.0
    assert (
        result["single_updates_per_smoothing_window"]
        == 4 * result["joint_updates_per_smoothing_window"]
    )
    with (output / "series.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    assert {int(r["global_transitions"]) for r in rows if r["iteration"] == "7"} == {384}
    assert {
        int(r["global_transitions"])
        for r in rows
        if r["policy"] == "joint" and r["iteration"] == "1"
    } == {384}
    assert rows[0]["episode_return"] == ""  # Missing does not become a zero return.
    assert {
        int(r["source_transitions"])
        for r in rows
        if r["policy"] == "joint" and r["iteration"] == "1"
    } == {96}
    with (output / "loop_time.csv").open() as stream:
        times = list(csv.DictReader(stream))
    shared = [r for r in times if r["policy"] == "joint"]
    assert len(shared) == 2 and sum(float(r["loop_seconds"]) for r in shared) == 9
    assert float(shared[-1]["cumulative_loop_seconds"]) == 9
    for name in (
        "comparison.png",
        "comparison.pdf",
        "loop_time.png",
        "loop_time.pdf",
        "comparison.json",
    ):
        assert (output / name).stat().st_size > 100


@pytest.mark.parametrize(
    "fault", ["source", "budget", "resume", "timing", "iterations", "nonempty", "audit"]
)
def test_report_rejects_invalid_experiment_before_output(experiment, monkeypatch, fault):
    root, joint, output, _, audit = experiment
    single_iterations = 8
    if fault in ("source", "budget"):
        monkeypatch.setattr(
            comparison,
            "audit_single_run",
            lambda run, **kw: {
                "source": "motrix" if fault == "source" else run.name,
                "total_transitions": 383 if fault == "budget" else 384,
            },
        )
    elif fault == "resume":
        audit["generations"] = [1]
    elif fault == "timing":
        audit["timing"]["missing_iterations"] = 1
    elif fault == "iterations":
        single_iterations = 7
    elif fault == "nonempty":
        output.mkdir()
        (output / "user.txt").write_text("preserved")
    else:

        def fail(*args, **kwargs):
            raise ValueError("training did not complete")

        monkeypatch.setattr(comparison, "audit_single_run", fail)
    with pytest.raises(ValueError):
        comparison.report(
            root, joint, output, single_iterations=single_iterations, joint_iterations=2, num_envs=2
        )
    assert not (output / "comparison.json").exists()
    if fault == "nonempty":
        assert (output / "user.txt").read_text() == "preserved"


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "nan", "late", "missing_time", "negative_time", "unpaired", "negative_length"],
)
def test_native_event_corruption(tmp_path, fault):
    rows = records()
    if fault == "duplicate":
        rows.append(("Train/mean_reward", 1, 5.0))
    elif fault == "nan":
        rows.append(("Train/mean_reward", 0, float("nan")))
    elif fault == "late":
        rows.append(("Train/mean_reward", 8, 3.0))
    elif fault == "missing_time":
        rows = [r for r in rows if r[:2] != ("Perf/collection_time", 2)]
    elif fault == "negative_time":
        rows = [(t, i, -1.0 if (t, i) == ("Perf/collection_time", 2) else v) for t, i, v in rows]
    elif fault == "unpaired":
        rows = [r for r in rows if r[:2] != ("Train/mean_episode_length", 3)]
    else:
        rows = [
            (t, i, -1.0 if (t, i) == ("Train/mean_episode_length", 2) else v) for t, i, v in rows
        ]
    write_events(tmp_path, rows)
    with pytest.raises(ValueError):
        comparison._native_series(tmp_path, 8)


def test_smoothing_uses_iteration_width_without_filling_missing():
    result = comparison._smooth([None, 2.0, 4.0, None, 10.0], 2)
    assert math.isnan(result[0]) and math.isnan(result[3])
    assert result[1:3] == [2.0, 3.0] and result[4] == 10.0
    assert comparison._recent([None, None], 2) == {"observed_updates": 0, "mean": None}
