"""Portable report completeness and evidence guards; tiny mock media only."""

import csv
import json
from hashlib import sha256

import pytest

from unilab.visualization import experiment_report as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path):
    clocks, intervals = [], []
    for source, index in report.expected_checkpoints():
        batch = 98304 if source == "joint" else 24576
        prefix = f"checkpoints/{source}/model_{index:05d}/"
        clock = dict(
            policy=source,
            iteration=index,
            completed_updates=index + 1,
            cumulative_transitions=(index + 1) * batch,
            cumulative_optimizer_steps=(index + 1) * 20,
            cumulative_loop_seconds=float(index + 1),
        )
        clocks.append(clock)
        score = dict(
            horizon_steps=224,
            horizon_seconds=4.48,
            error_scales=report.SCALES,
            survived_step_fraction=1.0,
            observed_steps_including_failure=224,
            padded_steps_after_failure=0,
            normalized_error_with_failure_padding=dict.fromkeys(report.SCALES, 0.2),
            return_with_zero_failure_padding=33.0,
            time_to_first_failure_seconds=None,
            first_failure_reference_frame=None,
            first_failure_terms=[],
        )
        metrics = dict(
            protocol="g1_native_tracking_v1",
            frames=1000,
            control_dt_seconds=0.02,
            fixed_first_opportunity=score,
            true_terminations=0,
            attempts=[],
        )
        result = dict(
            protocol="retrospective-500-v1",
            source=source,
            checkpoint_index=index,
            completed_updates=index + 1,
            planned_updates=5000 if source == "joint" else 20000,
            checkpoint_sha256="a" * 64,
            total_transitions=(index + 1) * batch,
            optimizer_steps=(index + 1) * 20,
            video=f"media/{source}_{index:05d}.mp4",
            poster=f"media/{source}_{index:05d}.jpg",
            video_mode="native",
            video_format=dict(frames=1000, width=1280, height=720, fps=50, seconds=20.0),
            metrics=metrics,
        )
        native = dict(
            checkpoint_sha256="a" * 64,
            backend="mujoco",
            strict_preflight=True,
            actor_normalizer_unchanged=True,
            seed=1,
        )
        artifacts = [result["video"], result["poster"], *(prefix + p for p in report.ARTIFACTS)]
        for artifact in artifacts:
            write(
                tmp_path / artifact,
                metrics
                if artifact.endswith("metrics.json")
                else native
                if artifact.endswith("native/verification.json")
                else "mock artifact",
            )
        result["artifact_sha256"] = {
            p: sha256((tmp_path / p).read_bytes()).hexdigest() for p in artifacts
        }
        write(tmp_path / prefix / "result.json", result)
    for source in report.SOURCES:
        for first in range(0, 5000 if source == "joint" else 20000, 500):
            row = dict(
                policy=source, first_completed_update=first + 1, last_completed_update=first + 500
            )
            keys = "collection_seconds learning_seconds loop_seconds event_boundary_elapsed_seconds cumulative_loop_seconds transitions_per_loop_second episode_return_mean episode_length_mean Loss/value_mean Loss/entropy_mean Loss/learning_rate_mean".split()
            row.update(dict.fromkeys(keys, 1.0))
            intervals.append(row)
    write(
        tmp_path / "training/intervals.json",
        dict(
            checkpoint_clocks=clocks,
            interval_updates=500,
            intervals=intervals,
            source_intervals=[{}] * 200,
        ),
    )
    write(
        tmp_path / "audit/config-comparison.json",
        {"audit_passed": True, "shared": {"task_name": "G1FlipTracking"}},
    )
    for relative in (
        *(
            f"training/{name}.csv"
            for name in ("intervals", "source_intervals", "scalar_intervals", "checkpoint_clocks")
        ),
        *(
            f"training/{name}.{suffix}"
            for name in ("episodes", "tracking", "timing_ppo")
            for suffix in ("png", "pdf")
        ),
    ):
        write(tmp_path / relative, "fixture")
    return tmp_path


def test_complete_report_has_all_videos_portable_links_exact_budgets_and_plots(experiment):
    pytest.importorskip("matplotlib")
    result = report.report(experiment)
    assert result["complete"] and result["checkpoints"] == result["embedded_videos"] == 175
    for filename in ("REPORT.md", "index.html"):
        text = (experiment / filename).read_text()
        assert text.count('<video controls preload="none"') == 175
        assert text.count('type="video/mp4"') == 175
        assert 'href="media/motrix_00500.mp4"' in text
        assert "model_500.pt · 完成 501 更新" in text
        assert "3.5秒" in text and "不是失败率" in text
        assert "https://" not in text and "file://" not in text
    with (experiment / "checkpoint_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 175
    shared = next(r for r in rows if r["source"] == "joint" and r["checkpoint_index"] == "500")
    assert shared["completed_updates"] == "501" and int(shared["total_transitions"]) == 501 * 98304
    for name in ("samples", "updates", "time"):
        assert (experiment / f"figures/holdout_{name}.png").stat().st_size > 100
        assert (experiment / f"figures/holdout_{name}.pdf").stat().st_size > 100
    for relative, digest in result["output_sha256"].items():
        assert sha256((experiment / relative).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "fault",
    "missing digest budget scales nan inline path training existing protocol media".split(),
)
def test_bad_evidence_never_produces_completed_report(experiment, fault):
    path = experiment / "checkpoints/motrix/model_00500/result.json"
    data = json.loads(path.read_text())
    if fault == "missing":
        path.unlink()
    elif fault == "digest":
        (experiment / data["video"]).write_text("changed")
    elif fault == "existing":
        (experiment / "REPORT.md").write_text("preserve")
    elif fault == "training":
        training = experiment / "training/intervals.json"
        value = json.loads(training.read_text())
        value["checkpoint_clocks"][0]["completed_updates"] = 5
        write(training, value)
    else:
        if fault == "budget":
            data["completed_updates"] = 500
        elif fault == "scales":
            data["metrics"]["fixed_first_opportunity"]["error_scales"]["joint_rmse_rad"] = 2.0
        elif fault == "nan":
            data["metrics"]["frames"] = float("nan")
        elif fault == "inline":
            data["metrics"]["true_terminations"] = 1
        elif fault == "protocol":
            data["protocol"] = "retrospective-500-v2"
        elif fault == "media":
            data["video"] = "media/motrix_00000.mp4"
        else:
            data["artifact_sha256"]["../escape"] = "a" * 64
        write(path, data)
    with pytest.raises((ValueError, FileNotFoundError)):
        report.report(experiment)
    assert not (experiment / "index.html").exists()
    assert not (experiment / "report_manifest.json").exists()
    if fault == "existing":
        assert (experiment / "REPORT.md").read_text() == "preserve"


def test_flatten_retains_descriptive_statistics_and_lists_without_input_path_columns():
    flat = report._flatten(
        {
            "control_statistics": {"action_rms": {"mean": 1.0}},
            "native_episodes": [{"return": 2}],
            "input_sha256": {"/tmp/a": "b"},
        }
    )
    assert flat == {"control_statistics.action_rms.mean": 1.0, "native_episodes": '[{"return": 2}]'}
