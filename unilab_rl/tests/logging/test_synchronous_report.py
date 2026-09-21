"""Reject fabricated budget evidence and audit actual checkpoint tensor fields."""

import hashlib
import json

import pytest
import torch

from uni_rl.logging.synchronous_report import SOURCE_ORDER, audit_run, report


def fixture_run(path, iterations=2, num_envs=2):
    path.mkdir()
    algorithm = {
        "num_envs": num_envs,
        "num_steps_per_env": 24,
        "save_interval": 500,
        "empirical_normalization": True,
        "algorithm": {"num_learning_epochs": 5, "num_mini_batches": 4, "schedule": "adaptive"},
    }
    sources = [{"source": name, "device": "cpu"} for name in SOURCE_ORDER]
    behavior = {"sources": sources, "algorithm": algorithm, "assets": {"robot.xml": "a" * 64}}
    digest = hashlib.sha256(json.dumps(behavior, sort_keys=True).encode()).hexdigest()
    (path / "sources_manifest.json").write_text(json.dumps({**behavior, "digest": digest}))
    (path / "run_config.json").write_text(
        json.dumps(
            {
                "run": {"mode": "synchronous_four_source", "manifest_digest": digest},
                "config": {
                    "algo": {**algorithm, "max_iterations": iterations},
                    "unidr": {
                        "sources": [{"name": name, "device": "cpu"} for name in SOURCE_ORDER]
                    },
                },
            }
        )
    )
    rows = []
    for iteration in range(iterations):
        count, samples = iteration + 1, num_envs * 96
        rows.append(
            {
                "iteration": iteration,
                "generation": 0,
                "window_stamp": [iteration, iteration, iteration * 24],
                "policy_version": count,
                "normalizer_version": count * 24,
                "optimizer_steps": count * 20,
                "total_transitions": count * samples,
                "epoch_samples": [samples] * 5,
                "sources": {
                    name: {
                        "transitions": samples // 4,
                        "reward": 0.5,
                        "episode_return": 5.0,
                        "episode_length": 10.0,
                        "terminated": 1,
                        "truncated": 0,
                    }
                    for name in SOURCE_ORDER
                },
                "reward": 0.5,
                "episode_return": 5.0,
                "episode_length": 10.0,
                "losses": {"value": 0.1, "surrogate": -0.01, "entropy": 0.4},
                "learning_rate": 0.0001,
                "collect_seconds": 2.0,
                "learn_seconds": 0.5,
            }
        )
        if iteration not in (0, iterations - 1):
            continue
        states = {}
        for name, shape in (("actor", (2, 2)), ("critic", (1, 3))):
            states[f"{name}_state_dict"] = {
                "mlp.0.weight": torch.full(shape, float(iteration)),
                "obs_normalizer._mean": torch.zeros(1, shape[-1]),
                "obs_normalizer._var": torch.ones(1, shape[-1]),
                "obs_normalizer._std": torch.ones(1, shape[-1]),
                "obs_normalizer.count": torch.tensor(count * samples, dtype=torch.int64),
            }
        optimizer = {
            "param_groups": [{"params": [0, 1], "lr": 0.0001}],
            "state": {
                index: {
                    "step": torch.tensor(float(count * 20)),
                    "exp_avg": torch.zeros(shape),
                    "exp_avg_sq": torch.zeros(shape),
                }
                for index, shape in enumerate(((2, 2), (1, 3)))
            },
        }
        torch.save(
            {
                **states,
                "optimizer_state_dict": optimizer,
                "iter": iteration,
                "synchronous": {
                    "format": 1,
                    "complete": True,
                    "contract": {
                        "manifest_digest": digest,
                        "train_cfg": algorithm,
                        "num_envs": num_envs * 4,
                        "sources": [
                            (name, i * num_envs, (i + 1) * num_envs)
                            for i, name in enumerate(SOURCE_ORDER)
                        ],
                    },
                    "next_iteration": count,
                    "policy_version": count,
                    "normalizer_version": count * 24,
                    "optimizer_steps": count * 20,
                    "total_transitions": count * samples,
                    "generation": 0,
                    "learning_rate": 0.0001,
                },
            },
            path / f"model_{iteration}.pt",
        )
    (path / "synchronous_metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    return path


def test_formal_budget_audits_all_10000_iterations(tmp_path):
    run = fixture_run(tmp_path / "run", 10000, 1024)
    result = audit_run(run)
    assert result["scope"] == "formal_training_budget"
    assert result["total_transitions"] == 983040000
    assert result["optimizer_steps"] == 200000
    assert set(result["source_total_transitions"].values()) == {245760000}
    assert result["parameters_changed_since_iteration_0"] == {"actor": True, "critic": True}
    assert (
        result["final_checkpoint_sha256"]
        == hashlib.sha256((run / "model_9999.pt").read_bytes()).hexdigest()
    )


def test_bounded_run_requires_explicit_budget(tmp_path):
    run = fixture_run(tmp_path / "run")
    with pytest.raises(ValueError, match="max_iterations"):
        audit_run(run)
    result = audit_run(run, expected_iterations=2, num_envs=2)
    assert result["scope"] == "bounded_validation_budget" and result["total_transitions"] == 384


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "stamp", "boolean", "source", "epoch", "nan"]
)
def test_reject_bad_metric_evidence(tmp_path, fault):
    run = fixture_run(tmp_path / "run")
    path = run / "synchronous_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[1] = rows[0]
    elif fault == "stamp":
        rows[-1]["window_stamp"][2] = 0
    elif fault == "boolean":
        rows[-1]["iteration"] = True
    elif fault == "source":
        rows[-1]["sources"]["genesis"]["transitions"] -= 1
    elif fault == "epoch":
        rows[-1]["epoch_samples"][-1] -= 1
    elif fault == "nan":
        rows[-1]["losses"]["value"] = float("nan")
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError):
        audit_run(run, expected_iterations=2, num_envs=2)


@pytest.mark.parametrize(
    "fault",
    [
        "partial",
        "counter",
        "adam",
        "count",
        "float_count",
        "missing_count",
        "nan",
        "moment_shape",
        "missing_param",
        "unchanged",
    ],
)
@pytest.mark.parametrize("prefix", [False, True])
def test_reject_corrupt_saved_learner(tmp_path, fault, prefix):
    run = fixture_run(tmp_path / "run")
    path = run / "model_1.pt"
    saved = torch.load(path, weights_only=False)
    if fault == "partial":
        saved["synchronous"]["complete"] = False
    elif fault == "counter":
        saved["synchronous"]["optimizer_steps"] -= 1
    elif fault == "adam":
        saved["optimizer_state_dict"]["state"][0]["step"] -= 1
    elif fault == "count":
        saved["actor_state_dict"]["obs_normalizer.count"] -= 1
    elif fault == "float_count":
        saved["actor_state_dict"]["obs_normalizer.count"] = torch.tensor(384.0)
    elif fault == "missing_count":
        saved["actor_state_dict"].pop("obs_normalizer.count")
    elif fault == "nan":
        saved["critic_state_dict"]["mlp.0.weight"].fill_(float("nan"))
    elif fault == "moment_shape":
        saved["optimizer_state_dict"]["state"][0]["exp_avg"] = torch.zeros(17)
    elif fault == "missing_param":
        saved["optimizer_state_dict"]["state"].pop(0)
    elif fault == "unchanged":
        saved["critic_state_dict"]["mlp.0.weight"].zero_()
    torch.save(saved, path)
    with pytest.raises(ValueError):
        audit_run(run, expected_iterations=2, num_envs=2, final_checkpoint=path if prefix else None)


@pytest.mark.parametrize("fault", ["intent", "capacity", "fingerprint"])
def test_reject_mismatched_metadata(tmp_path, fault):
    run = fixture_run(tmp_path / "run")
    path = run / "run_config.json"
    cfg = json.loads(path.read_text())
    if fault == "intent":
        cfg["config"]["algo"]["max_iterations"] = 10000
    elif fault == "capacity":
        cfg["config"]["algo"]["num_envs"] = 1024
    elif fault == "fingerprint":
        cfg["run"]["manifest_digest"] = "b" * 64
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        audit_run(run, expected_iterations=2, num_envs=2)


def test_report_never_overwrites_run_or_existing_outputs(tmp_path):
    run = fixture_run(tmp_path / "run")
    for output in (run, tmp_path):
        with pytest.raises(ValueError, match="separate"):
            report(run, output, expected_iterations=2, num_envs=2)
    output = tmp_path / "report"
    output.mkdir()
    (output / "keep.txt").write_text("unrelated")
    with pytest.raises(ValueError, match="empty"):
        report(run, output, expected_iterations=2, num_envs=2)
    assert (output / "keep.txt").read_text() == "unrelated"


@pytest.mark.parametrize("prefix", [False, True])
def test_report_creates_json_and_standalone_curves(tmp_path, prefix):
    pytest.importorskip("matplotlib")
    run = fixture_run(tmp_path / "run")
    output = tmp_path / "report"
    result = report(
        run,
        output,
        expected_iterations=2,
        num_envs=2,
        final_checkpoint=run / "model_1.pt" if prefix else None,
    )
    assert json.loads((output / "audit.json").read_text()) == result
    assert (output / "curves.png").read_bytes().startswith(b"\x89PNG")
    assert (output / "curves.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("fault", [None, "cycle", "manifest", "generation", "gap"])
def test_resume_follows_only_selected_checkpoint_prefix(tmp_path, fault):
    parent = fixture_run(tmp_path / "parent", iterations=3)
    child = fixture_run(tmp_path / "child", iterations=3)
    cfg_path = child / "run_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["config"]["algo"].update(resume=True, resume_path=str(parent / "model_0.pt"))
    if fault == "cycle":
        cfg["config"]["algo"]["resume_path"] = str(child / "model_0.pt")
    cfg_path.write_text(json.dumps(cfg))
    metrics_path = child / "synchronous_metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()][1:]
    for row in rows:
        row["generation"] = 0 if fault == "generation" else 1
    if fault == "gap":
        rows.pop(0)
    metrics_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    checkpoint = child / "model_2.pt"
    saved = torch.load(checkpoint, weights_only=False)
    saved["synchronous"]["generation"] = 1
    torch.save(saved, checkpoint)
    if fault == "manifest":
        path = parent / "sources_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["assets"]["robot.xml"] = "b" * 64
        path.write_text(json.dumps(manifest))
    if fault:
        with pytest.raises(ValueError):
            audit_run(child, expected_iterations=3, num_envs=2)
    else:
        # A torn abandoned tail must not invalidate the selected completed prefix.
        with (parent / "synchronous_metrics.jsonl").open("a") as stream:
            stream.write('{"iteration":')
        result = audit_run(child, expected_iterations=3, num_envs=2)
        assert result["total_transitions"] == 576 and result["generations"] == [0, 1]
        assert [stage["through_iteration"] for stage in result["history"]] == [0, 2]


def test_explicit_checkpoint_preserves_original_intent_and_excludes_uncommitted_tail(tmp_path):
    run = fixture_run(tmp_path / "run", iterations=3)
    config_path = run / "run_config.json"
    cfg = json.loads(config_path.read_text())
    cfg["config"]["algo"]["max_iterations"] = 10000
    config_path.write_text(json.dumps(cfg))
    config_bytes = config_path.read_bytes()
    selected = run / "model_2.pt"
    checkpoint_bytes = selected.read_bytes()
    tail = '{"iteration": 3}\n{"iteration":'
    with (run / "synchronous_metrics.jsonl").open("a") as stream:
        stream.write(tail)
    with pytest.raises(ValueError, match="max_iterations"):
        audit_run(run, expected_iterations=3, num_envs=2)
    result = audit_run(run, expected_iterations=3, num_envs=2, final_checkpoint=selected)
    assert result["scope"] == "explicit_checkpoint_prefix"
    assert result["audit_mode"] == "selected_checkpoint_prefix"
    assert result["configured_max_iterations"] == 10000 and result["iterations"] == 3
    assert result["total_transitions"] == 576 and result["optimizer_steps"] == 60
    assert set(result["source_total_transitions"].values()) == {144}
    assert result["history"][-1]["excluded_metric_tail_bytes"] == len(tail.encode())
    assert result["final_checkpoint"] == str(selected)
    assert result["final_checkpoint_sha256"] == hashlib.sha256(checkpoint_bytes).hexdigest()
    assert config_path.read_bytes() == config_bytes and selected.read_bytes() == checkpoint_bytes


@pytest.mark.parametrize("fault", ["outside", "filename", "intent", "missing_metric"])
def test_explicit_checkpoint_still_requires_the_selected_committed_boundary(tmp_path, fault):
    run = fixture_run(tmp_path / "run", iterations=3)
    selected = run / "model_2.pt"
    if fault == "outside":
        selected = fixture_run(tmp_path / "other", iterations=3) / "model_2.pt"
    elif fault == "filename":
        selected = run / "model_0.pt"
    elif fault == "intent":
        path = run / "run_config.json"
        cfg = json.loads(path.read_text())
        cfg["config"]["algo"]["max_iterations"] = 2
        path.write_text(json.dumps(cfg))
    elif fault == "missing_metric":
        path = run / "synchronous_metrics.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError):
        audit_run(run, expected_iterations=3, num_envs=2, final_checkpoint=selected)
