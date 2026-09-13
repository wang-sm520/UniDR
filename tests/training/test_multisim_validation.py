from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from unilab.training import validation


@pytest.mark.parametrize("name", ["unilab_rl", "unisim"])
def test_dependency_root_prefers_bundle_and_preserves_sibling_fallback(tmp_path, monkeypatch, name):
    root = tmp_path / "UniDR"
    monkeypatch.setattr(validation, "_ROOT", root)
    assert validation._dependency_root(name) == tmp_path / name
    bundled = root / "vendor" / name
    bundled.mkdir(parents=True)
    assert validation._dependency_root(name) == bundled


def test_source_revision_tracks_each_vendored_source_tree(tmp_path, monkeypatch):
    root = tmp_path / "UniDR"
    roots = [root, root / "vendor/unilab_rl", root / "vendor/unisim"]
    for source in roots:
        (source / "src").mkdir(parents=True)
        (source / "src/example.py").write_text("value = 1\n")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setattr(validation, "_ROOT", root)
    baseline = validation.source_revision_evidence()
    assert set(baseline) == {"UniDR", "unilab_rl", "unisim"}
    assert len({entry["base"] for entry in baseline.values()}) == 1
    (roots[1] / "src/example.py").write_text("value = 2\n")
    modified = validation.source_revision_evidence()
    assert modified["unilab_rl"] != baseline["unilab_rl"]
    assert modified["unisim"] == baseline["unisim"]
    assert modified["UniDR"] == baseline["UniDR"]
    (roots[2] / "src/additional.py").write_text("value = 3\n")
    untracked = validation.source_revision_evidence()
    assert untracked["unisim"] != modified["unisim"]
    assert untracked["unilab_rl"] == modified["unilab_rl"]


@pytest.mark.parametrize("configured_scene", [None, "/configured/g1.xml"])
def test_native_physics_checks_run_from_vendored_dependency(
    tmp_path, monkeypatch, configured_scene
):
    root = tmp_path / "UniDR"
    bundled = root / "vendor/unisim"
    bundled.mkdir(parents=True)
    monkeypatch.setattr(validation, "_ROOT", root)
    monkeypatch.setattr(validation, "source_revision_evidence", lambda: {})
    if configured_scene is None:
        monkeypatch.delenv("UNILAB_G1_SCENE", raising=False)
    else:
        monkeypatch.setenv("UNILAB_G1_SCENE", configured_scene)
    native_directories = []
    native_scenes = []

    def run(command, logfile, **kwargs):
        from pathlib import Path

        target = next(
            value.partition("=")[2] for value in command if value.startswith("--junitxml=")
        )
        logfile.parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("<testsuites><testsuite><testcase /></testsuite></testsuites>")
        if "cwd" in kwargs:
            native_directories.append(kwargs["cwd"])
            native_scenes.append(kwargs["extra_env"]["UNILAB_G1_SCENE"])

    monkeypatch.setattr(validation, "_run_logged", run)
    validation._execute_gate("physics", tmp_path / "reports")
    assert native_directories == [bundled] * 4
    assert (
        native_scenes
        == [configured_scene or str(root / "src/unilab/assets/robots/g1/scene_flat.xml")] * 4
    )


def test_memory_snapshot_includes_current_process():
    import os

    report = validation.process_memory_snapshot()
    current = next(record for record in report["processes"] if record["pid"] == os.getpid())
    assert current["rss_bytes"] > 0
    assert report["rss_sum_bytes"] >= current["rss_bytes"]


def test_failed_preflight_never_runs_gpu_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validation, "collect_preflight", lambda: {"status": "blocked", "reasons": ["NVML mismatch"]}
    )
    monkeypatch.setattr(
        validation, "_execute_gate", lambda *args: pytest.fail("must not run a GPU gate")
    )
    assert validation.main(["--gate", "soak", "--output-root", str(tmp_path)]) == 2
    assert json.loads((tmp_path / "preflight-soak.json").read_text())["status"] == "blocked"
    assert not (tmp_path / "G6.json").exists()


def test_gates_require_same_source_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "source_revision_evidence", lambda: {"UniLab": "current"})
    with pytest.raises(ValueError, match="requires passing G1"):
        validation._require_gates(tmp_path, "single")
    (tmp_path / "G1.json").write_text(
        json.dumps({"status": "passed", "repositories": {"UniLab": "previous"}})
    )
    with pytest.raises(ValueError, match="current three-repository"):
        validation._require_gates(tmp_path, "single")
    (tmp_path / "G1.json").write_text(
        json.dumps({"status": "passed", "repositories": {"UniLab": "current"}})
    )
    validation._require_gates(tmp_path, "single")


def test_soak_budget_is_fixed_and_owned_by_uni_rl(tmp_path, monkeypatch):
    import uni_rl.algos.rsl_rl_validation as algorithm_validation

    captured = {}

    def run(runner, **kwargs):
        captured.update(kwargs)
        return {"status": "passed"}

    monkeypatch.setattr(algorithm_validation, "run_bounded_ppo", run)
    cfg = OmegaConf.create(
        {
            "training": {
                "validation": {"gate": "soak", "warmup_updates": 2, "output_dir": str(tmp_path)}
            }
        }
    )
    env = SimpleNamespace(source_statistics={"a": {"num_envs": 2000}})
    assert validation.run_ppo_acceptance(cfg, object(), env)["status"] == "passed"
    assert captured["duration_seconds"] == 3600
    assert captured["max_updates"] is None
    assert captured["warmup_updates"] == 2
    assert captured["source_statistics"]() == env.source_statistics


def test_explicit_physics_gate_rejects_skipped_tests(tmp_path, monkeypatch):
    def run(command, logfile, **kwargs):
        target = next(
            value.partition("=")[2] for value in command if value.startswith("--junitxml=")
        )
        validation._write_report(tmp_path / "unused.json", {})
        logfile.parent.mkdir(parents=True, exist_ok=True)
        from pathlib import Path

        Path(target).write_text(
            "<testsuites><testsuite><testcase><skipped /></testcase></testsuite></testsuites>"
        )

    monkeypatch.setattr(validation, "_run_logged", run)
    with pytest.raises(RuntimeError, match="did not actually pass"):
        validation._execute_gate("physics", tmp_path)


def test_non_multisim_cannot_request_acceptance():
    cfg = OmegaConf.create(
        {"training": {"sim_backend": "motrix", "validation": {"gate": "capacity"}}}
    )
    with pytest.raises(ValueError, match="multisim owner"):
        validation.validate_acceptance_request(cfg)
