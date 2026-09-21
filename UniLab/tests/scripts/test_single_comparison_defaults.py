"""Single-source budget defaults must agree across owners, preparation and queue."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.training import single_comparison

ROOT = Path(__file__).parents[2]
SOURCES = ("motrix", "isaacsim", "isaacgym", "genesis")
SEEDS = {"isaacsim": 2, "isaacgym": 3, "genesis": 4, "motrix": 5}
STAGES = ("prepare", "train", "audit", "sim2sim")


@pytest.mark.parametrize("source", SOURCES)
def test_single_owner_changes_only_budget(source):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        actual = compose("config", [f"task=g1_flip_tracking/{source}_comparison"])
        overrides = [f"task=g1_flip_tracking/unidr_{source}", f"+env.seed={SEEDS[source]}"]
        overrides += ["algo.num_envs=4096", "algo.max_iterations=5000"]
        expected = compose("config", overrides)
    assert OmegaConf.to_container(actual, resolve=True) == OmegaConf.to_container(
        expected, resolve=True
    )
    assert actual.training.sim2sim_strict and not actual.algo.resume
    assert actual.algo.num_steps_per_env == 24
    assert actual.algo.algorithm.num_learning_epochs * actual.algo.algorithm.num_mini_batches == 20
    assert "unidr" not in actual


@pytest.mark.parametrize("layout", ["single", "four"])
def test_joint_defaults_are_unchanged(layout):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", [f"task=g1_flip_tracking/unidr_{layout}_gpu"])
    assert (cfg.algo.num_envs, cfg.algo.max_iterations) == (1024, 10000)
    assert cfg.algo.num_steps_per_env == 24
    assert tuple(item.name for item in cfg.unidr.sources) == tuple(SEEDS)
    assert "adaptive" not in cfg.unidr


@pytest.mark.parametrize("source", SOURCES)
def test_prepare_defaults_match_single_owner(source, tmp_path, monkeypatch):
    monkeypatch.setattr(single_comparison, "build_manifest", lambda *_: {"digest": "test"})
    result = single_comparison.prepare_run(ROOT, tmp_path / source, source)
    assert (result["num_envs"], result["expected_iterations"]) == (4096, 5000)
    assert result["config"]["algo"]["num_envs"] == 4096
    assert result["config"]["algo"]["max_iterations"] == 5000
    assert json.loads((tmp_path / source / "single_manifest.json").read_text()) == result


def test_prepare_cli_defaults(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["prepare", "motrix", str(tmp_path / "new")])
    monkeypatch.setattr(
        single_comparison,
        "prepare_run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"digest": "test"},
    )
    single_comparison.main()
    assert calls == [((ROOT, tmp_path / "new", "motrix"), {"num_envs": 4096, "iterations": 5000})]


FAKE_UV = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "unilab.training.single_comparison" in args:
    i = args.index("unilab.training.single_comparison")
    source, run = args[i+1], Path(args[i+2])
    run.mkdir()
    stage = "prepare"
elif "train" in args:
    source = args[args.index("--sim")+1]
    stage = "train"
elif "uni_rl.logging.single_run_audit" in args:
    source = Path(args[args.index("uni_rl.logging.single_run_audit")+1]).name
    stage = "audit"
else:
    source = Path(args[args.index("scripts/play_single_reference.py")+1]).parent.name
    stage = "sim2sim"
with Path(os.environ["FAKE_UV_EVENTS"]).open("a") as stream:
    stream.write(json.dumps([source, stage, args])+"\n")
"""


@pytest.mark.parametrize("budget", [None, (2, 2)], ids=["defaults", "explicit-smoke"])
def test_queue_propagates_budget_through_all_stages(tmp_path, budget):
    executable = tmp_path / "uv"
    executable.write_text(f"#!{sys.executable}\n" + FAKE_UV)
    executable.chmod(0o755)
    events = tmp_path / "events.jsonl"
    run_root = tmp_path / "experiment with spaces"
    command = ["bash", str(ROOT / "scripts/run_single_comparison.sh"), str(run_root)]
    if budget is not None:
        command.extend(map(str, budget))
    env = dict(os.environ, FAKE_UV_EVENTS=str(events))
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    result = subprocess.run(command, capture_output=True, text=True, timeout=20, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in events.read_text().splitlines()]
    expected = [[source, stage] for source in SOURCES for stage in STAGES]
    assert [record[:2] for record in records] == expected
    num_envs, iterations = budget or (4096, 5000)
    for source, stage, args in records:
        if stage == "train":
            assert "--profile" in args and args[args.index("--profile") + 1] == "comparison"
            assert f"algo.num_envs={num_envs}" in args
            assert f"algo.max_iterations={iterations}" in args
        else:
            assert args[args.index("--num-envs") + 1] == str(num_envs)
            flag = "--iterations" if stage == "prepare" else "--expected-iterations"
            assert args[args.index(flag) + 1] == str(iterations)
        if stage == "sim2sim":
            assert str(run_root / source / f"model_{iterations - 1}.pt") in args
    assert (run_root / "status.tsv").read_text().split("\t")[1] == "completed"
