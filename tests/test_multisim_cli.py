from __future__ import annotations

from pathlib import Path

import pytest

from unilab import cli


def test_multisim_selects_owner_and_checks_all_real_backends(monkeypatch) -> None:
    checked = []
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda algo, sim: checked.append(sim))
    command = cli.build_command(
        mode="train", algo="ppo", task="g1_walk_flat", sim="multisim", overrides=[]
    )
    assert "task=g1_walk_flat/multisim" in command
    assert checked == ["isaacgym", "isaacsim", "motrix", "genesis"]
    assert "multisim" not in cli.SUPPORTED_SIMS


@pytest.mark.parametrize(
    "mode,algo,task",
    [
        ("eval", "ppo", "g1_walk_flat"),
        ("train", "sac", "g1_walk_flat"),
        ("train", "ppo", "go2_joystick_flat"),
    ],
)
def test_multisim_is_not_a_general_physical_backend(mode, algo, task) -> None:
    with pytest.raises(SystemExit, match="training owner"):
        cli.build_command(mode=mode, algo=algo, task=task, sim="multisim", overrides=[])


@pytest.mark.parametrize("backend", ["isaacsim", "mujoco"])
def test_metrics_routes_real_evaluation_instead_of_skipping(monkeypatch, backend: str) -> None:
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda *args: None)
    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="g1_walk_flat",
        sim=backend,
        profile="multisim",
        metrics=True,
        load_run="example_multisim",
        overrides=[],
    )
    assert f"task=g1_walk_flat/{backend}_multisim" in command
    assert "training.evaluation.enabled=true" in command
    assert "training.play_render_mode=none" in command
    assert "training.play_only=true" in command
    assert "algo.load_run=example_multisim" in command


def test_metrics_rejects_rendering_and_route_overrides(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda *args: None)
    with pytest.raises(SystemExit, match="without rendering"):
        cli.build_command(
            mode="eval",
            algo="ppo",
            task="g1_walk_flat",
            sim="isaacsim",
            profile="multisim",
            metrics=True,
            render_mode="record",
            overrides=[],
        )
    with pytest.raises(SystemExit, match="CLI flags"):
        cli.build_command(
            mode="eval",
            algo="ppo",
            task="g1_walk_flat",
            sim="isaacsim",
            profile="multisim",
            overrides=["training.evaluation.enabled=true"],
        )


def test_eval_parser_never_accepts_multisim_backend() -> None:
    with pytest.raises(SystemExit):
        cli._train_eval_parser(mode="eval").parse_args(
            ["--algo", "ppo", "--task", "g1_walk_flat", "--sim", "multisim"]
        )


def test_metrics_has_no_sibling_owner_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_check_runtime_requirements", lambda *args: None)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/train_rsl_rl.py").touch()
    owner = tmp_path / "conf/ppo/task/g1_walk_flat"
    owner.mkdir(parents=True)
    (owner / "motrix_multisim.yaml").touch()
    with pytest.raises(SystemExit, match="No owner config"):
        cli.build_command(
            mode="eval",
            algo="ppo",
            task="g1_walk_flat",
            sim="isaacsim",
            profile="multisim",
            metrics=True,
            overrides=[],
            root=tmp_path,
        )
