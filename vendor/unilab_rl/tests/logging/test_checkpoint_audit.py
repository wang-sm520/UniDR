"""Intermediate audits retain completed-run provenance and exact saved budgets."""

import copy
import json

import pytest
import torch
from test_single_run_audit import run  # noqa: F401

from uni_rl.logging.checkpoint_audit import audit_single_checkpoint, main
from uni_rl.logging.single_run_audit import audit_single_run


@pytest.fixture
def intermediate(run):  # noqa: F811 - imported pytest fixture
    path, cfg, summary, final, write = run
    saved = copy.deepcopy(final)
    saved["iter"] = 0
    saved["unilab_logger_state"] = {"tot_timesteps": 48, "tot_time": 1.25}
    for name in ("actor", "critic"):
        saved[f"{name}_state_dict"]["obs_normalizer.count"] = torch.tensor(48, dtype=torch.int64)
    for state in saved["optimizer_state_dict"]["state"].values():
        state["step"] = torch.tensor(20.0)
    checkpoint = path / "model_0.pt"
    torch.save(saved, checkpoint)
    return path, checkpoint, saved, cfg, summary, write


def _audit(path, checkpoint):
    return audit_single_checkpoint(path, checkpoint=checkpoint, expected_iterations=2, num_envs=2)


def test_selected_checkpoint_and_unchanged_final_audit(intermediate, monkeypatch, capsys):
    path, checkpoint, *_ = intermediate
    result = _audit(path, checkpoint)
    assert result["checkpoint_index"] == 0
    assert result["completed_iterations"] == 1
    assert result["total_transitions"] == 48
    assert result["optimizer_steps"] == 20
    assert result["normalizer_counts"] == {"actor": 48, "critic": 48}
    assert result["saved_logger_elapsed_seconds"] == 1.25
    assert len(result["checkpoint_sha256"]) == 64
    assert audit_single_run(path, expected_iterations=2, num_envs=2)["total_transitions"] == 96
    monkeypatch.setattr(
        "sys.argv",
        ["audit", str(path), str(checkpoint), "--expected-iterations", "2", "--num-envs", "2"],
    )
    main()
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("name", ["model_2.pt", "model_-1.pt", "model_00.pt", "other.pt"])
def test_invalid_index_and_filename(intermediate, name):
    path, checkpoint, saved, *_ = intermediate
    alternate = path / name
    torch.save(saved, alternate)
    with pytest.raises(ValueError):
        _audit(path, alternate)


def test_other_run_and_escaping_symlink(intermediate, tmp_path_factory):
    path, checkpoint, saved, *_ = intermediate
    outside = tmp_path_factory.mktemp("other") / checkpoint.name
    torch.save(saved, outside)
    with pytest.raises(ValueError, match="belong"):
        _audit(path, outside)
    checkpoint.unlink()
    checkpoint.symlink_to(outside)
    with pytest.raises(ValueError, match="belong"):
        _audit(path, checkpoint)


def test_symlink_cannot_change_requested_index(intermediate):
    path, checkpoint, *_ = intermediate
    checkpoint.unlink()
    checkpoint.symlink_to(path / "model_1.pt")
    with pytest.raises(ValueError, match="requested index"):
        _audit(path, checkpoint)


@pytest.mark.parametrize(
    "damage", ["iteration", "logger", "time", "count", "actor", "critic", "adam", "step", "missing"]
)
def test_corrupt_selected_checkpoint(intermediate, damage):
    path, checkpoint, saved, *_ = intermediate
    if damage == "iteration":
        saved["iter"] = 1
    elif damage == "logger":
        saved["unilab_logger_state"]["tot_timesteps"] = 96
    elif damage == "time":
        saved["unilab_logger_state"]["tot_time"] = float("inf")
    elif damage == "count":
        saved["actor_state_dict"]["obs_normalizer.count"] = torch.tensor(48.0)
    elif damage in ("actor", "critic"):
        saved[f"{damage}_state_dict"]["mlp.0.weight"][0, 0] = float("nan")
    elif damage == "adam":
        saved["optimizer_state_dict"]["state"][0]["exp_avg"][0] = float("nan")
    elif damage == "step":
        saved["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(40.0)
    else:
        del saved["unilab_logger_state"]
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError):
        _audit(path, checkpoint)


@pytest.mark.parametrize("damage", ["incomplete", "resume", "final_path", "source"])
def test_invalid_complete_run_provenance(intermediate, damage):
    path, checkpoint, _, cfg, summary, write = intermediate
    if damage == "incomplete":
        summary["status"] = "running"
    elif damage == "resume":
        cfg["algo"]["resume"] = True
    elif damage == "final_path":
        summary["last_checkpoint"] = str(checkpoint)
    else:
        summary["sim_backend"] = "mujoco"
    write()
    with pytest.raises(ValueError):
        _audit(path, checkpoint)
