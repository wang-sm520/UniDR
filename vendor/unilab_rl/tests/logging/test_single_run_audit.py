"""A completed summary cannot conceal incorrect saved single-source learner state."""

import json

import pytest
import torch

from uni_rl.logging.single_run_audit import audit_single_run, main


@pytest.fixture
def run(tmp_path):
    cfg = {
        "training": {"sim_backend": "motrix", "task_name": "G1FlipTracking"},
        "algo": {
            "algo": "ppo",
            "max_iterations": 2,
            "num_envs": 2,
            "num_steps_per_env": 24,
            "save_interval": 500,
            "resume": False,
            "load_run": "-1",
            "resume_path": None,
            "empirical_normalization": True,
            "algorithm": {"num_learning_epochs": 5, "num_mini_batches": 4, "schedule": "adaptive"},
            "policy": {"actor_hidden_dims": [4, 3, 2], "critic_hidden_dims": [4, 3, 2]},
        },
    }
    summary = {
        "status": "completed",
        "completed_iterations": 1,
        "total_env_steps": 96,
        "run_env_steps": 96,
        "world_size": 1,
        "num_envs_per_rank": 2,
        "global_num_envs": 2,
        "samples_per_iteration": 48,
        "last_checkpoint": str(tmp_path / "model_1.pt"),
        "algo": "ppo",
        "task": "G1FlipTracking",
        "sim_backend": "motrix",
    }
    saved = {"iter": 1, "unilab_logger_state": {"tot_timesteps": 96}}
    params = []
    for name, obs, output in (("actor", 3, 2), ("critic", 5, 1)):
        state = {f"obs_normalizer.{key}": torch.ones(1, obs) for key in ("_mean", "_var", "_std")}
        state["obs_normalizer.count"] = torch.tensor(96, dtype=torch.int64)
        if name == "actor":
            state["distribution.std_param"] = torch.ones(output)
        dims = [obs, 4, 3, 2, output]
        for index, (before, after) in enumerate(zip(dims, dims[1:])):
            state[f"mlp.{2 * index}.weight"] = torch.ones(after, before)
            state[f"mlp.{2 * index}.bias"] = torch.ones(after)
        params.extend(
            value for key, value in state.items() if not key.startswith("obs_normalizer.")
        )
        saved[f"{name}_state_dict"] = state
    saved["optimizer_state_dict"] = {
        "param_groups": [{"params": list(range(17)), "lr": 0.001}],
        "state": {
            index: {
                "step": torch.tensor(40.0),
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
            }
            for index, param in enumerate(params)
        },
    }

    def write():
        (tmp_path / "run_summary.json").write_text(json.dumps(summary))
        (tmp_path / "run_config.json").write_text(
            json.dumps({"config": cfg, "run": {"sim_backend": "motrix"}})
        )
        torch.save(saved, tmp_path / "model_1.pt")

    write()
    return tmp_path, cfg, summary, saved, write


def test_complete_saved_budget_and_cli(run, monkeypatch, capsys):
    path = run[0]
    result = audit_single_run(path, expected_iterations=2, num_envs=2)
    assert result["total_transitions"] == 96 and result["optimizer_steps"] == 40
    assert result["normalizer_counts"] == {"actor": 96, "critic": 96}
    assert len(result["final_checkpoint_sha256"]) == 64
    monkeypatch.setattr(
        "sys.argv", ["audit", str(path), "--expected-iterations", "2", "--num-envs", "2"]
    )
    main()
    assert json.loads(capsys.readouterr().out) == result


def test_constant_observations_allow_zero_normalizer_variance(run):
    path, _, _, saved, write = run
    # RSL divides by std + eps; constant reference features legitimately have std=0.
    for name in ("actor", "critic"):
        state = saved[f"{name}_state_dict"]
        state["obs_normalizer._var"][0, 0] = 0
        state["obs_normalizer._std"][0, 0] = 0
    write()
    assert audit_single_run(path, expected_iterations=2, num_envs=2)["audit_passed"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("completed_iterations", 2),
        ("total_env_steps", 95),
        ("run_env_steps", 95),
        ("world_size", 4),
        ("global_num_envs", 8),
    ],
)
def test_reject_incomplete_summary(run, field, value):
    path, _, summary, _, write = run
    summary[field] = value
    write()
    with pytest.raises(ValueError):
        audit_single_run(path, expected_iterations=2, num_envs=2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("resume", True),
        ("load_run", -1),
        ("resume_path", "old.pt"),
        ("max_iterations", 3),
        ("num_envs", 3),
        ("num_steps_per_env", 12),
    ],
)
def test_reject_fresh_config_or_budget_mismatch(run, field, value):
    path, cfg, _, _, write = run
    cfg["algo"][field] = value
    write()
    with pytest.raises(ValueError):
        audit_single_run(path, expected_iterations=2, num_envs=2)


@pytest.mark.parametrize(
    "damage", ["weight", "count", "iteration", "logger", "step", "shape", "nan", "missing"]
)
def test_reject_corrupt_learner_state(run, damage):
    path, _, _, saved, write = run
    actor, optimizer = saved["actor_state_dict"], saved["optimizer_state_dict"]["state"]
    if damage == "weight":
        actor["mlp.0.weight"][0, 0] = float("nan")
    elif damage == "count":
        actor["obs_normalizer.count"] = torch.tensor(96.0)
    elif damage == "iteration":
        saved["iter"] = 0
    elif damage == "logger":
        saved["unilab_logger_state"]["tot_timesteps"] = 95
    elif damage == "step":
        optimizer[16]["step"] = torch.tensor(39.0)
    elif damage == "shape":
        optimizer[0]["exp_avg"] = torch.zeros(3)
    elif damage == "nan":
        optimizer[0]["exp_avg_sq"][0] = float("nan")
    else:
        optimizer.pop(16)
    write()
    with pytest.raises(ValueError):
        audit_single_run(path, expected_iterations=2, num_envs=2)
