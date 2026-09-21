"""Verify native single-source PPO completion from saved budgets and learner state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, expected: int, name: str) -> None:
    _require(type(value) is int and value == expected, f"{name}: expected integer {expected}")


def _model(state: dict, hidden: list[int], samples: int, actor: bool) -> list[torch.Tensor]:
    _require(len(hidden) == 3, "expected three hidden layers")
    names = [f"mlp.{index}.{kind}" for index in (0, 2, 4, 6) for kind in ("weight", "bias")]
    if actor:
        names.insert(0, "distribution.std_param")
    buffers = [f"obs_normalizer.{name}" for name in ("_mean", "_var", "_std", "count")]
    _require(set(state) == set(names + buffers), "incomplete or unexpected model state")
    for tensor in state.values():
        _require(
            isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all()),
            "non-finite model state",
        )
    count = state["obs_normalizer.count"]
    _require(
        count.dtype == torch.int64 and count.numel() == 1 and int(count) == samples,
        "normalizer count mismatch or non-int64 count",
    )
    width = state["mlp.0.weight"].shape[1]
    output = state["distribution.std_param"].numel() if actor else 1
    dimensions = [width, *hidden, output]
    for index, (before, after) in enumerate(zip(dimensions, dimensions[1:])):
        _require(
            state[f"mlp.{index * 2}.weight"].shape == (after, before)
            and state[f"mlp.{index * 2}.bias"].shape == (after,),
            "model layer shape mismatch",
        )
    for name in ("_mean", "_var", "_std"):
        tensor = state[f"obs_normalizer.{name}"]
        _require(
            tensor.dtype == torch.float32 and tensor.shape == (1, width),
            "normalizer shape or dtype mismatch",
        )
    _require(
        bool((state["obs_normalizer._var"] >= 0).all())
        and bool((state["obs_normalizer._std"] >= 0).all()),
        "invalid normalizer variance or standard deviation",
    )
    if actor:
        _require(
            state["distribution.std_param"].shape == (output,)
            and bool((state["distribution.std_param"] > 0).all()),
            "invalid policy standard deviation",
        )
    return [state[name] for name in names]


def _audit(run_dir: Path, iterations: int, num_envs: int) -> dict:
    _require(type(iterations) is int and iterations > 0, "positive iterations required")
    _require(type(num_envs) is int and num_envs > 0, "positive num_envs required")
    samples, steps = num_envs * 24 * iterations, iterations * 20
    summary = json.loads((run_dir / "run_summary.json").read_text())
    run = json.loads((run_dir / "run_config.json").read_text())
    cfg, algo = run["config"], run["config"]["algo"]
    _require(summary["status"] == "completed", "training did not complete")
    for key, value in {
        "completed_iterations": iterations - 1,
        "total_env_steps": samples,
        "run_env_steps": samples,
        "world_size": 1,
        "num_envs_per_rank": num_envs,
        "global_num_envs": num_envs,
        "samples_per_iteration": num_envs * 24,
    }.items():
        _integer(summary[key], value, f"summary {key}")
    for key, value in {
        "max_iterations": iterations,
        "num_envs": num_envs,
        "num_steps_per_env": 24,
        "save_interval": 500,
    }.items():
        _integer(algo[key], value, f"config {key}")
    for key, value in {"num_learning_epochs": 5, "num_mini_batches": 4}.items():
        _integer(algo["algorithm"][key], value, f"config {key}")
    _require(
        algo["resume"] is False and algo["load_run"] == "-1" and algo["resume_path"] is None,
        "fresh training requires resume=false, string load_run='-1', resume_path=null",
    )
    _require(
        algo["empirical_normalization"] is True and algo["algorithm"]["schedule"] == "adaptive",
        "normalized native adaptive PPO required",
    )
    source = cfg["training"]["sim_backend"]
    _require(
        source in ("motrix", "isaacsim", "isaacgym", "genesis")
        and summary["sim_backend"] == run["run"]["sim_backend"] == source
        and summary["algo"] == algo["algo"] == "ppo"
        and summary["task"] == cfg["training"]["task_name"] == "G1FlipTracking",
        "single-source PPO task provenance mismatch",
    )
    final_path = run_dir / f"model_{iterations - 1}.pt"
    _require(
        Path(summary["last_checkpoint"]).resolve(strict=True) == final_path,
        "summary final checkpoint mismatch",
    )
    saved = torch.load(final_path, map_location="cpu", weights_only=False)
    _integer(saved["iter"], iterations - 1, "checkpoint iteration")
    _integer(saved["unilab_logger_state"]["tot_timesteps"], samples, "checkpoint transitions")
    params = [
        tensor
        for name in ("actor", "critic")
        for tensor in _model(
            saved[f"{name}_state_dict"],
            algo["policy"][f"{name}_hidden_dims"],
            samples,
            name == "actor",
        )
    ]
    optimizer = saved["optimizer_state_dict"]
    _require(len(optimizer["param_groups"]) == 1, "one shared Adam parameter group required")
    group = optimizer["param_groups"][0]
    ids, state = group["params"], optimizer["state"]
    _require(
        len(ids) == len(params) == 17 and len(set(ids)) == 17 and set(ids) == set(state),
        "incomplete optimizer parameter state",
    )
    lr = group["lr"]
    _require(type(lr) in (int, float) and math.isfinite(lr) and lr > 0, "invalid learning rate")
    for index, parameter in zip(ids, params):
        step = state[index]["step"]
        _require(
            isinstance(step, torch.Tensor) and step.numel() == 1 and float(step) == steps,
            "actual Adam step mismatch",
        )
        for name in ("exp_avg", "exp_avg_sq"):
            moment = state[index][name]
            _require(
                isinstance(moment, torch.Tensor)
                and moment.shape == parameter.shape
                and bool(torch.isfinite(moment).all()),
                "invalid optimizer moment",
            )
        _require(bool((state[index]["exp_avg_sq"] >= 0).all()), "negative Adam second moment")
    digest = hashlib.sha256()
    with final_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "audit_passed": True,
        "scope": "native_single_source_final_budget",
        "run_dir": str(run_dir),
        "source": source,
        "iterations": iterations,
        "num_envs": num_envs,
        "transitions_per_iteration": num_envs * 24,
        "total_transitions": samples,
        "optimizer_steps": steps,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": digest.hexdigest(),
        "normalizer_counts": {"actor": samples, "critic": samples},
        "learning_rate": lr,
        "evidence_limit": "Saved budgets and learner state; does not prove task or holdout success.",
    }


def audit_single_run(
    run_dir: Path, *, expected_iterations: int = 20000, num_envs: int = 1024
) -> dict:
    """Reject incomplete, resumed, or inconsistent native single-source PPO runs."""
    try:
        return _audit(Path(run_dir).resolve(), expected_iterations, num_envs)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"invalid single-source audit schema: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-iterations", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(audit_single_run(**vars(args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
