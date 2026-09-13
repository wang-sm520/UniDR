from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from uni_rl.algos.rsl_rl_validation import run_bounded_ppo


class _Runner:
    def __init__(self, clock):
        self.clock = clock
        self.num_updates = 0
        self.saved = []
        self.closed = 0
        self.cfg = {
            "num_steps_per_env": 24,
            "algorithm": {"num_mini_batches": 4, "num_learning_epochs": 5},
        }
        self.env = SimpleNamespace(num_envs=8)
        self.alg = SimpleNamespace(actor=torch.nn.Linear(2, 1), critic=torch.nn.Linear(2, 1))
        self.logger = SimpleNamespace(log=lambda **kwargs: None, stop_logging_writer=self.stop)
        self.loss = 0.5

    def stop(self):
        self.closed += 1

    def learn(self, *, num_learning_iterations, init_at_random_ep_len):
        assert init_at_random_ep_len
        for iteration in range(num_learning_iterations):
            self.num_updates += 1
            self.clock[0] += 2.0
            self.logger.log(
                it=iteration,
                collect_time=1.5,
                learn_time=0.5,
                loss_dict={"value": self.loss},
                action_std=torch.ones(1),
                learning_rate=0.001,
            )

    def save(self, path, *, infos):
        self.saved.append((path, infos))
        torch.save(self.alg.actor.state_dict(), path)

    def stats(self):
        return {
            name: {"num_envs": 2, "transitions": self.num_updates * 48}
            for name in ("a", "b", "c", "d")
        }


def test_duration_stops_after_update_and_excludes_warmup(tmp_path):
    clock = [0.0]
    runner = _Runner(clock)
    original_logger = runner.logger
    result = run_bounded_ppo(
        runner,
        output_dir=tmp_path,
        duration_seconds=5,
        warmup_updates=2,
        source_statistics=runner.stats,
        clock=lambda: clock[0],
    )
    assert runner.num_updates == 5
    assert result["updates"] == 5
    assert result["measured_elapsed_seconds"] == 6
    assert result["measured_updates"] == 3
    assert len(runner.saved) == 1
    assert runner.closed == 1
    assert runner.logger is original_logger
    rows = [json.loads(line) for line in (tmp_path / "updates.jsonl").read_text().splitlines()]
    assert all(row["source_samples"] == {name: 48 for name in ("a", "b", "c", "d")} for row in rows)
    assert [row["warmup"] for row in rows] == [True, True, False, False, False]
    restored = torch.nn.Linear(2, 1)
    restored.load_state_dict(torch.load(result["checkpoint"], weights_only=True))


def test_capacity_finishes_exactly_one_complete_update(tmp_path):
    clock = [0.0]
    runner = _Runner(clock)
    result = run_bounded_ppo(runner, output_dir=tmp_path, max_updates=1, clock=lambda: clock[0])
    assert result["updates"] == runner.num_updates == 1
    assert result["transitions"] == 192


def test_nonfinite_update_does_not_claim_success(tmp_path):
    runner = _Runner([0.0])
    runner.loss = float("nan")
    with pytest.raises(FloatingPointError, match="loss/value"):
        run_bounded_ppo(runner, output_dir=tmp_path, max_updates=1)
    assert not runner.saved
    assert runner.closed == 1
    assert not (tmp_path / "validation_report.json").exists()


def test_source_drop_is_a_failure_not_a_resized_rollout(tmp_path):
    runner = _Runner([0.0])
    with pytest.raises(RuntimeError, match="expected 48 samples, got 0"):
        run_bounded_ppo(
            runner,
            output_dir=tmp_path,
            max_updates=1,
            source_statistics=lambda: {"a": {"num_envs": 2, "transitions": 0}},
        )
    assert not runner.saved


@pytest.mark.parametrize(
    "budgets",
    [
        {},
        {"max_updates": 1, "duration_seconds": 1},
        {"max_updates": 0},
        {"duration_seconds": float("inf")},
    ],
)
def test_invalid_budget_is_rejected(tmp_path, budgets):
    with pytest.raises(ValueError):
        run_bounded_ppo(_Runner([0.0]), output_dir=tmp_path, **budgets)


def test_epoch_must_not_drop_transitions(tmp_path):
    runner = _Runner([0.0])
    runner.cfg["algorithm"]["num_mini_batches"] = 5
    with pytest.raises(ValueError, match="entire rollout"):
        run_bounded_ppo(runner, output_dir=tmp_path, max_updates=1)


def test_warmup_telemetry_is_not_counted_as_training(tmp_path):
    clock = [0.0]
    runner = _Runner(clock)

    def sample_resources():
        clock[0] += 10.0
        return {}

    result = run_bounded_ppo(
        runner,
        output_dir=tmp_path,
        duration_seconds=1,
        warmup_updates=1,
        sample_resources=sample_resources,
        clock=lambda: clock[0],
    )
    assert result["updates"] == 2
    assert result["measured_elapsed_seconds"] == 2
