"""Real native PPO lifecycle, recovery boundaries, and checkpoint validation."""

import json
import os
import random

import numpy as np
import pytest
import torch
from examples.unidr_fake import FakeSource

from uni_rl.algos.synchronous_ppo import CentralVecEnv
from uni_rl.algos.synchronous_runner import SynchronousPPORunner
from uni_rl.ipc.synchronous_env import SourceSpec, SynchronousEnv


def make_runner(directory, digest="test-config-and-assets", source=None, adaptive=None):
    torch.set_num_threads(1)
    if source is None:
        source = FakeSource(0, 8)
        source.source_slices = {f"source{i}": slice(2 * i, 2 * i + 2) for i in range(4)}
    model = {"class_name": "rsl_rl.models.MLPModel", "hidden_dims": [8], "obs_normalization": True}
    cfg = {
        "num_steps_per_env": 24,
        "save_interval": 500,
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "actor": {
            **model,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": 0.5,
            },
        },
        "critic": dict(model),
        "algorithm": {
            "class_name": "uni_rl.algos.rsl_rl_ppo.FinalObservationAwarePPO",
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
        },
    }
    return SynchronousPPORunner(
        CentralVecEnv(source), cfg, str(directory), manifest_digest=digest, adaptive=adaptive
    )


def fake_service(count, override):
    return FakeSource(override["source"], count)


def test_real_four_process_timing_reaches_completed_ppo_window(tmp_path):
    sources = [SourceSpec(f"source{i}", fake_service, 2, {"source": i}) for i in range(4)]
    runner = make_runner(tmp_path, source=SynchronousEnv(sources))
    runner.learn(2)
    assert runner.optimizer_steps == 40 and runner.total_transitions == 384
    metrics = runner.last_metrics
    for source in metrics["sources"].values():
        assert source["timing_steps"] == 24
        assert 0 <= source["env_step_seconds"] <= source["request_response_seconds"]
        assert 0 <= source["barrier_wait_seconds"]
        assert (
            source["request_response_seconds"] + source["barrier_wait_seconds"]
            <= metrics["collect_seconds"]
        )


def test_native_budget_checkpoint_and_resume(tmp_path):
    runner = make_runner(tmp_path / "first")
    runner.learn(2)
    assert runner.env.env.closed
    assert runner.next_iteration == runner.policy_version == 2
    assert runner.normalizer_version == 48 and runner.optimizer_steps == 40
    assert runner.total_transitions == 384
    assert runner.last_metrics["epoch_samples"] == [192] * 5
    records = [
        json.loads(line)
        for line in (tmp_path / "first/synchronous_metrics.jsonl").read_text().splitlines()
    ]
    assert [record["iteration"] for record in records] == [0, 1]
    assert list(records[-1]["sources"]) == [f"source{i}" for i in range(4)]
    assert sorted(path.name for path in (tmp_path / "first").glob("*.pt")) == [
        "model_0.pt",
        "model_1.pt",
    ]
    checkpoint = tmp_path / "first/model_1.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["iter"] == 1 and saved["synchronous"]["complete"]
    rng = saved["synchronous"]["rng"]
    torch.set_rng_state(rng["torch"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    expected_rng = (torch.rand(3), random.random(), np.random.rand())
    resumed = make_runner(tmp_path / "resumed")
    resumed.load(str(checkpoint))
    assert resumed.next_iteration == 2 and resumed.generation == 1
    assert not resumed.collector.episode_ids.any()
    torch.testing.assert_close(torch.rand(3), expected_rng[0], rtol=0, atol=0)
    assert (random.random(), np.random.rand()) == expected_rng[1:]
    assert resumed.alg.learning_rate == saved["synchronous"]["learning_rate"]
    for model, key in (
        (resumed.alg.actor, "actor_state_dict"),
        (resumed.alg.critic, "critic_state_dict"),
    ):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[key][name], rtol=0, atol=0)
    resumed.learn(1)
    assert resumed.next_iteration == 3 and resumed.optimizer_steps == 60
    assert resumed.normalizer_version == 72 and resumed.total_transitions == 576
    assert resumed.latest_window.spec.stamp == (2, 2, 48)
    assert resumed.latest_window.generation == 1
    assert resumed.env.env.closed
    assert (tmp_path / "resumed/model_2.pt").exists()


def test_source_timings_sum_each_24_step_window_and_reach_tensorboard(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    runner = make_runner(tmp_path)
    source = runner.env.env
    step = source.step

    def measured(actions):
        state = step(actions)
        state.info["source_timings"] = {
            key: {
                "env_step_seconds": source.tick * 0.001,
                "request_response_seconds": source.tick * 0.002,
                "barrier_wait_seconds": index * 0.01,
            }
            for index, key in enumerate(source.source_slices)
        }
        return state

    monkeypatch.setattr(source, "step", measured)
    runner.learn(2)
    rows = [
        json.loads(line)
        for line in (tmp_path / "synchronous_metrics.jsonl").read_text().splitlines()
    ]
    events = EventAccumulator(str(tmp_path)).Reload()
    for iteration, row in enumerate(rows):
        expected = sum(range(iteration * 24 + 1, iteration * 24 + 25)) * 0.001
        for index, (key, metrics) in enumerate(row["sources"].items()):
            assert metrics["timing_steps"] == 24
            assert metrics["env_step_seconds"] == pytest.approx(expected)
            assert metrics["request_response_seconds"] == pytest.approx(expected * 2)
            assert metrics["barrier_wait_seconds"] == pytest.approx(index * 0.24)
            assert events.Scalars(f"Source/{key}/env_step_seconds")[
                iteration
            ].value == pytest.approx(expected)
            assert "learn_seconds" not in metrics


@pytest.mark.parametrize("damage", ["missing_step", "missing_source", "nan", "negative"])
def test_partial_or_invalid_timing_cannot_commit_window(tmp_path, monkeypatch, damage):
    runner = make_runner(tmp_path)
    source, step = runner.env.env, runner.env.env.step

    def measured(actions):
        state = step(actions)
        if damage == "missing_step" and source.tick == 1:
            return state
        timing = {
            key: dict(
                env_step_seconds=0.01, request_response_seconds=0.02, barrier_wait_seconds=0.0
            )
            for key in source.source_slices
        }
        if damage == "missing_source":
            timing.pop("source3")
        elif damage in ("nan", "negative"):
            timing["source0"]["env_step_seconds"] = float("nan") if damage == "nan" else -1.0
        state.info["source_timings"] = timing
        return state

    monkeypatch.setattr(source, "step", measured)
    with pytest.raises(ValueError, match="timing"):
        runner.learn(1)
    assert source.closed and runner.last_metrics == {}
    assert not list(tmp_path.glob("model_*.pt"))
    assert not (tmp_path / "synchronous_metrics.jsonl").exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "fingerprint",
        "counter",
        "partial",
        "optimizer",
        "duplicate_parameters",
        "normalizer",
        "truncated",
        "missing_count",
        "nonfinite",
        "moment",
        "source_order",
    ],
)
def test_resume_rejects_incompatible_or_partial_state(tmp_path, corruption):
    runner = make_runner(tmp_path / "first")
    runner.learn(1)
    checkpoint = tmp_path / "first/model_0.pt"
    saved = torch.load(checkpoint, weights_only=False)
    if corruption == "fingerprint":
        saved["synchronous"]["contract"]["manifest_digest"] = "other-assets"
    elif corruption == "counter":
        saved["synchronous"]["next_iteration"] = 0
    elif corruption == "partial":
        saved["synchronous"]["complete"] = False
    elif corruption == "optimizer":
        next(iter(saved["optimizer_state_dict"]["state"].values()))["step"] -= 1
    elif corruption == "duplicate_parameters":
        optimizer = saved["optimizer_state_dict"]
        ids, states = optimizer["param_groups"][0]["params"], optimizer["state"]
        duplicate, missing = next(
            (a, b)
            for a in ids
            for b in ids
            if a < b and states[a]["exp_avg"].shape == states[b]["exp_avg"].shape
        )
        ids[ids.index(missing)] = duplicate
        states.pop(missing)
    elif corruption == "normalizer":
        saved["actor_state_dict"]["obs_normalizer.count"] -= 1
    elif corruption == "missing_count":
        saved["actor_state_dict"].pop("obs_normalizer.count")
    elif corruption == "nonfinite":
        saved["actor_state_dict"]["obs_normalizer._mean"].fill_(float("nan"))
    elif corruption == "moment":
        next(iter(saved["optimizer_state_dict"]["state"].values()))["exp_avg"].fill_(float("nan"))
    elif corruption == "source_order":
        saved["synchronous"]["contract"]["sources"].reverse()
    torch.save(saved, checkpoint)
    if corruption == "truncated":
        checkpoint.write_bytes(checkpoint.read_bytes()[:200])
    resumed = make_runner(tmp_path / "resumed")
    with pytest.raises((ValueError, RuntimeError)):
        resumed.load(str(checkpoint))
    assert resumed.env.env.closed
    assert not list((tmp_path / "resumed").glob("*.pt"))


def test_failed_window_cannot_checkpoint_and_closes_writer(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    with pytest.raises(RuntimeError, match="complete iteration"):
        runner.save(str(tmp_path / "partial.pt"))

    def save_mid_rollout(*_):
        runner.save(str(tmp_path / "partial.pt"))

    monkeypatch.setattr(runner, "_on_step", save_mid_rollout)
    with pytest.raises(RuntimeError, match="complete iteration"):
        runner.learn(1)
    assert runner.env.env.closed and runner.collector.failed
    assert runner.logger.writer.file_writer is None
    assert not list(tmp_path.glob("*.pt"))


def test_optimizer_failure_keeps_previous_complete_checkpoint(tmp_path):
    runner = make_runner(tmp_path)

    def fail_after_first_step_of_second_iteration(optimizer, *_):
        if int(next(iter(optimizer.state.values()))["step"]) == 21:
            raise RuntimeError("injected optimizer failure")

    runner.alg.optimizer.register_step_post_hook(fail_after_first_step_of_second_iteration)
    with pytest.raises(RuntimeError, match="injected optimizer failure"):
        runner.learn(2)
    assert runner.env.env.closed and runner.logger.writer.file_writer is None
    with pytest.raises(RuntimeError, match="complete iteration"):
        runner.save(str(tmp_path / "model_1.pt"))
    assert [path.name for path in tmp_path.glob("*.pt")] == ["model_0.pt"]
    saved = torch.load(tmp_path / "model_0.pt", weights_only=False)
    assert saved["iter"] == 0 and saved["synchronous"]["optimizer_steps"] == 20


def test_native_random_initial_episode_lengths_reach_environment(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    initial_lengths = []
    monkeypatch.setattr(
        runner.env.env,
        "set_episode_length_buf",
        lambda values: initial_lengths.append(values.copy()),
        raising=False,
    )
    runner.learn(1, init_at_random_ep_len=True)
    assert len(initial_lengths) == 1 and initial_lengths[0].shape == (8,)
    assert np.all((initial_lengths[0] >= 0) & (initial_lengths[0] < 50))


def test_writer_initialization_failure_closes_created_writer(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    initialize = runner.logger.init_logging_writer

    def fail_after_initialize():
        initialize()
        raise RuntimeError("logging setup failure")

    monkeypatch.setattr(runner.logger, "init_logging_writer", fail_after_initialize)
    with pytest.raises(RuntimeError, match="logging setup failure"):
        runner.learn(1)
    assert runner.env.env.closed and runner.logger.writer.file_writer is None


def test_failed_atomic_replace_preserves_previous_checkpoint(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    runner.learn(1)
    path = tmp_path / "model_0.pt"
    previous = path.read_bytes()

    def fail_replace(*_):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        runner.save(str(path))
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))
