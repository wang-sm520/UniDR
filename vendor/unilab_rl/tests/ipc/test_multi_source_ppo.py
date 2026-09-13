"""Stock CPU PPO rollout, update, epoch source coverage and checkpoint integration."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from multi_source_fakes import make_deterministic_env
from rsl_rl.runners import OnPolicyRunner

from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper
from uni_rl.algos.rsl_rl_source_timing import record_source_timing
from uni_rl.algos.rsl_rl_validation import run_bounded_ppo
from uni_rl.ipc.multi_source_env import EnvSourceSpec, MultiSourceOptions, make_multi_source_env


@pytest.mark.parametrize("algorithm", ["PPO", "uni_rl.algos.rsl_rl_ppo:FinalObservationAwarePPO"])
@pytest.mark.parametrize("bounded", [False, True])
def test_stock_cpu_ppo_24_step_update_checkpoint_and_epoch_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, algorithm: str, bounded: bool
) -> None:
    torch.manual_seed(73)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env: Any = make_multi_source_env(
        [
            EnvSourceSpec("left", make_deterministic_env, 2, {"marker": 10}, seed=11),
            EnvSourceSpec("right", make_deterministic_env, 2, {"marker": 20}, seed=12),
            EnvSourceSpec("front", make_deterministic_env, 2, {"marker": 30}, seed=13),
            EnvSourceSpec("back", make_deterministic_env, 2, {"marker": 40}, seed=14),
        ],
        options=MultiSourceOptions(30, 10, 1),
    )
    with ExitStack() as stack:
        stack.callback(torch.set_num_threads, previous_threads)
        stack.callback(env.close)
        wrapper = RslRlVecEnvWrapper(env, device="cpu")
        model = {"class_name": "MLPModel", "hidden_dims": [16], "obs_normalization": True}
        config = {
            "actor": {
                **model,
                "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
            },
            "critic": dict(model),
            "algorithm": {"class_name": algorithm, "num_learning_epochs": 2, "num_mini_batches": 4},
            "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
            "num_steps_per_env": 24,
            "save_interval": 10,
        }
        restore_config = deepcopy(config)
        runner = OnPolicyRunner(wrapper, config, device="cpu")
        initial_parameters = {
            name: value.clone() for name, value in runner.alg.actor.state_dict().items()
        }
        original_generator = runner.alg.storage.mini_batch_generator
        epoch_counts: list[Counter[int]] = []
        loss_snapshots: list[dict[str, Any]] = []
        original_log = runner.logger.log

        def observe_epochs(num_mini_batches: int, num_epochs: int = 8) -> Iterator[Any]:
            counts: Counter[int] = Counter()
            for batch_number, batch in enumerate(original_generator(num_mini_batches, num_epochs)):
                observations = batch.observations
                labels = observations["policy"][:, 0].to(torch.int64).tolist()
                counts.update(labels)
                yield batch
                if (batch_number + 1) % num_mini_batches == 0:
                    epoch_counts.append(counts)
                    counts = Counter()

        def observe_log(**values: Any) -> Any:
            loss_snapshots.append(dict(values["loss_dict"]))
            return original_log(**values)

        monkeypatch.setattr(runner.alg.storage, "mini_batch_generator", observe_epochs)
        monkeypatch.setattr(runner.logger, "log", observe_log)
        stack.enter_context(
            record_source_timing(
                runner,
                output_path=tmp_path / "source_timing.jsonl",
                source_statistics=lambda: env.source_statistics,
            )
        )
        if bounded:
            result = run_bounded_ppo(
                runner,
                output_dir=tmp_path / "bounded",
                max_updates=1,
                source_statistics=lambda: env.source_statistics,
            )
            assert result["updates"] == 1
            assert result["samples_per_update"] == 192
            assert Path(result["checkpoint"]).is_file()
        else:
            runner.learn(1, init_at_random_ep_len=True)
        assert len(loss_snapshots) == 1
        timing = json.loads((tmp_path / "source_timing.jsonl").read_text())
        assert timing["source_order"] == ["left", "right", "front", "back"]
        assert timing["rollout_steps"] == 24
        for source in timing["sources"].values():
            assert source["transitions"] == 48
            assert len(source["step_seconds"]) == 24
            assert source["step_seconds_sum"] > 0
        assert all(
            torch.isfinite(torch.as_tensor(value)).all() for value in loss_snapshots[0].values()
        )
        assert len(epoch_counts) == 2
        for counts in epoch_counts:
            assert counts == Counter({label: 24 for label in (10, 11, 20, 21, 30, 31, 40, 41)})
            assert {counts[marker] + counts[marker + 1] for marker in (10, 20, 30, 40)} == {48}
        for stats in env.source_statistics.values():
            assert stats["step_calls"] == 24
            assert stats["transitions"] == 48
        updated_parameters = runner.alg.actor.state_dict()
        assert all(torch.isfinite(value).all() for value in updated_parameters.values())
        assert any(
            not torch.equal(initial_parameters[name], value)
            for name, value in updated_parameters.items()
        )
        checkpoint_path = tmp_path / "completed_update.pt"
        runner.save(str(checkpoint_path), infos={"rollout_steps": 24})
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["optimizer_state_dict"]["state"]
        assert checkpoint["infos"] == {"rollout_steps": 24}
        restored = OnPolicyRunner(wrapper, restore_config, device="cpu")
        restored.load(str(checkpoint_path), map_location="cpu")
        for name, value in restored.alg.actor.state_dict().items():
            torch.testing.assert_close(value, checkpoint["actor_state_dict"][name])
