"""Bounded acceptance runs using the unchanged RSL-RL PPO learning loop."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch


class _BudgetCompleteError(Exception):
    """An update completed and the acceptance budget has been met."""


def _assert_finite(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite PPO validation value: {name}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{name}/{key}")
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise FloatingPointError(f"Non-finite PPO validation value: {name}")


class _ValidationLogger:
    def __init__(self, logger: Any, on_update: Callable[[dict[str, Any]], None]):
        self._logger = logger
        self._on_update = on_update

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)

    def log(self, **kwargs: Any) -> None:
        self._logger.log(**kwargs)
        self._on_update(kwargs)


def run_bounded_ppo(
    runner: Any,
    *,
    output_dir: Path,
    max_updates: int | None = None,
    duration_seconds: float | None = None,
    warmup_updates: int = 1,
    source_statistics: Callable[[], Mapping[str, Mapping[str, Any]]] | None = None,
    sample_resources: Callable[[], Mapping[str, Any]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Stop only after a complete PPO update, save, and report measured evidence.

    Exactly one budget is required. ``max_updates`` counts all updates, whereas
    ``duration_seconds`` excludes the first ``warmup_updates``. The temporary
    logger adapter observes the stock runner's post-update boundary; it never
    interrupts an environment operation or changes rollout/GAE/loss behavior.
    A failure propagates without producing an acceptance checkpoint or report.
    The caller retains ownership of environment cleanup.
    """
    if (max_updates is None) == (duration_seconds is None):
        raise ValueError("Specify exactly one of max_updates or duration_seconds")
    if max_updates is not None and (
        isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates < 1
    ):
        raise ValueError("max_updates must be a positive integer")
    if duration_seconds is not None and (
        not math.isfinite(duration_seconds) or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be positive and finite")
    if (
        isinstance(warmup_updates, bool)
        or not isinstance(warmup_updates, int)
        or warmup_updates < 0
    ):
        raise ValueError("warmup_updates must be a non-negative integer")
    if getattr(runner, "is_distributed", False):
        raise ValueError("Bounded PPO validation requires a single learner")
    samples_per_update = int(runner.env.num_envs) * int(runner.cfg["num_steps_per_env"])
    minibatches = int(runner.cfg["algorithm"]["num_mini_batches"])
    epochs = int(runner.cfg["algorithm"]["num_learning_epochs"])
    if minibatches <= 0 or epochs <= 0 or samples_per_update % minibatches:
        raise ValueError("Every PPO epoch must consume the entire rollout without dropped samples")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "validation_final.pt"
    report_path = output_dir / "validation_report.json"
    updates_path = output_dir / "updates.jsonl"
    if any(target.exists() for target in (checkpoint, report_path, updates_path)):
        raise FileExistsError(
            "Use a fresh validation output directory; evidence is never overwritten"
        )
    original_logger = runner.logger
    completed_updates = 0
    measured_updates = 0
    started = clock()
    measured_start = started if warmup_updates == 0 else None
    last_finished = started
    previous_stats = (
        {name: dict(stats) for name, stats in source_statistics().items()}
        if source_statistics is not None
        else {}
    )
    latest_stats: dict[str, Any] = previous_stats

    with updates_path.open("x", encoding="utf-8") as update_stream:

        def record_update(values: dict[str, Any]) -> None:
            nonlocal completed_updates, measured_updates, measured_start, last_finished
            nonlocal previous_stats, latest_stats
            _assert_finite(values["loss_dict"], "loss")
            _assert_finite(values["action_std"], "action_std")
            _assert_finite(values["learning_rate"], "learning_rate")
            for model_name in ("actor", "critic"):
                for parameter_name, parameter in getattr(runner.alg, model_name).named_parameters():
                    _assert_finite(parameter, f"{model_name}/{parameter_name}")
            samples: dict[str, int] = {}
            if source_statistics is not None:
                latest_stats = {name: dict(stats) for name, stats in source_statistics().items()}
                if latest_stats.keys() != previous_stats.keys():
                    raise RuntimeError("Environment sources changed during PPO validation")
                for name, stats in latest_stats.items():
                    expected = int(stats["num_envs"]) * int(runner.cfg["num_steps_per_env"])
                    samples[name] = int(stats["transitions"]) - int(
                        previous_stats[name]["transitions"]
                    )
                    if samples[name] != expected:
                        raise RuntimeError(
                            f"Source {name}: expected {expected} samples, got {samples[name]}"
                        )
                if sum(samples.values()) != samples_per_update:
                    raise RuntimeError("Source quotas do not cover the complete PPO rollout")
                previous_stats = latest_stats
            finished = clock()
            completed_updates += 1
            is_measured = completed_updates > warmup_updates
            if is_measured:
                measured_updates += 1
            elif completed_updates == warmup_updates:
                measured_start = finished
            measured_elapsed = 0.0 if measured_start is None else finished - measured_start
            record = {
                "update": completed_updates,
                "runner_iteration": int(values["it"]),
                "warmup": not is_measured,
                "collect_seconds": float(values["collect_time"]),
                "update_seconds": float(values["learn_time"]),
                "wall_seconds": finished - last_finished,
                "measured_elapsed_seconds": measured_elapsed,
                "samples": samples_per_update,
                "source_samples": samples,
                "source_samples_per_epoch": samples,
                "ppo_epochs": epochs,
                "source_statistics": latest_stats,
                "resources": dict(sample_resources()) if sample_resources is not None else {},
            }
            update_stream.write(json.dumps(record, allow_nan=False) + "\n")
            update_stream.flush()
            last_finished = finished
            if completed_updates == warmup_updates:
                measured_start = clock()
            if max_updates is not None and completed_updates >= max_updates:
                raise _BudgetCompleteError
            if (
                duration_seconds is not None
                and is_measured
                and measured_elapsed >= duration_seconds
            ):
                raise _BudgetCompleteError

        runner.logger = _ValidationLogger(original_logger, record_update)
        try:
            try:
                runner.learn(
                    num_learning_iterations=max_updates if max_updates is not None else 2**31 - 1,
                    init_at_random_ep_len=True,
                )
            except _BudgetCompleteError:
                pass
            else:
                raise RuntimeError("PPO runner returned without reaching the validation boundary")
            elapsed = last_finished - started
            measured_elapsed = 0.0 if measured_start is None else last_finished - measured_start
            result = {
                "status": "passed",
                "checkpoint": str(checkpoint.resolve()),
                "updates": completed_updates,
                "warmup_updates": min(completed_updates, warmup_updates),
                "measured_updates": measured_updates,
                "elapsed_seconds": elapsed,
                "measured_elapsed_seconds": measured_elapsed,
                "requested_duration_seconds": duration_seconds,
                "samples_per_update": samples_per_update,
                "transitions": completed_updates * samples_per_update,
                "measured_transitions_per_second": (
                    measured_updates * samples_per_update / measured_elapsed
                    if measured_elapsed > 0
                    else None
                ),
                "source_statistics": latest_stats,
                "updates_file": str(updates_path.resolve()),
            }
            runner.save(str(checkpoint), infos={"bounded_validation": result})
            report_path.write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            return result
        finally:
            runner.logger = original_logger
            original_logger.stop_logging_writer()
