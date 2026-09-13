"""Per-source rollout timing without modifying the PPO learning loop or IPC."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

import numpy as np

SourceStatistics = Callable[[], Mapping[str, Mapping[str, Any]]]


class _SourceTimingLogger:
    def __init__(
        self,
        logger: Any,
        statistics: SourceStatistics,
        stream: TextIO,
        rollout_steps: int,
        num_envs: int,
    ) -> None:
        self._logger = logger
        self._statistics = statistics
        self._stream = stream
        self._rollout_steps = rollout_steps
        self._previous = self._snapshot()
        if (
            not self._previous
            or sum(int(stats["num_envs"]) for stats in self._previous.values()) != num_envs
        ):
            raise ValueError("Source timing requires statistics covering every environment")
        self._steps: dict[str, list[float]] = {name: [] for name in self._previous}
        self._timings: dict[str, dict[str, float]] = {name: {} for name in self._previous}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)

    def _snapshot(self) -> dict[str, dict[str, Any]]:
        return {name: dict(stats) for name, stats in self._statistics().items()}

    def process_env_step(self, *args: Any, **kwargs: Any) -> None:
        self._logger.process_env_step(*args, **kwargs)
        current = self._snapshot()
        if tuple(current) != tuple(self._previous):
            raise RuntimeError("Source timing: source order changed during rollout")
        for name, stats in current.items():
            previous = self._previous[name]
            num_envs = int(stats["num_envs"])
            if (
                num_envs != int(previous["num_envs"])
                or int(stats["step_calls"]) - int(previous["step_calls"]) != 1
                or int(stats["transitions"]) - int(previous["transitions"]) != num_envs
            ):
                raise RuntimeError(f"Source timing: {name} did not publish exactly one full step")
            elapsed = float(stats["last_step_seconds"])
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError(f"Source timing: {name} has invalid step duration")
            self._steps[name].append(elapsed)
            for key, total in stats.items():
                if key.startswith("timing/") and key.endswith("/total"):
                    timer = key.removeprefix("timing/").removesuffix("/total")
                    delta = float(total) - float(previous.get(key, 0.0))
                    if not math.isfinite(delta):
                        raise ValueError(f"Source timing: {name}/{timer} has invalid timer delta")
                    self._timings[name][timer] = self._timings[name].get(timer, 0.0) + delta
        self._previous = current

    def log(self, **kwargs: Any) -> None:
        if any(len(steps) != self._rollout_steps for steps in self._steps.values()):
            raise RuntimeError("Source timing: cannot report an incomplete PPO rollout")
        sources: dict[str, dict[str, Any]] = {}
        step_matrix = np.asarray(list(self._steps.values()), dtype=np.float64)
        step_maxima = step_matrix.max(axis=0)
        for source_index, (name, steps) in enumerate(self._steps.items()):
            stats = self._previous[name]
            sources[name] = {
                "num_envs": int(stats["num_envs"]),
                "step_calls": len(steps),
                "transitions": len(steps) * int(stats["num_envs"]),
                "step_seconds": steps,
                "step_seconds_sum": float(sum(steps)),
                "step_seconds_mean": float(np.mean(steps)),
                "step_seconds_p50": float(np.percentile(steps, 50)),
                "step_seconds_p95": float(np.percentile(steps, 95)),
                "step_seconds_max": float(max(steps)),
                "slowest_steps": int(np.count_nonzero(step_matrix[source_index] == step_maxima)),
                "environment_timing_totals": self._timings[name],
                "pid": stats.get("pid"),
                "rss_bytes": stats.get("rss_bytes"),
            }
        record = {
            "schema_version": 1,
            "runner_iteration": int(kwargs["it"]),
            "rollout_steps": self._rollout_steps,
            "source_order": list(sources),
            "collect_seconds": float(kwargs["collect_time"]),
            "learn_seconds": float(kwargs["learn_time"]),
            "max_source_step_seconds_sum": float(step_maxima.sum()),
            "collection_minus_max_source_step_seconds": float(kwargs["collect_time"])
            - float(step_maxima.sum()),
            "sources": sources,
        }
        serialized = json.dumps(record, allow_nan=False)
        self._logger.log(**kwargs)
        self._stream.write(serialized + "\n")
        self._stream.flush()
        writer = self._logger.writer
        if writer is not None:
            for name, source in sources.items():
                for metric in (
                    "step_seconds_sum",
                    "step_seconds_mean",
                    "step_seconds_p50",
                    "step_seconds_p95",
                    "step_seconds_max",
                    "slowest_steps",
                    "transitions",
                    "rss_bytes",
                ):
                    if source[metric] is not None:
                        writer.add_scalar(
                            f"source/{name}/sampling/{metric}", source[metric], kwargs["it"]
                        )
                for timer, total in self._timings[name].items():
                    writer.add_scalar(
                        f"source/{name}/timing/{timer}/rollout_sum", total, kwargs["it"]
                    )
            for metric in (
                "max_source_step_seconds_sum",
                "collection_minus_max_source_step_seconds",
            ):
                writer.add_scalar(f"Perf/{metric}", record[metric], kwargs["it"])
        summary = " | ".join(
            f"{name}: {1000 * source['step_seconds_mean']:.1f} ms/step "
            f"(p95 {1000 * source['step_seconds_p95']:.1f}), "
            f"{source['step_seconds_sum']:.3f} s/rollout"
            for name, source in sources.items()
        )
        print(f"Source sampling [{record['runner_iteration']}]: {summary}", flush=True)
        for name in self._steps:
            self._steps[name] = []
            self._timings[name] = {}


@contextmanager
def record_source_timing(
    runner: Any,
    *,
    output_path: Path,
    source_statistics: SourceStatistics,
) -> Iterator[None]:
    """Record every source step and completed rollout in JSONL and the PPO writer.

    Step seconds are worker wall time inside env.step, including autoresets but
    excluding the outer IPC transfer. Existing environment timers retain their
    original units and can overlap. The sum of per-step maximum worker durations
    is not a measured barrier duration; its residual against collection includes
    inference, IPC, copying, scheduling and logger overhead, not just IPC.
    No CUDA synchronization is added. Files are created exclusively, never reused.
    """
    if getattr(runner, "is_distributed", False):
        raise ValueError("Source timing requires a single learner")
    rollout_steps = int(runner.cfg["num_steps_per_env"])
    if rollout_steps < 1:
        raise ValueError("Source timing requires a positive rollout length")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_logger = runner.logger
    with output_path.open("x", encoding="utf-8") as stream:
        runner.logger = _SourceTimingLogger(
            original_logger, source_statistics, stream, rollout_steps, int(runner.env.num_envs)
        )
        try:
            yield
        finally:
            runner.logger = original_logger
