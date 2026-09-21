"""Retrospective, read-only 500-update summaries of native and shared PPO logs."""

from __future__ import annotations

import csv
import importlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from uni_rl.logging.single_run_audit import audit_single_run
from uni_rl.logging.synchronous_report import (
    SOURCE_ORDER,
    SOURCE_TIMES,
    _read_audited,
    _require,
    _sha256,
)

TIMES = ("Perf/collection_time", "Perf/learning_time")
PPO = ("Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate", "Policy/mean_std")
TRACKING = tuple(
    f"error_{part}_{kind}"
    for part, kinds in (
        ("anchor", ("pos", "rot", "lin_vel", "ang_vel")),
        ("body", ("pos", "rot", "lin_vel", "ang_vel")),
        ("joint", ("pos", "vel")),
    )
    for kind in kinds
)
EPISODES = {"episode_return": "Train/mean_reward", "episode_length": "Train/mean_episode_length"}


def _read_events(run: Path, iterations: int) -> tuple[dict, dict, dict]:
    cls = importlib.import_module("tensorboard.backend.event_processing.event_accumulator")
    events = cls.EventAccumulator(
        str(run), size_guidance={"scalars": 0}, purge_orphaned_data=False
    ).Reload()
    result, clock = {}, {}
    inventory: dict = {}
    for tag in sorted(events.Tags()["scalars"]):
        records = events.Scalars(tag)
        inventory[tag] = {"records": len(records), "included": not tag.endswith("/time")}
        if tag.endswith("/time"):
            inventory[tag]["reason"] = "Seconds-axis duplicate of iteration-axis metric."
            continue
        values: list[float | None] = [None] * iterations
        for event in records:
            i, value = event.step, event.value
            _require(type(i) is int and 0 <= i < iterations, f"invalid iteration: {tag}")
            _require(values[i] is None, f"duplicate iteration: {tag}")
            _require(math.isfinite(value), f"non-finite scalar: {tag}")
            values[i] = float(value)
            if tag == TIMES[0]:
                _require(math.isfinite(event.wall_time), "invalid event timestamp")
                clock[i] = event.wall_time
        result[tag] = values
    for tag in (*TIMES, *PPO):
        _require(tag in result and all(v is not None for v in result[tag]), f"missing {tag}")
    _require(all(v is not None and v >= 0 for tag in TIMES for v in result[tag]), "negative timing")
    _require(
        all(
            a is not None and b is not None and a + b > 0
            for a, b in zip(result[TIMES[0]], result[TIMES[1]])
        ),
        "zero loop duration",
    )
    _require(all(clock[i] >= clock[i - 1] for i in range(1, iterations)), "event clock reversed")
    return result, inventory, clock


def _stats(values: list, start: int, stop: int) -> dict:
    observed = [v for v in values[start:stop] if v is not None]
    result = {"observed_updates": len(observed), "missing_updates": stop - start - len(observed)}
    return {
        **result,
        **dict.fromkeys(("mean", "std", "min", "p95", "max", "last"), None),
        **(
            dict(
                mean=float(np.mean(observed)),
                std=float(np.std(observed)),
                min=min(observed),
                p95=float(np.percentile(observed, 95)),
                max=max(observed),
                last=observed[-1],
            )
            if observed
            else {}
        ),
    }


def _bounds(iterations: int, interval: int):
    for start in range(0, iterations, interval):
        stop = min(start + interval, iterations)
        yield (
            start,
            stop,
            {
                "first_completed_update": start + 1,
                "last_completed_update": stop,
                "first_iteration": start,
                "last_iteration": stop - 1,
                "updates": stop - start,
            },
        )


def _csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(
    policy: str, events: dict, clock: dict, iterations: int, interval: int, batch: int, joint: list
) -> list[dict]:
    rows, cumulative = [], 0.0
    for start, stop, boundary in _bounds(iterations, interval):
        collect = [r["collect_seconds"] for r in joint] if joint else events[TIMES[0]]
        learn = [r["learn_seconds"] for r in joint] if joint else events[TIMES[1]]
        c, l = sum(collect[start:stop]), sum(learn[start:stop])
        cumulative += c + l
        row = dict(
            policy=policy,
            **boundary,
            interval_transitions=(stop - start) * batch,
            cumulative_transitions=stop * batch,
            interval_optimizer_steps=(stop - start) * 20,
            cumulative_optimizer_steps=stop * 20,
            collection_seconds=c,
            learning_seconds=l,
            loop_seconds=c + l,
            cumulative_loop_seconds=cumulative,
            transitions_per_loop_second=(stop - start) * batch / (c + l),
            event_boundary_elapsed_seconds=clock[stop - 1] - clock[start - 1] if start else None,
            interval_last_event_utc=datetime.fromtimestamp(
                clock[stop - 1], timezone.utc
            ).isoformat(),
        )
        for name, tag in {**EPISODES, **{tag: tag for tag in PPO}}.items():
            stat = _stats(events.get(tag, [None] * iterations), start, stop)
            row[name + "_mean"] = stat["mean"]
            row[name + "_observed_updates"] = stat["observed_updates"]
        rows.append(row)
    return rows


def _source_rows(
    policy: str, events: dict, iterations: int, interval: int, batch: int, joint: list
) -> list[dict]:
    rows = []
    for source in SOURCE_ORDER if joint else (policy,):
        prefix = source + "/" if joint else ""
        for start, stop, boundary in _bounds(iterations, interval):
            row: dict = dict(policy=policy, source=source, **boundary)
            row["interval_source_transitions"] = (stop - start) * batch
            row["cumulative_source_transitions"] = stop * batch
            tags = {name: prefix + "Metrics/motion/" + name for name in TRACKING}
            for tag in events:
                if tag.startswith(prefix + "reward/"):
                    tags[tag.removeprefix(prefix)] = tag
            for name, tag in {**EPISODES, **tags}.items():
                values = (
                    [r["sources"][source][name] for r in joint]
                    if joint and name in EPISODES
                    else events.get(tag, [None] * iterations)
                )
                stat = _stats(values, start, stop)
                row[name + "_mean"] = stat["mean"]
                row[name + "_observed_updates"] = stat["observed_updates"]
            if joint:
                data = [r["sources"][source] for r in joint[start:stop]]
                for key in (*SOURCE_TIMES, "terminated", "truncated"):
                    row[key + "_sum"] = sum(r[key] for r in data)
                row["true_termination_flags_per_transition"] = (
                    row["terminated_sum"] / row["interval_source_transitions"]
                )
                row["timeout_flags_per_transition"] = (
                    row["truncated_sum"] / row["interval_source_transitions"]
                )
                row["transition_reward_mean"] = float(np.mean([r["reward"] for r in data]))
            rows.append(row)
    return rows


def _plots(output: Path, intervals: list[dict], sources: list[dict]) -> None:
    importlib.import_module("matplotlib").use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for ax, key, label in zip(
            axes.flat,
            (
                "loop_seconds",
                "cumulative_loop_seconds",
                "transitions_per_loop_second",
                "Loss/value_mean",
            ),
            (
                "Seconds per interval",
                "Cumulative loop seconds",
                "Transitions / loop second",
                "Value loss (interval mean)",
            ),
        ):
            for policy in (*SOURCE_ORDER, "joint"):
                rows = [r for r in intervals if r["policy"] == policy]
                ax.plot(
                    [r["cumulative_transitions"] / 1e6 for r in rows],
                    [r[key] for r in rows],
                    label=policy,
                )
            ax.set(xlabel="Global transitions (millions)", ylabel=label)
            ax.legend(fontsize=8)
        fig.suptitle("500-update intervals | joint interval has four times as many samples")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"timing_ppo.{suffix}", dpi=150)
        plt.close(fig)
        for filename, metrics in (
            ("episodes", ("episode_return", "episode_length")),
            (
                "tracking",
                ("error_anchor_pos", "error_body_pos", "error_joint_pos", "error_body_rot"),
            ),
        ):
            fig, axes = plt.subplots(
                len(metrics),
                4,
                figsize=(17, 3 * len(metrics)),
                constrained_layout=True,
                squeeze=False,
            )
            for column, source in enumerate(SOURCE_ORDER):
                for row, metric in enumerate(metrics):
                    ax = axes[row, column]
                    for policy, factor, label in (
                        (source, 1, "single"),
                        ("joint", 4, "shared PPO"),
                    ):
                        data = [
                            r for r in sources if r["policy"] == policy and r["source"] == source
                        ]
                        ax.plot(
                            [r["cumulative_source_transitions"] * factor / 1e6 for r in data],
                            [r[metric + "_mean"] for r in data],
                            label=label,
                        )
                    ax.set(title=source, xlabel="Global transitions (millions)", ylabel=metric)
                    ax.legend(fontsize=8)
            fig.suptitle(
                "500-update means; tracking sampled only at resets, not trajectory error"
                if filename == "tracking"
                else "Rolling training episode statistics; not holdout success rates"
            )
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"{filename}.{suffix}", dpi=150)
            plt.close(fig)


def report(
    single_root: str | Path,
    joint_run: str | Path,
    output_dir: str | Path,
    *,
    interval: int = 500,
    single_iterations: int = 20000,
    joint_iterations: int = 5000,
    num_envs: int = 1024,
) -> dict:
    """Audit finals, preserve raw scalar semantics, and summarize completed updates."""
    _require(type(interval) is int and interval > 0, "positive integer interval required")
    root, joint_path, output = (Path(p).resolve() for p in (single_root, joint_run, output_dir))
    _require(not output.exists(), "interval output must be new")
    _require(
        all(output != p and output not in p.parents for p in (root, joint_path)),
        "output cannot contain training inputs",
    )
    _require(single_iterations == joint_iterations * 4, "unequal final per-policy sample budgets")
    audits, native, inventories, provenance = {}, {}, {}, {}
    for source in SOURCE_ORDER:
        path = root / source
        audits[source] = audit_single_run(
            path, expected_iterations=single_iterations, num_envs=num_envs
        )
        _require(audits[source]["source"] == source, "source directory mismatch")
        native[source] = _read_events(path, single_iterations)
    joint_audit, joint = _read_audited(joint_path, joint_iterations, num_envs, None)
    _require(joint_audit["timing"]["missing_iterations"] == 0, "missing joint source timing")
    _require(
        all(a["total_transitions"] == joint_audit["total_transitions"] for a in audits.values()),
        "final audited budgets differ",
    )
    native["joint"] = _read_events(joint_path, joint_iterations)
    intervals, sources, scalars, checkpoint_clocks = [], [], [], []
    for policy, (events, inventory, clock) in native.items():
        shared = policy == "joint"
        path = joint_path if shared else root / policy
        count, batch = (
            (joint_iterations, num_envs * 96) if shared else (single_iterations, num_envs * 24)
        )
        rows = joint if shared else []
        intervals.extend(_summary_rows(policy, events, clock, count, interval, batch, rows))
        sources.extend(_source_rows(policy, events, count, interval, num_envs * 24, rows))
        inventories[policy] = inventory
        provenance[policy] = {
            p.name: _sha256(p) for p in sorted(path.glob("events.out.tfevents.*"))
        }
        if shared:
            provenance[policy]["synchronous_metrics.jsonl"] = _sha256(
                path / "synchronous_metrics.jsonl"
            )
        for start, stop, boundary in _bounds(count, interval):
            for tag, values in events.items():
                scalars.append(
                    dict(policy=policy, **boundary, tag=tag, **_stats(values, start, stop))
                )
        loops = np.cumsum(
            [r["collect_seconds"] + r["learn_seconds"] for r in rows]
            if shared
            else [a + b for a, b in zip(events[TIMES[0]], events[TIMES[1]])]
        )
        for checkpoint in sorted(path.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1])):
            iteration = int(checkpoint.stem.split("_")[-1])
            _require(0 <= iteration < count, "checkpoint outside logged updates")
            checkpoint_clocks.append(
                dict(
                    policy=policy,
                    checkpoint=str(checkpoint),
                    iteration=iteration,
                    completed_updates=iteration + 1,
                    cumulative_transitions=(iteration + 1) * batch,
                    cumulative_optimizer_steps=(iteration + 1) * 20,
                    cumulative_loop_seconds=float(loops[iteration]),
                )
            )
    result = dict(
        scope="retrospective_training_interval_statistics",
        interval_updates=interval,
        single_audits=audits,
        joint_audit=joint_audit,
        provenance_sha256=provenance,
        metric_inventory=inventories,
        intervals=intervals,
        source_intervals=sources,
        checkpoint_clocks=checkpoint_clocks,
        interpretation=[
            "Interval 1-500 contains zero-based iterations 0-499; model_500.pt has completed 501 updates. Final model_19999/model_4999 are 20000/5000 updates.",
            "All scalar statistics are equally weighted over observed update-level log values; missing values remain missing. std is descriptive population SD, not seed uncertainty.",
            "Metrics/motion/error_* are reset-time post-step errors, averaged over resetting environments then logged reset steps; not time-averaged trajectory errors or independent evaluations.",
            "Anchor position/rotation/velocity errors are L2/angle errors; body metrics average per-body errors; joint errors are full-vector L2, not per-joint RMSE. Position m, rotation rad, velocity m/s or rad/s.",
            "reward/* are weighted pre-dt reward rates; Episode_Reward/* is completed-episode reward sum divided by fixed max episode seconds. These differ from rolling episodic return.",
            "Episode_Termination/* are original reset-step count summaries, not episode failure probabilities; simultaneous reasons can overlap. Native exact failure/timeout totals cannot be reconstructed from these means.",
            "Joint terminated/truncated sums are flag counts; simultaneous flags may overlap, so their sum is not a unique episode count. Per-transition flag frequencies are not success rates.",
            "Joint source timings overlap; do not sum them as elapsed time. Native GAE is in learning, joint GAE in collection: compare total loop.",
            "Loop time excludes startup, some logging and checkpoint overhead. Event-boundary elapsed includes between-event overhead; first interval is unknown because no preceding event exists.",
            "500 joint updates contain 4x the samples of 500 native updates. Final total samples match; native has 4x the optimizer updates. This is observational timing, not an isolated benchmark.",
            "This retrospective report does not rank/select checkpoints or feed MuJoCo outcomes back into training.",
        ],
    )
    output.mkdir(parents=True)
    for filename, data in (
        ("intervals", intervals),
        ("source_intervals", sources),
        ("scalar_intervals", scalars),
        ("checkpoint_clocks", checkpoint_clocks),
    ):
        _csv(output / f"{filename}.csv", data)
    _plots(output, intervals, sources)
    (output / "intervals.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
