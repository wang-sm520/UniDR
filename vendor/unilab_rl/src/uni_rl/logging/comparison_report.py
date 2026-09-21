"""Compare completed native and synchronous PPO runs at equal sample budgets."""

from __future__ import annotations

import csv
import importlib
import json
import math
from pathlib import Path

from uni_rl.logging.single_run_audit import audit_single_run
from uni_rl.logging.synchronous_report import SOURCE_ORDER, _read_audited, _require, _sha256

METRICS = {"episode_return": "Train/mean_reward", "episode_length": "Train/mean_episode_length"}
TIMES = ("Perf/collection_time", "Perf/learning_time")


def _native_series(run: Path, iterations: int) -> dict[str, list[float | None]]:
    cls = importlib.import_module("tensorboard.backend.event_processing.event_accumulator")
    events = cls.EventAccumulator(
        str(run), size_guidance={"scalars": 0}, purge_orphaned_data=False
    ).Reload()
    available = events.Tags()["scalars"]
    result: dict[str, list[float | None]] = {}
    for name, tag in {**METRICS, **{tag: tag for tag in TIMES}}.items():
        values: list[float | None] = [None] * iterations
        for event in events.Scalars(tag) if tag in available else ():
            i, value = event.step, event.value
            _require(type(i) is int and 0 <= i < iterations, f"invalid iteration: {tag}")
            _require(values[i] is None, f"duplicate iteration: {tag}")
            _require(math.isfinite(value), f"non-finite scalar: {tag}")
            values[i] = float(value)
        if tag in TIMES:
            _require(
                all(v is not None and v >= 0 for v in values), f"missing/negative timing: {tag}"
            )
        result[name] = values
    _require(
        [v is None for v in result["episode_return"]]
        == [v is None for v in result["episode_length"]],
        "episode return/length coverage mismatch",
    )
    _require(all(v is None or v > 0 for v in result["episode_length"]), "invalid episode length")
    result["loop_seconds"] = [
        float(a) + float(b)
        for a, b in zip(result[TIMES[0]], result[TIMES[1]])
        if a is not None and b is not None
    ]
    _require(all(v is not None and v > 0 for v in result["loop_seconds"]), "zero loop duration")
    return {name: result[name] for name in (*METRICS, "loop_seconds")}


def _smooth(values: list[float | None], window: int) -> list[float]:
    """Same transition-width window; missing observations stay missing."""
    total, count = 0.0, 0
    result = []
    for i, value in enumerate(values):
        if value is not None:
            total, count = total + value, count + 1
        old = values[i - window] if i >= window else None
        if old is not None:
            total, count = total - old, count - 1
        result.append(total / count if value is not None and count else float("nan"))
    return result


def _recent(values: list[float | None], window: int) -> dict:
    tail = [v for v in values[-window:] if v is not None]
    return {"observed_updates": len(tail), "mean": sum(tail) / len(tail) if tail else None}


def _plots(output: Path, singles: dict, joint: dict, batch: int, windows: tuple[int, int]) -> None:
    importlib.import_module("matplotlib").use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
        for column, source in enumerate(SOURCE_ORDER):
            for row, metric in enumerate(METRICS):
                ax = axes[row, column]
                for label, series, step, window, color in (
                    ("Single", singles[source], batch, windows[0], "#2876a8"),
                    ("Shared PPO", joint[source], batch * 4, windows[1], "#c96a29"),
                ):
                    values = series[metric]
                    x = [(i + 1) * step / 1e6 for i in range(len(values))]
                    ax.plot(x, values, color=color, alpha=0.15, linewidth=0.6)
                    ax.plot(x, _smooth(values, window), label=label, color=color, linewidth=1.5)
                ax.set(title=source, xlabel="Global transitions (millions)", ylabel=metric)
                ax.legend()
        fig.suptitle(
            "Single-source vs shared PPO | same samples per policy, different update budgets"
        )
        fig.supxlabel(
            "Rolling episode statistics; no independent success rate. Reference resets can occur within an episode.\n"
            f"Smoothing spans {windows[0] * batch:,} transitions for both policies; missing metrics stay missing.",
            fontsize=10,
        )
        try:
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"comparison.{suffix}", dpi=150)
        finally:
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for label, series, step in (
            *((source, singles[source], batch) for source in SOURCE_ORDER),
            ("Shared PPO (all sources)", joint[SOURCE_ORDER[0]], batch * 4),
        ):
            elapsed, y = 0.0, []
            for value in series["loop_seconds"]:
                elapsed += value
                y.append(elapsed / 3600)
            ax.plot([(i + 1) * step / 1e6 for i in range(len(y))], y, label=label)
        ax.set(
            xlabel="Global transitions (millions)",
            ylabel="Recorded loop hours",
            title="Collection + GAE + PPO | excludes startup, some logging and checkpoint time",
        )
        ax.legend()
        try:
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"loop_time.{suffix}", dpi=150)
        finally:
            plt.close(fig)


def report(
    single_root: str | Path,
    joint_run: str | Path,
    output_dir: str | Path,
    *,
    single_iterations: int = 20000,
    joint_iterations: int = 5000,
    num_envs: int = 1024,
) -> dict:
    """Re-audit fixed finals before producing the five-policy training comparison."""
    _require(single_iterations == 4 * joint_iterations, "unequal per-policy sample budgets")
    root, joint_path, output = (Path(p).resolve() for p in (single_root, joint_run, output_dir))
    _require(not output.exists(), "comparison output must be new")
    _require(
        all(output != p and output not in p.parents for p in (root, joint_path)),
        "output cannot contain training inputs",
    )
    audits, singles, event_hashes = {}, {}, {}
    for source in SOURCE_ORDER:
        run = root / source
        audit = audit_single_run(run, expected_iterations=single_iterations, num_envs=num_envs)
        _require(audit["source"] == source, "single-source directory/backend mismatch")
        singles[source] = _native_series(run, single_iterations)
        audits[source] = audit
        files = sorted(run.glob("events.out.tfevents.*"))
        _require(bool(files), "native TensorBoard events missing")
        event_hashes[source] = {p.name: _sha256(p) for p in files}
    joint_audit, rows = _read_audited(joint_path, joint_iterations, num_envs, None)
    cfg = json.loads((joint_path / "run_config.json").read_text())["config"]["algo"]
    _require(
        cfg.get("resume") is False
        and cfg.get("resume_path") is None
        and cfg.get("load_run") == "-1"
        and joint_audit["generations"] == [0]
        and len(joint_audit["history"]) == 1
        and joint_audit["history"][0]["resume_checkpoint"] is None,
        "comparison requires fresh joint training",
    )
    timing = joint_audit.get("timing")
    _require(
        timing is not None
        and timing["missing_iterations"] == 0
        and timing["timed_iterations"] == joint_iterations,
        "complete source timing required",
    )
    _require(
        all(a["total_transitions"] == joint_audit["total_transitions"] for a in audits.values()),
        "audited sample budgets differ",
    )
    joint = {
        source: {
            **{key: [r["sources"][source][key] for r in rows] for key in METRICS},
            "loop_seconds": [r["collect_seconds"] + r["learn_seconds"] for r in rows],
        }
        for source in SOURCE_ORDER
    }
    batch, windows = num_envs * 24, (200, 50)
    result = {
        "scope": "completed_training_statistics_comparison",
        "single_audits": audits,
        "joint_audit": joint_audit,
        "native_event_sha256": event_hashes,
        "joint_metrics_sha256": _sha256(joint_path / "synchronous_metrics.jsonl"),
        "smoothing_transitions": batch * windows[0],
        "single_updates_per_smoothing_window": windows[0],
        "joint_updates_per_smoothing_window": windows[1],
        "recent_training_metrics": {
            source: {
                policy: {key: _recent(series[key], window) for key in METRICS}
                for policy, series, window in (
                    ("single", singles[source], windows[0]),
                    ("joint", joint[source], windows[1]),
                )
            }
            for source in SOURCE_ORDER
        },
        "loop_seconds": {
            **{
                name: sum(v for v in series["loop_seconds"] if v is not None)
                for name, series in singles.items()
            },
            "joint": joint_audit["recorded_training_seconds"],
        },
        "interpretation": [
            "Each policy receives the same total samples; the shared policy sees one quarter from each source.",
            "The joint batch is four times larger and executes one quarter as many optimizer steps.",
            "Native mean_reward and joint source episode_return are rolling completed-episode returns.",
            "Joint per-transition reward and completion-weighted global episode return are not used as source comparisons.",
            "Compare total loop times: native GAE is inside learning time, joint GAE is inside collection time.",
            "Source service durations overlap; do not add them to estimate elapsed time.",
            "Configuration/assets and holdout-video integrity require the separate UniLab gate; this report does not revalidate them.",
            "Training curves and 500-step episodes do not establish flip, landing or MuJoCo success.",
        ],
    }
    output.mkdir(parents=True)
    with (output / "series.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["policy", "source", "iteration", "global_transitions", "source_transitions", *METRICS]
        )
        for policy, series, step in (("single", singles, batch), ("joint", joint, batch * 4)):
            for source in SOURCE_ORDER:
                data = series[source]
                for i in range(len(data["loop_seconds"])):
                    writer.writerow(
                        [
                            policy,
                            source,
                            i,
                            (i + 1) * step,
                            (i + 1) * batch,
                            *(data[key][i] for key in METRICS),
                        ]
                    )
    with (output / "loop_time.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["policy", "iteration", "global_transitions", "loop_seconds", "cumulative_loop_seconds"]
        )
        for name, data, step in (
            *((source, singles[source], batch) for source in SOURCE_ORDER),
            ("joint", joint[SOURCE_ORDER[0]], batch * 4),
        ):
            elapsed = 0.0
            for i, value in enumerate(data["loop_seconds"]):
                elapsed += value
                writer.writerow([name, i, (i + 1) * step, value, elapsed])
    _plots(output, singles, joint, batch, windows)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
