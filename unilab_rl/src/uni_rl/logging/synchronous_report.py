"""Offline evidence audit and curves for complete synchronous PPO runs."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import torch

SOURCE_ORDER = ("isaacsim", "isaacgym", "genesis", "motrix")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, expected: int, name: str) -> None:
    _require(type(value) is int and value == expected, f"{name}: expected integer {expected}")


def _number(value: Any, name: str, *, nullable: bool = False) -> float:
    if value is None and nullable:
        return float("nan")
    _require(type(value) in (int, float) and math.isfinite(value), f"non-finite {name}")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counters(iteration: int, samples: int) -> dict[str, int]:
    count = iteration + 1
    return dict(
        next_iteration=count,
        policy_version=count,
        normalizer_version=count * 24,
        optimizer_steps=count * 20,
        total_transitions=count * samples,
    )


def _model_state(state: dict, samples: int) -> str:
    count = state["obs_normalizer.count"]
    _require(
        isinstance(count, torch.Tensor)
        and count.dtype == torch.int64
        and count.numel() == 1
        and int(count) == samples,
        "invalid normalizer sample count",
    )
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        _require(
            isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all()),
            "non-finite model state",
        )
        if name.startswith("obs_normalizer."):
            digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.contiguous().numpy().tobytes())
    for name in ("_mean", "_var", "_std"):
        _require(
            state[f"obs_normalizer.{name}"].dtype == torch.float32, "float32 normalization required"
        )
    _require(
        bool((state["obs_normalizer._var"] >= 0).all())
        and bool((state["obs_normalizer._std"] >= 0).all()),
        "negative normalizer variance/std",
    )
    return digest.hexdigest()


def _checkpoint(path: Path, iteration: int, samples: int, contract: dict) -> tuple[dict, dict]:
    saved = torch.load(path, weights_only=False, map_location="cpu")
    _integer(saved["iter"], iteration, "checkpoint iteration")
    meta = saved["synchronous"]
    _integer(meta["format"], 1, "checkpoint format")
    _require(
        meta["complete"] is True and meta["contract"] == contract,
        "incomplete or mismatched checkpoint contract",
    )
    for name, value in _counters(iteration, samples).items():
        _integer(meta[name], value, f"checkpoint {name}")
    _require(
        type(meta["generation"]) is int and meta["generation"] >= 0, "invalid checkpoint generation"
    )
    lr = _number(meta["learning_rate"], "checkpoint learning rate")
    _require(lr > 0, "positive learning rate required")
    optimizer = saved["optimizer_state_dict"]
    ids = [index for group in optimizer["param_groups"] for index in group["params"]]
    _require(
        bool(ids) and len(ids) == len(set(ids)) and set(ids) == set(optimizer["state"]),
        "incomplete optimizer parameter state",
    )
    for group in optimizer["param_groups"]:
        _require(group["lr"] == lr, "optimizer learning rate mismatch")
    params = [
        value
        for name in ("actor", "critic")
        for key, value in saved[f"{name}_state_dict"].items()
        if not key.startswith("obs_normalizer.")
    ]
    _require(len(params) == len(ids), "optimizer/model parameter count mismatch")
    for index, parameter in zip(ids, params):
        state = optimizer["state"][index]
        step = state["step"]
        _require(
            isinstance(step, torch.Tensor)
            and step.numel() == 1
            and float(step) == (iteration + 1) * 20,
            "actual Adam step mismatch",
        )
        for name in ("exp_avg", "exp_avg_sq"):
            _require(
                isinstance(state[name], torch.Tensor)
                and state[name].shape == parameter.shape
                and bool(torch.isfinite(state[name]).all()),
                "invalid optimizer moment",
            )
    hashes = {
        name: _model_state(saved[f"{name}_state_dict"], (iteration + 1) * samples)
        for name in ("actor", "critic")
    }
    return saved, hashes


def _history(
    run_dir: Path, context: dict, cutoff: int | None = None, seen=None
) -> tuple[list[dict], list[dict]]:
    seen = set() if seen is None else seen
    _require(run_dir not in seen, "cyclic resume ancestry")
    seen.add(run_dir)
    manifest = json.loads((run_dir / "sources_manifest.json").read_text())
    run = json.loads((run_dir / "run_config.json").read_text())
    contract, algo = context["contract"], run["config"]["algo"]
    _require(
        run["run"]["mode"] == "synchronous_four_source"
        and {key: manifest[key] for key in context["behavior"]} == context["behavior"]
        and manifest["digest"] == run["run"]["manifest_digest"] == contract["manifest_digest"],
        "resume ancestry manifest mismatch",
    )
    _require(
        run["config"]["unidr"]["sources"] == context["sources"], "resume ancestry sources mismatch"
    )
    for key in ("num_envs", "num_steps_per_env", "save_interval"):
        _integer(algo[key], contract["train_cfg"][key], f"ancestry {key}")
    for key in ("num_learning_epochs", "num_mini_batches"):
        _integer(algo["algorithm"][key], contract["train_cfg"]["algorithm"][key], f"ancestry {key}")
    _require(
        algo["empirical_normalization"] is True and algo["algorithm"]["schedule"] == "adaptive",
        "ancestry PPO mismatch",
    )
    prefix: list[dict] = []
    chain: list[dict] = []
    generation = 0
    origin = None
    if algo.get("resume", False):
        origin = Path(algo["resume_path"])
        _require(origin.is_absolute(), "resume ancestry requires an absolute checkpoint path")
        origin = origin.resolve(strict=True)
        boundary = torch.load(origin, weights_only=False, map_location="cpu")["iter"]
        _require(
            type(boundary) is int and boundary >= 0 and (cutoff is None or boundary < cutoff),
            "invalid resume ancestry boundary",
        )
        parent, _ = _checkpoint(origin, boundary, context["samples"], contract)
        prefix, chain = _history(origin.parent, context, boundary, seen)
        _require(
            prefix[-1]["generation"] == parent["synchronous"]["generation"]
            and prefix[-1]["learning_rate"] == parent["synchronous"]["learning_rate"],
            "ancestry checkpoint metrics mismatch",
        )
        generation = parent["synchronous"]["generation"] + 1
    else:
        _require(not algo.get("resume_path"), "resume path provided with resume disabled")
    rows = []
    with (run_dir / "synchronous_metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            rows.append(row)
            if cutoff is not None and row["iteration"] == cutoff:
                break  # Abandoned updates after the selected checkpoint are not this run's history.
        excluded_tail_bytes = len(stream.read().encode("utf-8"))
    _require(
        bool(rows) and (cutoff is None or rows[-1]["iteration"] == cutoff),
        "missing ancestry iteration rows",
    )
    _require(
        type(algo["max_iterations"]) is int and algo["max_iterations"] > rows[-1]["iteration"],
        "ancestry intent does not cover its history",
    )
    _require(
        all(type(row["generation"]) is int and row["generation"] == generation for row in rows),
        "resume generation mismatch",
    )
    chain.append(
        {
            "run_dir": str(run_dir),
            "through_iteration": rows[-1]["iteration"],
            "resume_checkpoint": str(origin) if origin else None,
            "resume_checkpoint_sha256": _sha256(origin) if origin else None,
            "excluded_metric_tail_bytes": excluded_tail_bytes,
        }
    )
    return prefix + rows, chain


def _audit(
    run_dir: Path, expected_iterations: int, num_envs: int, final_checkpoint: str | Path | None
) -> tuple[dict, list[dict]]:
    _require(
        type(expected_iterations) is int and expected_iterations >= 2,
        "at least two expected iterations required",
    )
    _require(type(num_envs) is int and num_envs > 0, "positive per-source num_envs required")
    samples = num_envs * 4 * 24
    manifest = json.loads((run_dir / "sources_manifest.json").read_text())
    run = json.loads((run_dir / "run_config.json").read_text())
    behavior = {key: manifest[key] for key in ("sources", "algorithm", "assets")}
    digest = hashlib.sha256(json.dumps(behavior, sort_keys=True, default=str).encode()).hexdigest()
    _require(
        digest == manifest["digest"] == run["run"]["manifest_digest"], "manifest digest mismatch"
    )
    _require(run["run"]["mode"] == "synchronous_four_source", "unexpected run mode")
    _require(bool(manifest["assets"]), "missing asset fingerprints")
    for value in manifest["assets"].values():
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            "invalid asset fingerprint",
        )
    cfg = run["config"]
    configured_iterations = cfg["algo"]["max_iterations"]
    final_path = run_dir / f"model_{expected_iterations - 1}.pt"
    if final_checkpoint is None:
        _integer(configured_iterations, expected_iterations, "run intent max_iterations")
    else:
        _require(
            Path(final_checkpoint).resolve(strict=True) == final_path
            and type(configured_iterations) is int
            and configured_iterations >= expected_iterations,
            "selected checkpoint must match the explicit budget within the original run intent",
        )
    for algorithm in (cfg["algo"], manifest["algorithm"]):
        for name, value in (
            ("num_envs", num_envs),
            ("num_steps_per_env", 24),
            ("save_interval", 500),
        ):
            _integer(algorithm[name], value, f"run intent {name}")
        for name, value in (("num_learning_epochs", 5), ("num_mini_batches", 4)):
            _integer(algorithm["algorithm"][name], value, f"run intent {name}")
        _require(
            algorithm["empirical_normalization"] is True
            and algorithm["algorithm"]["schedule"] == "adaptive",
            "native adaptive normalized PPO required",
        )
    _require(
        tuple(source["source"] for source in manifest["sources"]) == SOURCE_ORDER,
        "manifest source order mismatch",
    )
    _require(
        tuple(source["name"] for source in cfg["unidr"]["sources"]) == SOURCE_ORDER,
        "run intent source order mismatch",
    )
    for actual, intended in zip(manifest["sources"], cfg["unidr"]["sources"]):
        _require(actual["device"] == intended["device"], "run intent source device mismatch")
    contract = {
        "manifest_digest": digest,
        "train_cfg": manifest["algorithm"],
        "num_envs": 4 * num_envs,
        "sources": [
            (source, i * num_envs, (i + 1) * num_envs) for i, source in enumerate(SOURCE_ORDER)
        ],
    }
    rows, history = _history(
        run_dir,
        {
            "behavior": behavior,
            "contract": contract,
            "sources": cfg["unidr"]["sources"],
            "samples": samples,
        },
        cutoff=expected_iterations - 1 if final_checkpoint is not None else None,
    )
    _require(len(rows) == expected_iterations, "missing or extra iteration rows")
    loss_names = set(rows[0]["losses"])
    _require(bool(loss_names), "missing PPO losses")
    for iteration, row in enumerate(rows):
        _integer(row["iteration"], iteration, "iteration sequence")
        _require(
            row["window_stamp"] == [iteration, iteration, iteration * 24]
            and all(type(value) is int for value in row["window_stamp"]),
            "window stamp mismatch",
        )
        for name, value in _counters(iteration, samples).items():
            if name != "next_iteration":
                _integer(row[name], value, name)
        _require(
            row["epoch_samples"] == [samples] * 5
            and all(type(value) is int for value in row["epoch_samples"]),
            "epoch sample budget mismatch",
        )
        _require(
            type(row["generation"]) is int
            and row["generation"] >= (rows[iteration - 1]["generation"] if iteration else 0),
            "invalid recovery generation sequence",
        )
        _require(tuple(row["sources"]) == SOURCE_ORDER, "metric source order mismatch")
        for name in SOURCE_ORDER:
            source = row["sources"][name]
            _integer(source["transitions"], num_envs * 24, "source transitions")
            for flag in ("terminated", "truncated"):
                _require(
                    type(source[flag]) is int and 0 <= source[flag] <= num_envs * 24,
                    "invalid termination count",
                )
            for key in ("reward", "episode_return", "episode_length"):
                _number(source[key], f"{name}/{key}", nullable=key != "reward")
        for key in ("reward", "episode_return", "episode_length"):
            _number(row[key], key, nullable=key != "reward")
        _require(
            math.isclose(
                row["reward"],
                sum(source["reward"] for source in row["sources"].values()) / 4,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ),
            "global/source reward mismatch",
        )
        _require(
            _number(row["learning_rate"], "learning rate") > 0, "positive learning rate required"
        )
        _require(set(row["losses"]) == loss_names, "PPO loss schema mismatch")
        for value in row["losses"].values():
            _number(value, "PPO loss")
        times = [_number(row[key], key) for key in ("collect_seconds", "learn_seconds")]
        _require(min(times) >= 0 and sum(times) > 0, "invalid iteration duration")
    final, normalizer_hashes = _checkpoint(final_path, expected_iterations - 1, samples, contract)
    initial, initial_hashes = _checkpoint(
        Path(history[0]["run_dir"]) / "model_0.pt", 0, samples, contract
    )
    _require(
        final["synchronous"]["generation"] == rows[-1]["generation"]
        and initial["synchronous"]["generation"] == rows[0]["generation"],
        "checkpoint/metric generation mismatch",
    )
    _require(
        final["synchronous"]["learning_rate"] == rows[-1]["learning_rate"]
        and initial["synchronous"]["learning_rate"] == rows[0]["learning_rate"],
        "checkpoint/metric learning rate mismatch",
    )
    changed = {}
    for name in ("actor", "critic"):
        before, after = initial[f"{name}_state_dict"], final[f"{name}_state_dict"]
        _require(
            before.keys() == after.keys()
            and all(
                before[key].shape == after[key].shape and before[key].dtype == after[key].dtype
                for key in before
            ),
            "model checkpoint schema mismatch",
        )
        changed[name] = any(
            not torch.equal(before[key], after[key])
            for key in before
            if not key.startswith("obs_normalizer.")
        )
        _require(changed[name], f"{name} parameters did not change after iteration 0")
    seconds = sum(row["collect_seconds"] + row["learn_seconds"] for row in rows)
    result = {
        "audit_passed": True,
        "scope": "formal_training_budget"
        if (expected_iterations, num_envs) == (10000, 1024)
        else "bounded_validation_budget",
        "audit_mode": "full_run" if final_checkpoint is None else "selected_checkpoint_prefix",
        "configured_max_iterations": configured_iterations,
        "run_dir": str(run_dir),
        "history": history,
        "iterations": expected_iterations,
        "num_envs_per_source": num_envs,
        "transitions_per_iteration": samples,
        "total_transitions": samples * expected_iterations,
        "optimizer_steps": 20 * expected_iterations,
        "source_order": list(SOURCE_ORDER),
        "generations": sorted({row["generation"] for row in rows}),
        "source_total_transitions": {
            name: num_envs * 24 * expected_iterations for name in SOURCE_ORDER
        },
        "manifest_digest": digest,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": _sha256(final_path),
        "parameters_changed_since_iteration_0": changed,
        "normalizer_sha256": normalizer_hashes,
        "iteration_0_normalizer_sha256": initial_hashes,
        "recorded_training_seconds": seconds,
        "mean_transitions_per_second": samples * expected_iterations / seconds,
        "final_metrics": rows[-1],
        "evidence_limit": "Budget and saved learner state only; curves do not establish task success or holdout performance.",
    }
    if final_checkpoint is not None:
        result["scope"] = "explicit_checkpoint_prefix"
        result["evidence_limit"] += (
            " Only committed history through the explicit checkpoint is audited; later metrics"
            " and any in-flight collection or optimization are excluded. Original run intent is unchanged."
        )
    return result, rows


def _read_audited(
    run_dir: str | Path,
    expected_iterations: int,
    num_envs: int,
    final_checkpoint: str | Path | None,
) -> tuple[dict, list[dict]]:
    try:
        return _audit(Path(run_dir).resolve(), expected_iterations, num_envs, final_checkpoint)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"invalid synchronous report input schema: {exc}") from exc


def audit_run(
    run_dir: str | Path,
    *,
    expected_iterations: int = 10000,
    num_envs: int = 1024,
    final_checkpoint: str | Path | None = None,
) -> dict:
    """Audit a full run, or an explicitly selected committed prefix without changing intent."""
    return _read_audited(run_dir, expected_iterations, num_envs, final_checkpoint)[0]


def _plot(rows: list[dict], output_dir: Path, samples: int) -> None:
    plt = importlib.import_module("matplotlib.pyplot")

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
        iterations = [row["iteration"] for row in rows]
        for axis, key, label in zip(
            axes.flat,
            ("reward", "episode_return", "episode_length"),
            ("Mean step reward", "Rolling episode return", "Rolling episode length (steps)"),
        ):
            axis.plot(
                iterations,
                [row[key] for row in rows],
                label="All sources",
                color="black",
                linewidth=1.8,
            )
            for source in SOURCE_ORDER:
                axis.plot(
                    iterations,
                    [row["sources"][source][key] for row in rows],
                    label=source,
                    linewidth=1,
                )
            axis.set_ylabel(label)
            axis.legend(fontsize=8)
        axes.flat[3].plot(iterations, [row["learning_rate"] for row in rows])
        axes.flat[3].set_ylabel("Learning rate")
        axes.flat[3].set_yscale("log")
        for name in rows[0]["losses"]:
            axes.flat[4].plot(iterations, [row["losses"][name] for row in rows], label=name)
        axes.flat[4].set_ylabel("Native PPO losses / entropy")
        axes.flat[4].set_yscale("symlog", linthresh=0.01)
        axes.flat[4].legend()
        axes.flat[5].plot(
            iterations, [samples / (row["collect_seconds"] + row["learn_seconds"]) for row in rows]
        )
        axes.flat[5].set_ylabel("Transitions / second")
        for axis in axes.flat:
            axis.set_xlabel("Completed iteration (zero based)")
        fig.suptitle(
            f"Synchronous PPO · {len(rows):,} iterations · {samples:,} transitions/iteration"
        )
        try:
            fig.savefig(output_dir / "curves.png", dpi=160)
            fig.savefig(output_dir / "curves.pdf")
        finally:
            plt.close(fig)


def report(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_iterations: int = 10000,
    num_envs: int = 1024,
    final_checkpoint: str | Path | None = None,
) -> dict:
    """Audit, then write JSON and standalone curves into a fresh output directory."""
    run_path, output_path = Path(run_dir).resolve(), Path(output_dir).resolve()
    _require(
        output_path != run_path and output_path not in run_path.parents,
        "report output must be separate from the run directory",
    )
    _require(
        not output_path.exists() or (output_path.is_dir() and not any(output_path.iterdir())),
        "report output directory must be empty",
    )
    result, rows = _read_audited(run_path, expected_iterations, num_envs, final_checkpoint)
    output_path.mkdir(parents=True, exist_ok=True)
    _plot(rows, output_path, result["transitions_per_iteration"])
    (output_path / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
