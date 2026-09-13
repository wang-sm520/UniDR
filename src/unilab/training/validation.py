"""Reproducible, fail-closed acceptance gates for the fixed G1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from unilab.training.multi_source import SOURCE_ORDER, build_multi_source_plan, is_multi_source

_ROOT = Path(__file__).resolve().parents[3]
_GATES = {"physics": "G1", "single": "G4", "capacity": "G5", "soak": "G6", "evaluate": "G7"}
_REQUIRES = {
    "physics": (),
    "single": ("G1",),
    "capacity": ("G1", "G4"),
    "soak": ("G1", "G4", "G5"),
    "evaluate": ("G6",),
}


def _command(arguments: list[str], *, cwd: Path = _ROOT, timeout: float = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            arguments, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "command": arguments,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": arguments, "returncode": -1, "stdout": "", "stderr": str(error)}


def source_revision_evidence() -> dict[str, Any]:
    """Fingerprint local development sources without rewriting any repository."""
    evidence = {}
    for repository in (_ROOT, _ROOT.parent / "unilab_rl", _ROOT.parent / "unisim"):
        revision = _command(["git", "rev-parse", "HEAD"], cwd=repository)
        patch = _command(["git", "diff", "HEAD", "--", "src", "tests"], cwd=repository)
        untracked = _command(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "tests"],
            cwd=repository,
        )
        if any(item["returncode"] for item in (revision, patch, untracked)):
            raise RuntimeError(f"A development repository is unavailable: {repository}")
        digest = hashlib.sha256((revision["stdout"] + patch["stdout"]).encode())
        for name in sorted(untracked["stdout"].splitlines()):
            digest.update(name.encode())
            digest.update((repository / name).read_bytes())
        evidence[repository.name] = {
            "path": str(repository),
            "base": revision["stdout"],
            "source_sha256": digest.hexdigest(),
        }
    return evidence


def gpu_resource_snapshot() -> dict[str, Any]:
    """Read resource evidence; never change device state or stop other jobs."""
    return {
        "process_memory": process_memory_snapshot(),
        "gpu": _command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
        "processes": _command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def process_memory_snapshot() -> dict[str, Any]:
    """Read RSS of this learner/driver and its descendants, never other jobs."""
    import psutil

    parent = psutil.Process()
    records: list[dict[str, Any]] = []
    for process in (parent, *parent.children(recursive=True)):
        try:
            records.append(
                {
                    "pid": process.pid,
                    "ppid": process.ppid(),
                    "name": process.name(),
                    "rss_bytes": process.memory_info().rss,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "processes": records,
        "rss_sum_bytes": sum(record["rss_bytes"] for record in records),
        "accounting": "per-process RSS; shared pages may be counted more than once",
    }


def collect_preflight() -> dict[str, Any]:
    """Check interpreter discovery, editable sources, and an idle usable GPU."""
    from unisim.backend.isaacgym.dependencies import resolve_isaacgym_runtime
    from unisim.backend.isaacsim.dependencies import resolve_isaacsim_runtime

    packages: dict[str, str | None] = {}
    for package in (
        "unilab",
        "unilab-rl",
        "unisim-core",
        "torch",
        "rsl-rl-lib",
        "motrixsim-core",
        "genesis-world",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    runtimes = {}
    for backend, resolver, expected_python in (
        ("isaacgym", resolve_isaacgym_runtime, (3, 8)),
        ("isaacsim", resolve_isaacsim_runtime, (3, 11)),
    ):
        try:
            executable = resolver().python
        except Exception as error:
            runtimes[backend] = {
                "command": [],
                "returncode": -1,
                "stdout": "",
                "stderr": str(error),
            }
            continue
        probe = (
            "import importlib.util,json,sys; print(json.dumps({'python':sys.version,'available':importlib.util.find_spec("
            + repr(backend)
            + ") is not None})); assert importlib.util.find_spec("
            + repr(backend)
            + ") is not None"
            + f"; assert sys.version_info[:2] == {expected_python!r}"
        )
        runtimes[backend] = _command(
            ["uv", "run", "--no-project", "--python", str(executable), "python", "-c", probe]
        )
    resources = gpu_resource_snapshot()
    reasons = []
    if str(packages.get("torch", "")).endswith("+cpu"):
        reasons.append(
            "CPU-only verification environment; GPU acceptance requires the locked CUDA torch build"
        )
    if resources["gpu"]["returncode"] or resources["processes"]["returncode"]:
        reasons.append(
            "NVIDIA driver/NVML is unavailable; no GPU physics, capacity or stability gate may run"
        )
    elif resources["processes"]["stdout"]:
        reasons.append(
            "Existing GPU compute jobs must release resources; no processes were stopped"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        reasons.append(
            "The fixed experiment requires host GPU 0, without an alternate CUDA device remapping"
        )
    for backend, result in runtimes.items():
        if result["returncode"]:
            reasons.append(f"{backend} external Python runtime discovery failed")
    for package, version in packages.items():
        if version is None:
            reasons.append(f"Required package is absent: {package}")
    modules = {}
    for package, repository in (("uni_rl", "unilab_rl"), ("unisim", "unisim")):
        spec = importlib.util.find_spec(package)
        origin = None if spec is None else spec.origin
        modules[package] = origin
        expected = _ROOT.parent / repository / "src" / package
        if origin is None or not Path(origin).resolve().is_relative_to(expected):
            reasons.append(f"{package} must resolve to its editable development repository")
    kernel_version = Path("/proc/driver/nvidia/version")
    return {
        "status": "blocked" if reasons else "passed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasons": reasons,
        "python": sys.version,
        "packages": packages,
        "module_origins": modules,
        "repositories": source_revision_evidence(),
        "runtimes": runtimes,
        "resources": resources,
        "loaded_nvidia_kernel": kernel_version.read_text() if kernel_version.exists() else None,
        "environment": {
            "UV_NO_SYNC": os.environ.get("UV_NO_SYNC"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def _write_report(target: Path, report: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _require_gates(output_root: Path, stage: str) -> None:
    revisions = source_revision_evidence()
    for gate in _REQUIRES[stage]:
        target = output_root / f"{gate}.json"
        if not target.is_file():
            raise ValueError(f"{stage} requires passing {gate} evidence: {target}")
        report = json.loads(target.read_text())
        if report.get("status") != "passed" or report.get("repositories") != revisions:
            raise ValueError(
                f"{gate} has not passed for the current three-repository source revision"
            )


def validate_acceptance_request(cfg: DictConfig) -> None:
    """Reject invalid acceptance runs before simulator or CUDA construction."""
    gate = OmegaConf.select(cfg, "training.validation.gate")
    if gate is None:
        return
    if gate not in ("capacity", "soak") or not is_multi_source(cfg):
        raise ValueError("PPO acceptance requires the multisim owner and gate capacity or soak")
    output_dir = OmegaConf.select(cfg, "training.validation.output_dir")
    if not output_dir:
        raise ValueError("PPO acceptance requires training.validation.output_dir")
    _require_gates(Path(str(output_dir)).resolve().parent, str(gate))


def run_ppo_acceptance(cfg: DictConfig, runner: Any, env: Any) -> dict[str, Any]:
    """Adapt the experiment budget to the algorithm-owned bounded runner."""
    from uni_rl.algos.rsl_rl_validation import run_bounded_ppo

    gate = str(cfg.training.validation.gate)
    warmup = int(cfg.training.validation.warmup_updates)
    if warmup < 1:
        raise ValueError("Acceptance timing requires at least one complete warmup update")
    result: dict[str, Any] = run_bounded_ppo(
        runner,
        output_dir=Path(str(cfg.training.validation.output_dir)).resolve(),
        max_updates=1 if gate == "capacity" else None,
        duration_seconds=3600.0 if gate == "soak" else None,
        warmup_updates=warmup,
        source_statistics=lambda: env.source_statistics,
        sample_resources=gpu_resource_snapshot,
    )
    return result


def _multisim_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(_ROOT / "src/unilab/conf/ppo")):
        return compose(config_name="config", overrides=["task=g1_walk_flat/multisim"])


def _single_source(backend: str, output: Path) -> None:
    import resource

    import numpy as np

    from unilab.training.run import apply_env_nan_guard

    cfg = _multisim_config()
    plan = build_multi_source_plan(cfg, root_dir=_ROOT)
    source = next(source for source in plan.sources if source.name == backend)
    started = time.monotonic()
    env = source.factory(source.num_envs, source.env_cfg_override)
    construction_seconds = time.monotonic() - started
    try:
        if (
            env.num_envs != 2000
            or env.obs_groups_spec != {"obs": 98, "critic": 101}
            or env.action_space.shape != (29,)
        ):
            raise ValueError(
                "Single-source full-scale environment violates fixed G1 dimensions/quota"
            )
        apply_env_nan_guard(env, cfg.training)
        env.init_state()
        started = time.monotonic()
        env.reset(np.asarray([0, 1999], dtype=np.int64))
        reset_seconds = time.monotonic() - started
        steps = math.ceil(float(cfg.env.max_episode_seconds) / float(cfg.env.ctrl_dt))
        latencies = []
        actions = np.zeros((2000, 29), dtype=np.float32)
        for _ in range(steps):
            started = time.monotonic()
            state = env.step(actions)
            latencies.append(time.monotonic() - started)
            if not all(np.isfinite(value).all() for value in (*state.obs.values(), state.reward)):
                raise FloatingPointError(f"Non-finite core output in full-scale {backend}")
        report = {
            "status": "passed",
            "backend": backend,
            "num_envs": 2000,
            "steps": steps,
            "construction_seconds": construction_seconds,
            "selected_reset_seconds": reset_seconds,
            "step_seconds_mean": float(np.mean(latencies)),
            "step_seconds_max": max(latencies),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "resources": gpu_resource_snapshot(),
        }
    finally:
        env.close()
    _write_report(output, report)


def _run_logged(
    command: list[str],
    logfile: Path,
    *,
    extra_env: dict[str, str] | None = None,
    cwd: Path = _ROOT,
) -> None:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command}) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "UV_NO_SYNC": "1",
                "UV_PROJECT_ENVIRONMENT": sys.prefix,
                **(extra_env or {}),
            },
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Acceptance subprocess exited {result.returncode}; see {logfile}")


def _required_test_count(junit: Path, backend: str) -> int:
    root = ET.parse(junit).getroot()
    cases = list(root.iter("testcase"))
    if not cases or any(
        case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")
    ):
        raise RuntimeError(f"Required {backend} physical tests did not actually pass")
    return len(cases)


def _execute_gate(stage: str, output_root: Path) -> dict[str, Any]:
    from unilab.cli import build_command

    results: dict[str, Any] = {}
    if stage in ("physics", "single"):
        for backend in SOURCE_ORDER:
            directory = output_root / stage
            if stage == "physics":
                junit = directory / f"{backend}.xml"
                _run_logged(
                    [
                        "uv",
                        "run",
                        "--no-sync",
                        "pytest",
                        "tests/backends/test_g1_multisim_physics.py",
                        "-q",
                        "-k",
                        backend,
                        "-m",
                        "",
                        f"--junitxml={junit}",
                    ],
                    directory / f"{backend}.log",
                    extra_env={"UNILAB_RUN_MULTISIM_PHYSICS": "1"},
                )
                effect_tests = _required_test_count(junit, backend)
                native_junit = directory / f"{backend}-native.xml"
                native_test = (
                    "tests/test_isaac_native_readback.py"
                    if backend in ("isaacgym", "isaacsim")
                    else "tests/test_multisim_native_readback.py"
                )
                _run_logged(
                    [
                        "uv",
                        "run",
                        "--no-sync",
                        "pytest",
                        native_test,
                        "-q",
                        "-k",
                        backend,
                        "-m",
                        "",
                        f"--junitxml={native_junit}",
                    ],
                    directory / f"{backend}-native.log",
                    extra_env={"UNILAB_RUN_MULTISIM_PHYSICS": "1", "UNISIM_ISAAC_READBACK": "1"},
                    cwd=_ROOT.parent / "unisim",
                )
                results[backend] = {
                    "effect_tests": effect_tests,
                    "junit": str(junit),
                    "native_readback_tests": _required_test_count(native_junit, backend),
                    "native_junit": str(native_junit),
                }
            else:
                target = directory / f"{backend}.json"
                _run_logged(
                    [
                        "uv",
                        "run",
                        "--no-sync",
                        "-m",
                        "unilab.training.validation",
                        "--single-backend",
                        backend,
                        "--output-root",
                        str(target),
                    ],
                    directory / f"{backend}.log",
                )
                results[backend] = json.loads(target.read_text())
    elif stage in ("capacity", "soak"):
        directory = output_root / stage
        command = build_command(
            mode="train",
            algo="ppo",
            task="g1_walk_flat",
            sim="multisim",
            overrides=[
                f"training.validation.gate={stage}",
                f"training.validation.output_dir={directory}",
                f"training.log_dir={directory}",
                "training.logger=tensorboard",
            ],
        )
        _run_logged(command, output_root / f"{stage}.log")
        results = json.loads((directory / "validation_report.json").read_text())
        if stage == "soak" and results["measured_elapsed_seconds"] < 3600:
            raise RuntimeError("The stability test did not complete 60 measured minutes")
    elif stage == "evaluate":
        from unilab.training.evaluation import aggregate_ppo_metrics_reports

        checkpoint = output_root / "soak/validation_final.pt"
        reports = []
        for backend in SOURCE_ORDER:
            directory = output_root / "evaluation" / backend
            command = build_command(
                mode="eval",
                algo="ppo",
                task="g1_walk_flat",
                sim=backend,
                profile="multisim",
                metrics=True,
                load_run=str(checkpoint),
                overrides=[f"training.evaluation.output_dir={directory}"],
            )
            _run_logged(command, output_root / "evaluation" / f"{backend}.log")
            reports.append(directory / "metrics.json")
            results[backend] = {"directory": str(directory), "checkpoint": str(checkpoint)}
        results["overall_report"] = str(
            aggregate_ppo_metrics_reports(
                reports, output_path=output_root / "evaluation/overall.json"
            )
        )
    return {
        "status": "passed",
        "gate": _GATES[stage],
        "repositories": source_revision_evidence(),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("preflight", *_GATES), default="preflight")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--single-backend", choices=SOURCE_ORDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    if args.single_backend:
        _single_source(args.single_backend, output_root)
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = collect_preflight()
    _write_report(output_root / f"preflight-{args.gate}.json", preflight)
    if preflight["status"] != "passed":
        print(
            json.dumps(
                {"status": "blocked", "gate": args.gate, "reasons": preflight["reasons"]}, indent=2
            )
        )
        return 2
    if args.gate == "preflight":
        return 0
    target = output_root / f"{_GATES[args.gate]}.json"
    if target.exists():
        raise FileExistsError(f"Evidence already exists: {target}; use a fresh output root")
    try:
        _require_gates(output_root, args.gate)
        report = _execute_gate(args.gate, output_root)
    except Exception as error:
        _write_report(
            target,
            {
                "status": "failed",
                "gate": _GATES[args.gate],
                "error": str(error),
                "repositories": source_revision_evidence(),
            },
        )
        raise
    _write_report(target, report)
    print(f"{_GATES[args.gate]} passed: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
