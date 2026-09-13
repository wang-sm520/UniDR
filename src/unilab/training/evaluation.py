"""Headless PPO checkpoint evaluation assembled from the existing policy runtime.

``run_ppo_metrics_evaluation`` returns the written ``metrics.json`` path. Exactly
one first episode per environment and seed contributes to its episode-weighted
summary; ``episodes.csv`` contains the same unsummarized records. Optional video
captures environment zero's first episode of the first seed via deferred
rendering. No training, visual play profile, or backend-private API is used.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, get_policy_obs_dims, normalize_ppo_train_cfg
from uni_rl.algos.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.base.process_device import (
    apply_backend_env_device_override,
    configure_backend_process_device,
)
from unilab.tasks.locomotion.g1.evaluation import (
    G1EvaluationSpec,
    G1TransitionMetrics,
    apply_g1_evaluation_overrides,
    g1_evaluation_spec,
    g1_metric_provenance,
)
from unilab.training.run import (
    algo_config_dict,
    format_play_checkpoint_error,
    get_log_root,
    parse_checkpoint_path,
)
from unilab.utils.checkpoint import get_entrypoint_log_root
from unilab.utils.sim2sim import (
    extract_contract_snapshot,
    policy_load_dim_guard,
    resolve_sim2sim_config,
)
from unilab.visualization.interactive_playback import (
    RslRlPlaybackConfig,
    create_rsl_rl_playback_session,
    infer_checkpoint_actor_input_dim,
)

_EVALUATION_SEEDS = (101, 102, 103, 104)


def _inference_train_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = cast(dict[str, Any], normalize_ppo_train_cfg(train_cfg))
    normalized.setdefault("algorithm", {})["enable_compile"] = False
    return normalized


def _checkpoint_sha256(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inference_policy(runner: Any, checkpoint: str, *, device: str) -> None:
    from uni_rl.algos.rsl_rl_training_state import TrainingStateOnPolicyRunner

    options = (
        {"restore_training_state": False} if isinstance(runner, TrainingStateOnPolicyRunner) else {}
    )
    runner.load(
        checkpoint,
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
        map_location=device,
        strict=True,
        **options,
    )


def _validate_checkpoint_dimensions(
    cfg: DictConfig, checkpoint: Path, spec: G1EvaluationSpec
) -> None:
    """Strictly load the configured CPU models before any simulator construction."""
    from rsl_rl.models import MLPModel
    from rsl_rl.utils import resolve_callable, resolve_obs_groups
    from tensordict import TensorDict

    train_cfg = _inference_train_cfg(algo_config_dict(cfg))
    observations = TensorDict(
        {
            "actor": torch.zeros(1, spec.obs_groups_spec["obs"]),
            "policy": torch.zeros(1, spec.obs_groups_spec["obs"]),
            "critic": torch.zeros(1, spec.obs_groups_spec["critic"]),
        },
        batch_size=[1],
    )
    groups = resolve_obs_groups(observations, train_cfg["obs_groups"], ["actor", "critic"])
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Not an RSL-RL checkpoint: {checkpoint}")
    for name, output_dim in (("actor", spec.action_dim), ("critic", 1)):
        model_cfg = copy.deepcopy(train_cfg[name])
        model_class = resolve_callable(model_cfg.pop("class_name"))
        if model_class is not MLPModel:
            raise ValueError("G1 metrics currently support the owner RSL-RL MLPModel policy only")
        state_dict = loaded.get(f"{name}_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint is missing {name}_state_dict: {checkpoint}")
        model = model_class(observations, groups, name, output_dim, **model_cfg)
        with policy_load_dim_guard(
            env_obs_dim=sum(int(observations[group].shape[-1]) for group in groups[name]),
            env_action_dim=spec.action_dim,
            algo_name=f"ppo metrics {name}",
        ):
            model.load_state_dict(state_dict, strict=True)


def _collect_first_episodes(
    session: Any,
    metrics: G1TransitionMetrics,
    *,
    seed: int,
    max_steps: int,
    ctrl_dt: float,
    recorder: Any | None = None,
) -> list[dict[str, Any]]:
    env = session.env
    env.seed(seed)
    session.reset()
    active = np.ones(env.num_envs, dtype=bool)
    lengths = np.zeros(env.num_envs, dtype=np.int64)
    totals: dict[str, np.ndarray] = {}
    episodes = []
    with torch.inference_mode():
        for _ in range(max_steps):
            if recorder is not None and active[0]:
                recorder.snapshot()
            command_vx = metrics.command_vx(env.state.obs)
            session.step_once()
            state = env.state
            terminated = np.asarray(state.terminated, dtype=bool)
            truncated = np.asarray(state.truncated, dtype=bool)
            if terminated.shape != active.shape or truncated.shape != active.shape:
                raise ValueError("Evaluation requires one termination/timeout flag per environment")
            samples = metrics.measure(state, command_vx, active)
            lengths[active] += 1
            for name, values in samples.items():
                totals.setdefault(name, np.zeros(env.num_envs, dtype=np.float64))[active] += values[
                    active
                ]
            completed = active & (terminated | truncated)
            for env_index in np.flatnonzero(completed):
                length = int(lengths[env_index])
                episodes.append(
                    {
                        "seed": seed,
                        "env_index": int(env_index),
                        "episode_length_steps": length,
                        "episode_time_seconds": length * ctrl_dt,
                        "terminated": bool(terminated[env_index]),
                        "truncated": bool(truncated[env_index]),
                        "full_episode_success": bool(
                            truncated[env_index]
                            and not terminated[env_index]
                            and length == max_steps
                        ),
                        "full_20s_success": bool(
                            truncated[env_index]
                            and not terminated[env_index]
                            and length * ctrl_dt >= 20.0
                        ),
                        **{
                            name: float(values[env_index] / length)
                            for name, values in totals.items()
                        },
                    }
                )
            active[completed] = False
            if not np.any(active):
                return sorted(episodes, key=lambda episode: episode["env_index"])
    raise RuntimeError(
        f"Seed {seed}: {int(np.sum(active))} environments did not finish within {max_steps} steps; "
        "refusing to publish an incomplete evaluation"
    )


def _summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "episode_count": len(episodes),
        "non_timeout_termination_rate": float(
            np.mean([episode["terminated"] for episode in episodes])
        ),
        "full_episode_success_rate": float(
            np.mean([episode["full_episode_success"] for episode in episodes])
        ),
        "full_20s_success_rate": float(
            np.mean([episode["full_20s_success"] for episode in episodes])
        ),
    }
    summary["fall_rate"] = summary["non_timeout_termination_rate"]
    for name in (
        "episode_length_steps",
        "episode_time_seconds",
        "vx_mae",
        "abs_vy_mean",
        "abs_wz_mean",
    ):
        values = np.asarray([episode[name] for episode in episodes], dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
    return summary


def run_ppo_metrics_evaluation(cfg: DictConfig, *, device: str, root_dir: Path) -> Path:
    """Evaluate the selected G1 PPO checkpoint headlessly, reusing one seeded env.

    Strict Sim2Sim validation always sees the original profile, even when
    ``training.sim2sim_strict`` is false. Only afterwards are evaluation-specific
    noise/DR changes applied to a private copy. CPU actor and critic strict loads
    also precede environment construction. Genesis is initialized at most once.
    """
    if OmegaConf.select(cfg, "algo.algo") != "ppo":
        raise ValueError("PPO metrics evaluation requires algo.algo=ppo")
    root_dir = root_dir.resolve()
    seeds = OmegaConf.select(cfg, "training.evaluation.seeds", default=list(_EVALUATION_SEEDS))
    if OmegaConf.is_config(seeds):
        seeds = OmegaConf.to_container(seeds, resolve=True)
    num_envs = OmegaConf.select(cfg, "training.evaluation.num_envs", default=25)
    record_video = OmegaConf.select(cfg, "training.evaluation.record_video", default=False)
    if not isinstance(record_video, bool):
        raise ValueError("training.evaluation.record_video must be a boolean")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("Evaluation seeds must be a non-empty list of non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Evaluation seeds must be unique")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("training.evaluation.num_envs must be a positive integer")
    ctrl_dt = float(cfg.env.ctrl_dt)
    horizon = float(cfg.env.max_episode_seconds)
    if not math.isfinite(ctrl_dt) or ctrl_dt <= 0 or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Evaluation requires positive finite ctrl_dt and max_episode_seconds")
    max_steps = math.ceil(horizon / ctrl_dt)
    checkpoint, run_dir = parse_checkpoint_path(cfg, root_dir=root_dir)
    if checkpoint is None or run_dir is None or not checkpoint.is_file():
        raise FileNotFoundError(
            format_play_checkpoint_error(
                cfg,
                task_log_root=get_log_root(root_dir, cfg) / str(cfg.training.task_name),
                load_path=checkpoint,
                load_path_dir=run_dir,
            )
        )
    checkpoint = checkpoint.resolve()
    resolve_sim2sim_config(run_dir, cfg, algo_name="ppo", strict=True)
    spec = g1_evaluation_spec(cfg)
    checkpoint_digest = _checkpoint_sha256(checkpoint)
    _validate_checkpoint_dimensions(cfg, checkpoint, spec)
    effective = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    overrides = apply_g1_evaluation_overrides(effective)
    for path, value in {
        "training.play_render_mode": "none",
        "training.no_play": True,
        "training.sim2sim_strict": True,
        "algo.algorithm.enable_compile": False,
        "algo.num_envs": num_envs,
        "env.seed": seeds[0],
    }.items():
        OmegaConf.update(effective, path, value, force_add=True)
    backend = str(effective.training.sim_backend)
    timestamp = datetime.now(timezone.utc)
    configured_output = OmegaConf.select(cfg, "training.evaluation.output_dir")
    output_dir = (
        Path(str(configured_output))
        if configured_output
        else (
            checkpoint.parent
            / "evaluation"
            / f"{backend}_{checkpoint.stem}_{timestamp:%Y%m%dT%H%M%S%fZ}"
        )
    )
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    video_path = output_dir / f"{checkpoint.stem}_{backend}_seed{seeds[0]}_env0.mp4"
    if record_video and video_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation video: {video_path}")
    bound_device = configure_backend_process_device(backend, device)
    device = bound_device or device
    env_overrides = BackendAdapter(
        effective, root_dir=root_dir, algo_name="ppo"
    ).build_task_env_cfg_override()
    env_overrides = apply_backend_env_device_override(env_overrides, backend, learner_device=device)
    if backend == "isaacsim":
        env_overrides["isaacsim_render_mode"] = "none"
    rl_cfg = algo_config_dict(effective)
    runtime = resolve_rsl_rl_ppo_runtime(rl_cfg, default_wrapper_cls=RslRlVecEnvWrapper)
    from rsl_rl.runners import OnPolicyRunner

    env = create_env(effective, num_envs=num_envs, env_cfg_override=env_overrides)
    video_metadata = None
    try:
        if (
            env.num_envs != num_envs
            or env.obs_groups_spec != spec.obs_groups_spec
            or env.action_space.shape != (spec.action_dim,)
        ):
            raise ValueError(
                "Constructed environment disagrees with the cold-path G1 policy dimensions"
            )
        metrics = G1TransitionMetrics(env)
        session, _, loaded_checkpoint = create_rsl_rl_playback_session(
            playback_cfg=RslRlPlaybackConfig(
                task=str(effective.training.task_name),
                load_run=str(checkpoint),
                checkpoint=None,
                action_mode="policy",
                policy_obs_mode="flat",
                algo_log_name=str(effective.algo.algo_log_name),
                log_root=None,
                num_envs=num_envs,
            ),
            env_factory=lambda count: env,
            algo_config=rl_cfg,
            root_dir=root_dir,
            device=device,
            checkpoint_resolver=lambda *args: str(checkpoint),
            checkpoint_input_dim_reader=infer_checkpoint_actor_input_dim,
            entrypoint_log_root=get_entrypoint_log_root,
            wrapper_cls=runtime.wrapper_cls,
            runner_cls=runtime.runner_cls or OnPolicyRunner,
            policy_obs_dims_getter=get_policy_obs_dims,
            train_cfg_normalizer=_inference_train_cfg,
            runner_loader=lambda runner, path: _load_inference_policy(runner, path, device=device),
            guard_algo_name="ppo",
        )
        if session.policy is None or loaded_checkpoint != str(checkpoint):
            raise RuntimeError("Metrics evaluation requires the specified checkpoint policy")
        recorder = None
        if record_video:
            from unilab.visualization.playback_session import SnapshotPlaybackSession

            recorder = SnapshotPlaybackSession(
                env, frame_state_getter=lambda: env.get_physics_state_snapshot()[:1]
            )
        episodes = []
        for seed in seeds:
            episodes.extend(
                _collect_first_episodes(
                    session,
                    metrics,
                    seed=seed,
                    max_steps=max_steps,
                    ctrl_dt=ctrl_dt,
                    recorder=recorder if seed == seeds[0] else None,
                )
            )
        if recorder is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fps = max(1, round(1.0 / ctrl_dt))
            camera = {
                "cam_tracking": True,
                "cam_tracking_env_idx": 0,
                "cam_tracking_extra_envs": 0,
                "cam_distance": float(effective.training.cam_distance),
                "cam_elevation": float(effective.training.cam_elevation),
                "cam_azimuth": float(effective.training.cam_azimuth),
            }
            rendered = recorder.render_snapshots(output_video=video_path, camera=camera, fps=fps)
            if rendered is None or not video_path.is_file() or video_path.stat().st_size == 0:
                raise RuntimeError("Evaluation video rendering failed; no nonempty video produced")
            video_metadata = {
                "path": str(video_path),
                "seed": seeds[0],
                "env_index": 0,
                "selection": "first_episode_of_first_seed_env0; no retries or reselection",
                "frame_timing": "pre-step; stops before first terminal autoreset",
                "frame_count": len(recorder),
                "fps": fps,
                "duration_seconds": len(recorder) / fps,
                "width": 1280,
                "height": 720,
                "camera": camera,
                "episode": episodes[0],
            }
    finally:
        env.close()
    if _checkpoint_sha256(checkpoint) != checkpoint_digest:
        raise RuntimeError("Checkpoint changed during evaluation; refusing to misattribute metrics")
    report = {
        "schema_version": 1,
        "metadata": {
            "created_at": timestamp.isoformat(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest,
            "source_run_dir": str(run_dir.resolve()),
            "device": device,
            "backend": backend,
            "seeds": seeds,
            "num_envs": num_envs,
            "expected_episode_count": len(seeds) * num_envs,
            "ctrl_dt": ctrl_dt,
            "episode_horizon_seconds": horizon,
            "max_episode_steps": max_steps,
            "seed_strategy": "one_environment_public_seed_then_full_reset",
            "episode_selection": "first_completed_episode_per_seed_and_env",
            "deterministic_actions": True,
            "render_mode": "none",
            "video": video_metadata,
            "sim2sim_strict": True,
            "aggregation": "equal_episode_weight; population_std; linear_quantiles",
            "metric_source": "named raw critic terms; final_observation on terminal transitions",
            "metric_provenance": g1_metric_provenance(),
            "vx_reference": "commanded vx at action selection, before command resampling",
            "fall_definition": (
                "non-timeout termination; simultaneous termination and timeout is failure"
            ),
            "units": {"vx_mae": "m/s", "abs_vy_mean": "m/s", "abs_wz_mean": "rad/s"},
            "contract_snapshot": extract_contract_snapshot(cfg),
            "evaluation_overrides": overrides,
            "effective_config": OmegaConf.to_container(effective, resolve=True),
            "effective_env_overrides": env_overrides,
            "critic_term_slices": {
                name: [span.start, span.stop] for name, span in metrics.slices.items()
            },
        },
        "summary": _summarize(episodes),
        "per_seed": {
            str(seed): _summarize([episode for episode in episodes if episode["seed"] == seed])
            for seed in seeds
        },
        "episodes": episodes,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as episode_file:
        writer = csv.DictWriter(episode_file, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    report_path = output_dir / "metrics.json"
    report_path.write_text(serialized, encoding="utf-8")
    return report_path


def _validate_metrics_report(report: Any, *, backend: str) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError(f"{backend}: unsupported metrics report schema")
    metadata = report.get("metadata", {})
    episodes = report.get("episodes")
    if not isinstance(metadata, dict) or metadata.get("backend") != backend:
        raise ValueError(f"Metrics reports must follow SOURCE_ORDER; expected {backend}")
    required = {
        "seeds": list(_EVALUATION_SEEDS),
        "num_envs": 25,
        "expected_episode_count": 100,
        "ctrl_dt": 0.02,
        "episode_horizon_seconds": 20.0,
        "max_episode_steps": 1000,
        "deterministic_actions": True,
        "render_mode": "none",
        "sim2sim_strict": True,
        "episode_selection": "first_completed_episode_per_seed_and_env",
        "seed_strategy": "one_environment_public_seed_then_full_reset",
        "metric_source": "named raw critic terms; final_observation on terminal transitions",
        "metric_provenance": g1_metric_provenance(),
        "vx_reference": "commanded vx at action selection, before command resampling",
        "units": {"vx_mae": "m/s", "abs_vy_mean": "m/s", "abs_wz_mean": "rad/s"},
    }
    for field, value in required.items():
        if metadata.get(field) != value:
            raise ValueError(f"{backend}: expected metadata.{field}={value!r}")
    checkpoint_sha = metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha) is None:
        raise ValueError(f"{backend}: missing or invalid checkpoint SHA256")
    if not isinstance(episodes, list) or len(episodes) != 100:
        raise ValueError(f"{backend}: evaluation must contain exactly 100 episodes")
    seen = set()
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError(f"{backend}: invalid episode record")
        seed, env_index = episode.get("seed"), episode.get("env_index")
        if (
            type(seed) is not int
            or seed not in _EVALUATION_SEEDS
            or type(env_index) is not int
            or not 0 <= env_index < 25
            or (seed, env_index) in seen
        ):
            raise ValueError(f"{backend}: expected one episode per seed/environment pair")
        seen.add((seed, env_index))
        length = episode.get("episode_length_steps")
        terminated, truncated = episode.get("terminated"), episode.get("truncated")
        if (
            type(length) is not int
            or not 1 <= length <= 1000
            or type(terminated) is not bool
            or type(truncated) is not bool
            or not (terminated or truncated)
        ):
            raise ValueError(f"{backend}: invalid completed-episode length or termination flags")
        for field in ("episode_time_seconds", "vx_mae", "abs_vy_mean", "abs_wz_mean"):
            value = episode.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{backend}: invalid episode metric {field}")
        success = truncated and not terminated and length == 1000
        if (
            not math.isclose(episode["episode_time_seconds"], length * 0.02, abs_tol=1e-9)
            or episode.get("full_episode_success") is not success
            or episode.get("full_20s_success") is not success
        ):
            raise ValueError(f"{backend}: inconsistent episode time or success flags")
    effective_config = metadata.get("effective_config")
    source_contract = metadata.get("contract_snapshot")
    if (
        not isinstance(effective_config, dict)
        or not isinstance(source_contract, dict)
        or not source_contract
    ):
        raise ValueError(f"{backend}: missing effective config or validated profile contract")
    effective = cast(DictConfig, OmegaConf.create(effective_config))
    g1_evaluation_spec(effective)
    for field, value in {
        "algo.algo": "ppo",
        "algo.num_envs": 25,
        "training.sim_backend": backend,
        "training.play_render_mode": "none",
        "training.sim2sim_strict": True,
        "training.no_play": True,
        "training.evaluation.seeds": list(_EVALUATION_SEEDS),
        "training.evaluation.num_envs": 25,
        "env.ctrl_dt": 0.02,
        "env.max_episode_seconds": 20.0,
        "env.events.base_mass": None,
        "env.events.pd_gains": None,
    }.items():
        if OmegaConf.select(effective, field) != value:
            raise ValueError(f"{backend}: inconsistent effective configuration at {field}")
    expected_contract = copy.deepcopy(source_contract)
    try:
        expected_contract["env.observations"][effective.env.policy_observation_group][
            "enable_corruption"
        ] = False
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{backend}: invalid original observation contract") from exc
    effective_contract = extract_contract_snapshot(effective)
    if effective_contract != expected_contract:
        raise ValueError(f"{backend}: effective policy contract differs beyond actor corruption")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "contract_snapshot": source_contract,
        "effective_contract": effective_contract,
        "initial_state_and_commands": {
            "events": OmegaConf.to_container(effective.env.events, resolve=True),
            "commands": OmegaConf.to_container(effective.env.commands, resolve=True),
            "joint_names": list(effective.env.scene.entities.robot.joint_names),
            "actuator_names": list(effective.env.scene.entities.robot.actuator_names),
        },
        "evaluation_overrides": metadata.get("evaluation_overrides"),
        "metric_source": metadata.get("metric_source"),
        "metric_provenance": metadata["metric_provenance"],
        "vx_reference": metadata.get("vx_reference"),
        "units": metadata.get("units"),
    }


def aggregate_ppo_metrics_reports(reports: Sequence[Path], *, output_path: Path) -> Path:
    """Pool four validated 100-episode reports in the owner's ``SOURCE_ORDER``.

    The approved cohort uses seeds 101--104, 25 envs per seed and the 20-second
    horizon. Checkpoint identity, policy contracts, reset distributions and metric
    definitions must agree. Summaries are recomputed from all 400 equally weighted
    episode records, never averaged percentiles or duration-weighted means. These
    descriptive checkpoint metrics do not establish training convergence.
    """
    from unilab.training.multi_source import SOURCE_ORDER

    if len(reports) != len(SOURCE_ORDER):
        raise ValueError(f"Expected exactly four metrics reports in SOURCE_ORDER {SOURCE_ORDER}")
    combined: list[dict[str, Any]] = []
    per_backend = {}
    source_reports = {}
    common_contract = None
    for backend, path in zip(SOURCE_ORDER, reports, strict=True):
        path = Path(path).resolve()
        report_text = path.read_text(encoding="utf-8")
        report = json.loads(report_text)
        comparison = _validate_metrics_report(report, backend=backend)
        if common_contract is None:
            common_contract = comparison
        elif comparison != common_contract:
            differing = [
                field for field in comparison if comparison[field] != common_contract[field]
            ]
            raise ValueError(
                f"{backend}: reports disagree on checkpoint/config contract: {differing}"
            )
        episodes = report["episodes"]
        per_backend[backend] = _summarize(episodes)
        combined.extend({**episode, "backend": backend} for episode in episodes)
        source_reports[backend] = {
            "path": str(path),
            "sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        }
    result = {
        "schema_version": 1,
        "metadata": {
            "source_order": list(SOURCE_ORDER),
            "source_reports": source_reports,
            "seeds": list(_EVALUATION_SEEDS),
            "num_envs_per_seed_per_backend": 25,
            "episodes_per_backend": 100,
            "expected_episode_count": 400,
            "episode_horizon_seconds": 20.0,
            "ctrl_dt": 0.02,
            "aggregation": "equal_episode_weight; population_std; linear_quantiles",
            "convergence_assessed": False,
            "interpretation": "Descriptive fixed-checkpoint metrics; not evidence of training convergence.",
            **cast(dict[str, Any], common_contract),
        },
        "summary": _summarize(combined),
        "per_backend": per_backend,
        "episodes": combined,
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")
    return output_path
