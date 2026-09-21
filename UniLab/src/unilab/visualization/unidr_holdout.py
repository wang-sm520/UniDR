"""Fixed final-checkpoint MuJoCo playback with strict preflight and evidence."""

from __future__ import annotations

import json
import subprocess
from contextlib import closing
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from uni_rl.algos.rsl_rl import normalize_ppo_train_cfg
from unisim.backend.base import CameraCfg

from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.training import algo_config_dict
from unilab.utils.sim2sim import extract_contract_snapshot, resolve_sim2sim_config
from unilab.visualization.playback_session import SnapshotPlaybackSession

SOURCE_ORDER = ("isaacsim", "isaacgym", "genesis", "motrix")
STEPS, FPS = 1000, 50


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _task_profile(cfg: DictConfig) -> dict[str, Any]:
    profile = {key: OmegaConf.to_container(cfg[key], resolve=True) for key in ("env", "reward")}
    env = profile["env"]
    assert isinstance(env, dict)
    # These source-only adapter controls have no effect on the native MuJoCo
    # target. Retain them in training manifests; compare all task fields strictly.
    for name in ("motrix_disable_self_collision", "genesis_enable_self_collision"):
        value = env.pop(name, None)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} must be bool or None")
    return profile


def _models(cfg: DictConfig) -> tuple[MLPModel, MLPModel]:
    normalized = normalize_ppo_train_cfg(algo_config_dict(cfg))
    observations = TensorDict(
        {"actor": torch.zeros(1, 160), "critic": torch.zeros(1, 286)}, batch_size=[1]
    )
    models = []
    for name, output_dim in (("actor", 29), ("critic", 1)):
        options = normalized[name].copy()
        if options.pop("class_name") != "rsl_rl.models.MLPModel":
            raise ValueError("The final holdout requires the native MLP policy")
        models.append(MLPModel(observations, normalized["obs_groups"], name, output_dim, **options))
    return models[0], models[1]


def _validate_adam(
    saved: dict[str, Any], models: tuple[MLPModel, MLPModel], expected_steps: int
) -> None:
    optimizer = saved.get("optimizer_state_dict", {})
    states, groups = optimizer.get("state", {}), optimizer.get("param_groups", [])
    ids = [index for group in groups for index in group["params"]]
    params = [parameter for model in models for parameter in model.parameters()]
    learning_rate = saved["synchronous"].get("learning_rate", float("nan"))
    if (
        len(ids) != len(params)
        or len(set(ids)) != len(ids)
        or set(ids) != set(states)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
        or any(group.get("lr") != learning_rate for group in groups)
    ):
        raise ValueError("Incomplete final Adam parameter state or learning rate")
    for index, parameter in zip(ids, params, strict=True):
        entry = states[index]
        if float(entry.get("step", -1)) != expected_steps or any(
            not isinstance(entry.get(name), torch.Tensor)
            or entry[name].shape != parameter.shape
            or not bool(torch.isfinite(entry[name]).all())
            for name in ("exp_avg", "exp_avg_sq")
        ):
            raise ValueError(
                f"Final Adam state must contain {expected_steps} complete finite updates"
            )


@dataclass(frozen=True)
class HoldoutPlan:
    checkpoint: Path
    checkpoint_sha256: str
    config: DictConfig
    actor: MLPModel
    metadata: dict[str, Any]


def preflight(checkpoint: Path, root: Path, *, expected_iterations: int = 10000) -> HoldoutPlan:
    """Reject incomplete artifacts or incompatible models before constructing any env."""
    if type(expected_iterations) is not int or not 1 <= expected_iterations <= 10000:
        raise ValueError("Expected iterations must be an integer between 1 and 10000")
    final_index = expected_iterations - 1
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.name != f"model_{final_index}.pt" or not checkpoint.is_file():
        raise ValueError(
            f"Only the fixed final model_{final_index}.pt may enter the MuJoCo holdout"
        )
    run = _read_json(checkpoint.parent / "run_config.json")
    manifest = _read_json(checkpoint.parent / "sources_manifest.json")
    if not {"sources", "algorithm", "assets"} <= manifest.keys():
        raise ValueError("The source manifest must include its complete behavior")
    behavior = {key: manifest[key] for key in ("sources", "algorithm", "assets")}
    digest = sha256(json.dumps(behavior, sort_keys=True, default=str).encode()).hexdigest()
    if digest != manifest.get("digest"):
        raise ValueError("The source manifest digest does not match its behavior")
    if not isinstance(run.get("config"), dict):
        raise ValueError("A complete training config is required")
    source = OmegaConf.create(run["config"])
    planned_iterations = OmegaConf.select(source, "algo.max_iterations")
    if (
        OmegaConf.select(source, "training.task_name") != "G1FlipTracking"
        or type(planned_iterations) is not int
        or planned_iterations < expected_iterations
        or (expected_iterations == 10000 and planned_iterations != 10000)
        or OmegaConf.select(source, "algo.num_envs") != 1024
        or OmegaConf.select(source, "training.sim2sim_strict") is not True
        or tuple(item.name for item in OmegaConf.select(source, "unidr.sources", default=[]))
        != SOURCE_ORDER
    ):
        raise ValueError("Expected the formal aligned four-source G1FlipTracking run")
    snapshot = run.get("contract_snapshot")
    if not isinstance(snapshot, dict) or snapshot != extract_contract_snapshot(source):
        raise ValueError("A complete, consistent training contract snapshot is required")
    with initialize_config_dir(config_dir=str(root / "src/unilab/conf/ppo"), version_base="1.3"):
        target = compose("config", overrides=["task=g1_flip_tracking/mujoco"])
    resolve_sim2sim_config(checkpoint.parent, target, algo_name="ppo", strict=True)
    if _task_profile(source) != _task_profile(target):
        raise ValueError("The final holdout must preserve the native MuJoCo task profile")
    model_file = root / str(target.env.scene.model_file)
    model_file.resolve(strict=True)
    motion = root / "src/unilab/assets" / str(target.env.commands.motion.params.motion_file)
    motion.resolve(strict=True)
    assets = manifest.get("assets")
    if (
        not isinstance(assets, dict)
        or not {str(model_file.relative_to(root)), str(motion.relative_to(root))} <= assets.keys()
    ):
        raise ValueError("The training manifest must include the robot model and motion")
    for filename, expected in assets.items():
        if sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Training asset changed: {filename}")
    # Synchronous checkpoints contain Python/NumPy RNG state in addition to tensors.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sync = saved.get("synchronous", {})
    contract = sync.get("contract", {})
    if (
        saved.get("iter") != final_index
        or sync.get("format") != 1
        or sync.get("complete") is not True
        or sync.get("next_iteration") != expected_iterations
        or sync.get("policy_version") != expected_iterations
        or sync.get("normalizer_version") != expected_iterations * 24
        or sync.get("optimizer_steps") != expected_iterations * 20
        or not manifest.get("digest")
        or contract.get("manifest_digest") != manifest["digest"]
        or run.get("run", {}).get("manifest_digest") != manifest["digest"]
        or run.get("run", {}).get("mode") != "synchronous_four_source"
        or contract.get("num_envs") != 4096
        or contract.get("sources")
        != [(name, index * 1024, (index + 1) * 1024) for index, name in enumerate(SOURCE_ORDER)]
        or sync.get("total_transitions") != expected_iterations * 98304
        or contract.get("train_cfg") != manifest["algorithm"]
    ):
        raise ValueError(
            f"The fixed checkpoint must contain all {expected_iterations} completed central PPO updates"
        )
    models = _models(target)
    for name, model in zip(("actor", "critic"), models, strict=True):
        state = saved.get(f"{name}_state_dict")
        if not isinstance(state, dict) or any(
            not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
            for value in state.values()
        ):
            raise ValueError(f"Missing or non-finite {name} model state")
        count = state.get("obs_normalizer.count")
        if (
            count is None
            or count.numel() != 1
            or count.dtype != torch.int64
            or int(count) != sync.get("total_transitions")
            or int(count) <= 0
        ):
            raise ValueError(f"Missing or incomplete frozen {name} normalizer")
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
    _validate_adam(saved, models, expected_iterations * 20)
    target.training.play_only = True
    target.training.play_env_num = 1
    target.training.device = "cpu"
    OmegaConf.update(target, "env.seed", 1, force_add=True)
    metadata = {
        key: sync[key] for key in ("policy_version", "normalizer_version", "total_transitions")
    }
    metadata["manifest_digest"] = manifest["digest"]
    metadata["expected_iterations"] = expected_iterations
    metadata["planned_iterations"] = planned_iterations
    return HoldoutPlan(
        checkpoint, sha256(checkpoint.read_bytes()).hexdigest(), target, models[0], metadata
    )


@dataclass
class PhaseTelemetry:
    """Conservative kinematic landing evidence, scoped to one uninterrupted attempt."""

    episode: int = 0
    attempt: int = 0
    seen_airborne: bool = False
    seen_inversion: bool = False
    stable_frames: int = 0
    landing_recorded: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)

    def observe(
        self,
        *,
        frame: int,
        next_frame: int,
        upright: float,
        height: float,
        vertical_speed: float,
        feet_z: list[float],
        terminated: bool,
        truncated: bool,
        step: int,
    ) -> dict[str, Any]:
        inverted = upright < 0.0
        airborne = min(feet_z) > 0.12
        done = terminated or truncated
        wrapped = next_frame < frame and not done
        self.seen_airborne |= airborne
        self.seen_inversion |= inverted
        stable = (
            self.seen_airborne
            and self.seen_inversion
            and frame >= 150
            and upright > 0.8
            and max(feet_z) < 0.12
            and abs(vertical_speed) < 0.5
            and not done
            and not wrapped
        )
        self.stable_frames = self.stable_frames + 1 if stable else 0
        landing = self.stable_frames >= 10 and not self.landing_recorded
        self.landing_recorded |= landing
        row = dict(
            step=step,
            time_seconds=(step + 1) / FPS,
            episode=self.episode,
            attempt=self.attempt,
            reference_frame=frame,
            next_reference_frame=next_frame,
            reference_phase=frame / 224,
            upright_z=upright,
            root_height=height,
            root_vertical_speed=vertical_speed,
            feet_height=feet_z,
            inverted=inverted,
            airborne=airborne,
            landing_candidate=landing,
            terminated=terminated,
            truncated=truncated,
            reset=done,
            clip_wrap=wrapped,
        )
        self.rows.append(row)
        if done or wrapped:
            self.episode += int(done)
            self.attempt += 1
            self.seen_airborne = self.seen_inversion = self.landing_recorded = False
            self.stable_frames = 0
        return row


def validate_video(path: Path) -> dict[str, Any]:
    """Fully decode the exported stream and verify its real frame count and format."""
    frames = imageio_ffmpeg.read_frames(str(path))
    try:
        metadata = next(frames)
    finally:
        frames.close()
    decoded = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-xerror",
            "-threads",
            "2",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    counts = [
        int(line.split("=", 1)[1])
        for line in decoded.stdout.splitlines()
        if line.startswith("frame=")
    ]
    if (
        metadata["size"] != (1280, 720)
        or metadata["fps"] != FPS
        or metadata["duration"] < 20
        or not counts
        or counts[-1] != STEPS
    ):
        raise ValueError(
            "Holdout video must decode all 1000 frames at 1280x720, 50 fps, 20 seconds"
        )
    return {
        "frames": counts[-1],
        "width": 1280,
        "height": 720,
        "fps": FPS,
        "seconds": metadata["duration"],
    }


def record_holdout(
    checkpoint: Path, output: Path, *, root: Path, expected_iterations: int = 10000
) -> dict[str, Any]:
    """Record the sole fixed final policy; no model search, training, or action fallback."""
    plan = preflight(checkpoint, root, expected_iterations=expected_iterations)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "mujoco_config.json").write_text(
        json.dumps(OmegaConf.to_container(plan.config, resolve=True), indent=2) + "\n"
    )
    override = BackendAdapter(
        plan.config, root_dir=root, algo_name="ppo"
    ).build_task_env_cfg_override()
    override["seed"] = 1
    before = {key: value.clone() for key, value in plan.actor.state_dict().items()}
    telemetry = PhaseTelemetry()
    with closing(registry_env_factory("G1FlipTracking", "mujoco")(1, override)) as env:
        env.set_autoreset(False)
        obs, _ = env.reset(seed=1)
        if env.action_space.shape != (29,) or {key: value.shape for key, value in obs.items()} != {
            "obs": (1, 160),
            "critic": (1, 286),
        }:
            raise ValueError("MuJoCo policy dimensions must be 160/286/29")
        motion = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        feet = [
            robot.body_names.index(name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        session = SnapshotPlaybackSession(env, width=1280, height=720, num_processes=2)
        with torch.inference_mode(), (output / "telemetry.jsonl").open("x") as stream:
            for step in range(STEPS):
                frame = int(motion.time_steps[0])
                tensor_obs = TensorDict({"actor": torch.as_tensor(obs["obs"])}, batch_size=[1])
                actions = plan.actor(tensor_obs).cpu().numpy()
                if actions.shape != (1, 29) or not np.isfinite(actions).all():
                    raise ValueError("The loaded policy produced invalid actions")
                state = env.step(actions)
                if not np.isfinite(state.reward).all() or not all(
                    np.isfinite(v).all() for v in state.obs.values()
                ):
                    raise ValueError("Non-finite MuJoCo transition")
                session.snapshot()
                quat = robot.data.body_link_quat_w[0, 0]
                row = telemetry.observe(
                    frame=frame,
                    next_frame=int(motion.time_steps[0]),
                    upright=float(1 - 2 * (quat[1] ** 2 + quat[2] ** 2)),
                    height=float(robot.data.body_link_pos_w[0, 0, 2]),
                    vertical_speed=float(robot.data.body_link_lin_vel_w[0, 0, 2]),
                    feet_z=robot.data.body_link_pos_w[0, feet, 2].tolist(),
                    terminated=bool(state.terminated[0]),
                    truncated=bool(state.truncated[0]),
                    step=step,
                )
                row["termination_terms"] = [
                    name
                    for name in env.termination_manager.active_terms
                    if env.termination_manager.get_term(name)[0]
                ]
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                obs = env.reset()[0] if row["reset"] else state.obs
        if any(
            not torch.equal(value, plan.actor.state_dict()[key]) for key, value in before.items()
        ):
            raise RuntimeError("Playback modified actor or normalizer state")
        video = session.render_snapshots(
            output_video=output / "holdout.mp4",
            fps=FPS,
            camera=CameraCfg(
                cam_tracking=True,
                cam_tracking_env_idx=0,
                cam_tracking_extra_envs=0,
                cam_distance=3.2,
                cam_elevation=-12.0,
                cam_azimuth=0.0,
            ),
        )
        if video is None:
            raise RuntimeError("Holdout playback did not produce a video")
    if sha256(plan.checkpoint.read_bytes()).hexdigest() != plan.checkpoint_sha256:
        raise RuntimeError("Playback modified its checkpoint")
    result = dict(
        checkpoint=str(plan.checkpoint),
        checkpoint_sha256=plan.checkpoint_sha256,
        backend="mujoco",
        action_mode="frozen_deterministic_policy",
        seed=1,
        video=validate_video(Path(video)),
        **plan.metadata,
        resets=sum(row["reset"] for row in telemetry.rows),
        inverted_frames=sum(row["inverted"] for row in telemetry.rows),
        landing_candidates=sum(row["landing_candidate"] for row in telemetry.rows),
    )
    (output / "holdout.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
