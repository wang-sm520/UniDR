"""Fixed single-source policy holdout with phase-matched reference rendering."""

from __future__ import annotations

import json
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tensordict import TensorDict
from uni_rl.logging.single_run_audit import audit_single_run
from unisim.backend.playback_common import write_playback_video
from unisim.visualization import render_many

from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.utils.sim2sim import extract_contract_snapshot, resolve_sim2sim_config
from unilab.visualization.playback_session import SnapshotPlaybackSession
from unilab.visualization.unidr_holdout import (
    FPS,
    STEPS,
    HoldoutPlan,
    PhaseTelemetry,
    _models,
    _read_json,
    _task_profile,
    validate_video,
)

SOURCES = ("motrix", "isaacsim", "isaacgym", "genesis")
RGBA = (0.0, 0.85, 1.0, 0.55)


def _profile(cfg: DictConfig) -> dict[str, Any]:
    profile = _task_profile(cfg)
    for key in (
        "seed",
        "isaacsim_device_id",
        "isaacgym_device_id",
        "genesis_device_id",
        "isaacsim_worker_timeout_s",
        "genesis_integrator",
    ):
        profile["env"].pop(key, None)
    # Binding names differ where the source exposes bodies instead of MuJoCo geoms.
    profile["env"]["scene"]["entities"]["robot"].pop("geom_names", None)
    return profile


def preflight(
    checkpoint: Path,
    root: Path,
    *,
    expected_iterations: int = 20000,
    expected_num_envs: int = 1024,
) -> HoldoutPlan:
    """Validate the fixed completed budget and strict contract before env creation."""
    if any(
        type(value) is not int or value < 1 for value in (expected_iterations, expected_num_envs)
    ):
        raise ValueError("Expected iterations and environments must be positive integers")
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.name != f"model_{expected_iterations - 1}.pt":
        raise ValueError("Only the explicitly requested final checkpoint is allowed")
    audit = audit_single_run(
        checkpoint.parent, expected_iterations=expected_iterations, num_envs=expected_num_envs
    )
    run = _read_json(checkpoint.parent / "run_config.json")
    source = OmegaConf.create(run["config"])
    if (
        source.training.task_name != "G1FlipTracking"
        or source.training.sim_backend not in SOURCES
        or source.algo.max_iterations != expected_iterations
        or source.algo.num_envs != expected_num_envs
        or source.algo.num_steps_per_env != 24
        or source.algo.algorithm.num_learning_epochs != 5
        or source.algo.algorithm.num_mini_batches != 4
        or source.algo.resume
        or source.algo.resume_path
        or source.training.sim2sim_strict is not True
        or run.get("contract_snapshot") != extract_contract_snapshot(source)
    ):
        raise ValueError("Expected a complete fresh aligned single-source G1FlipTracking run")
    with initialize_config_dir(config_dir=str(root / "src/unilab/conf/ppo"), version_base="1.3"):
        target = compose("config", overrides=["task=g1_flip_tracking/mujoco"])
    resolve_sim2sim_config(checkpoint.parent, target, algo_name="ppo", strict=True)
    if _profile(source) != _profile(target):
        raise ValueError("Single-source holdout must preserve the native task and rewards")
    manifest = _read_json(checkpoint.parent / "single_manifest.json")
    behavior = {key: manifest[key] for key in ("sources", "algorithm", "assets")}
    digest = sha256(json.dumps(behavior, sort_keys=True, default=str).encode()).hexdigest()
    if (
        digest != manifest.get("digest")
        or manifest.get("source") != str(source.training.sim_backend)
        or manifest.get("expected_iterations") != expected_iterations
        or manifest.get("num_envs") != expected_num_envs
    ):
        raise ValueError("Training manifest provenance or digest mismatch")
    assets = manifest.get("assets", {})
    required = {
        str(target.env.scene.model_file),
        "src/unilab/assets/robots/g1/g1.xml",
        "src/unilab/assets/" + str(target.env.commands.motion.params.motion_file),
    }
    if not isinstance(assets, dict) or not required <= assets.keys():
        raise ValueError("Training asset fingerprints must include the model and motion")
    for filename, expected in assets.items():
        path = (root / filename).resolve(strict=True)
        if (
            not path.is_relative_to(root.resolve())
            or sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Training asset changed: {filename}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_hash = sha256(checkpoint.read_bytes()).hexdigest()
    if audit["final_checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Checkpoint changed after its native training audit")
    models = _models(target)
    transitions = expected_iterations * expected_num_envs * 24
    for name, model in zip(("actor", "critic"), models, strict=True):
        state = saved.get(f"{name}_state_dict")
        if not isinstance(state, dict) or any(
            not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
            for value in state.values()
        ):
            raise ValueError(f"Missing or non-finite {name} state")
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
    target.training.play_only, target.training.play_env_num, target.training.device = True, 1, "cpu"
    OmegaConf.update(target, "env.seed", 1, force_add=True)
    metadata = dict(
        source=str(source.training.sim_backend),
        expected_iterations=expected_iterations,
        training_num_envs=expected_num_envs,
        total_transitions=transitions,
        optimizer_steps=expected_iterations * 20,
        assets=assets,
        training_audit=audit,
        manifest_digest=digest,
    )
    return HoldoutPlan(checkpoint, checkpoint_hash, target, models[0], metadata)


def _render_models(model_file: Path, output: Path) -> tuple[Any, list[str]]:
    """Build detached visual copies; never alter the physics environment or assets."""
    spec = mujoco.MjSpec.from_file(str(model_file))
    original = spec.compile()
    for texture in spec.textures:
        if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            texture.rgb1, texture.rgb2 = [0.3, 0.5, 0.7], [0.0, 0.0, 0.0]
        elif texture.name == "groundplane":
            texture.rgb1, texture.rgb2 = [0.2, 0.3, 0.4], [0.1, 0.2, 0.3]
            texture.mark, texture.markrgb = mujoco.mjtMark.mjMARK_EDGE, [0.8, 0.8, 0.8]
    for material in spec.materials:
        if material.name == "groundplane":
            material.texrepeat, material.texuniform, material.reflectance = [5.0, 5.0], True, 0.2
    spec.visual.headlight.diffuse, spec.visual.headlight.ambient = [0.6] * 3, [0.3] * 3
    spec.visual.headlight.specular = [0.0] * 3
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
    spec.visual.map.fogstart, spec.visual.map.fogend = 3.0, 10.0
    model = spec.compile()
    for field in (
        "qpos0",
        "body_mass",
        "body_inertia",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
        "actuator_gainprm",
        "actuator_biasprm",
    ):
        np.testing.assert_array_equal(getattr(original, field), getattr(model, field))
    paths = [str(output / name) for name in ("classic.mjb", "reference.mjb")]
    mujoco.mj_saveModel(model, paths[0])
    visual = (model.geom_group == 2) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    if int(visual.sum()) < 20:
        raise ValueError("G1 reference model is missing its visual meshes")
    model.geom_rgba[visual], model.geom_matid[visual] = RGBA, -1
    mujoco.mj_saveModel(model, paths[1])
    return model, paths


def _reference_states(
    model: Any, motion_file: Path, phases: np.ndarray, joints: list[str]
) -> np.ndarray:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, 30)]
    if (model.nq, model.nv, model.nbody) != (36, 35, 31) or names != joints:
        raise ValueError("G1 reference DoF, body layout or joint order mismatch")
    data, cache = mujoco.MjData(model), []
    with np.load(motion_file) as clip:
        if (
            int(clip["fps"][0]) != FPS
            or clip["joint_pos"].shape != (225, 29)
            or np.any((phases < 0) | (phases >= 225))
        ):
            raise ValueError("Reference clip or recorded phase mismatch")
        for frame in range(225):
            data.qpos[:3], data.qpos[3:7], data.qpos[7:] = (
                clip["body_pos_w"][frame, 1],
                clip["body_quat_w"][frame, 1],
                clip["joint_pos"][frame],
            )
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.xpos, clip["body_pos_w"][frame], atol=1e-6)
            np.testing.assert_allclose(
                np.abs(np.sum(data.xquat * clip["body_quat_w"][frame], axis=-1)), 1.0, atol=1e-6
            )
            data.qvel[:3] = clip["body_lin_vel_w"][frame, 1]
            data.qvel[3:6] = data.xmat[1].reshape(3, 3).T @ clip["body_ang_vel_w"][frame, 1]
            data.qvel[6:] = clip["joint_vel"][frame]
            cache.append(np.concatenate(([frame / FPS], data.qpos, data.qvel)))
    return np.stack(cache)[phases, None, :]


def record_reference(
    checkpoint: Path,
    output: Path,
    *,
    root: Path,
    expected_iterations: int = 20000,
    expected_num_envs: int = 1024,
) -> dict[str, Any]:
    """Run genuine MuJoCo physics once, rendering actual and reference from those states."""
    plan = preflight(
        checkpoint,
        root,
        expected_iterations=expected_iterations,
        expected_num_envs=expected_num_envs,
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "mujoco_config.json").write_text(
        json.dumps(OmegaConf.to_container(plan.config, resolve=True), indent=2) + "\n"
    )
    override = BackendAdapter(
        plan.config, root_dir=root, algo_name="ppo"
    ).build_task_env_cfg_override()
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
        motion, robot = env.command_manager.get_term("motion"), env.scene["robot"]
        if int(motion.time_steps[0]) != 0:
            raise ValueError("Holdout must start at reference frame zero")
        feet = [
            robot.body_names.index(name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        session = SnapshotPlaybackSession(env, width=1280, height=720, num_processes=2)
        with torch.inference_mode(), (output / "telemetry.jsonl").open("x") as stream:
            for step in range(STEPS):
                frame = int(motion.time_steps[0])
                actions = (
                    plan.actor(TensorDict({"actor": torch.as_tensor(obs["obs"])}, batch_size=[1]))
                    .cpu()
                    .numpy()
                )
                if actions.shape != (1, 29) or not np.isfinite(actions).all():
                    raise ValueError("Loaded policy produced invalid actions")
                state = env.step(actions)
                if not np.isfinite(state.reward).all() or not all(
                    np.isfinite(v).all() for v in state.obs.values()
                ):
                    raise ValueError("Non-finite MuJoCo transition")
                session.snapshot()  # Capture the terminal state before manual reset.
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
        states = np.stack(session.snapshots)
    if states.shape != (STEPS, 1, 72) or not np.isfinite(states).all():
        raise ValueError("Incomplete or non-finite physics snapshots")
    if any(not torch.equal(value, plan.actor.state_dict()[key]) for key, value in before.items()):
        raise RuntimeError("Playback modified actor or normalizer state")
    phases = np.array([row["next_reference_frame"] for row in telemetry.rows])
    model, paths = _render_models(root / str(plan.config.env.scene.model_file), output)
    refs = _reference_states(
        model,
        root / "src/unilab/assets" / str(plan.config.env.commands.motion.params.motion_file),
        phases,
        list(plan.config.env.scene.entities.robot.joint_names),
    )
    np.savez_compressed(
        output / "physics_snapshots.npz", states=states, reference=refs, phase=phases
    )
    frames = render_many.render_states_get_frames_tracking(
        list(np.concatenate((states, refs), axis=1)),
        paths,
        width=1280,
        height=720,
        max_extra_envs=1,
        render_spacing=0.0,
        cam_distance=3.6,
        cam_elevation=-12.0,
        cam_azimuth=90.0,
    )
    if len(frames) != STEPS or any(float(frame.std()) < 5 for frame in frames):
        raise RuntimeError("Missing or blank rendered frames")
    sheet = Image.new("RGB", (1600, 450))
    for i, index in enumerate((0, 109, 124, 139, 169, 172, 224, 449)):
        sheet.paste(
            Image.fromarray(frames[index]).resize((400, 225)), ((i % 4) * 400, (i // 4) * 225)
        )
    sheet.save(output / "preview.jpg")
    video = output / "front-reference.mp4"
    write_playback_video(str(video), frames, fps=FPS)
    if sha256(checkpoint.read_bytes()).hexdigest() != plan.checkpoint_sha256:
        raise RuntimeError("Playback checkpoint changed")
    result = dict(
        checkpoint=str(plan.checkpoint),
        checkpoint_sha256=plan.checkpoint_sha256,
        **plan.metadata,
        backend="mujoco",
        strict_preflight=True,
        fresh_policy_steps=STEPS,
        seed=1,
        actor_normalizer_unchanged=True,
        video=str(video),
        format=validate_video(video),
        video_sha256=sha256(video.read_bytes()).hexdigest(),
        text_overlays=False,
        reference_rgba=RGBA,
        reference_phase="post-step phase before manual reset; original world XY/yaw",
        terminated=sum(row["terminated"] for row in telemetry.rows),
        timeout_resets=sum(row["truncated"] for row in telemetry.rows),
        reference_physics_resets=sum(row["clip_wrap"] for row in telemetry.rows),
        landing_candidates=sum(row["landing_candidate"] for row in telemetry.rows),
    )
    (output / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
