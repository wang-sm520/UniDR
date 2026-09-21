"""Repeated fixed-final MuJoCo trials with an explicit single-flip success rule."""

from __future__ import annotations

import csv
import json
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from uni_rl.logging.synchronous_report import audit_run

from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand, MotionCommandCfg
from unilab.visualization.failure_tail import FailureTailTerminations
from unilab.visualization.single_reference import preflight
from unilab.visualization.unidr_holdout import preflight as joint_preflight

SOURCES = ("motrix", "isaacsim", "isaacgym", "genesis", "joint")
PROTOCOL: dict[str, Any] = dict(
    name="g1_flip_then_five_seconds_without_fall_v2",
    trials_per_policy=10,
    seeds=list(range(1, 11)),
    motion_steps=224,
    observation_steps=250,
    horizon_steps=474,
    control_dt_seconds=0.02,
    deterministic_actor=True,
    extra_randomization=False,
    minimum_airborne_steps=3,
    minimum_feet_height_m=0.12,
    contact_distance_tolerance_m=0.002,
    rotation_range_degrees=[330.0, 390.0],
    final_stable_steps=10,
    minimum_upright_z=0.8,
    maximum_abs_vertical_speed_m_s=0.5,
    post_motion_minimum_upright_z=0.5,
    post_motion_minimum_pelvis_height_m=0.3,
    native_done="record only; continue the same live policy without reset",
    reference_end="hold original final reference frame; no physical state writes",
    forbid_reference_reset=True,
    forbid_nonfoot_ground_contact=True,
    interpretation="Repeated original deterministic scenario; not independent randomized robustness trials",
)


class _TrialMotion(MotionCommand):
    def _update_command(self, env_ids: np.ndarray | None) -> None:
        if env_ids is None and np.all(self.time_steps == self.sampler.current_clip_end_frames):
            # Refresh the existing final reference; never call the resample path
            # which teleports the robot. Policy/physics continue normally.
            env_ids = np.arange(self.num_envs, dtype=np.int32)
        super()._update_command(env_ids)


@dataclass(kw_only=True)
class TrialMotionCfg(MotionCommandCfg):
    """Evaluation-only final-reference hold for uninterrupted landing observation."""

    def build(self, env: Any) -> MotionCommand:
        if env.num_envs != 1:
            raise ValueError("Flip trials require exactly one environment")
        return _TrialMotion(self, env)


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False) + "\n")


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _angles(quaternions: np.ndarray, basis: np.ndarray) -> np.ndarray:
    forward = np.array([_rotation(q)[:, 0] for q in quaternions])
    angles = np.unwrap(np.arctan2(-forward @ basis[:, 2], forward @ basis[:, 0]))
    return np.degrees(angles - angles[0])


def classify(trace: dict[str, np.ndarray], rows: list[dict], direction: float) -> dict:
    """Score a full clip followed by five unreset seconds, separately from native done."""
    count = len(rows)
    if count < 1 or direction not in (-1.0, 1.0):
        raise ValueError("A nonempty attempt and reference rotation direction are required")
    if [row["step"] for row in rows] != list(range(count)):
        raise ValueError("Non-contiguous attempt steps")
    if any(len(value) != count or not np.isfinite(value).all() for value in trace.values()):
        raise ValueError("Non-finite or misaligned attempt telemetry")
    motion_end = min(count, PROTOCOL["motion_steps"])
    airborne = ~trace["feet_contact"][:motion_end].any(axis=1) & (
        trace["feet_height"][:motion_end].min(axis=1) > PROTOCOL["minimum_feet_height_m"]
    )
    consecutive, longest = 0, 0
    for flag in airborne:
        consecutive = consecutive + 1 if flag else 0
        longest = max(longest, consecutive)
    final_turn = float(trace["rotation_degrees"][motion_end - 1] * direction)
    stable = (
        trace["feet_contact"].all(axis=1)
        & (trace["upright_z"] > PROTOCOL["minimum_upright_z"])
        & (np.abs(trace["vertical_speed"]) < PROTOCOL["maximum_abs_vertical_speed_m_s"])
    )
    terminal = next((r for r in rows if r["terminated"] or r["truncated"]), None)
    wrapped = any(r["next_reference_frame"] < r["reference_frame"] for r in rows)
    fallen = (
        trace["nonfoot_ground_contact"]
        | (trace["upright_z"] < PROTOCOL["post_motion_minimum_upright_z"])
        | (trace["pelvis_height"] < PROTOCOL["post_motion_minimum_pelvis_height_m"])
    )
    post_falls = np.flatnonzero(fallen[PROTOCOL["motion_steps"] :])
    conditions = dict(
        complete_horizon=count == PROTOCOL["horizon_steps"],
        no_reference_reset=not wrapped,
        airborne=longest >= PROTOCOL["minimum_airborne_steps"],
        inverted_in_air=bool(np.any(airborne & (trace["upright_z"][:motion_end] < 0))),
        full_backflip=PROTOCOL["rotation_range_degrees"][0]
        <= final_turn
        <= PROTOCOL["rotation_range_degrees"][1],
        stable_landing_at_clip_end=count >= PROTOCOL["motion_steps"]
        and bool(stable[motion_end - PROTOCOL["final_stable_steps"] : motion_end].all()),
        no_nonfoot_contact_during_motion=not bool(
            trace["nonfoot_ground_contact"][:motion_end].any()
        ),
        five_seconds_without_fall=count == PROTOCOL["horizon_steps"] and not len(post_falls),
    )
    return dict(
        success=all(conditions.values()),
        conditions=conditions,
        failed_conditions=[name for name, passed in conditions.items() if not passed],
        steps=count,
        duration_seconds=count * PROTOCOL["control_dt_seconds"],
        rotation_degrees_in_reference_direction=final_turn,
        longest_airborne_steps=longest,
        terminal_terms=[] if terminal is None else terminal["termination_terms"],
        first_native_failure_seconds=(
            None if terminal is None else (terminal["step"] + 1) * PROTOCOL["control_dt_seconds"]
        ),
        first_post_motion_fall_seconds=(
            None
            if not len(post_falls)
            else (int(post_falls[0]) + 1) * PROTOCOL["control_dt_seconds"]
        ),
        return_sum=sum(r["reward"] for r in rows),
    )


def inspect_states(model: Any, states: np.ndarray) -> dict[str, np.ndarray]:
    """Detached FK and geometric contact detection; never estimate contact support force."""
    if states.ndim != 2 or states.shape[1] != 72 or not np.isfinite(states).all():
        raise ValueError("Expected finite initial+post-step G1 physics snapshots")
    if not np.allclose(np.linalg.norm(states[:, 4:8], axis=1), 1, atol=1e-5):
        raise ValueError("Non-unit pelvis quaternion")
    feet = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    ground = np.flatnonzero(
        (model.geom_bodyid == 0) & (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    )
    if min(feet) <= 0 or not len(ground):
        raise ValueError("Missing feet or world ground plane")
    pairs = [
        (int(plane), geom, int(model.geom_bodyid[geom]))
        for plane in ground
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] != 0
        and (
            (model.geom_contype[geom] & model.geom_conaffinity[plane])
            or (model.geom_contype[plane] & model.geom_conaffinity[geom])
        )
    ]
    data, contacts, heights, other = mujoco.MjData(model), [], [], []
    for state in states[1:]:
        data.qpos[:], data.qvel[:] = state[1:37], state[37:]
        mujoco.mj_forward(model, data)
        touched, nonfoot = set(), False
        for plane, geom, body in pairs:
            # Active contacts omit positive gaps when margin=0. Use signed
            # geometry distance to implement the declared 2 mm tolerance.
            distance = mujoco.mj_geomDistance(model, data, plane, geom, 10.0, None)
            if distance > PROTOCOL["contact_distance_tolerance_m"]:
                continue
            if body in feet:
                touched.add(body)
            elif body != 0:
                nonfoot = True
        contacts.append([foot in touched for foot in feet])
        heights.append(data.xpos[feet, 2].copy())
        other.append(nonfoot)
    return dict(
        feet_contact=np.asarray(contacts, dtype=bool),
        feet_height=np.asarray(heights),
        nonfoot_ground_contact=np.asarray(other, dtype=bool),
        upright_z=np.array([_rotation(q)[2, 2] for q in states[1:, 4:8]]),
        vertical_speed=states[1:, 39],
        pelvis_height=states[1:, 3],
        rotation_degrees=_angles(states[:, 4:8], _rotation(states[0, 4:8]))[1:],
    )


def checkpoint_schedule(series: bool) -> list[tuple[str, int]]:
    """Enumerate all historical save points or just the predetermined final models."""
    return [
        (source, index)
        for source in SOURCES
        for index in (
            [
                *range(0, 5000 if source == "joint" else 20000, 500),
                4999 if source == "joint" else 19999,
            ]
            if series
            else [4999 if source == "joint" else 19999]
        )
    ]


def write_checkpoint_table(output: Path, results: list[dict]) -> None:
    """Write an index-aligned table only when the complete historical series exists."""
    lookup = {(row["source"], row["checkpoint_index"]): row for row in results}
    if len(lookup) != len(results) or set(lookup) != set(checkpoint_schedule(True)):
        raise ValueError("Checkpoint table requires all 175 unique scheduled evaluations")
    for (_, index), row in lookup.items():
        if (
            row["seed"] != 1
            or row["completed_updates"] != index + 1
            or row["steps"] != PROTOCOL["horizon_steps"]
            or type(row["success"]) is not bool
        ):
            raise ValueError("Incomplete or inconsistent checkpoint evaluation")
    columns = [
        "checkpoint 编号",
        "实际完成更新",
        "Motrix",
        "Isaac Sim",
        "Isaac Gym",
        "Genesis",
        "四源联合",
    ]
    table = []
    for index in sorted({index for _, index in lookup}):
        statuses = [
            ("成功" if lookup[source, index]["success"] else "失败")
            if (source, index) in lookup
            else "—"
            for source in SOURCES
        ]
        table.append([str(index), str(index + 1), *statuses])
    with (output / "checkpoint_success.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(table)
    text = [
        "# 每 500 个 checkpoint 刻度的 MuJoCo 动作成功表",
        "",
        "成功标准：完整后空翻并落地，动作完成后连续 5 秒不摔倒。"
        "复用已冻结的五秒协议，每个模型 seed=1、确定性策略、无额外 DR，"
        "224 步动作 + 250 步末帧保持，共 9.48 秒；全程不 reset。",
        "",
        "这是对既有固定 checkpoint 的补充评测：旧视频会循环/reset，"
        "不能单凭旧视频认证连续五秒。本表没有训练新模型，也不用于 checkpoint 选择。"
        "每格只有一次确定性试验，表示通过/未通过，不是统计成功率。",
        "",
        "编号按实际文件名：model_500.pt 完成 501 次更新，model_0.pt 已完成 1 次更新；"
        "单源最终为 model_19999.pt（20000 次），联合最终为 model_4999.pt（5000 次）。"
        "— 表示该训练没有这个 checkpoint。相同行编号不代表相同采样预算。",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
        *["| " + " | ".join(row) + " |" for row in table],
        "",
        "[下载宽表 CSV](checkpoint_success.csv) · [逐模型判定与失败原因](trials.csv) · "
        "[原始汇总](summary.json) · [固定判定参数](protocol.json)",
        "",
        "每个 来源/model_XXXXX/trial_01/ 目录包含原始状态、动作、相位、原生终止项、"
        "几何触地证据及模型/数据哈希。真正终止条件仍记录，跟踪误差触发本身不等于摔倒。"
        "完整判定包括离地、倒立、参考方向旋转330–390°、片段末稳定落地；"
        "之后250步出现非足部距地≤2mm、骨盆upright_z<0.5或高度<0.3m均判失败。",
        "",
    ]
    (output / "SUCCESS.md").write_text("\n".join(text))


def run_trials(
    root: Path,
    output: Path,
    *,
    checkpoint_series: bool = False,
    single_runs: list[tuple[str, Path]] | None = None,
    expected_iterations: int = 20000,
    expected_num_envs: int = 1024,
) -> dict:
    """Evaluate fixed final repeats or every historical checkpoint once, without selection."""
    root, output = root.resolve(), output.resolve()
    if single_runs is not None:
        sources = [source for source, _ in single_runs]
        if (
            checkpoint_series
            or not sources
            or len(set(sources)) != len(sources)
            or not set(sources) <= set(SOURCES[:-1])
            or any(
                type(value) is not int or value < 1
                for value in (expected_iterations, expected_num_envs)
            )
        ):
            raise ValueError(
                "Explicit single runs require unique training sources and positive budgets"
            )
        runs = {source: (root / run).resolve() for source, run in single_runs}
        schedule = [(source, expected_iterations - 1) for source in sources]
    else:
        if (expected_iterations, expected_num_envs) != (20000, 1024):
            raise ValueError("Custom budgets require explicit single runs")
        schedule = checkpoint_schedule(checkpoint_series)
        sources = list(SOURCES)
        runs = {
            source: root
            / (
                "logs/comparison-joint-1024-5000-20260916/train"
                if source == "joint"
                else f"logs/comparison-1024-20000-20260916/{source}"
            )
            for source in sources
        }
    output.mkdir(parents=True, exist_ok=False)
    protocol = {**PROTOCOL, "trials_per_policy": 1, "seeds": [1]} if checkpoint_series else PROTOCOL
    _write(output / "protocol.json", protocol)
    _write(output / "schedule.json", schedule)
    _write(output / "runs.json", {source: str(run) for source, run in runs.items()})
    results = []
    for source, index in schedule:
        run = runs[source]
        checkpoint = run / f"model_{index}.pt"
        if source == "joint":
            audit_run(
                run, expected_iterations=index + 1, num_envs=1024, final_checkpoint=checkpoint
            )
            plan = joint_preflight(checkpoint, root, expected_iterations=index + 1)
        else:
            plan = preflight(
                checkpoint,
                root,
                expected_iterations=expected_iterations,
                expected_num_envs=expected_num_envs,
                selected_checkpoint=checkpoint_series,
            )
            if plan.metadata["source"] != source:
                raise ValueError("Source/checkpoint mismatch")
        params = plan.config.env.commands.motion.params
        if (
            params.sampling_mode != "start"
            or params.truncate_on_clip_end
            or any(
                value != 0
                for ranges in (params.pose_range, params.velocity_range)
                for pair in ranges.values()
                for value in pair
            )
            or any(value != 0 for value in params.joint_position_range)
        ):
            raise ValueError("This protocol requires the original zero-perturbation reset")
        model = mujoco.MjModel.from_xml_path(str(root / plan.config.env.scene.model_file))
        with np.load(root / "src/unilab/assets" / params.motion_file) as motion_file:
            reference = motion_file["body_quat_w"][:, 1]
            turn = _angles(reference, _rotation(reference[0]))[-1]
        if not 330 <= abs(turn) <= 390:
            raise ValueError("Reference must contain one full sagittal flip")
        direction = float(np.sign(turn))
        folder = output / source
        if checkpoint_series:
            folder /= f"model_{index:05d}"
        folder.mkdir(parents=True)
        _write(folder / "config.json", OmegaConf.to_container(plan.config, resolve=True))
        override = BackendAdapter(
            plan.config, root_dir=root, algo_name="ppo"
        ).build_task_env_cfg_override()
        override["commands"]["motion"]["_target_"] = (
            "unilab.visualization.flip_trials.TrialMotionCfg"
        )
        _write(
            folder / "evaluation_overrides.json",
            dict(
                motion_target=override["commands"]["motion"]["_target_"],
                termination=PROTOCOL["native_done"],
                reference_end=PROTOCOL["reference_end"],
            ),
        )
        frozen = {key: tensor.clone() for key, tensor in plan.actor.state_dict().items()}
        for seed in protocol["seeds"]:
            trial = folder / f"trial_{seed:02d}"
            trial.mkdir()
            rows, actions = [], []
            with closing(registry_env_factory("G1FlipTracking", "mujoco")(1, override)) as env:
                continuation = FailureTailTerminations(env, PROTOCOL["horizon_steps"])
                env.termination_manager = continuation
                env.set_autoreset(False)
                obs, _ = env.reset(seed=seed)
                motion = env.command_manager.get_term("motion")
                if (
                    env.step_dt != 0.02
                    or int(motion.time_steps[0]) != 0
                    or int(motion.sampler.current_clip_end_frames[0]) != PROTOCOL["motion_steps"]
                ):
                    raise ValueError("Unexpected initial phase or control timestep")
                states = [env.get_physics_state_snapshot()[0].copy()]
                with torch.inference_mode():
                    for step in range(PROTOCOL["horizon_steps"]):
                        frame = int(motion.time_steps[0])
                        action = (
                            plan.actor(
                                TensorDict({"actor": torch.as_tensor(obs["obs"])}, batch_size=[1])
                            )
                            .cpu()
                            .numpy()
                        )
                        if action.shape != (1, 29) or not np.isfinite(action).all():
                            raise ValueError("Invalid policy action")
                        state = env.step(action)
                        if not np.isfinite(state.reward).all() or not all(
                            np.isfinite(v).all() for v in state.obs.values()
                        ):
                            raise ValueError("Non-finite native transition")
                        states.append(env.get_physics_state_snapshot()[0].copy())
                        actions.append(action[0].copy())
                        row = dict(
                            step=step,
                            reference_frame=frame,
                            next_reference_frame=int(motion.time_steps[0]),
                            terminated=continuation.native_terminated,
                            truncated=continuation.native_truncated,
                            reward=float(state.reward[0]),
                            termination_terms=[
                                name
                                for name in env.termination_manager.active_terms
                                if env.termination_manager.get_term(name)[0]
                            ],
                        )
                        rows.append(row)
                        if state.terminated[0] or state.truncated[0]:
                            raise ValueError("Evaluation ended before five-second observation")
                        if row["next_reference_frame"] != min(step + 1, PROTOCOL["motion_steps"]):
                            raise ValueError("Unexpected reference phase or physical clip reset")
                        obs = state.obs
            if any(
                not torch.equal(value, plan.actor.state_dict()[name])
                for name, value in frozen.items()
            ):
                raise ValueError("Trial changed actor/normalizer state")
            array = np.stack(states)
            trace = inspect_states(model, array)
            result = dict(
                source=source,
                seed=seed,
                checkpoint=str(checkpoint),
                checkpoint_index=index,
                completed_updates=index + 1,
                total_transitions=plan.metadata["total_transitions"],
                checkpoint_sha256=plan.checkpoint_sha256,
                reference_rotation_degrees=float(turn),
                actor_normalizer_unchanged=True,
                initial_state_sha256=sha256(array[0].tobytes()).hexdigest(),
                trajectory_sha256=sha256(array.tobytes()).hexdigest(),
                **classify(trace, rows, direction),
            )
            np.savez_compressed(
                trial / "trajectory.npz", states=array, actions=np.stack(actions), **trace
            )
            _write(trial / "telemetry.json", rows)
            result["artifact_sha256"] = {
                name: sha256((trial / name).read_bytes()).hexdigest()
                for name in ("trajectory.npz", "telemetry.json")
            }
            _write(trial / "result.json", result)
            results.append(result)
            print(f"{source} model_{index} seed={seed} success={result['success']}", flush=True)
        if sha256(checkpoint.read_bytes()).hexdigest() != plan.checkpoint_sha256:
            raise ValueError("Checkpoint changed during trials")
    summary = {
        source: dict(
            successes=sum(r["success"] for r in results if r["source"] == source),
            trials=sum(r["source"] == source for r in results),
            unique_trajectories=len(
                {r["trajectory_sha256"] for r in results if r["source"] == source}
            ),
            unique_initial_states=len(
                {r["initial_state_sha256"] for r in results if r["source"] == source}
            ),
        )
        for source in sources
    }
    _write(output / "summary.json", dict(protocol=protocol, policies=summary, trials=results))
    with (output / "trials.csv").open("w", newline="") as stream:
        names = [k for k, v in results[0].items() if not isinstance(v, dict)]
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    if checkpoint_series:
        write_checkpoint_table(output, results)
    return summary
