"""Cold-path G1 observation contracts and terminal-transition velocity metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from uni_rl.utils.final_observation import resolve_terminal_observation_contract

_TERM_FUNCTIONS = {
    "base_ang_vel": "unilab.envs.mdp.builtin_sensor",
    "projected_gravity": "unilab.envs.mdp.projected_gravity_from_sensor",
    "joint_pos": "unilab.envs.mdp.joint_pos_rel",
    "joint_vel": "unilab.envs.mdp.joint_vel_rel",
    "actions": "unilab.envs.mdp.last_action",
    "command": "unilab.envs.mdp.generated_commands",
    "gait_phase": "unilab.tasks.locomotion.g1.manager_terms.G1GaitPhase",
    "base_lin_vel": "unilab.envs.mdp.builtin_sensor",
}
_TERM_PARAMS = {
    "base_ang_vel": {"sensor_name": "torso_gyro"},
    "projected_gravity": {"sensor_name": "torso_upvector"},
    "joint_pos": {},
    "joint_vel": {},
    "actions": {"action_name": "joint_pos"},
    "command": {"command_name": "twist"},
    "base_lin_vel": {"sensor_name": "pelvis_local_linvel"},
}
_METRIC_TERMS = ("base_lin_vel", "base_ang_vel", "command")


def g1_metric_provenance() -> dict[str, Any]:
    """Describe the distinct sensor-local frames retained by the G1 critic."""
    linear = {
        "observation_term": "base_lin_vel",
        "sensor": "pelvis_local_linvel",
        "sensor_site": "imu_in_pelvis",
        "body": "pelvis",
        "frame": "imu_in_pelvis local frame",
    }
    return {
        "components": {
            "vx_mae": {
                **linear,
                "axis": "x",
                "reference": "twist linear-x command at action selection",
            },
            "abs_vy_mean": {**linear, "axis": "y"},
            "abs_wz_mean": {
                "observation_term": "base_ang_vel",
                "sensor": "torso_gyro",
                "sensor_site": "imu_in_torso",
                "body": "torso_link",
                "frame": "imu_in_torso local frame",
                "axis": "z",
                "is_pelvis_angular_velocity": False,
            },
        },
        "limitations": [
            "Angular velocity is torso-gyro-local, not pelvis/root angular velocity: waist joints allow relative motion.",
            "Linear velocity is measured at the pelvis IMU site, not necessarily the pelvis origin or center of mass.",
            "The unchanged 98/101 observation contract has no terminal pelvis gyro or root-state snapshot; post-reset scene reads cannot recover it.",
        ],
    }


@dataclass(frozen=True)
class G1EvaluationSpec:
    """Policy dimensions derived from the explicit, supported G1 owner layout."""

    obs_groups_spec: dict[str, int]
    action_dim: int


def g1_evaluation_spec(cfg: DictConfig) -> G1EvaluationSpec:
    """Resolve dimensions without constructing an environment or parsing assets.

    Unknown functions, filtered joint selections and observation histories fail
    closed instead of guessing dimensions before the checkpoint preflight.
    """
    if OmegaConf.select(cfg, "training.task_name") != "G1WalkFlat":
        raise ValueError("PPO metrics evaluation currently supports only G1WalkFlat")
    joints = list(cfg.env.scene.entities.robot.joint_names)
    actuators = list(cfg.env.scene.entities.robot.actuator_names)
    if len(joints) != 29 or len(set(joints)) != 29 or actuators != joints:
        raise ValueError("G1 metrics require the explicit 29-joint/actuator owner declaration")
    widths = dict.fromkeys(_TERM_FUNCTIONS, 3)
    widths.update(
        joint_pos=len(joints), joint_vel=len(joints), actions=len(actuators), gait_phase=2
    )
    dimensions = {}
    groups = {"obs": cfg.env.policy_observation_group, "critic": cfg.env.critic_observation_group}
    for output_name, group_name in groups.items():
        group = cfg.env.observations[group_name]
        if (
            not group.get("concatenate_terms", True)
            or group.get("concatenate_dim", -1) != -1
            or group.get("history_length") not in (None, 0)
        ):
            raise ValueError("G1 metrics require concatenated, history-free observation groups")
        terms = {name: term for name, term in group.terms.items() if term is not None}
        expected = set(_TERM_FUNCTIONS) - ({"base_lin_vel"} if output_name == "obs" else set())
        if set(terms) != expected:
            raise ValueError(f"Unsupported G1 {group_name} observation terms: {sorted(terms)}")
        for name, term in terms.items():
            if term.func != _TERM_FUNCTIONS[name]:
                raise ValueError(f"Cannot infer dimensions of G1 observation term {name!r}")
            params = OmegaConf.to_container(term.get("params", OmegaConf.create({})), resolve=True)
            if name in _TERM_PARAMS and params != _TERM_PARAMS[name]:
                raise ValueError(f"Unsupported G1 observation parameters for {name!r}: {params}")
            if term.get("history_length", 0) != 0:
                raise ValueError("G1 metrics require history-free observation terms")
        if output_name == "critic":
            if group.get("enable_corruption", False):
                raise ValueError("G1 metrics require an uncorrupted raw critic group")
            for name in _METRIC_TERMS:
                term = terms[name]
                scale = term.get("scale")
                if (
                    term.get("clip") is not None
                    or (scale is not None and not np.all(np.asarray(scale) == 1))
                    or term.get("delay_min_lag", 0) != 0
                    or term.get("delay_max_lag", 0) != 0
                    or term.get("delay_hold_prob", 0.0) != 0.0
                    or term.get("delay_update_period", 0) != 0
                ):
                    raise ValueError(f"G1 metrics require raw, current critic term {name!r}")
        dimensions[output_name] = sum(widths[name] for name in terms)
    action = cfg.env.actions.joint_pos
    if (
        set(cfg.env.actions) != {"joint_pos"}
        or action.get("_target_") != "unilab.envs.mdp.JointPositionActionCfg"
        or action.get("entity_name") != "robot"
        or list(action.actuator_names) != [".*"]
    ):
        raise ValueError("G1 metrics require the owner joint-position action declaration")
    if cfg.env.get("is_finite_horizon", False):
        raise ValueError("G1 metrics require separate termination and timeout flags")
    return G1EvaluationSpec(dimensions, len(actuators))


def apply_g1_evaluation_overrides(cfg: DictConfig) -> dict[str, Any]:
    """Disable actor noise and reset mass/PD DR, preserving initial-state sampling.

    Call only on a copy after strict validation of the original training profile.
    """
    overrides: dict[str, Any] = {
        f"env.observations.{cfg.env.policy_observation_group}.enable_corruption": False,
    }
    for name in ("base_mass", "pd_gains"):
        path = f"env.events.{name}"
        event = OmegaConf.select(cfg, path)
        if event is not None:
            if event.get("mode") != "reset":
                raise ValueError(f"Expected reset DR at {path}; refusing a broader override")
            overrides[path] = None
    for path, value in overrides.items():
        OmegaConf.update(cfg, path, value, force_add=True)
    return overrides


class G1TransitionMetrics:
    """Read named raw critic terms using the public observation-manager layout.

    Metrics use the post-physics, pre-autoreset state. The vx reference is copied
    before action selection so command resampling at a timeout cannot change the
    target retrospectively. No backend state is read after an automatic reset.
    Linear components use the pelvis velocimeter; angular z uses the torso gyro,
    which is not a pelvis/root angular velocity when the waist joints move.
    """

    def __init__(self, env: Any):
        manager = env.observation_manager
        group = env.cfg.critic_observation_group
        names = manager.active_terms[group]
        shapes = manager.group_obs_term_dim[group]
        if not manager.group_obs_concatenate[group] or len(names) != len(shapes):
            raise ValueError("Invalid public G1 critic observation layout")
        self.slices: dict[str, slice] = {}
        offset = 0
        for name, shape in zip(names, shapes, strict=True):
            width = int(np.prod(shape))
            if name in _METRIC_TERMS:
                if tuple(shape) != (3,):
                    raise ValueError(f"G1 metric term {name!r} must have shape (3,), got {shape}")
                self.slices[name] = slice(offset, offset + width)
            offset += width
        if set(self.slices) != set(_METRIC_TERMS) or offset != env.obs_groups_spec["critic"]:
            raise ValueError("Public G1 critic layout does not match the metric contract")
        self.shape = (env.num_envs, offset)

    def command_vx(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(obs["critic"])[:, self.slices["command"]][:, 0].copy()

    def measure(
        self, state: Any, command_vx: np.ndarray, active: np.ndarray
    ) -> dict[str, np.ndarray]:
        critic = np.asarray(state.obs["critic"])
        if critic.shape != self.shape:
            raise ValueError(f"Unexpected G1 critic shape: {critic.shape}, expected {self.shape}")
        done = np.asarray(state.terminated) | np.asarray(state.truncated)
        terminal = active & done
        values = critic.copy()
        if np.any(terminal):
            contract = resolve_terminal_observation_contract(
                self.shape[0], state.final_observation, done, state.info, state.truncated
            )
            final = contract.terminal_critic
            if final is None or np.asarray(final).shape != self.shape:
                raise ValueError(
                    "G1 metrics require full-batch pre-reset final_observation['critic']"
                )
            values[terminal] = np.asarray(final)[terminal]
        linear = values[:, self.slices["base_lin_vel"]]
        angular = values[:, self.slices["base_ang_vel"]]
        metrics = {
            "vx_mae": np.abs(linear[:, 0] - command_vx),
            "abs_vy_mean": np.abs(linear[:, 1]),
            "abs_wz_mean": np.abs(angular[:, 2]),
        }
        if any(not np.all(np.isfinite(value[active])) for value in metrics.values()):
            raise ValueError("Non-finite G1 transition metrics; evaluation is incomplete")
        return metrics
