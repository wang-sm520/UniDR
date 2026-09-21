"""Independent native flip probes on resident training pools, without extra SDKs."""

from __future__ import annotations

import random
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from typing import Any, cast

import numpy as np
from uni_rl.env_contract import EnvFactory, EnvProtocol

from unilab.base.env_factory import make_registry_env
from unilab.base.np_env import NpEnvState
from unilab.envs import ManagerBasedRlEnv
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand


class SourceProbeEnv:
    """Expose a fixed 224-control-step opportunity through the existing env service.

    Normal training is passed through verbatim. During a probe, manual autoreset
    lets us measure terminal physical states before resetting failed rows. The
    runtime counts only the first episode of each row and owns E/S/R aggregation.
    There is no policy, optimization, metric normalization, or scheduling here.
    """

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self.env = env
        self.task_cfg = cast(ManagerBasedRlEnvCfg, env.cfg)
        self.motion = cast(MotionCommand, env.command_manager.get_term("motion"))
        params = self.motion.cfg.params
        ranges = [
            *params.pose_range.values(),
            *params.velocity_range.values(),
            params.joint_position_range,
            params.joint_default_position_range,
        ]
        if (
            params.sampling_mode != "start"
            or self.motion.motion.num_clips != 1
            or self.motion.motion.num_frames != 225
            or self.motion.motion.fps != 50
            or not np.isclose(env.cfg.ctrl_dt, 0.02)
            or not self.task_cfg.auto_reset
            or any(not np.all(np.asarray(value) == 0) for value in ranges)
            or any(value is not None for value in self.task_cfg.events.values())
            or any(value is not None for value in self.task_cfg.curriculum.values())
            or any(
                group is not None and group.enable_corruption
                for group in self.task_cfg.observations.values()
            )
        ):
            raise ValueError("Probe requires the nominal start-phase 225-frame flip without DR")
        self._probe = False
        self._steps = 0
        self._saved: dict[str, Any] = {}
        self._probe_state: NpEnvState | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def state(self) -> NpEnvState | None:
        return self._probe_state if self._probe else self.env.state

    def reset_probe(self, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset all resident rows to the common nominal reference initial state."""
        if self._probe or type(seed) is not int or seed < 0:
            raise ValueError("Probe reset requires an inactive probe and nonnegative integer seed")
        self._saved = {
            "rng": deepcopy(self.env.rng.bit_generator.state),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "training": self.env.export_training_state(),
            "seed": self.task_cfg.seed,
        }
        self._probe = True
        self._steps = 0
        self.env.set_autoreset(False)
        try:
            obs, info = self.env.reset(seed=seed)
            self._probe_state = self.env.state
            if np.any(self.motion.time_steps != 0):
                raise ValueError("Probe reset did not select frame zero")
            self._body_error()  # Validate shape/finiteness before the reset tolerance.
            reset_error = np.linalg.norm(
                self.motion.body_pos_w - self.motion.robot_body_pos_w, axis=-1
            )
            if np.max(reset_error) > 0.005:
                raise ValueError("Probe reset does not match reference body positions")
            return obs, info
        except BaseException:
            self.finish_probe()
            raise

    def _body_error(self) -> np.ndarray:
        # Both arrays use scene origins; their difference removes pool placement.
        error = np.mean(
            np.linalg.norm(self.motion.body_pos_w - self.motion.robot_body_pos_w, axis=-1),
            axis=-1,
        ).astype(np.float32)
        if error.shape != (self.env.num_envs,) or not np.isfinite(error).all():
            raise ValueError("Probe body tracking error is non-finite or has invalid shape")
        return error

    def step(self, actions: np.ndarray) -> NpEnvState:
        if not self._probe:
            return self.env.step(actions)
        if self._steps >= 224:
            raise RuntimeError("Probe cannot advance beyond the original clip opportunity")
        state = self.env.step(actions)
        error = self._body_error()
        reward = state.reward.copy()
        terminated, truncated = state.terminated.copy(), state.truncated.copy()
        if not np.isfinite(reward).all() or any(
            not np.isfinite(value).all() for value in state.obs.values()
        ):
            raise ValueError("Probe transition contains non-finite data")
        done = terminated | truncated
        final = {key: value.copy() for key, value in state.obs.items()} if done.any() else None
        info = deepcopy(state.info)
        info["probe"] = {"error": error}
        if done.any():
            # Capture terminal states before public reset mutates the same buffers.
            self.env.reset(np.flatnonzero(done).astype(np.int32))
            info["final_observation"], info["_final_observation"] = final, done.copy()
        self._steps += 1
        self._probe_state = NpEnvState(
            {key: value.copy() for key, value in state.obs.items()},
            reward,
            terminated,
            truncated,
            info,
            final,
        )
        return self._probe_state

    def finish_probe(self) -> None:
        """Restore training RNG/counters; caller must reset all rows before collecting."""
        if not self._probe:
            raise RuntimeError("No probe is active")
        self.env.set_autoreset(True)
        self.env.rng.bit_generator.state = self._saved["rng"]
        self.task_cfg.seed = self._saved["seed"]
        np.random.set_state(self._saved["numpy"])
        random.setstate(self._saved["python"])
        self.env.import_training_state(self._saved["training"])
        self._saved.clear()
        self._probe_state = None
        self._probe = False


def _make_probe_env(
    backend: str, num_envs: int, env_cfg_override: Mapping[str, Any] | None = None
) -> EnvProtocol:
    env = cast(
        ManagerBasedRlEnv, make_registry_env("G1FlipTracking", backend, num_envs, env_cfg_override)
    )
    try:
        return cast(EnvProtocol, SourceProbeEnv(env))
    except BaseException:
        env.close()
        raise


def probe_env_factory(backend: str) -> EnvFactory:
    """Picklable task-owned wrapper; MuJoCo is deliberately excluded from online probes."""
    if backend not in {"isaacsim", "isaacgym", "genesis", "motrix"}:
        raise ValueError("Probe factory requires one of the four training sources")
    return partial(_make_probe_env, backend)
