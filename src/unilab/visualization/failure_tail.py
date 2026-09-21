"""Playback-only continuation after native failure, without changing training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.managers.termination_manager import TerminationManager
from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand, MotionCommandCfg


class _HoldLastMotion(MotionCommand):
    def _update_command(self, env_ids: np.ndarray | None) -> None:
        if np.any(self.time_steps > self.sampler.current_clip_end_frames):
            raise ValueError("Diagnostic reference exceeded the original clip")
        if (
            env_ids is None
            and isinstance(self._env.termination_manager, FailureTailTerminations)
            and self._env.termination_manager.first_done is not None
            and np.all(self.time_steps == self.sampler.current_clip_end_frames)
        ):
            # The existing row-refresh path preserves the final reference frame
            # after native failure, without writing a new robot state. Before
            # failure retain native clip loops so later failures stay reachable.
            env_ids = np.arange(self.num_envs, dtype=np.int32)
        super()._update_command(env_ids)


@dataclass(kw_only=True)
class HoldLastMotionCfg(MotionCommandCfg):
    """Preserve native loops; hold the last frame only during a diagnostic tail."""

    def build(self, env: Any) -> MotionCommand:
        if env.num_envs != 1:
            raise ValueError("Failure-tail playback requires exactly one environment")
        return _HoldLastMotion(self, env)


class FailureTailTerminations(TerminationManager):
    """Observe native conditions but delay their reset by a fixed number of steps."""

    def __init__(self, env: Any, tail_steps: int):
        if env.num_envs != 1 or type(tail_steps) is not int or tail_steps < 1:
            raise ValueError("Failure tail requires one environment and positive steps")
        # Use unbound term configs; a live manager contains callables bound to
        # the simulator, which must never be deep-copied.
        super().__init__(env.cfg.terminations, env)
        self.tail_steps = tail_steps
        self.tick = 0
        self.first_done: int | None = None
        self.native_terminated = False
        self.native_truncated = False
        self.seen_termination = False
        self.steps_after_failure: int | None = None

    def compute(self) -> np.ndarray:
        super().compute()
        self.native_terminated = bool(self.terminated[0])
        self.native_truncated = bool(self.time_outs[0])
        self.seen_termination |= self.native_terminated
        if self.first_done is None and (self.native_terminated or self.native_truncated):
            self.first_done = self.tick
        self.steps_after_failure = None if self.first_done is None else self.tick - self.first_done
        expired = (
            self.steps_after_failure is not None and self.steps_after_failure >= self.tail_steps
        )
        self.terminated.fill(expired and self.seen_termination)
        self.time_outs.fill(expired and not self.seen_termination)
        self.tick += 1
        return self.dones

    def reset(self, env_ids: np.ndarray | slice | None = None) -> dict[str, int]:
        extras = super().reset(env_ids)
        self.tick = 0
        self.first_done = None
        self.native_terminated = self.native_truncated = self.seen_termination = False
        self.steps_after_failure = None
        return extras
