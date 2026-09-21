"""Same-backend flip diagnostics, independent of learners and checkpoint selection."""

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from unilab.envs import ManagerBasedRlEnv
from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand


def validate_flip_episode(
    env: ManagerBasedRlEnv,
    actions: Callable[[], np.ndarray],
    *,
    seed: int = 1,
) -> dict[str, Any]:
    """Measure the first full-clip opportunity per env, before manual resets.

    Body error is capped at 0.5 m and unobserved steps after failure receive
    that cap. Return is the raw sum up to the first done, with zero padding.
    These are baseline diagnostics, not the future scheduler's E/S/R contract.
    The caller owns and closes this dedicated evaluation environment.
    """
    env.set_autoreset(False)
    env.reset(seed=seed)
    motion = cast(MotionCommand, env.command_manager.get_term("motion"))
    if motion.cfg.params.sampling_mode != "start" or not motion.cfg.params.truncate_on_clip_end:
        raise ValueError("Flip validation requires start sampling and clip-end truncation")
    horizon = int(motion.motion.num_frames)
    if motion.motion.num_clips != 1 or horizon < 2:
        raise ValueError("Flip validation requires one complete motion clip")
    if np.any(motion.time_steps != 0) or not np.isclose(motion.motion.fps * env.step_dt, 1.0):
        raise ValueError(
            "Flip validation requires frame zero and one reference frame per control step"
        )
    reset_error = float(
        np.max(np.linalg.norm(motion.body_pos_w - motion.robot_body_pos_w, axis=-1))
    )
    if not np.isfinite(reset_error) or reset_error > 0.005:
        raise ValueError(f"Reference reset body positions differ by {reset_error:.6f} m")

    active = np.ones(env.num_envs, dtype=bool)
    completed = np.zeros(env.num_envs, dtype=bool)
    lengths = np.zeros(env.num_envs, dtype=int)
    returns = np.zeros(env.num_envs)
    error_sum = np.zeros(env.num_envs)
    cap = 0.5
    padded_error = np.full(env.num_envs, horizon * cap)
    termination_counts: dict[str, int] = {}
    for _ in range(horizon):
        state = env.step(actions())
        if not all(np.isfinite(x).all() for x in (*state.obs.values(), state.reward)):
            raise ValueError("Non-finite flip evaluation transition")
        error = np.mean(
            np.linalg.norm(motion.body_pos_w - motion.robot_body_pos_w, axis=-1), axis=-1
        )
        if not np.isfinite(error).all():
            raise ValueError("Non-finite flip body error")
        lengths[active] += 1
        returns[active] += state.reward[active]
        error_sum[active] += error[active]
        padded_error[active] += np.minimum(error[active], cap) - cap
        done = (state.terminated | state.truncated) & active
        clip_end = motion.time_steps >= motion.sampler.current_clip_end_frames
        completed[done] = state.truncated[done] & ~state.terminated[done] & clip_end[done]
        for name in env.termination_manager.active_terms:
            count = int(np.count_nonzero(env.termination_manager.get_term(name) & done))
            termination_counts[name] = termination_counts.get(name, 0) + count
        active &= ~done
        if not active.any():
            break
        # NpEnv requires done rows to be reset before the next batched step.
        # They remain inactive: later episodes never contribute to this report.
        reset_rows = np.flatnonzero(state.terminated | state.truncated).astype(np.int32)
        if len(reset_rows):
            env.reset(reset_rows)
    return {
        "opportunities": env.num_envs,
        "horizon_steps": horizon,
        "reset_body_error_m": reset_error,
        "clip_completion_rate": float(completed.mean()),
        "mean_steps": float(lengths.mean()),
        "mean_return_until_done": float(returns.mean()),
        "observed_mean_body_error_m": float(error_sum.sum() / lengths.sum()),
        "capped_horizon_body_error_m": float(padded_error.mean() / horizon),
        "failure_padding_body_error_m": cap,
        "termination_counts": termination_counts,
    }
