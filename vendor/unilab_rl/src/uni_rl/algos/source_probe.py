"""Independent frozen-policy probes; no probe observation enters training statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from uni_rl.algos.synchronous_ppo import _revision


def validate_probe_config(config: dict) -> None:
    for key in ("interval", "horizon", "seed"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid probe {key}")
    for key in ("error_cap", "return_min", "return_max"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]):
            raise ValueError(f"invalid probe {key}")
    if config["error_cap"] <= 0 or config["return_min"] >= config["return_max"]:
        raise ValueError("invalid fixed probe normalization scales")


@torch.no_grad()
def run_source_probe(ppo: Any, env: Any, config: dict) -> dict:
    """Evaluate one first-episode opportunity per pool row, padding early failures."""
    try:
        validate_probe_config(config)
        revision = _revision(ppo)
        ppo.eval_mode()
        horizon, cap = config["horizon"], config["error_cap"]
        alive = np.ones(env.num_envs, dtype=bool)
        error_sum = np.zeros(env.num_envs, dtype=np.float64)
        returns = np.zeros(env.num_envs, dtype=np.float64)
        cursor, sizes = 0, []
        for part in env.env.source_slices.values():
            if part.start != cursor or part.stop <= cursor or part.step not in (None, 1):
                raise ValueError("invalid probe source pools")
            sizes.append(part.stop - part.start)
            cursor = part.stop
        if len(sizes) != 4 or len(set(sizes)) != 1 or cursor != env.num_envs:
            raise ValueError("probe requires four equal nonempty source pools")
        raw, _ = env.env.reset_probe(config["seed"])
        obs = env._obs_to_tensordict(raw)
        for _ in range(horizon):
            action = ppo.actor(obs, stochastic_output=False)
            state = env.env.step(action.cpu().numpy().copy())
            for name, part in env.env.source_slices.items():
                probe = state.info["source_logs"][name].get("probe")
                if not isinstance(probe, dict):
                    raise ValueError("source did not supply independent probe metrics")
                error = np.asarray(probe["error"], dtype=np.float64)
                if (
                    error.shape != (part.stop - part.start,)
                    or not np.isfinite(error).all()
                    or (error < 0).any()
                ):
                    raise ValueError("invalid probe tracking error")
                active = alive[part]
                error_sum[part] += np.where(active, np.minimum(error, cap), cap)
                returns[part] += np.where(active, state.reward[part], 0)
            alive &= ~(state.terminated | state.truncated)
            obs = env._obs_to_tensordict(state.obs)
        result = {}
        for name, part in env.env.source_slices.items():
            raw_error, raw_return = (
                float(error_sum[part].mean() / horizon),
                float(returns[part].mean()),
            )
            result[name] = {
                "error": float(np.clip(raw_error / cap, 0, 1)),
                "survival": float(alive[part].mean()),
                "return": float(
                    np.clip(
                        (raw_return - config["return_min"])
                        / (config["return_max"] - config["return_min"]),
                        0,
                        1,
                    )
                ),
                "raw_error_m": raw_error,
                "raw_return": raw_return,
                "opportunities": part.stop - part.start,
                "horizon_steps": horizon,
            }
        if revision != _revision(ppo):
            raise ValueError("probe changed policy or normalizer")
        env.env.finish_probe()
        env.episode_returns.zero_()
        env.episode_lengths.zero_()
        return result
    except BaseException:
        env.close()
        raise
