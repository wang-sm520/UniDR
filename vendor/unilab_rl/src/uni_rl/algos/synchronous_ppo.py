"""Central inference with native per-step normalization and source-local GAE."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass
from typing import Any, cast

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from uni_rl.algos.multi_source_ppo import (
    SourceRollout,
    WindowSpec,
    _prepare_ppo_window,
    _require,
)
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper
from uni_rl.algos.rsl_rl_ppo import FinalObservationAwarePPO


class CentralVecEnv(RslRlVecEnvWrapper):
    """Retain all observation groups and give true termination precedence."""

    def _obs_to_tensordict(self, obs: dict[str, Any], info=None) -> TensorDict:
        result = super()._obs_to_tensordict(obs, info)
        for key, value in obs.items():
            tensor = torch.as_tensor(value, device=self.device)
            if key in result:
                _require(torch.equal(result[key], tensor), "reserved observation group collision")
            else:
                result[key] = tensor
        return result

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        state = self.env.state
        logs = {}
        for source, info in state.info.get("source_logs", {}).items():
            for key, value in info.get("log", {}).items():
                logs[f"{source}/{key}"] = value
        if logs:
            extras["log"] = logs
        return obs, rewards, dones, extras


def _check_central_ppo(ppo: FinalObservationAwarePPO) -> None:
    _require(not (ppo.actor.is_recurrent or ppo.critic.is_recurrent), "feedforward PPO only")
    _require(not (ppo.rnd or ppo.symmetry or ppo.is_multi_gpu), "one plain PPO learner required")
    _require(not ppo.enable_compile, "central PPO requires the native uncompiled update")
    _require(not getattr(ppo, "use_mixed_precision", False), "float32 PPO required")
    _require(not ppo.normalize_advantage_per_mini_batch, "global advantage normalization required")
    _require(ppo.num_learning_epochs == 5 and ppo.num_mini_batches == 4, "PPO budget must be 5x4")
    _require(ppo.schedule in {"adaptive", "fixed"}, "unsupported PPO schedule")
    for model in (ppo.actor, ppo.critic):
        norm = model.obs_normalizer
        _require(
            hasattr(norm, "count") and norm.count.dtype == torch.int64,
            "int64 empirical count required",
        )
        _require(norm.until is None, "normalizer must remain enabled throughout training")


def _revision(ppo: FinalObservationAwarePPO) -> tuple[tuple[int, int], ...]:
    return tuple(
        (id(value), value._version)
        for model in (ppo.actor, ppo.critic)
        for value in (*model.parameters(), *model.buffers())
    )


@dataclass(frozen=True)
class CentralWindow:
    spec: WindowSpec
    sources: tuple[SourceRollout, ...]
    normalizer_versions: torch.Tensor  # [T], pre-act version; bootstrap uses version + 1.
    generation: int = 0
    learner_revision: tuple[tuple[int, int], ...] = ()

    def prepare(self, ppo: FinalObservationAwarePPO) -> RolloutStorage:
        _check_central_ppo(ppo)
        _require(self.learner_revision == _revision(ppo), "learner state changed after collection")
        versions = self.normalizer_versions
        steps = self.sources[0].raw.num_transitions_per_env
        _require(
            versions.dtype == torch.int64
            and torch.equal(
                versions, torch.arange(steps, device=versions.device) + self.spec.normalizer_version
            ),
            "nonconsecutive normalizer versions",
        )
        _require(
            all(source.raw.num_transitions_per_env == steps for source in self.sources),
            "all sources must participate in every control step",
        )
        ppo.eval_mode()  # Explicitly freeze statistics for bootstrap/optimization.
        return _prepare_ppo_window(ppo, self.sources, self.spec)


class CentralCollector:
    """Collect one complete window; any error poisons this collector and closes envs."""

    def __init__(
        self,
        ppo: FinalObservationAwarePPO,
        env: CentralVecEnv,
        source_slices: Mapping[str, slice],
    ) -> None:
        _check_central_ppo(ppo)
        self.ppo, self.env = ppo, env
        self.source_slices = dict(source_slices)
        cursor = 0
        for part in self.source_slices.values():
            _require(
                part.start == cursor and part.stop > cursor and part.step in (None, 1),
                "invalid source slices",
            )
            cursor = part.stop
        _require(
            cursor == env.num_envs and len(source_slices) == 4,
            "four complete source pools required",
        )
        self.steps = ppo.storage.num_transitions_per_env
        self.episode_ids = torch.zeros(env.num_envs, dtype=torch.int64, device=ppo.device)
        self.failed = False
        self.generation = 0
        self.next_stamp: tuple[int, int, int] | None = None

    @torch.no_grad()
    def collect(
        self,
        window_id: int,
        policy_version: int,
        normalizer_version: int,
        on_step: Callable[..., None] | None = None,
    ) -> CentralWindow:
        _require(not self.failed, "failed collector must be recreated from a checkpoint")
        stamp = (window_id, policy_version, normalizer_version)
        _require(all(type(v) is int and v >= 0 for v in stamp), "invalid window versions")
        _require(
            self.next_stamp is None or stamp == self.next_stamp, "stale or skipped window version"
        )
        try:
            window = self._collect(stamp, on_step)
        except BaseException:
            self.failed = True
            self.env.close()
            raise
        self.next_stamp = (window_id + 1, policy_version + 1, normalizer_version + self.steps)
        return window

    def _collect(
        self, stamp: tuple[int, int, int], on_step: Callable[..., None] | None
    ) -> CentralWindow:
        ppo, env = self.ppo, self.env
        _check_central_ppo(ppo)
        ppo.train_mode()
        obs = env.get_observations().to(ppo.device)
        n, t = env.num_envs, self.steps
        storage = RolloutStorage("rl", n, t, obs, ppo.storage.actions_shape, ppo.device)
        ppo.storage = storage
        raw_rewards = torch.empty_like(storage.rewards)
        terminated = torch.empty((t, n), dtype=torch.bool, device=ppo.device)
        truncated = torch.empty_like(terminated)
        episodes = torch.empty((t, n), dtype=torch.int64, device=ppo.device)
        bootstrap = torch.zeros_like(storage.values)
        final = TensorDict(
            {key: torch.zeros_like(value) for key, value in storage.observations.items()}, [t, n]
        )
        params = tuple(ppo.actor.parameters()) + tuple(ppo.critic.parameters())
        weight_versions = [p._version for p in params]
        for step in range(t):
            for key in obs.keys():
                value = cast(torch.Tensor, obs[key])
                _require(
                    value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                    "non-finite or non-float32 observation",
                )
            episodes[step].copy_(self.episode_ids)
            actions = ppo.act(
                obs.clone()
            )  # EnvProtocol permits reuse of numpy observation buffers.
            if step:
                ongoing = ~(terminated[step - 1] | truncated[step - 1])
                assert ppo.transition.values is not None
                bootstrap[step - 1, ongoing] = ppo.transition.values[ongoing]
            obs, rewards, dones, extras = env.step(actions)
            _require(
                set(obs.keys()) == set(storage.observations.keys()), "observation groups mismatch"
            )
            for key, value in obs.items():
                _require(
                    value.shape == storage.observations[key].shape[1:]
                    and value.dtype == torch.float32
                    and bool(torch.isfinite(value).all()),
                    "invalid post-step observation",
                )
            state = env.env.state
            terminated[step].copy_(torch.as_tensor(state.terminated, device=ppo.device))
            truncated[step].copy_(torch.as_tensor(state.truncated, device=ppo.device))
            raw_rewards[step, :, 0].copy_(rewards)
            _require(bool(torch.isfinite(rewards).all()), "non-finite reward")
            if bool(dones.any()):
                _require(state.final_observation is not None, "missing final observation")
                cast(TensorDict, final[step]).copy_(env._obs_to_tensordict(state.final_observation))
            counts = [
                cast(torch.Tensor, model.obs_normalizer.count).clone()
                for model in (ppo.actor, ppo.critic)
            ]
            ppo.process_env_step(obs, rewards, dones, extras)
            for model, count in zip((ppo.actor, ppo.critic), counts):
                _require(
                    bool(model.obs_normalizer.count == count + n),
                    "normalizer updated other than once",
                )
                _require(
                    all(
                        bool(torch.isfinite(value).all())
                        for value in model.obs_normalizer.buffers()
                    ),
                    "non-finite normalizer statistics",
                )
            timeout = truncated[step] & ~terminated[step]
            if bool(timeout.any()):
                bootstrap[step, timeout] = ppo.critic(final[step])[timeout].detach()
            self.episode_ids += dones.long()
            if on_step is not None:
                on_step(obs, rewards, dones, extras)
        tail = ppo.critic(obs).detach()
        ongoing = ~(terminated[-1] | truncated[-1])
        bootstrap[-1, ongoing] = tail[ongoing]
        _require(
            weight_versions == [p._version for p in params], "weights changed during collection"
        )
        quotas = {name: t * (part.stop - part.start) for name, part in self.source_slices.items()}
        spec = WindowSpec(*stamp, quotas, t * n)
        sources = []
        for name, part in self.source_slices.items():
            raw = copy(storage)
            raw.num_envs = part.stop - part.start
            for field in (
                "observations",
                "actions",
                "dones",
                "values",
                "actions_log_prob",
                "returns",
                "advantages",
            ):
                setattr(raw, field, getattr(storage, field)[:, part])
            raw.rewards = raw_rewards[:, part]
            assert storage.distribution_params is not None
            raw.distribution_params = tuple(value[:, part] for value in storage.distribution_params)
            sources.append(
                SourceRollout(
                    name,
                    stamp[0],
                    stamp,
                    raw,
                    torch.arange(raw.num_envs, device=ppo.device),
                    episodes[:, part],
                    terminated[:, part],
                    truncated[:, part],
                    cast(TensorDict, final[:, part]),
                    obs[part].clone(),
                    bootstrap[:, part],
                )
            )
        return CentralWindow(
            spec,
            tuple(sources),
            torch.arange(t, device=ppo.device) + stamp[2],
            self.generation,
            _revision(ppo),
        )
