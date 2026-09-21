"""Whole-pool quota collection, with one central policy and source-local GAE."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from uni_rl.algos.multi_source_ppo import SourceRollout, WindowSpec, _prepare_ppo_window, _require
from uni_rl.algos.synchronous_ppo import (
    CentralCollector,
    CentralWindow,
    _check_central_ppo,
    _revision,
)


def source_observations(env: Any, obs: Mapping) -> TensorDict:
    """Apply the central wrapper's observation aliases to a selected batch."""
    tensors = {key: torch.as_tensor(value, device=env.device) for key, value in obs.items()}
    result = {"actor": tensors["obs"], "policy": tensors["obs"]}
    if env.policy_obs_mode == "flat":
        result["policy"] = torch.cat([v for k, v in tensors.items() if k != "critic"], -1)
    for key, value in tensors.items():
        _require(
            key not in result or torch.equal(result[key], value), "observation alias collision"
        )
        result[key] = value
    return TensorDict(result, [len(tensors["obs"])], device=env.device)


@dataclass(frozen=True)
class QuotaWindow(CentralWindow):
    def prepare(self, ppo):
        _check_central_ppo(ppo)
        _require(self.learner_revision == _revision(ppo), "learner state changed after collection")
        waves = max(source.raw.num_transitions_per_env for source in self.sources)
        _require(
            self.normalizer_versions.dtype == torch.int64
            and torch.equal(
                self.normalizer_versions,
                torch.arange(waves, device=ppo.device) + self.spec.normalizer_version,
            ),
            "nonconsecutive normalizer versions",
        )
        ppo.eval_mode()
        return _prepare_ppo_window(ppo, self.sources, self.spec)


class QuotaCollector(CentralCollector):
    """Sources advance consecutively until their quota is exhausted, then pause."""

    def set_quotas(self, quotas: Mapping[str, int]) -> None:
        _require(tuple(quotas) == tuple(self.source_slices), "quota source order mismatch")
        _require(all(type(t) is int and t > 0 for t in quotas.values()), "invalid step quota")
        _require(sum(quotas.values()) == 96, "whole-pool step budget must remain 96")
        _require(
            len({s.stop - s.start for s in self.source_slices.values()}) == 1,
            "equal resident pool capacities required",
        )
        self.quotas = dict(quotas)
        self.steps = max(quotas.values())

    def _collect(self, stamp: tuple[int, int, int], on_step: Callable | None) -> QuotaWindow:
        ppo, env = self.ppo, self.env
        _check_central_ppo(ppo)
        ppo.train_mode()
        obs = env.get_observations().to(ppo.device).clone()
        params = tuple(ppo.actor.parameters()) + tuple(ppo.critic.parameters())
        revisions = [p._version for p in params]
        sources = {}
        for name, part in self.source_slices.items():
            n, t = part.stop - part.start, self.quotas[name]
            raw = RolloutStorage("rl", n, t, obs[part], ppo.storage.actions_shape, ppo.device)
            flags = torch.zeros((t, n), dtype=torch.bool, device=ppo.device)
            sources[name] = SourceRollout(
                name,
                stamp[0],
                stamp,
                raw,
                torch.arange(n, device=ppo.device),
                flags.to(torch.int64),
                flags.clone(),
                flags.clone(),
                TensorDict({k: torch.zeros_like(v) for k, v in raw.observations.items()}, [t, n]),
                obs[part].clone(),
                torch.zeros_like(raw.values),
            )
        for step in range(self.steps):
            names = [key for key, quota in self.quotas.items() if step < quota]
            ids = torch.cat(
                [
                    torch.arange(
                        self.source_slices[k].start, self.source_slices[k].stop, device=ppo.device
                    )
                    for k in names
                ]
            )
            before = obs[ids].clone()
            _require(
                all(
                    v.dtype == torch.float32 and bool(torch.isfinite(v).all())
                    for v in before.values()
                ),
                "invalid observation",
            )
            actions = ppo.act(before)
            counts = [sources[k].raw.num_envs for k in names]
            pieces = torch.split(actions, counts)
            states = env.env.step_selected(
                {k: a.cpu().numpy().copy() for k, a in zip(names, pieces)}
            )
            after = cast(
                TensorDict, torch.cat([source_observations(env, states[k].obs) for k in names])
            )
            _require(
                set(after.keys()) == set(before.keys())
                and all(
                    after[k].shape == before[k].shape
                    and after[k].dtype == torch.float32
                    and bool(torch.isfinite(after[k]).all())
                    for k in before.keys()
                ),
                "invalid post-step observation",
            )
            for model in (ppo.actor, ppo.critic):
                count = cast(torch.Tensor, model.obs_normalizer.count).clone()
                model.update_normalization(after)
                _require(
                    bool(model.obs_normalizer.count == count + len(ids)),
                    "normalizer updated other than once",
                )
                _require(
                    all(bool(torch.isfinite(v).all()) for v in model.obs_normalizer.buffers()),
                    "non-finite normalizer statistics",
                )
            obs[ids] = after
            offset = 0
            rewards = torch.zeros(env.num_envs, device=ppo.device)
            dones = torch.zeros(env.num_envs, device=ppo.device, dtype=torch.bool)
            for name in names:
                source, state = sources[name], states[name]
                part = self.source_slices[name]
                selected = slice(offset, offset + source.raw.num_envs)
                transition = RolloutStorage.Transition()
                for field in ("observations", "actions", "values", "actions_log_prob"):
                    setattr(transition, field, getattr(ppo.transition, field)[selected].clone())
                assert ppo.transition.distribution_params is not None
                transition.distribution_params = tuple(
                    v[selected].clone() for v in ppo.transition.distribution_params
                )
                transition.rewards = torch.as_tensor(state.reward, device=ppo.device).clone()
                source.terminated[step] = torch.as_tensor(state.terminated, device=ppo.device)
                source.truncated[step] = torch.as_tensor(state.truncated, device=ppo.device)
                done = source.terminated[step] | source.truncated[step]
                transition.dones = done
                source.episode_ids[step] = self.episode_ids[part]
                if step:
                    ongoing = ~(source.terminated[step - 1] | source.truncated[step - 1])
                    assert transition.values is not None
                    source.bootstrap_values[step - 1, ongoing] = transition.values[ongoing]
                if bool(done.any()):
                    _require(state.final_observation is not None, "missing final observation")
                    final = source_observations(env, state.final_observation)
                    assert source.final_observations is not None
                    source.final_observations[step].copy_(final)
                    timeout = source.truncated[step] & ~source.terminated[step]
                    source.bootstrap_values[step, timeout] = ppo.critic(final)[timeout].detach()
                source.raw.add_transition(transition)
                source.last_observations.copy_(obs[part])
                self.episode_ids[part] += done.long()
                rewards[part], dones[part] = transition.rewards, done
                offset = selected.stop
            ppo.transition.clear()
            ppo.actor.reset(dones[ids])
            ppo.critic.reset(dones[ids])
            env.episode_returns += rewards
            env.episode_lengths[ids] += 1
            env.episode_returns[dones] = env.episode_lengths[dones] = 0
            if on_step is not None:
                logs = {
                    f"{name}/{key}": value
                    for name, state in states.items()
                    for key, value in state.info.get("log", {}).items()
                }
                on_step(obs, rewards, dones, {"active_env_ids": ids, "log": logs})
        for source in sources.values():
            ongoing = ~(source.terminated[-1] | source.truncated[-1])
            source.bootstrap_values[-1, ongoing] = ppo.critic(source.last_observations)[
                ongoing
            ].detach()
        _require(revisions == [p._version for p in params], "weights changed during collection")
        quotas = {k: v.raw.num_envs * v.raw.num_transitions_per_env for k, v in sources.items()}
        return QuotaWindow(
            WindowSpec(*stamp, quotas, sum(quotas.values())),
            tuple(sources.values()),
            torch.arange(self.steps, device=ppo.device) + stamp[2],
            self.generation,
            _revision(ppo),
        )
