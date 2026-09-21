"""Synchronous, in-memory PPO windows; raw trajectories survive batch preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import cast

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from uni_rl.algos.rsl_rl_ppo import FinalObservationAwarePPO
from uni_rl.env_contract import EnvProtocol


@dataclass(frozen=True)
class WindowSpec:
    window_id: int
    policy_version: int
    normalizer_version: int
    quotas: Mapping[str, int]  # Transitions, independently of resident pool capacity.
    transitions: int

    @property
    def stamp(self) -> tuple[int, int, int]:
        return self.window_id, self.policy_version, self.normalizer_version


@dataclass(frozen=True)
class SourceRollout:
    """One contiguous vector segment. Episode IDs are local to each segment/env."""

    source_id: str
    segment_id: int
    stamp: tuple[int, int, int]
    raw: RolloutStorage
    env_ids: torch.Tensor
    episode_ids: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_observations: TensorDict | None  # Dense [T,N]; only done rows are meaningful.
    last_observations: TensorDict
    bootstrap_values: torch.Tensor  # V(next pre-reset), zero for true terminations.


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _check_ppo(ppo: FinalObservationAwarePPO) -> None:
    _require(not (ppo.actor.is_recurrent or ppo.critic.is_recurrent), "feedforward PPO only")
    _require(not (ppo.rnd or ppo.symmetry or ppo.is_multi_gpu), "plain single-learner PPO only")
    _require(not ppo.enable_compile, "compiled PPO is outside this window contract")
    _require(not getattr(ppo, "use_mixed_precision", False), "float32 PPO only")
    _require(not ppo.normalize_advantage_per_mini_batch, "global advantage normalization required")
    _require(ppo.num_learning_epochs == 5 and ppo.num_mini_batches == 4, "PPO budget must be 5x4")
    _require(ppo.schedule == "fixed", "fixed PPO schedule required")
    for model in (ppo.actor, ppo.critic):
        _require(
            not model.training and not model.obs_normalizer.training,
            "policy/normalizer must be frozen in eval mode",
        )


def _tensor(data, shape, dtype, device) -> torch.Tensor:
    value = torch.as_tensor(data, device=device)
    _require(value.shape == shape and value.dtype == dtype, "invalid tensor shape or dtype")
    _require(bool(torch.isfinite(value).all()), "non-finite rollout")
    return value.clone()


def _obs(data: Mapping, num_envs: int, ppo: FinalObservationAwarePPO) -> TensorDict:
    schema = ppo.storage.observations
    _require(set(data) == set(schema.keys()), "observation groups mismatch")
    return TensorDict(
        {
            key: _tensor(value, (num_envs, *schema[key].shape[2:]), torch.float32, ppo.device)
            for key, value in data.items()
        },
        batch_size=[num_envs],
        device=ppo.device,
    )


@torch.no_grad()
def collect_source(
    ppo: FinalObservationAwarePPO,
    env: EnvProtocol,
    spec: WindowSpec,
    source_id: str,
    steps: int,
    *,
    segment_id: int = 0,
) -> SourceRollout:
    """Collect a full-pool segment. The caller owns and closes the injected env.

    No optimizer or normalizer update runs here. Version stamps are supplied by
    the caller; snapshot publication and interprocess validation are not C1.
    """
    _check_ppo(ppo)
    _require(steps > 0 and env.num_envs > 0, "positive segment dimensions required")
    state = env.state if env.state is not None else env.init_state()
    obs = _obs(state.obs, env.num_envs, ppo)
    raw = RolloutStorage("rl", env.num_envs, steps, obs, env.action_space.shape, ppo.device)
    flags = torch.zeros(steps, env.num_envs, dtype=torch.bool, device=ppo.device)
    terminated, truncated = flags.clone(), flags.clone()
    episodes = torch.zeros_like(flags, dtype=torch.int64)
    episode = torch.zeros(env.num_envs, dtype=torch.int64, device=ppo.device)
    final = raw.observations.clone()
    bootstrap = torch.zeros_like(raw.values)
    for step in range(steps):
        transition = RolloutStorage.Transition()
        transition.observations = obs.clone()
        actions = ppo.actor(obs, stochastic_output=True).clone()
        transition.actions = actions
        transition.values = ppo.critic(obs).clone()
        transition.actions_log_prob = ppo.actor.get_output_log_prob(actions).clone()
        transition.distribution_params = tuple(
            p.clone() for p in ppo.actor.output_distribution_params
        )
        state = env.step(actions.cpu().numpy().copy())
        obs = _obs(state.obs, env.num_envs, ppo)
        terminated[step] = _tensor(state.terminated, (env.num_envs,), torch.bool, ppo.device)
        truncated[step] = _tensor(state.truncated, (env.num_envs,), torch.bool, ppo.device)
        done = terminated[step] | truncated[step]
        next_obs = obs.clone()
        if done.any():
            if state.final_observation is None:
                raise ValueError("done transition missing final observation")
            terminal_obs = _obs(state.final_observation, env.num_envs, ppo)
            final[step].copy_(terminal_obs)
            next_obs[done] = terminal_obs[done]
        bootstrap[step] = ppo.critic(next_obs)
        bootstrap[step, terminated[step]] = 0
        episodes[step] = episode
        episode = episode + done.long()
        transition.rewards = _tensor(state.reward, (env.num_envs,), torch.float32, ppo.device)
        transition.dones = done
        raw.add_transition(transition)
    return SourceRollout(
        source_id=source_id,
        segment_id=segment_id,
        stamp=spec.stamp,
        raw=raw,
        env_ids=torch.arange(env.num_envs, device=ppo.device),
        episode_ids=episodes,
        terminated=terminated,
        truncated=truncated,
        final_observations=final,
        last_observations=obs,
        bootstrap_values=bootstrap,
    )


def _validate(source: SourceRollout, spec: WindowSpec, ppo: FinalObservationAwarePPO) -> None:
    st = source.raw
    t, n = st.num_transitions_per_env, st.num_envs
    _require(
        source.source_id in spec.quotas and source.segment_id >= 0,
        "unknown source or invalid segment",
    )
    _require(source.stamp == spec.stamp, "window/policy/normalizer version mismatch")
    _require(t > 0 and n > 0 and st.step == t, "incomplete source rollout")
    _require(
        source.env_ids.shape == (n,) and source.env_ids.dtype == torch.int64, "invalid env IDs"
    )
    _require(
        bool((source.env_ids >= 0).all()) and source.env_ids.unique().numel() == n,
        "duplicate/invalid env IDs",
    )
    _require(
        source.episode_ids.shape == (t, n) and source.episode_ids.dtype == torch.int64,
        "invalid episode IDs",
    )
    _require(bool((source.episode_ids >= 0).all()), "invalid episode IDs")
    for flag in (source.terminated, source.truncated):
        _require(flag.shape == (t, n) and flag.dtype == torch.bool, "invalid termination flags")
    done = source.terminated | source.truncated
    _require(torch.equal(st.dones, done.unsqueeze(-1).to(st.dones.dtype)), "done flags disagree")
    _require(
        torch.equal(source.episode_ids[1:] - source.episode_ids[:-1], done[:-1].long()),
        "episode continuity mismatch",
    )
    expected = {key: value.shape[2:] for key, value in ppo.storage.observations.items()}
    groups = [(st.observations, (t, n)), (source.last_observations, (n,))]
    if source.final_observations is not None:
        groups.append((source.final_observations, (t, n)))
    _require(
        not bool(done.any()) or source.final_observations is not None, "missing final observation"
    )
    tensors = []
    for observations, batch_shape in groups:
        _require(set(observations.keys()) == set(expected), "observation groups mismatch")
        for key, shape in expected.items():
            value = observations[key]
            _require(tuple(value.shape) == (*batch_shape, *shape), "observation shape mismatch")
            tensors.append(value)
    for value in (st.rewards, st.values, st.actions_log_prob, source.bootstrap_values):
        _require(value.shape == (t, n, 1), "scalar field shape mismatch")
        tensors.append(value)
    _require(st.actions.shape == (t, n, *ppo.storage.actions_shape), "action shape mismatch")
    _require(
        st.distribution_params is not None and len(st.distribution_params) == 2,
        "Gaussian distribution parameters required",
    )
    assert st.distribution_params is not None
    for parameter in st.distribution_params:
        _require(parameter.shape == st.actions.shape, "distribution shape mismatch")
    _require(bool((st.distribution_params[1] > 0).all()), "distribution std must be positive")
    tensors.extend((st.actions, *st.distribution_params))
    for value in tensors:
        _require(value.dtype == torch.float32, "float32 rollout required")
        _require(
            value.device == st.values.device == torch.device(ppo.device), "rollout device mismatch"
        )
        _require(bool(torch.isfinite(value).all()), "non-finite rollout")
    _require(
        bool((source.bootstrap_values[source.terminated] == 0).all()),
        "true termination must not bootstrap",
    )


@torch.no_grad()
def prepare_ppo_window(
    ppo: FinalObservationAwarePPO, sources: Sequence[SourceRollout], spec: WindowSpec
) -> RolloutStorage:
    """Validate the complete budget, run existing GAE per segment, then merge.

    The returned [1,B] storage is an optimization view, never a trajectory.
    The caller retains sources and assigns this storage to the single learner.
    """
    _check_ppo(ppo)
    return _prepare_ppo_window(ppo, sources, spec)


@torch.no_grad()
def _prepare_ppo_window(
    ppo: FinalObservationAwarePPO, sources: Sequence[SourceRollout], spec: WindowSpec
) -> RolloutStorage:
    """Shared validated GAE/batch preparation after the caller checks its protocol."""
    _require(all(type(v) is int and v >= 0 for v in spec.stamp), "invalid window versions")
    _require(
        bool(spec.quotas) and all(type(v) is int and v > 0 for v in spec.quotas.values()),
        "invalid quotas",
    )
    _require(
        spec.transitions == sum(spec.quotas.values()) and spec.transitions % 4 == 0,
        "budget must equal quotas and be divisible by 4",
    )
    counts = dict.fromkeys(spec.quotas, 0)
    seen: set[tuple[str, int]] = set()
    for source in sources:
        _validate(source, spec, ppo)
        key = source.source_id, source.segment_id
        _require(key not in seen, "duplicate source segment")
        seen.add(key)
        counts[source.source_id] += source.raw.num_envs * source.raw.num_transitions_per_env
    _require(counts == dict(spec.quotas), "source budget mismatch")
    original = ppo.storage
    prepared = []
    try:
        ppo.normalize_advantage_per_mini_batch = True  # Defer normalization until after merging.
        for source in sources:
            work = copy(source.raw)
            timeout = (source.truncated & ~source.terminated).unsqueeze(-1)
            work.rewards = source.raw.rewards + ppo.gamma * source.bootstrap_values * timeout
            work.returns = torch.empty_like(source.raw.values)
            ppo.storage = work
            ppo.compute_returns(source.last_observations)
            prepared.append(work)
    finally:
        ppo.storage = original
        ppo.normalize_advantage_per_mini_batch = False
    observations = cast(TensorDict, torch.cat([st.observations.flatten(0, 1) for st in prepared]))
    merged = RolloutStorage(
        "rl", spec.transitions, 1, observations, original.actions_shape, ppo.device
    )
    merged.observations[0].copy_(observations)
    for name in ("actions", "rewards", "dones", "values", "actions_log_prob", "returns"):
        getattr(merged, name)[0].copy_(
            torch.cat([getattr(st, name).flatten(0, 1) for st in prepared])
        )
    merged.distribution_params = tuple(
        torch.cat([st.distribution_params[i].flatten(0, 1) for st in prepared]).unsqueeze(0)
        for i in range(2)
    )
    advantage = merged.returns - merged.values
    merged.advantages.copy_((advantage - advantage.mean()) / (advantage.std() + 1e-8))
    merged.step = 1
    return merged
