"""Real shared PPO over whole-pool adaptive quotas and isolated source trajectories."""

import os
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from examples.unidr_fake import FakeSource, make_ppo
from rsl_rl.storage import RolloutStorage

from uni_rl.algos.quota_ppo import QuotaCollector
from uni_rl.algos.synchronous_ppo import CentralCollector, CentralVecEnv
from uni_rl.ipc.synchronous_env import SourceSpec, SynchronousEnv

NAMES = tuple(f"source{i}" for i in range(4))


class AuditSource(FakeSource):
    fault = None

    def step(self, actions):
        state = super().step(actions)
        state.truncated |= state.terminated  # Explicit dual flags on true terminations.
        if self.fault == "missing":
            state.final_observation = None
        elif self.fault == "nan":
            state.reward[0] = np.nan
        elif self.fault == "groups":
            state.obs["unexpected"] = state.obs["obs"]
        return state


def factory(num_envs, override):
    source = AuditSource(override["index"], num_envs)
    source.fault = override.get("fault")
    return source


@pytest.fixture
def make():
    services = []

    def build(steps=(25, 24, 24, 23), fault=None, collector_type=QuotaCollector):
        torch.set_num_threads(1)
        env = SynchronousEnv(
            [
                SourceSpec(name, factory, 2, {"index": i, "fault": fault if i == 0 else None})
                for i, name in enumerate(NAMES)
            ]
        )
        services.append(env)
        wrapped = CentralVecEnv(env)
        ppo = make_ppo(normalize=True)
        ppo.schedule = "adaptive"
        ppo.storage = RolloutStorage("rl", 8, 24, wrapped.get_observations(), [2])
        collector = collector_type(ppo, wrapped, env.source_slices)
        if isinstance(collector, QuotaCollector):
            collector.set_quotas(dict(zip(NAMES, steps)))
        return ppo, wrapped, collector

    yield build
    for env in services:
        env.close()


def test_uniform_matches_native_central_rollout_gae_and_normalization(make):
    ppo, env, collector = make((24,) * 4)
    rng = torch.get_rng_state()
    actual = collector.collect(0, 0, 0)
    env.close()
    reference, _, native = make(collector_type=CentralCollector)
    torch.set_rng_state(rng)
    expected = native.collect(0, 0, 0)
    a, b = actual.prepare(ppo), expected.prepare(reference)
    for field in (
        "observations",
        "actions",
        "values",
        "rewards",
        "actions_log_prob",
        "returns",
        "advantages",
    ):
        torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=2e-5, atol=2e-6)
    for model, ref in zip((ppo.actor, ppo.critic), (reference.actor, reference.critic)):
        for key, value in model.obs_normalizer.state_dict().items():
            torch.testing.assert_close(value, ref.obs_normalizer.state_dict()[key], rtol=0, atol=0)


def test_nonuniform_full_budget_gae_real_update_and_no_inactive_steps(make, monkeypatch):
    ppo, env, collector = make()
    observed = []
    window = collector.collect(
        0, 0, 0, lambda *args: observed.append(args[3]["active_env_ids"].tolist())
    )
    assert [len(ids) for ids in observed] == [8] * 23 + [6, 2]
    assert window.spec.transitions == 192
    assert list(window.spec.quotas.values()) == [50, 48, 48, 46]
    assert env.env.state.obs["critic"][:, -1].tolist() == [25, 25, 24, 24, 24, 24, 23, 23]
    manual_returns = []
    for source in window.sources:
        raw, advantage = source.raw, torch.zeros((2, 1))
        expected = torch.empty_like(raw.values)
        for t in reversed(range(raw.num_transitions_per_env)):
            done = (source.terminated[t] | source.truncated[t])[:, None]
            delta = raw.rewards[t] + ppo.gamma * source.bootstrap_values[t] - raw.values[t]
            advantage = delta + ppo.gamma * ppo.lam * (~done) * advantage
            expected[t] = advantage + raw.values[t]
        manual_returns.append(expected.flatten(0, 1))
        assert torch.all(source.bootstrap_values[source.terminated] == 0)
        assert torch.any(source.truncated & ~source.terminated)
        assert torch.equal(
            source.episode_ids[1:] - source.episode_ids[:-1],
            (source.terminated | source.truncated)[:-1].long(),
        )
    ppo.storage = window.prepare(ppo)
    torch.testing.assert_close(
        ppo.storage.returns[0], torch.cat(manual_returns), rtol=2e-5, atol=2e-6
    )
    advantage = ppo.storage.returns - ppo.storage.values
    torch.testing.assert_close(
        ppo.storage.advantages, (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    )
    ppo.storage.observations["sample_id"] = torch.arange(192).reshape(1, 192, 1).float()
    generator, epochs = ppo.storage.mini_batch_generator, [[] for _ in range(5)]

    def audit(*args):
        for index, batch in enumerate(generator(*args)):
            epochs[index // 4].extend(batch.observations["sample_id"].flatten().tolist())
            yield batch

    monkeypatch.setattr(ppo.storage, "mini_batch_generator", audit)
    before = [
        torch.nn.utils.parameters_to_vector(m.parameters()).detach().clone()
        for m in (ppo.actor, ppo.critic)
    ]
    normals = [deepcopy(m.obs_normalizer.state_dict()) for m in (ppo.actor, ppo.critic)]
    ppo.update()
    assert all(sorted(epoch) == list(range(192)) for epoch in epochs)
    assert {int(state["step"]) for state in ppo.optimizer.state.values()} == {20}
    for model, old, normal in zip((ppo.actor, ppo.critic), before, normals):
        assert not torch.equal(old, torch.nn.utils.parameters_to_vector(model.parameters()))
        assert int(model.obs_normalizer.count) == 194
        for key, value in normal.items():
            torch.testing.assert_close(
                value, model.obs_normalizer.state_dict()[key], rtol=0, atol=0
            )


def test_version_checks_count_precision_and_continuity(make):
    ppo, _, collector = make()
    for model in (ppo.actor, ppo.critic):
        model.obs_normalizer.count.fill_(2**24 + 1)
    window = collector.collect(0, 0, 0)
    for model in (ppo.actor, ppo.critic):
        assert int(model.obs_normalizer.count) == 2**24 + 193
    with pytest.raises(ValueError, match="version"):
        collector.collect(0, 0, 0)
    with pytest.raises(ValueError, match="normalizer versions"):
        replace(window, normalizer_versions=window.normalizer_versions + 1).prepare(ppo)
    with pytest.raises(ValueError, match="version mismatch"):
        replace(
            window, sources=(replace(window.sources[0], stamp=(0, 1, 0)), *window.sources[1:])
        ).prepare(ppo)
    ppo.storage = window.prepare(ppo)
    ppo.update()
    episodes = collector.episode_ids.clone()
    collector.set_quotas(dict.fromkeys(NAMES, 24))
    following = collector.collect(1, 1, 25)
    assert torch.equal(torch.cat([s.episode_ids[0] for s in following.sources]), episodes)


@pytest.mark.parametrize("fault", ["missing", "nan", "groups"])
def test_fault_closes_all_sources_and_poisoned_collector(make, fault):
    _, env, collector = make(fault=fault)
    pids = [process.pid for process in env.env._processes]
    with pytest.raises((ValueError, RuntimeError)):
        collector.collect(0, 0, 0)
    assert collector.failed and env.env._closed
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    with pytest.raises(ValueError, match="failed collector"):
        collector.collect(0, 0, 0)
