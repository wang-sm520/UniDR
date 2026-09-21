"""Compare the central collector against the installed native PPO step order."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from examples.unidr_fake import FakeSource, FakeState, make_ppo
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from uni_rl.algos.synchronous_ppo import CentralCollector, CentralVecEnv
from uni_rl.ipc.synchronous_env import SourceSpec, SynchronousEnv


class SerialReference(FakeSource):
    """Independent serial vector environment used only as a test oracle."""

    def __init__(self):
        super().__init__(0, 8)
        self.children = [FakeSource(i, 2) for i in range(4)]
        self.source_slices = {f"source{i}": slice(2 * i, 2 * i + 2) for i in range(4)}
        self.fault = None

    def merge(self, states):
        def merge_obs(name):
            return {
                key: np.concatenate([getattr(s, name)[key] for s in states])
                for key in states[0].obs
            }

        self.state = FakeState(
            merge_obs("obs"),
            np.concatenate([s.reward for s in states]),
            np.concatenate([s.terminated for s in states]),
            np.concatenate([s.truncated for s in states]),
            merge_obs("final_observation") if states[0].final_observation is not None else None,
        )
        return self.state

    def init_state(self):
        return self.merge([child.init_state() for child in self.children])

    def reset(self, env_indices):
        for child in self.children:
            child.reset(np.arange(2))
            child.init_state()
        self.init_state()
        return self.state.obs, {}

    def step(self, actions):
        state = self.merge(
            [child.step(actions[2 * i : 2 * i + 2]) for i, child in enumerate(self.children)]
        )
        # Exercise dual flags on some genuinely terminal transitions.
        state.truncated[0] |= state.terminated[0]
        if self.fault == "missing":
            state.final_observation = None
        elif self.fault == "nan":
            state.reward[0] = np.nan
        return state

    def close(self):
        for child in self.children:
            child.close()
        super().close()


def setup():
    torch.set_num_threads(1)
    env = CentralVecEnv(SerialReference())
    ppo = make_ppo(normalize=True)
    ppo.schedule = "adaptive"
    ppo.storage = RolloutStorage("rl", 8, 24, env.get_observations(), [2])
    return ppo, env, CentralCollector(ppo, env, env.env.source_slices)


def norm_state(ppo):
    return [deepcopy(m.obs_normalizer.state_dict()) for m in (ppo.actor, ppo.critic)]


def assert_norm_equal(a, b):
    for x, y in zip(a, b):
        for key in x:
            torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)


def test_stepwise_native_oracle_and_real_adaptive_update():
    ppo, env, collector = setup()
    reference = deepcopy(ppo)
    ref_env = CentralVecEnv(SerialReference())
    observed = []
    rng = torch.get_rng_state()
    window = collector.collect(0, 0, 0, lambda *_: observed.append(norm_state(ppo)))
    actual = window.prepare(ppo)
    torch.set_rng_state(rng)
    reference.train_mode()
    obs = ref_env.get_observations()
    with torch.no_grad():
        for step in range(24):
            actions = reference.act(obs)
            obs, rewards, dones, extras = ref_env.step(actions)
            reference.process_env_step(obs, rewards, dones, extras)
            assert_norm_equal(observed[step], norm_state(reference))
        reference.eval_mode()
        reference.compute_returns(obs)
    # The oracle runs native [T,N] GAE, then orders samples source/time/env.
    st = reference.storage
    expected = RolloutStorage("rl", 192, 1, actual.observations[0], [2])
    for field in (
        "observations",
        "actions",
        "rewards",
        "values",
        "actions_log_prob",
        "returns",
        "advantages",
        "dones",
    ):
        value = getattr(st, field)
        ordered = torch.cat(
            [value[:, part].flatten(0, 1) for part in ref_env.env.source_slices.values()]
        )
        getattr(expected, field)[0].copy_(ordered)
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=2e-5, atol=2e-6
        )
    expected.distribution_params = tuple(
        torch.cat([value[:, part].flatten(0, 1) for part in ref_env.env.source_slices.values()])[
            None
        ]
        for value in st.distribution_params
    )
    expected.step = 1
    assert window.spec.transitions == 192
    assert list(window.spec.quotas.values()) == [48] * 4
    assert torch.equal(window.normalizer_versions, torch.arange(24))
    for source in window.sources:
        assert (source.bootstrap_values[source.terminated] == 0).all()
        assert torch.equal(
            source.episode_ids[1:] - source.episode_ids[:-1],
            (source.terminated | source.truncated)[:-1].long(),
        )
        assert "obs" in source.raw.observations  # Original dictionary survives the adapter.
        corrected = (
            source.raw.rewards
            + ppo.gamma
            * source.bootstrap_values
            * (source.truncated & ~source.terminated)[..., None]
        )
        name = source.source_id
        torch.testing.assert_close(corrected, st.rewards[:, ref_env.env.source_slices[name]])
    ppo.storage, reference.storage = actual, expected
    before = [
        torch.nn.utils.parameters_to_vector(m.parameters()).detach().clone()
        for m in (ppo.actor, ppo.critic)
    ]
    norm_before = norm_state(ppo)
    update_rng = torch.get_rng_state()
    losses = ppo.update()
    torch.set_rng_state(update_rng)
    ref_losses = reference.update()
    assert losses == pytest.approx(ref_losses, rel=1e-4, abs=1e-6)
    assert ppo.learning_rate == reference.learning_rate
    for old, model, ref in zip(
        before, (ppo.actor, ppo.critic), (reference.actor, reference.critic)
    ):
        assert not torch.equal(old, torch.nn.utils.parameters_to_vector(model.parameters()))
        for value, ref_value in zip(model.parameters(), ref.parameters()):
            torch.testing.assert_close(value, ref_value, rtol=1e-4, atol=1e-6)
    assert {int(state["step"]) for state in ppo.optimizer.state.values()} == {20}
    assert_norm_equal(norm_before, norm_state(ppo))
    assert all(int(m.obs_normalizer.count) == 194 for m in (ppo.actor, ppo.critic))


def test_every_epoch_uses_every_sample(monkeypatch):
    ppo, _, collector = setup()
    window = collector.collect(0, 0, 0)
    ppo.storage = window.prepare(ppo)
    generator = ppo.storage.mini_batch_generator
    epochs = [[] for _ in range(5)]
    # Record exact per-sample IDs through an unused observation group.
    ppo.storage.observations["sample_id"] = torch.arange(192).reshape(1, 192, 1).float()

    def audited(*args):
        for index, batch in enumerate(generator(*args)):
            epochs[index // 4].extend(batch.observations["sample_id"].flatten().tolist())
            yield batch

    monkeypatch.setattr(ppo.storage, "mini_batch_generator", audited)
    ppo.update()
    assert all(sorted(epoch) == list(range(192)) for epoch in epochs)


@pytest.mark.parametrize("fault", ["missing", "nan"])
def test_failed_window_closes_all_sources(fault):
    _, env, collector = setup()
    env.env.fault = fault
    with pytest.raises((ValueError, RuntimeError)):
        collector.collect(0, 0, 0)
    assert collector.failed and env.env.closed
    assert all(child.closed for child in env.env.children)
    with pytest.raises(ValueError, match="failed collector"):
        collector.collect(0, 0, 0)


def test_versions_and_episode_continuity_across_windows():
    ppo, _, collector = setup()
    first = collector.collect(0, 0, 0)
    ppo.storage = first.prepare(ppo)
    ppo.update()
    previous = collector.episode_ids.clone()
    with pytest.raises(ValueError, match="version"):
        collector.collect(0, 0, 0)
    second = collector.collect(1, 1, 24)
    assert torch.equal(torch.cat([s.episode_ids[0] for s in second.sources]), previous)
    with pytest.raises(ValueError, match="normalizer versions"):
        replace(second, normalizer_versions=second.normalizer_versions + 1).prepare(ppo)
    sources = (replace(second.sources[0], stamp=(0, 0, 0)), *second.sources[1:])
    with pytest.raises(ValueError, match="version mismatch"):
        replace(second, sources=sources).prepare(ppo)


def test_normalizer_integer_count_above_float32_limit():
    ppo, _, collector = setup()
    for model in (ppo.actor, ppo.critic):
        model.obs_normalizer.count.fill_(2**24 + 1)
    window = collector.collect(0, 0, 0)
    window.prepare(ppo)
    for model in (ppo.actor, ppo.critic):
        assert model.obs_normalizer.count.dtype == torch.int64
        assert int(model.obs_normalizer.count) == 2**24 + 1 + 192


def fake_factory(num_envs, override):
    return FakeSource(override["index"], num_envs)


def test_four_spawn_services_drive_one_real_ppo():
    services = SynchronousEnv(
        [SourceSpec(f"source{i}", fake_factory, 2, {"index": i}) for i in range(4)]
    )
    try:
        wrapped = CentralVecEnv(services)
        ppo = make_ppo(normalize=True)
        ppo.schedule = "adaptive"
        ppo.storage = RolloutStorage("rl", 8, 24, wrapped.get_observations(), [2])
        window = CentralCollector(ppo, wrapped, services.source_slices).collect(0, 0, 0)
        ppo.storage = window.prepare(ppo)
        ppo.update()
        assert {int(state["step"]) for state in ppo.optimizer.state.values()} == {20}
        assert all(source.raw.step == 24 for source in window.sources)
    finally:
        services.close()


def test_rejects_changed_learner_before_gae():
    ppo, _, collector = setup()
    window = collector.collect(0, 0, 0)
    with torch.no_grad():
        next(ppo.critic.parameters()).add_(1)
    with pytest.raises(ValueError, match="learner state changed"):
        window.prepare(ppo)


def test_raw_rollout_survives_cross_window_environment_buffer_reuse(monkeypatch):
    ppo, env, collector = setup()
    initial = env.env.state.obs["obs"].copy()
    step = env.env.step

    def reuse(actions):
        previous = env.env.state.obs
        state = step(actions)
        for key, value in state.obs.items():
            previous[key][:] = value
        state.obs = previous
        return state

    monkeypatch.setattr(env.env, "step", reuse)
    window = collector.collect(0, 0, 0)
    actual = torch.cat([source.raw.observations["obs"][0] for source in window.sources])
    np.testing.assert_array_equal(actual.numpy(), initial)
    before = deepcopy(window.sources)
    ppo.storage = window.prepare(ppo)
    ppo.update()
    collector.collect(1, 1, 24)
    for source, saved in zip(window.sources, before):
        for field in ("env_ids", "episode_ids", "terminated", "truncated", "bootstrap_values"):
            torch.testing.assert_close(
                getattr(source, field), getattr(saved, field), rtol=0, atol=0
            )
        for field in ("last_observations", "final_observations"):
            for key in getattr(saved, field).keys():
                torch.testing.assert_close(
                    getattr(source, field)[key], getattr(saved, field)[key], rtol=0, atol=0
                )
        for key in saved.raw.observations.keys():
            torch.testing.assert_close(
                source.raw.observations[key], saved.raw.observations[key], rtol=0, atol=0
            )
        for field in (
            "actions",
            "rewards",
            "dones",
            "values",
            "actions_log_prob",
            "returns",
            "advantages",
        ):
            torch.testing.assert_close(
                getattr(source.raw, field), getattr(saved.raw, field), rtol=0, atol=0
            )
        for value, original in zip(source.raw.distribution_params, saved.raw.distribution_params):
            torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_normalizer_overflow_aborts_before_update(monkeypatch):
    ppo, env, collector = setup()
    step = env.env.step

    def overflow(actions):
        state = step(actions)
        for value in state.obs.values():
            value[::2, 0], value[1::2, 0] = 1e20, -1e20
        return state

    monkeypatch.setattr(env.env, "step", overflow)
    with pytest.raises(ValueError, match="non-finite normalizer"):
        collector.collect(0, 0, 0)
    assert not ppo.optimizer.state and env.env.closed
