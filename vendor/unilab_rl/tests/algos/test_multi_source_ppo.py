from dataclasses import replace

import pytest
import torch
from examples import unidr_fake

from uni_rl.algos.multi_source_ppo import prepare_ppo_window


def snapshot(source):
    raw = source.raw
    tensors = [
        *raw.observations.values(),
        *(getattr(raw, key) for key in "actions rewards values actions_log_prob dones".split()),
        *raw.distribution_params,
        *(getattr(source, key) for key in "env_ids episode_ids terminated truncated".split()),
        *source.final_observations.values(),
        *source.last_observations.values(),
        source.bootstrap_values,
    ]
    return [value.clone() for value in tensors]


@pytest.mark.parametrize("steps", [(24, 24, 24, 24), (25, 24, 24, 23)])
def test_real_ppo_update_uses_every_sample_and_preserves_raw_window(steps, monkeypatch):
    ppo = unidr_fake.make_ppo(normalize=True)
    norm_before = [
        {key: value.clone() for key, value in model.obs_normalizer.state_dict().items()}
        for model in (ppo.actor, ppo.critic)
    ]
    assert all(state and any(bool(t.any()) for t in state.values()) for state in norm_before)
    spec, sources = unidr_fake.fake_window(ppo, steps=steps)
    originals = [snapshot(source) for source in sources]
    metadata = [(s.source_id, s.segment_id, s.stamp) for s in sources]
    storage = prepare_ppo_window(ppo, sources, spec)
    assert storage.num_envs == spec.transitions == 192
    assert storage.num_transitions_per_env == 1
    expected = storage.returns - storage.values
    torch.testing.assert_close(
        storage.advantages, (expected - expected.mean()) / (expected.std() + 1e-8)
    )
    seen, optimizer_steps = [], []
    original_generator = storage.mini_batch_generator

    def batches(*args, **kwargs):
        for batch in original_generator(*args, **kwargs):
            seen.append(batch.observations["critic"].clone())
            yield batch

    monkeypatch.setattr(storage, "mini_batch_generator", batches)
    before = [[p.detach().clone() for p in model.parameters()] for model in (ppo.actor, ppo.critic)]
    handle = ppo.optimizer.register_step_post_hook(lambda *_: optimizer_steps.append(1))
    ppo.storage = storage
    try:
        losses = ppo.update()
    finally:
        handle.remove()
    assert len(optimizer_steps) == len(seen) == 20
    expected_rows = {tuple(row) for row in storage.observations["critic"][0].tolist()}
    assert len(expected_rows) == 192
    for epoch in range(5):
        rows = torch.cat(seen[epoch * 4 : (epoch + 1) * 4]).tolist()
        assert len(rows) == 192 and {tuple(row) for row in rows} == expected_rows
    for model, previous, norm in zip((ppo.actor, ppo.critic), before, norm_before, strict=True):
        assert any(
            not torch.equal(old, new) for old, new in zip(previous, model.parameters(), strict=True)
        )
        for key, value in norm.items():
            assert torch.equal(value, model.obs_normalizer.state_dict()[key])
        assert not model.training and not model.obs_normalizer.training
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    for source, original, ids in zip(sources, originals, metadata, strict=True):
        assert (source.source_id, source.segment_id, source.stamp) == ids
        assert all(
            torch.equal(old, new) for old, new in zip(original, snapshot(source), strict=True)
        )


def test_gae_terminal_timeout_both_flags_and_window_tail(monkeypatch):
    ppo = unidr_fake.make_ppo()
    spec, sources = unidr_fake.fake_window(ppo)
    ppo.gamma, ppo.lam = 0.9, 0.8
    monkeypatch.setattr(
        ppo.critic, "forward", lambda obs, **_: torch.full((*obs.batch_size, 1), 2.0)
    )
    for source in sources:
        source.raw.rewards.fill_(1)
        source.raw.values.fill_(2)
        source.terminated.fill_(True)
        source.truncated.zero_()
        source.bootstrap_values.zero_()
    source = sources[0]
    source.raw.rewards[:4, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    source.terminated[:2, 0] = False
    source.truncated[1, 0] = source.truncated[3, 0] = True
    source.bootstrap_values[0, 0, 0] = 2
    source.bootstrap_values[1, 0, 0] = 5
    source.terminated[-1, 1] = False
    source.bootstrap_values[-1, 1, 0] = 2
    for source in sources:
        done = source.terminated | source.truncated
        source.raw.dones.copy_(done.unsqueeze(-1))
        source.episode_ids.zero_()
        source.episode_ids[1:] = done[:-1].long().cumsum(0)
    merged = prepare_ppo_window(ppo, sources, spec)
    returns = merged.returns[0, :48, 0].reshape(24, 2)
    torch.testing.assert_close(returns[:4, 0], torch.tensor([6.04, 6.5, 3.0, 4.0]))
    torch.testing.assert_close(returns[-1, 1], torch.tensor(2.8))
    assert torch.equal(merged.returns[0, 48:, 0], torch.ones(144))
    original = merged.returns[0].clone()
    sources[0].raw.rewards.add_(100)
    changed = prepare_ppo_window(ppo, sources, spec).returns[0]
    assert torch.equal(original[48:], changed[48:])
    assert not torch.equal(original[:48], changed[:48])
    reversed_returns = prepare_ppo_window(ppo, sources[::-1], spec).returns[0]
    assert torch.equal(reversed_returns.reshape(4, 48, 1).flip(0).flatten(0, 1), changed)


def test_collection_uses_final_observation_for_timeout_bootstrap():
    ppo = unidr_fake.make_ppo()
    spec, sources = unidr_fake.fake_window(ppo)
    for source in sources:
        assert source.stamp == spec.stamp
        assert source.terminated.any() and source.truncated.any()
        assert bool((source.bootstrap_values[source.terminated] == 0).all())
        timeout = source.truncated & ~source.terminated
        with torch.no_grad():
            expected = ppo.critic(source.final_observations[timeout])
            reset_values = ppo.critic(source.raw.observations[1:][timeout[:-1]])
            ppo.actor(source.raw.observations.flatten(0, 1), stochastic_output=True)
            log_prob = ppo.actor.get_output_log_prob(source.raw.actions.flatten(0, 1))
        torch.testing.assert_close(source.raw.actions_log_prob.flatten(), log_prob)
        torch.testing.assert_close(source.bootstrap_values[timeout], expected)
        assert not torch.allclose(source.bootstrap_values[:-1][timeout[:-1]], reset_values)
        assert torch.equal(
            source.episode_ids[1:] - source.episode_ids[:-1], source.raw.dones[:-1, :, 0].long()
        )


@pytest.mark.parametrize(
    "bad",
    "policy normalizer window duplicate under over total final nan groups shape episode bootstrap distribution".split(),
)
def test_malformed_window_rejected_before_gae(bad, monkeypatch):
    ppo = unidr_fake.make_ppo()
    spec, sources = unidr_fake.fake_window(ppo)
    source = sources[0]
    if bad in ("window", "policy", "normalizer"):
        field = "window_id" if bad == "window" else f"{bad}_version"
        spec = replace(spec, **{field: 1})  # Collected version zero is now stale.
    elif bad == "duplicate":
        sources[1] = source
    elif bad == "under":
        sources.pop()
    elif bad == "over":
        sources.append(replace(source, segment_id=1))
    elif bad == "total":
        spec = replace(spec, transitions=spec.transitions + 1)
    elif bad == "final":
        sources[0] = replace(source, final_observations=None)
    elif bad == "nan":
        source.raw.rewards[0, 0] = float("nan")
    elif bad == "groups":
        del source.raw.observations["critic"]
    elif bad == "shape":
        source.raw.observations["critic"] = source.raw.observations["critic"][..., :1]
    elif bad == "episode":
        source.episode_ids[1, 0] += 10
    elif bad == "bootstrap":
        source.bootstrap_values[source.terminated] = 1
    else:
        source.raw.distribution_params[1].zero_()
    original = ppo.storage
    monkeypatch.setattr(
        ppo, "compute_returns", lambda *_: pytest.fail("invalid window reached GAE")
    )
    with pytest.raises(ValueError):
        prepare_ppo_window(ppo, sources, spec)
    assert ppo.storage is original and not ppo.normalize_advantage_per_mini_batch


@pytest.mark.parametrize(
    "path,value",
    [
        ("actor.is_recurrent", True),
        ("actor.training", True),
        ("critic.obs_normalizer.training", True),
        ("num_learning_epochs", 4),
        ("num_mini_batches", 3),
        ("schedule", "adaptive"),
        ("normalize_advantage_per_mini_batch", True),
        ("enable_compile", True),
    ],
)
def test_unsupported_ppo_configuration_rejected(path, value, monkeypatch):
    ppo = unidr_fake.make_ppo()
    spec, sources = unidr_fake.fake_window(ppo)
    *parents, attr = path.split(".")
    target = ppo
    for parent in parents:
        target = getattr(target, parent)
    monkeypatch.setattr(target, attr, value)
    with pytest.raises(ValueError):
        prepare_ppo_window(ppo, sources, spec)


def test_gae_failure_restores_learner_storage(monkeypatch):
    ppo = unidr_fake.make_ppo()
    spec, sources = unidr_fake.fake_window(ppo)
    original = ppo.storage

    def fail(*_):
        raise RuntimeError("GAE failure")

    monkeypatch.setattr(ppo, "compute_returns", fail)
    with pytest.raises(RuntimeError, match="GAE failure"):
        prepare_ppo_window(ppo, sources, spec)
    assert ppo.storage is original and not ppo.normalize_advantage_per_mini_batch


@pytest.mark.parametrize("where", ["step", "constructor"])
def test_fake_failure_closes_every_created_source(where, monkeypatch):
    created, closed = [], []
    original_init = unidr_fake.FakeSource.__init__
    original_close = unidr_fake.FakeSource.close

    def init(self, *args, **kwargs):
        if where == "constructor" and len(created) == 2:
            raise RuntimeError("injected failure")
        original_init(self, *args, **kwargs)
        created.append(self)

    def close(self):
        closed.append(self)
        original_close(self)

    def fail(*_):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(unidr_fake.FakeSource, "__init__", init)
    monkeypatch.setattr(unidr_fake.FakeSource, "close", close)
    if where == "step":
        monkeypatch.setattr(unidr_fake.FakeSource, "step", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        unidr_fake.fake_window(unidr_fake.make_ppo())
    assert created and len(closed) == len(created) and set(closed) == set(created)


def test_fake_sources_are_deterministic():
    _, first = unidr_fake.fake_window(unidr_fake.make_ppo(seed=17))
    _, second = unidr_fake.fake_window(unidr_fake.make_ppo(seed=17))
    for left, right in zip(first, second, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(snapshot(left), snapshot(right), strict=True))


@pytest.mark.parametrize(
    "bad", "groups shape obs_nan final final_groups terminated truncated reward".split()
)
def test_invalid_env_output_rejected_before_storage_conversion(bad, monkeypatch):
    original = unidr_fake.FakeSource.step

    def step(self, actions):
        state = original(self, actions)
        if bad == "groups":
            state.obs["extra"] = state.obs["obs"]
        elif bad == "shape":
            state.obs["obs"] = state.obs["obs"][:, :1]
        elif bad == "obs_nan":
            state.obs["obs"][0, 0] = float("nan")
        elif bad == "final":
            state.final_observation = None
        elif bad == "final_groups":
            del state.final_observation["critic"]
        else:
            setattr(state, bad, getattr(state, bad).astype("float32") + float("nan"))
        return state

    monkeypatch.setattr(unidr_fake.FakeSource, "step", step)
    with pytest.raises(ValueError):
        unidr_fake.fake_window(unidr_fake.make_ppo())
