"""Independent hand-calculated first-episode probe outcomes, using actual PPO models."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from examples.unidr_fake import make_ppo
from tensordict import TensorDict

from uni_rl.algos.source_probe import run_source_probe

CONFIG = dict(interval=100, horizon=4, seed=3, error_cap=0.5, return_min=0, return_max=4)


class ProbeService:
    def __init__(self, ppo, fault=None):
        self.ppo, self.fault = ppo, fault
        self.source_slices = {f"s{i}": slice(2 * i, 2 * i + 2) for i in range(4)}
        self.obs = {"obs": np.ones((8, 4), np.float32), "critic": np.ones((8, 5), np.float32)}
        self.tick = 0
        self.closed = self.finished = False
        self.previous_done = np.zeros(8, bool)

    def reset_probe(self, seed):
        assert seed == CONFIG["seed"]
        return self.obs, {}

    def step(self, actions):
        assert actions.shape == (8, 2) and np.isfinite(actions).all()
        assert not self.ppo.actor.training and not self.ppo.critic.training
        self.tick += 1
        terminated, truncated = np.zeros(8, bool), np.zeros(8, bool)
        if self.tick == 1:
            terminated[0] = True
        if self.tick == 2:
            truncated[2] = True
        if self.tick == 3:
            terminated[4] = truncated[4] = True
        reward = np.where(self.previous_done, 1000, 1).astype(np.float32)
        self.previous_done |= terminated | truncated
        logs = {
            f"s{i}": {"probe": {"error": np.full(2, error, np.float32)}}
            for i, error in enumerate((0.1, 0.2, 0.8, 0.4))
        }
        if self.fault == "missing":
            logs["s0"] = {}
        elif self.fault in {"nan", "negative", "shape"}:
            logs["s0"]["probe"]["error"] = {
                "nan": np.array([np.nan, 0]),
                "negative": np.array([-1, 0]),
                "shape": np.zeros(3),
            }[self.fault]
        elif self.fault == "mutation":
            next(self.ppo.actor.parameters()).add_(1)
        return SimpleNamespace(
            obs=self.obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"source_logs": logs},
        )

    def finish_probe(self):
        self.finished = True

    def close(self):
        self.closed = True


def setup(fault=None):
    ppo = make_ppo(normalize=True)
    ppo.train_mode()
    service = ProbeService(ppo, fault)
    wrapped = SimpleNamespace(
        num_envs=8,
        env=service,
        close=service.close,
        episode_returns=torch.ones(8),
        episode_lengths=torch.ones(8),
        _obs_to_tensordict=lambda obs: TensorDict(
            {k: torch.as_tensor(v) for k, v in obs.items()}, [8]
        ),
    )
    return ppo, wrapped


def test_first_episode_padding_survival_and_frozen_actual_policy():
    ppo, env = setup()
    before = [deepcopy(model.state_dict()) for model in (ppo.actor, ppo.critic)]
    rng = torch.get_rng_state().clone()
    result = run_source_probe(ppo, env, CONFIG)
    # H=4, cap=.5: s0 failing row error=(.1+3*.5)/4; survivor=.1.
    # s1 timeout row=(2*.2+2*.5)/4; s2 clips observed .8 to .5.
    expected = [(0.25, 2.5, 0.5), (0.275, 3, 0.5), (0.5, 3.5, 0.5), (0.4, 4, 1)]
    assert tuple(result) == ("s0", "s1", "s2", "s3")
    for actual, (error, reward, survival) in zip(result.values(), expected):
        assert actual == pytest.approx(
            {
                "error": error / 0.5,
                "survival": survival,
                "return": reward / 4,
                "raw_error_m": error,
                "raw_return": reward,
                "opportunities": 2,
                "horizon_steps": 4,
            }
        )
    for model, saved in zip((ppo.actor, ppo.critic), before):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
    assert ppo.storage.step == 0 and not ppo.optimizer.state
    assert torch.equal(rng, torch.get_rng_state())
    assert not env.episode_returns.any() and not env.episode_lengths.any()
    assert env.env.finished and not env.env.closed and env.env.tick == 4


@pytest.mark.parametrize("fault", ["missing", "nan", "negative", "shape", "mutation"])
def test_invalid_metrics_or_actor_mutation_fail_and_close(fault):
    ppo, env = setup(fault)
    with pytest.raises(ValueError):
        run_source_probe(ppo, env, CONFIG)
    assert env.env.closed and not env.env.finished
    assert ppo.storage.step == 0 and not ppo.optimizer.state


@pytest.mark.parametrize("part", [slice(0, 0), slice(0, 1)])
def test_zero_or_unequal_opportunities_rejected(part):
    ppo, env = setup()
    env.env.source_slices["s0"] = part
    with pytest.raises(ValueError):
        run_source_probe(ppo, env, CONFIG)
    assert env.env.closed


def test_nonbinary_error_cap_keeps_perfect_probe_in_bounds(monkeypatch):
    ppo, env = setup()
    step = env.env.step

    def perfect(actions):
        state = step(actions)
        state.terminated[:] = state.truncated[:] = False
        for source in state.info["source_logs"].values():
            source["probe"]["error"].fill(0)
        return state

    monkeypatch.setattr(env.env, "step", perfect)
    result = run_source_probe(ppo, env, CONFIG | {"horizon": 224, "error_cap": 0.2})
    assert all(0 <= item["error"] < 1e-12 for item in result.values())


@pytest.mark.parametrize("config", [{}, CONFIG | {"horizon": -1}, CONFIG | {"error_cap": 0}])
def test_invalid_probe_configuration_also_closes_resources(config):
    ppo, env = setup()
    with pytest.raises((KeyError, ValueError)):
        run_source_probe(ppo, env, config)
    assert env.env.closed and env.env.tick == 0
