"""Deterministic fixtures for the C1 example; these are not robot simulators."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from uni_rl.algos.multi_source_ppo import SourceRollout, WindowSpec, collect_source
from uni_rl.algos.rsl_rl_ppo import FinalObservationAwarePPO

SOURCE_IDS = ("fake_isaacsim", "fake_isaacgym", "fake_genesis", "fake_motrix")


@dataclass
class FakeState:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_observation: dict[str, np.ndarray] | None = None
    info: dict[str, Any] = field(default_factory=dict)


class FakeSource:
    obs_groups_spec = {"obs": 4, "critic": 5}
    action_space = SimpleNamespace(shape=(2,))
    observation_space = SimpleNamespace(shape=(4,))
    cfg = SimpleNamespace(max_episode_seconds=1.0, ctrl_dt=0.02)
    play_capabilities = SimpleNamespace(supports_physics_state_playback=False)

    def __init__(self, source: int, num_envs: int):
        self.source, self.num_envs = source, num_envs
        self.position = np.zeros((num_envs, 2), dtype=np.float32)
        self.age = np.zeros(num_envs, dtype=np.int64)
        self.tick = 0
        self.closed = False
        self.state: FakeState | None = None

    def _obs(self) -> dict[str, np.ndarray]:
        obs = np.column_stack(
            (np.full(self.num_envs, self.source), np.arange(self.num_envs), self.position)
        ).astype(np.float32)
        return {
            "obs": obs,
            "critic": np.column_stack((obs, np.full(self.num_envs, self.tick))).astype(np.float32),
        }

    def init_state(self) -> FakeState:
        self.state = FakeState(
            self._obs(),
            np.zeros(self.num_envs, np.float32),
            self.age.astype(bool),
            self.age.astype(bool),
        )
        return self.state

    def step(self, actions: np.ndarray) -> FakeState:
        if self.closed:
            raise RuntimeError("closed fake source")
        self.position += 0.05 * np.tanh(actions) + 0.01 * (self.source + 1)
        self.age += 1
        self.tick += 1
        reward = (1 - np.square(self.position - 0.3).sum(axis=1)).astype(np.float32)
        terminated = (self.age >= 5 + self.source) & (np.arange(self.num_envs) % 2 == 0)
        truncated = self.age >= 7 + self.source
        final = self._obs()
        done = terminated | truncated
        self.position[done], self.age[done] = 0, 0
        self.state = FakeState(self._obs(), reward, terminated, truncated, final)
        return self.state

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        self.position[env_indices], self.age[env_indices] = 0, 0
        return {key: value[env_indices] for key, value in self._obs().items()}, {}

    def set_nan_guard(self, guard: Any) -> None:
        del guard

    def close(self) -> None:
        self.closed = True


def make_ppo(seed: int = 1, normalize: bool = False) -> FinalObservationAwarePPO:
    torch.manual_seed(seed)
    obs = TensorDict({"obs": torch.zeros(1, 4), "critic": torch.zeros(1, 5)}, [1])
    groups = {"actor": ["obs"], "critic": ["critic"]}
    actor = MLPModel(
        obs,
        groups,
        "actor",
        2,
        hidden_dims=[16, 16],
        obs_normalization=normalize,
        distribution_cfg={"class_name": "rsl_rl.modules:GaussianDistribution", "init_std": 0.5},
    )
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16, 16], obs_normalization=normalize)
    if normalize:
        calibration = TensorDict(
            {
                key: torch.stack((torch.full((dim,), 0.25), torch.full((dim,), 1.5)))
                for key, dim in FakeSource.obs_groups_spec.items()
            },
            [2],
        )
        for model in (actor, critic):
            model.update_normalization(calibration)
    return FinalObservationAwarePPO(
        actor.eval(), critic.eval(), RolloutStorage("rl", 1, 1, obs, [2]), schedule="fixed"
    )


def fake_window(
    ppo: FinalObservationAwarePPO, steps=(24, 24, 24, 24), num_envs: int = 2
) -> tuple[WindowSpec, list[SourceRollout]]:
    if len(steps) != 4 or num_envs <= 0:
        raise ValueError("four sources and positive pool capacity required")
    spec = WindowSpec(0, 0, 0, dict(zip(SOURCE_IDS, (num_envs * t for t in steps))), num_envs * 96)
    with ExitStack() as stack:
        envs = []
        for index in range(4):
            env = FakeSource(index, num_envs)
            stack.callback(env.close)
            envs.append(env)
        sources = [
            collect_source(ppo, env, spec, name, horizon)
            for env, name, horizon in zip(envs, SOURCE_IDS, steps)
        ]
    return spec, sources
