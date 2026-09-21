"""Run the C1 fake-data smoke using the runtime-owned rollout bridge."""

import json

import hydra
import torch
from omegaconf import DictConfig
from torch.nn.utils import parameters_to_vector
from unidr_fake import fake_window, make_ppo

from uni_rl.algos.multi_source_ppo import prepare_ppo_window


@hydra.main(version_base=None, config_path="configs", config_name="unidr_fake_ppo")
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(1)
    ppo = make_ppo(cfg.seed, cfg.normalize)
    models = (ppo.actor, ppo.critic)
    before = [parameters_to_vector(model.parameters()).detach().clone() for model in models]
    spec, sources = fake_window(ppo, cfg.steps, cfg.num_envs)
    ppo.storage = prepare_ppo_window(ppo, sources, spec)
    losses = ppo.update()
    changed = [
        not torch.equal(old, parameters_to_vector(model.parameters()))
        for old, model in zip(before, models)
    ]
    assert all(changed), "actor and critic must both update"
    print(
        json.dumps(
            {
                "evidence": "fake sources / real PPO",
                "source_counts": dict(spec.quotas),
                "transitions": spec.transitions,
                "actor_critic_changed": changed,
                "optimizer_steps": sorted(
                    {int(state["step"]) for state in ppo.optimizer.state.values()}
                ),
                "losses": losses,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
