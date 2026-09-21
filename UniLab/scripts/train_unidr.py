"""Hydra entrypoint for the runtime-owned synchronous four-source PPO runner."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from unilab.training.synchronous import run_synchronous_training


@hydra.main(version_base="1.3", config_path="../src/unilab/conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    run_synchronous_training(cfg, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
