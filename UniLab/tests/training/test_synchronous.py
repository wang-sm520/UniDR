"""Owner-to-runtime wiring without importing a vendor SDK or constructing physics."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from unilab.training.synchronous import SOURCE_ORDER, build_sources

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("owner,ids", [("single", (0, 0, 0)), ("four", (0, 1, 2))])
def test_build_sources_preserves_mapping_and_capacity(owner, ids):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", [f"task=g1_flip_tracking/unidr_{owner}_gpu"])
        specs, snapshots = build_sources(cfg, ROOT)
    assert tuple(s.source_id for s in specs) == SOURCE_ORDER
    assert [s.num_envs for s in specs] == [1024] * 4
    for source, device_id in zip(specs, ids):
        assert source.env_cfg_override[f"{source.source_id}_device_id"] == device_id
    assert [s.env_cfg_override["seed"] for s in specs] == [2, 3, 4, 5]
    assert snapshots[-1]["device"] == "cpu"


@pytest.mark.parametrize(
    "override,match",
    [
        ("unidr.sources.0.name=mujoco", "fixed order"),
        ("training.sim2sim_strict=false", "strict"),
        ("reward.motion_body_pos.weight=1.0", "behavior"),
        ("env.actions.joint_pos.scale=0.25", "policy contract"),
        ("unidr.sources.1.device=cuda:2", "device"),
        ("+unidr.sources.3.env_overrides.sim_dt=0.01", "only bind"),
        ("algo.algorithm.gamma=0.5", "PPO parameters"),
        ("algo.policy.activation=relu", "PPO parameters"),
        ("training.logger=wandb", "local TensorBoard"),
    ],
)
def test_invalid_source_configuration_fails_before_env_construction(override, match):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", ["task=g1_flip_tracking/unidr_single_gpu", override])
        with pytest.raises(ValueError, match=match):
            build_sources(cfg, ROOT)
