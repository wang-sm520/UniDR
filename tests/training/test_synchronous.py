"""Owner-to-runtime wiring without importing a vendor SDK or constructing physics."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.training.source_probe import probe_env_factory
from unilab.training.synchronous import SOURCE_ORDER, build_manifest, build_sources

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


def test_adaptive_owner_preserves_task_ppo_capacity_and_source_factories():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", ["task=g1_flip_tracking/unidr_adaptive"])
        fixed = compose("config", ["task=g1_flip_tracking/unidr_comparison"])
        specs, _ = build_sources(cfg, ROOT)
    for key in ("env", "reward", "algo"):
        assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(
            fixed[key], resolve=True
        )
    assert cfg.algo.num_envs == 1024 and cfg.algo.max_iterations == 5000
    assert cfg.unidr.adaptive.enabled and cfg.unidr.adaptive.probe.interval == 100
    assert cfg.unidr.adaptive.probe.horizon == 224
    for spec in specs:
        expected = probe_env_factory(spec.source_id)
        assert spec.factory.func is expected.func and spec.factory.args == expected.args


def test_adaptive_four_gpu_mapping_and_complete_flip_horizon():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", ["task=g1_flip_tracking/unidr_adaptive_four_gpu"])
        specs, snapshots = build_sources(cfg, ROOT)
        assert cfg.training.device == cfg.unidr.learner_device == "cuda:3"
        assert [item["device"] for item in snapshots] == ["cuda:0", "cuda:1", "cuda:2", "cpu"]
        assert [spec.num_envs for spec in specs] == [1024] * 4
        assert cfg.algo.max_iterations == 5000 and cfg.unidr.adaptive.enabled
        cfg.unidr.adaptive.probe.horizon = 100
        with pytest.raises(ValueError, match="complete 224-step"):
            build_sources(cfg, ROOT)


@pytest.mark.parametrize("override", ["enabled=false", "probe.interval=50", "probe.error_cap=1.0"])
def test_manifest_fingerprints_adaptive_mode_and_probe_scales(override, monkeypatch):
    monkeypatch.setattr("unilab.training.synchronous.ensure_robot_assets_for_paths", lambda _: None)
    monkeypatch.setattr("unilab.training.synchronous.resolve_motion_files", lambda _: None)
    monkeypatch.setattr("unilab.training.synchronous.get_git_info", lambda _: {})
    monkeypatch.setattr("unilab.training.synchronous.importlib.metadata.version", lambda _: "test")
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", ["task=g1_flip_tracking/unidr_adaptive"])
        _, snapshots = build_sources(cfg, ROOT)
        original = build_manifest(cfg, ROOT, snapshots)
        cfg = compose(
            "config", ["task=g1_flip_tracking/unidr_adaptive", f"unidr.adaptive.{override}"]
        )
        changed = build_manifest(cfg, ROOT, snapshots)
    assert original["digest"] != changed["digest"]
    assert original["sources"] == changed["sources"]
    assert original["algorithm"] == changed["algorithm"]
