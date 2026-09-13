from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from uni_rl.env_contract import EnvAlgoCapabilities, get_algo_capabilities
from uni_rl.utils.nan_guard import NanGuard, NanGuardCfg

from unilab.base.env_factory import make_registry_env
from unilab.training.multi_source import (
    SOURCE_ORDER,
    build_multi_source_plan,
    create_multi_source_training_env,
    source_timing_context,
    validate_multi_source_topology,
)
from unilab.training.run import apply_env_nan_guard
from unilab.utils.sim2sim import extract_contract_snapshot

ROOT = Path(__file__).resolve().parents[2]


def _config(owner: str = "multisim", *overrides: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "src/unilab/conf/ppo")):
        return compose(config_name="config", overrides=[f"task=g1_walk_flat/{owner}", *overrides])


def test_shared_g1_profile_has_fixed_task_and_dr() -> None:
    cfg = _config()
    assert cfg.training.source_timing.enabled is True
    assert cfg.algo.num_envs == 8000
    assert cfg.algo.num_steps_per_env == 24
    assert cfg.algo.empirical_normalization is False
    assert list(cfg.algo.obs_groups.actor) == ["actor"]
    assert list(cfg.algo.obs_groups.critic) == ["critic"]
    assert cfg.env.actions.joint_pos.scale == 0.25
    assert cfg.env.ctrl_dt == 0.02
    assert cfg.env.max_episode_seconds == 20.0
    assert list(cfg.env.commands.twist.ranges.lin_vel_x) == [0.4, 0.7]
    assert list(cfg.env.commands.twist.ranges.lin_vel_y) == [0.0, 0.0]
    assert list(cfg.env.commands.twist.ranges.ang_vel_z) == [0.0, 0.0]
    assert list(cfg.env.events.base_mass.params.mass_distribution_params) == [0.8, 1.2]
    assert list(cfg.env.events.base_mass.params.asset_cfg.body_names) == ["pelvis"]
    assert cfg.env.events.base_mass.params.recompute_inertia is False
    assert list(cfg.env.events.pd_gains.params.kp_range) == [0.9, 1.1]
    assert list(cfg.env.events.pd_gains.params.kd_range) == [0.9, 1.1]
    assert cfg.reward.feet_air_time.weight == 0.25
    assert cfg.env.terminations.tilt.params.max_tilt_deg == 25.0
    assert cfg.env.terminations.base_height.params.minimum_height == 0.55
    assert list(cfg.env.events.reset_root_state_uniform.params.velocity_range.x) == [-0.5, 0.5]


def test_source_timing_disabled_for_single_backend() -> None:
    cfg = _config("isaacgym_multisim")
    assert cfg.training.source_timing.enabled is False
    with source_timing_context(cfg, None, None, log_dir=None):
        pass


def test_source_timing_owner_passes_live_statistics(tmp_path, monkeypatch) -> None:
    from contextlib import nullcontext

    import uni_rl.algos.rsl_rl_source_timing as timing

    captured = {}
    runner = object()
    env = SimpleNamespace(source_statistics={"isaacgym": {"step_calls": 0}})

    def capture(actual_runner, **kwargs):
        assert actual_runner is runner
        captured.update(kwargs)
        return nullcontext()

    monkeypatch.setattr(timing, "record_source_timing", capture)
    with source_timing_context(_config(), runner, env, log_dir=tmp_path):
        assert captured["output_path"] == tmp_path / "source_timing.jsonl"
        env.source_statistics = {"isaacgym": {"step_calls": 24}}
        assert captured["source_statistics"]() == env.source_statistics


def test_source_timing_requires_multisource_run_directory() -> None:
    with pytest.raises(ValueError, match="run directory"):
        source_timing_context(_config(), None, None, log_dir=None)
    cfg = _config("isaacgym_multisim", "training.source_timing.enabled=true")
    with pytest.raises(ValueError, match="multi-source training"):
        source_timing_context(cfg, None, None, log_dir="unused")


def test_sources_share_policy_contract_and_evaluation_profiles() -> None:
    cfg = _config()
    plan = build_multi_source_plan(cfg, root_dir=ROOT)
    assert tuple(source.name for source in plan.sources) == SOURCE_ORDER
    assert [source.num_envs for source in plan.sources] == [2000] * 4
    assert [source.seed for source in plan.sources] == [2, 3, 4, 5]
    assert plan.manifest["samples_per_iteration"] == 192000
    assert [entry["fraction"] for entry in plan.manifest["sources"]] == [0.25] * 4
    contract = extract_contract_snapshot(cfg)
    for name, source_manifest in zip(SOURCE_ORDER, plan.manifest["sources"], strict=True):
        assert source_manifest["contract_snapshot"] == contract
        evaluation = _config(f"{name}_multisim")
        assert extract_contract_snapshot(evaluation) == contract
        assert OmegaConf.to_container(evaluation.reward) == OmegaConf.to_container(cfg.reward)
        assert evaluation.training.sim_backend == name


def test_mujoco_evaluation_profile_preserves_multisim_training_contract() -> None:
    training = _config()
    evaluation = _config("mujoco_multisim")
    assert evaluation.training.sim_backend == "mujoco"
    assert evaluation.training.multi_source is None
    assert evaluation.play_profile.enabled is False
    assert extract_contract_snapshot(evaluation) == extract_contract_snapshot(training)
    assert OmegaConf.to_container(evaluation.env) == OmegaConf.to_container(training.env)
    assert OmegaConf.to_container(evaluation.reward) == OmegaConf.to_container(training.reward)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("algo.num_envs=4000", "8000"),
        ("algo.num_steps_per_env=12", "24-step"),
        ("training.no_play=false", "no_play"),
        ("training.play_render_mode=record", "play_render_mode"),
        ("training.play_only=true", "real backend"),
        ("training.device=cpu", "cuda:0"),
    ],
)
def test_invalid_multisim_topology_fails_early(override: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_multi_source_topology(_config("multisim", override))


@pytest.mark.parametrize("world_size,devices", [(2, None), (1, [0, 1]), (1, [1])])
def test_multisim_never_launches_torchrun(world_size, devices) -> None:
    with pytest.raises(ValueError, match="one learner"):
        validate_multi_source_topology(_config(), world_size=world_size, devices=devices)


def test_physical_backend_cannot_enable_multisource_accidentally() -> None:
    cfg = _config()
    cfg.training.sim_backend = "isaacgym"
    with pytest.raises(ValueError, match="task owner"):
        validate_multi_source_topology(cfg)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"num_envs": 1999}, "exactly 2000"),
        ({"backend": "motrix"}, "must select backend"),
        ({"env": {"actions": {}}}, "shared task"),
        ({"env": {"isaacgym_device_id": 1}}, "GPU 0"),
    ],
)
def test_source_overrides_cannot_change_task_or_quota(change, message) -> None:
    cfg = _config()
    cfg.training.multi_source.sources[0] = OmegaConf.merge(
        OmegaConf.to_container(cfg.training.multi_source.sources[0], resolve=True), change
    )
    with pytest.raises(ValueError, match=message):
        build_multi_source_plan(cfg, root_dir=ROOT)


def test_source_order_is_not_completion_order_or_configurable() -> None:
    cfg = _config()
    values = OmegaConf.to_container(cfg.training.multi_source.sources)
    cfg.training.multi_source.sources = list(reversed(values))
    with pytest.raises(ValueError, match="source order"):
        build_multi_source_plan(cfg, root_dir=ROOT)


@pytest.mark.parametrize(
    "override",
    [
        "algo.empirical_normalization=true",
        "env.ctrl_dt=0.01",
        "env.commands.twist.ranges.lin_vel_x=[0.1,1.0]",
        "reward.feet_air_time.weight=0.0",
        "env.events.pd_gains.params.kp_range=[1.0,1.0]",
    ],
)
def test_fixed_experiment_cannot_silently_disable_features(override: str) -> None:
    with pytest.raises(ValueError, match="multisim"):
        validate_multi_source_topology(_config("multisim", override))


def test_actual_environment_dimensions_fail_closed(monkeypatch) -> None:
    import uni_rl.ipc.multi_source_env as runtime

    plan = build_multi_source_plan(_config(), root_dir=ROOT)
    closed = []
    env = SimpleNamespace(obs_groups_spec={"obs": 1}, close=lambda: closed.append(True))
    monkeypatch.setattr(runtime, "make_multi_source_env", lambda *args, **kwargs: env)
    with pytest.raises(ValueError, match="obs=98/critic=101"):
        create_multi_source_training_env(plan)
    assert closed == [True]


def test_real_g1_reports_resolved_action_order() -> None:
    plan = build_multi_source_plan(_config(), root_dir=ROOT)
    env = make_registry_env("G1WalkFlat", "mujoco", 2, plan.sources[2].env_cfg_override)
    try:
        capabilities = get_algo_capabilities(env)
        assert capabilities.joint_names == plan.joint_names
        np.testing.assert_array_equal(capabilities.action_low, env.action_space.low)
        np.testing.assert_array_equal(capabilities.action_high, env.action_space.high)
    finally:
        env.close()


def test_actual_environment_joint_order_fails_closed(monkeypatch) -> None:
    import uni_rl.ipc.multi_source_env as runtime

    plan = build_multi_source_plan(_config(), root_dir=ROOT)
    closed = []
    env = SimpleNamespace(
        obs_groups_spec={"obs": 98, "critic": 101},
        action_space=SimpleNamespace(shape=(29,)),
        algo_capabilities=EnvAlgoCapabilities(joint_names=tuple(reversed(plan.joint_names))),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(runtime, "make_multi_source_env", lambda *args, **kwargs: env)
    with pytest.raises(ValueError, match="action joint order"):
        create_multi_source_training_env(plan)
    assert closed == [True]


def test_training_attaches_the_shared_runtime_nan_guard() -> None:
    attached = []
    env = SimpleNamespace(
        num_envs=8000,
        play_capabilities=SimpleNamespace(supports_physics_state_playback=False),
        set_nan_guard=attached.append,
    )
    apply_env_nan_guard(env, _config().training)
    assert len(attached) == 1
    assert isinstance(attached[0], NanGuard)
    assert isinstance(attached[0].cfg, NanGuardCfg)
    assert attached[0].cfg.enabled
