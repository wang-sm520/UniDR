"""Single-backend comparison owners preserve the shared flip training profile."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from uni_rl.algos.rsl_rl import normalize_ppo_train_cfg

from unilab.base.config_adapter import BackendAdapter
from unilab.training import algo_config_dict
from unilab.training.experiment import write_run_config_snapshot
from unilab.training.single_comparison import prepare_run
from unilab.utils.sim2sim import (
    CrossBackendIncompatibleError,
    extract_contract_snapshot,
    resolve_sim2sim_config,
)

ROOT = Path(__file__).parents[2]
BACKENDS = ("motrix", "isaacsim", "isaacgym", "genesis")
ENV_SEEDS = {"isaacsim": 2, "isaacgym": 3, "genesis": 4, "motrix": 5}


def config(owner, *overrides):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task=g1_flip_tracking/{owner}", *overrides])


@pytest.mark.parametrize("backend", BACKENDS)
def test_comparison_preserves_shared_task_and_effective_ppo(backend):
    cfg = config(f"{backend}_comparison")
    inherited = config(
        f"unidr_{backend}", "algo.max_iterations=20000", f"+env.seed={ENV_SEEDS[backend]}"
    )
    shared = config("unidr_single_gpu", "algo.max_iterations=20000")
    assert cfg.training.task_name == "G1FlipTracking"
    assert cfg.training.sim_backend == backend
    assert cfg.training.sim2sim_strict and cfg.training.no_play
    assert not cfg.algo.resume and cfg.algo.load_run == "-1"
    assert cfg.algo.resume_path is None
    assert cfg.env.seed == ENV_SEEDS[backend] and cfg.algo.seed == 1
    assert "unidr" not in cfg  # Native single-backend runner, no four-source services.
    assert OmegaConf.to_container(cfg, resolve=True) == OmegaConf.to_container(
        inherited, resolve=True
    )
    assert BackendAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override() == (
        BackendAdapter(inherited, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    )
    actual = normalize_ppo_train_cfg(algo_config_dict(cfg))
    assert actual == normalize_ppo_train_cfg(algo_config_dict(shared))
    assert extract_contract_snapshot(cfg) == extract_contract_snapshot(shared)
    assert (cfg.algo.num_envs, cfg.algo.max_iterations, cfg.algo.num_steps_per_env) == (
        1024,
        20000,
        24,
    )
    assert cfg.algo.save_interval == 500
    assert (cfg.algo.algorithm.num_learning_epochs, cfg.algo.algorithm.num_mini_batches) == (
        5,
        4,
    )
    assert cfg.env.motrix_disable_self_collision is True
    assert cfg.env.genesis_enable_self_collision is False
    assert cfg.env.commands.motion.params.sampling_mode == "start"
    assert not cfg.env.commands.motion.params.truncate_on_clip_end


@pytest.mark.parametrize("backend", BACKENDS)
def test_comparison_snapshot_strict_mujoco_contract_and_legacy_rejection(backend, tmp_path):
    cfg = config(f"{backend}_comparison")
    snapshot = extract_contract_snapshot(cfg)
    write_run_config_snapshot(
        tmp_path,
        run_metadata={"sim_backend": backend},
        full_cfg=cfg,
        contract_snapshot=snapshot,
    )
    saved = json.loads((tmp_path / "run_config.json").read_text())
    assert saved["config"] == OmegaConf.to_container(cfg, resolve=True)
    assert saved["contract_snapshot"] == snapshot
    mujoco = config("mujoco")
    assert resolve_sim2sim_config(tmp_path, mujoco, strict=True) is mujoco
    legacy = config(f"{backend}_baseline")
    with pytest.raises(CrossBackendIncompatibleError, match="env.actions"):
        resolve_sim2sim_config(tmp_path, legacy, strict=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_comparison_allows_explicit_small_smoke_budget(backend):
    cfg = config(f"{backend}_comparison", "algo.num_envs=2", "algo.max_iterations=2")
    assert (cfg.algo.num_envs, cfg.algo.max_iterations, cfg.algo.num_steps_per_env) == (2, 2, 24)
    assert cfg.algo.algorithm.num_learning_epochs * cfg.algo.algorithm.num_mini_batches == 20
    assert extract_contract_snapshot(cfg) == extract_contract_snapshot(config("unidr_single_gpu"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_prepare_single_4096_5000_without_adaptive_config(backend, tmp_path, monkeypatch):
    monkeypatch.setattr("unilab.training.synchronous.ensure_robot_assets_for_paths", lambda _: None)
    monkeypatch.setattr("unilab.training.synchronous.resolve_motion_files", lambda _: None)
    monkeypatch.setattr("unilab.training.synchronous.get_git_info", lambda _: {})
    monkeypatch.setattr("unilab.training.synchronous.importlib.metadata.version", lambda _: "test")
    run_dir = tmp_path / backend
    manifest = prepare_run(ROOT, run_dir, backend, num_envs=4096, iterations=5000)
    assert manifest["num_envs"] == 4096 and manifest["expected_iterations"] == 5000
    assert "adaptive" not in manifest and "unidr" not in manifest["config"]
    assert json.loads((run_dir / "single_manifest.json").read_text()) == manifest
