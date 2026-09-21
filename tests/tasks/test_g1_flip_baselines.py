"""Config guards and opt-in real-physics PPO baseline validation."""

import json
import os
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner
from scripts.validate_g1_flip import main as validate_main
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from uni_rl.algos.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime

from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.tasks.motion_tracking.g1.validation import validate_flip_episode
from unilab.training import algo_config_dict
from unilab.training.experiment import write_run_config_snapshot
from unilab.utils.sim2sim import (
    DENYLIST,
    CrossBackendIncompatibleError,
    extract_contract_snapshot,
    resolve_sim2sim_config,
)

ROOT = Path(__file__).parents[2]
BACKENDS = ("motrix", "isaacsim", "isaacgym", "genesis")


def config(backend, *, baseline=True):
    owner = f"{backend}_baseline" if baseline else backend
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task=g1_flip_tracking/{owner}"])


def test_native_motrix_matches_mujoco_flip_training_profile():
    motrix = config("motrix", baseline=False)
    mujoco = config("mujoco", baseline=False)
    for section in ("env", "reward"):
        assert OmegaConf.to_container(motrix[section], resolve=True) == OmegaConf.to_container(
            mujoco[section], resolve=True
        )
    assert motrix.training.sim_backend == "motrix"
    assert motrix.training.task_name == "G1FlipTracking"
    assert motrix.algo.num_envs == 1024
    assert motrix.algo.max_iterations == 30000
    assert motrix.algo.empirical_normalization is True
    actual = normalize_ppo_train_cfg(algo_config_dict(motrix))
    expected = normalize_ppo_train_cfg(algo_config_dict(mujoco))
    for model in ("actor", "critic"):
        assert actual[model] == expected[model]
        assert actual[model]["obs_normalization"] is True


@pytest.mark.parametrize(
    ("field", "old_value"),
    (("env.actions.joint_pos.scale", 0.25), ("algo.empirical_normalization", False)),
)
def test_native_motrix_rejects_legacy_policy_contract(tmp_path, field, old_value):
    old = config("motrix", baseline=False)
    OmegaConf.update(old, field, old_value, merge=False)
    (tmp_path / "run_config.json").write_text(
        json.dumps({"contract_snapshot": extract_contract_snapshot(old)})
    )
    with pytest.raises(CrossBackendIncompatibleError) as exc:
        resolve_sim2sim_config(tmp_path, config("motrix", baseline=False), strict=True)
    assert ("env.actions" if field.startswith("env.actions") else field) in str(exc.value)


def test_baselines_share_flip_policy_contract():
    reference = extract_contract_snapshot(config("motrix"))
    for backend in BACKENDS:
        cfg = config(backend)
        snapshot = extract_contract_snapshot(cfg)
        assert {k: v for k, v in snapshot.items() if k in DENYLIST} == {
            k: v for k, v in reference.items() if k in DENYLIST
        }
        assert cfg.training.sim_backend == backend
        assert cfg.training.task_name == "G1FlipTracking"
        assert cfg.env.commands.motion.params.motion_file == "motions/g1/flip_360_001__A304.npz"
        assert cfg.env.commands.motion.params.truncate_on_clip_end
        assert cfg.env.terminations.motion_clip_end.time_out
        assert cfg.env.terminations.undesired_contacts is None
        assert not cfg.algo.algorithm.disable_finite_checks
        assert cfg.env.sim_dt == 0.005 and cfg.env.ctrl_dt == 0.02
        assert len(cfg.env.scene.entities.robot.joint_names) == 29
        assert cfg.env.actions.joint_pos.scale == 0.25
        assert cfg.algo.empirical_normalization is False
        for term in ("motion_global_root_pos", "motion_body_pos", "motion_body_ori"):
            assert cfg.reward[term].weight == 1.0
        assert cfg.reward.action_rate_l2.weight == -0.05
        assert cfg.reward.motion_ee_body_pos_z is None
        assert cfg.reward.undesired_contacts is None


def test_validation_stops_at_first_done_and_pads_failures():
    motion = SimpleNamespace(
        cfg=SimpleNamespace(
            params=SimpleNamespace(sampling_mode="start", truncate_on_clip_end=True)
        ),
        motion=SimpleNamespace(num_frames=3, num_clips=1, fps=50),
        body_pos_w=np.zeros((2, 1, 3)),
        robot_body_pos_w=np.zeros((2, 1, 3)),
        time_steps=np.array([0, 0]),
        sampler=SimpleNamespace(current_clip_end_frames=np.array([2, 2])),
    )
    step = 0
    pending = False

    def reset(_rows=None, **_kwargs):
        nonlocal pending
        pending = False

    def advance(_actions):
        nonlocal step, pending
        assert not pending, "done rows require manual reset before another step"
        pending = True
        step += 1
        motion.time_steps[:] = step - 1
        motion.robot_body_pos_w[:, :, 0] = 0.1
        return SimpleNamespace(
            obs={"obs": np.zeros((2, 1))},
            reward=np.array([1.0, 1.0]),
            terminated=np.array([True, False]),
            truncated=np.array([False, step == 3]),
        )

    env = SimpleNamespace(
        num_envs=2,
        step_dt=0.02,
        set_autoreset=lambda flag: None,
        reset=reset,
        command_manager=SimpleNamespace(get_term=lambda name: motion),
        termination_manager=SimpleNamespace(active_terms=[]),
        step=advance,
    )
    result = validate_flip_episode(env, lambda: np.zeros((2, 1)))
    assert result["mean_steps"] == 2
    assert result["mean_return_until_done"] == 2
    assert result["clip_completion_rate"] == 0.5
    assert result["observed_mean_body_error_m"] == pytest.approx(0.1)
    assert result["capped_horizon_body_error_m"] == pytest.approx((1.1 + 0.3) / 6)
    motion.robot_body_pos_w[:] = np.nan
    motion.time_steps[:] = 0
    with pytest.raises(ValueError, match="Reference reset"):
        validate_flip_episode(env, lambda: np.zeros((2, 1)))


@pytest.mark.parametrize("destination", ["checkpoint", "manifest", "hardlink"])
def test_validation_cannot_overwrite_training_artifacts(tmp_path, monkeypatch, destination):
    checkpoint = tmp_path / "model.pt"
    manifest = tmp_path / "run_config.json"
    checkpoint.write_bytes(b"preserve checkpoint")
    manifest.write_text("{}")
    output = checkpoint if destination == "checkpoint" else manifest
    if destination == "hardlink":
        output = tmp_path / "alias.json"
        output.hardlink_to(checkpoint)
    monkeypatch.setattr(sys, "argv", ["validate", str(checkpoint), "--output", str(output)])
    with pytest.raises(ValueError, match="separate JSON"):
        validate_main()
    assert checkpoint.read_bytes() == b"preserve checkpoint" and manifest.read_text() == "{}"


def test_validation_excludes_holdout_before_environment_creation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "contract_snapshot": {"test": True},
                "config": {"training": {"sim_backend": "mujoco"}},
            }
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["validate", str(checkpoint), "--output", str(tmp_path / "eval.json")]
    )
    with pytest.raises(ValueError, match="excludes the MuJoCo holdout"):
        validate_main()


@pytest.mark.parametrize("snapshot", [None, [], [1], {"algo.seed": 1}])
def test_validation_rejects_incomplete_contract_before_env(tmp_path, monkeypatch, snapshot):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "contract_snapshot": snapshot,
                "config": OmegaConf.to_container(config("motrix"), resolve=True),
            }
        )
    )

    def unexpected_env(*_args, **_kwargs):
        pytest.fail("An incomplete training contract must fail before environment construction")

    monkeypatch.setattr("scripts.validate_g1_flip.registry_env_factory", unexpected_env)
    monkeypatch.setattr(
        sys, "argv", ["validate", str(checkpoint), "--output", str(tmp_path / "eval.json")]
    )
    with pytest.raises(ValueError, match="complete training contract snapshot"):
        validate_main()


@pytest.mark.slow
def test_real_flip_ppo_update_checkpoint_and_validation(tmp_path):
    backend = os.environ.get("UNIDR_REAL_BACKEND")
    if backend not in BACKENDS:
        pytest.skip("Set UNIDR_REAL_BACKEND to opt into a real single-backend training smoke")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "from tests.tasks.test_g1_flip_baselines import _train_and_validate; "
            "_train_and_validate(Path(sys.argv[1]), sys.argv[2])",
            str(tmp_path),
            backend,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _train_and_validate(tmp_path, backend):
    torch.set_num_threads(4)
    torch.manual_seed(1)
    cfg = config(backend)
    cfg.algo.num_envs = 2
    cfg.algo.save_interval = 1
    cfg.algo.max_iterations = 2
    cfg.training.device = "cpu"
    override = BackendAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    factory = registry_env_factory("G1FlipTracking", backend)
    rl_cfg = algo_config_dict(cfg)
    runtime = resolve_rsl_rl_ppo_runtime(rl_cfg, default_wrapper_cls=RslRlVecEnvWrapper)
    runner_cls = runtime.runner_cls or OnPolicyRunner

    def runner_for(env, log_dir):
        return runner_cls(
            runtime.wrapper_cls(env, device="cpu"),
            normalize_ppo_train_cfg(rl_cfg),
            log_dir=log_dir,
            device="cpu",
        )

    with closing(factory(2, override)) as env:
        runner = runner_for(env, str(tmp_path))
        before = [
            [p.detach().clone() for p in model.parameters()]
            for model in (runner.alg.actor, runner.alg.critic)
        ]
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
        for old, model in zip(before, (runner.alg.actor, runner.alg.critic), strict=True):
            assert any(not torch.equal(x, y) for x, y in zip(old, model.parameters(), strict=True))
        assert {int(s["step"].item()) for s in runner.alg.optimizer.state.values()} == {40}
        assert runner.logger.tot_timesteps == 2 * 2 * 24
        trained = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    checkpoint = tmp_path / "model_1.pt"
    assert checkpoint.is_file()
    write_run_config_snapshot(
        tmp_path,
        run_metadata={"backend": backend},
        full_cfg=cfg,
        contract_snapshot=extract_contract_snapshot(cfg),
    )
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert all(torch.equal(v, saved["actor_state_dict"][k]) for k, v in trained.items())
    # Genesis permits one scene lifetime per process. Reload validation always
    # uses a fresh process, as does the standalone same-backend entrypoint.
    output = tmp_path / "validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_g1_flip.py"),
            str(checkpoint),
            "--num-envs",
            "2",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    assert result["horizon_steps"] == 225 and result["mean_steps"] > 0
