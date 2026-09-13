"""CPU-only headless evaluation tests with deliberately different reset states."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_obs_groups
from tensordict import TensorDict
from uni_rl.algos.rsl_rl import normalize_ppo_train_cfg

from unilab.tasks.locomotion.g1.evaluation import (
    G1TransitionMetrics,
    apply_g1_evaluation_overrides,
    g1_evaluation_spec,
    g1_metric_provenance,
)
from unilab.training import evaluation
from unilab.utils.sim2sim import CrossBackendIncompatibleError, extract_contract_snapshot

ROOT_DIR = Path(__file__).resolve().parents[2]
_WIDTHS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "joint_pos": 29,
    "joint_vel": 29,
    "actions": 29,
    "command": 3,
    "gait_phase": 2,
    "base_lin_vel": 3,
}


def _config(tmp_path: Path, *, num_envs: int = 4, backend: str = "motrix") -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "src/unilab/conf/ppo"), version_base="1.3"
    ):
        cfg = compose("config", overrides=[f"task=g1_walk_flat/{backend}_multisim"])
    for path, value in {
        "algo.policy.actor_hidden_dims": [8, 6],
        "algo.policy.critic_hidden_dims": [8, 6],
        "algo.num_steps_per_env": 2,
        "algo.load_run": str(tmp_path / "run"),
        "algo.checkpoint": 7,
        "training.play_only": True,
        "training.evaluation.enabled": True,
        "training.evaluation.num_envs": num_envs,
        "training.evaluation.output_dir": "report",
        "env.max_episode_seconds": 0.06,
    }.items():
        OmegaConf.update(cfg, path, value, force_add=True)
    return cfg


def _checkpoint(
    cfg: DictConfig,
    *,
    actor_dim: int = 98,
    critic_dim: int = 101,
    action_dim: int = 29,
) -> Path:
    run_dir = Path(cfg.algo.load_run)
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({"contract_snapshot": extract_contract_snapshot(cfg)}), encoding="utf-8"
    )
    train_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))
    obs = TensorDict(
        {
            "actor": torch.zeros(1, actor_dim),
            "policy": torch.zeros(1, actor_dim),
            "critic": torch.zeros(1, critic_dim),
        },
        batch_size=[1],
    )
    groups = resolve_obs_groups(obs, train_cfg["obs_groups"], ["actor", "critic"])
    checkpoint: dict[str, Any] = {"iter": 7, "infos": {}}
    for name, output_dim in (("actor", action_dim), ("critic", 1)):
        model_cfg = copy.deepcopy(train_cfg[name])
        model_cfg.pop("class_name")
        model = MLPModel(obs, groups, name, output_dim, **model_cfg)
        checkpoint[f"{name}_state_dict"] = model.state_dict()
    path = run_dir / "model_7.pt"
    torch.save(checkpoint, path)
    (run_dir / "model_9.pt").write_bytes(b"not the requested checkpoint")
    return path


class _SyntheticEnv:
    def __init__(self, cfg: DictConfig, *, num_envs: int, **kwargs: Any):
        self.overrides = kwargs.get("env_cfg_override")
        self.cfg = cfg.env
        self.num_envs = num_envs
        self.obs_groups_spec = {"obs": 98, "critic": 101}
        self.action_space = gym.spaces.Box(-1.0, 1.0, (29,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (199,), dtype=np.float32)
        names = list(cfg.env.observations.critic.terms)
        self.observation_manager = SimpleNamespace(
            active_terms={"critic": names},
            group_obs_term_dim={"critic": [(_WIDTHS[name],) for name in names]},
            group_obs_concatenate={"critic": True},
        )
        self.slices = {}
        offset = 0
        for name in names:
            self.slices[name] = slice(offset, offset + _WIDTHS[name])
            offset += _WIDTHS[name]
        self.lengths = np.minimum(np.arange(num_envs) + 1, 3)
        self.steps = np.zeros(num_envs, dtype=int)
        self.seed_calls: list[int] = []
        self.reset_samples: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rng = np.random.default_rng(0)
        self.state: Any = None
        self.closed = False
        self.missing_final = False
        self.compat_final = False
        self.never_done = False
        self.nonfinite = False

    def seed(self, seed: int) -> int:
        self.seed_calls.append(seed)
        self.rng = np.random.default_rng(seed)
        return seed

    def init_state(self) -> Any:
        self.reset()
        return self.state

    def reset(self, env_indices: Any = None, *, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self.seed(seed)
        if env_indices is not None:
            np.testing.assert_array_equal(env_indices, np.arange(self.num_envs))
        self.steps.fill(0)
        self.reset_samples.append(self.rng.uniform(size=self.num_envs))
        critic = np.zeros((self.num_envs, 101), dtype=np.float32)
        critic[:, self.slices["command"]] = [1.0, 0.0, 0.0]
        self.state = SimpleNamespace(
            obs={"obs": np.zeros((self.num_envs, 98), dtype=np.float32), "critic": critic},
            reward=np.zeros(self.num_envs, dtype=np.float32),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=np.zeros(self.num_envs, dtype=bool),
            info={},
            final_observation=None,
        )
        return self.state.obs, {}

    def step(self, actions: np.ndarray) -> Any:
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        self.actions.append(np.asarray(actions).copy())
        self.steps += 1
        critic = self.state.obs["critic"]
        critic[:, self.slices["base_lin_vel"]] = np.column_stack(
            [1.0 + 2.0 * self.steps, -0.5 * self.steps, np.zeros(self.num_envs)]
        )
        critic[:, self.slices["base_ang_vel"]] = np.column_stack(
            [np.zeros(self.num_envs), np.zeros(self.num_envs), 1.5 * self.steps]
        )
        critic[:, self.slices["command"]] = [1.0, 0.0, 0.0]
        done = self.steps >= self.lengths
        if self.never_done:
            done.fill(False)
        indices = np.arange(self.num_envs)
        self.state.terminated = done & (indices % 4 < 3) & (indices % 4 != 1)
        self.state.truncated = done & (indices % 4 != 0)
        final = {"obs": self.state.obs["obs"].copy(), "critic": critic.copy()}
        final["critic"][~done] = np.nan
        final["critic"][done, self.slices["command"]] = [77.0, 0.0, 0.0]
        if self.nonfinite:
            final["critic"][done, self.slices["base_lin_vel"]] = np.nan
        self.state.final_observation = None if self.missing_final or self.compat_final else final
        self.state.info = {"final_observation": final} if self.compat_final else {}
        critic[done] = 1000.0
        self.steps[done] = 0
        return self.state

    def render(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("Headless metrics must not construct or invoke a renderer")

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, env: _SyntheticEnv):
        self.env = env

    def reset(self) -> None:
        self.env.reset()

    def step_once(self) -> None:
        self.env.step(np.zeros((self.env.num_envs, 29)))


@pytest.mark.parametrize("compat_final", [False, True])
def test_terminal_metrics_use_pre_reset_values_and_action_time_command(
    tmp_path: Path, compat_final: bool
):
    env = _SyntheticEnv(_config(tmp_path), num_envs=4)
    env.compat_final = compat_final
    records = evaluation._collect_first_episodes(
        _Session(env),
        G1TransitionMetrics(env),
        seed=101,
        max_steps=3,
        ctrl_dt=0.02,
    )
    assert len(records) == 4
    assert [record["episode_length_steps"] for record in records] == [1, 2, 3, 3]
    np.testing.assert_allclose(
        [record["episode_time_seconds"] for record in records], [0.02, 0.04, 0.06, 0.06]
    )
    np.testing.assert_allclose([record["vx_mae"] for record in records], [2, 3, 4, 4])
    np.testing.assert_allclose([record["abs_vy_mean"] for record in records], [0.5, 0.75, 1, 1])
    np.testing.assert_allclose([record["abs_wz_mean"] for record in records], [1.5, 2.25, 3, 3])
    summary = evaluation._summarize(records)
    assert summary["vx_mae"]["mean"] == 3.25
    assert summary["vx_mae"]["median"] == 3.5
    assert summary["vx_mae"]["p95"] == 4.0
    assert summary["fall_rate"] == 0.5
    assert summary["non_timeout_termination_rate"] == 0.5
    assert summary["full_episode_success_rate"] == 0.25
    assert summary["full_20s_success_rate"] == 0.0


def test_full_20s_success_requires_surviving_all_1000_transitions(tmp_path: Path):
    env = _SyntheticEnv(_config(tmp_path), num_envs=4)
    env.lengths[3] = 1000
    records = evaluation._collect_first_episodes(
        _Session(env),
        G1TransitionMetrics(env),
        seed=101,
        max_steps=1000,
        ctrl_dt=0.02,
    )
    assert records[3]["episode_length_steps"] == 1000
    assert records[3]["episode_time_seconds"] == 20.0
    assert records[3]["vx_mae"] == 1001.0
    assert evaluation._summarize(records)["full_20s_success_rate"] == 0.25


def test_video_snapshots_stop_at_first_env0_episode_before_autoreset(tmp_path: Path):
    env = _SyntheticEnv(_config(tmp_path), num_envs=4)
    env.lengths[0] = 2
    captured_steps = []
    recorder = SimpleNamespace(snapshot=lambda: captured_steps.append(int(env.steps[0])))
    records = evaluation._collect_first_episodes(
        _Session(env),
        G1TransitionMetrics(env),
        seed=101,
        max_steps=3,
        ctrl_dt=0.02,
        recorder=recorder,
    )
    assert captured_steps == [0, 1]
    assert records[0]["episode_length_steps"] == len(captured_steps)
    assert len(records) == 4


def test_repeated_seed_reproduces_initial_samples_and_episode_results(tmp_path: Path):
    env = _SyntheticEnv(_config(tmp_path), num_envs=4)
    session = _Session(env)
    metrics = G1TransitionMetrics(env)
    first = evaluation._collect_first_episodes(
        session, metrics, seed=101, max_steps=3, ctrl_dt=0.02
    )
    second = evaluation._collect_first_episodes(
        session, metrics, seed=102, max_steps=3, ctrl_dt=0.02
    )
    repeated = evaluation._collect_first_episodes(
        session, metrics, seed=101, max_steps=3, ctrl_dt=0.02
    )
    assert first == repeated
    assert second[0]["seed"] == 102
    np.testing.assert_array_equal(env.reset_samples[0], env.reset_samples[2])
    assert not np.array_equal(env.reset_samples[0], env.reset_samples[1])


def test_named_layout_supports_reordered_critic_terms(tmp_path: Path):
    cfg = _config(tmp_path)
    terms = OmegaConf.to_container(cfg.env.observations.critic.terms, resolve=True)
    cfg.env.observations.critic.terms = dict(reversed(list(terms.items())))
    assert g1_evaluation_spec(cfg).obs_groups_spec == {"obs": 98, "critic": 101}
    env = _SyntheticEnv(cfg, num_envs=4)
    metrics = G1TransitionMetrics(env)
    assert metrics.slices["base_lin_vel"] == slice(0, 3)
    records = evaluation._collect_first_episodes(
        _Session(env), metrics, seed=101, max_steps=3, ctrl_dt=0.02
    )
    assert [record["vx_mae"] for record in records] == [2, 3, 4, 4]


@pytest.mark.parametrize("backend", ["motrix", "genesis", "isaacgym", "isaacsim", "mujoco"])
def test_singlebackend_profiles_have_cold_path_dimensions(tmp_path: Path, backend: str):
    spec = g1_evaluation_spec(_config(tmp_path, backend=backend))
    assert spec.obs_groups_spec == {"obs": 98, "critic": 101}
    assert spec.action_dim == 29


@pytest.mark.parametrize("field,value", [("scale", 2), ("clip", [-1, 1]), ("delay_max_lag", 1)])
def test_rejects_processed_critic_metrics(tmp_path: Path, field: str, value: Any):
    cfg = _config(tmp_path)
    OmegaConf.update(
        cfg, f"env.observations.critic.terms.base_lin_vel.{field}", value, force_add=True
    )
    with pytest.raises(ValueError, match="raw, current critic"):
        g1_evaluation_spec(cfg)


def test_overrides_preserve_initial_distribution_and_original_config(tmp_path: Path):
    original = _config(tmp_path)
    cfg = copy.deepcopy(original)
    before = OmegaConf.to_container(original, resolve=True)
    overrides = apply_g1_evaluation_overrides(cfg)
    assert set(overrides) == {
        "env.observations.policy.enable_corruption",
        "env.events.base_mass",
        "env.events.pd_gains",
    }
    assert cfg.env.observations.policy.enable_corruption is False
    assert cfg.env.events.base_mass is None
    assert cfg.env.events.pd_gains is None
    assert cfg.env.events.reset_root_state_uniform == original.env.events.reset_root_state_uniform
    assert cfg.env.events.reset_scene_to_default == original.env.events.reset_scene_to_default
    assert cfg.env.commands == original.env.commands
    assert OmegaConf.to_container(original, resolve=True) == before


@pytest.mark.parametrize("backend", ["motrix", "genesis", "isaacgym", "isaacsim", "mujoco"])
def test_real_policy_loader_reuses_one_env_for_100_seeded_first_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
):
    cfg = _config(tmp_path, num_envs=25, backend=backend)
    cfg.algo.empirical_normalization = True
    checkpoint = _checkpoint(cfg)
    original = OmegaConf.to_container(cfg, resolve=True)
    envs = []

    def factory(*args: Any, **kwargs: Any) -> _SyntheticEnv:
        env = _SyntheticEnv(*args, **kwargs)
        envs.append(env)
        return env

    monkeypatch.setattr(evaluation, "create_env", factory)
    monkeypatch.setattr(
        evaluation, "configure_backend_process_device", lambda backend, device: device
    )
    monkeypatch.setattr(
        evaluation.BackendAdapter,
        "build_play_env_cfg_override",
        lambda self: pytest.fail("visual play profile used"),
    )
    report_path = evaluation.run_ppo_metrics_evaluation(cfg, device="cpu", root_dir=tmp_path)
    assert report_path == tmp_path / "report/metrics.json"
    report = json.loads(report_path.read_text())
    assert len(envs) == 1
    env = envs[0]
    assert env.closed
    assert env.seed_calls == [101, 102, 103, 104]
    for seed, sample in zip(env.seed_calls, env.reset_samples[-4:], strict=True):
        np.testing.assert_array_equal(sample, np.random.default_rng(seed).uniform(size=25))
    assert len(report["episodes"]) == report["summary"]["episode_count"] == 100
    assert len({(episode["seed"], episode["env_index"]) for episode in report["episodes"]}) == 100
    assert all(summary["episode_count"] == 25 for summary in report["per_seed"].values())
    assert report["metadata"]["checkpoint"] == str(checkpoint)
    assert (
        report["metadata"]["checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert (
        report["metadata"]["effective_config"]["env"]["observations"]["policy"]["enable_corruption"]
        is False
    )
    assert report["metadata"]["effective_env_overrides"]["events"]["base_mass"] is None
    assert report["metadata"]["effective_env_overrides"]["events"]["pd_gains"] is None
    assert report["metadata"]["render_mode"] == "none"
    assert report["metadata"]["deterministic_actions"] is True
    provenance = report["metadata"]["metric_provenance"]
    assert provenance["components"]["vx_mae"]["sensor"] == "pelvis_local_linvel"
    assert provenance["components"]["abs_vy_mean"]["frame"] == "imu_in_pelvis local frame"
    assert provenance["components"]["abs_wz_mean"]["sensor"] == "torso_gyro"
    assert provenance["components"]["abs_wz_mean"]["frame"] == "imu_in_torso local frame"
    assert provenance["components"]["abs_wz_mean"]["is_pelvis_angular_velocity"] is False
    assert "waist joints" in provenance["limitations"][0]
    if backend == "isaacsim":
        assert env.overrides["isaacsim_render_mode"] == "none"
    assert (
        env.overrides["events"]["reset_root_state_uniform"]
        == original["env"]["events"]["reset_root_state_uniform"]
    )
    assert OmegaConf.to_container(cfg, resolve=True) == original
    for actions in env.actions[1:]:
        np.testing.assert_array_equal(actions, env.actions[0])
    with (report_path.parent / "episodes.csv").open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 100
    assert float(rows[0]["vx_mae"]) == report["episodes"][0]["vx_mae"]


@pytest.mark.parametrize("dimensions", [{"actor_dim": 99}, {"critic_dim": 102}, {"action_dim": 28}])
def test_checkpoint_dimension_mismatches_fail_before_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dimensions: dict
):
    cfg = _config(tmp_path)
    _checkpoint(cfg, **dimensions)
    monkeypatch.setattr(
        evaluation,
        "create_env",
        lambda *args, **kwargs: pytest.fail("environment constructed before dimension check"),
    )
    with pytest.raises(CrossBackendIncompatibleError, match="dimensions"):
        evaluation.run_ppo_metrics_evaluation(cfg, device="cpu", root_dir=tmp_path)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(
    "field,value",
    [("env.observations.policy.enable_corruption", False), ("env.actions.joint_pos.scale", 0.5)],
)
def test_strict_original_profile_contract_checked_before_overrides_or_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
):
    cfg = _config(tmp_path)
    _checkpoint(cfg)
    OmegaConf.update(cfg, field, value)
    cfg.training.sim2sim_strict = False
    monkeypatch.setattr(
        evaluation,
        "create_env",
        lambda *args, **kwargs: pytest.fail("environment constructed before strict preflight"),
    )
    with pytest.raises(CrossBackendIncompatibleError):
        evaluation.run_ppo_metrics_evaluation(cfg, device="cpu", root_dir=tmp_path)


def test_missing_checkpoint_fails_before_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr(
        evaluation,
        "create_env",
        lambda *args, **kwargs: pytest.fail("environment constructed without checkpoint"),
    )
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        evaluation.run_ppo_metrics_evaluation(cfg, device="cpu", root_dir=tmp_path)


@pytest.mark.parametrize(
    "failure,match",
    [("missing_final", "pre-reset"), ("never_done", "incomplete"), ("nonfinite", "Non-finite")],
)
def test_incomplete_rollout_closes_env_without_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, match: str
):
    cfg = _config(tmp_path)
    _checkpoint(cfg)
    env = _SyntheticEnv(cfg, num_envs=4)
    setattr(env, failure, True)
    monkeypatch.setattr(evaluation, "create_env", lambda *args, **kwargs: env)
    with pytest.raises((ValueError, RuntimeError), match=match):
        evaluation.run_ppo_metrics_evaluation(cfg, device="cpu", root_dir=tmp_path)
    assert env.closed
    assert not (tmp_path / "report").exists()


def test_default_output_is_checkpoint_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = _config(tmp_path)
    cfg.training.evaluation.output_dir = None
    checkpoint = _checkpoint(cfg)
    monkeypatch.setattr(evaluation, "create_env", _SyntheticEnv)
    report_path = evaluation.run_ppo_metrics_evaluation(cfg, device="cpu", root_dir=tmp_path)
    assert report_path.parent.parent == checkpoint.parent / "evaluation"
    assert report_path.is_file()


@pytest.mark.parametrize("render_success", [False, True])
def test_optional_video_uses_evaluated_first_seed_env0_and_checks_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, render_success: bool
):
    from unilab.visualization import playback_session

    cfg = _config(tmp_path, backend="mujoco")
    cfg.training.evaluation.record_video = True
    _checkpoint(cfg)
    env = _SyntheticEnv(cfg, num_envs=4)
    env.get_physics_state_snapshot = lambda: env.steps[:, None].copy()
    monkeypatch.setattr(evaluation, "create_env", lambda *args, **kwargs: env)
    captured = []

    class Recorder:
        def __init__(self, recorded_env: Any, *, frame_state_getter: Any):
            assert recorded_env is env
            self.getter = frame_state_getter

        def __len__(self) -> int:
            return len(captured)

        def snapshot(self) -> None:
            assert env.seed_calls == [101]
            captured.append(self.getter())

        def render_snapshots(self, *, output_video: Path, camera: dict, fps: int) -> str | None:
            assert not env.closed
            assert env.seed_calls == [101, 102, 103, 104]
            assert camera["cam_tracking"] is True
            assert camera["cam_tracking_env_idx"] == camera["cam_tracking_extra_envs"] == 0
            assert fps == 50
            if not render_success:
                return None
            output_video.write_bytes(b"synthetic video")
            return str(output_video)

    monkeypatch.setattr(playback_session, "SnapshotPlaybackSession", Recorder)
    if not render_success:
        with pytest.raises(RuntimeError, match="video rendering failed"):
            evaluation.run_ppo_metrics_evaluation(cfg, device="cpu", root_dir=tmp_path)
        assert env.closed
        assert not (tmp_path / "report/metrics.json").exists()
        return
    report_path = evaluation.run_ppo_metrics_evaluation(cfg, device="cpu", root_dir=tmp_path)
    report = json.loads(report_path.read_text())
    assert env.closed
    assert report["summary"]["episode_count"] == 16
    video = report["metadata"]["video"]
    assert video["seed"] == 101
    assert video["env_index"] == 0
    assert video["frame_count"] == 1
    assert video["duration_seconds"] == 0.02
    assert video["episode"] == report["episodes"][0]
    np.testing.assert_array_equal(captured, np.zeros((1, 1, 1)))
    monkeypatch.setattr(
        evaluation, "create_env", lambda *args, **kwargs: pytest.fail("overwrite must fail early")
    )
    with pytest.raises(FileExistsError, match="overwrite evaluation video"):
        evaluation.run_ppo_metrics_evaluation(cfg, device="cpu", root_dir=tmp_path)


def test_policy_load_maps_device_and_does_not_restore_training_progress():
    from uni_rl.algos.rsl_rl_training_state import TrainingStateOnPolicyRunner

    calls = []
    runner = TrainingStateOnPolicyRunner.__new__(TrainingStateOnPolicyRunner)
    runner.load = lambda path, **options: calls.append((path, options))
    evaluation._load_inference_policy(runner, "model_7.pt", device="cpu")
    assert calls == [
        (
            "model_7.pt",
            {
                "load_cfg": {
                    "actor": True,
                    "critic": False,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
                "map_location": "cpu",
                "strict": True,
                "restore_training_state": False,
            },
        )
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", []),
        ("seeds", [101, 101]),
        ("seeds", [True]),
        ("seeds", [-1]),
        ("num_envs", 0),
        ("record_video", "false"),
    ],
)
def test_invalid_evaluation_options_rejected(tmp_path: Path, field: str, value: Any):
    cfg = _config(tmp_path)
    OmegaConf.update(cfg, f"training.evaluation.{field}", value)
    with pytest.raises(ValueError):
        evaluation.run_ppo_metrics_evaluation(cfg, device="cpu", root_dir=tmp_path)


@pytest.fixture
def backend_reports(tmp_path: Path) -> list[Path]:
    from unilab.training.multi_source import SOURCE_ORDER

    reports = []
    for backend_index, backend in enumerate(SOURCE_ORDER):
        cfg = _config(tmp_path, num_envs=25, backend=backend)
        cfg.env.max_episode_seconds = 20.0
        snapshot = extract_contract_snapshot(cfg)
        overrides = apply_g1_evaluation_overrides(cfg)
        cfg.algo.num_envs = 25
        episodes = []
        metric_value = (1.0, 2.0, 4.0, 100.0)[backend_index]
        for seed in (101, 102, 103, 104):
            for env_index in range(25):
                success = backend_index % 2 == 0
                length = 1000 if success else env_index + 1
                episodes.append(
                    {
                        "seed": seed,
                        "env_index": env_index,
                        "episode_length_steps": length,
                        "episode_time_seconds": length * 0.02,
                        "terminated": not success,
                        "truncated": success,
                        "full_episode_success": success,
                        "full_20s_success": success,
                        "vx_mae": metric_value,
                        "abs_vy_mean": metric_value / 2,
                        "abs_wz_mean": metric_value / 4,
                    }
                )
        report = {
            "schema_version": 1,
            "metadata": {
                "backend": backend,
                "seeds": [101, 102, 103, 104],
                "num_envs": 25,
                "expected_episode_count": 100,
                "ctrl_dt": 0.02,
                "episode_horizon_seconds": 20.0,
                "max_episode_steps": 1000,
                "deterministic_actions": True,
                "render_mode": "none",
                "sim2sim_strict": True,
                "episode_selection": "first_completed_episode_per_seed_and_env",
                "seed_strategy": "one_environment_public_seed_then_full_reset",
                "checkpoint_sha256": "a" * 64,
                "contract_snapshot": snapshot,
                "effective_config": OmegaConf.to_container(cfg, resolve=True),
                "evaluation_overrides": overrides,
                "metric_source": "named raw critic terms; final_observation on terminal transitions",
                "metric_provenance": g1_metric_provenance(),
                "vx_reference": "commanded vx at action selection, before command resampling",
                "units": {"vx_mae": "m/s", "abs_vy_mean": "m/s", "abs_wz_mean": "rad/s"},
            },
            "episodes": episodes,
            "summary": evaluation._summarize(episodes),
        }
        path = tmp_path / backend / "metrics.json"
        path.parent.mkdir()
        path.write_text(json.dumps(report), encoding="utf-8")
        reports.append(path)
    return reports


def test_aggregate_pools_400_episodes_instead_of_averaging_quantiles(
    backend_reports: list[Path],
    tmp_path: Path,
):
    from unilab.training.multi_source import SOURCE_ORDER

    path = evaluation.aggregate_ppo_metrics_reports(
        backend_reports, output_path=tmp_path / "overall.json"
    )
    report = json.loads(path.read_text())
    assert report["summary"]["episode_count"] == 400
    assert report["summary"]["vx_mae"]["mean"] == 26.75
    assert report["summary"]["vx_mae"]["median"] == 3.0
    assert report["summary"]["vx_mae"]["p95"] == 100.0
    assert report["summary"]["fall_rate"] == 0.5
    assert report["summary"]["full_20s_success_rate"] == 0.5
    assert list(report["per_backend"]) == list(SOURCE_ORDER)
    assert all(summary["episode_count"] == 100 for summary in report["per_backend"].values())
    assert (
        len(
            {
                (episode["backend"], episode["seed"], episode["env_index"])
                for episode in report["episodes"]
            }
        )
        == 400
    )
    assert report["metadata"]["checkpoint_sha256"] == "a" * 64
    assert report["metadata"]["convergence_assessed"] is False
    assert "not evidence" in report["metadata"]["interpretation"]


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "unordered"])
def test_aggregate_requires_exact_real_backend_source_order(
    backend_reports: list[Path],
    tmp_path: Path,
    invalid: str,
):
    if invalid == "missing":
        backend_reports.pop()
    elif invalid == "duplicate":
        backend_reports[-1] = backend_reports[0]
    else:
        backend_reports.reverse()
    with pytest.raises(ValueError, match="SOURCE_ORDER"):
        evaluation.aggregate_ppo_metrics_reports(
            backend_reports, output_path=tmp_path / "overall.json"
        )
    assert not (tmp_path / "overall.json").exists()


@pytest.mark.parametrize(
    "invalid",
    [
        "count",
        "seed",
        "duplicate_episode",
        "checkpoint",
        "source_contract",
        "effective_contract",
        "initial_distribution",
        "seeds_metadata",
        "horizon",
        "nan",
        "success",
        "reset_dr",
        "units",
        "sensor_frame",
    ],
)
def test_aggregate_rejects_incomparable_or_incomplete_reports(
    backend_reports: list[Path],
    tmp_path: Path,
    invalid: str,
):
    path = backend_reports[-1]
    report = json.loads(path.read_text())
    metadata = report["metadata"]
    episode = report["episodes"][0]
    if invalid == "count":
        report["episodes"].pop()
    elif invalid == "seed":
        episode["seed"] = 105
    elif invalid == "duplicate_episode":
        report["episodes"][1] = episode.copy()
    elif invalid == "checkpoint":
        metadata["checkpoint_sha256"] = "b" * 64
    elif invalid == "source_contract":
        metadata["contract_snapshot"]["env.actions"]["joint_pos"]["scale"] = 0.5
    elif invalid == "effective_contract":
        metadata["effective_config"]["env"]["actions"]["joint_pos"]["scale"] = 0.5
    elif invalid == "initial_distribution":
        metadata["effective_config"]["env"]["events"]["reset_root_state_uniform"]["params"][
            "pose_range"
        ]["yaw"] = [0, 0]
    elif invalid == "seeds_metadata":
        metadata["seeds"] = [1, 2, 3, 4]
    elif invalid == "horizon":
        metadata["episode_horizon_seconds"] = 10.0
    elif invalid == "nan":
        episode["vx_mae"] = float("nan")
    elif invalid == "success":
        episode["full_20s_success"] = True
    elif invalid == "reset_dr":
        metadata["effective_config"]["env"]["events"]["base_mass"] = {"mode": "reset"}
    elif invalid == "units":
        metadata["units"]["vx_mae"] = "km/h"
    elif invalid == "sensor_frame":
        metadata["metric_provenance"]["components"]["abs_wz_mean"]["frame"] = "pelvis root frame"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.aggregate_ppo_metrics_reports(
            backend_reports, output_path=tmp_path / "overall.json"
        )
    assert not (tmp_path / "overall.json").exists()
