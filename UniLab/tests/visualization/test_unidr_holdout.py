"""Engineering fixtures only: no trained checkpoint or MuJoCo environment is used."""

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.utils.sim2sim import extract_contract_snapshot
from unilab.visualization import unidr_holdout as holdout

ROOT = Path(__file__).parents[2]


@pytest.fixture
def artifact(tmp_path):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=g1_flip_tracking/unidr_single_gpu"])
    actor, critic = holdout._models(cfg)
    with torch.no_grad():
        actor.mlp[-1].weight.zero_()
        actor.mlp[-1].bias.fill_(0.25)
        for model in (actor, critic):
            model.obs_normalizer.count.fill_(4096 * 24 * 10000)
    params = [parameter for model in (actor, critic) for parameter in model.parameters()]
    saved = {
        "iter": 9999,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": {
            "param_groups": [{"params": list(range(len(params))), "lr": 1.0e-5}],
            "state": {
                index: {
                    "step": torch.tensor(200000.0),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
                for index, parameter in enumerate(params)
            },
        },
        "synchronous": {
            "format": 1,
            "complete": True,
            "next_iteration": 10000,
            "policy_version": 10000,
            "normalizer_version": 240000,
            "optimizer_steps": 200000,
            "learning_rate": 1.0e-5,
            "total_transitions": 4096 * 24 * 10000,
            "contract": {
                "num_envs": 4096,
                "sources": [
                    (name, i * 1024, (i + 1) * 1024) for i, name in enumerate(holdout.SOURCE_ORDER)
                ],
            },
        },
    }
    run = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "contract_snapshot": extract_contract_snapshot(cfg),
        "run": {"mode": "synchronous_four_source", "manifest_digest": "engineering-fixture"},
    }
    asset_paths = [
        cfg.env.scene.model_file,
        "src/unilab/assets/" + cfg.env.commands.motion.params.motion_file,
    ]
    manifest = {
        "sources": [{"source": name} for name in holdout.SOURCE_ORDER],
        "algorithm": OmegaConf.to_container(cfg.algo, resolve=True),
        "assets": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in asset_paths},
    }
    digest = sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()
    manifest["digest"] = digest
    saved["synchronous"]["contract"]["manifest_digest"] = run["run"]["manifest_digest"] = digest
    saved["synchronous"]["contract"]["train_cfg"] = manifest["algorithm"]
    checkpoint = tmp_path / "model_9999.pt"
    torch.save(saved, checkpoint)
    (tmp_path / "run_config.json").write_text(json.dumps(run))
    (tmp_path / "sources_manifest.json").write_text(json.dumps(manifest))
    return checkpoint, saved, run


@pytest.mark.parametrize("legacy", [False, True])
def test_source_collision_controls_preserve_native_holdout(artifact, legacy):
    checkpoint, _, run = artifact
    fields = ("motrix_disable_self_collision", "genesis_enable_self_collision")
    if legacy:
        for name in fields:
            run["config"]["env"].pop(name)
        (checkpoint.parent / "run_config.json").write_text(json.dumps(run))
    before = (checkpoint.parent / "run_config.json").read_bytes()
    plan = holdout.preflight(checkpoint, ROOT)
    for name in fields:
        assert OmegaConf.select(plan.config, f"env.{name}") is None
    assert (checkpoint.parent / "run_config.json").read_bytes() == before


@pytest.mark.parametrize("damage", [None, "default", "name", "counter", "adam", "intent"])
def test_explicit_shortened_budget_preserves_original_training_config(artifact, damage):
    checkpoint, saved, run = artifact
    selected = 5001
    checkpoint = checkpoint.with_name("model_5000.pt")
    saved["iter"] = selected - 1
    sync = saved["synchronous"]
    sync.update(
        next_iteration=selected,
        policy_version=selected,
        normalizer_version=selected * 24,
        optimizer_steps=selected * 20,
        total_transitions=selected * 98304,
    )
    for name in ("actor_state_dict", "critic_state_dict"):
        saved[name]["obs_normalizer.count"].fill_(selected * 98304)
    for state in saved["optimizer_state_dict"]["state"].values():
        state["step"].fill_(selected * 20)
    if damage == "name":
        checkpoint = checkpoint.with_name("model_4999.pt")
    elif damage == "counter":
        sync["total_transitions"] -= 98304
    elif damage == "adam":
        saved["optimizer_state_dict"]["state"][0]["step"] -= 20
    elif damage == "intent":
        run["config"]["algo"]["max_iterations"] = 5000
        (checkpoint.parent / "run_config.json").write_text(json.dumps(run))
    torch.save(saved, checkpoint)
    before = (checkpoint.parent / "run_config.json").read_bytes()
    if damage:
        with pytest.raises(ValueError):
            holdout.preflight(
                checkpoint, ROOT, expected_iterations=10000 if damage == "default" else selected
            )
    else:
        plan = holdout.preflight(checkpoint, ROOT, expected_iterations=selected)
        assert plan.metadata["expected_iterations"] == selected
        assert plan.metadata["planned_iterations"] == 10000
        assert plan.metadata["total_transitions"] == 491618304
        assert not plan.actor.training
    assert (checkpoint.parent / "run_config.json").read_bytes() == before


@pytest.mark.parametrize("invalid", [0, -1, 10001, True, 5001.0])
def test_rejects_invalid_shortened_budget_before_checkpoint_access(tmp_path, invalid):
    with pytest.raises(ValueError, match="Expected iterations"):
        holdout.preflight(tmp_path / "missing.pt", ROOT, expected_iterations=invalid)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_checkpoint",
        "wrong_name",
        "missing_config",
        "missing_manifest",
        "missing_snapshot",
        "wrong_iteration",
        "incomplete_update",
        "wrong_manifest",
        "manifest_tamper",
        "wrong_pool_size",
        "wrong_source_slice",
        "adam_step",
        "missing_adam_state",
        "missing_actor",
        "actor_input",
        "critic_input",
        "action_output",
        "missing_normalizer",
        "normalizer_count",
        "nonfinite",
        "action_scale",
        "reward",
        "missing_model",
    ],
)
def test_rejects_bad_artifact_before_env_or_output(artifact, tmp_path, monkeypatch, damage):
    checkpoint, saved, run = artifact
    output = tmp_path / "evidence"
    monkeypatch.setattr(
        holdout, "registry_env_factory", lambda *_: pytest.fail("env before preflight")
    )
    if damage == "missing_checkpoint":
        checkpoint.unlink()
    elif damage == "wrong_name":
        checkpoint = checkpoint.rename(tmp_path / "model_9500.pt")
    elif damage in {"missing_config", "missing_manifest"}:
        (
            tmp_path
            / ("run_config.json" if damage == "missing_config" else "sources_manifest.json")
        ).unlink()
    elif damage == "missing_snapshot":
        run.pop("contract_snapshot")
    elif damage == "wrong_iteration":
        saved["iter"] = 9500
    elif damage == "incomplete_update":
        saved["synchronous"]["complete"] = False
    elif damage == "wrong_manifest":
        saved["synchronous"]["contract"]["manifest_digest"] = "different"
    elif damage == "manifest_tamper":
        path = tmp_path / "sources_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["sources"][0]["source"] = "substitute"
        path.write_text(json.dumps(manifest))
    elif damage == "wrong_pool_size":
        saved["synchronous"]["contract"]["num_envs"] = 8
        saved["synchronous"]["total_transitions"] = 8 * 24 * 10000
        for name in ("actor_state_dict", "critic_state_dict"):
            saved[name]["obs_normalizer.count"].fill_(8 * 24 * 10000)
    elif damage == "wrong_source_slice":
        saved["synchronous"]["contract"]["sources"][0] = ("isaacsim", 0, 1023)
    elif damage == "adam_step":
        saved["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(199980.0)
    elif damage == "missing_adam_state":
        saved["optimizer_state_dict"]["state"].pop(0)
    elif damage == "missing_actor":
        saved.pop("actor_state_dict")
    elif damage in {"actor_input", "critic_input"}:
        model = damage.split("_", 1)[0] + "_state_dict"
        saved[model]["mlp.0.weight"] = saved[model]["mlp.0.weight"][:, :-1]
    elif damage == "action_output":
        saved["actor_state_dict"]["mlp.6.weight"] = saved["actor_state_dict"]["mlp.6.weight"][:-1]
    elif damage == "missing_normalizer":
        saved["actor_state_dict"].pop("obs_normalizer._mean")
    elif damage == "normalizer_count":
        saved["actor_state_dict"]["obs_normalizer.count"] = torch.tensor(1)
    elif damage == "nonfinite":
        saved["actor_state_dict"]["mlp.0.weight"][0, 0] = float("nan")
    elif damage == "action_scale":
        run["config"]["env"]["actions"]["joint_pos"]["scale"] = 0.25
        run["contract_snapshot"] = extract_contract_snapshot(OmegaConf.create(run["config"]))
    elif damage == "reward":
        run["config"]["reward"]["action_rate_l2"]["weight"] = -0.05
    elif damage == "missing_model":
        resolve = Path.resolve

        def missing_model(path, *args, **kwargs):
            if path.name == "scene_flat.xml":
                raise FileNotFoundError(path)
            return resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", missing_model)
    if damage not in {"missing_checkpoint", "wrong_name"}:
        torch.save(saved, checkpoint)
    if damage != "missing_config":
        (tmp_path / "run_config.json").write_text(json.dumps(run))
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        holdout.record_holdout(checkpoint, output, root=ROOT)
    assert not output.exists()


def observation(telemetry, **kwargs):
    values = dict(
        frame=180,
        next_frame=181,
        upright=1.0,
        height=0.8,
        vertical_speed=0.0,
        feet_z=[0.04, 0.04],
        terminated=False,
        truncated=False,
        step=len(telemetry.rows),
    )
    return telemetry.observe(**(values | kwargs))


def test_landing_requires_airborne_inversion_and_sustained_upright_feet():
    telemetry = holdout.PhaseTelemetry()
    for _ in range(15):
        assert not observation(telemetry)["landing_candidate"]
    observation(telemetry, frame=124, upright=-1.0, feet_z=[0.8, 0.8])
    for _ in range(9):
        assert not observation(telemetry)["landing_candidate"]
    assert observation(telemetry)["landing_candidate"]
    assert not observation(telemetry)["landing_candidate"]


@pytest.mark.parametrize(
    "boundary", [{"terminated": True}, {"truncated": True}, {"frame": 224, "next_frame": 0}]
)
def test_reset_and_clip_wrap_never_count_as_landing(boundary):
    telemetry = holdout.PhaseTelemetry()
    observation(telemetry, frame=124, upright=-1.0, feet_z=[0.8, 0.8])
    row = observation(telemetry, **boundary)
    assert not row["landing_candidate"]
    assert row["reset"] or row["clip_wrap"]
    for _ in range(12):
        assert not observation(telemetry)["landing_candidate"]


@pytest.mark.parametrize("render_failure", [False, True])
def test_recording_uses_loaded_policy_and_preserves_terminal_frames(
    artifact, tmp_path, monkeypatch, render_failure
):
    checkpoint, _, _ = artifact
    physics = SimpleNamespace(
        body_link_quat_w=np.array([[[1, 0, 0, 0]]]),
        body_link_pos_w=np.zeros((1, 3, 3)),
        body_link_lin_vel_w=np.zeros((1, 1, 3)),
    )
    motion = SimpleNamespace(time_steps=np.array([0]))
    actions_seen, captured = [], []
    obs = {"obs": np.zeros((1, 160), np.float32), "critic": np.zeros((1, 286), np.float32)}
    reset_count = 0
    closed = False

    def reset(**_kwargs):
        nonlocal reset_count
        reset_count += 1
        motion.time_steps[:] = 0
        return obs, {}

    def step(actions):
        actions_seen.append(actions.copy())
        motion.time_steps += 1
        return SimpleNamespace(
            obs=obs,
            reward=np.ones(1),
            terminated=np.array([len(actions_seen) % 13 == 0]),
            truncated=np.zeros(1, bool),
        )

    def close():
        nonlocal closed
        closed = True

    env = SimpleNamespace(
        set_autoreset=lambda value: None,
        reset=reset,
        step=step,
        close=close,
        action_space=SimpleNamespace(shape=(29,)),
        command_manager=SimpleNamespace(get_term=lambda name: motion),
        scene={
            "robot": SimpleNamespace(
                data=physics, body_names=("pelvis", "left_ankle_roll_link", "right_ankle_roll_link")
            )
        },
        termination_manager=SimpleNamespace(active_terms=[]),
    )

    class Recorder:
        def __init__(self, actual_env, **kwargs):
            assert actual_env is env and kwargs == {
                "width": 1280,
                "height": 720,
                "num_processes": 2,
            }

        def snapshot(self):
            captured.append(int(motion.time_steps[0]))

        def render_snapshots(self, *, output_video, fps, camera):
            assert fps == 50 and camera.cam_tracking
            assert (camera.cam_distance, camera.cam_elevation, camera.cam_azimuth) == (3.2, -12, 0)
            if render_failure:
                return None
            output_video.write_bytes(b"engineering video fixture")
            return str(output_video)

    def make_env(_num_envs, override):
        assert override["seed"] == 1
        return env

    monkeypatch.setattr(holdout, "registry_env_factory", lambda *_: make_env)
    monkeypatch.setattr(holdout, "SnapshotPlaybackSession", Recorder)
    monkeypatch.setattr(holdout, "validate_video", lambda path: {"frames": 1000})
    if render_failure:
        with pytest.raises(RuntimeError, match="did not produce a video"):
            holdout.record_holdout(checkpoint, tmp_path / "evidence", root=ROOT)
        assert closed and not (tmp_path / "evidence/holdout.json").exists()
        return
    result = holdout.record_holdout(checkpoint, tmp_path / "evidence", root=ROOT)
    assert closed and len(actions_seen) == len(captured) == 1000
    assert all(np.allclose(action, 0.25) for action in actions_seen)
    assert captured[12] == 13 and captured[13] == 1
    assert reset_count == 77 and result["resets"] == 76
    assert result["inverted_frames"] == result["landing_candidates"] == 0
    assert len((tmp_path / "evidence/telemetry.jsonl").read_text().splitlines()) == 1000
    assert json.loads((tmp_path / "evidence/mujoco_config.json").read_text())["env"]["seed"] == 1


def test_video_validation_fully_decodes_real_encoding(tmp_path):
    path = tmp_path / "encoding-fixture.mp4"
    subprocess.run(
        [
            holdout.imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=1280x720:r=50",
            "-frames:v",
            "1000",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "2",
            str(path),
        ],
        check=True,
        timeout=60,
    )
    assert holdout.validate_video(path) == {
        "frames": 1000,
        "width": 1280,
        "height": 720,
        "fps": 50,
        "seconds": 20.0,
    }
