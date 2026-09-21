"""Engineering fixtures for fixed-policy holdout; no trained result is claimed."""

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.utils.sim2sim import extract_contract_snapshot
from unilab.visualization import single_reference as holdout

ROOT = Path(__file__).parents[2]


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "task=g1_flip_tracking/unidr_motrix",
                "algo.num_envs=2",
                "algo.max_iterations=2",
            ],
        )
    actor, critic = holdout._models(cfg)
    with torch.no_grad():
        actor.mlp[-1].weight.zero_()
        actor.mlp[-1].bias.fill_(0.25)
    checkpoint = tmp_path / "model_1.pt"
    saved = {"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict()}
    torch.save(saved, checkpoint)
    run = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "contract_snapshot": extract_contract_snapshot(cfg),
    }
    (tmp_path / "run_config.json").write_text(json.dumps(run))
    paths = [
        str(cfg.env.scene.model_file),
        "src/unilab/assets/robots/g1/g1.xml",
        "src/unilab/assets/" + str(cfg.env.commands.motion.params.motion_file),
    ]
    manifest = {
        "sources": [],
        "algorithm": {},
        "assets": {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
    }
    manifest["digest"] = sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode()
    ).hexdigest()
    manifest.update(source="motrix", expected_iterations=2, num_envs=2)
    (tmp_path / "single_manifest.json").write_text(json.dumps(manifest))

    # Native completion evidence is independently tested by uni_rl.single_run_audit.
    def audit(path, *, expected_iterations, num_envs):
        assert path == tmp_path and expected_iterations == num_envs == 2
        return {
            "engineering_fixture": True,
            "final_checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(holdout, "audit_single_run", audit)
    return checkpoint, saved, run


def test_retrospective_selection_is_explicit_and_keeps_native_contract(artifact, monkeypatch):
    from uni_rl.logging import checkpoint_audit

    final, _, _ = artifact
    selected = final.with_name("model_0.pt")
    final.rename(selected)
    with pytest.raises(ValueError, match="explicitly requested final"):
        holdout.preflight(selected, ROOT, expected_iterations=2, expected_num_envs=2)
    audit = {
        "checkpoint_sha256": sha256(selected.read_bytes()).hexdigest(),
        "completed_iterations": 1,
    }

    def check(run, *, checkpoint, expected_iterations, num_envs):
        assert run == selected.parent and checkpoint == selected
        assert expected_iterations == num_envs == 2
        return audit

    monkeypatch.setattr(checkpoint_audit, "audit_single_checkpoint", check)
    plan = holdout.preflight(
        selected, ROOT, expected_iterations=2, expected_num_envs=2, selected_checkpoint=True
    )
    assert plan.metadata["completed_iterations"] == 1
    assert plan.metadata["total_transitions"] == 48 and plan.metadata["optimizer_steps"] == 20
    assert not plan.actor.training and not any(p.requires_grad for p in plan.actor.parameters())
    audit["checkpoint_sha256"] = "bad"
    with pytest.raises(ValueError, match="changed"):
        holdout.preflight(
            selected, ROOT, expected_iterations=2, expected_num_envs=2, selected_checkpoint=True
        )


@pytest.mark.parametrize("source", holdout.SOURCES)
def test_preflight_accepts_all_aligned_sources_and_freezes_models(artifact, source):
    checkpoint, _, run = artifact
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                f"task=g1_flip_tracking/unidr_{source}",
                "algo.num_envs=2",
                "algo.max_iterations=2",
            ],
        )
    run.update(
        config=OmegaConf.to_container(cfg, resolve=True),
        contract_snapshot=extract_contract_snapshot(cfg),
    )
    (checkpoint.parent / "run_config.json").write_text(json.dumps(run))
    manifest_path = checkpoint.parent / "single_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"] = source
    manifest_path.write_text(json.dumps(manifest))
    plan = holdout.preflight(checkpoint, ROOT, expected_iterations=2, expected_num_envs=2)
    assert plan.metadata["source"] == source
    assert not plan.actor.training and not any(p.requires_grad for p in plan.actor.parameters())
    assert plan.config.training.sim_backend == "mujoco"


@pytest.mark.parametrize(
    "damage",
    ["budget", "snapshot", "reward", "scale", "dimensions", "normalizer", "nan", "asset", "audit"],
)
def test_bad_artifact_fails_before_env_and_output(artifact, tmp_path, monkeypatch, damage):
    checkpoint, saved, run = artifact
    monkeypatch.setattr(
        holdout, "registry_env_factory", lambda *_: pytest.fail("env before validation")
    )
    if damage == "budget":
        run["config"]["algo"]["max_iterations"] = 20000
    elif damage == "snapshot":
        run.pop("contract_snapshot")
    elif damage == "reward":
        run["config"]["reward"]["action_rate_l2"]["weight"] = -0.1
    elif damage == "scale":
        run["config"]["env"]["actions"]["joint_pos"]["scale"] = 0.25
        run["contract_snapshot"] = extract_contract_snapshot(OmegaConf.create(run["config"]))
    elif damage == "dimensions":
        saved["actor_state_dict"]["mlp.0.weight"] = saved["actor_state_dict"]["mlp.0.weight"][
            :, :-1
        ]
    elif damage == "normalizer":
        saved["actor_state_dict"].pop("obs_normalizer._mean")
    elif damage == "nan":
        saved["critic_state_dict"]["mlp.0.weight"][0, 0] = float("nan")
    elif damage == "asset":
        path = tmp_path / "single_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["assets"] = {}
        path.write_text(json.dumps(manifest))
    else:

        def reject(*args, **kwargs):
            raise ValueError("Incomplete native training")

        monkeypatch.setattr(holdout, "audit_single_run", reject)
    torch.save(saved, checkpoint)
    (tmp_path / "run_config.json").write_text(json.dumps(run))
    with pytest.raises((ValueError, RuntimeError)):
        holdout.record_reference(
            checkpoint, tmp_path / "out", root=ROOT, expected_iterations=2, expected_num_envs=2
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("failure", [False, True])
def test_terminal_snapshot_reference_phase_and_resource_cleanup(
    artifact, tmp_path, monkeypatch, failure
):
    checkpoint, _, _ = artifact
    phase, actions, closed = np.array([0]), [], []
    obs = {"obs": np.zeros((1, 160), np.float32), "critic": np.zeros((1, 286), np.float32)}

    def reset(**kwargs):
        phase[:] = 0
        return obs, {}

    def step(action):
        actions.append(action.copy())
        phase[:] += 1
        return SimpleNamespace(
            obs=obs,
            reward=np.ones(1),
            terminated=np.array([len(actions) % 13 == 0]),
            truncated=np.zeros(1, bool),
        )

    robot = SimpleNamespace(
        body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        data=SimpleNamespace(
            body_link_quat_w=np.array([[[1, 0, 0, 0]]]),
            body_link_pos_w=np.zeros((1, 2, 3)),
            body_link_lin_vel_w=np.zeros((1, 1, 3)),
        ),
    )
    env = SimpleNamespace(
        set_autoreset=lambda value: None,
        reset=reset,
        step=step,
        close=lambda: closed.append(True),
        action_space=SimpleNamespace(shape=(29,)),
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(time_steps=phase)),
        scene={"robot": robot},
        termination_manager=SimpleNamespace(active_terms=[]),
        play_capabilities=SimpleNamespace(supports_physics_state_playback=True),
        get_physics_state_snapshot=lambda: np.full((1, 72), phase[0]),
    )
    monkeypatch.setattr(holdout, "registry_env_factory", lambda *_: lambda *_: env)
    monkeypatch.setattr(holdout, "_render_models", lambda *_: (None, ["classic", "reference"]))

    def references(model, motion, phases, joints):
        assert phases[12] == 13 and phases[13] == 1
        return np.zeros((1000, 1, 72))

    monkeypatch.setattr(holdout, "_reference_states", references)

    def render(states, paths, **kwargs):
        assert len(states) == 1000 and states[12][0, 0] == 13 and states[13][0, 0] == 1
        assert kwargs["cam_azimuth"] == 90 and kwargs["max_extra_envs"] == 1
        if failure:
            raise RuntimeError("render failed")
        return [np.arange(192, dtype=np.uint8).reshape(8, 8, 3)] * 1000

    monkeypatch.setattr(holdout.render_many, "render_states_get_frames_tracking", render)
    monkeypatch.setattr(
        holdout,
        "write_playback_video",
        lambda path, frames, fps: Path(path).write_bytes(b"fixture"),
    )
    monkeypatch.setattr(holdout, "validate_video", lambda _: {"engineering_fixture": True})
    if failure:
        with pytest.raises(RuntimeError, match="render failed"):
            holdout.record_reference(
                checkpoint, tmp_path / "out", root=ROOT, expected_iterations=2, expected_num_envs=2
            )
        assert not (tmp_path / "out/verification.json").exists()
    else:
        result = holdout.record_reference(
            checkpoint, tmp_path / "out", root=ROOT, expected_iterations=2, expected_num_envs=2
        )
        assert result["terminated"] == 76 and result["actor_normalizer_unchanged"]
    assert closed == [True] and len(actions) == 1000
    assert all(np.allclose(action, 0.25) for action in actions)


def test_real_reference_forward_kinematics_and_visual_only_copy(tmp_path):
    source = ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"
    before = sha256(source.read_bytes()).hexdigest()
    model, paths = holdout._render_models(source, tmp_path)
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, 30)]
    references = holdout._reference_states(
        model, ROOT / "src/unilab/assets/motions/g1/flip_360_001__A304.npz", np.arange(225), joints
    )
    assert references.shape == (225, 1, 72) and np.isfinite(references).all()
    assert all(Path(path).is_file() for path in paths)
    assert sha256(source.read_bytes()).hexdigest() == before
