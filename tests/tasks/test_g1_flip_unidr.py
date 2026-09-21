"""Aligned four-source policy contract and opt-in reference-phase integrity."""

import json
import os
import xml.etree.ElementTree as ET
from contextlib import closing
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from unisim.utils.rotation import np_quat_apply_inverse

from unilab.base.backend_factory import env_backend_kwargs
from unilab.base.base import EnvCfg
from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.utils.sim2sim import (
    CrossBackendIncompatibleError,
    extract_contract_snapshot,
    resolve_sim2sim_config,
)

ROOT = Path(__file__).parents[2]
SOURCES = ("isaacsim", "isaacgym", "genesis", "motrix")
BACKEND_FIELDS = {
    "isaacsim_device_id",
    "isaacsim_worker_timeout_s",
    "isaacgym_device_id",
    "genesis_device_id",
    "genesis_integrator",
    "motrix_disable_self_collision",
    "genesis_enable_self_collision",
}
# Independently recorded from the completed run_config.json, not composed YAML.
# See docs/unidr-four-source-config.md for provenance and normalization rules.
SUCCESSFUL_PROFILE_SHA256 = "558789835ea6a0025b5b2508303b27d4d4f0dc72efaf96f860aa2094e2ebabde"


def config(owner, *overrides):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task=g1_flip_tracking/{owner}", *overrides])


def semantic_profile(cfg):
    plain = OmegaConf.to_container(cfg, resolve=True)
    profile = {key: plain[key] for key in ("env", "reward", "algo")}
    for key in BACKEND_FIELDS:
        profile["env"].pop(key, None)
    profile["env"]["scene"]["entities"]["robot"].pop("geom_names", None)
    # Hydra's native owner uses a string; the saved CLI override was numeric.
    profile["algo"]["load_run"] = str(profile["algo"]["load_run"])
    return profile


@pytest.mark.parametrize("backend", SOURCES)
def test_aligned_source_matches_successful_semantic_profile(backend, tmp_path):
    cfg = config(f"unidr_{backend}")
    profile = semantic_profile(cfg)
    encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    assert sha256(encoded).hexdigest() == SUCCESSFUL_PROFILE_SHA256
    reference = config("motrix", "algo.max_iterations=10000")
    assert profile == semantic_profile(reference)
    (tmp_path / "run_config.json").write_text(
        json.dumps({"contract_snapshot": extract_contract_snapshot(reference)})
    )
    assert resolve_sim2sim_config(tmp_path, cfg, strict=True) is cfg
    assert extract_contract_snapshot(cfg) == extract_contract_snapshot(reference)
    assert cfg.training.task_name == "G1FlipTracking"
    assert cfg.training.sim_backend == backend
    assert cfg.training.sim2sim_strict and cfg.training.no_play
    assert (cfg.algo.num_envs, cfg.algo.max_iterations, cfg.algo.num_steps_per_env) == (
        1024,
        10000,
        24,
    )
    assert cfg.algo.algorithm.class_name == "uni_rl.algos.rsl_rl_ppo:FinalObservationAwarePPO"
    assert cfg.algo.empirical_normalization and cfg.algo.algorithm.schedule == "adaptive"
    assert (cfg.algo.algorithm.num_learning_epochs, cfg.algo.algorithm.num_mini_batches) == (
        5,
        4,
    )
    assert cfg.env.commands.motion.params.sampling_mode == "start"
    assert not cfg.env.commands.motion.params.truncate_on_clip_end
    assert "motion_clip_end" not in cfg.env.terminations
    assert cfg.env.terminations.anchor_ori.params.threshold == 1.0e9
    assert cfg.env.terminations.undesired_contacts is not None
    assert cfg.env.motrix_disable_self_collision is True
    assert cfg.env.genesis_enable_self_collision is False
    assert OmegaConf.select(reference, "env.motrix_disable_self_collision") is None
    assert OmegaConf.select(reference, "env.genesis_enable_self_collision") is None
    override = BackendAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    fields = ("motrix_disable_self_collision", "genesis_enable_self_collision")
    env_cfg = EnvCfg(**{name: override[name] for name in fields})
    env_cfg.validate()
    kwargs = env_backend_kwargs(env_cfg)
    assert kwargs[fields[0]] is True and kwargs[fields[1]] is False
    if backend != "motrix":
        assert cfg.env.scene.entities.robot.geom_names is None
    else:
        assert (
            cfg.env.scene.entities.robot.geom_names == reference.env.scene.entities.robot.geom_names
        )


@pytest.mark.parametrize(
    "field", ["motrix_disable_self_collision", "genesis_enable_self_collision"]
)
@pytest.mark.parametrize("value", ["false", 0, 1])
def test_collision_controls_reject_non_boolean_values(field, value):
    with pytest.raises(ValueError, match="bool or None"):
        EnvCfg(**{field: value}).validate()


@pytest.mark.parametrize("backend", SOURCES)
def test_aligned_source_rejects_legacy_baseline_contract(backend, tmp_path):
    legacy = config(f"{backend}_baseline")
    (tmp_path / "run_config.json").write_text(
        json.dumps({"contract_snapshot": extract_contract_snapshot(legacy)})
    )
    with pytest.raises(CrossBackendIncompatibleError) as exc:
        resolve_sim2sim_config(tmp_path, config(f"unidr_{backend}"), strict=True)
    assert "env.actions" in str(exc.value)
    assert "algo.empirical_normalization" in str(exc.value)


@pytest.mark.parametrize(
    ("owner", "learner", "source_ids"),
    [("unidr_four_gpu", "cuda:3", (0, 1, 2)), ("unidr_single_gpu", "cuda:0", (0, 0, 0))],
)
def test_four_source_topology_resolves_owner_configs(owner, learner, source_ids):
    cfg = config(owner)
    assert cfg.unidr.learner_device == cfg.training.device == learner
    assert tuple(source.name for source in cfg.unidr.sources) == SOURCES
    assert cfg.training.devices is None
    for source, device_id in zip(cfg.unidr.sources[:3], source_ids, strict=True):
        assert source.device == f"cuda:{device_id}"
        assert source.env_overrides == {f"{source.name}_device_id": device_id}
        resolved = config(source.task.removeprefix("g1_flip_tracking/"))
        assert resolved.training.sim_backend == source.name
        assert semantic_profile(resolved) == semantic_profile(cfg)
    assert cfg.unidr.sources[3].device == "cpu"
    assert cfg.unidr.sources[3].env_overrides == {}
    assert cfg.unidr.sources[3].task == "g1_flip_tracking/unidr_motrix"


@pytest.mark.slow
def test_real_aligned_flip_reference_phases(monkeypatch):
    backend = os.environ.get("UNIDR_REAL_BACKEND")
    if backend is None:
        pytest.skip("set UNIDR_REAL_BACKEND for a two-environment reference-phase smoke")
    assert backend in SOURCES
    cfg = config(f"unidr_{backend}", "algo.num_envs=2", "training.device=cpu")
    override = BackendAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    site = ET.parse(ROOT / "src/unilab/assets/robots/g1/g1.xml").find(
        ".//site[@name='imu_in_pelvis']"
    )
    assert site is not None
    offset = np.fromstring(site.attrib["pos"], sep=" ")
    with closing(registry_env_factory("G1FlipTracking", backend)(2, override)) as env:
        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        rows = np.arange(2, dtype=np.int32)
        assert command.motion.num_frames == 225 and command.motion.fps == 50
        assert env.action_space.shape == (29,)

        def fixed_phase(ids):
            # Test injection exercises the normal reset path at recorded phases.
            command.sampler.current_frames[ids] = phase
            command.sampler.current_clip_indices[ids] = 0
            command.sampler.current_clip_end_frames[ids] = 224
            return command.sampler.current_frames[ids]

        monkeypatch.setattr(command.sampler, "sample_frames", fixed_phase)
        for phase in (0, 100, 124, 180, 224):
            obs, _ = env.reset(rows)
            assert np.array_equal(command.time_steps, [phase, phase])
            reference = command.motion.get_motion_at_frame(command.time_steps)
            assert {key: value.shape for key, value in obs.items()} == {
                "obs": (2, 160),
                "critic": (2, 286),
            }
            assert all(np.isfinite(value).all() for value in obs.values())
            np.testing.assert_allclose(robot.data.joint_pos, reference.joint_pos, atol=1e-5, rtol=0)
            np.testing.assert_allclose(robot.data.joint_vel, reference.joint_vel, atol=1e-5, rtol=0)
            np.testing.assert_allclose(
                robot.data.body_link_pos_w,
                reference.body_pos_w + env.scene.env_origins[:, None, :],
                atol=1e-4,
                rtol=0,
            )
            dot = np.sum(robot.data.body_link_quat_w * reference.body_quat_w, axis=-1)
            np.testing.assert_allclose(np.abs(dot), 1, atol=1e-5, rtol=0)
            for actual, expected in (
                (robot.data.body_link_lin_vel_w[:, 0], reference.body_lin_vel_w[:, 0]),
                (robot.data.body_link_ang_vel_w[:, 0], reference.body_ang_vel_w[:, 0]),
            ):
                np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
            root_quat = reference.body_quat_w[:, 0]
            expected_vel = np_quat_apply_inverse(root_quat, reference.body_lin_vel_w[:, 0])
            expected_vel += np.cross(
                np_quat_apply_inverse(root_quat, reference.body_ang_vel_w[:, 0]), offset
            )
            index = env.observation_manager.active_terms["actor"].index("base_lin_vel")
            start = sum(
                int(np.prod(dim))
                for dim in env.observation_manager.group_obs_term_dim["actor"][:index]
            )
            np.testing.assert_allclose(
                obs["obs"][:, start : start + 3], expected_vel, atol=1e-4, rtol=0
            )
            state = env.step(np.zeros((2, 29), dtype=np.float32))
            assert all(np.isfinite(value).all() for value in state.obs.values())
            assert np.isfinite(state.reward).all()
            print(f"{backend}: phase={phase}, terminated={state.terminated.tolist()}")
