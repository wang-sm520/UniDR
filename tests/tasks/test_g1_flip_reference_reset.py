"""Opt-in real flip reset integrity, separate from PPO and behavioral success."""

import os
import xml.etree.ElementTree as ET
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from unisim.utils.rotation import np_quat_apply_inverse

from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory


@pytest.mark.slow
def test_real_g1_flip_reference_reset():
    backend = os.environ.get("UNIDR_REAL_BACKEND")
    if backend is None:
        pytest.skip("set UNIDR_REAL_BACKEND to opt into real backend execution")
    assert backend in {"isaacsim", "isaacgym", "genesis", "motrix"}
    root = Path(__file__).parents[2]
    with initialize_config_dir(config_dir=str(root / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                f"task=g1_flip_tracking/{backend}_baseline",
                "algo.num_envs=2",
                "training.device=cpu",
                "training.play_render_mode=none",
                "env.max_episode_seconds=0.04",
            ],
        )
    factory = registry_env_factory("G1FlipTracking", backend)
    override = BackendAdapter(cfg, root_dir=root, algo_name="ppo").build_task_env_cfg_override()
    with closing(factory(2, env_cfg_override=override)) as env:
        rows = np.arange(2, dtype=np.int32)
        obs, _ = env.reset(rows)
        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        reference = command.motion.get_motion_at_frame(command.time_steps)
        assert np.array_equal(command.time_steps, [0, 0])
        assert command.motion.num_frames == 225 and command.motion.fps == 50
        assert env.action_space.shape == (29,)
        assert {key: value.shape for key, value in obs.items()} == {
            "obs": (2, 160),
            "critic": (2, 286),
        }
        assert all(np.isfinite(value).all() for value in obs.values())
        np.testing.assert_allclose(robot.data.joint_pos, reference.joint_pos, atol=1e-5, rtol=0)
        np.testing.assert_allclose(robot.data.joint_vel, reference.joint_vel, atol=1e-5, rtol=0)
        expected_positions = reference.body_pos_w + env.scene.env_origins[:, None, :]
        np.testing.assert_allclose(
            robot.data.body_link_pos_w, expected_positions, atol=1e-4, rtol=0
        )
        quat_dot = np.sum(robot.data.body_link_quat_w * reference.body_quat_w, axis=-1)
        np.testing.assert_allclose(np.abs(quat_dot), 1, atol=1e-5, rtol=0)
        # Root motion must survive set_state; non-root velocities can differ from
        # finite-difference reference velocities without indicating a reset bug.
        np.testing.assert_allclose(
            robot.data.body_link_lin_vel_w[:, 0], reference.body_lin_vel_w[:, 0], atol=1e-4, rtol=0
        )
        np.testing.assert_allclose(
            robot.data.body_link_ang_vel_w[:, 0], reference.body_ang_vel_w[:, 0], atol=1e-4, rtol=0
        )
        # The policy observes the offset IMU site, not the pelvis link origin.
        site = ET.parse(root / "src/unilab/assets/robots/g1/g1.xml").find(
            ".//site[@name='imu_in_pelvis']"
        )
        assert site is not None
        offset = np.fromstring(site.attrib["pos"], sep=" ")
        root_quat = reference.body_quat_w[:, 0]
        expected_vel = np_quat_apply_inverse(root_quat, reference.body_lin_vel_w[:, 0])
        expected_vel += np.cross(
            np_quat_apply_inverse(root_quat, reference.body_ang_vel_w[:, 0]), offset
        )
        term_index = env.observation_manager.active_terms["actor"].index("base_lin_vel")
        start = sum(
            int(np.prod(dim))
            for dim in env.observation_manager.group_obs_term_dim["actor"][:term_index]
        )
        np.testing.assert_allclose(
            obs["obs"][:, start : start + 3], expected_vel, atol=1e-4, rtol=0
        )
        for index in range(2):
            state = env.step(np.zeros((2, 29), dtype=np.float32))
            assert all(np.isfinite(value).all() for value in state.obs.values())
            assert np.isfinite(state.reward).all()
            assert not state.terminated.any()
            if index == 0:
                assert not state.truncated.any()
            else:
                assert state.truncated.all()
                assert state.final_observation is not None
                assert {key: value.shape for key, value in state.final_observation.items()} == {
                    key: value.shape for key, value in obs.items()
                }
                assert all(np.isfinite(value).all() for value in state.final_observation.values())
        print(f"{backend}: 2 G1 flip resets match reference; 2 steps and timeout final obs passed")
