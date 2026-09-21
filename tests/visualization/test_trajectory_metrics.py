"""Analytical tracking/early-failure checks on detached G1 kinematics."""

from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from unilab.visualization.trajectory_metrics import evaluate_snapshots

ROOT = Path(__file__).parents[2]
MODEL = ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"


@pytest.fixture
def fixture():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    states = np.zeros((224, 1, 72))
    states[:, 0, 0] = np.arange(1, 225) / 50
    states[:, 0, 1:37] = model.qpos0
    rows = [
        dict(
            step=index,
            reference_frame=index,
            next_reference_frame=index + 1,
            terminated=False,
            truncated=False,
            reset=False,
            clip_wrap=False,
            termination_terms=[],
            reward=0.2,
        )
        for index in range(224)
    ]
    options = dict(
        model_file=MODEL,
        joint_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, 30)],
        tracked_body_names=[
            "pelvis",
            "torso_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ],
        end_effector_names=[
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ],
    )
    return states.copy(), states.copy(), rows, options


def test_identical_states_have_zero_errors_and_full_fixed_opportunity(fixture):
    states, refs, rows, options = fixture
    result = evaluate_snapshots(states, refs, rows, **options)
    assert all(metric["max"] < 1e-7 for metric in result["descriptive_tracking_errors"].values())
    score = result["fixed_first_opportunity"]
    assert score["horizon_seconds"] == 4.48
    assert score["survived_step_fraction"] == 1
    assert score["return_with_zero_failure_padding"] == pytest.approx(44.8)
    assert score["completed_without_failure"]
    assert result["true_terminations"] == result["reference_physics_resets"] == 0
    assert result["native_episodes"][0]["right_censored"]
    assert result["native_episodes"][0]["return"] == pytest.approx(44.8)
    assert result["control_statistics"]["reward"]["rms"] == pytest.approx(0.2)


def test_world_translation_has_analytical_body_root_and_effector_errors(fixture):
    states, refs, rows, options = fixture
    states[:, 0, 1:4] += [0.3, 0, 0.4]
    result = evaluate_snapshots(states, refs, rows, **options)
    errors = result["descriptive_tracking_errors"]
    assert errors["root_position_error_m"]["mean"] == pytest.approx(0.5)
    assert errors["body_position_rmse_m"]["mean"] == pytest.approx(0.5)
    assert errors["ee_height_rmse_m"]["mean"] == pytest.approx(0.4)
    assert errors["joint_rmse_rad"]["mean"] == 0
    assert result["fixed_first_opportunity"]["normalized_error_with_failure_padding"][
        "ee_height_rmse_m"
    ] == pytest.approx(0.8)


def test_joint_and_velocity_indices_and_quaternion_sign(fixture):
    states, refs, rows, options = fixture
    states[:, 0, 4:8] *= -1
    states[:, 0, 8] += 0.29
    states[:, 0, 43] += 0.58
    result = evaluate_snapshots(states, refs, rows, **options)
    errors = result["descriptive_tracking_errors"]
    assert errors["root_orientation_error_rad"]["max"] < 1e-7
    assert errors["joint_rmse_rad"]["rms"] == pytest.approx(0.29 / np.sqrt(29))
    assert errors["joint_velocity_rmse_rad_s"]["rms"] == pytest.approx(0.58 / np.sqrt(29))


@pytest.mark.parametrize("double", [False, True])
def test_terminal_observation_included_and_remainder_padded_without_reset_leak(fixture, double):
    states, refs, rows, options = fixture
    rows[9].update(terminated=True, truncated=double, reset=True, termination_terms=["ee_body_pos"])
    # Continue a different reset episode, whose large error must not leak into
    # the fixed first opportunity. Native termination may hold the phase.
    rows[9]["next_reference_frame"] = 9
    refs[9, 0, 0] = 9 / 50
    for index in range(10, 224):
        rows[index].update(reference_frame=index - 10, next_reference_frame=index - 9)
        refs[index, 0, 0] = (index - 9) / 50
    states[9, 0, 1] += 0.5
    states[10:, 0, 1] += 100
    result = evaluate_snapshots(states, refs, rows, **options)
    score = result["fixed_first_opportunity"]
    assert score["observed_steps_including_failure"] == 10
    assert score["padded_steps_after_failure"] == 214
    assert score["survived_step_fraction"] == 9 / 224
    assert score["time_to_first_failure_seconds"] == 0.2
    assert score["return_with_zero_failure_padding"] == pytest.approx(2.0)
    assert score["normalized_error_with_failure_padding"]["root_position_error_m"] == pytest.approx(
        214.5 / 224
    )
    assert result["double_termination_flags"] == int(double)
    assert result["true_terminations"] == 1 and result["timeout_only_resets"] == 0
    assert result["native_episodes"][0]["return"] == pytest.approx(2.0)
    assert not result["native_episodes"][0]["right_censored"]
    assert result["native_episodes"][1]["right_censored"]


def test_early_failure_cannot_appear_perfect_by_only_scoring_surviving_frames(fixture):
    states, refs, rows, options = fixture
    rows[0].update(terminated=True, reset=True, termination_terms=["anchor_pos"])
    result = evaluate_snapshots(states[:1], refs[:1], rows[:1], **options)
    score = result["fixed_first_opportunity"]
    assert score["survived_step_fraction"] == 0
    assert all(
        value == pytest.approx(223 / 224)
        for value in score["normalized_error_with_failure_padding"].values()
    )
    assert result["termination_reason_counts"] == {"anchor_pos": 1}


def test_clip_reset_snapshots_are_not_counted_as_perfect_tracking_or_landing(fixture):
    states, refs, rows, options = fixture
    rows.append(
        dict(rows[-1], step=224, reference_frame=224, next_reference_frame=0, clip_wrap=True)
    )
    states = np.concatenate((states, states[:1]))
    refs = np.concatenate((refs, refs[:1]))
    refs[-1, 0, 0] = 0
    states[-1, 0, 1] = 100  # Artificial reset jump must not enter errors.
    result = evaluate_snapshots(states, refs, rows, **options)
    assert result["reference_physics_resets"] == 1
    assert result["descriptive_frames_excluding_clip_resets"] == 224
    assert result["descriptive_tracking_errors"]["root_position_error_m"]["max"] == 0
    assert result["attempts"][0]["landing_candidate_steps"] == []


@pytest.mark.parametrize(
    "damage",
    [
        "nan",
        "shape",
        "quaternion",
        "step",
        "phase",
        "names",
        "joint_order",
        "tail",
        "timeout",
        "short",
        "reward",
        "partial_reward",
        "action_nan",
    ],
)
def test_invalid_or_censored_input_is_rejected(fixture, damage):
    states, refs, rows, options = deepcopy(fixture)
    if damage == "nan":
        states[0, 0, 8] = np.nan
    elif damage == "shape":
        refs = refs[:, :, :-1]
    elif damage == "quaternion":
        states[0, 0, 4:8] = 0
    elif damage == "step":
        rows[2]["step"] = 1
    elif damage == "phase":
        refs[3, 0, 0] += 1
    elif damage == "names":
        options["tracked_body_names"] = ["missing"]
    elif damage == "joint_order":
        options["joint_names"] = list(reversed(options["joint_names"]))
    elif damage == "tail":
        rows[0]["native_terminated"] = False
    elif damage == "timeout":
        rows[-1].update(truncated=True, reset=True)
    elif damage == "short":
        states, refs, rows = states[:1], refs[:1], rows[:1]
    elif damage == "reward":
        rows[0]["reward"] = np.nan
    elif damage == "partial_reward":
        rows[0].pop("reward")
    else:
        for row in rows:
            row["action_rms"] = np.nan
    with pytest.raises(ValueError):
        evaluate_snapshots(states, refs, rows, **options)
