"""Success requires a complete, unreset flip followed by final stable ground contact."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand
from unilab.visualization.flip_trials import (
    PROTOCOL,
    _TrialMotion,
    checkpoint_schedule,
    classify,
    inspect_states,
    run_trials,
    write_checkpoint_table,
)


@pytest.fixture
def attempt():
    feet = np.ones((474, 2), dtype=bool)
    feet[80:140] = False
    heights = np.full((474, 2), 0.04)
    heights[80:140] = 0.4
    upright = np.ones(474)
    upright[100:120] = -1
    trace = dict(
        feet_contact=feet,
        feet_height=heights,
        upright_z=upright,
        nonfoot_ground_contact=np.zeros(474, bool),
        vertical_speed=np.zeros(474),
        pelvis_height=np.full(474, 0.75),
        rotation_degrees=np.concatenate((np.linspace(0, -360, 224), np.full(250, -360))),
    )
    rows = [
        dict(
            step=i,
            reference_frame=min(i, 224),
            next_reference_frame=min(i + 1, 224),
            terminated=False,
            truncated=False,
            termination_terms=[],
            reward=0.1,
        )
        for i in range(474)
    ]
    return trace, rows


def test_complete_flip_and_stable_landing_succeeds(attempt):
    trace, rows = attempt
    result = classify(trace, rows, -1.0)
    assert result["success"] and all(result["conditions"].values())
    assert result["longest_airborne_steps"] == 60
    assert result["return_sum"] == pytest.approx(47.4)


@pytest.mark.parametrize(
    "defect",
    [
        "no_flight",
        "no_inversion",
        "partial_turn",
        "wrong_direction",
        "double_flip",
        "no_contact",
        "nonfoot_contact",
        "vertical_speed",
        "not_upright",
        "late_instability",
    ],
)
def test_required_motion_and_contact_evidence_cannot_be_skipped(attempt, defect):
    trace, rows = attempt
    if defect == "no_flight":
        trace["feet_height"][:] = 0.04
    elif defect == "no_inversion":
        trace["upright_z"][:] = 1
    elif defect in ("partial_turn", "wrong_direction", "double_flip"):
        trace["rotation_degrees"] *= {"partial_turn": 0.5, "wrong_direction": -1, "double_flip": 2}[
            defect
        ]
    elif defect == "no_contact":
        trace["feet_contact"][150:] = False
    elif defect == "nonfoot_contact":
        trace["nonfoot_ground_contact"][160] = True
    elif defect == "vertical_speed":
        trace["vertical_speed"][214:224] = 0.6
    elif defect == "not_upright":
        trace["upright_z"][214:224] = 0.5
    elif defect == "late_instability":
        trace["feet_contact"][223, 0] = False
    assert not classify(trace, rows, -1.0)["success"]


@pytest.mark.parametrize("flags", [(True, False), (False, True), (True, True)])
def test_native_tracking_done_is_diagnostic_not_physical_fall(attempt, flags):
    trace, rows = attempt
    rows[-1].update(terminated=flags[0], truncated=flags[1], termination_terms=["native"])
    result = classify(trace, rows, -1.0)
    assert result["success"] and result["terminal_terms"] == ["native"]


def test_reset_jump_cannot_supply_a_landing(attempt):
    trace, rows = attempt
    rows[170]["next_reference_frame"] = 0
    assert not classify(trace, rows, -1.0)["success"]


def test_short_attempt_cannot_count_as_complete(attempt):
    trace, rows = attempt
    trace = {name: value[:-1] for name, value in trace.items()}
    assert not classify(trace, rows[:-1], -1.0)["success"]


def test_nonfinite_and_misaligned_evidence_is_rejected(attempt):
    trace, rows = attempt
    broken = deepcopy(trace)
    broken["vertical_speed"][12] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        classify(broken, rows, -1.0)
    rows[-1]["step"] = 1
    with pytest.raises(ValueError, match="Non-contiguous"):
        classify(trace, rows, -1.0)


@pytest.mark.parametrize("step", [224, 300, 473])
@pytest.mark.parametrize(
    "field,value",
    [
        ("nonfoot_ground_contact", True),
        ("upright_z", 0.49),
        ("pelvis_height", 0.29),
    ],
)
def test_fall_anywhere_in_five_seconds_cannot_recover_to_success(attempt, step, field, value):
    trace, rows = attempt
    trace[field][step] = value
    result = classify(trace, rows, -1.0)
    assert not result["success"]
    assert not result["conditions"]["five_seconds_without_fall"]
    assert result["first_post_motion_fall_seconds"] == pytest.approx((step - 223) * 0.02)


def test_clip_alone_and_499_seconds_are_insufficient(attempt):
    trace, rows = attempt
    for count in (224, 473):
        result = classify({key: value[:count] for key, value in trace.items()}, rows[:count], -1.0)
        assert not result["success"] and not result["conditions"]["five_seconds_without_fall"]
    assert PROTOCOL["observation_steps"] * PROTOCOL["control_dt_seconds"] == 5.0


def test_positive_gap_contact_tolerance_without_active_contacts():
    import mujoco

    joints = "".join(
        f'<body><joint name="j{i}" axis="0 1 0"/><geom type="sphere" size=".01" mass="1"/></body>'
        for i in range(29)
    )
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
        <geom type="plane" size="10 10 .1"/>
        <body><freejoint/><geom type="sphere" size=".01" mass="1"/>
          <body pos="0 .1 -.6" name="left_ankle_roll_link">
            <geom type="sphere" size=".05" mass="1"/></body>
          <body pos="0 -.1 -.6" name="right_ankle_roll_link">
            <geom type="sphere" size=".05" mass="1"/></body>
          {joints}
        </body></worldbody></mujoco>""")
    states = np.zeros((3, 72))
    states[:, 4] = 1
    states[:, 3] = [0.651, 0.651, 0.653]
    trace = inspect_states(model, states)
    np.testing.assert_array_equal(trace["feet_contact"], [[True, True], [False, False]])
    assert not trace["nonfoot_ground_contact"].any()


def test_final_reference_refresh_cannot_enter_physical_resample_path(monkeypatch):
    motion = object.__new__(_TrialMotion)
    motion._env = SimpleNamespace(num_envs=1)
    motion.time_steps = np.array([223])
    motion.sampler = SimpleNamespace(current_clip_end_frames=np.array([224]))
    calls = []
    monkeypatch.setattr(MotionCommand, "_update_command", lambda self, ids: calls.append(ids))
    motion._update_command(None)
    assert calls[-1] is None
    motion.time_steps[0] = 224
    for _ in range(250):
        motion._update_command(None)
        np.testing.assert_array_equal(calls[-1], [0])


def test_historical_schedule_includes_zero_periodic_and_final_without_duplicates():
    schedule = checkpoint_schedule(True)
    assert len(schedule) == len(set(schedule)) == 175
    assert [index for source, index in schedule if source == "joint"] == [
        0,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        4999,
    ]
    assert all(
        sum(source == name for source, _ in schedule) == 41
        for name in ("motrix", "isaacsim", "isaacgym", "genesis")
    )
    assert set(checkpoint_schedule(False)) <= set(schedule)
    assert len(checkpoint_schedule(False)) == 5


def test_table_keeps_joint_final_separate_from_model_5000_and_rejects_missing(tmp_path):
    import csv

    results = [
        dict(
            source=source,
            checkpoint_index=index,
            completed_updates=index + 1,
            seed=1,
            steps=474,
            success=True,
        )
        for source, index in checkpoint_schedule(True)
    ]
    with pytest.raises(ValueError, match="175 unique"):
        write_checkpoint_table(tmp_path, results[:-1])
    write_checkpoint_table(tmp_path, results)
    with (tmp_path / "checkpoint_success.csv").open(encoding="utf-8-sig") as stream:
        table = {row["checkpoint 编号"]: row for row in csv.DictReader(stream)}
    assert table["4999"]["实际完成更新"] == "5000"
    assert table["4999"]["Motrix"] == "—"
    assert table["4999"]["四源联合"] == "成功"
    assert table["5000"]["四源联合"] == "—"
    assert table["5000"]["实际完成更新"] == "5001"


@pytest.mark.parametrize("source", ["genesis", "motrix", "isaacgym"])
def test_explicit_final_run_uses_requested_checkpoint_and_strict_budget(
    tmp_path, monkeypatch, source
):
    import json

    from unilab.visualization import flip_trials

    requested = tmp_path / "new-experiment" / source

    def strict_preflight(checkpoint, root, **kwargs):
        assert checkpoint == requested / "model_4999.pt"
        assert root == tmp_path
        assert kwargs == {
            "expected_iterations": 5000,
            "expected_num_envs": 4096,
            "selected_checkpoint": False,
        }
        raise ValueError("strict preflight rejected incomplete run")

    monkeypatch.setattr(flip_trials, "preflight", strict_preflight)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="strict preflight rejected"):
        run_trials(
            tmp_path,
            output,
            single_runs=[(source, requested)],
            expected_iterations=5000,
            expected_num_envs=4096,
        )
    assert json.loads((output / "schedule.json").read_text()) == [[source, 4999]]
    assert json.loads((output / "runs.json").read_text()) == {source: str(requested)}
    assert json.loads((output / "protocol.json").read_text()) == PROTOCOL
    assert not (output / source).exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"single_runs": []},
        {"single_runs": [("genesis", Path("a")), ("genesis", Path("b"))]},
        {"single_runs": [("mujoco", Path("a"))]},
        {"single_runs": [("joint", Path("a"))]},
        {"single_runs": [("genesis", Path("a"))], "checkpoint_series": True},
        {"single_runs": [("genesis", Path("a"))], "expected_iterations": 0},
        {"single_runs": [("genesis", Path("a"))], "expected_num_envs": -1},
        {"expected_iterations": 5000},
        {"expected_num_envs": 4096},
    ],
)
def test_explicit_selection_rejects_ambiguous_or_invalid_requests(tmp_path, kwargs):
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="(Explicit single runs|Custom budgets)"):
        run_trials(tmp_path, output, **kwargs)
    assert not output.exists()
