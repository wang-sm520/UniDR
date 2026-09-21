"""Offline, phase-matched G1 tracking diagnostics; no policy or simulation steps."""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

# Fixed before checkpoint evaluation, shared by every source and checkpoint.
# These are reporting scales, not changes to training rewards or termination.
ERROR_SCALES = {
    "joint_rmse_rad": 1.0,
    "root_position_error_m": 1.0,
    "root_orientation_error_rad": float(np.pi),
    "body_position_rmse_m": 0.5,
    "ee_height_rmse_m": 0.5,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _quaternion_error(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(actual, axis=-1), np.linalg.norm(reference, axis=-1)
    _require(all(np.allclose(n, 1, atol=1e-5) for n in norms), "Non-unit quaternion")
    dot = np.sum(actual * reference, axis=-1) / (norms[0] * norms[1])
    return 2 * np.arccos(np.clip(np.abs(dot), 0, 1))


def _rmse(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=axis))


def _ids(model: Any, names: list[str]) -> np.ndarray:
    _require(bool(names) and len(set(names)) == len(names), "Empty or duplicated body names")
    indices = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names])
    _require(bool(np.all(indices > 0)), "Unknown or world body in tracked body set")
    return indices


def _kinematics(model: Any, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    positions, quaternions = [], []
    for state in states:
        data.qpos[:], data.qvel[:] = state[1:37], state[37:]
        mujoco.mj_forward(model, data)
        positions.append(data.xpos.copy())
        quaternions.append(data.xquat.copy())
    return np.array(positions), np.array(quaternions)


def _validate_rows(rows: list[dict[str, Any]], size: int) -> None:
    _require(len(rows) == size and size > 0, "Missing or empty telemetry")
    for field in ("reward", "action_rms", "action_delta_rms"):
        present = [field in row for row in rows]
        if any(present):
            _require(all(present), f"Partial {field} telemetry")
            _require(
                all(type(row[field]) in (int, float) and np.isfinite(row[field]) for row in rows),
                f"Non-finite or non-scalar {field}",
            )
    for step, row in enumerate(rows):
        _require(row.get("step") == step, "Non-contiguous telemetry steps")
        _require(
            not {"native_terminated", "steps_after_failure", "first_native_failure"} & row.keys(),
            "Diagnostic failure-tail playback cannot be used for native scoring",
        )
        for name in ("terminated", "truncated", "reset", "clip_wrap"):
            _require(type(row.get(name)) is bool, f"Missing or invalid {name}")
        for name in ("reference_frame", "next_reference_frame"):
            _require(type(row.get(name)) is int and 0 <= row[name] <= 224, "Invalid phase")
        expected = (
            0 if step == 0 or rows[step - 1]["reset"] else rows[step - 1]["next_reference_frame"]
        )
        _require(row["reference_frame"] == expected, "Reference phase discontinuity")
        done = row["terminated"] or row["truncated"]
        _require(row["reset"] == done, "Native reset flags mismatch")
        _require(
            row["clip_wrap"] == (row["next_reference_frame"] < row["reference_frame"] and not done),
            "Clip-wrap flag mismatch",
        )
        terms = row.get("termination_terms")
        _require(
            isinstance(terms, list) and all(isinstance(n, str) for n in terms),
            "Missing termination terms",
        )


def _fixed_horizon(
    errors: dict[str, np.ndarray], rows: list[dict[str, Any]], steps: int, dt: float
) -> dict[str, Any]:
    _require(type(steps) is int and 1 <= steps <= 224, "Scoring horizon must be 1..224 steps")
    failed: int | None = None
    observed = 0
    for index, row in enumerate(rows[:steps]):
        _require(not row["clip_wrap"], "Physical clip reset inside fixed scoring horizon")
        _require(not row["truncated"] or row["terminated"], "Timeout censors scoring horizon")
        observed += 1
        if row["terminated"]:
            failed = index
            break  # Subsequent reset episodes must not enter this fixed opportunity.
    _require(observed == steps or failed is not None, "Incomplete fixed scoring horizon")
    padded = steps - observed
    normalized = {
        name: float((np.clip(values[:observed] / ERROR_SCALES[name], 0, 1).sum() + padded) / steps)
        for name, values in errors.items()
        if name in ERROR_SCALES
    }
    rewards = [row.get("reward") for row in rows[:observed]]
    reward_available = all(value is not None for value in rewards)
    if reward_available:
        _require(bool(np.isfinite(rewards).all()), "Non-finite reward")
    return {
        "horizon_steps": steps,
        "horizon_seconds": steps * dt,
        "observed_steps_including_failure": observed,
        "padded_steps_after_failure": padded,
        "completed_without_failure": failed is None,
        "survived_step_fraction": (steps if failed is None else failed) / steps,
        "time_to_first_failure_seconds": None if failed is None else (failed + 1) * dt,
        "first_failure_terms": [] if failed is None else rows[failed]["termination_terms"],
        "first_failure_reference_frame": (
            None if failed is None else rows[failed]["next_reference_frame"]
        ),
        "normalized_error_with_failure_padding": normalized,
        "return_with_zero_failure_padding": (
            float(np.sum(np.asarray(rewards, dtype=np.float64))) if reward_available else None
        ),
        "error_scales": ERROR_SCALES.copy(),
        "padding_rule": "Include terminal observation/error/reward; after first true failure pad normalized errors with 1 and rewards with 0; never borrow samples from the next episode",
    }


def _attempts(
    rows: list[dict[str, Any]], rotation: np.ndarray, ref_rotation: np.ndarray
) -> list[dict[str, Any]]:
    result, start = [], 0
    for index, row in enumerate(rows):
        if not (row["reset"] or row["clip_wrap"] or index == len(rows) - 1):
            continue
        stop = index if row["clip_wrap"] else index + 1
        valid = np.arange(start, stop)
        angles: list[float | None] = [None, None]
        if len(valid) > 1:
            basis = ref_rotation[valid[0]]
            for target, matrices in enumerate((rotation, ref_rotation)):
                forward = matrices[valid, :, 0]
                angle = np.unwrap(np.arctan2(-forward @ basis[:, 2], forward @ basis[:, 0]))
                angles[target] = float(np.degrees(angle[-1] - angle[0]))
        result.append(
            {
                "first_step": start,
                "last_step": index,
                "sagittal_rotation_degrees": angles[0],
                "reference_sagittal_rotation_degrees": angles[1],
                "terminated": row["terminated"],
                "timeout_only": row["truncated"] and not row["terminated"],
                "reference_physics_reset": row["clip_wrap"],
                "landing_candidate_steps": [
                    r["step"] for r in rows[start : index + 1] if r.get("landing_candidate", False)
                ],
            }
        )
        start = index + 1
    return result


def _episodes(rows: list[dict[str, Any]], dt: float) -> list[dict[str, Any]]:
    """Native timeouts/terminations bound episodes; clip wraps are separately counted."""
    result, start = [], 0
    for index, row in enumerate(rows):
        if not (row["reset"] or index == len(rows) - 1):
            continue
        chunk = rows[start : index + 1]
        result.append(
            {
                "first_step": start,
                "last_step": index,
                "duration_seconds": len(chunk) * dt,
                "return": sum(r["reward"] for r in chunk) if "reward" in row else None,
                "terminated": row["terminated"],
                "timeout_only": row["truncated"] and not row["terminated"],
                "right_censored": not row["reset"],
                "reference_physics_resets": sum(r["clip_wrap"] for r in chunk),
            }
        )
        start = index + 1
    return result


def evaluate_snapshots(
    states: np.ndarray,
    reference: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    model_file: Path,
    joint_names: list[str],
    tracked_body_names: list[str],
    end_effector_names: list[str],
    anchor_body_name: str = "torso_link",
    dt: float = 0.02,
    horizon_steps: int = 224,
) -> dict[str, Any]:
    """Measure native snapshots; first-opportunity padding prevents survivorship bias.

    Inputs are post-step, pre-manual-reset states with their actual post-step
    reference phases. Physical clip wraps are excluded from descriptive errors.
    This measures one deterministic rollout, not a statistical success rate.
    """
    _require(
        states.shape == reference.shape and states.ndim == 3 and states.shape[1:] == (1, 72),
        "Expected matching (T, 1, 72) G1 physics snapshots",
    )
    _require(bool(np.isfinite(states).all() and np.isfinite(reference).all()), "Non-finite state")
    _require(np.isfinite(dt) and dt > 0, "Invalid control timestep")
    _validate_rows(rows, len(states))
    actual, ref = states[:, 0].astype(np.float64), reference[:, 0].astype(np.float64)
    _require(
        bool(
            np.allclose(
                ref[:, 0], np.array([r["next_reference_frame"] for r in rows]) * dt, atol=1e-7
            )
        ),
        "Snapshot/reference phase mismatch",
    )
    model = mujoco.MjModel.from_xml_path(str(model_file))
    expected = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)
    ]
    _require(
        (model.nq, model.nv) == (36, 35) and joint_names == expected,
        "G1 DoF or joint order mismatch",
    )
    bodies, effectors = _ids(model, tracked_body_names), _ids(model, end_effector_names)
    anchor = int(_ids(model, [anchor_body_name])[0])
    # mj_forward on detached MjData removes the live mj_step body-cache lag.
    pos, quat = _kinematics(model, actual)
    ref_pos, ref_quat = _kinematics(model, ref)
    errors = {
        "joint_rmse_rad": _rmse(actual[:, 8:37] - ref[:, 8:37], 1),
        "joint_velocity_rmse_rad_s": _rmse(actual[:, 43:] - ref[:, 43:], 1),
        "root_position_error_m": np.linalg.norm(actual[:, 1:4] - ref[:, 1:4], axis=1),
        "root_orientation_error_rad": _quaternion_error(actual[:, 4:8], ref[:, 4:8]),
        "root_linear_velocity_error_m_s": np.linalg.norm(actual[:, 37:40] - ref[:, 37:40], axis=1),
        "anchor_position_error_m": np.linalg.norm(pos[:, anchor] - ref_pos[:, anchor], axis=1),
        "anchor_orientation_error_rad": _quaternion_error(quat[:, anchor], ref_quat[:, anchor]),
        "body_position_rmse_m": np.sqrt(
            np.mean(np.sum((pos[:, bodies] - ref_pos[:, bodies]) ** 2, axis=2), axis=1)
        ),
        "body_orientation_rmse_rad": _rmse(
            _quaternion_error(quat[:, bodies], ref_quat[:, bodies]), 1
        ),
        "ee_height_rmse_m": _rmse(pos[:, effectors, 2] - ref_pos[:, effectors, 2], 1),
        "ee_height_max_error_m": np.max(
            np.abs(pos[:, effectors, 2] - ref_pos[:, effectors, 2]), axis=1
        ),
    }
    mask = np.array([not row["clip_wrap"] for row in rows])
    _require(bool(mask.any()), "No non-reset snapshots")
    descriptive = {
        name: {
            "mean": float(np.mean(values[mask])),
            "rms": float(_rmse(values[mask], 0)),
            "p95": float(np.quantile(values[mask], 0.95)),
            "max": float(np.max(values[mask])),
        }
        for name, values in errors.items()
    }
    fail = [row for row in rows if row["terminated"]]
    root = int(_ids(model, ["pelvis"])[0])
    rotation: list[np.ndarray] = []
    ref_rotation: list[np.ndarray] = []
    for values, target in ((quat[:, root], rotation), (ref_quat[:, root], ref_rotation)):
        for q in values:
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, q)
            target.append(matrix.reshape(3, 3))
    return {
        "protocol": "g1_native_tracking_v1",
        "frames": len(rows),
        "control_dt_seconds": dt,
        "descriptive_frames_excluding_clip_resets": int(mask.sum()),
        "tracked_body_names": tracked_body_names,
        "end_effector_names": end_effector_names,
        "anchor_body_name": anchor_body_name,
        "descriptive_tracking_errors": descriptive,
        "control_statistics": {
            field: {
                "mean": float(np.mean([row[field] for row in rows])),
                "rms": float(_rmse(np.array([row[field] for row in rows]), 0)),
                "max": float(np.max([row[field] for row in rows])),
            }
            for field in ("reward", "action_rms", "action_delta_rms")
            if field in rows[0]
        },
        "native_episodes": _episodes(rows, dt),
        "fixed_first_opportunity": _fixed_horizon(errors, rows, horizon_steps, dt),
        "true_terminations": len(fail),
        "timeout_only_resets": sum(row["truncated"] and not row["terminated"] for row in rows),
        "double_termination_flags": sum(row["truncated"] and row["terminated"] for row in rows),
        "reference_physics_resets": sum(row["clip_wrap"] for row in rows),
        "first_failure_step": fail[0]["step"] if fail else None,
        "first_failure_seconds": (fail[0]["step"] + 1) * dt if fail else None,
        "termination_reason_counts": dict(
            Counter(name for row in fail for name in row["termination_terms"])
        ),
        "attempts": _attempts(rows, np.array(rotation), np.array(ref_rotation)),
        "semantics": {
            "errors": "Post-step qpos/qvel versus phase-matched reference in original world coordinates; body positions/orientations recomputed with detached MuJoCo forward kinematics",
            "fixed_horizon": "First 224 controls end at reference frame 224, before the native 225th-control physical clip reset; never count a reset as landing",
            "descriptive": "All non-clip-reset snapshots including terminal observations, across reset episodes; survivor- and phase-occupancy-dependent, not the primary cross-checkpoint score",
            "rotation": "Unwrapped sagittal projection relative to the first reference root axes of each uninterrupted attempt; excludes clip-reset frames, not a full 3D rotation metric",
            "landing": "Existing telemetry landing flags are kinematic candidates, can lag the snapshot by one physics substep, and are not contact-force or task-success certification",
            "scope": "One deterministic native rollout, not a statistical success rate; no policy tuning or checkpoint selection",
        },
    }


def evaluate_directory(directory: Path, *, root: Path) -> dict[str, Any]:
    """Read a native holdout directory without changing any input artifact."""
    config_path, states_path, telemetry_path = (
        directory / name
        for name in ("mujoco_config.json", "physics_snapshots.npz", "telemetry.jsonl")
    )
    config = json.loads(config_path.read_text())
    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    with np.load(states_path, allow_pickle=False) as saved:
        _require(set(saved.files) == {"states", "reference", "phase"}, "Unexpected snapshot schema")
        _require(
            np.array_equal(saved["phase"], [row["next_reference_frame"] for row in rows]),
            "Snapshot phase mismatch",
        )
        env = config["env"]
        command = env["commands"]["motion"]["params"]
        result = evaluate_snapshots(
            saved["states"],
            saved["reference"],
            rows,
            model_file=root / env["scene"]["model_file"],
            joint_names=env["scene"]["entities"]["robot"]["joint_names"],
            tracked_body_names=command["body_names"],
            end_effector_names=env["terminations"]["ee_body_pos"]["params"]["body_names"],
            anchor_body_name=command["anchor_body_name"],
        )
    result["input_sha256"] = {
        str(path): sha256(path.read_bytes()).hexdigest()
        for path in (config_path, states_path, telemetry_path)
    }
    return result
