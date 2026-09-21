"""Diagnostic videos must preserve native behavior up to the first failure."""

import json

import numpy as np
import pytest

from unilab.visualization.checkpoint_series import validate_native_prefix


def _fixture(tmp_path):
    native, video = tmp_path / "native", tmp_path / "video"
    native.mkdir()
    video.mkdir()
    rows = [dict(terminated=i == 3, truncated=False) for i in range(6)]
    diagnostic = [
        dict(native_terminated=i == 3, native_truncated=False, first_native_failure=i == 3)
        for i in range(6)
    ]
    for folder, telemetry in ((native, rows), (video, diagnostic)):
        (folder / "telemetry.jsonl").write_text("\n".join(map(json.dumps, telemetry)))
        np.savez(
            folder / "physics_snapshots.npz", states=np.arange(12).reshape(6, 2), phase=np.arange(6)
        )
    return native, video


def test_prefix_includes_failed_physics_but_allows_post_failure_continuation(tmp_path):
    native, video = _fixture(tmp_path)
    states = np.arange(12).reshape(6, 2)
    states[4:] += 100
    np.savez(video / "physics_snapshots.npz", states=states, phase=[0, 1, 2, 4, 5, 5])
    assert validate_native_prefix(native, video) == 3


@pytest.mark.parametrize("change", ["physics", "phase", "flag", "first"])
def test_changed_prefailure_behavior_is_rejected(tmp_path, change):
    native, video = _fixture(tmp_path)
    if change in ("physics", "phase"):
        states, phase = np.arange(12).reshape(6, 2), np.arange(6)
        (states if change == "physics" else phase)[2] += 1
        np.savez(video / "physics_snapshots.npz", states=states, phase=phase)
    else:
        path = video / "telemetry.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[3]["native_terminated" if change == "flag" else "first_native_failure"] = False
        path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises((AssertionError, ValueError)):
        validate_native_prefix(native, video)
