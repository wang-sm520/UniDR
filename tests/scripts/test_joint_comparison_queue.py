"""A follow-on queue must observe successful completion without controlling singles."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/run_joint_comparison.sh"
INVOCATION = "a" * 32
UNIT = "original-single-comparison.service"
COMPLETE = {
    "LoadState": "loaded",
    "InvocationID": INVOCATION,
    "ActiveState": "inactive",
    "SubState": "dead",
    "MainPID": "0",
    "Result": "success",
    "ExecMainCode": "1",
    "ExecMainStatus": "0",
    "ExecMainExitTimestampMonotonic": "123456789",
}
SYSTEMCTL = r"""
import json, os, sys
from pathlib import Path
with Path(os.environ["FAKE_CONTROL_CALLS"]).open("a") as stream:
    stream.write(json.dumps(sys.argv[1:])+"\n")
properties = json.loads(Path(os.environ["FAKE_UNIT_STATE"]).read_text())
for key, value in properties.items():
    print(key+"="+str(value))
"""
UV = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
stage = ("preflight" if "unilab.training.joint_comparison" in args else
         "train" if "scripts/train_unidr.py" in args else "audit_report")
with Path(os.environ["FAKE_STAGE_EVENTS"]).open("a") as stream:
    stream.write(json.dumps([stage, args])+"\n")
if os.environ.get("FAKE_FAIL_STAGE") == stage:
    sys.exit(19)
"""


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def until(predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("Timed out waiting for follow-on queue")


@pytest.fixture
def queue(tmp_path):
    singles = tmp_path / "existing singles"
    singles.mkdir()
    joint = tmp_path / "new joint"
    state, calls, events = [
        tmp_path / name for name in ("unit.json", "calls.jsonl", "events.jsonl")
    ]
    for name, code in (
        ("systemctl", SYSTEMCTL),
        ("uv", UV),
        ("sleep", "import time; time.sleep(.01)"),
    ):
        executable = tmp_path / name
        executable.write_text(f"#!{sys.executable}\n" + code)
        executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "FAKE_UNIT_STATE": str(state),
        "FAKE_CONTROL_CALLS": str(calls),
        "FAKE_STAGE_EVENTS": str(events),
    }
    processes = []

    def set_state(**changes):
        temporary = state.with_suffix(".tmp")
        temporary.write_text(json.dumps({**COMPLETE, **changes}))
        temporary.replace(state)

    def launch(**settings):
        process = subprocess.Popen(
            ["bash", str(SCRIPT), str(singles), str(joint), UNIT, INVOCATION],
            env={**env, **settings},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        return process

    set_state()
    yield SimpleNamespace(
        singles=singles,
        joint=joint,
        state=state,
        calls=calls,
        events=events,
        set_state=set_state,
        launch=launch,
    )
    for process in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_running_success_defaults_do_not_start_training(queue):
    queue.set_state(
        ActiveState="active",
        SubState="running",
        MainPID="4242",
        ExecMainCode="0",
        ExecMainExitTimestampMonotonic="0",
    )
    process = queue.launch()
    until(lambda: len(records(queue.calls)) >= 2)
    assert process.poll() is None
    assert records(queue.events) == []
    assert (queue.joint / "status.tsv").read_text().split("\t")[1] == "waiting"
    queue.set_state()
    out, err = process.communicate(timeout=10)
    assert process.returncode == 0, out + err
    stages = records(queue.events)
    assert [stage for stage, _ in stages] == ["preflight", "train", "audit_report"]
    assert str(queue.singles) in stages[0][1]
    assert "task=g1_flip_tracking/unidr_comparison" in stages[1][1]
    assert f"training.log_dir={queue.joint / 'train'}" in stages[1][1]
    assert "--expected-iterations" in stages[2][1] and "5000" in stages[2][1]
    assert "--num-envs" in stages[2][1] and "1024" in stages[2][1]
    assert not any("resume" in arg or ".pt" in arg for arg in stages[1][1])
    assert (queue.joint / "status.tsv").read_text().split("\t")[1] == "completed"
    assert (queue.joint / "stage.pid").read_text().strip() == "0"
    assert (queue.joint / "single-unit-completed.txt").is_file()


@pytest.mark.parametrize(
    "changes",
    [
        {"ActiveState": "failed", "Result": "exit-code", "ExecMainStatus": "1"},
        {"InvocationID": "b" * 32},
        {"LoadState": "not-found", "InvocationID": ""},
        {"ExecMainCode": "0", "ExecMainExitTimestampMonotonic": "0"},
        {"ExecMainStatus": "15", "Result": "signal"},
        {"MainPID": "4242"},
        {"SubState": "exited"},
        {"ExecMainExitTimestampMonotonic": "0"},
    ],
)
def test_incomplete_or_wrong_unit_never_starts_preflight(queue, changes):
    queue.set_state(**changes)
    process = queue.launch()
    out, err = process.communicate(timeout=10)
    assert process.returncode != 0, out + err
    assert records(queue.events) == []
    assert (queue.joint / "status.tsv").read_text().split("\t")[1] == "failed"


@pytest.mark.parametrize("failed_stage", ["preflight", "train", "audit_report"])
def test_stage_failure_stops_follow_on(queue, failed_stage):
    process = queue.launch(FAKE_FAIL_STAGE=failed_stage)
    out, err = process.communicate(timeout=10)
    assert process.returncode == 19, out + err
    stages = ["preflight", "train", "audit_report"]
    assert [stage for stage, _ in records(queue.events)] == stages[: stages.index(failed_stage) + 1]
    assert (queue.joint / "status.tsv").read_text().split("\t")[1] == "failed"


def test_stopping_waiter_does_not_control_original_service(queue):
    queue.set_state(ActiveState="active", SubState="running", MainPID="4242")
    original = queue.state.read_bytes()
    process = queue.launch()
    until(lambda: len(records(queue.calls)) >= 2)
    process.terminate()
    out, err = process.communicate(timeout=10)
    assert process.returncode == 143, out + err
    assert records(queue.events) == []
    assert queue.state.read_bytes() == original
    assert all(args[:3] == ["--user", "show", UNIT] for args in records(queue.calls))
    assert not any(
        arg in {"stop", "kill", "restart", "start"} for args in records(queue.calls) for arg in args
    )


def test_follow_on_refuses_existing_output(queue):
    queue.joint.mkdir()
    marker = queue.joint / "existing-artifact"
    marker.write_bytes(b"keep")
    process = queue.launch()
    process.communicate(timeout=10)
    assert process.returncode != 0
    assert marker.read_bytes() == b"keep"
    assert records(queue.events) == records(queue.calls) == []
