"""Exercise the real queue with stub commands, without training or SDK imports."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

QUEUE = Path(__file__).parents[2] / "scripts/run_single_comparison.sh"
SOURCES = ("motrix", "isaacsim", "isaacgym", "genesis")
STAGES = ("prepare", "train", "audit", "sim2sim")
FAKE_UV = r'''
import json, os, signal, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
events = Path(os.environ["FAKE_UV_EVENTS"])
if "unilab.training.single_comparison" in args:
    i = args.index("unilab.training.single_comparison")
    source, run = args[i+1], Path(args[i+2])
    stage = "prepare"
    run.mkdir()
elif "train" in args:
    source = args[args.index("--sim")+1]
    run = Path(next(a.split("=", 1)[1] for a in args if a.startswith("training.log_dir=")))
    stage = "train"
elif "uni_rl.logging.single_run_audit" in args:
    run = Path(args[args.index("uni_rl.logging.single_run_audit")+1])
    source, stage = run.name, "audit"
else:
    run = Path(args[args.index("scripts/play_single_reference.py")+1]).parent
    source, stage = run.name, "sim2sim"
with events.open("a") as stream:
    stream.write(json.dumps([source, stage, args])+"\n")
key = source + ":" + stage
if os.environ.get("FAKE_UV_FAIL") == key:
    sys.exit(17)
if os.environ.get("FAKE_UV_HOLD") == key:
    while not events.with_suffix(".release").exists():
        time.sleep(.01)
if os.environ.get("FAKE_UV_CHILD") == key:
    child_code = """
import signal, sys, time
from pathlib import Path
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
Path(sys.argv[1]).write_text('ready')
while True: time.sleep(.1)
"""
    child = subprocess.Popen([sys.executable, "-c", child_code, str(run / "child.ready")])
    def stop(*_):
        child.wait(timeout=3)
        sys.exit(130)
    signal.signal(signal.SIGINT, stop)
    (run / "child.pid").write_text(str(child.pid))
    while True:
        signal.pause()
'''


def until(predicate, *, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("Timed out waiting for queue progress")


@pytest.fixture
def queue(tmp_path):
    executable = tmp_path / "uv"
    executable.write_text(f"#!{sys.executable}\n" + FAKE_UV)
    executable.chmod(0o755)
    events = tmp_path / "events.jsonl"
    root = tmp_path / "comparison with spaces"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "FAKE_UV_EVENTS": str(events)}
    processes = []

    def launch(order=None, train_only=False, **settings):
        args = ["bash", str(QUEUE), str(root), "2", "2"]
        if order is not None or train_only:
            args.append(order if order is not None else ",".join(SOURCES))
        if train_only:
            args.append("--train-only")
        process = subprocess.Popen(
            args,
            env={**env, **settings},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        return process

    yield root, events, launch
    for process in processes:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pid_file = root / "stage.pid"
                if pid_file.exists() and int(pid_file.read_text()) > 0:
                    try:
                        os.killpg(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.kill()
                process.wait(timeout=5)


def records(events):
    return [json.loads(line) for line in events.read_text().splitlines()]


def test_queue_audits_before_advancing_and_uses_fixed_final_checkpoint(queue):
    root, events, launch = queue
    process = launch(FAKE_UV_HOLD="motrix:audit")
    until(lambda: events.exists() and len(records(events)) == 3)
    assert [record[:2] for record in records(events)] == [["motrix", stage] for stage in STAGES[:3]]
    assert not (root / "isaacsim").exists()
    events.with_suffix(".release").touch()
    out, err = process.communicate(timeout=10)
    assert process.returncode == 0, out + err
    assert [record[:2] for record in records(events)] == [
        [source, stage] for source in SOURCES for stage in STAGES
    ]
    for source, stage, args in records(events):
        if stage == "sim2sim":
            assert str(root / source / "model_1.pt") in args
        if stage == "train":
            assert "--profile" in args and "comparison" in args
            assert "algo.num_envs=2" in args and "algo.max_iterations=2" in args
    assert (root / "status.tsv").read_text().split("\t")[1] == "completed"
    assert (root / "stage.pid").read_text().strip() == "0"


@pytest.mark.parametrize("failed_stage", ["train", "audit"])
def test_failed_stage_prevents_playback_and_next_backend(queue, failed_stage):
    root, events, launch = queue
    process = launch(FAKE_UV_FAIL=f"motrix:{failed_stage}")
    out, err = process.communicate(timeout=10)
    assert process.returncode == 17, out + err
    assert [record[:2] for record in records(events)] == [
        ["motrix", stage] for stage in STAGES[: STAGES.index(failed_stage) + 1]
    ]
    assert not (root / "isaacsim").exists()
    assert (root / "status.tsv").read_text().split("\t")[1:3] == ["failed", "motrix"]


def test_queue_cancellation_reaps_owned_grandchild(queue):
    root, events, launch = queue
    process = launch(FAKE_UV_CHILD="motrix:train")
    until(lambda: (root / "motrix/child.ready").exists())
    grandchild = int((root / "motrix/child.pid").read_text())
    process.terminate()
    out, err = process.communicate(timeout=10)
    assert process.returncode == 143, out + err
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)
    assert [record[:2] for record in records(events)] == [
        ["motrix", "prepare"],
        ["motrix", "train"],
    ]
    assert not (root / "isaacsim").exists()
    assert (root / "status.tsv").read_text().split("\t")[1:3] == ["failed", "motrix"]


def test_queue_refuses_existing_experiment_directory(queue):
    root, events, launch = queue
    root.mkdir()
    marker = root / "existing-checkpoint"
    marker.write_bytes(b"preserve")
    process = launch()
    process.communicate(timeout=10)
    assert process.returncode != 0
    assert marker.read_bytes() == b"preserve"
    assert not events.exists()


def test_custom_order_train_only_audits_all_four_before_advancing(queue):
    root, events, launch = queue
    order = ("genesis", "motrix", "isaacgym", "isaacsim")
    process = launch(order=",".join(order), train_only=True, FAKE_UV_HOLD="genesis:audit")
    until(lambda: events.exists() and len(records(events)) == 3)
    assert not (root / "motrix").exists()
    events.with_suffix(".release").touch()
    out, err = process.communicate(timeout=10)
    assert process.returncode == 0, out + err
    assert [record[:2] for record in records(events)] == [
        [source, stage] for source in order for stage in STAGES[:3]
    ]
    cfg = json.loads((root / "queue_config.json").read_text())
    assert cfg == dict(num_envs=2, iterations=2, source_order=",".join(order), holdout=False)
    assert (root / "status.tsv").read_text().split("\t")[1:] == ["completed", "isaacsim", "audit\n"]


@pytest.mark.parametrize(
    "order",
    [
        "genesis,motrix,isaacgym",
        "genesis,motrix,isaacgym,genesis",
        "genesis,motrix,isaacgym,mujoco",
    ],
)
def test_invalid_order_fails_before_creating_experiment(queue, order):
    root, events, launch = queue
    process = launch(order=order, train_only=True)
    process.communicate(timeout=10)
    assert process.returncode == 2
    assert not root.exists() and not events.exists()
