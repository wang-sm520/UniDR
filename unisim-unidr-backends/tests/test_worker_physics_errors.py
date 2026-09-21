"""Native diagnostics must invalidate successful protocol replies before use."""

from __future__ import annotations

import subprocess
import sys
from multiprocessing import shared_memory
from pathlib import Path

import pytest

from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.backend import MjcfSubprocessBackend, SubprocessWorkerError
from unisim.scene import SceneCfg

_WORKER = """
import importlib.util
import os
import sys
spec = importlib.util.spec_from_file_location('protocol', sys.argv[1])
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
while True:
    message = protocol.recv_message(sys.stdin.buffer)
    if message['cmd'] == protocol.CMD_SHUTDOWN:
        break
    payload = message.get('payload') or {}
    os.write(2, payload.get('diagnostic', '').encode())
    protocol.send_message(sys.stdout.buffer, protocol.CMD_READY, {'accepted': True})
"""


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    instance = MjcfSubprocessBackend(SceneCfg(model_file="unused.xml"), 1, 0.005)
    instance._open_worker_log()
    instance._proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _WORKER, protocol.__file__],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=instance._stderr_file,
        bufsize=0,
    )
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("cmd", [protocol.CMD_INIT, protocol.CMD_STEP, protocol.CMD_SET_STATE])
@pytest.mark.parametrize(
    "diagnostic",
    [
        "PhysX error: increase totalAggregatePairsCapacity, "
        "otherwise, the simulation will miss interactions\n",
        "PhysX error: PxgCudaDeviceMemoryAllocator failed to allocate memory 4294967296 bytes!\n",
        "[Error] [omni.physx.plugin] Failed to create PhysX Scene\n",
        "PhysX error: Patch buffer overflow detected\n",
        "PhysX error: Contact buffer overflow detected\n",
        "PhysX error: Collision stack overflow\n",
    ],
)
def test_fatal_reply_is_rejected_and_resources_closed(backend, cmd, diagnostic):
    worker = backend._proc
    log_path = Path(backend._stderr_file.name)
    memory = shared_memory.SharedMemory(create=True, size=16)
    memory_name = memory.name
    backend._shm_handles["test"] = memory
    with pytest.raises(SubprocessWorkerError, match="fatal physics diagnostic") as error:
        backend._request(cmd, {"diagnostic": diagnostic}, expect=protocol.CMD_READY)
    assert str(log_path) in str(error.value)
    assert diagnostic in error.value.stderr_tail
    assert worker.poll() is not None
    assert backend._proc is None and backend._slots == {} and backend._shm_handles == {}
    assert backend._stderr_file is None and backend._stderr_reader is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=memory_name)
    assert log_path.read_text() == diagnostic
    with pytest.raises(SubprocessWorkerError, match="earlier failure"):
        backend._request(protocol.CMD_STEP, {}, expect=protocol.CMD_READY)


def test_warnings_remain_usable_and_read_cursor_does_not_move_writer(backend):
    warning = "[Warning] PhysX imported scene; GPU capacity configured.\n"
    assert backend._request(
        protocol.CMD_STEP, {"diagnostic": warning}, expect=protocol.CMD_READY
    ) == {"accepted": True}
    reader = backend._stderr_reader
    assert reader.tell() == len(warning)
    assert backend._stderr_tail() == warning
    assert reader.tell() == len(warning)
    assert backend._request(
        protocol.CMD_SET_STATE, {"diagnostic": warning}, expect=protocol.CMD_READY
    ) == {"accepted": True}
    assert Path(backend._stderr_file.name).read_text() == warning * 2
    assert reader.tell() == 2 * len(warning)
    # Consumed bytes are not read again on an otherwise empty barrier.
    backend._check_worker_stderr(protocol.CMD_STEP)
    assert reader.tell() == 2 * len(warning)


def test_fatal_native_write_split_across_barriers(backend):
    backend._request(
        protocol.CMD_STEP,
        {"diagnostic": "[Error] PhysX error: simulation will miss inter"},
        expect=protocol.CMD_READY,
    )
    with pytest.raises(SubprocessWorkerError, match="fatal physics diagnostic"):
        backend._request(
            protocol.CMD_SET_STATE, {"diagnostic": "actions\n"}, expect=protocol.CMD_READY
        )


def test_fatal_native_write_crosses_scan_block(backend):
    diagnostic = "x" * (65536 - 20) + "simulation will miss interactions\n"
    with pytest.raises(SubprocessWorkerError, match="fatal physics diagnostic"):
        backend._request(protocol.CMD_STEP, {"diagnostic": diagnostic}, expect=protocol.CMD_READY)


def test_pending_fatal_diagnostic_prevents_next_stateful_request(backend):
    log_path = Path(backend._stderr_file.name)
    backend._stderr_file.write(b"PhysX error: simulation will miss interactions\n")
    backend._stderr_file.flush()
    with pytest.raises(SubprocessWorkerError, match="fatal physics diagnostic"):
        backend._request(
            protocol.CMD_STEP, {"diagnostic": "STEP WAS EXECUTED\n"}, expect=protocol.CMD_READY
        )
    assert "STEP WAS EXECUTED" not in log_path.read_text()
