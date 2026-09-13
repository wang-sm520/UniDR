"""Source-isolated CPython resource trackers for crash-safe nested SHM cleanup."""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
from multiprocessing import resource_tracker
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any


class SourceResources:
    """Own a tracker outside the source group and wait for its EOF cleanup."""

    def __init__(self) -> None:
        if os.name != "posix" or sys.implementation.name != "cpython":
            raise RuntimeError("multi-source resource isolation requires POSIX CPython")
        reader, self.writer = mp.Pipe(duplex=False)
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"from multiprocessing.resource_tracker import main; main({reader.fileno()})",
                ],
                pass_fds=(reader.fileno(),),
                start_new_session=True,
            )
        except BaseException:
            self.writer.close()
            raise
        finally:
            reader.close()

    def release_parent_writer(self) -> None:
        self.writer.close()

    def finish(self, timeout: float) -> None:
        """EOF occurs only after the source and its nested workers release their FDs."""
        self.writer.close()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"source resource tracker {self.process.pid} did not finish cleanup; "
                "a descendant may have escaped its source process group"
            ) from error


def install_source_tracker(connection: Connection) -> None:
    """Replace only this spawned process's inherited tracker before factory execution.

    CPython spawn installs a borrowed tracker FD in each child. Rebinding that
    FD makes subsequently spawned descendants share this source's tracker rather
    than the learner's. The source tracker is owned and reaped by the coordinator.
    """
    tracker: Any = resource_tracker._resource_tracker
    if not all(hasattr(tracker, name) for name in ("_lock", "_fd", "_pid")):
        raise RuntimeError("unsupported CPython resource-tracker implementation")
    with tracker._lock:
        previous_fd = tracker._fd
        tracker._fd = os.dup(connection.fileno())
        tracker._pid = None
        if previous_fd is not None:
            os.close(previous_fd)
    connection.close()


def exclude_transport_attachment(memory: SharedMemory) -> None:
    """Keep coordinator-owned transport out of the source's resource ownership."""
    resource_tracker.unregister(f"/{memory.name}", "shared_memory")


def reserve_transport_memory(memory: SharedMemory, size: int) -> None:
    """Reserve Linux tmpfs storage before mapped writes can receive SIGBUS."""
    if sys.platform == "linux":
        file_descriptor = getattr(memory, "_fd", None)
        if type(file_descriptor) is not int or file_descriptor < 0:
            raise RuntimeError("unsupported CPython shared-memory file descriptor")
        os.posix_fallocate(file_descriptor, 0, size)
