"""IPC primitives for multi-process RL training."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uni_rl.ipc.async_runner import AsyncRunner
    from uni_rl.ipc.replay_buffer import ReplayBuffer
    from uni_rl.ipc.rollout_ring_buffer import RolloutRingBuffer
    from uni_rl.ipc.weight_sync import SharedWeightSync


def __getattr__(name: str) -> Any:
    """Load torch-based primitives only when requested."""
    modules = {
        "AsyncRunner": "async_runner",
        "ReplayBuffer": "replay_buffer",
        "RolloutRingBuffer": "rollout_ring_buffer",
        "SharedWeightSync": "weight_sync",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"uni_rl.ipc.{modules[name]}"), name)
    globals()[name] = value
    return value


__all__ = [
    "SharedWeightSync",
    "RolloutRingBuffer",
    "AsyncRunner",
    "ReplayBuffer",
]
