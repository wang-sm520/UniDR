"""Utility exports with lazy loading of torch-dependent helpers."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uni_rl.utils.device import get_default_device
    from uni_rl.utils.tensor import to_numpy, to_torch


def __getattr__(name: str) -> Any:
    """Keep numpy-only utilities independent of torch at import time."""
    modules = {"get_default_device": "device", "to_numpy": "tensor", "to_torch": "tensor"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"uni_rl.utils.{modules[name]}"), name)
    globals()[name] = value
    return value


__all__ = ["get_default_device", "to_numpy", "to_torch"]
