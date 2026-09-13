"""CPU coverage of the separate native acceptance probe's runtime wiring."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from test_isaac_native_readback import _native_worker_launch


@pytest.mark.parametrize("kind", ("isaacgym", "isaacsim"))
def test_native_probe_reuses_worker_runtime(kind, monkeypatch, tmp_path):
    dependencies = importlib.import_module(f"unisim.backend.{kind}.dependencies")
    runtime = SimpleNamespace(
        python=tmp_path / "bin/python",
        isaacgym_python=tmp_path / "isaacgym/python",
        isaaclab_source=tmp_path / "IsaacLab/source",
    )
    expected_environment = {"NATIVE_RUNTIME_TEST": kind}
    selected_runtimes = []
    monkeypatch.setattr(dependencies, f"resolve_{kind}_runtime", lambda: runtime)

    def build_environment(selected):
        selected_runtimes.append(selected)
        return expected_environment

    monkeypatch.setattr(dependencies, "build_worker_env", build_environment)
    interpreter, environment, payload = _native_worker_launch(kind)
    assert interpreter == str(runtime.python)
    assert environment == expected_environment
    assert selected_runtimes == [runtime]
    field = "isaacgym_python" if kind == "isaacgym" else "isaaclab_source"
    assert payload == {field: str(getattr(runtime, field))}
