"""Optional native setter/readback checks; G1 physical acceptance is a separate gate."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.backend import MjcfSubprocessBackend
from unisim.scene import SceneCfg


def _native_worker_launch(kind):
    dependencies = importlib.import_module(f"unisim.backend.{kind}.dependencies")
    runtime = getattr(dependencies, f"resolve_{kind}_runtime")()
    environment = dependencies.build_worker_env(runtime)
    payload = {}
    if kind == "isaacgym":
        payload["isaacgym_python"] = str(runtime.isaacgym_python)
    elif runtime.isaaclab_source is not None:
        payload["isaaclab_source"] = str(runtime.isaaclab_source)
    return str(runtime.python), environment, payload


@pytest.mark.slow
@pytest.mark.optional
@pytest.mark.parametrize("kind", ("isaacgym", "isaacsim"))
def test_native_selected_property_readback(kind, tmp_path):
    if os.environ.get("UNISIM_ISAAC_READBACK") != "1":
        pytest.skip("opt in with UNISIM_ISAAC_READBACK=1; this is not the G1 effect gate")
    utility = shutil.which("nvidia-smi")
    assert utility is not None, "native readback BLOCKED: nvidia-smi is unavailable"
    gpu = subprocess.run(
        [utility, "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert gpu.returncode == 0, "native readback BLOCKED: " + gpu.stdout + gpu.stderr
    interpreter, worker_environment, runtime_payload = _native_worker_launch(kind)
    tests = Path(__file__).parent
    model = tests / "assets" / "isaac_reset.xml"
    backend = MjcfSubprocessBackend(scene=SceneCfg(model_file=str(model)), num_envs=3, sim_dt=0.002)
    try:
        metadata = backend._get_scene_metadata()
        payload = {
            "protocol_version": protocol.PROTOCOL_VERSION,
            "required_reset_terms": list(protocol.RESET_TERMS),
            "model_file": str(model),
            "root_body_name": "base",
            "mjcf_body_names": list(metadata.body_names),
            "mjcf_joint_names": list(metadata.joint_names),
            "num_envs": 3,
            "sim_dt": 0.002,
            "device_id": int(os.environ.get("UNISIM_ISAAC_EFFECT_DEVICE", "0")),
            "keyframe_qpos": backend.get_default_qpos().tolist(),
            "render_mode": "none",
            "render_width": 1280,
            "render_height": 720,
            **backend._position_actuation_payload(),
            **runtime_payload,
        }
        config = tmp_path / "native_readback_config.json"
        config.write_text(json.dumps(payload))
        result = subprocess.run(
            [
                interpreter,
                str(tests / "isaac_native_readback_worker.py"),
                "--backend",
                kind,
                "--config",
                str(config),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env=worker_environment,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-12000:]
        prefix = "UNISIM_ISAAC_READBACK_RESULT "
        records = [
            json.loads(line[len(prefix) :])
            for line in result.stdout.splitlines()
            if line.startswith(prefix)
        ]
        assert records == [
            {"backend": kind, "status": "passed", "physical_effects_validated": False}
        ]
    finally:
        backend.close()
