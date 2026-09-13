"""CPU checks for the native GPU-solver/CPU-tensor reset configuration."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from unisim.backend.isaacgym.worker import _WorkerContext


@pytest.mark.parametrize("device_id", (-1, 0, 2))
def test_solver_device_is_independent_of_cpu_tensor_transport(monkeypatch, device_id):
    class SetupCapturedError(Exception):
        pass

    def create_sim(compute_device, graphics_device, engine, params):
        assert compute_device == graphics_device == device_id
        assert engine == "physx"
        assert params.physx.use_gpu is (device_id >= 0)
        assert params.use_gpu_pipeline is False
        assert params.dt == 1.0 / 150.0
        assert params.substeps == 1
        assert params.physx.num_position_iterations == 4
        assert params.physx.num_velocity_iterations == 1
        assert worker.device == "cpu"
        raise SetupCapturedError

    sdk = ModuleType("isaacgym")
    sdk.gymapi = SimpleNamespace(
        acquire_gym=lambda: SimpleNamespace(create_sim=create_sim),
        SimParams=lambda: SimpleNamespace(physx=SimpleNamespace()),
        Vec3=lambda *values: values,
        UpAxis=SimpleNamespace(UP_AXIS_Z="z"),
        SIM_PHYSX="physx",
    )
    sdk.gymtorch = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "isaacgym", sdk)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    worker = _WorkerContext(SimpleNamespace(validate_init=lambda payload: None))
    with pytest.raises(SetupCapturedError):
        worker.init_sim(
            {
                "isaacgym_python": sys.path[0],
                "num_envs": 4,
                "sim_dt": 1.0 / 150.0,
                "device_id": device_id,
            }
        )
