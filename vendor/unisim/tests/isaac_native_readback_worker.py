"""Opt-in vendor property readback probe; Python 3.8 compatible, not a physics effect gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


def native_properties(worker, kind):
    if kind == "isaacsim":
        view = worker.robot.root_physx_view
        return {
            "mass": as_numpy(view.get_masses())[:, worker.native_body_for_contract],
            "kp": as_numpy(view.get_dof_stiffnesses())[:, worker.native_joint_for_contract],
            "kd": as_numpy(view.get_dof_dampings())[:, worker.native_joint_for_contract],
            "inertia": as_numpy(view.get_inertias()),
        }
    masses, stiffness, damping, inertia = [], [], [], []
    for env, actor in zip(worker.env_handles, worker.actor_handles):
        bodies = worker.gym.get_actor_rigid_body_properties(env, actor)
        joints = worker.gym.get_actor_dof_properties(env, actor)
        masses.append([body.mass for body in bodies])
        stiffness.append(joints["stiffness"])
        damping.append(joints["damping"])
        inertia.append(
            [
                [
                    getattr(getattr(body.inertia, axis), component)
                    for axis in ("x", "y", "z")
                    for component in ("x", "y", "z")
                ]
                for body in bodies
            ]
        )
    return {
        "mass": np.asarray(masses)[:, worker.native_body_for_contract],
        "kp": np.asarray(stiffness)[:, worker.native_joint_for_contract],
        "kd": np.asarray(damping)[:, worker.native_joint_for_contract],
        "inertia": np.asarray(inertia),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "src" / "unisim" / "backend"
    protocol = load_module(root / "subprocess_ipc" / "protocol.py", "readback_protocol")
    module = load_module(root / args.backend / "worker.py", "readback_worker")
    payload = json.loads(Path(args.config).read_text())
    worker = module._WorkerContext(protocol)
    try:
        meta = worker.init_sim(payload)
        protocol.validate_worker_metadata(meta)
        worker.slots = {
            name: np.zeros(shape, dtype=protocol.slot_dtype(name))
            for name, shape in protocol.slot_shapes(3, worker.num_dof, worker.num_bodies).items()
        }
        worker.refresh_state_slots()
        before = native_properties(worker, args.backend)
        worker.slots["reset_env_ids"][:2] = [2, 0]
        worker.slots["reset_qpos"][:2] = payload["keyframe_qpos"]
        worker.slots["reset_qvel"][:] = 0
        mass = np.asarray(meta["nominal_body_mass"])
        worker.slots["reset_body_mass"][:2] = mass[None, :] * np.array([[2], [1.5]])
        worker.slots["reset_kp"][:2] = [[12, 34], [56, 78]]
        worker.slots["reset_kd"][:2] = [[1, 3], [5, 7]]
        for _ in range(2):
            worker.set_state({"count": 2, "randomization_terms": list(protocol.RESET_TERMS)})
            current = native_properties(worker, args.backend)
            for term, field in (("body_mass", "mass"), ("kp", "kp"), ("kd", "kd")):
                np.testing.assert_allclose(
                    current[field][[2, 0]], worker.slots["reset_" + term][:2]
                )
                np.testing.assert_allclose(current[field][1], before[field][1])
            np.testing.assert_allclose(current["inertia"], before["inertia"], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(worker.get_meta()["nominal_body_mass"], mass)
            assert not worker.slots["contact_force"][[2, 0]].any()
        print(
            "UNISIM_ISAAC_READBACK_RESULT "
            + json.dumps(
                {
                    "backend": args.backend,
                    "status": "passed",
                    "physical_effects_validated": False,
                }
            ),
            flush=True,
        )
    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()
