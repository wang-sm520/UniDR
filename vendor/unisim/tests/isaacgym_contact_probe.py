"""Diagnostic native G1 contact trace; worker branch remains Python 3.8 compatible.

The optional term ablations isolate SDK setter failures, not acceptance tests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_PREFIX = "UNISIM_GYM_CONTACT_TRACE "


def _worker_main():
    path = Path(__file__).resolve().parents[1] / "src/unisim/backend/isaacgym/worker.py"
    spec = importlib.util.spec_from_file_location("contact_probe_worker", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module._WorkerContext

    class TracedWorker(original):
        def _commit_pending_reset(self):
            record = {
                "kind": "reset_commit",
                "root_before": self._root_state.view(self.num_envs, 13).cpu().numpy().tolist(),
                "indices": sorted(self._pending_reset_env_ids),
            }
            super()._commit_pending_reset()
            record["root_after"] = self._root_state.view(self.num_envs, 13).cpu().numpy().tolist()
            if record["indices"]:
                print(_PREFIX + json.dumps(record), file=sys.stderr, flush=True)

        def init_sim(self, payload):
            result = super().init_sim(payload)
            params = self.gym.get_sim_params(self.sim)
            ranges = self.gym.get_actor_rigid_body_shape_indices(
                self.env_handles[0], self.actor_handles[0]
            )
            shapes = self.gym.get_actor_rigid_shape_properties(
                self.env_handles[0], self.actor_handles[0]
            )
            record = {
                "kind": "init",
                "dt": params.dt,
                "substeps": params.substeps,
                "contact_collection": int(params.physx.contact_collection),
                "gpu_physx": params.physx.use_gpu,
                "gpu_pipeline": params.use_gpu_pipeline,
                "native_body_names": self.body_names,
                "native_body_for_contract": self.native_body_for_contract.tolist(),
                "shape_ranges": [{"start": value.start, "count": value.count} for value in ranges],
                "shapes": [
                    {
                        "filter": shape.filter,
                        "contact_offset": shape.contact_offset,
                        "rest_offset": shape.rest_offset,
                        "friction": shape.friction,
                    }
                    for shape in shapes
                ],
            }
            print(_PREFIX + json.dumps(record), file=sys.stderr, flush=True)
            return result

        def step(self, payload):
            result = super().step(payload)
            record = {
                "kind": "step",
                "root": self._root_state.view(self.num_envs, 13).cpu().numpy().tolist(),
                "body_positions": self._body_state.view(self.num_envs, self.num_bodies, 13)
                .cpu()
                .numpy()[:, :, :3]
                .tolist(),
                "forces": self._contact_force.view(self.num_envs, self.num_bodies, 3)
                .cpu()
                .numpy()
                .tolist(),
                "contact_valid": self._contact_valid.tolist(),
            }
            print(_PREFIX + json.dumps(record), file=sys.stderr, flush=True)
            return result

    module._WorkerContext = TracedWorker
    return module.main(sys.argv[1:])


def _host_main():
    from unisim.backend.isaacgym.backend import IsaacGymBackend
    from unisim.dr.types import ResetRandomizationPayload
    from unisim.scene import SceneCfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--terms", choices=("all", "mass", "gains", "none"), default="all")
    args = parser.parse_args()
    assert args.output.is_absolute()
    scene = (
        Path(__file__).resolve().parents[2] / "UniLab/src/unilab/assets/robots/g1/scene_flat.xml"
    )

    class TracedBackend(IsaacGymBackend):
        def _worker_entrypoint(self):
            return Path(__file__).resolve()

    backend = TracedBackend(
        SceneCfg(model_file=str(scene), default_keyframe_name="stand"),
        4,
        1.0 / 150.0,
        base_name="pelvis",
        device_id=0,
        worker_timeout_s=120,
    )
    report = {"scene": str(scene), "host": []}
    try:
        backend.materialize()
        rows = np.arange(4, dtype=np.int32)
        joints = backend.get_actuator_joint_names()
        foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        foot_ids = backend.get_body_ids(foot_names)
        qpos = np.tile(np.asarray(backend.get_keyframe_qpos("stand"), dtype=np.float32), (4, 1))
        qpos[[0, 2], 2] += 2.0
        qvel = np.tile(backend.get_init_qvel(), (4, 1))
        kp, kd = (np.tile(values, (4, 1)) for values in backend.get_actuator_gains())
        backend.set_state(
            rows,
            qpos,
            qvel,
            randomization=(
                None
                if args.terms == "none"
                else ResetRandomizationPayload(
                    body_mass=(
                        np.tile(backend.get_body_mass(), (4, 1))
                        if args.terms in ("all", "mass")
                        else None
                    ),
                    kp=kp if args.terms in ("all", "gains") else None,
                    kd=kd if args.terms in ("all", "gains") else None,
                )
            ),
        )
        ctrl = qpos[:, backend.get_joint_state_qpos_indices(joints)].copy()
        sensors = tuple(
            "%s_foot_contact_%d" % (side, corner)
            for side in ("left", "right")
            for corner in range(4)
        )
        report["requested_root"] = qpos[:, :7].tolist()
        report["contract_foot_ids"] = foot_ids.tolist()
        for step in range(args.steps):
            backend.step(ctrl.copy(), nsteps=1)
            report["host"].append(
                {
                    "step": step,
                    "root": backend.get_base_pos().copy().tolist(),
                    "feet": backend.get_body_pos_w(foot_ids).copy().tolist(),
                    "forces": backend._slots["contact_force"].copy().tolist(),
                    "sensors": backend.get_sensor_data_batch(sensors).copy().tolist(),
                }
            )
    finally:
        if backend._stderr_file is not None:
            backend._stderr_file.seek(0)
            stderr = backend._stderr_file.read().decode(errors="replace")
            report["native"] = [
                json.loads(line[len(_PREFIX) :])
                for line in stderr.splitlines()
                if line.startswith(_PREFIX)
            ]
            args.output.with_suffix(".stderr.log").write_text(stderr)
        backend.close()
        args.output.write_text(json.dumps(report, indent=2))
    print(str(args.output))


if __name__ == "__main__":
    if "--protocol" in sys.argv:
        sys.exit(_worker_main())
    _host_main()
