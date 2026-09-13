"""Python 3.8 worker process for the out-of-process IsaacGym backend.

PYTHON 3.8 COMPATIBILITY: IsaacGym (Preview 4, EOL) only supports Python
3.6-3.8, so this file runs on the dedicated ``hsgym`` conda interpreter.  Keep
it stdlib + numpy + torch + isaacgym only, and never import ``unilab`` — the
shared protocol module is loaded by file path (``--protocol``) because the
worker interpreter has no access to the main environment's site-packages.

Message loop: read one framed command from stdin, dispatch, write one framed
reply to stdout.  Bulk state crosses the process boundary through shared
memory slots declared by the host (see ``protocol.slot_shapes``); the pipe
only carries commands, metadata, and error payloads.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np


def _load_protocol(path: str) -> Any:
    """Load the shared protocol module by file path (no package import)."""
    spec = importlib.util.spec_from_file_location("unisim_subprocess_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _WorkerContext:
    """Owns the IsaacGym sim, tensor views, and attached shared-memory slots."""

    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol
        # IsaacGym/torch modules are imported inside init_sim; they do not
        # exist on the host interpreter, so these stay ``Any``.
        self.gymapi: Any = None
        self.gymtorch: Any = None
        self.torch: Any = None
        self.gym: Any = None
        self.sim: Any = None
        self.num_envs = 0
        self.num_dof = 0
        self.num_bodies = 0
        self.sim_dt = 0.0
        self.device = "cpu"
        self.use_gpu_pipeline = False
        self.env_handles: List[Any] = []
        self.actor_handles: List[Any] = []
        self.slots: Dict[str, np.ndarray] = {}
        self._shm_handles: List[Any] = []
        self._root_state: Any = None
        self._dof_state: Any = None
        self._body_state: Any = None
        self._contact_force: Any = None
        self._tensors_need_refresh = True
        self._pending_reset_env_ids: Set[int] = set()
        self._reset_actor_indices: Any = None
        self._body_properties: List[Any] = []
        self._dof_properties: List[Any] = []
        self._reset_metadata: Dict[str, Any] = {}
        self._contact_valid = np.empty(0, dtype=bool)
        # Native rendering state (viewer and/or camera sensor).  Both live in
        # this process because the sim handle does.
        self.graphics_device_id = -1
        self.viewer: Any = None
        self.camera_handle: Any = None
        self.camera_env: Any = None
        self.camera_width = 0
        self.camera_height = 0
        # Defaults match the repository's MuJoCo playback camera convention
        # (elevation is the height angle above the horizon, in degrees).
        self.camera_distance = 2.0
        self.camera_elevation_deg = 20.0
        self.camera_azimuth_deg = 90.0

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #

    def init_sim(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create PhysX with CPU state tensors and the requested solver device.

        Preview 4 runtime rigid-body mass writes invalidate root resets on the
        GPU tensor pipeline, even for unchanged masses. CPU tensors preserve
        those resets while PhysX still runs on the requested GPU. The IPC
        boundary already exchanges host arrays; no physics step is added.
        """
        self.protocol.validate_init(payload)
        isaacgym_python = payload["isaacgym_python"]
        if isaacgym_python not in sys.path:
            sys.path.insert(0, isaacgym_python)
        # isaacgym must be imported before torch (it enforces this itself).
        from isaacgym import gymapi, gymtorch  # noqa: PLC0415, I001
        import torch  # noqa: PLC0415

        self.gymapi = gymapi
        self.gymtorch = gymtorch
        self.torch = torch

        gymapi = self.gymapi
        self.num_envs = int(payload["num_envs"])
        self.sim_dt = float(payload["sim_dt"])
        device_id = int(payload.get("device_id", 0))
        self.use_gpu_pipeline = False
        self.device = "cpu"

        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = self.sim_dt
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = device_id >= 0
        sim_params.use_gpu_pipeline = self.use_gpu_pipeline
        # The graphics context is enabled whenever the sim runs on a GPU
        # device.  It opens no window by itself (only create_viewer does) and
        # is required for both the interactive viewer and headless camera
        # capture; the cost for training-only runs is negligible. CPU-only
        # sims get no graphics context and fail closed on render requests.
        self.graphics_device_id = device_id if device_id >= 0 else -1
        self.sim = self.gym.create_sim(
            device_id, self.graphics_device_id, gymapi.SIM_PHYSX, sim_params
        )
        if self.sim is None:
            raise RuntimeError(
                "isaacgym create_sim failed (device_id=%d, gpu_pipeline=%s)"
                % (device_id, self.use_gpu_pipeline)
            )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        model_file = os.fspath(payload["model_file"])
        asset_root, asset_file = os.path.split(model_file)
        if not asset_file.lower().endswith((".xml", ".mjcf")):
            raise RuntimeError(
                "isaacgym backend currently loads MJCF scenes only; got asset file "
                "%r. Convert the task scene or extend the worker asset loader." % asset_file
            )
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if asset is None:
            raise RuntimeError(
                "isaacgym load_asset failed for %r. MJCF import requires the file to be "
                "self-contained for IsaacGym's importer (some MuJoCo elements are "
                "unsupported); run the worker command manually for the importer log." % model_file
            )

        self.num_dof = int(self.gym.get_asset_dof_count(asset))
        self.num_bodies = int(self.gym.get_asset_rigid_body_count(asset))
        self.dof_names: List[str] = list(self.gym.get_asset_dof_names(asset))
        self.body_names = list(self.gym.get_asset_rigid_body_names(asset))
        self.contract_joint_names = list(payload["mjcf_joint_names"])
        self.contract_body_names = list(payload["mjcf_body_names"])
        self.native_joint_for_contract = self.protocol.name_permutation(
            self.dof_names, self.contract_joint_names, "joint"
        )
        self.native_body_for_contract = self.protocol.name_permutation(
            self.body_names, self.contract_body_names, "body"
        )
        self._contact_valid = np.zeros(self.num_envs, dtype=bool)

        dof_props = self.gym.get_asset_dof_properties(asset)
        # Position-controlled dofs: ctrl is the per-dof position target,
        # matching MuJoCo <position kp kv forcerange> actuator semantics
        # (PhysX applies force = kp * (target - pos) - kv * vel, clamped to
        # the symmetric effort limit).  All parameters come from the host's
        # MJCF scan because the importer drops kv/frictionloss/joint ranges.
        self._apply_actuator_props(dof_props, payload)

        spacing = 2.0
        num_per_row = max(1, int(np.ceil(np.sqrt(self.num_envs))))
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, 0.0)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        for env_index in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            # collision_group=env_index isolates envs; filter=1 disables
            # self-collision.  The MJCF <contact><exclude> pairs (e.g. G1's
            # elbow/wrist and pelvis/hip overlaps) cannot be reproduced
            # per-link-pair through gymapi, and with self-collision on those
            # overlapping capsules generate permanent contact forces that
            # destabilize the drives.  Disabling self-collision is the
            # ecosystem-standard approximation (legged_gym, MetaSim) and a
            # superset of the MJCF exclusions.
            actor_handle = self.gym.create_actor(env_handle, asset, pose, "robot", env_index, 1)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self.env_handles.append(env_handle)
            self.actor_handles.append(actor_handle)

        self._cache_reset_properties()
        self.gym.prepare_sim(self.sim)
        self._acquire_tensors()
        self._refresh_tensors()

        keyframe_qpos = payload.get("keyframe_qpos")
        if keyframe_qpos is not None:
            # Apply the scene's task-initial pose (AGENTS.md: the keyframe is
            # the task initial state) so the post-INIT state matches the
            # host-side get_default_qpos()/get_default_dof_pos() contract.
            self._apply_initial_keyframe(keyframe_qpos, payload.get("mjcf_joint_names") or [])
        lower = np.asarray(dof_props["lower"], dtype=np.float64)
        upper = np.asarray(dof_props["upper"], dtype=np.float64)
        effort = np.asarray(dof_props["effort"], dtype=np.float64)
        return {
            **self._reset_metadata,
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            "dof_names": list(self.contract_joint_names),
            "body_names": list(self.contract_body_names),
            "dof_lower": lower[self.native_joint_for_contract].tolist(),
            "dof_upper": upper[self.native_joint_for_contract].tolist(),
            "effort": effort[self.native_joint_for_contract].tolist(),
            "gravity": [0.0, 0.0, -9.81],
            "use_gpu_pipeline": self.use_gpu_pipeline,
            "graphics_enabled": self.graphics_device_id >= 0,
        }

    def _cache_reset_properties(self) -> None:
        self._body_properties = [
            self.gym.get_actor_rigid_body_properties(env, actor)
            for env, actor in zip(self.env_handles, self.actor_handles)
        ]
        self._dof_properties = [
            self.gym.get_actor_dof_properties(env, actor)
            for env, actor in zip(self.env_handles, self.actor_handles)
        ]
        masses = np.asarray([prop.mass for prop in self._body_properties[0]])
        self._reset_metadata = {
            "protocol_version": self.protocol.PROTOCOL_VERSION,
            "supported_reset_terms": list(self.protocol.RESET_TERMS),
            "contact_reporter": self.protocol.CONTACT_REPORTER_ISAACGYM,
            "nominal_body_mass": masses[self.native_body_for_contract].tolist(),
            "nominal_kp": self._dof_properties[0]["stiffness"][
                self.native_joint_for_contract
            ].tolist(),
            "nominal_kd": self._dof_properties[0]["damping"][
                self.native_joint_for_contract
            ].tolist(),
        }

    def _apply_reset_randomization(
        self,
        env_ids: np.ndarray,
        values: Dict[str, np.ndarray],
    ) -> None:
        for row, env_id in enumerate(env_ids):
            env = self.env_handles[env_id]
            actor = self.actor_handles[env_id]
            if "body_mass" in values:
                properties = self._body_properties[env_id]
                for column, native_id in enumerate(self.native_body_for_contract):
                    properties[native_id].mass = float(values["body_mass"][row, column])
                if not self.gym.set_actor_rigid_body_properties(env, actor, properties, False):
                    raise RuntimeError("IsaacGym rejected selected-environment body mass reset")
            if "kp" in values or "kd" in values:
                properties = self._dof_properties[env_id]
                for term, field in (("kp", "stiffness"), ("kd", "damping")):
                    if term in values:
                        properties[field][self.native_joint_for_contract] = values[term][row]
                if not self.gym.set_actor_dof_properties(env, actor, properties):
                    raise RuntimeError("IsaacGym rejected selected-environment gain reset")

    def _apply_actuator_props(self, dof_props: Any, payload: Dict[str, Any]) -> None:
        """Set per-dof PD/limit/dynamics properties from the host MJCF scan.

        The host sends arrays in MJCF joint document order
        (``mjcf_joint_names``); they are mapped onto the asset's dofs by NAME,
        because IsaacGym's MJCF importer is free to reorder joints.
        """
        gymapi = self.gymapi
        joint_names = [str(name) for name in (payload.get("mjcf_joint_names") or [])]
        if len(joint_names) != self.num_dof:
            raise RuntimeError(
                "mjcf_joint_names has %d entries but the asset exposes %d dofs; "
                "IsaacGym's MJCF importer must preserve one dof per single-DoF joint"
                % (len(joint_names), self.num_dof)
            )
        index_by_name = {}
        for index, name in enumerate(joint_names):
            index_by_name[name] = index
        fields = (
            ("stiffness", payload["dof_stiffness"]),
            ("damping", payload["dof_damping"]),
            ("effort", payload["dof_effort"]),
            ("armature", payload["dof_armature"]),
            ("friction", payload["dof_friction"]),
        )
        for dof_index, dof_name in enumerate(self.dof_names):
            if dof_name not in index_by_name:
                raise RuntimeError(
                    "isaacgym asset dof %r is missing from mjcf_joint_names; the MJCF "
                    "importer may have dropped or renamed the joint" % dof_name
                )
            source = index_by_name[dof_name]
            dof_props["driveMode"][dof_index] = int(gymapi.DOF_MODE_POS)
            for field, values in fields:
                dof_props[field][dof_index] = float(values[source])

    def _apply_initial_keyframe(self, qpos_values: Any, joint_names: Any) -> None:
        """Write the scene keyframe pose into every env via the tensor API.

        ``qpos_values`` follows the MJCF layout: 7 free-root columns
        (xyz + wxyz quat) plus one column per single-DoF joint in document
        order (``joint_names``).  DoF values are mapped onto the asset's dofs
        by NAME, because IsaacGym's MJCF importer is free to reorder joints.
        """
        protocol = self.protocol
        torch = self.torch
        qpos = np.asarray(qpos_values, dtype=np.float32).reshape(-1)
        expected = 7 + self.num_dof
        if qpos.size != expected:
            raise RuntimeError(
                "keyframe qpos has %d entries; expected %d (7 root + %d dofs)"
                % (qpos.size, expected, self.num_dof)
            )
        joint_names = [str(name) for name in joint_names]
        if len(joint_names) != self.num_dof:
            raise RuntimeError(
                "mjcf_joint_names has %d entries but the asset exposes %d dofs; "
                "IsaacGym's MJCF importer must preserve one dof per single-DoF joint"
                % (len(joint_names), self.num_dof)
            )
        index_by_name = {}
        for index, name in enumerate(joint_names):
            index_by_name[name] = index
        dof_pos = np.zeros((self.num_envs, self.num_dof), dtype=np.float32)
        for dof_index, dof_name in enumerate(self.dof_names):
            if dof_name not in index_by_name:
                raise RuntimeError(
                    "isaacgym asset dof %r is missing from mjcf_joint_names; the MJCF "
                    "importer may have dropped or renamed the joint" % dof_name
                )
            dof_pos[:, dof_index] = qpos[7 + index_by_name[dof_name]]

        root = np.zeros((self.num_envs, 13), dtype=np.float32)
        root[:, 0:3] = qpos[0:3]
        root[:, 3:7] = protocol.wxyz_to_xyzw(qpos[None, 3:7])
        root_view = self._root_state.view(self.num_envs, -1, 13)
        root_view[:, 0, :] = torch.from_numpy(root).to(self.device)
        dof = np.zeros((self.num_envs, self.num_dof, 2), dtype=np.float32)
        dof[:, :, 0] = dof_pos
        dof_view = self._dof_state.view(self.num_envs, self.num_dof, 2)
        dof_view[:, :, :] = torch.from_numpy(dof).to(self.device)
        self._pending_reset_env_ids.update(range(self.num_envs))

    def _acquire_tensors(self) -> None:
        gym = self.gym
        gymtorch = self.gymtorch
        self._root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(self.sim))
        self._dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(self.sim))
        self._body_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(self.sim))
        self._contact_force = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(self.sim))

    # ------------------------------------------------------------------ #
    # Shared-memory slots
    # ------------------------------------------------------------------ #

    def attach_slots(self, payload: Dict[str, Any]) -> None:
        """Attach host-created shm slots and detach them from resource tracking.

        Python's shared_memory resource tracker would otherwise unlink the
        host-owned segments when this worker exits (CPython issue 39959), so
        every attached name is unregistered here; the host owns unlinking.
        """
        from multiprocessing import resource_tracker, shared_memory  # noqa: PLC0415

        self.protocol.validate_slot_specs(payload, self.num_envs, self.num_dof, self.num_bodies)
        for name, spec in payload["slots"].items():
            handle = shared_memory.SharedMemory(name=spec["shm"], create=False)
            resource_tracker.unregister(handle._name, "shared_memory")  # type: ignore[attr-defined]
            array = np.ndarray(
                tuple(spec["shape"]), dtype=np.dtype(spec["dtype"]), buffer=handle.buf
            )
            self.slots[name] = array
            self._shm_handles.append(handle)
        self.refresh_state_slots()

    # ------------------------------------------------------------------ #
    # State exchange
    # ------------------------------------------------------------------ #

    def _refresh_tensors(self) -> None:
        if not self._tensors_need_refresh:
            return
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._tensors_need_refresh = False

    def _commit_pending_reset(self) -> None:
        """Submit each GPU setter once per simulate, retaining its index storage."""
        if not self._pending_reset_env_ids:
            return
        indices = np.asarray(sorted(self._pending_reset_env_ids), dtype=np.int32)
        self._reset_actor_indices = self.torch.from_numpy(indices).to(self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            self.gymtorch.unwrap_tensor(self._root_state),
            self.gymtorch.unwrap_tensor(self._reset_actor_indices),
            len(indices),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            self.gymtorch.unwrap_tensor(self._dof_state),
            self.gymtorch.unwrap_tensor(self._reset_actor_indices),
            len(indices),
        )
        self._pending_reset_env_ids.clear()

    def refresh_state_slots(self) -> None:
        """Copy the latest tensor state into every host-visible shm slot."""
        protocol = self.protocol
        self._refresh_tensors()
        root = self._root_state.view(self.num_envs, -1, 13)[:, 0, :].cpu().numpy()
        root_slot = self.slots["root_state"]
        root_slot[:, 0:3] = root[:, 0:3]
        root_slot[:, 3:7] = protocol.xyzw_to_wxyz(root[:, 3:7])
        root_slot[:, 7:13] = root[:, 7:13]
        np.copyto(
            self.slots["dof_state"],
            self._dof_state.view(self.num_envs, self.num_dof, 2)
            .cpu()
            .numpy()[:, self.native_joint_for_contract, :],
        )
        bodies = (
            self._body_state.view(self.num_envs, self.num_bodies, 13)
            .cpu()
            .numpy()[:, self.native_body_for_contract, :]
        )
        body_slot = self.slots["body_state"]
        body_slot[:, :, 0:3] = bodies[:, :, 0:3]
        body_slot[:, :, 3:7] = protocol.xyzw_to_wxyz(bodies[:, :, 3:7])
        body_slot[:, :, 7:13] = bodies[:, :, 7:13]
        np.copyto(
            self.slots["contact_force"],
            self._contact_force.view(self.num_envs, self.num_bodies, 3)
            .cpu()
            .numpy()[:, self.native_body_for_contract, :],
        )
        self.slots["contact_force"][~self._contact_valid] = 0.0

    def step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        nsteps = int(payload["nsteps"])
        if nsteps <= 0:
            raise ValueError("nsteps must be positive, got %d" % nsteps)
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        self._commit_pending_reset()
        native_ctrl = np.empty_like(self.slots["ctrl"])
        native_ctrl[:, self.native_joint_for_contract] = self.slots["ctrl"]
        torch_ctrl = self.torch.from_numpy(native_ctrl).to(self.device)
        # ctrl carries per-dof position targets (MuJoCo <position> actuator
        # semantics); PhysX runs the PD loop with the INIT-time kp/kv/effort.
        self.gym.set_dof_position_target_tensor(
            self.sim, self.gymtorch.unwrap_tensor(torch_ctrl.reshape(-1).contiguous())
        )
        timings["control_upload_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        for _ in range(nsteps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        self._tensors_need_refresh = True
        self._contact_valid[:] = True
        timings["physics_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self.refresh_state_slots()
        timings["state_refresh_ms"] = (time.perf_counter() - t0) * 1000.0
        return {"timing": timings}

    def set_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        protocol = self.protocol
        torch = self.torch
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        env_ids, qpos, qvel, values = protocol.read_reset(
            payload, self.slots, self.num_envs, self.num_dof, self.num_bodies
        )
        count = len(env_ids)
        if count == 0:
            return {"timing": timings}
        self._apply_reset_randomization(env_ids, values)

        root = np.zeros((count, 13), dtype=np.float32)
        root[:, 0:3] = qpos[:, 0:3]
        root[:, 3:7] = protocol.wxyz_to_xyzw(qpos[:, 3:7])
        root[:, 7:10] = qvel[:, 0:3]
        # Contract qvel carries body-frame angular velocity; IsaacGym root
        # states take world-frame angular velocity.
        root[:, 10:13] = protocol.quat_rotate(qpos[:, 3:7], qvel[:, 3:6]).astype(np.float32)
        env_id_tensor = torch.from_numpy(env_ids.astype(np.int32)).to(self.device)
        root_view = self._root_state.view(self.num_envs, -1, 13)
        root_view[env_id_tensor.long(), 0, :] = torch.from_numpy(root).to(self.device)

        dof = np.zeros((count, self.num_dof, 2), dtype=np.float32)
        dof[:, self.native_joint_for_contract, 0] = qpos[:, 7 : 7 + self.num_dof]
        dof[:, self.native_joint_for_contract, 1] = qvel[:, 6 : 6 + self.num_dof]
        dof_view = self._dof_state.view(self.num_envs, self.num_dof, 2)
        dof_view[env_id_tensor.long(), :, :] = torch.from_numpy(dof).to(self.device)
        self._pending_reset_env_ids.update(env_ids.tolist())
        timings["set_state_reset_upload_ms"] = (time.perf_counter() - t0) * 1000.0

        # IsaacGym has no kinematics-only forward call; root/dof slots reflect
        # the applied state immediately, while body/contact slots stay as of
        # the last physics step until the next STEP.
        t0 = time.perf_counter()
        self._contact_valid[env_ids] = False
        self.refresh_state_slots()
        timings["set_state_host_cache_refresh_ms"] = (time.perf_counter() - t0) * 1000.0
        return {"timing": timings}

    def get_meta(self) -> Dict[str, Any]:
        return {
            **self._reset_metadata,
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            "use_gpu_pipeline": self.use_gpu_pipeline,
            "graphics_enabled": self.graphics_device_id >= 0,
        }

    # ------------------------------------------------------------------ #
    # Native rendering (viewer + camera sensor)
    # ------------------------------------------------------------------ #

    def _require_graphics(self) -> None:
        if self.graphics_device_id < 0:
            raise RuntimeError(
                "isaacgym rendering requires a GPU sim (device_id >= 0); this sim was "
                "created without a graphics context"
            )

    def init_renderer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create the interactive viewer and/or the headless capture camera."""
        gym = self.gym
        gymapi = self.gymapi
        self._require_graphics()
        headless = bool(payload.get("headless", False))
        capture = bool(payload.get("capture", False))

        if not headless and self.viewer is None:
            viewer = gym.create_viewer(self.sim, gymapi.CameraProperties())
            if viewer is None:
                raise RuntimeError(
                    "isaacgym create_viewer failed (no display reachable); use "
                    "play_render_mode=record for headless video capture"
                )
            # Default view: env 0 area, slightly above the grid.
            gym.viewer_camera_look_at(
                viewer,
                None,
                gymapi.Vec3(2.5, 2.5, 1.8),
                gymapi.Vec3(0.0, 0.0, 0.5),
            )
            self.viewer = viewer

        if capture and self.camera_handle is None:
            camera = payload.get("camera") or {}
            self.camera_distance = float(camera.get("distance", 2.0))
            self.camera_elevation_deg = float(camera.get("elevation_deg", 20.0))
            self.camera_azimuth_deg = float(camera.get("azimuth_deg", 90.0))
            self.camera_width = int(payload.get("width", 1280))
            self.camera_height = int(payload.get("height", 720))
            cam_props = gymapi.CameraProperties()
            cam_props.width = self.camera_width
            cam_props.height = self.camera_height
            self.camera_env = self.env_handles[0]
            self.camera_handle = gym.create_camera_sensor(self.camera_env, cam_props)
            self._position_tracking_camera()

        return {
            "viewer": self.viewer is not None,
            "capture": self.camera_handle is not None,
        }

    def _position_tracking_camera(self) -> None:
        """Aim the capture camera at env 0's root on a spherical offset."""
        import math  # noqa: PLC0415

        gymapi = self.gymapi
        root = self._root_state.view(self.num_envs, -1, 13)[0, 0, :].cpu().numpy()
        target = np.asarray(root[0:3], dtype=np.float64)
        elevation = math.radians(self.camera_elevation_deg)
        azimuth = math.radians(self.camera_azimuth_deg)
        offset = self.camera_distance * np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        eye = target + offset
        self.gym.set_camera_location(
            self.camera_handle,
            self.camera_env,
            gymapi.Vec3(float(eye[0]), float(eye[1]), float(eye[2])),
            gymapi.Vec3(float(target[0]), float(target[1]), float(target[2])),
        )

    def render_frame(self) -> Dict[str, Any]:
        """Draw one viewer frame; report whether the user closed the window."""
        if self.viewer is None:
            raise RuntimeError("isaacgym viewer is not initialized; call INIT_RENDERER first")
        gym = self.gym
        if gym.query_viewer_has_closed(self.viewer):
            self._destroy_viewer()
            return {"closed": True}
        gym.step_graphics(self.sim)
        gym.draw_viewer(self.viewer, self.sim, True)
        if gym.query_viewer_has_closed(self.viewer):
            self._destroy_viewer()
            return {"closed": True}
        return {"closed": False}

    def capture_frame(self) -> Dict[str, Any]:
        """Render the capture camera and return one RGB uint8 frame."""
        if self.camera_handle is None:
            raise RuntimeError(
                "isaacgym capture camera is not initialized; call INIT_RENDERER first"
            )
        gym = self.gym
        self._position_tracking_camera()
        gym.step_graphics(self.sim)
        gym.render_all_camera_sensors(self.sim)
        image = np.asarray(
            gym.get_camera_image(
                self.sim, self.camera_env, self.camera_handle, self.gymapi.IMAGE_COLOR
            )
        )
        frame = np.ascontiguousarray(
            image.reshape(self.camera_height, self.camera_width, 4)[:, :, :3]
        )
        return {
            "frame": frame,
            "width": self.camera_width,
            "height": self.camera_height,
        }

    def _destroy_viewer(self) -> None:
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None

    def shutdown(self) -> None:
        if self.gym is not None:
            self._destroy_viewer()
        if self.gym is not None and self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
        for handle in self._shm_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._shm_handles = []


def _dispatch(ctx: _WorkerContext, protocol: Any, cmd: str, payload: Any) -> Tuple[str, Any]:
    if cmd == protocol.CMD_INIT:
        return protocol.CMD_META, ctx.init_sim(payload)
    if cmd == protocol.CMD_ATTACH:
        ctx.attach_slots(payload)
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_STEP:
        return protocol.CMD_READY, ctx.step(payload)
    if cmd == protocol.CMD_SET_STATE:
        return protocol.CMD_READY, ctx.set_state(payload)
    if cmd == protocol.CMD_REFRESH:
        ctx.refresh_state_slots()
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_GET_META:
        return protocol.CMD_META, ctx.get_meta()
    if cmd == protocol.CMD_INIT_RENDERER:
        return protocol.CMD_META, ctx.init_renderer(payload)
    if cmd == protocol.CMD_RENDER_FRAME:
        return protocol.CMD_META, ctx.render_frame()
    if cmd == protocol.CMD_CAPTURE_FRAME:
        return protocol.CMD_META, ctx.capture_frame()
    raise ValueError(f"unknown command {cmd!r}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, help="path to protocol.py")
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol)
    ctx = _WorkerContext(protocol)

    stdin = sys.stdin.buffer
    # IsaacGym's native extension prints banners straight to fd 1, which would
    # corrupt the framed protocol. Keep a private copy of the original stdout
    # for protocol messages and reroute fd 1 (and with it sys.stdout) to
    # stderr, where the parent captures it for crash diagnostics.
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    stdout = protocol_out
    while True:
        try:
            message = protocol.recv_message(stdin)
        except (EOFError, protocol.WorkerDisconnectedError):
            return 0
        cmd = message["cmd"]
        payload = message.get("payload")
        if cmd == protocol.CMD_SHUTDOWN:
            try:
                ctx.shutdown()
            finally:
                protocol.send_message(stdout, protocol.CMD_READY)
            return 0
        try:
            reply_cmd, reply_payload = _dispatch(ctx, protocol, cmd, payload)
        except Exception as exc:  # noqa: BLE001 - every worker error crosses the wire
            protocol.send_message(stdout, protocol.CMD_ERROR, protocol.serialize_exception(exc))
            continue
        protocol.send_message(stdout, reply_cmd, reply_payload)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
