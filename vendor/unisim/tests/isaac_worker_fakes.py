"""NumPy-only SDK doubles used by the real worker reset/state methods."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np

from unisim.backend.isaacgym.worker import _WorkerContext as GymWorker
from unisim.backend.isaacsim.worker import _WorkerContext as SimWorker
from unisim.backend.subprocess_ipc import protocol


def _index(value):
    if isinstance(value, Tensor):
        return value.array
    if isinstance(value, tuple):
        return tuple(_index(part) for part in value)
    return value


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, index):
        return Tensor(self.array[_index(index)])

    def __setitem__(self, index, value):
        self.array[_index(index)] = np.asarray(value)

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.array

    def to(self, device):
        return self

    def long(self):
        return Tensor(self.array.astype(np.int64))

    def clone(self):
        return Tensor(self.array.copy())

    def view(self, *shape):
        return Tensor(self.array.reshape(*shape))

    reshape = view

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.array))


class Torch:
    long = np.int64
    float32 = np.float32

    @staticmethod
    def from_numpy(value):
        return Tensor(value)

    @staticmethod
    def as_tensor(value, dtype=None, device=None):
        return Tensor(np.asarray(value, dtype=dtype))


class Gym:
    def __init__(self):
        self.mass = np.tile([2, 3, 5], (3, 1)).astype(np.float32)
        self.inertia = np.arange(27, dtype=np.float32).reshape(3, 3, 3) + 1
        self.properties = np.zeros(
            (3, 2),
            dtype=[
                (name, "f4")
                for name in (
                    "stiffness",
                    "damping",
                    "effort",
                    "armature",
                    "friction",
                    "driveMode",
                )
            ],
        )
        self.properties["stiffness"] = [10, 20]
        self.properties["damping"] = [1, 2]
        self.properties["effort"] = [70, 80]
        self.calls = []
        self.reads = 0
        self.reject = None

    def get_actor_rigid_body_properties(self, env, actor):
        self.reads += 1
        return [
            SimpleNamespace(mass=float(mass), inertia=inertia.copy())
            for mass, inertia in zip(self.mass[env], self.inertia[env])
        ]

    def get_actor_dof_properties(self, env, actor):
        self.reads += 1
        return self.properties[env].copy()

    def set_actor_rigid_body_properties(self, env, actor, properties, recompute):
        self.calls.append(("mass", env, actor, recompute))
        assert recompute is False
        if self.reject == "mass":
            return False
        self.mass[env] = [prop.mass for prop in properties]
        np.testing.assert_array_equal(self.inertia[env], [prop.inertia for prop in properties])
        return True

    def set_actor_dof_properties(self, env, actor, properties):
        self.calls.append(("gains", env, actor))
        if self.reject == "gains":
            return False
        self.properties[env] = properties
        return True

    def set_actor_root_state_tensor_indexed(self, sim, values, rows, count):
        self.calls.append(("root", np.asarray(rows).copy()))
        return True

    def set_dof_state_tensor_indexed(self, sim, values, rows, count):
        self.calls.append(("dof", np.asarray(rows).copy()))
        return True

    def set_dof_position_target_tensor(self, sim, targets):
        self.targets = np.asarray(targets).reshape(3, 2).copy()

    def refresh_actor_root_state_tensor(self, sim):
        pass

    refresh_dof_state_tensor = refresh_actor_root_state_tensor
    refresh_rigid_body_state_tensor = refresh_actor_root_state_tensor
    refresh_net_contact_force_tensor = refresh_actor_root_state_tensor
    simulate = refresh_actor_root_state_tensor

    def fetch_results(self, sim, wait):
        pass


class PhysX:
    def __init__(self):
        self.mass = np.tile([2, 3, 5], (3, 1)).astype(np.float32)
        self.inertia = np.arange(81, dtype=np.float32).reshape(3, 3, 9)
        self.calls = []
        self.reads = 0
        self.reject = False

    def get_masses(self):
        self.reads += 1
        return Tensor(self.mass.copy())

    def set_masses(self, masses, indices):
        if self.reject:
            raise RuntimeError("IsaacSim rejected body mass reset")
        rows = np.asarray(indices)
        assert np.asarray(masses).shape == (3, 3)
        self.calls.append(rows.copy())
        self.mass[rows] = np.asarray(masses)[rows]


class Robot:
    def __init__(self):
        root = np.zeros((3, 13), dtype=np.float32)
        root[:, 3] = 1
        body = np.zeros((3, 3, 13), dtype=np.float32)
        body[:, :, 3] = 1
        self.data = SimpleNamespace(
            root_link_state_w=Tensor(root),
            body_link_state_w=Tensor(body),
            joint_pos=Tensor(np.zeros((3, 2), dtype=np.float32)),
            joint_vel=Tensor(np.zeros((3, 2), dtype=np.float32)),
            joint_stiffness=Tensor(np.tile([10, 20], (3, 1)).astype(np.float32)),
            joint_damping=Tensor(np.tile([1, 2], (3, 1)).astype(np.float32)),
        )
        self.root_physx_view = PhysX()
        self.actuators = {
            "all": SimpleNamespace(
                stiffness=self.data.joint_stiffness.clone(),
                damping=self.data.joint_damping.clone(),
                joint_indices=slice(None),
            )
        }
        self.calls = []

    def write_root_pose_to_sim(self, pose, env_ids):
        self.calls.append(("root", np.asarray(env_ids).copy()))
        self.data.root_link_state_w[env_ids, :7] = pose

    def write_root_link_velocity_to_sim(self, velocity, env_ids):
        self.data.root_link_state_w[env_ids, 7:] = velocity

    def write_joint_state_to_sim(self, positions, velocities, env_ids):
        self.data.joint_pos[env_ids] = positions
        self.data.joint_vel[env_ids] = velocities

    def write_joint_stiffness_to_sim(self, values, env_ids):
        self.calls.append(("kp", np.asarray(env_ids).copy()))
        self.data.joint_stiffness[env_ids] = values

    def write_joint_damping_to_sim(self, values, env_ids):
        self.calls.append(("kd", np.asarray(env_ids).copy()))
        self.data.joint_damping[env_ids] = values

    def reset(self, env_ids):
        self.calls.append(("reset", np.asarray(env_ids).copy()))

    def update(self, dt):
        pass

    def set_joint_position_target(self, targets):
        self.targets = np.asarray(targets).copy()

    def write_data_to_sim(self):
        pass


class ContactSensor:
    def __init__(self):
        self.body_names = ["foot", "base", "leg"]
        self.is_initialized = True
        self.forces = np.arange(27, dtype=np.float32).reshape(3, 3, 3) + 1
        self.data = SimpleNamespace(net_forces_w=Tensor(self.forces.copy()))
        self.resets = []

    def reset(self, env_ids=None):
        self.resets.append(None if env_ids is None else np.asarray(env_ids).copy())
        self.data.net_forces_w[slice(None) if env_ids is None else env_ids] = 0

    def update(self, dt, force_recompute):
        assert force_recompute
        self.data.net_forces_w[:] = self.forces


def make_worker(kind):
    worker = GymWorker(protocol) if kind == "isaacgym" else SimWorker(protocol)
    worker.num_envs, worker.num_dof, worker.num_bodies = 3, 2, 3
    worker.sim_dt = 0.01
    worker.device = "cpu"
    worker.torch = Torch()
    worker.slots = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.slot_shapes(3, 2, 3).items()
    }
    worker.contract_joint_names = ["hip", "knee"]
    worker.contract_body_names = ["base", "leg", "foot"]
    worker.native_joint_for_contract = protocol.name_permutation(
        ["knee", "hip"], worker.contract_joint_names, "joint"
    )
    worker.native_body_for_contract = protocol.name_permutation(
        ["leg", "foot", "base"], worker.contract_body_names, "body"
    )
    if kind == "isaacgym":
        worker.gym = Gym()
        worker.gymtorch = SimpleNamespace(unwrap_tensor=lambda tensor: tensor)
        worker.env_handles = list(range(3))
        worker.actor_handles = [100, 101, 102]
        worker._cache_reset_properties()
        root = np.zeros((3, 13), dtype=np.float32)
        root[:, 6] = 1
        body = np.zeros((3, 3, 13), dtype=np.float32)
        body[:, :, 6] = 1
        worker._root_state = Tensor(root)
        worker._dof_state = Tensor(np.zeros((3, 2, 2), dtype=np.float32))
        worker._body_state = Tensor(body)
        worker._contact_force = Tensor(np.arange(27, dtype=np.float32).reshape(3, 3, 3) + 1)
        worker._contact_valid = np.ones(3, dtype=bool)
    else:
        worker.robot = Robot()
        worker.sim = SimpleNamespace(step=lambda render: None)
        worker.env_origins = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float32)
        worker.robot.data.root_link_state_w[:, :3] = worker.env_origins
        worker.robot.data.body_link_state_w[:, :, :3] = worker.env_origins[:, None, :]
        worker.contact_sensor = ContactSensor()
        worker._bind_contact_reporter()
        worker.contact_sensor.update(0.01, force_recompute=True)
        worker._contact_valid[:] = True
        worker._cache_reset_properties()
    worker.refresh_state_slots()
    return worker


def fill_reset(worker, rows=(2, 0), terms=protocol.RESET_TERMS):
    count = len(rows)
    worker.slots["reset_env_ids"][:count] = rows
    positions = worker.slots["reset_qpos"][:count]
    positions[:] = 0
    positions[:, 3] = 1
    positions[:, :3] = [1, 2, 3]
    positions[:, 7:] = [0.4, 0.8]
    worker.slots["reset_qvel"][:count] = 0
    worker.slots["reset_qvel"][:count, 6:] = [0.6, 0.9]
    for term, values in (("body_mass", [11, 12, 13]), ("kp", [31, 32]), ("kd", [3, 4])):
        worker.slots["reset_" + term][:count] = (
            np.asarray(values)[None, :] + np.arange(count)[:, None]
        )
    return {"count": count, "randomization_terms": list(terms)}


def physical_properties(worker, kind):
    if kind == "isaacgym":
        return {
            "mass": worker.gym.mass.copy(),
            "kp": worker.gym.properties["stiffness"].copy(),
            "kd": worker.gym.properties["damping"].copy(),
            "inertia": worker.gym.inertia.copy(),
        }
    return {
        "mass": worker.robot.root_physx_view.mass.copy(),
        "kp": np.asarray(worker.robot.data.joint_stiffness).copy(),
        "kd": np.asarray(worker.robot.data.joint_damping).copy(),
        "inertia": worker.robot.root_physx_view.inertia.copy(),
    }


def nominal_metadata(worker):
    return copy.deepcopy(worker._reset_metadata)
