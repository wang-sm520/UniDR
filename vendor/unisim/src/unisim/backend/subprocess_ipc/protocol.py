"""Canonical pipe and shared-memory protocol for subprocess backends.

The module is loaded both by the host interpreter and by external workers via
an explicit file path.  Keep it compatible with Python 3.8 and import only the
standard library plus NumPy.
"""

from __future__ import annotations

import pickle
import struct
import traceback
from typing import Any, BinaryIO, Dict, Tuple

import numpy as np

CMD_INIT = "INIT"
CMD_ATTACH = "ATTACH_SLOTS"
CMD_STEP = "STEP"
CMD_SET_STATE = "SET_STATE"
CMD_REFRESH = "REFRESH"
CMD_GET_META = "GET_META"
CMD_INIT_RENDERER = "INIT_RENDERER"
CMD_RENDER_FRAME = "RENDER_FRAME"
CMD_CAPTURE_FRAME = "CAPTURE_FRAME"
CMD_SHUTDOWN = "SHUTDOWN"

CMD_READY = "READY"
CMD_META = "META"
CMD_ERROR = "ERROR"

PROTOCOL_VERSION = 2
RESET_TERMS = ("body_mass", "kp", "kd")
CONTACT_REPORTER_ISAACGYM = "isaacgym_net_contact_force"
CONTACT_REPORTER_ISAACSIM = "isaaclab_contact_sensor"

_PICKLE_PROTOCOL = 4
_HEADER = struct.Struct("<Q")
HEADER_SIZE = _HEADER.size


def pack_message(cmd: str, payload: Any = None) -> bytes:
    return pickle.dumps({"cmd": cmd, "payload": payload}, protocol=_PICKLE_PROTOCOL)


def unpack_header(data: bytes) -> int:
    (size,) = _HEADER.unpack(data)
    return int(size)


def decode_message(body: bytes) -> Dict[str, Any]:
    message = pickle.loads(body)
    if not isinstance(message, dict) or "cmd" not in message:
        raise ValueError(f"malformed worker message: {message!r}")
    return message


class WorkerDisconnectedError(EOFError):
    """Raised when a worker pipe closes before a complete message arrives."""


def send_message(stream: BinaryIO, cmd: str, payload: Any = None) -> None:
    body = pack_message(cmd, payload)
    stream.write(_HEADER.pack(len(body)))
    stream.write(body)
    stream.flush()


def _read_exactly(stream: BinaryIO, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise WorkerDisconnectedError(
                f"pipe closed while reading {size} bytes (got {size - remaining})"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(stream: BinaryIO) -> Dict[str, Any]:
    size = unpack_header(_read_exactly(stream, _HEADER.size))
    return decode_message(_read_exactly(stream, size))


_SLOT_DTYPES: Dict[str, str] = {
    "ctrl": "float32",
    "root_state": "float32",
    "dof_state": "float32",
    "body_state": "float32",
    "contact_force": "float32",
    "reset_env_ids": "int32",
    "reset_qpos": "float32",
    "reset_qvel": "float32",
    "reset_body_mass": "float32",
    "reset_kp": "float32",
    "reset_kd": "float32",
}

SLOT_NAMES = tuple(_SLOT_DTYPES)


def slot_shapes(num_envs: int, num_dof: int, num_bodies: int) -> Dict[str, Tuple[int, ...]]:
    if num_envs <= 0 or num_dof < 0 or num_bodies <= 0:
        raise ValueError(
            "slot shapes require num_envs>0, num_dof>=0, num_bodies>0; "
            f"got {num_envs}, {num_dof}, {num_bodies}"
        )
    return {
        "ctrl": (num_envs, num_dof),
        "root_state": (num_envs, 13),
        "dof_state": (num_envs, num_dof, 2),
        "body_state": (num_envs, num_bodies, 13),
        "contact_force": (num_envs, num_bodies, 3),
        "reset_env_ids": (num_envs,),
        "reset_qpos": (num_envs, 7 + num_dof),
        "reset_qvel": (num_envs, 6 + num_dof),
        "reset_body_mass": (num_envs, num_bodies),
        "reset_kp": (num_envs, num_dof),
        "reset_kd": (num_envs, num_dof),
    }


def validate_version(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("subprocess handshake must be a dictionary")
    version = payload.get("protocol_version")
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise ValueError(
            "incompatible subprocess protocol: expected version %d, got %r; "
            "update the host and worker together" % (PROTOCOL_VERSION, version)
        )


def validate_terms(terms: Any) -> Tuple[str, ...]:
    if not isinstance(terms, (list, tuple)) or not all(isinstance(term, str) for term in terms):
        raise ValueError("randomization terms must be a list of strings")
    if len(set(terms)) != len(terms) or set(terms).difference(RESET_TERMS):
        raise ValueError("unsupported or duplicate reset randomization terms: %r" % (terms,))
    return tuple(terms)


def validate_init(payload: Dict[str, Any]) -> None:
    validate_version(payload)
    validate_terms(payload.get("required_reset_terms"))


def validate_worker_metadata(meta: Dict[str, Any]) -> Tuple[str, ...]:
    validate_version(meta)
    terms = validate_terms(meta.get("supported_reset_terms"))
    if set(terms) != set(RESET_TERMS):
        raise ValueError("worker does not support required reset terms: %r" % (RESET_TERMS,))
    if "contact_reporter" not in meta or meta["contact_reporter"] not in (
        None,
        CONTACT_REPORTER_ISAACGYM,
        CONTACT_REPORTER_ISAACSIM,
    ):
        raise ValueError("worker must declare a supported contact_reporter or None")
    return terms


def name_permutation(native: Any, contract: Any, kind: str) -> np.ndarray:
    """Resolve a complete contract-to-native mapping once during INIT."""
    if (
        len(native) != len(contract)
        or len(set(native)) != len(native)
        or len(set(contract)) != len(contract)
        or set(native) != set(contract)
    ):
        raise ValueError(
            "%s name mapping mismatch: native=%r, contract=%r" % (kind, native, contract)
        )
    native_ids = {name: index for index, name in enumerate(native)}
    return np.asarray([native_ids[name] for name in contract], dtype=np.int64)


def float_array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError("%s must contain real numeric values" % name)
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("%s must have shape %r and finite float32 values" % (name, shape))
    return array


def reset_values(value: Any, shape: Tuple[int, ...], term: str) -> np.ndarray:
    array = float_array(value, shape, term)
    invalid = (array <= 0).any() if term == "body_mass" else (array < 0).any()
    if invalid:
        bound = "positive" if term == "body_mass" else "nonnegative"
        raise ValueError("%s must be %s" % (term, bound))
    return array


def validate_reset_state(
    env_ids: Any,
    qpos: Any,
    qvel: Any,
    num_envs: int,
    num_dof: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(env_ids)
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError("reset environment ids must be a one-dimensional integer array")
    if (rows < 0).any() or (rows >= num_envs).any() or np.unique(rows).size != rows.size:
        raise ValueError("reset environment ids must be unique and in range")
    positions = float_array(qpos, (rows.size, 7 + num_dof), "reset qpos")
    velocities = float_array(qvel, (rows.size, 6 + num_dof), "reset qvel")
    if (np.linalg.norm(positions[:, 3:7].astype(np.float64), axis=1) < 1e-8).any():
        raise ValueError("reset qpos must contain nonzero root quaternions")
    return rows.astype(np.int32), positions, velocities


def read_reset(
    payload: Dict[str, Any],
    slots: Dict[str, np.ndarray],
    num_envs: int,
    num_dof: int,
    num_bodies: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Validate and snapshot the complete transaction before any SDK writes."""
    count = payload.get("count")
    if type(count) is not int or not 0 <= count <= num_envs:
        raise ValueError("reset count must be an integer in [0, %d]" % num_envs)
    terms = validate_terms(payload.get("randomization_terms"))
    rows, qpos, qvel = validate_reset_state(
        slots["reset_env_ids"][:count],
        slots["reset_qpos"][:count],
        slots["reset_qvel"][:count],
        num_envs,
        num_dof,
    )
    randomization = {
        term: reset_values(
            slots["reset_" + term][:count],
            (count, num_bodies if term == "body_mass" else num_dof),
            term,
        )
        for term in terms
    }
    return rows, qpos, qvel, randomization


def validate_slot_specs(
    payload: Dict[str, Any],
    num_envs: int,
    num_dof: int,
    num_bodies: int,
) -> None:
    validate_version(payload)
    specs = payload["slots"]
    shapes = slot_shapes(num_envs, num_dof, num_bodies)
    if set(specs) != set(SLOT_NAMES):
        raise ValueError("incompatible shared-memory slot names")
    for name, shape in shapes.items():
        if tuple(specs[name]["shape"]) != shape or np.dtype(specs[name]["dtype"]) != slot_dtype(
            name
        ):
            raise ValueError("incompatible shared-memory slot layout for %s" % name)


def slot_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(_SLOT_DTYPES[name])
    except KeyError as exc:
        raise ValueError(f"unknown shm slot {name!r}; known: {sorted(_SLOT_DTYPES)}") from exc


def slot_nbytes(name: str, shape: Tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) * int(slot_dtype(name).itemsize)


def serialize_exception(exc: BaseException) -> Dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def format_worker_error(payload: Dict[str, str], backend: str = "subprocess") -> str:
    return (
        f"{backend} worker raised {payload.get('type', 'Error')}: "
        f"{payload.get('message', '')}\n"
        f"worker traceback:\n{payload.get('traceback', '<unavailable>')}"
    )


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat)[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat)[..., [1, 2, 3, 0]]


def quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (w * uv + uuv)


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).copy()
    q[..., 1:4] = -q[..., 1:4]
    return quat_rotate(q, vec)


__all__ = [
    "CMD_ATTACH",
    "CMD_CAPTURE_FRAME",
    "CMD_ERROR",
    "CMD_GET_META",
    "CMD_INIT",
    "CMD_INIT_RENDERER",
    "CMD_META",
    "CMD_READY",
    "CMD_REFRESH",
    "CMD_RENDER_FRAME",
    "CMD_SET_STATE",
    "CMD_SHUTDOWN",
    "CMD_STEP",
    "CONTACT_REPORTER_ISAACGYM",
    "CONTACT_REPORTER_ISAACSIM",
    "HEADER_SIZE",
    "PROTOCOL_VERSION",
    "RESET_TERMS",
    "SLOT_NAMES",
    "WorkerDisconnectedError",
    "decode_message",
    "format_worker_error",
    "float_array",
    "name_permutation",
    "pack_message",
    "quat_rotate",
    "quat_rotate_inverse",
    "recv_message",
    "read_reset",
    "reset_values",
    "send_message",
    "serialize_exception",
    "slot_dtype",
    "slot_nbytes",
    "slot_shapes",
    "unpack_header",
    "validate_init",
    "validate_reset_state",
    "validate_slot_specs",
    "validate_terms",
    "validate_version",
    "validate_worker_metadata",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
