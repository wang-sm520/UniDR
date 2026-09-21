"""Private, Python-3.8-compatible MJCF FK for state writes before Gym simulates.

Compile once during INIT. Reset evaluation only uses cached numeric arrays;
the supported subset is one free root and fixed/hinge/slide child bodies.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np


def _rotate(q, v):
    uv = np.cross(q[..., 1:], v)
    return v + 2 * (q[..., :1] * uv + np.cross(q[..., 1:], uv))


def _multiply(a, b):
    return np.concatenate(
        (
            a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True),
            a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:]),
        ),
        axis=-1,
    )


def _vector(attributes, key, default, size):
    value = np.fromstring(attributes.get(key, default), sep=" ", dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError("invalid MJCF %s for IsaacGym reset FK" % key)
    return value


def _load_xml(path, ancestors=()):
    path = os.path.realpath(path)
    if path in ancestors:
        raise ValueError("cyclic MJCF include in IsaacGym reset FK")
    root = ET.parse(path).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "include":
                included = _load_xml(
                    os.path.join(os.path.dirname(path), child.attrib["file"]), ancestors + (path,)
                )
                index = list(parent).index(child)
                parent.remove(child)
                for item in included:
                    parent.insert(index, item)
                    index += 1
    return root


class ResetKinematics:
    def __init__(self, path, body_names, dof_names):
        root = _load_xml(path)
        compilers = root.findall("compiler")
        if any(c.get("coordinate", "local") != "local" for c in compilers):
            raise ValueError("IsaacGym reset FK requires local MJCF coordinates")
        angles = {c.get("angle") for c in compilers if c.get("angle") is not None}
        if len(angles) > 1 or angles - {"radian", "degree"}:
            raise ValueError("ambiguous MJCF angle units for IsaacGym reset FK")
        angle_scale = 1.0 if angles == {"radian"} else np.pi / 180
        defaults = {"": {}}

        def collect_defaults(node, inherited):
            attrs = dict(inherited)
            joint = node.find("joint")
            if joint is not None:
                attrs.update(joint.attrib)
            name = node.get("class", "")
            if name and name in defaults:
                raise ValueError("duplicate MJCF default class %r" % name)
            defaults[name] = attrs
            for child in node.findall("default"):
                collect_defaults(child, attrs)

        for node in root.findall("default"):
            collect_defaults(node, defaults[""])
        worldbodies = root.findall("worldbody")
        if any(
            node.tag in {"frame", "replicate", "composite", "flexcomp"}
            for world in worldbodies
            for node in world.iter()
        ):
            raise ValueError("unsupported MJCF body construction for IsaacGym reset FK")
        roots = [body for world in worldbodies for body in world.findall("body")]
        if len(roots) != 1:
            raise ValueError("IsaacGym reset FK requires exactly one free-root body")
        dof_indices = {name: index for index, name in enumerate(dof_names)}
        self._nodes = []
        names = []
        joints = []

        def visit(body, parent, inherited_class):
            name = body.get("name")
            if not name or name in names:
                raise ValueError("IsaacGym reset FK requires unique named bodies")
            if any(key in body.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
                raise ValueError("IsaacGym reset FK requires body pos/quat orientation")
            pos = _vector(body.attrib, "pos", "0 0 0", 3)
            quat = _vector(body.attrib, "quat", "1 0 0 0", 4)
            norm = np.linalg.norm(quat)
            if norm < 1e-12:
                raise ValueError("zero body quaternion for IsaacGym reset FK")
            quat /= norm
            childclass = body.get("childclass", inherited_class)
            body_joints = [child for child in body if child.tag in ("joint", "freejoint")]
            kind, dof, axis, anchor, ref = "fixed", -1, np.zeros(3), np.zeros(3), 0.0
            if len(body_joints) > 1:
                raise ValueError("IsaacGym reset FK supports one joint per body")
            if body_joints:
                joint = body_joints[0]
                cls = joint.get("class", childclass)
                if cls not in defaults:
                    raise ValueError("unknown MJCF joint default class %r" % cls)
                attrs = dict(defaults[cls])
                attrs.update(joint.attrib)
                kind = "free" if joint.tag == "freejoint" else attrs.get("type", "hinge")
                if parent < 0:
                    if kind != "free":
                        raise ValueError("IsaacGym reset FK requires a free root")
                else:
                    if kind not in ("hinge", "slide"):
                        raise ValueError("unsupported IsaacGym reset FK joint %r" % kind)
                    joint_name = attrs.get("name")
                    if joint_name not in dof_indices or joint_name in joints:
                        raise ValueError("IsaacGym reset FK joint names differ from native asset")
                    joints.append(joint_name)
                    dof = dof_indices[joint_name]
                    axis = _vector(attrs, "axis", "0 0 1", 3)
                    norm = np.linalg.norm(axis)
                    if norm < 1e-12:
                        raise ValueError("zero joint axis for IsaacGym reset FK")
                    axis /= norm
                    anchor = _vector(attrs, "pos", "0 0 0", 3)
                    ref = float(attrs.get("ref", "0")) * (angle_scale if kind == "hinge" else 1)
                    if not np.isfinite(ref):
                        raise ValueError("invalid joint ref for IsaacGym reset FK")
            elif parent < 0:
                raise ValueError("IsaacGym reset FK requires a free root")
            index = len(names)
            names.append(name)
            self._nodes.append((parent, pos, quat, kind, dof, axis, anchor, ref))
            for child in body.findall("body"):
                visit(child, index, childclass)

        visit(roots[0], -1, "")
        if set(names) != set(body_names) or len(names) != len(body_names):
            raise ValueError("IsaacGym reset FK body names differ from native asset")
        if set(joints) != set(dof_names) or len(joints) != len(dof_names):
            raise ValueError("IsaacGym reset FK joint names differ from native asset")
        self._native_order = np.asarray([names.index(name) for name in body_names])

    def evaluate(self, root, dof):
        """Return native-order link states, with wxyz quats and world velocities."""
        states = np.zeros((len(root), len(self._nodes), 13), dtype=np.float64)
        states[:, 0] = root
        for index, (parent, pos, quat, kind, joint, axis, anchor, ref) in enumerate(self._nodes):
            if parent < 0:
                continue
            parent_state = states[:, parent]
            state = states[:, index]
            offset = _rotate(parent_state[:, 3:7], pos)
            rotation = _multiply(parent_state[:, 3:7], quat)
            state[:, :3] = parent_state[:, :3] + offset
            state[:, 7:10] = parent_state[:, 7:10] + np.cross(parent_state[:, 10:13], offset)
            state[:, 10:13] = parent_state[:, 10:13]
            if kind in ("hinge", "slide"):
                value, speed = dof[:, joint, 0] - ref, dof[:, joint, 1:2]
                world_axis = _rotate(rotation, axis)
                if kind == "hinge":
                    half = value[:, None] / 2
                    delta = np.concatenate((np.cos(half), np.sin(half) * axis), axis=-1)
                    moved_rotation = _multiply(rotation, delta)
                    moved_anchor = _rotate(moved_rotation, anchor)
                    shift = _rotate(rotation, anchor) - moved_anchor
                    state[:, 7:10] -= np.cross(world_axis * speed, moved_anchor)
                    state[:, 10:13] += world_axis * speed
                    rotation = moved_rotation
                else:
                    shift = world_axis * value[:, None]
                    state[:, 7:10] += world_axis * speed
                state[:, :3] += shift
                state[:, 7:10] += np.cross(parent_state[:, 10:13], shift)
            state[:, 3:7] = rotation
        return states[:, self._native_order]
