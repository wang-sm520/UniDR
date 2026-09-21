"""Offline playback keeps visual geometry outside the stripped physics model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
mjbatch = pytest.importorskip("mjbatch")

from unisim import MuJoCoBackend  # noqa: E402
from unisim.backend.mujoco.playback import resolve_render_play_model_files  # noqa: E402
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402


def _visual_scene(path: Path, *, radius: float, color: str) -> Path:
    path.write_text(
        f"""<mujoco>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" width="8" height="48"
             rgb1="0.5 0.5 0.5" rgb2="0.2 0.2 0.2"/>
    <material name="paint" rgba="{color} 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom name="collision" type="sphere" size="0.05" mass="1" group="3"/>
      <geom name="robot_visual" type="sphere" size="{radius}" material="paint"
            contype="0" conaffinity="0" density="0" group="2"/>
    </body>
  </worldbody>
</mujoco>"""
    )
    return path


def _assert_visual_model(path: str, *, radius: float, color: list[float]) -> None:
    model = mujoco.MjModel.from_binary_path(path)
    visual_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_visual")
    assert visual_id >= 0, "offline playback lost the robot's visual geometry"
    assert model.geom_group[visual_id] == 2
    assert model.ntex == 1
    assert model.geom_size[visual_id, 0] == pytest.approx(radius)
    np.testing.assert_allclose(model.mat_rgba[model.geom_matid[visual_id]], [*color, 1])


@pytest.mark.parametrize("visual_override", [False, True])
def test_nominal_playback_resolves_visual_model_without_changing_physics(
    tmp_path: Path, visual_override: bool
) -> None:
    source = _visual_scene(tmp_path / "scene.xml", radius=0.1, color="1 0 0")
    visual = _visual_scene(tmp_path / "visual.xml", radius=0.2, color="0 1 0")
    backend = MuJoCoBackend(
        SceneCfg(
            model_file=str(source), visual_model_file=str(visual) if visual_override else None
        ),
        num_envs=2,
        sim_dt=0.002,
    )
    physics = backend.model
    before = backend.get_physics_state().copy()
    assert physics.ngeom == 2
    assert physics.ntex == 0
    output = tmp_path / "playback"
    output.mkdir()

    files = resolve_render_play_model_files(backend, num_envs=2, tmp_dir=output)

    assert isinstance(files, list)
    assert len(files) == 2
    for path in files:
        _assert_visual_model(
            path,
            radius=0.2 if visual_override else 0.1,
            color=[0, 1, 0] if visual_override else [1, 0, 0],
        )
    assert backend.model is physics
    assert physics.ngeom == 2
    assert physics.ntex == 0
    np.testing.assert_array_equal(backend.get_physics_state(), before)


@pytest.mark.skipif(
    not hasattr(mjbatch.Batch, "from_variant_pack"), reason="mjbatch VariantPack API is required"
)
def test_fixed_variant_playback_keeps_each_world_visual_source(tmp_path: Path) -> None:
    sources = [
        _visual_scene(tmp_path / "red.xml", radius=0.1, color="1 0 0"),
        _visual_scene(tmp_path / "green.xml", radius=0.2, color="0 1 0"),
    ]
    assignment = np.array([1, 0], dtype=np.int32)
    plan = FixedVariantPlan(assignment, tuple(ModelSourceDescriptor(str(path)) for path in sources))
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(sources[0]), fixed_variant_plan=plan),
        num_envs=2,
        sim_dt=0.002,
    )
    output = tmp_path / "playback"
    output.mkdir()

    files = resolve_render_play_model_files(backend, num_envs=2, tmp_dir=output)

    assert isinstance(files, list)
    _assert_visual_model(files[0], radius=0.2, color=[0, 1, 0])
    _assert_visual_model(files[1], radius=0.1, color=[1, 0, 0])
