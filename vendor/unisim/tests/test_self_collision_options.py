"""Construction options must remove internal contacts without removing ground contacts."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from unisim import create_backend
from unisim.backend.genesis.materialization import build_genesis_scene
from unisim.backend.motrix.scene import _disable_motrix_robot_self_collision
from unisim.scene import SceneCfg

MODEL = """<mujoco><option timestep="0.001" gravity="0 0 0"/><worldbody>
  <geom name="floor" type="plane" size="2 2 .05"/>
  <body name="base" pos="0 0 .5"><freejoint/>
    <geom name="rootgeom" type="sphere" size=".02" mass="1"/>
    <body name="left" pos="-.3 0 -.42">
      <joint name="left_slide" type="slide" axis="1 0 0"/>
      <geom name="leftgeom" type="sphere" size=".1" mass="1"/></body>
    <body name="right" pos=".3 0 -.42">
      <joint name="right_slide" type="slide" axis="1 0 0"/>
      <geom name="rightgeom" type="sphere" size=".1" mass="1"/></body>
  </body></worldbody><actuator><motor joint="left_slide"/>
    <motor joint="right_slide"/></actuator></mujoco>"""


@pytest.mark.parametrize("option", [None, False, True])
def test_genesis_option_forwarding(option):
    gs = SimpleNamespace(
        Scene=lambda **kwargs: kwargs,
        options=SimpleNamespace(SimOptions=lambda **kw: kw, RigidOptions=lambda **kw: kw),
    )
    scene = build_genesis_scene(
        SimpleNamespace(genesis=gs),
        sim_dt=0.005,
        gravity=np.zeros(3),
        integrator=None,
        constraint_solver=None,
        friction_cone=None,
        solver_iterations=None,
        enable_self_collision=option,
    )
    if option is None:
        assert "enable_self_collision" not in scene["rigid_options"]
    else:
        assert scene["rigid_options"]["enable_self_collision"] is option


@pytest.mark.parametrize(
    "option", ["motrix_disable_self_collision", "genesis_enable_self_collision"]
)
@pytest.mark.parametrize("bad", [0, 1, "false"])
def test_factory_rejects_invalid_collision_options(option, bad):
    with pytest.raises(TypeError, match=option):
        create_backend("mujoco", SceneCfg(model_file="unused.xml"), **{option: bad})


def test_motrix_requires_named_articulation_root():
    root = SimpleNamespace(name="base", disable_self_collision=False)
    other = SimpleNamespace(name="object", disable_self_collision=False)
    world = SimpleNamespace(
        hierarchy=SimpleNamespace(
            bodies=[
                SimpleNamespace(link=root),
                SimpleNamespace(link=other),
            ]
        )
    )
    with pytest.raises(ValueError, match="articulation root"):
        _disable_motrix_robot_self_collision(world, "missing")
    _disable_motrix_robot_self_collision(world, "base")
    assert root.disable_self_collision is True
    assert other.disable_self_collision is False


@pytest.mark.parametrize("disabled", [False, True])
def test_motrix_native_self_contacts_and_ground(tmp_path, disabled):
    pytest.importorskip("motrixsim")
    path = tmp_path / "contact.xml"
    # Overlapping siblings and floor exercise three distinct contact pairs.
    path.write_text(MODEL.replace("-.3 0", "-.075 0").replace(".3 0", ".075 0"))
    backend = create_backend(
        "motrix",
        SceneCfg(model_file=str(path)),
        num_envs=2,
        sim_dt=0.001,
        base_name="base",
        motrix_disable_self_collision=disabled,
    )
    try:
        backend.step(np.zeros((2, 2), dtype=np.float32))
        geoms = {g.name: int(g.index) for g in backend._model.geoms}
        pairs = np.array(
            [
                [geoms["leftgeom"], geoms["rightgeom"]],
                [geoms["leftgeom"], geoms["floor"]],
                [geoms["rightgeom"], geoms["floor"]],
            ],
            dtype=np.uint32,
        )
        actual = backend._model.get_contact_query(backend._data).is_colliding(pairs)
        np.testing.assert_array_equal(actual, [[not disabled, True, True]] * 2)
    finally:
        del backend  # Headless Motrix owns in-process SDK objects, with no close method.


def test_genesis_native_self_contacts_and_imported_floor(tmp_path):
    pytest.importorskip("genesis")
    path = tmp_path / "contact.xml"
    path.write_text(MODEL)
    # Isolate Genesis's process-global init/destroy lifecycle from other tests.
    script = """
import sys
from types import SimpleNamespace
import numpy as np
import genesis as gs
from unisim.backend.genesis.materialization import build_genesis_scene
gs.init(backend=gs.cpu, logging_level='warning')
try:
    for enabled in (True, False):
        scene = build_genesis_scene(
            SimpleNamespace(genesis=gs), sim_dt=.001, gravity=np.zeros(3),
            integrator=None, constraint_solver=None, friction_cone=None,
            solver_iterations=None, enable_self_collision=enabled,
        )
        entity = scene.add_entity(gs.morphs.MJCF(file=sys.argv[1]))
        scene.build(n_envs=1)
        # Move initially separate siblings into contact after neutral-pose filtering.
        qpos = entity.get_qpos().clone()
        qpos[:, -2:] = qpos.new_tensor([.225, -.225])
        entity.set_qpos(qpos)
        pairs = {tuple(sorted(map(int, pair))) for pair in entity.detect_collision()}
        ids = {g.link.name: int(g.idx) for g in entity.geoms}
        assert ((ids['left'], ids['right']) in pairs) == enabled, pairs
        assert (ids['world'], ids['left']) in pairs, pairs
        assert (ids['world'], ids['right']) in pairs, pairs
        scene.destroy()
finally:
    gs.destroy()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
