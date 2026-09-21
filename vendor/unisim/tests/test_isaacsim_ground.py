"""USD composition regression; also runnable in the vendor Python without Kit."""

from pathlib import Path

import pytest

pytest.importorskip("pxr.Usd")
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from unisim.backend.isaacsim.worker import _share_static_ground  # noqa: E402


def _scene(tmp_path: Path, *, tilt=False, dynamic=False, num_envs=2):
    source = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    root = UsdGeom.Xform.Define(source, "/scene").GetPrim()
    source.SetDefaultPrim(root)
    robot = UsdGeom.Xform.Define(source, "/scene/robot").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(robot)
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Sphere.Define(source, "/scene/robot/shape").GetPrim())
    world = UsdGeom.Xform.Define(source, "/scene/world").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(world)
    floor = UsdGeom.Xform.Define(source, "/scene/world/floor")
    floor.AddTranslateOp().Set((0, 0, 0.15))
    if tilt:
        floor.AddRotateXOp().Set(10)
    UsdPhysics.RigidBodyAPI.Apply(floor.GetPrim()).CreateKinematicEnabledAttr(not dynamic)
    plane = UsdGeom.Plane.Define(source, "/scene/world/floor/plane")
    plane.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    material = UsdShade.Material.Define(source, "/scene/Looks/ground")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(0.7)
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(material, materialPurpose="physics")
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateInMemory()
    paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
    for i, path in enumerate(paths):
        UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set((i * 2.0, 0, 0))
        stage.DefinePrim(path + "/Robot").GetReferences().AddReference(
            source.GetRootLayer().identifier
        )
    return stage, source, paths


@pytest.mark.parametrize("num_envs", [1, 2])
def test_clones_share_one_ground_and_preserve_physics(tmp_path, num_envs):
    stage, source, paths = _scene(tmp_path, num_envs=num_envs)
    ground = _share_static_ground(stage, source.GetRootLayer().identifier, paths)
    planes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Plane)]
    assert len(planes) == 1
    assert ground == ["/World/sharedGround"]
    plane = planes[0]
    assert UsdPhysics.CollisionAPI(plane).GetCollisionEnabledAttr().Get()
    assert (
        UsdGeom.Xformable(plane).ComputeLocalToWorldTransform(Usd.TimeCode.Default())[3][2] == 0.15
    )
    material, _ = UsdShade.MaterialBindingAPI(plane).ComputeBoundMaterial("physics")
    assert UsdPhysics.MaterialAPI(material).GetStaticFrictionAttr().Get() == pytest.approx(0.7)
    assert sum(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()) == num_envs
    assert sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()) == num_envs
    assert len([p for p in stage.Traverse() if p.IsA(UsdGeom.Sphere)]) == num_envs
    assert source.GetPrimAtPath("/scene/world/floor/plane").IsActive()


@pytest.mark.parametrize("kwargs", [{"tilt": True}, {"dynamic": True}])
def test_nonshareable_plane_fails_before_mutating_stage(tmp_path, kwargs):
    stage, source, paths = _scene(tmp_path, **kwargs)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="static horizontal"):
        _share_static_ground(stage, source.GetRootLayer().identifier, paths)
    assert stage.GetRootLayer().ExportToString() == before


def test_inherited_material_and_time_varying_plane(tmp_path):
    stage, source, paths = _scene(tmp_path)
    plane = source.GetPrimAtPath("/scene/world/floor/plane")
    UsdShade.MaterialBindingAPI(plane).UnbindAllBindings()
    material = UsdShade.Material.Get(source, "/scene/Looks/ground")
    UsdShade.MaterialBindingAPI.Apply(plane.GetParent()).Bind(material, materialPurpose="physics")
    source.GetRootLayer().Save()
    _share_static_ground(stage, source.GetRootLayer().identifier, paths)
    result = stage.GetPrimAtPath("/World/sharedGround/world/floor/plane")
    bound, _ = UsdShade.MaterialBindingAPI(result).ComputeBoundMaterial("physics")
    assert UsdPhysics.MaterialAPI(bound).GetStaticFrictionAttr().Get() == pytest.approx(0.7)
    floor = source.GetPrimAtPath("/scene/world/floor")
    floor.GetAttribute("xformOp:translate").Set((0, 0, 0.2), time=1)
    floor.GetAttribute("xformOp:translate").Set((0, 0, 0.3), time=2)
    source.GetRootLayer().Save()
    with pytest.raises(ValueError, match="static horizontal"):
        _share_static_ground(stage, source.GetRootLayer().identifier, paths)


def test_scene_without_planes_is_unchanged(tmp_path):
    stage, source, paths = _scene(tmp_path)
    source.GetPrimAtPath("/scene/world").SetActive(False)
    source.GetRootLayer().Save()
    before = stage.GetRootLayer().ExportToString()
    assert _share_static_ground(stage, source.GetRootLayer().identifier, paths) == []
    assert stage.GetRootLayer().ExportToString() == before
