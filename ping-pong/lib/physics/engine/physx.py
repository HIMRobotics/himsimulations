"""
Author the canonical PhysX settings on the Isaac Sim 6.0.1 stage, and read them back.

Every function takes prim paths and :class:`lib.physics.engine.settings` values;
nothing here decides a value. Call :func:`configure_scene` and the body
functions on a stopped world, before ``world.reset()``. Each returns what it
authored, read back from the stage, so a run records the settings that were
really in force (for instance that the solver is PGS).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lib.physics.engine.settings import BallBodySettings, PaddleBodySettings, SceneSettings

if TYPE_CHECKING:
    from pxr import Usd

BALL_MATERIAL_PATH: str = "/World/TableTennis/PhysicsMaterials/ball"


def _applied(api: Any, what: str) -> Any:
    """An applied schema, or a named failure (``Apply`` returns an invalid schema on a bad prim)."""

    if not api:
        raise RuntimeError(f"could not apply {what}")
    return api


def configure_scene(world: Any, stage: Usd.Stage, settings: SceneSettings) -> dict[str, object]:
    """
    Set the PhysX scene: solver, CPU dynamics, CCD, stabilization, bounce threshold, gravity.

    :param world: The Isaac Sim ``World``.
    :param stage: The active stage.
    :param settings: Scene settings.
    :return: The values read back from the physics scene prim.
    :raises RuntimeError: If the solver or CCD did not take
    """

    from pxr import Gf, PhysxSchema, UsdPhysics

    context = world.get_physics_context()
    context.enable_gpu_dynamics(settings.gpu_dynamics)
    context.enable_ccd(settings.ccd)
    context.enable_stabilization(settings.stabilization)
    context.set_solver_type(settings.solver)
    context.set_gravity(-settings.gravity_mps2)
    scene_path = context.get_current_physics_scene_prim_path() if hasattr(context, "get_current_physics_scene_prim_path") else None
    scene_prim = stage.GetPrimAtPath(scene_path) if scene_path else None
    if scene_prim is None or not scene_prim.IsValid():
        scene_prim = next((prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)), None)
    if scene_prim is None:
        raise RuntimeError("no UsdPhysics.Scene prim on the stage after configuring the physics context")
    physx_scene = _applied(PhysxSchema.PhysxSceneAPI.Apply(scene_prim), "PhysxSceneAPI")
    physx_scene.CreateSolverTypeAttr().Set(settings.solver)
    physx_scene.CreateEnableCCDAttr().Set(settings.ccd)
    physx_scene.CreateEnableStabilizationAttr().Set(settings.stabilization)
    physx_scene.CreateEnableGPUDynamicsAttr().Set(settings.gpu_dynamics)
    physx_scene.CreateBounceThresholdAttr().Set(float(settings.bounce_threshold_mps))
    scene: Any = UsdPhysics.Scene(scene_prim)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(float(settings.gravity_mps2))
    readback = {
        "scene_prim": str(scene_prim.GetPath()),
        "solver": str(physx_scene.GetSolverTypeAttr().Get()),
        "ccd": bool(physx_scene.GetEnableCCDAttr().Get()),
        "stabilization": bool(physx_scene.GetEnableStabilizationAttr().Get()),
        "gpu_dynamics": bool(physx_scene.GetEnableGPUDynamicsAttr().Get()),
        "bounce_threshold_mps": float(physx_scene.GetBounceThresholdAttr().Get()),
        "gravity_mps2": float(scene.GetGravityMagnitudeAttr().Get()),
        "physics_dt_s": float(context.get_physics_dt()),
    }
    if readback["solver"] != settings.solver or readback["ccd"] != settings.ccd:
        raise RuntimeError(f"the physics scene did not take the canonical settings: {readback}")
    return readback


def _ball_material(stage: Usd.Stage, settings: BallBodySettings) -> str:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, BALL_MATERIAL_PATH)
    api = _applied(UsdPhysics.MaterialAPI.Apply(material.GetPrim()), "MaterialAPI")
    api.CreateRestitutionAttr().Set(float(settings.restitution))
    api.CreateStaticFrictionAttr().Set(float(settings.friction))
    api.CreateDynamicFrictionAttr().Set(float(settings.friction))
    physx = _applied(PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()), "PhysxMaterialAPI")
    physx.CreateRestitutionCombineModeAttr().Set(settings.combine_mode)
    physx.CreateFrictionCombineModeAttr().Set(settings.combine_mode)
    return BALL_MATERIAL_PATH


def author_ball_body(stage: Usd.Stage, ball_path: str, collider_path: str, settings: BallBodySettings, *, filtered: tuple[str, ...] = ()) -> dict[str, object]:
    """
    Make the ball prim a CCD rigid body with the canonical mass, inertia, limits and material.

    :param stage: The active stage, stopped.
    :param ball_path: The ball prim (its transform is the body's).
    :param collider_path: The sphere collider under it.
    :param settings: Ball settings.
    :param filtered: Collider prims PhysX must not collide the ball with (hybrid surfaces).
    :return: What was authored, read back.
    """

    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics, UsdShade

    prim = stage.GetPrimAtPath(ball_path)
    collider = stage.GetPrimAtPath(collider_path)
    if not prim.IsValid() or not collider.IsValid():
        raise RuntimeError(f"ball prim {ball_path} or collider {collider_path} is missing")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = _applied(UsdPhysics.MassAPI.Apply(prim), "MassAPI")
    mass.CreateMassAttr().Set(float(settings.mass_kg))
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(settings.inertia_kgm2),) * 3))
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    body = _applied(PhysxSchema.PhysxRigidBodyAPI.Apply(prim), "PhysxRigidBodyAPI")
    body.CreateEnableCCDAttr().Set(settings.ccd)
    body.CreateMaxAngularVelocityAttr().Set(float(settings.max_angular_velocity_deg_s))
    body.CreateLinearDampingAttr().Set(float(settings.linear_damping))
    body.CreateAngularDampingAttr().Set(float(settings.angular_damping))
    body.CreateSleepThresholdAttr().Set(float(settings.sleep_threshold))
    body.CreateStabilizationThresholdAttr().Set(float(settings.stabilization_threshold))
    body.CreateMaxDepenetrationVelocityAttr().Set(float(settings.max_depenetration_velocity_mps))
    body.CreateSolverPositionIterationCountAttr().Set(int(settings.position_iterations))
    body.CreateSolverVelocityIterationCountAttr().Set(int(settings.velocity_iterations))
    collision = _applied(PhysxSchema.PhysxCollisionAPI.Apply(collider), "PhysxCollisionAPI")
    collision.CreateContactOffsetAttr().Set(float(settings.contact_offset_m))
    collision.CreateRestOffsetAttr().Set(float(settings.rest_offset_m))
    material = _ball_material(stage, settings)
    UsdShade.MaterialBindingAPI.Apply(collider).Bind(UsdShade.Material(stage.GetPrimAtPath(material)), UsdShade.Tokens.weakerThanDescendants, "physics")
    if filtered:
        pairs = _applied(UsdPhysics.FilteredPairsAPI.Apply(prim), "FilteredPairsAPI")
        pairs.CreateFilteredPairsRel().SetTargets([Sdf.Path(path) for path in filtered])
    return {
        "ball_prim": ball_path,
        "mass_kg": float(mass.GetMassAttr().Get()),
        "inertia_kgm2": float(mass.GetDiagonalInertiaAttr().Get()[0]),
        "ccd": bool(body.GetEnableCCDAttr().Get()),
        "max_angular_velocity_deg_s": float(body.GetMaxAngularVelocityAttr().Get()),
        "angular_damping": float(body.GetAngularDampingAttr().Get()),
        "contact_offset_m": float(collision.GetContactOffsetAttr().Get()),
        "material": material,
        "filtered": list(filtered),
    }


def make_passive_marker(stage: Usd.Stage, ball_path: str, collider_path: str) -> dict[str, object]:
    """
    Turn the game ball's body into a passive marker for the ``rk4`` ball integrator: no gravity, no CCD and no
    collisions, so PhysX leaves it exactly where the adapter writes it (once per engine step) and nothing meets it.

    :return: What was authored.
    """

    from pxr import PhysxSchema, UsdPhysics

    prim = stage.GetPrimAtPath(ball_path)
    collider = stage.GetPrimAtPath(collider_path)
    if not prim.IsValid() or not collider.IsValid():
        raise RuntimeError(f"ball prim {ball_path} or collider {collider_path} is missing")
    body = _applied(PhysxSchema.PhysxRigidBodyAPI.Apply(prim), "PhysxRigidBodyAPI")
    body.CreateDisableGravityAttr().Set(True)
    body.CreateEnableCCDAttr().Set(False)
    collision: Any = UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr()
    collision.Set(False)
    return {
        "passive_marker": True,
        "gravity_disabled": bool(body.GetDisableGravityAttr().Get()),
        "ccd": bool(body.GetEnableCCDAttr().Get()),
        "collision_enabled": bool(collision.Get()),
    }


def strip_colliders(stage: Usd.Stage, root_path: str) -> list[str]:
    """Remove every collision API under a prim (the paddle must never collide with the ball in PhysX)."""

    from pxr import Usd, UsdPhysics

    removed: list[str] = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        for schema in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI):
            if prim.HasAPI(schema):
                prim.RemoveAPI(schema)
                removed.append(str(prim.GetPath()))
    return removed


def author_paddle_body(stage: Usd.Stage, paddle_path: str, settings: PaddleBodySettings) -> dict[str, object]:
    """
    Make a paddle prim a dynamic, gravity-free, collision-free body driven by velocity.

    :param stage: The active stage, stopped.
    :param paddle_path: The paddle prim (blade centre at its origin).
    :param settings: Paddle settings.
    :return: What was authored.
    :raises RuntimeError: If the settings ask for a kinematic or ball-colliding paddle
    """

    from pxr import Gf, PhysxSchema, UsdPhysics

    if settings.kinematic or settings.collides_with_ball:
        raise RuntimeError("the game paddle is a dynamic, non-colliding body: velocities written to kinematic PhysX bodies are ignored")
    prim = stage.GetPrimAtPath(paddle_path)
    if not prim.IsValid():
        raise RuntimeError(f"paddle prim {paddle_path} is missing")
    removed = strip_colliders(stage, paddle_path)
    rigid: Any = _applied(UsdPhysics.RigidBodyAPI.Apply(prim), "RigidBodyAPI")
    rigid.CreateKinematicEnabledAttr().Set(False)
    mass = _applied(UsdPhysics.MassAPI.Apply(prim), "MassAPI")
    mass.CreateMassAttr().Set(float(settings.mass_kg))
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(value) for value in settings.inertia_kgm2)))
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    body = _applied(PhysxSchema.PhysxRigidBodyAPI.Apply(prim), "PhysxRigidBodyAPI")
    body.CreateDisableGravityAttr().Set(not settings.gravity)
    body.CreateLinearDampingAttr().Set(0.0)
    body.CreateAngularDampingAttr().Set(0.0)
    body.CreateSleepThresholdAttr().Set(0.0)
    body.CreateStabilizationThresholdAttr().Set(0.0)
    body.CreateMaxAngularVelocityAttr().Set(100000.0)
    return {"paddle_prim": paddle_path, "mass_kg": settings.mass_kg, "colliders_removed": removed, "gravity": settings.gravity}


__all__ = ["BALL_MATERIAL_PATH", "author_ball_body", "author_paddle_body", "configure_scene", "make_passive_marker", "strip_colliders"]
