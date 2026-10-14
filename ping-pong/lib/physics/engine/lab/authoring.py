"""
Author the ball, paddle, table and net of every environment on the Isaac Lab stage.

Runs after Kit has booted; every simulator import is inside a function.
Environment 0 is authored by Isaac Lab spawners and cloned to the others;
:func:`apply_env_physics` then writes each environment's own values (materials,
masses, caps, colliders, drive) before ``sim.reset()`` parses the stage, which
is why a batch that varies them must not replicate physics. Every physics
attribute is written explicitly, so nothing depends on an engine default.

Frames: each environment's local frame is the :mod:`lib.physics` table frame
(Z up, origin at the centre of the playing surface). The blade's local X
points towards the handle and local Z is the forehand face normal, as
:class:`lib.physics.reference.surfaces.PaddlePose` defines them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from lib.physics.engine.lab.batch import PARK_POSITION, BatchSpec, EngineConfig
from lib.physics.engine.lab.materials import BALL_RESTITUTION, COMBINE_MODE, EnvPhysics

if TYPE_CHECKING:
    from pxr import Usd

BALL_NAME = "Ball"
PADDLE_NAME = "Paddle"
TABLE_NAME = "Table"
NET_NAME = "Net"
GEOM = "geom"
MATERIAL = "PhysMat"

ENV_ROOT = "/World/envs"


def _collider(stage: Usd.Stage, geom: Usd.Prim, material_path: str, engine: EngineConfig, restitution: float, friction: float, enabled: bool) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    collision: Any = UsdPhysics.CollisionAPI.Apply(geom)
    collision.CreateCollisionEnabledAttr().Set(bool(enabled))
    physx = PhysxSchema.PhysxCollisionAPI.Apply(geom)
    physx.CreateContactOffsetAttr().Set(float(engine.contact_offset_m))
    physx.CreateRestOffsetAttr().Set(float(engine.rest_offset_m))
    material = UsdShade.Material.Define(stage, material_path)
    api: Any = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateRestitutionAttr().Set(float(restitution))
    api.CreateStaticFrictionAttr().Set(float(friction))
    api.CreateDynamicFrictionAttr().Set(float(friction))
    combine = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    combine.CreateRestitutionCombineModeAttr().Set(COMBINE_MODE)
    combine.CreateFrictionCombineModeAttr().Set(COMBINE_MODE)
    UsdShade.MaterialBindingAPI.Apply(geom).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _body(
    prim: Usd.Prim,
    engine: EngineConfig,
    mass: float,
    inertia: Sequence[float],
    *,
    kinematic: bool,
    gravity: bool,
    max_angular_velocity_deg_s: float,
    damping: dict[str, float],
) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    body: Any = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateKinematicEnabledAttr().Set(bool(kinematic))
    mass_api: Any = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in inertia]))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx.CreateMaxAngularVelocityAttr().Set(float(max_angular_velocity_deg_s))
    physx.CreateLinearDampingAttr().Set(float(damping["linear_damping"]))
    physx.CreateAngularDampingAttr().Set(float(damping["angular_damping"]))
    physx.CreateSleepThresholdAttr().Set(float(damping["sleep_threshold"]))
    physx.CreateStabilizationThresholdAttr().Set(float(damping["stabilization_threshold"]))
    physx.CreateMaxDepenetrationVelocityAttr().Set(float(damping["max_depenetration_velocity_mps"]))
    physx.CreateSolverPositionIterationCountAttr().Set(int(engine.position_iterations))
    physx.CreateSolverVelocityIterationCountAttr().Set(int(engine.velocity_iterations))
    physx.CreateEnableCCDAttr().Set(False)
    physx.CreateDisableGravityAttr().Set(not gravity)


def _xform(stage: Usd.Stage, path: str, translation: Sequence[float]) -> Usd.Prim:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    return xform.GetPrim()


def _box(stage: Usd.Stage, path: str, size: Sequence[float]) -> Usd.Prim:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr().Set(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    return cube.GetPrim()


def make_authors(spec: BatchSpec, first: EnvPhysics, ball_friction: float, damping: dict[str, float]) -> dict[str, Callable[[Usd.Stage, str], Usd.Prim]]:
    """
    Closures that author each asset of environment 0 at a prim path.

    :param spec: The batch (shared geometry and engine configuration).
    :param first: Environment 0's values; the others are written by :func:`apply_env_physics`.
    :param ball_friction: The ball material's own friction (above every surface's).
    :param damping: Ball damping, thresholds and depenetration velocity shared with the game adapter.
    :return: ``{"ball", "paddle", "table", "net"}`` to an author callable.
    """

    geometry = spec.geometry
    engine = spec.engine
    half_span = geometry.net_half_span

    def ball(stage: Usd.Stage, path: str) -> Usd.Prim:
        from pxr import UsdGeom

        prim = _xform(stage, path, (0.0, 0.0, 1.0))
        _body(
            prim,
            engine,
            first.ball_mass,
            (first.ball_inertia,) * 3,
            kinematic=False,
            gravity=True,
            max_angular_velocity_deg_s=first.max_angular_velocity_deg_s,
            damping=damping,
        )
        sphere = UsdGeom.Sphere.Define(stage, f"{path}/{GEOM}")
        sphere.CreateRadiusAttr().Set(float(first.ball_radius))
        _collider(stage, sphere.GetPrim(), f"{path}/{MATERIAL}", engine, BALL_RESTITUTION, ball_friction, True)
        return prim

    def paddle(stage: Usd.Stage, path: str) -> Usd.Prim:
        prim = _xform(stage, path, PARK_POSITION)
        no_damping = {**damping, "linear_damping": 0.0, "angular_damping": 0.0}
        _body(
            prim,
            engine,
            first.paddle_mass,
            first.paddle_inertia,
            kinematic=first.paddle_kinematic,
            gravity=False,
            max_angular_velocity_deg_s=100000.0,
            damping=no_damping,
        )
        size = (2.0 * geometry.paddle.blade_half_length, 2.0 * geometry.paddle.blade_half_width, geometry.paddle.thickness)
        blade = _box(stage, f"{path}/{GEOM}", size)
        _collider(stage, blade, f"{path}/{MATERIAL}", engine, first.rubber_restitution, first.rubber_friction, first.paddle_collides)
        return prim

    def table(stage: Usd.Stage, path: str) -> Usd.Prim:
        prim = _xform(stage, path, (0.0, 0.0, -0.5 * engine.table_thickness_m))
        top = _box(stage, f"{path}/{GEOM}", (geometry.table.length, geometry.table.width, engine.table_thickness_m))
        _collider(stage, top, f"{path}/{MATERIAL}", engine, first.table_restitution, first.table_friction, first.table_collides)
        return prim

    def net(stage: Usd.Stage, path: str) -> Usd.Prim:
        prim = _xform(stage, path, (0.0, 0.0, 0.5 * geometry.net.height))
        slab = _box(stage, f"{path}/{GEOM}", (geometry.net.thickness, 2.0 * half_span, geometry.net.height))
        _collider(stage, slab, f"{path}/{MATERIAL}", engine, first.net_restitution, first.net_friction, first.net_collides)
        return prim

    return {"ball": ball, "paddle": paddle, "table": table, "net": net}


def build_scene(spec: BatchSpec, authors: dict[str, Callable[[Usd.Stage, str], Usd.Prim]]) -> tuple[Any, Any]:
    """
    Create Lab's simulation context and the replicated scene (not yet reset).

    :param spec: The batch.
    :param authors: :func:`make_authors` output.
    :return: ``(simulation, scene)``.
    """

    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg, SpawnerCfg
    from isaaclab.sim.utils import clone, get_current_stage
    from isaaclab.utils import configclass
    from isaaclab_physx.physics import PhysxCfg

    engine = spec.engine

    def spawner(name: str, rigid: bool) -> Any:
        @clone
        def _spawn(prim_path: str, spawn_cfg: Any, translation: Any = None, orientation: Any = None, **kwargs: Any) -> Any:
            return authors[name](get_current_stage(), str(prim_path))

        base = RigidObjectSpawnerCfg if rigid else SpawnerCfg
        return base(func=_spawn)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        table = AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/{TABLE_NAME}", spawn=spawner("table", False))
        net = AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/{NET_NAME}", spawn=spawner("net", False))
        ball = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{BALL_NAME}", spawn=spawner("ball", True), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0))
        )
        paddle = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{PADDLE_NAME}", spawn=spawner("paddle", True), init_state=RigidObjectCfg.InitialStateCfg(pos=PARK_POSITION)
        )

    physics = PhysxCfg(
        solver_type=0 if engine.solver == "pgs" else 1,
        enable_ccd=False,
        enable_stabilization=False,
        bounce_threshold_velocity=engine.bounce_threshold_mps,
    )
    gravity = spec.geometry.aero.gravity
    simulation = SimulationContext(SimulationCfg(dt=engine.physics_dt_s, device=engine.device, gravity=(0.0, 0.0, -gravity), physics=physics))
    scene = InteractiveScene(SceneCfg(num_envs=spec.num_envs, env_spacing=engine.env_spacing_m, replicate_physics=engine.replicate_physics))
    return simulation, scene


def apply_env_physics(stage: Usd.Stage, values: Sequence[EnvPhysics]) -> int:
    """
    Write every environment's own physics values over the cloned copy of environment 0.

    :param stage: The stage, before ``sim.reset()``.
    :param values: One entry per environment, in order.
    :return: Number of attributes written.
    :raises RuntimeError: If an environment's prims are missing
    """

    from pxr import Gf, Sdf

    first = values[0]
    written = 0

    def put(path: str, name: str, value: Any) -> None:
        nonlocal written
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"prim {path} is missing")
        attribute = prim.GetAttribute(name)
        if not attribute.IsValid():
            raise RuntimeError(f"{path} has no attribute {name}")
        attribute.Set(value)
        written += 1

    with Sdf.ChangeBlock():
        for index, env in enumerate(values):
            if index == 0 or env == first:
                continue
            root = f"{ENV_ROOT}/env_{index}"
            if (env.ball_mass, env.ball_inertia) != (first.ball_mass, first.ball_inertia):
                put(f"{root}/{BALL_NAME}", "physics:mass", float(env.ball_mass))
                put(f"{root}/{BALL_NAME}", "physics:diagonalInertia", Gf.Vec3f(*(float(env.ball_inertia),) * 3))
            if env.ball_radius != first.ball_radius:
                put(f"{root}/{BALL_NAME}/{GEOM}", "radius", float(env.ball_radius))
            if env.max_angular_velocity_deg_s != first.max_angular_velocity_deg_s:
                put(f"{root}/{BALL_NAME}", "physxRigidBody:maxAngularVelocity", float(env.max_angular_velocity_deg_s))
            for surface, restitution, friction in (
                (TABLE_NAME, env.table_restitution, env.table_friction),
                (PADDLE_NAME, env.rubber_restitution, env.rubber_friction),
                (NET_NAME, env.net_restitution, env.net_friction),
            ):
                put(f"{root}/{surface}/{MATERIAL}", "physics:restitution", float(restitution))
                put(f"{root}/{surface}/{MATERIAL}", "physics:staticFriction", float(friction))
                put(f"{root}/{surface}/{MATERIAL}", "physics:dynamicFriction", float(friction))
            put(f"{root}/{TABLE_NAME}/{GEOM}", "physics:collisionEnabled", bool(env.table_collides))
            put(f"{root}/{NET_NAME}/{GEOM}", "physics:collisionEnabled", bool(env.net_collides))
            put(f"{root}/{PADDLE_NAME}/{GEOM}", "physics:collisionEnabled", bool(env.paddle_collides))
            put(f"{root}/{PADDLE_NAME}", "physics:kinematicEnabled", bool(env.paddle_kinematic))
            put(f"{root}/{PADDLE_NAME}", "physics:mass", float(env.paddle_mass))
            put(f"{root}/{PADDLE_NAME}", "physics:diagonalInertia", Gf.Vec3f(*[float(v) for v in env.paddle_inertia]))
    return written


def lattice_origins(count: int, spacing: float) -> list[tuple[float, float, float]]:
    """
    Environment origins on a centred cube lattice, filled x fastest, then y, then z.

    :param count: Number of environments.
    :param spacing: Lattice spacing, m.
    :return: One origin per environment, world frame.
    """

    side = max(1, math.ceil(count ** (1.0 / 3.0) - 1e-9))
    while side**3 < count:
        side += 1
    layers = math.ceil(count / (side * side))
    centre = ((side - 1) / 2.0, (side - 1) / 2.0, (layers - 1) / 2.0)
    out = []
    for index in range(count):
        ix, iy, iz = index % side, (index // side) % side, index // (side * side)
        out.append(((ix - centre[0]) * spacing, (iy - centre[1]) * spacing, (iz - centre[2]) * spacing))
    return out


def place_envs(stage: Usd.Stage, origins: Sequence[tuple[float, float, float]]) -> None:
    """
    Move every environment's root prim to its origin (before ``sim.reset()``).

    :param stage: The stage.
    :param origins: One origin per environment.
    :raises RuntimeError: If an environment root has no translate op
    """

    from pxr import Gf, Sdf

    with Sdf.ChangeBlock():
        for index, origin in enumerate(origins):
            attribute = stage.GetPrimAtPath(f"{ENV_ROOT}/env_{index}").GetAttribute("xformOp:translate")
            if not attribute.IsValid():
                raise RuntimeError(f"{ENV_ROOT}/env_{index} has no xformOp:translate")
            attribute.Set(Gf.Vec3d(*origin))


def read_back(stage: Usd.Stage, indices: Sequence[int]) -> list[dict[str, object]]:
    """
    The physics values the stage holds for some environments, for the run record.

    :param stage: The stage.
    :param indices: Environments to read.
    :return: One mapping per environment.
    """

    out: list[dict[str, object]] = []
    for index in indices:
        root = f"{ENV_ROOT}/env_{index}"

        def get(path: str, name: str) -> object:
            attribute = stage.GetPrimAtPath(path).GetAttribute(name)
            value = attribute.Get() if attribute.IsValid() else None
            if value is None or isinstance(value, bool | int | float | str):
                return value
            try:
                return [float(v) for v in value]
            except TypeError:
                return str(value)

        out.append(
            {
                "env": index,
                "ball_mass": get(f"{root}/{BALL_NAME}", "physics:mass"),
                "ball_inertia": get(f"{root}/{BALL_NAME}", "physics:diagonalInertia"),
                "ball_radius": get(f"{root}/{BALL_NAME}/{GEOM}", "radius"),
                "ball_max_angular_velocity_deg_s": get(f"{root}/{BALL_NAME}", "physxRigidBody:maxAngularVelocity"),
                "ball_restitution": get(f"{root}/{BALL_NAME}/{MATERIAL}", "physics:restitution"),
                "ball_friction": get(f"{root}/{BALL_NAME}/{MATERIAL}", "physics:dynamicFriction"),
                "combine": get(f"{root}/{BALL_NAME}/{MATERIAL}", "physxMaterial:restitutionCombineMode"),
                "table_restitution": get(f"{root}/{TABLE_NAME}/{MATERIAL}", "physics:restitution"),
                "table_friction": get(f"{root}/{TABLE_NAME}/{MATERIAL}", "physics:dynamicFriction"),
                "table_collides": get(f"{root}/{TABLE_NAME}/{GEOM}", "physics:collisionEnabled"),
                "rubber_restitution": get(f"{root}/{PADDLE_NAME}/{MATERIAL}", "physics:restitution"),
                "rubber_friction": get(f"{root}/{PADDLE_NAME}/{MATERIAL}", "physics:dynamicFriction"),
                "paddle_collides": get(f"{root}/{PADDLE_NAME}/{GEOM}", "physics:collisionEnabled"),
                "paddle_kinematic": get(f"{root}/{PADDLE_NAME}", "physics:kinematicEnabled"),
                "paddle_mass": get(f"{root}/{PADDLE_NAME}", "physics:mass"),
                "net_restitution": get(f"{root}/{NET_NAME}/{MATERIAL}", "physics:restitution"),
            }
        )
    return out


__all__ = [
    "BALL_NAME",
    "ENV_ROOT",
    "NET_NAME",
    "PADDLE_NAME",
    "TABLE_NAME",
    "apply_env_physics",
    "build_scene",
    "lattice_origins",
    "make_authors",
    "place_envs",
    "read_back",
]
