"""
A humanoid with its paddle at the HIM table, in Isaac Lab: :func:`build_table_scene`.

**What is in each env** (origin on the floor below the table-top centre, the robot behind the ``-x`` end line):

- **The robot** from its :class:`~lib.robots.spec.RobotSpec`: its USD with the hand pruned, the paddle fixed to its
  wrist as one more link, every link colliding with the URDF's collider (:func:`author_robot`), its motors and PD
  gains as Isaac Lab ``DCMotorCfg`` actuators;
- **the table and net** for the robot to touch: invisible static boxes (the board, the undercarriage, the net);
- **the published table, net and paddle meshes** to look at (render only);
- **the ball**, which is not a PhysX body. It is the calibrated ball physics of :mod:`lib.physics.robot_coupling`,
  flown in 1 ms sub-steps of each 5 ms PhysX step against the robot's real blade (and an opponent's), pushing back on
  the paddle. Its table, net and floor are the calibrated contact laws, never PhysX's.

**One policy step** (50 Hz, four 5 ms physics steps, Isaac Lab's own loop). Each physics step:

1. the hooks hand the coupling the opponent's blade (:meth:`TableHooks.opponent_blade`);
2. the coupling flies the ball over the step that just ended, resolves its contacts with the calibrated laws, and
   pushes the paddle back (``F = -J / h``);
3. the joints are driven to ``default_pose + action_scale * action`` through their PD gains.

After the four steps a fallen robot is reset; then the observation is read (:func:`observe`, and the hooks' columns).

**Resets** put the robot at its ready stance 0.35-0.75 m behind the end line (``RobotSpec.stance_m``), within 0.35 m
of the centre line and 0.15 rad of facing the table, its joints within 0.05 rad of the default pose, and take its
env's ball out of play. A fall is the pelvis below ``RobotSpec.fall.pelvis_below_m`` or tilted beyond
``tilt_above_rad``.

**For another humanoid** nothing here changes: everything robot-specific comes from its spec.

Isaac Lab and torch are imported inside functions, so this module imports anywhere.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from lib.robots.spec import LinkColliders, PaddleMount, RobotSpec

if TYPE_CHECKING:
    from pxr import Usd

Tensor = Any

PHYSICS_HZ: int = 200
"""PhysX's rate: 5 ms steps."""

POLICY_HZ: int = 50
"""The policy's rate (4 physics steps a policy step)."""

BALL_SUBSTEP_S: float = 1e-3
"""The ball's sub-step inside each physics step."""

TABLE_TOP_Z_M: float = 0.76
TABLE_ORIGIN_ENV: tuple[float, float, float] = (0.0, 0.0, TABLE_TOP_Z_M)
"""The table-top centre in the env frame, m: the table frame's origin, in which the ball physics works."""

TABLE_SIZE_M: tuple[float, float] = (2.74, 1.525)
BOARD_THICKNESS_M: float = 0.025
UNDERCARRIAGE_HALF_XY_M: tuple[float, float] = (1.212, 0.642)
UNDERCARRIAGE_TOP_M: float = 0.735
NET_SIZE_M: tuple[float, float, float] = (0.003, 1.83, 0.1525)

ENV_SPACING_M: float = 2.5
"""Env spacing; every env collides only with itself and the ground."""

STANCE_LATERAL_M: float = 0.35
STANCE_YAW_RAD: float = 0.15
RESET_JOINT_NOISE_RAD: float = 0.05
"""The reset spread: lateral offset, yaw and joint noise (each uniform within +-)."""

PUSH_INTERVAL_S: tuple[float, float] = (5.0, 8.0)
"""Random shoves: the time between them, s, uniform."""

MAX_LINK_ANGULAR_VELOCITY_DEG_S: float = 1e5
"""PhysX's per-link angular speed cap, deg/s: far above any stroke, so only the joints' own limits bind (Isaac Lab's
robot configs use 1000 deg/s, which a stroking forearm and paddle exceed)."""

PHYSX_GPU_BUFFERS: dict[str, int] = {
    "gpu_max_rigid_patch_count": 2**20,
    "gpu_max_rigid_contact_count": 2**23,
    "gpu_found_lost_pairs_capacity": 2**21,
    "gpu_found_lost_aggregate_pairs_capacity": 2**25,
    "gpu_total_aggregate_pairs_capacity": 2**21,
    "gpu_temp_buffer_capacity": 2**24,
    "gpu_collision_stack_size": 2**26,
    "gpu_heap_capacity": 2**26,
}
"""PhysX's fixed GPU buffers. A full buffer drops contacts, so every run reads PhysX's own peak use against these
(:class:`PhysxBufferMonitor`) and fails on one. All but the patch buffer are Isaac Lab's defaults: Lab's 163,840 patches
is 20 per robot, which standing robots with every link colliding fill; 2**20 holds 8192 envs with room to spare."""


def observation_terms(num_joints: int) -> tuple[tuple[str, int], ...]:
    """
    The robot's observation (:func:`observe`), in order: ``3 J + 29`` values. Base-frame terms are in the pelvis frame;
    ``base_pose_table_xy_yaw`` is the pelvis in the table frame.
    """

    return (
        ("base_lin_vel_b", 3),
        ("base_ang_vel_b", 3),
        ("projected_gravity_b", 3),
        ("base_height", 1),
        ("base_pose_table_xy_yaw", 3),
        ("joint_pos_rel", num_joints),
        ("joint_vel", num_joints),
        ("last_action", num_joints),
        ("paddle_pos_b", 3),
        ("paddle_normal_b", 3),
        ("paddle_vel_b", 3),
        ("ball_pos_b", 3),
        ("ball_vel_b", 3),
        ("ball_active", 1),
    )


class TableHooks(Protocol):
    """
    What a task plugs into the scene (every method optional): the rally (:class:`lib.policy.rally.CoupledRally`), or the
    template's ball watch. Each receives the Lab env (``coupling``, ``num_envs``, ``step_dt``, ``episode_length_buf``,
    ``table_offset``, ``base_frame()``).

    - ``opponent_blade(env, t_end)``: every physics step, before the ball flies: the opponent's blade ``(p, v, q,
      teleport)`` at ``t_end`` (table frame; game axes ``(w, x, y, z)``), or ``None``;
    - ``physics_step(env, contacts)``: every physics step, after the ball flew (its contacts, or ``None``);
    - ``observe(env)``: every policy step, after resets: extra observation columns ``(N, K)``, or ``None``;
    - ``reset(env, env_ids)``: those envs were reset.
    """


def data_tensor(value: Any) -> Tensor:
    """An Isaac Lab data field as a torch tensor (warp-backed fields expose ``.torch``)."""

    return value.torch if hasattr(value, "torch") else value


def quat_xyzw_to_matrix(q: Tensor) -> Tensor:
    """Rotation matrices ``(..., 3, 3)`` from Isaac Lab's ``(x, y, z, w)`` quaternions."""

    import torch

    x, y, z, w = q.unbind(-1)
    norm = torch.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


# ---- the scene ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneFiles:
    """Local paths of the published meshes the scene draws (render only)."""

    table: str
    net: str
    paddle: str
    ball: str


def fetch_scene_files() -> SceneFiles:
    """Fetch (or reuse the session's cached) table, net, paddle and ball meshes at their pinned versions."""

    import antioch

    from lib.assets import ASSETS

    def fetch(name: str) -> str:
        asset = ASSETS[f"ping-pong/{name}"]
        return str(antioch.fetch_asset(asset.name, version=asset.version))

    return SceneFiles(table=fetch("table"), net=fetch("net"), paddle=fetch("paddle"), ball=fetch("ball"))


def table_collider_boxes() -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """The robot's table colliders as boxes, ``name: (centre, size)`` in the env frame, m: the board, the undercarriage
    (one block over the published table's frame footprint) and the net."""

    length, width = TABLE_SIZE_M
    hx, hy = UNDERCARRIAGE_HALF_XY_M
    return {
        "TableBoard": ((0.0, 0.0, TABLE_TOP_Z_M - 0.5 * BOARD_THICKNESS_M), (length, width, BOARD_THICKNESS_M)),
        "TableFrame": ((0.0, 0.0, 0.5 * UNDERCARRIAGE_TOP_M), (2.0 * hx, 2.0 * hy, UNDERCARRIAGE_TOP_M)),
        "Net": ((0.0, 0.0, TABLE_TOP_Z_M + 0.5 * NET_SIZE_M[2]), NET_SIZE_M),
    }


def author_table(stage: Usd.Stage, env_path: str, files: SceneFiles | None) -> None:
    """The robot's invisible table colliders under one env, and (``files``) the published table and net to look at."""

    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    for name, (centre, size) in table_collider_boxes().items():
        cube = UsdGeom.Cube.Define(stage, f"{env_path}/RobotColliders/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        collision.CreateContactOffsetAttr(0.01)
        collision.CreateRestOffsetAttr(0.0)
    if files is not None:
        for name, usd in (("TableVisual", files.table), ("NetVisual", files.net)):
            prim = UsdGeom.Xform.Define(stage, f"{env_path}/{name}").GetPrim()
            prim.GetReferences().AddReference(usd)
            prim.SetInstanceable(True)


# ---- the robot ------------------------------------------------------------------------------------------------------


def author_robot(stage: Usd.Stage, robot_path: str, robot: RobotSpec, paddle_visual: str | None) -> None:
    """
    After the robot's USD is spawned: prune its hand, fix the paddle to its wrist, and give every link the URDF's
    collider (:func:`author_paddle`, :func:`author_link_colliders`).

    :raises RuntimeError: If no prim matches the prune patterns (the asset changed).
    """

    from pxr import Usd

    root = stage.GetPrimAtPath(robot_path)
    if robot.prune:
        pruned = sorted(
            (prim.GetPath() for prim in Usd.PrimRange(root) if any(name in prim.GetName() for name in robot.prune)), key=lambda path: len(str(path))
        )
        count = 0
        for path in pruned:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsActive() and not prim.IsInstanceProxy():
                prim.SetActive(False)
                count += 1
        if count == 0:
            raise RuntimeError(f"no prim under {robot_path!r} matches the prune patterns {robot.prune}")
    author_paddle(stage, robot_path, robot.paddle, paddle_visual)
    author_link_colliders(stage, robot_path, robot.colliders)


def author_paddle(stage: Usd.Stage, robot_path: str, mount: PaddleMount, visual_usd: str | None) -> str:
    """
    The paddle body under the robot prim, on a fixed joint to its wrist link, so PhysX parses it into the articulation
    as one more link: its mass properties, an invisible blade collider (an elliptic slab, convex hull) and handle box
    (robot-table and self contact only; the ball meets the calibrated blade), and the published mesh as its visual.

    :raises RuntimeError: If the wrist link is not under the robot.
    """

    import numpy as np
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    root = stage.GetPrimAtPath(robot_path)
    link = next((prim for prim in Usd.PrimRange(root) if prim.GetName() == mount.link), None)
    if link is None:
        raise RuntimeError(f"paddle link {mount.link!r} not found under {robot_path!r}")
    cache = UsdGeom.XformCache()
    pose = (
        Gf.Matrix4d().SetTranslate(Gf.Vec3d(mount.offset_m, 0.0, 0.0))
        * cache.GetLocalToWorldTransform(link)
        * cache.GetLocalToWorldTransform(root).GetInverse()
    )
    path = f"{robot_path}/{mount.body}"
    paddle = UsdGeom.Xform.Define(stage, path)
    transform = Gf.Transform(pose)
    paddle.ClearXformOpOrder()
    paddle.AddTranslateOp().Set(Gf.Vec3d(transform.GetTranslation()))
    paddle.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(transform.GetRotation().GetQuat()))
    prim = paddle.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass: Any = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(mount.mass_kg)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*mount.com_m))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*mount.inertia))
    body: Any = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateMaxDepenetrationVelocityAttr(10.0)
    body.CreateMaxAngularVelocityAttr(MAX_LINK_ANGULAR_VELOCITY_DEG_S)
    material = UsdShade.Material.Define(stage, f"{path}/rubber")
    surface: Any = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    surface.CreateStaticFrictionAttr(1.0)
    surface.CreateDynamicFrictionAttr(1.0)
    surface.CreateRestitutionAttr(0.0)
    # the blade: an elliptic prism (x tip, y normal, z across), 24 vertices a face (a GPU convex hull holds 64)
    segments = 24
    angle = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = np.stack([mount.blade_half_length * np.cos(angle), np.zeros_like(angle), mount.blade_half_width * np.sin(angle)], -1)
    back, front = ring.copy(), ring.copy()
    back[:, 1], front[:, 1] = -0.5 * mount.blade_thickness, 0.5 * mount.blade_thickness
    blade = UsdGeom.Mesh.Define(stage, f"{path}/blade")
    blade.CreatePointsAttr([Gf.Vec3f(*point) for point in np.concatenate([back, front]).tolist()])
    indices = list(range(segments - 1, -1, -1)) + list(range(segments, 2 * segments))
    for k in range(segments):
        indices += [k, (k + 1) % segments, segments + (k + 1) % segments, segments + k]
    blade.CreateFaceVertexCountsAttr([segments, segments] + [4] * segments)
    blade.CreateFaceVertexIndicesAttr(indices)
    hull: Any = UsdPhysics.MeshCollisionAPI.Apply(blade.GetPrim())
    hull.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    handle = UsdGeom.Cube.Define(stage, f"{path}/handle")
    handle.CreateSizeAttr(1.0)
    handle.AddTranslateOp().Set(Gf.Vec3d(mount.handle_centre_x, 0.0, 0.0))
    handle.AddScaleOp().Set(Gf.Vec3f(*mount.handle_size))
    for geometry in (blade.GetPrim(), handle.GetPrim()):
        UsdPhysics.CollisionAPI.Apply(geometry)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(geometry)
        collision.CreateContactOffsetAttr(0.01)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(geometry).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    if visual_usd is not None:
        for geometry in (blade, handle):
            geometry.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        visual = UsdGeom.Xform.Define(stage, f"{path}/visual")
        visual.GetPrim().GetReferences().AddReference(visual_usd)
        visual.ClearXformOpOrder()
        visual.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*mount.visual_orientation))
        visual.GetPrim().SetInstanceable(True)
    else:
        blade.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.05, 0.05)])
        handle.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.35, 0.18)])
    joint: Any = UsdPhysics.FixedJoint.Define(stage, f"{path}/mount_joint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(link.GetPath()))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(mount.offset_m, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return path


def author_link_colliders(stage: Usd.Stage, robot_path: str, colliders: LinkColliders) -> None:
    """
    The URDF's colliders on the links the USD leaves without one (:class:`~lib.robots.spec.LinkColliders`): a mesh
    link's convex hull of its visual mesh, a cylinder, or a capsule fitted to the visual mesh's extent; then the
    filtered pairs.

    :raises RuntimeError: If a link is missing, already carries a collider, or has no visual mesh to take one from.
    """

    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    def link_path(link: str) -> str:
        path = f"{robot_path}/{link}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"no link {link!r} under {robot_path!r}")
        if any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
            raise RuntimeError(f"link {link!r} already carries a collider: the asset changed")
        return path

    def visual_mesh(path: str) -> tuple[list[Any], list[int], list[int]]:
        """Every visual mesh under a link, in the link's frame: points, face vertex counts, face vertex indices."""

        time = Usd.TimeCode.Default()
        to_link = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(time).GetInverse()
        points: list[Any] = []
        counts: list[int] = []
        indices: list[int] = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI) or "collision" in str(prim.GetPath()).lower():
                continue
            mesh = UsdGeom.Mesh(prim)
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time) * to_link
            offset = len(points)
            points.extend(transform.Transform(Gf.Vec3d(p)) for p in (mesh.GetPointsAttr().Get() or []))
            counts.extend(int(c) for c in (mesh.GetFaceVertexCountsAttr().Get() or []))
            indices.extend(offset + int(i) for i in (mesh.GetFaceVertexIndicesAttr().Get() or []))
        if not points:
            raise RuntimeError(f"link {path!r} has no visual mesh to take a collider from")
        return points, counts, indices

    def collider(prim: Any) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)

    for link in colliders.mesh_hull_links:
        path = link_path(link)
        points, counts, indices = visual_mesh(path)
        mesh = UsdGeom.Mesh.Define(stage, f"{path}/urdf_collision")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        collider(mesh.GetPrim())
        hull: Any = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        hull.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    for link, cylinder_spec in colliders.cylinders.items():
        cylinder = UsdGeom.Cylinder.Define(stage, f"{link_path(link)}/urdf_collision")
        cylinder.CreateRadiusAttr(cylinder_spec.radius_m)
        cylinder.CreateHeightAttr(cylinder_spec.length_m)
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        xform = UsdGeom.Xformable(cylinder)
        xform.AddTranslateOp().Set(Gf.Vec3d(*cylinder_spec.xyz_m))
        roll, pitch, yaw = cylinder_spec.rpy_rad
        # URDF rpy: R = Rz(yaw) Ry(pitch) Rx(roll)
        rotation = (
            Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw))
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(pitch))
            * Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(roll))
        )
        xform.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
        collider(cylinder.GetPrim())
    for link in colliders.fitted_capsule_links:
        path = link_path(link)
        points, _counts, _indices = visual_mesh(path)
        lo = [min(p[k] for p in points) for k in range(3)]
        hi = [max(p[k] for p in points) for k in range(3)]
        size = [hi[k] - lo[k] for k in range(3)]
        axis = max(range(3), key=lambda k: size[k])
        radius = 0.5 * max(size[k] for k in range(3) if k != axis)
        capsule = UsdGeom.Capsule.Define(stage, f"{path}/fitted_collision")
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(max(size[axis] - 2.0 * radius, 0.0))
        capsule.CreateAxisAttr("XYZ"[axis])
        UsdGeom.Xformable(capsule).AddTranslateOp().Set(Gf.Vec3d(*[0.5 * (lo[k] + hi[k]) for k in range(3)]))
        collider(capsule.GetPrim())
    for a, b in colliders.filtered_pairs:
        pairs: Any = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(f"{robot_path}/{a}"))
        pairs.CreateFilteredPairsRel().AddTarget(f"{robot_path}/{b}")


def robot_cfg(robot: RobotSpec, *, fixed_base: bool, paddle_visual: str | None, base_position: tuple[float, float, float]) -> Any:
    """
    The robot as an Isaac Lab ``ArticulationCfg``: its USD (:func:`author_robot` after spawn), self-collisions on,
    solver iterations 8 / 4, no link damping, the link speed cap far above any stroke, each motor group a ``DCMotorCfg``
    (peak torque flat to the knee speed, falling to zero at the no-load speed; armature; the spec's PD gains).
    """

    import isaaclab.sim as lab_sim
    from isaaclab.actuators import DCMotorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    sim_utils: Any = lab_sim

    def spawn_robot(prim_path: str, cfg: Any, translation: Any = None, orientation: Any = None, **kwargs: Any) -> Any:
        from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
        from isaaclab.sim.utils.stage import get_current_stage

        prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
        author_robot(get_current_stage(), prim_path, robot, paddle_visual)
        return prim

    spawn = sim_utils.UsdFileCfg(
        usd_path=robot.usd.format(ISAAC_NUCLEUS_DIR=ISAAC_NUCLEUS_DIR),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=MAX_LINK_ANGULAR_VELOCITY_DEG_S,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=fixed_base, enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    )
    spawn.func = clone(spawn_robot)
    actuators: dict[str, Any] = {
        group.name: DCMotorCfg(
            joint_names_expr=list(group.joints),
            effort_limit=group.peak_torque,
            saturation_effort=group.saturation_effort,
            velocity_limit=group.no_load_speed,
            velocity_limit_sim=group.speed_limit,
            stiffness=dict(group.kp),
            damping=dict(group.kd),
            armature=group.armature,
        )
        for group in robot.actuators
    }
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(pos=base_position, joint_pos=dict(robot.default_pose), joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.9,
        actuators=actuators,
    )


@dataclass(frozen=True)
class JointFrame:
    """One joint on the chain from the robot's root to its paddle, as PhysX simulates it (local frames as authored)."""

    name: str | None
    parent_body: str
    child_body: str
    type: str
    local_pos0: tuple[float, float, float]
    local_rot0: tuple[float, float, float, float]
    local_pos1: tuple[float, float, float]
    local_rot1: tuple[float, float, float, float]
    axis: str


def paddle_chain(stage: Usd.Stage, robot_path: str, body: str) -> list[JointFrame]:
    """
    The joints from the articulation root to ``body`` (the paddle), root first, read from the live stage: the coupling
    builds the paddle's Jacobian from them and checks it against PhysX's after the first step.

    :raises RuntimeError: If ``body`` is not reached, a joint on the way is neither revolute nor fixed, or a body on the
        way carries a non-unit scale (the Jacobian assumes rigid, unscaled frames).
    """

    from pxr import Usd, UsdGeom, UsdPhysics

    def vec(value: Any) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0) if value is None else (float(value[0]), float(value[1]), float(value[2]))

    def quat(value: Any) -> tuple[float, float, float, float]:
        if value is None:
            return (1.0, 0.0, 0.0, 0.0)
        imaginary = value.GetImaginary()
        return (float(value.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2]))

    def scale(prim: Any) -> tuple[float, ...]:
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                return tuple(float(v) for v in op.Get())
        return (1.0, 1.0, 1.0)

    parents: dict[str, tuple[Any, str, Any]] = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or not prim.IsA(UsdPhysics.Joint):
            continue
        joint: Any = UsdPhysics.Joint(prim)
        if joint.GetExcludeFromArticulationAttr().Get():
            continue
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if body0 and body1:
            if body1[0].name in parents:
                raise RuntimeError(f"body {body1[0].name!r} has more than one parent joint")
            parents[body1[0].name] = (prim, body0[0].name, stage.GetPrimAtPath(body1[0]))
    chain: list[JointFrame] = []
    current = body
    while current in parents and len(chain) <= len(parents):
        prim, parent, child = parents[current]
        if child.IsValid() and any(abs(v - 1.0) > 1e-9 for v in scale(child)):
            raise RuntimeError(f"body {current!r} on the paddle's chain carries a non-unit scale {scale(child)}")
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute: Any = UsdPhysics.RevoluteJoint(prim)
            kind, name, axis = "revolute", prim.GetName(), str(revolute.GetAxisAttr().Get() or "X")
        elif prim.IsA(UsdPhysics.FixedJoint):
            kind, name, axis = "fixed", None, "X"
        else:
            raise RuntimeError(f"joint {prim.GetPath()} on the paddle's chain is {prim.GetTypeName()}, not revolute or fixed")
        frames: Any = UsdPhysics.Joint(prim)
        chain.append(
            JointFrame(
                name=name,
                parent_body=parent,
                child_body=current,
                type=kind,
                local_pos0=vec(frames.GetLocalPos0Attr().Get()),
                local_rot0=quat(frames.GetLocalRot0Attr().Get()),
                local_pos1=vec(frames.GetLocalPos1Attr().Get()),
                local_rot1=quat(frames.GetLocalRot1Attr().Get()),
                axis=axis,
            )
        )
        current = parent
    if not chain or len(chain) > len(parents):
        raise RuntimeError(f"no joint chain leads to body {body!r} under {robot_path!r}")
    return list(reversed(chain))


# ---- PhysX's own account --------------------------------------------------------------------------------------------

PHYSX_BUFFER_STATS: dict[str, str] = {
    "gpu_max_rigid_patch_count": "gpu_mem_rigid_patch_count",
    "gpu_max_rigid_contact_count": "gpu_mem_rigid_contact_count",
    "gpu_found_lost_pairs_capacity": "gpu_mem_found_lost_pairs",
    "gpu_found_lost_aggregate_pairs_capacity": "gpu_mem_found_lost_aggregate_pairs",
    "gpu_total_aggregate_pairs_capacity": "gpu_mem_total_aggregate_pairs",
    "gpu_temp_buffer_capacity": "gpu_mem_temp_buffer_capacity",
}
"""Each checked GPU buffer and the PhysX scene statistic reporting its peak use (above the size when it overflowed)."""


class PhysxBufferMonitor:
    """PhysX's GPU buffers' peak use against their sizes, and every PhysX error from construction on."""

    def __init__(self, sizes: dict[str, int]) -> None:
        import omni.physx

        self.sizes = dict(sizes)
        self.errors: list[str] = []
        self._subscription = omni.physx.get_physx_interface().get_error_event_stream().create_subscription_to_pop(self._on_error)

    def _on_error(self, event: Any) -> None:
        payload = getattr(event, "payload", None)
        try:
            fields = dict(payload) if payload is not None else {}
            message = fields.get("errorString") or fields.get("message") or (str(fields) if fields else None)
        except Exception:
            message = str(payload)
        self.errors.append(str(message if message is not None else event.type))

    def summary(self) -> dict[str, Any]:
        """Per buffer its peak use, size and share; the full ones (peak at or above the size); PhysX's errors."""

        from isaaclab.sim.utils.stage import get_current_stage
        from omni.physx import get_physx_statistics_interface
        from omni.physx.bindings._physx import PhysicsSceneStats
        from pxr import PhysicsSchemaTools, UsdPhysics, UsdUtils

        stage = get_current_stage()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        interface = get_physx_statistics_interface()
        peaks = dict.fromkeys(PHYSX_BUFFER_STATS, 0)
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Scene):
                continue
            stats = PhysicsSceneStats()
            if not interface.get_physx_scene_statistics(stage_id, PhysicsSchemaTools.sdfPathToInt(prim.GetPath()), stats):  # pyright: ignore[reportAttributeAccessIssue]
                raise RuntimeError(f"PhysX gave no statistics for scene {prim.GetPath()}")
            for name, field in PHYSX_BUFFER_STATS.items():
                peaks[name] += int(getattr(stats, field))
        buffers = {name: {"peak": peaks[name], "size": self.sizes[name], "use": peaks[name] / self.sizes[name]} for name in PHYSX_BUFFER_STATS}
        overflows = [m for m in self.errors if "overflow" in m.lower() or ("increase" in m.lower() and "capacity" in m.lower())]
        return {
            "buffers": buffers,
            "full": sorted(name for name, b in buffers.items() if b["peak"] >= b["size"]),
            "errors": len(self.errors),
            "overflow_errors": len(overflows),
            "first_errors": self.errors[:3],
        }


def physx_ok(summary: dict[str, Any]) -> bool:
    """Every buffer has headroom and PhysX reported no overflow."""

    return bool(summary.get("buffers")) and not summary.get("full") and summary.get("overflow_errors", 0) == 0


# ---- the Isaac Lab environment --------------------------------------------------------------------------------------


def make_env(
    robot: RobotSpec,
    num_envs: int,
    *,
    hooks: Any,
    opponent: bool,
    files: SceneFiles | None,
    fixed_base: bool,
    reset_spread: bool,
    push_speed_mps: float,
    episode_s: float,
    seed: int,
    env_spacing_m: float,
) -> Any:
    """The Isaac Lab ``DirectRLEnv`` behind :class:`TableScene` (see the module notes), reset and ready to step."""

    import isaaclab.sim as lab_sim
    import torch
    from isaaclab.assets import Articulation as LabArticulation
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab.utils import configclass
    from isaaclab_physx.physics import PhysxCfg

    from lib.physics.robot_coupling import CouplingConfig, RobotBallCoupling

    sim_utils: Any = lab_sim
    articulation_class: Any = LabArticulation
    num_joints = robot.num_joints
    behind = 0.5 * (robot.stance_m[0] + robot.stance_m[1])
    base = (-(0.5 * TABLE_SIZE_M[0] + behind), 0.0, robot.base_height_m)
    extra = int(getattr(hooks, "observation_size", 0))
    articulation = robot_cfg(robot, fixed_base=fixed_base, paddle_visual=files.paddle if files is not None else None, base_position=base)
    buffers: dict[str, Any] = dict(PHYSX_GPU_BUFFERS)

    @configclass
    class TableSceneCfg(DirectRLEnvCfg):
        decimation: int = PHYSICS_HZ // POLICY_HZ
        episode_length_s: float = episode_s
        action_space: int = num_joints
        observation_space: int = 3 * num_joints + 29 + extra
        sim: SimulationCfg = SimulationCfg(
            dt=1.0 / PHYSICS_HZ,
            render_interval=PHYSICS_HZ // POLICY_HZ,
            gravity=(0.0, 0.0, -9.81),
            physics=PhysxCfg(solver_type=0, **buffers),  # PGS: under TGS a contact early in a step loses restitution
        )
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing_m, replicate_physics=True)
        robot: Any = articulation

    cfg = TableSceneCfg()
    cfg.seed = seed

    class TableEnv(DirectRLEnv):
        """The robot at the table (see the module notes)."""

        def __init__(self, cfg: Any, **kwargs: Any) -> None:
            super().__init__(cfg, **kwargs)
            device = self.device
            articulation: Any = self.robot
            missing = sorted(set(robot.joint_names) ^ set(articulation.joint_names))
            if missing:
                raise RuntimeError(f"the articulation's joints differ from {robot.name}'s spec: {missing}")
            self.spec = robot
            self.hooks = hooks
            self.paddle_id = articulation.find_bodies(robot.paddle.body)[0][0]
            self.table_offset = (self.scene.env_origins + torch.tensor(TABLE_ORIGIN_ENV, device=device)).to(torch.float64)
            self.actions = torch.zeros(self.num_envs, num_joints, device=device)
            scales = robot.action_scale
            self.action_scale = torch.tensor(
                [next(scales[e] for e in scales if re.fullmatch(e, name)) for name in articulation.joint_names], dtype=torch.float32, device=device
            )
            self.coupling = RobotBallCoupling(
                articulation,
                self.scene.env_origins,
                self.physics_dt,
                CouplingConfig(
                    paddle_body=robot.paddle.body,
                    table_origin=TABLE_ORIGIN_ENV,
                    precision="float64",
                    substep=BALL_SUBSTEP_S,
                    opponent=opponent,
                    compliance="articulated",
                    track="line",
                    timing=False,
                    jacobian="torch",
                    wrench="direct",
                ),
                chain=paddle_chain(get_current_stage(), "/World/envs/env_0/Robot", robot.paddle.body),
            )
            self.opponent_pose: Any = None
            self.falls = 0
            self.physx = PhysxBufferMonitor(PHYSX_GPU_BUFFERS)
            low, high = PUSH_INTERVAL_S
            self.next_push_s = torch.empty(self.num_envs, dtype=torch.float64, device=device).uniform_(low, high)

        def _setup_scene(self) -> None:
            config: Any = self.cfg
            self.robot: Any = articulation_class(config.robot)
            sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
            light = sim_utils.DomeLightCfg(intensity=900.0, color=(0.95, 0.95, 1.0))
            light.func("/World/dome", light)
            author_table(get_current_stage(), "/World/envs/env_0", files)
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
            self.scene.articulations["robot"] = self.robot

        def sim_time(self) -> float:
            """Time at the start of the current physics step, s."""

            return (self._sim_step_counter - 1) * self.physics_dt

        def base_frame(self) -> tuple[Tensor, Tensor]:
            """The pelvis's world position ``(N, 3)`` and rotation ``(N, 3, 3)``."""

            data: Any = self.robot.data
            return data_tensor(data.root_link_pos_w), quat_xyzw_to_matrix(data_tensor(data.root_link_quat_w))

        def _pre_physics_step(self, actions: Tensor) -> None:
            if push_speed_mps > 0.0 and not fixed_base:
                self._push()
            self.actions = actions.clone()

        def _apply_action(self) -> None:
            t = self.sim_time()
            blade = getattr(hooks, "opponent_blade", None)
            if opponent and blade is not None:
                # the opponent's blade at the end of the step the coupling flies now, which ends at this PhysX time
                pose = blade(self, t)
                if pose is not None:
                    p, v, q, teleport = pose
                    self.coupling.set_opponent(p, v, q, teleport)
                    self.opponent_pose = (p, q)
            contacts = self.coupling.step()
            watch = getattr(hooks, "physics_step", None)
            if watch is not None:
                watch(self, contacts)
            target = data_tensor(self.robot.data.default_joint_pos) + self.action_scale * self.actions
            self.robot.set_joint_position_target_index(target=target)

        def _get_observations(self) -> dict[str, Tensor]:
            columns = [observe(self)]
            extra_columns = getattr(hooks, "observe", None)
            if extra_columns is not None:
                more = extra_columns(self)
                if more is not None:
                    columns.append(more.to(columns[0].dtype))
            return {"policy": torch.cat(columns, -1)}

        def _get_rewards(self) -> Tensor:
            return torch.zeros(self.num_envs, device=self.device)

        def fallen(self) -> Tensor:
            """Per env: the pelvis below the fall height, or tilted beyond the fall angle."""

            data: Any = self.robot.data
            if fixed_base:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            height = data_tensor(data.root_link_pos_w)[:, 2] - self.scene.env_origins[:, 2]
            tilt = torch.acos((-data_tensor(data.projected_gravity_b)[:, 2]).clamp(-1.0, 1.0))
            return (height < robot.fall.pelvis_below_m) | (tilt > robot.fall.tilt_above_rad)

        def _get_dones(self) -> tuple[Tensor, Tensor]:
            fell = self.fallen()
            self.falls += int(fell.sum())
            return fell, self.episode_length_buf >= self.max_episode_length - 1

        def _push(self) -> None:
            """Random shoves: every env whose shove is due gets a horizontal velocity kick of up to ``push_speed_mps``."""

            due = self.episode_length_buf.to(torch.float64) * self.step_dt >= self.next_push_s
            if not bool(due.any()):
                return
            rows = torch.nonzero(due).flatten()
            data: Any = self.robot.data
            velocity = data_tensor(data.root_link_vel_w)[rows].clone()
            velocity[:, :2] += torch.empty(rows.numel(), 2, device=self.device).uniform_(-push_speed_mps, push_speed_mps)
            self.robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=rows)
            low, high = PUSH_INTERVAL_S
            self.next_push_s[rows] += torch.empty(rows.numel(), dtype=torch.float64, device=self.device).uniform_(low, high)

        def _reset_idx(self, env_ids: Any) -> None:
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            env_ids = torch.as_tensor(env_ids, device=self.device).long()
            super()._reset_idx(env_ids)
            data: Any = self.robot.data
            root = data_tensor(data.default_root_state)[env_ids].clone()
            root[:, :3] += self.scene.env_origins[env_ids]
            q = data_tensor(data.default_joint_pos)[env_ids].clone()
            if reset_spread:
                count = env_ids.numel()
                low, high = robot.stance_m
                root[:, 0] += behind - torch.empty(count, device=self.device).uniform_(low, high)
                root[:, 1] += torch.empty(count, device=self.device).uniform_(-STANCE_LATERAL_M, STANCE_LATERAL_M)
                yaw = torch.empty(count, device=self.device).uniform_(-STANCE_YAW_RAD, STANCE_YAW_RAD)
                root[:, 3:7] = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)], -1)
                q += torch.empty_like(q).uniform_(-RESET_JOINT_NOISE_RAD, RESET_JOINT_NOISE_RAD)
            self.robot.write_root_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
            self.robot.write_root_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=env_ids)
            self.robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=env_ids)
            if getattr(self, "coupling", None) is not None:
                self.coupling.reset(env_ids)
            if getattr(self, "next_push_s", None) is not None:
                low, high = PUSH_INTERVAL_S
                self.next_push_s[env_ids] = torch.empty(env_ids.numel(), dtype=torch.float64, device=self.device).uniform_(low, high)
            forget = getattr(hooks, "reset", None)
            if forget is not None and getattr(self, "coupling", None) is not None:
                forget(self, env_ids)

    env = TableEnv(cfg)
    if env._physics_handles_decimation:
        raise RuntimeError("the per-step ball coupling needs Isaac Lab's own decimation loop (PhysX)")
    env.reset()
    env.coupling.start()
    return env


def observe(env: Any) -> Tensor:
    """
    The robot's observation (:func:`observation_terms`): its base's velocities, gravity, height and pose at the table,
    its joints relative to the default pose, their velocities, its last action, its paddle, and the ball (zero while
    it is out of play) in its base frame.
    """

    import torch

    data: Any = env.robot.data
    position, rotation = env.base_frame()
    to_base = rotation.transpose(-1, -2)
    table = (position.to(torch.float64) - env.table_offset).to(position.dtype)
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    paddle_p = data_tensor(data.body_link_pos_w)[:, env.paddle_id] - position
    paddle_v = data_tensor(data.body_link_lin_vel_w)[:, env.paddle_id]
    paddle_normal = quat_xyzw_to_matrix(data_tensor(data.body_link_quat_w)[:, env.paddle_id])[:, :, 1]
    ball = env.coupling.ball.to(position.dtype)
    ball_p = ball[:, :3] + env.table_offset.to(position.dtype) - position
    active = env.coupling.active.to(position.dtype)[:, None]

    def base(vector: Tensor) -> Tensor:
        return (to_base @ vector[..., None])[..., 0]

    return torch.cat(
        [
            data_tensor(data.root_lin_vel_b),
            data_tensor(data.root_ang_vel_b),
            data_tensor(data.projected_gravity_b),
            position[:, 2:3],
            torch.stack([table[:, 0], table[:, 1], yaw], -1),
            data_tensor(data.joint_pos) - data_tensor(data.default_joint_pos),
            data_tensor(data.joint_vel),
            env.actions,
            base(paddle_p),
            base(paddle_normal),
            base(paddle_v),
            base(ball_p),
            base(ball[:, 3:6]),
            active,
        ],
        -1,
    )


class TableScene:
    """
    The robot at the table in every env: step it with joint actions, read its observation. Built by
    :func:`build_table_scene`; the Isaac Lab env is :attr:`env`.
    """

    def __init__(self, env: Any, robot: RobotSpec, files: SceneFiles | None) -> None:
        self.env = env
        self.robot = robot
        self.files = files

    @property
    def num_envs(self) -> int:
        return int(self.env.num_envs)

    @property
    def step_dt(self) -> float:
        """The policy step, s."""

        return float(self.env.step_dt)

    @property
    def coupling(self) -> Any:
        """The ball physics (:class:`lib.physics.robot_coupling.RobotBallCoupling`)."""

        return self.env.coupling

    @property
    def falls(self) -> int:
        """Falls so far (each reset the env)."""

        return int(self.env.falls)

    @property
    def joint_names(self) -> list[str]:
        """The articulation's joints, in the order of actions and joint observations."""

        return list(self.env.robot.joint_names)

    def reset_all(self) -> Tensor:
        """Reset every env (robot at its stance, ball out of play, hooks told); the first observation."""

        observation, _extras = self.env.reset()
        return observation["policy"]

    def step(self, actions: Tensor) -> Tensor:
        """One policy step with joint actions ``(N, J)`` (units of each joint's action scale); the next observation."""

        observation, _reward, _terminated, _truncated, _extras = self.env.step(actions)
        return observation["policy"]

    def hold_default_pose(self) -> Tensor:
        """The actions that hold the ready stance: zeros. Replace them with your controller's."""

        import torch

        return torch.zeros(self.num_envs, self.robot.num_joints, device=self.env.device)

    def physx_buffers(self) -> dict[str, Any]:
        """PhysX's GPU buffers' peak use and its errors so far (:class:`PhysxBufferMonitor`)."""

        return self.env.physx.summary()

    def close(self) -> None:
        self.env.close()


def build_table_scene(
    robot: RobotSpec,
    num_envs: int,
    *,
    hooks: Any = None,
    opponent: bool = False,
    visuals: bool = True,
    fixed_base: bool = False,
    reset_spread: bool = True,
    push_speed_mps: float = 0.0,
    episode_s: float = 60.0,
    seed: int = 0,
    env_spacing_m: float = ENV_SPACING_M,
) -> TableScene:
    """
    A humanoid with its paddle at the table in ``num_envs`` parallel worlds, with the calibrated ball (Kit must be up).

    :param robot: The humanoid (:data:`lib.robots.ROBOTS`).
    :param hooks: The task (:class:`TableHooks`): the rally, or the template's ball watch.
    :param opponent: A second, kinematic blade the hooks drive (the opponent's paddle).
    :param visuals: Draw the published table, net and paddle (fetched from the asset store); off, the colliders only.
    :param fixed_base: Pin the pelvis to the world.
    :param reset_spread: Resets spread the stance and the joints (see the module notes); off, every env starts alike.
    :param push_speed_mps: Random shoves of up to this horizontal speed every 5-8 s (0: none).
    :param episode_s: An episode's longest span before a time-out reset, s.
    :param seed: Isaac Lab's seed.
    """

    files = fetch_scene_files() if visuals else None
    env = make_env(
        robot,
        num_envs,
        hooks=hooks,
        opponent=opponent,
        files=files,
        fixed_base=fixed_base,
        reset_spread=reset_spread,
        push_speed_mps=push_speed_mps,
        episode_s=episode_s,
        seed=seed,
        env_spacing_m=env_spacing_m,
    )
    return TableScene(env, robot, files)


__all__ = [
    "BALL_SUBSTEP_S",
    "PHYSICS_HZ",
    "POLICY_HZ",
    "TABLE_ORIGIN_ENV",
    "TableHooks",
    "TableScene",
    "build_table_scene",
    "observation_terms",
    "observe",
    "physx_ok",
]
