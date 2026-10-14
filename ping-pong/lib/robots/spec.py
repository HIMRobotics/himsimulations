"""
What the simulation needs to know about a humanoid, as data: :class:`RobotSpec`.

One spec drives every robot feature: the table scene and the template build the articulation from it
(``lib/robots/table_scene.py``), the policy reads its joints, action scale and fall rule (``lib/policy``), and the
retargeter builds its kinematic model from its URDF (``lib/retarget/model.py``). A new humanoid is one module like
``unitree_g1.py`` that defines its spec, registered in :data:`lib.robots.ROBOTS`.

Joint and link fields are names in the robot's USD (and its URDF, which must use the same names). Joint groups are
regular expressions matched in full against joint names, as Isaac Lab's actuator configs take them.

**What a new humanoid must provide** (each a field of :class:`RobotSpec`):

1. **Its body:** ``usd`` (an articulation whose links and joints carry the URDF's names) and ``urdf`` with its
   ``urdf_sha256``. ``prune`` lists name fragments of the prims the paddle replaces (a hand); at least one must match.
2. **Its joints and motors:** ``joint_names``, and ``actuators`` covering every joint exactly once, each group with its
   torque-speed envelope, rotor armature and PD gains. ``action_fraction`` sets the action scale.
3. **Its stance:** ``default_pose`` (the action origin), ``base_height_m`` and ``stance_m`` behind the end line.
4. **Its paddle:** ``paddle``, the wrist link it replaces the hand on and the blade centre's offset along that link's
   ``+x``. The paddle frame is fixed to the link's axes: ``x`` along the handle to the tip, ``y`` the forehand normal,
   ``z`` across the blade. The mass properties are the paddle's own, so they carry over unchanged.
5. **Its links:** ``links`` (the pelvis is the articulation's root) and ``fall``, when it has fallen.
6. **Its colliders:** ``colliders``, the URDF's collision for every link the USD leaves without one, so the whole body
   meets the table and itself.
7. **Optionally:** ``retarget``, for the stroke retargeter.

The table scene, the template and the film then work unchanged. The table scene refuses at build time if the articulation's joints
differ from ``joint_names``, a prune pattern matches nothing, or a collider names a missing link or one that already
has a collider. The trained policy is the G1's alone: its inputs are this robot's ``3 J + 29`` values plus the ball
forecast, and its outputs are its 29 joints. A new humanoid needs its own policy, which the template is the start of.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MotorGroup(BaseModel):
    """
    One motor type, the joints it drives and their PD gains (Isaac Lab ``DCMotorCfg``).

    Torque is flat at ``peak_torque`` up to ``knee_speed`` and falls linearly to zero at ``no_load_speed``.

    :ivar name: The motor type, used as the actuator group's name.
    :ivar joints: Joint-name regular expressions.
    :ivar peak_torque: Peak torque, N m.
    :ivar speed_limit: Joint speed limit, rad/s (PhysX clamps here).
    :ivar knee_speed: Speed up to which the peak torque holds, rad/s.
    :ivar no_load_speed: Speed at which the torque reaches zero, rad/s.
    :ivar armature: Reflected rotor inertia, kg m^2.
    :ivar kp: PD stiffness per joint expression, N m/rad.
    :ivar kd: PD damping per joint expression, N m s/rad.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    joints: tuple[str, ...] = Field(min_length=1)
    peak_torque: float = Field(gt=0.0)
    speed_limit: float = Field(gt=0.0)
    knee_speed: float = Field(gt=0.0)
    no_load_speed: float = Field(gt=0.0)
    armature: float = Field(gt=0.0)
    kp: dict[str, float]
    kd: dict[str, float]

    @model_validator(mode="after")
    def _gains_cover_the_joints(self) -> MotorGroup:
        if set(self.kp) != set(self.joints) or set(self.kd) != set(self.joints):
            raise ValueError(f"{self.name}: kp and kd must name exactly the group's joints {self.joints}")
        if self.no_load_speed <= self.knee_speed:
            raise ValueError(f"{self.name}: the no-load speed must exceed the knee speed")
        return self

    @property
    def saturation_effort(self) -> float:
        """Torque at zero speed of the line through ``(knee_speed, peak_torque)`` and ``(no_load_speed, 0)``, N m."""

        return self.peak_torque * self.no_load_speed / (self.no_load_speed - self.knee_speed)


class PaddleMount(BaseModel):
    """
    The paddle body, rigidly fixed to a wrist link in place of the hand.

    Paddle frame: origin at the blade centre, x along the handle towards the tip, y the forehand face normal, z across
    the blade.

    :ivar link: The wrist link it is fixed to.
    :ivar body: The paddle body's name in the articulation.
    :ivar offset_m: Blade centre from the link origin along the link's +x, m.
    :ivar mass_kg: Paddle mass, kg.
    :ivar com_m: Centre of mass in the paddle frame, m.
    :ivar inertia: Principal moments about the centre of mass (x, y, z), kg m^2.
    :ivar blade_half_length: Blade semi-axis along x, m.
    :ivar blade_half_width: Blade semi-axis along z, m.
    :ivar blade_thickness: Blade with both rubbers, along y, m.
    :ivar handle_size: Handle box (x, y, z), m.
    :ivar handle_centre_x: Handle box centre along x, m.
    :ivar visual_orientation: Rotation (w, x, y, z) from the paddle asset's frame (X across, Y tip, Z forehand) to the
        paddle frame.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    link: str
    body: str = "paddle"
    offset_m: float = Field(gt=0.0)
    mass_kg: float = Field(gt=0.0)
    com_m: tuple[float, float, float]
    inertia: tuple[float, float, float]
    blade_half_length: float = Field(gt=0.0)
    blade_half_width: float = Field(gt=0.0)
    blade_thickness: float = Field(gt=0.0)
    handle_size: tuple[float, float, float]
    handle_centre_x: float
    visual_orientation: tuple[float, float, float, float] = (0.5, -0.5, -0.5, -0.5)


class RobotLinks(BaseModel):
    """
    The links the features name.

    :ivar pelvis: The floating base.
    :ivar torso: The upper body.
    :ivar feet: Left and right foot links.
    :ivar racket_arm: The paddle arm, shoulder to wrist.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pelvis: str
    torso: str
    feet: tuple[str, str]
    racket_arm: tuple[str, ...] = Field(min_length=1)


class FallRule(BaseModel):
    """
    When the robot has fallen.

    :ivar pelvis_below_m: The pelvis is lower than this above the floor, m.
    :ivar tilt_above_rad: Or the pelvis is tilted further than this from upright, rad.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pelvis_below_m: float = Field(gt=0.0)
    tilt_above_rad: float = Field(gt=0.0)


class CylinderCollider(BaseModel):
    """
    A URDF ``<cylinder>`` collision: its axis is z of its origin frame.

    :ivar radius_m: Radius, m.
    :ivar length_m: Length along the axis, m.
    :ivar xyz_m: Origin in the link frame, m.
    :ivar rpy_rad: Origin rotation, URDF roll-pitch-yaw (``R = Rz(yaw) Ry(pitch) Rx(roll)``), rad.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    radius_m: float = Field(gt=0.0)
    length_m: float = Field(gt=0.0)
    xyz_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]


class LinkColliders(BaseModel):
    """
    The colliders the table scene authors on links whose USD body has none, as the URDF gives them, so every link of the
    robot meets the table and itself (``lib/robots/table_scene.py``). Links not named keep the USD's own colliders.

    :ivar mesh_hull_links: Links the URDF collides with their own mesh: the convex hull of the USD's visual mesh.
    :ivar cylinders: Links the URDF collides with a cylinder.
    :ivar fitted_capsule_links: Links the URDF gives no collision: a capsule fitted to the visual mesh's extent.
    :ivar filtered_pairs: Non-adjacent link pairs that overlap in the ready stance, kept from colliding with each other.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mesh_hull_links: tuple[str, ...] = ()
    cylinders: dict[str, CylinderCollider] = Field(default_factory=dict)
    fitted_capsule_links: tuple[str, ...] = ()
    filtered_pairs: tuple[tuple[str, str], ...] = ()


class RetargetSpec(BaseModel):
    """
    What the stroke retargeter needs beyond the URDF (``lib/retarget``).

    :ivar racket_arm: The paddle arm's shoulder, elbow and wrist links.
    :ivar free_arm: The other arm's shoulder, elbow and wrist links.
    :ivar urdf_drop_links: URDF links the kinematic model leaves out (the hands the paddle replaces).
    :ivar sole_corners_m: The sole's contact corners in the foot link's frame, m.
    :ivar collision_spheres: Coarse self-collision spheres per link: (centre in the link frame m, radius m).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    racket_arm: tuple[str, str, str]
    free_arm: tuple[str, str, str]
    urdf_drop_links: tuple[str, ...] = ()
    sole_corners_m: tuple[tuple[float, float, float], ...] = Field(min_length=3)
    collision_spheres: dict[str, tuple[tuple[tuple[float, float, float], float], ...]] = Field(default_factory=dict)


class RobotSpec(BaseModel):
    """
    A humanoid with a paddle, as every feature needs it.

    :ivar name: The robot's name in :data:`lib.robots.ROBOTS`.
    :ivar usd: The robot's USD, with ``{ISAAC_NUCLEUS_DIR}`` for NVIDIA's asset root.
    :ivar prune: Prims whose name contains one of these are deactivated at spawn (the hand the paddle replaces).
    :ivar urdf: The robot's URDF: per-link colliders and the retargeter's kinematics.
    :ivar urdf_sha256: Its sha256.
    :ivar joint_names: The actuated joints, in URDF order (the simulator's own order is read from the articulation).
    :ivar actuators: Every motor group; together they drive every joint once.
    :ivar action_fraction: A unit action asks for this share of a joint's peak torque through its PD: the action scale
        is ``action_fraction * peak_torque / kp`` per joint (:attr:`action_scale`).
    :ivar default_pose: Joint angles of the ready stance by joint expression, rad (others 0); the policy's action origin.
    :ivar base_height_m: Pelvis height above the floor in the ready stance, m.
    :ivar paddle: The paddle and its mount.
    :ivar links: The named links.
    :ivar fall: When the robot has fallen.
    :ivar stance_m: Where the pelvis starts, behind its end line (nearest, furthest), m.
    :ivar colliders: The URDF's colliders for links the USD leaves without one (:class:`LinkColliders`).
    :ivar retarget: What the stroke retargeter needs (``None``: the robot cannot be retargeted onto yet).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9-]+$")
    usd: str
    prune: tuple[str, ...] = ()
    urdf: Path
    urdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    joint_names: tuple[str, ...] = Field(min_length=1)
    actuators: tuple[MotorGroup, ...] = Field(min_length=1)
    action_fraction: float = Field(gt=0.0, le=1.0)
    default_pose: dict[str, float]
    base_height_m: float = Field(gt=0.0)
    paddle: PaddleMount
    links: RobotLinks
    fall: FallRule
    stance_m: tuple[float, float]
    colliders: LinkColliders = LinkColliders()
    retarget: RetargetSpec | None = None

    @model_validator(mode="after")
    def _every_joint_driven_once(self) -> RobotSpec:
        for joint in self.joint_names:
            groups = [group.name for group in self.actuators for expression in group.joints if re.fullmatch(expression, joint)]
            if len(groups) != 1:
                raise ValueError(f"{self.name}: joint {joint} is driven by {len(groups)} motor groups {groups}, not one")
        for expression in self.default_pose:
            if not any(re.fullmatch(expression, joint) for joint in self.joint_names):
                raise ValueError(f"{self.name}: default pose expression {expression!r} names no joint")
        if not 0.0 <= self.stance_m[0] < self.stance_m[1]:
            raise ValueError(f"{self.name}: stance_m must be (nearest, furthest) behind the end line")
        return self

    @property
    def num_joints(self) -> int:
        """Actuated joints."""

        return len(self.joint_names)

    @property
    def action_scale(self) -> dict[str, float]:
        """Per joint expression, the joint angle a unit action adds to the default pose, rad."""

        return {expression: self.action_fraction * group.peak_torque / group.kp[expression] for group in self.actuators for expression in group.joints}

    def motor(self, joint: str) -> MotorGroup:
        """
        The motor group driving a joint.

        :param joint: A joint name.
        :raises KeyError: If no group drives it.
        """

        for group in self.actuators:
            if any(re.fullmatch(expression, joint) for expression in group.joints):
                return group
        raise KeyError(f"{self.name}: no motor group drives joint {joint!r}")


__all__ = ["CylinderCollider", "FallRule", "LinkColliders", "MotorGroup", "PaddleMount", "RetargetSpec", "RobotLinks", "RobotSpec"]
