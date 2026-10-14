"""
The robot's paddle and the calibrated GPU ball, coupled both ways, once per PhysX step.

:class:`RobotBallCoupling` owns the ball physics (:class:`lib.physics.vectorized.build.BallPhysics`: the
graph-captured :class:`lib.physics.vectorized.dense.DenseBallBatch`, the bounce predictor and the inverse law) for
every environment of an Isaac Lab scene, and reads the robot's articulation directly. Call :meth:`step` first thing in
``DirectRLEnv._apply_action``, which runs once per physics step; the state there is the end of the step that just
ended, ``[t - h, t]``. For that step it:

1. **reads the paddle** (body ``paddle``; its origin is the blade centre; frame x = tip, y = forehand face normal,
   z = across): the blade centre's position and orientation, in the table frame (``x_table = x_world -
   env_origin - table_origin``) and the law's blade axes (X across, Y tip, Z forehand: ``paddle_to_game``), and its
   linear velocity only for ``track="hermite"``. The angular velocity is never read (the slerp needs only the two
   orientations); a velocity read costs about as much as the ball (about 10 ms per physics step at 4,096
   environments), so ``track="line"`` reads no velocity at all;
2. **builds the blade's motion** over the step from its two end states: a cubic Hermite through the positions and
   velocities with slerp at constant angular velocity (``track="hermite"``), or PhysX's own straight line at
   ``(p1 - p0) / h`` (``"line"``); a blade whose env was reset this step is parked at its new pose
   (``p0 = p1``, ``v = 0``), and the opponent's blade likewise when it teleports;
3. **the contact law's paddle**: with ``compliance="articulated"``, ``J M^-1 J^T`` of the paddle link about the blade
   centre, with every actuator's armature added to PhysX's mass matrix (which leaves it out: 0.165 kg instead of the
   measured 0.222 kg at the blade centre); the law then takes ``K(r)`` at each contact's offset ``r``. ``J`` is
   PhysX's (``jacobian="physx"``: a per-step refresh, about 21 ms at 8,192 environments) or :class:`PaddleJacobian`'s,
   in float64 from the root's orientation and the joint positions (``jacobian="torch"``, checked against PhysX's in
   every environment after the first step). With ``"law"`` the contract's paddle mass and free-paddle inertia are
   used;
4. **flies the ball** over the step in sub-steps of the batch's ``dt`` (``h / dt`` of them, ``dt`` passed
   explicitly), each sub-step's blade motion the exact restriction of the step's;
5. **slices that step's contacts** from the batch's on-device log (emptied every step, so it never grows):
   :class:`StepContacts`, per environment and slot;
6. **pushes back on the paddle**: ``F = -sum J / h`` at the paddle's centre of mass, with the torque
   ``sum (r_i - com) x F_i`` about it (no ``-L/h`` term), both computed in the environment's frame in float64 (the
   composer's own lever, a float32 world position minus the centre of mass, loses about 3e-5 m at 360 m from the world
   origin), on the paddle body only. :attr:`RobotBallCoupling.reaction` holds the force, that torque and the centre of
   mass (world); the robot feels a contact one step later. Any other paddle wrench (a test wrench) goes through
   :meth:`RobotBallCoupling.add_external` and joins the reaction. Two routes (:attr:`CouplingConfig.wrench`):

   - ``"composer"`` (the default and the reference): both are added to Lab's instantaneous wrench composer
     (``add_forces_and_torques_index(..., positions=None, is_global=True)``; Lab clears it after every write, so the
     order of the additions within a step does not matter). Lab's ``write_data_to_sim`` composes it, which reads the
     links' centre of mass and so pays PhysX's per-step refresh (about 21 ms at 8,192 environments) whatever the
     coupling read;
   - ``"direct"``: one write per physics step, for every environment (zeros where nothing pushed), straight into
     ``robot.root_view.apply_forces_and_torques_at_position(..., position_data=None, is_global=True)`` (a global force
     at the link's centre of mass and a global torque), with no refresh. PhysX applies only the last force write before
     a physics step and clears it after the step (measured), so the composer must stay unused: the
     write is deferred into the articulation's ``write_data_to_sim`` (after Lab's own, which then writes no force, and
     after every :meth:`~RobotBallCoupling.add_external` of the step), and a step on which the composer was written
     raises.

Negative controls and A/Bs run through the same path (:class:`CouplingConfig`): ``track="line"``,
``blade_lag_steps > 0`` (the ball sees the blade where it was), a batch built with ``dt = h`` (no sub-steps),
``angular_torque_term=True`` (a ``-L/h`` torque added) and ``wrench="direct_unguarded"`` (the reaction written to
the view at once, which a later composer write overwrites).

Isaac Lab and torch are imported only when used, so this module imports anywhere.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from lib.physics.params import BallTableParams
from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.build import BallPhysics, Precision
from lib.physics.vectorized.dense import blade_motion
from lib.physics.vectorized.geometry import REGION, STATIC_COUNT, TABLE_TOP, blade_state, quat_axes
from lib.physics.vectorized.graphs import compiler
from lib.physics.vectorized.impact import _skew, inv3, mat_mul, mat_vec

Tensor = Any

PADDLE_TO_GAME = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
"""The law's blade axes in the paddle frame, as columns: game X (across) = paddle z, game Y (tip) = paddle x, game Z
(forehand face) = paddle y. ``R_game = R_paddle @ PADDLE_TO_GAME``."""

BLADE_EDGE_REGION = REGION["blade_edge"]
"""The log's region code of a touch on a blade's rounded edge."""

PARKED_OPPONENT = (0.0, 0.0, -10.0)
"""Where an absent opponent's blade waits (table frame, m): far below the floor."""


@dataclass(frozen=True)
class CouplingConfig:
    """
    How the coupling is built (the A/Bs and negative controls of design sections 8 and 11 included).

    :ivar paddle_body: The paddle's body name.
    :ivar table_origin: The table-top centre in the env frame, m (the demo env: the floor below it is the env origin).
    :ivar paddle_to_game: :data:`PADDLE_TO_GAME`.
    :ivar precision: The ball physics' dtype.
    :ivar substep: The ball's sub-step, s (``physics_dt / substep`` must be whole; ``substep = physics_dt`` is the
        no-sub-steps negative control).
    :ivar opponent: A second, kinematic blade (:meth:`RobotBallCoupling.set_opponent`).
    :ivar compliance: ``articulated`` (``J M^-1 J^T`` with armature, per contact point) or ``law`` (the contract's
        paddle mass and free-paddle inertia).
    :ivar track: ``hermite`` or ``line`` (PhysX's straight line at ``(p1 - p0) / h``).
    :ivar blade_lag_steps: The ball sees the robot's blade this many physics steps late (0: on time; a negative
        control otherwise).
    :ivar angular_torque_term: Add ``-sum dL_spin / h`` to the reaction torque (a negative control).
    :ivar timing: Synchronise CUDA around :meth:`RobotBallCoupling.step` to time it (``wall_s``).
    :ivar log_capacity: Contacts per environment per physics step the slice holds.
    :ivar jacobian: Where the articulated compliance's paddle Jacobian comes from: ``physx`` (Lab's
        ``body_link_jacobian_w``: a per-step PhysX refresh, about 21 ms at 8,192 environments) or ``torch``
        (:class:`PaddleJacobian` from the root's orientation and the joint positions; needs ``chain``).
    :ivar base_point: The root point whose velocity a floating base's base columns carry: ``com`` (PhysX) or ``link``.
    :ivar verify_jacobian: Check the torch Jacobian against PhysX's in every environment after the first physics step
        (one Jacobian fetch) and raise on a mismatch beyond ``2e-5 max(1, |origin| / 16 m)``: PhysX's own is float32
        in world coordinates.
    :ivar wrench: How the reaction (and :meth:`RobotBallCoupling.add_external`'s wrenches) reach PhysX: ``composer``
        (Lab's instantaneous wrench composer, the reference), ``direct`` (one write per physics step to the
        articulation's PhysX view; see the module notes) or ``direct_unguarded`` (a negative control: the reaction
        written to the view at once in :meth:`RobotBallCoupling.step`, lost when a composer write follows).
    :ivar blade_edge: Blades with their rounded edge (the oracle's; an edge strike is resolved with ``rim_contact``, an
        unvalidated prior, and counted in ``counters.edge_contacts``); False is the face-only blade, through whose edge
        a ball passes (an A/B and negative control).
    :ivar warm_up: Compile and capture every phase when the coupling is built (:meth:`RobotBallCoupling.warm_up`), so
        torch.compile's first-use cost (about 240 s on CUDA for the ball's phases) lands in the build, not in the first
        steps that meet a ball. It changes no result. False defers each phase to its first use.
    :ivar compile_cache: Before the warm-up compiles, ``restore`` unpacks the published torch.compile cache for this
        compiler and GPU (:mod:`lib.physics.vectorized.compile_cache`, an Antioch asset), so the phases load instead of
        compiling; ``restore+publish`` also publishes the cache after a warm-up that compiled (a seeding run). ``off``
        (the default) touches no asset. A cache changes no result.
    """

    paddle_body: str = "paddle"
    table_origin: tuple[float, float, float] = (0.0, 0.0, 0.76)
    paddle_to_game: tuple[tuple[float, float, float], ...] = PADDLE_TO_GAME
    precision: Precision = "float64"
    substep: float = 1e-3
    opponent: bool = True
    compliance: Literal["articulated", "law"] = "articulated"
    track: Literal["hermite", "line"] = "hermite"
    blade_lag_steps: int = 0
    angular_torque_term: bool = False
    timing: bool = True
    log_capacity: int = 16
    jacobian: Literal["physx", "torch"] = "physx"
    base_point: Literal["com", "link"] = "com"
    verify_jacobian: bool = True
    wrench: Literal["composer", "direct", "direct_unguarded"] = "composer"
    blade_edge: bool = True
    warm_up: bool = True
    compile_cache: Literal["off", "restore", "restore+publish"] = "off"


@dataclass
class StepContacts:
    """
    One physics step's contacts, per environment ``(N,)`` and log slot ``(N, L)`` (slots ``>= count`` are stale), in the
    table frame. ``blade`` is the blade index (robot blades first, then the opponent) or ``-1`` for a static surface.

    The log fields are views of the batch's buffers, valid until the next :meth:`RobotBallCoupling.step`.
    ``impulse`` is ``J = m_ball (v_after - v_before)``; ``offset`` the contact point from the blade centre; and
    ``paddle_mass`` ``1 / (n^T K(r) n)``, the surface's effective mass along the normal at the contact (blades only;
    ``nan`` otherwise). The blade's state the law resolved each blade contact with, at the contact time: ``blade_p``
    (centre), ``blade_v``, ``blade_q`` (``(w, x, y, z)``, game axes), ``blade_w`` (angular velocity), and the
    ``compliance`` ``K (3, 3)`` it used (zero for static surfaces). Per environment: ``robot_hit``, ``opponent_hit``, and ``table_side``, the sign of x of the
    step's first table-top contact (0: none), and ``robot_edge_hit``: a robot-blade contact on the blade's rounded edge
    this step (region ``blade_edge``, resolved with ``rim_contact``: the QA count of edge strikes,
    :attr:`RobotBallCoupling.counters` ``edge_contacts``).
    """

    count: Tensor
    time: Tensor
    surface: Tensor
    region: Tensor
    point: Tensor
    normal: Tensor
    before: Tensor
    after: Tensor
    blade: Tensor
    impulse: Tensor
    offset: Tensor
    paddle_mass: Tensor
    blade_p: Tensor
    blade_v: Tensor
    blade_q: Tensor
    blade_w: Tensor
    compliance: Tensor
    robot_hit: Tensor
    opponent_hit: Tensor
    table_side: Tensor
    robot_edge_hit: Tensor

    def valid(self) -> Tensor:
        """``(N, L)``: the slots that hold this step's contacts."""

        slots = torch.arange(self.surface.shape[1], device=self.surface.device)
        return slots[None, :] < self.count[:, None]

    def flat(self) -> dict[str, Tensor]:
        """The contacts as ``(K,)`` rows (``env`` first; one host sync)."""

        env, slot = torch.nonzero(self.valid(), as_tuple=True)
        out = {"env": env}
        for name in (
            "time",
            "surface",
            "region",
            "point",
            "normal",
            "before",
            "after",
            "blade",
            "impulse",
            "offset",
            "paddle_mass",
            "blade_p",
            "blade_v",
            "blade_q",
            "blade_w",
            "compliance",
        ):
            out[name] = getattr(self, name)[env, slot]
        return out


@dataclass
class Reaction:
    """
    The robot's wrench over the coming physics step, per environment (zero where nothing touched), world axes: the force,
    the torque about ``point_w`` (the paddle's centre of mass, world) and whether a robot blade touched.
    """

    force_w: Tensor
    torque_w: Tensor
    point_w: Tensor
    touched: Tensor


@dataclass
class _Counters:
    intervals: int = 0
    substeps: int = 0
    contacts: int = 0
    skipped_intervals: int = 0
    skipped_with_contact: int = 0
    wall_s: float = 0.0
    compliance_s: float = 0.0
    direct_writes: int = 0
    edge_contacts: int = 0


def _data(value: Any) -> Any:
    """An Isaac Lab data field as a torch tensor (warp-backed fields expose ``.torch``)."""

    return value.torch if hasattr(value, "torch") else value


def _view_array(tensor: Tensor) -> Any:
    """
    A flat float32 tensor as the PhysX tensor view takes it: a warp array (the tensor itself where warp is absent).
    ``wp.init()`` is idempotent; Kit has called it already, a bare interpreter with warp installed (an engine image's
    pytest, a tool) has not, and ``from_torch`` needs it.
    """

    try:
        import warp as wp
    except ImportError:
        return tensor
    wp.init()
    return wp.from_torch(tensor.contiguous())


def quat_xyzw_to_matrix(q: Tensor) -> Tensor:
    """Rotation matrices ``(..., 3, 3)`` of ``(x, y, z, w)`` quaternions (Isaac Lab's order)."""

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


def matrix_to_quat_wxyz(r: Tensor) -> Tensor:
    """``(w, x, y, z)`` quaternions of rotation matrices (Shepperd's method; ``w >= 0``)."""

    m00, m01, m02 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    m10, m11, m12 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    m20, m21, m22 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]
    t0, t1, t2, t3 = 1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22
    qa = torch.stack([t0, m21 - m12, m02 - m20, m10 - m01], -1)
    qb = torch.stack([m21 - m12, t1, m01 + m10, m02 + m20], -1)
    qc = torch.stack([m02 - m20, m01 + m10, t2, m12 + m21], -1)
    qd = torch.stack([m10 - m01, m02 + m20, m12 + m21, t3], -1)
    best = torch.stack([t0, t1, t2, t3], -1).argmax(-1)
    q = torch.where((best == 0)[..., None], qa, torch.where((best == 1)[..., None], qb, torch.where((best == 2)[..., None], qc, qd)))
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True))
    return torch.where(q[..., :1] < 0, -q, q)


def compliance_blocks(jacobian: Tensor, mass_matrix: Tensor, armature: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    ``A, B, C``: the blocks of ``J M^-1 J^T`` (``A`` translational, ``B`` coupling, ``C`` rotational), with the
    armature added to the mass matrix's diagonal.

    :param jacobian: The link's Jacobian, ``(N, 6, D)``: linear rows (its origin's velocity), then angular, world axes.
    :param mass_matrix: PhysX's generalised mass matrix ``(N, D, D)``.
    :param armature: Per generalised coordinate ``(N, D)`` (zero for the base's).
    """

    m = mass_matrix + torch.diag_embed(armature)
    lower, _info = torch.linalg.cholesky_ex(m)
    solved = torch.cholesky_solve(jacobian.transpose(-1, -2), lower)
    g = jacobian @ solved
    return g[:, :3, :3], g[:, :3, 3:], g[:, 3:, 3:]


def contact_compliance(a: Tensor, b: Tensor, c: Tensor, r: Tensor) -> Tensor:
    """``K(r) = A + B [r]x - [r]x B^T - [r]x C [r]x``: the compliance at offset ``r`` from the blocks' point."""

    skew = _skew(r)
    return a + mat_mul(b, skew) - mat_mul(skew, b.transpose(-1, -2)) - mat_mul(mat_mul(skew, c), skew)


def _wxyz_to_matrix(q: Sequence[float]) -> Any:
    w, x, y, z = (float(v) for v in q)
    return quat_xyzw_to_matrix(torch.tensor([x, y, z, w], dtype=torch.float64))


class PaddleJacobian:
    """
    The paddle link origin's (the blade centre's) world-frame geometric Jacobian in torch, from the root link's
    orientation and the joint positions: PhysX's layout ``(N, 6, base + J)`` (linear rows, then angular; the six base
    columns first on a floating base: the root's linear velocity at ``base_point``, then its angular velocity), so
    ``J M^-1 J^T`` with PhysX's mass matrix is the paddle's compliance. Only orientations and positions relative to the
    root enter, so the result does not depend on the environment's world offset, and nothing is read from PhysX but
    the root's orientation and the joint positions (a Jacobian or velocity fetch costs a per-step refresh: about
    21 ms at 8,192 environments).

    :param chain: The joints from the root link to the paddle, in that order: objects with ``name``, ``parent_body``,
        ``child_body``, ``type``, ``local_pos0``, ``local_rot0``, ``local_pos1``, ``local_rot1`` and ``axis`` (the
        authored USD ``physics:*`` joint attributes, read from the live stage).
    :param joint_names: The articulation's joint names (Lab's order: the Jacobian's joint columns).
    :param floating: A floating base (six base columns first).
    :param base_point: The root point whose linear velocity the base columns carry, in the root link's frame (PhysX:
        the root's centre of mass).
    """

    def __init__(
        self, chain: Sequence[Any], joint_names: Sequence[str], *, floating: bool, base_point: Sequence[float] = (0.0, 0.0, 0.0), device: Any = "cpu"
    ) -> None:
        f64 = torch.float64
        names = list(joint_names)
        self.columns = (6 if floating else 0) + len(names)
        self.floating = floating
        axes = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
        steps = []
        for joint in chain:
            if joint.type not in ("revolute", "fixed"):
                raise ValueError(f"joint {joint.name!r}: only revolute and fixed joints are supported, not {joint.type!r}")
            rot0 = _wxyz_to_matrix(joint.local_rot0)
            rot1 = _wxyz_to_matrix(joint.local_rot1)
            column = -1
            if joint.type == "revolute":
                if joint.name is None or joint.name not in names:
                    raise ValueError(f"joint {joint.name!r} is not one of the articulation's joints")
                column = (6 if floating else 0) + names.index(joint.name)
            steps.append(
                (
                    torch.tensor(joint.local_pos0, dtype=f64, device=device),
                    rot0.to(device),
                    rot1.transpose(-1, -2).to(device),
                    -(rot1.transpose(-1, -2) @ torch.tensor(joint.local_pos1, dtype=f64)).to(device),
                    torch.tensor(axes[joint.axis], dtype=f64, device=device),
                    "XYZ".index(joint.axis),
                    column,
                    names.index(joint.name) if joint.type == "revolute" and joint.name is not None else -1,
                )
            )
        self.steps = steps
        self.base_point = torch.tensor(base_point, dtype=f64, device=device)

    @staticmethod
    def _rotation(axis: int, angle: Tensor) -> Tensor:
        c, s = torch.cos(angle), torch.sin(angle)
        one, zero = torch.ones_like(angle), torch.zeros_like(angle)
        if axis == 0:
            rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
        elif axis == 1:
            rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
        else:
            rows = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
        return torch.stack([torch.stack(row, -1) for row in rows], -2)

    def forward(self, root_quat_xyzw: Tensor, joint_pos: Tensor) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor, int]]]:
        """
        The paddle frame relative to the root link's origin (world axes) and each revolute joint's anchor, world axis
        and column: ``(position (N, 3), rotation (N, 3, 3), [(anchor, axis, column), ...])``, float64.
        """

        f64 = torch.float64
        n = joint_pos.shape[0]
        rotation = quat_xyzw_to_matrix(root_quat_xyzw.to(f64))
        position = torch.zeros(n, 3, dtype=f64, device=joint_pos.device)
        q = joint_pos.to(f64)
        joints: list[tuple[Tensor, Tensor, int]] = []
        for pos0, rot0, rot1_t, pos1_back, axis, axis_index, column, joint in self.steps:
            frame_pos = position + (rotation @ pos0[:, None])[..., 0]
            frame_rot = rotation @ rot0
            if column >= 0:
                joints.append((frame_pos, (frame_rot @ axis[:, None])[..., 0], column))
                frame_rot = frame_rot @ self._rotation(axis_index, q[:, joint])
            rotation = frame_rot @ rot1_t
            position = frame_pos + (frame_rot @ pos1_back[:, None])[..., 0]
        return position, rotation, joints

    def __call__(self, root_quat_xyzw: Tensor, joint_pos: Tensor) -> Tensor:
        """
        :param root_quat_xyzw: The root link's world orientation, ``(N, 4)`` (Lab's ``(x, y, z, w)``).
        :param joint_pos: Joint positions, ``(N, J)`` (Lab's order).
        :return: ``(N, 6, base + J)``, float64.
        """

        f64 = torch.float64
        position, _rotation, joints = self.forward(root_quat_xyzw, joint_pos)
        n, device = joint_pos.shape[0], joint_pos.device
        jacobian = torch.zeros(n, 6, self.columns, dtype=f64, device=device)
        for anchor, axis, column in joints:
            jacobian[:, :3, column] = torch.cross(axis, position - anchor, dim=-1)
            jacobian[:, 3:, column] = axis
        if self.floating:
            root_rotation = quat_xyzw_to_matrix(root_quat_xyzw.to(f64))
            lever = position - (root_rotation @ self.base_point[:, None])[..., 0]
            jacobian[:, :3, :3] = torch.eye(3, dtype=f64, device=device)
            jacobian[:, :3, 3:6] = -_skew(lever)
            jacobian[:, 3:, 3:6] = torch.eye(3, dtype=f64, device=device)
        return jacobian


def sample_track_fn(p0: Tensor, v0: Tensor, q0: Tensor, p1: Tensor, v1: Tensor, q1: Tensor, h: float, parts: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    The blades' states at the ``parts + 1`` sub-step boundaries of their motion over ``h`` (Hermite and slerp):
    ``(P, V, Q)``, ``(parts + 1, N, B, 3 / 3 / 4)``, and the angular velocity at the step's end, ``(N, B, 3)``. Each
    sub-step's motion rebuilt from two of them is the exact restriction of the whole step's (:meth:`BladeTrack.split`).
    """

    n, b = p0.shape[0], p0.shape[1]
    tensors = blade_motion(p0, v0, q0, p1, v1, q1, h)
    env = torch.arange(n, device=p0.device).repeat_interleave(b)
    blade = torch.arange(b, device=p0.device).repeat(n)
    positions, velocities, orientations = [], [], []
    omega = p0
    for k in range(parts + 1):
        u = torch.full((n * b,), k / parts, dtype=p0.dtype, device=p0.device)
        position, orientation, velocity, omega = blade_state(tensors, h, u, env, blade)
        positions.append(position.reshape(n, b, 3))
        velocities.append(velocity.reshape(n, b, 3))
        orientations.append(orientation.reshape(n, b, 4))
    return torch.stack(positions), torch.stack(velocities), torch.stack(orientations), omega.reshape(n, b, 3)


def collect_fn(
    count: Tensor,
    time_: Tensor,
    surface: Tensor,
    point: Tensor,
    normal: Tensor,
    before: Tensor,
    after: Tensor,
    sub_states: tuple[Tensor, Tensor, Tensor],
    t_start: Tensor,
    dt: float,
    h: float,
    ball_mass: Tensor,
    ball_inertia: Tensor,
    blocks: tuple[Tensor, Tensor, Tensor, Tensor],
    paddle: tuple[Tensor, Tensor, Tensor, Tensor],
    robot_blades: int,
    angular_torque_term: bool,
) -> tuple[Tensor, ...]:
    """
    One step's contacts from the log slice: per slot the blade, ``J``, the offset from the blade centre and the
    effective mass along the normal; per environment the robot blades' ``sum J``, ``sum r_c x J`` and the reaction
    (force, torque about the first contact's point, that point), and the hit and bounce flags.

    :param sub_states: The blades' states at the sub-step boundaries (:func:`sample_track_fn`): the blade at a
        contact is evaluated on the sub-step motion the ball was flown with (its angular velocity is the sub-step's,
        which differs from the whole step's where the game's motion takes its normalised-lerp branch).
    :param dt: The ball's sub-step, s.
    :param blocks: Articulated compliance ``(A, B, C, on)`` per environment and blade.
    :param paddle: The law's paddle ``(mass (N,), inertia_xx, inertia_yy, inertia_zz)`` for blades without ``on``.
    """

    n, slots = surface.shape
    device = surface.device
    slot = torch.arange(slots, device=device)
    valid = slot[None, :] < count[:, None]
    blade = torch.where(valid & (surface >= STATIC_COUNT), surface - STATIC_COUNT, torch.full_like(surface, -1))
    impulse = ball_mass[:, None, None] * (after[..., 3:6] - before[..., 3:6])
    impulse = torch.where(valid[..., None], impulse, torch.zeros_like(impulse))
    # The blade at the contact time: its centre and axes.
    env = torch.arange(n, device=device).repeat_interleave(slots)
    which = blade.clamp(min=0).reshape(-1)
    parts = sub_states[0].shape[0] - 1
    position = ((time_ - t_start) / dt).reshape(-1)
    k = torch.floor(position).clamp(0, parts - 1)
    u = (position - k).to(point.dtype)
    p_sub, v_sub, q_sub = sub_states
    tensors = blade_motion(p_sub[:-1], v_sub[:-1], q_sub[:-1], p_sub[1:], v_sub[1:], q_sub[1:], dt)
    flat = tuple(t.reshape(parts * n, *t.shape[2:]) for t in tensors)
    centre, orientation, blade_velocity, blade_omega = blade_state(flat, dt, u, k.long() * n + env, which)
    offset = point - centre.reshape(n, slots, 3)
    r = offset.reshape(-1, 3)
    a, b, c, on = blocks
    k_arm = contact_compliance(a[env, which], b[env, which], c[env, which], r)
    mass, ixx, iyy, izz = paddle
    axes = quat_axes(orientation)
    inertia = ((axes * torch.stack([iyy[env], ixx[env], izz[env]], -1)[:, None, :])[:, :, :, None] * axes.transpose(-1, -2)[:, None, :, :]).sum(-2)
    eye = torch.eye(3, dtype=point.dtype, device=device).expand(n * slots, 3, 3)
    skew = _skew(r)
    k_law = eye / mass[env][:, None, None] + mat_mul(mat_mul(skew.transpose(-1, -2), inv3(inertia)), skew)
    k = torch.where(on[env, which][:, None, None], k_arm, k_law)
    n_flat = normal.reshape(-1, 3)
    paddle_mass = (1.0 / (n_flat * mat_vec(k, n_flat)).sum(-1)).reshape(n, slots)
    paddle_mass = torch.where(blade >= 0, paddle_mass, torch.full_like(paddle_mass, math.nan))
    offset = torch.where((blade >= 0)[..., None], offset, torch.zeros_like(offset))
    # The robot blades' push on the ball, and the equal and opposite wrench.
    robot = (blade >= 0) & (blade < robot_blades)
    j_robot = torch.where(robot[..., None], impulse, torch.zeros_like(impulse))
    total = j_robot.sum(1)
    moment = torch.cross(point, j_robot, dim=-1).sum(1)
    first = torch.where(robot, slot[None, :].expand(n, slots), torch.full_like(surface, slots)).min(1).values
    touched = first < slots
    ref = point[torch.arange(n, device=device), first.clamp(max=slots - 1)]
    force = -total / h
    torque = torch.cross(point - ref[:, None, :], -j_robot / h, dim=-1).sum(1)
    if angular_torque_term:
        spin_change = ball_inertia[:, None, None] * (after[..., 6:9] - before[..., 6:9])
        torque = torque - torch.where(robot[..., None], spin_change, torch.zeros_like(spin_change)).sum(1) / h
    ref = torch.where(touched[:, None], ref, torch.zeros_like(ref))
    opponent_hit = (blade >= robot_blades).any(1)
    tops = valid & (surface == TABLE_TOP)
    first_top = torch.where(tops, slot[None, :].expand(n, slots), torch.full_like(surface, slots)).min(1).values
    top_x = point[torch.arange(n, device=device), first_top.clamp(max=slots - 1), 0]
    table_side = torch.where(first_top < slots, torch.sign(top_x), torch.zeros_like(top_x)).to(torch.long)
    on_blade = (blade >= 0)[..., None]
    blade_p = torch.where(on_blade, centre.reshape(n, slots, 3), torch.zeros_like(offset))
    blade_v = torch.where(on_blade, blade_velocity.reshape(n, slots, 3), torch.zeros_like(offset))
    blade_q = torch.where(on_blade, orientation.reshape(n, slots, 4), torch.zeros_like(orientation.reshape(n, slots, 4)))
    blade_w = torch.where(on_blade, blade_omega.reshape(n, slots, 3), torch.zeros_like(offset))
    k_used = torch.where(on_blade[..., None], k.reshape(n, slots, 3, 3), torch.zeros_like(k.reshape(n, slots, 3, 3)))
    return blade, impulse, offset, paddle_mass, total, moment, force, torque, ref, touched, opponent_hit, table_side, blade_p, blade_v, blade_q, blade_w, k_used


def prepare_fn(
    raw: tuple[Tensor, Tensor, Tensor],
    frame: tuple[Tensor, Tensor, Tensor],
    history: tuple[Tensor, Tensor, Tensor],
    skip: Tensor,
    opponent: tuple[Tensor, ...] | None,
    ball: tuple[Tensor, Tensor],
    h: float,
    parts: int,
    lag: int,
    line: bool,
    reach: float,
) -> tuple[Tensor, ...]:
    """
    The step's blade motions from the paddle link's raw state (``lin`` is not read where ``line``): the blade now
    (table frame, game axes; its velocity and angular velocity are the track's at the step's end, the values the law
    used), the history shifted by one, the robot blade's ends (parked where ``skip``, PhysX's line where ``line``,
    ``lag`` steps late), the opponent's (parked where it teleports), their states at the sub-step boundaries, and which
    environments' balls are within ``reach`` of the robot blade (they need the arm's compliance).
    """

    pos, lin, quat = raw
    origins, table_origin, paddle_to_game = frame
    p = pos - origins - table_origin
    q = matrix_to_quat_wxyz(quat_xyzw_to_matrix(quat) @ paddle_to_game)
    hist_p, hist_v, hist_q = history
    hist_p = torch.cat([hist_p[1:], p[None]])
    hist_v = torch.cat([hist_v[1:], lin[None]])
    hist_q = torch.cat([hist_q[1:], q[None]])
    p0, v0, q0 = hist_p[-lag - 2], hist_v[-lag - 2], hist_q[-lag - 2]
    p1, v1, q1 = hist_p[-lag - 1], hist_v[-lag - 1], hist_q[-lag - 1]
    p0 = torch.where(skip[:, None], p1, p0)
    q0 = torch.where(skip[:, None], q1, q0)
    if line:
        v0 = v1 = (p1 - p0) / h
    v0 = torch.where(skip[:, None], torch.zeros_like(v0), v0)
    v1 = torch.where(skip[:, None], torch.zeros_like(v1), v1)
    ends = [[p0], [v0], [q0], [p1], [v1], [q1]]
    opponent_next: tuple[Tensor, ...] = ()
    if opponent is not None:
        prev_p, prev_v, prev_q, now_p, now_v, now_q, teleport = opponent
        o_p0 = torch.where(teleport[:, None], now_p, prev_p)
        o_q0 = torch.where(teleport[:, None], now_q, prev_q)
        o_v0 = torch.where(teleport[:, None], torch.zeros_like(now_v), prev_v)
        o_v1 = torch.where(teleport[:, None], torch.zeros_like(now_v), now_v)
        for column, value in zip(ends, (o_p0, o_v0, o_q0, now_p, o_v1, now_q), strict=True):
            column.append(value)
        opponent_next = (now_p, now_v, now_q)
    s0, s1, s2, s3, s4, s5 = (torch.stack(column, 1) for column in ends)
    positions, velocities, orientations, omega_end = sample_track_fn(s0, s1, s2, s3, s4, s5, h, parts)
    state, active = ball
    gap = state[:, :3] - p
    near = active & ((gap * gap).sum(-1) < reach * reach)
    return (p, velocities[-1, :, 0], q, omega_end[:, 0], hist_p, hist_v, hist_q, positions, velocities, orientations, near, *opponent_next)


class RobotBallCoupling:
    """
    The ball physics of every environment, coupled to the robot's paddle (see the module notes for the order).

    Each :meth:`step` costs a few launches outside CUDA graphs: two or three copies of the paddle's state, the arm's
    compliance for the environments whose ball is within ``reach`` of the blade (one host sync), six copies and the
    ball's graphs per sub-step, and the composer call.

    :param robot: The Isaac Lab ``Articulation`` (G1 with the ``paddle`` body).
    :param env_origins: Environment origins in the world, ``(N, 3)`` (``scene.env_origins``).
    :param physics_dt: The robot's physics step ``h``, s.
    :param config: :class:`CouplingConfig`.
    :param params: Ball and table physics: one set for every environment or one per environment (default nominal).
    :param physics: A ball physics built already (``BallPhysics.create(..., blades=robot_blades + opponent, log=True,
        log_history=False, compliance=True)``); built from ``params`` and ``config`` when ``None``.
    :param chain: The joints from the root link to the paddle (see :class:`PaddleJacobian`), for
        ``jacobian="torch"``; checked against PhysX's Jacobian once, after the first physics step.
    :param reach: A ball closer than this to the blade centre at a step's end gets the arm's compliance computed, m
        (it covers a step's relative motion at 50 m/s, the blade's half-size and the ball's radius).
    """

    def __init__(
        self,
        robot: Any,
        env_origins: Tensor,
        physics_dt: float,
        config: CouplingConfig | None = None,
        *,
        params: BallTableParams | Sequence[BallTableParams] | None = None,
        physics: BallPhysics | None = None,
        chain: Sequence[Any] | None = None,
        reach: float = 0.5,
    ) -> None:
        from lib.physics.params import nominal_params

        self.config = config = config if config is not None else CouplingConfig()
        self.robot = robot
        self.h = float(physics_dt)
        self.n = n = int(env_origins.shape[0])
        self.robot_blades = 1
        self.reach = float(reach)
        blades = self.robot_blades + int(config.opponent)
        if physics is None:
            physics = BallPhysics.create(
                params if params is not None else nominal_params(),
                n,
                device=env_origins.device,
                precision=config.precision,
                dt=config.substep,
                blades=blades,
                log=True,
                log_history=False,
                compliance=config.compliance == "articulated",
                log_capacity=config.log_capacity,
                blade_edge=config.blade_edge,
            )
        self.physics = physics
        self.batch = physics.batch
        if self.batch.blades != blades:
            raise ValueError(f"the ball physics has {self.batch.blades} blades, the coupling needs {blades}")
        self.parts = round(self.h / self.batch.dt)
        if self.parts < 1 or abs(self.parts * self.batch.dt - self.h) > 1e-9 * self.h:
            raise ValueError(f"physics_dt {self.h} is not a whole number of ball sub-steps of {self.batch.dt}")
        if config.compliance == "articulated" and not self.batch.compliance:
            raise ValueError("compliance='articulated' needs a ball physics built with compliance=True")
        dtype, device = physics.params.dtype, physics.params.device
        self.dtype, self.device = dtype, device
        self.origins = env_origins.to(device=device, dtype=dtype)
        self.table_origin = torch.tensor(config.table_origin, dtype=dtype, device=device)
        self.paddle_to_game = torch.tensor(config.paddle_to_game, dtype=dtype, device=device)
        self.paddle_id = int(robot.find_bodies(config.paddle_body)[0][0]) if robot is not None else 0
        com_b = (0.0, 0.0, 0.0)
        if robot is not None and hasattr(robot.data, "body_com_pos_b"):
            com = _data(robot.data.body_com_pos_b)
            com_b = tuple(float(v) for v in com.reshape(com.shape[0], -1, 3)[0, self.paddle_id].tolist())
        self.com_b = torch.tensor(com_b, dtype=dtype, device=device)
        """The paddle's centre of mass in its link frame (the reaction acts there)."""
        self.kinematics: PaddleJacobian | None = None
        if config.jacobian == "torch" and config.compliance == "articulated":
            if chain is None or robot is None:
                raise ValueError("jacobian='torch' needs the robot and the joint chain from the root to the paddle")
            if chain[-1].child_body != config.paddle_body:
                raise ValueError(f"the chain ends at {chain[-1].child_body!r}, not the paddle body {config.paddle_body!r}")
            floating = not bool(getattr(robot, "is_fixed_base", False))
            base_point = (0.0, 0.0, 0.0)
            if floating and config.base_point == "com":
                com = _data(robot.data.body_com_pos_b)
                base_point = tuple(float(v) for v in com.reshape(com.shape[0], -1, 3)[0, 0].tolist())
            self.kinematics = PaddleJacobian(chain, list(robot.joint_names), floating=floating, base_point=base_point, device=device)
        self.route = "none"
        """How the last physics step's reaction reached PhysX: ``composer``, ``direct`` (the step's single write to the
        PhysX view), ``unguarded`` (the negative control) or ``none`` (nothing written yet)."""
        self._pending = False
        self._external_force = torch.zeros(n, 3, dtype=torch.float64, device=env_origins.device)
        self._external_torque = torch.zeros(n, 3, dtype=torch.float64, device=env_origins.device)
        self._external_pending = False
        if config.wrench != "composer" and robot is not None:
            if getattr(robot, "root_view", None) is None:
                raise ValueError(f"wrench={config.wrench!r} needs the articulation's PhysX view (robot.root_view)")
            bodies = int(robot.num_bodies)
            self._direct_force = torch.zeros(n, bodies, 3, dtype=torch.float32, device=env_origins.device)
            self._direct_torque = torch.zeros(n, bodies, 3, dtype=torch.float32, device=env_origins.device)
            if config.wrench == "direct":
                lab_write = robot.write_data_to_sim

                def write_data_to_sim(*args: Any, **kwargs: Any) -> Any:
                    return self._write_step(lab_write, *args, **kwargs)

                robot.write_data_to_sim = write_data_to_sim
        self.t = 0.0
        self.started = False
        self._verify_pending = self.kinematics is not None and config.verify_jacobian
        self.graphs = str(device).startswith("cuda")
        wrap = compiler(self.graphs)
        self._prepare_fn = wrap(prepare_fn)
        self._collect_fn = wrap(collect_fn)

        def zeros(*shape: int, kind: Any = None) -> Tensor:
            return torch.zeros(shape, dtype=kind if kind is not None else dtype, device=device)

        depth = config.blade_lag_steps + 2
        level = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
        parked = torch.tensor(PARKED_OPPONENT, dtype=dtype, device=device)
        slots = config.log_capacity
        self._s: dict[str, Tensor] = {
            "raw_pos": zeros(n, 3),
            "raw_lin": zeros(n, 3),
            "raw_quat": zeros(n, 4),
            "hist_p": zeros(depth, n, 3),
            "hist_v": zeros(depth, n, 3),
            "hist_q": zeros(depth, n, 4) + level,
            "now_p": zeros(n, 3),
            "now_v": zeros(n, 3),
            "now_q": zeros(n, 4) + level,
            "now_w": zeros(n, 3),
            "skip": zeros(n, kind=torch.bool),
            "skipped": zeros(n, kind=torch.bool),
            "near": zeros(n, kind=torch.bool),
            "sub_p": zeros(self.parts + 1, n, blades, 3),
            "sub_v": zeros(self.parts + 1, n, blades, 3),
            "sub_q": zeros(self.parts + 1, n, blades, 4) + level,
            "opp_prev_p": zeros(n, 3) + parked,
            "opp_prev_v": zeros(n, 3),
            "opp_prev_q": zeros(n, 4) + level,
            "opp_now_p": zeros(n, 3) + parked,
            "opp_now_v": zeros(n, 3),
            "opp_now_q": zeros(n, 4) + level,
            "opp_teleport": zeros(n, kind=torch.bool),
            "t_start": torch.zeros((), dtype=torch.float64, device=device),
            "blade": zeros(n, slots, kind=torch.long),
            "impulse": zeros(n, slots, 3),
            "offset": zeros(n, slots, 3),
            "paddle_mass": zeros(n, slots),
            "impulse_sum": zeros(n, 3),
            "moment": zeros(n, 3),
            "force": zeros(n, 3),
            "torque": zeros(n, 3),
            "point": zeros(n, 3),
            "point_w": zeros(n, 3),
            "touched": zeros(n, kind=torch.bool),
            "opponent_hit": zeros(n, kind=torch.bool),
            "table_side": zeros(n, kind=torch.long),
            "blade_p": zeros(n, slots, 3),
            "blade_v": zeros(n, slots, 3),
            "blade_q": zeros(n, slots, 4),
            "blade_w": zeros(n, slots, 3),
            "compliance": zeros(n, slots, 3, 3),
            "count_contacts": zeros(kind=torch.long),
            "count_edge": zeros(kind=torch.long),
            "edge_hit": zeros(n, kind=torch.bool),
            "count_skipped": zeros(kind=torch.long),
            "count_skipped_hit": zeros(kind=torch.long),
        }
        self._blocks = (zeros(n, blades, 3, 3), zeros(n, blades, 3, 3), zeros(n, blades, 3, 3), zeros(n, blades, kind=torch.bool))
        self._graphs: dict[str, Any] = {}
        self._counters = _Counters()
        s_ = self._s
        view = self.batch.log_view()
        self.contacts = StepContacts(
            count=view["count"],
            time=view["time"],
            surface=view["surface"],
            region=view["region"],
            point=view["point"],
            normal=view["normal"],
            before=view["before"],
            after=view["after"],
            blade=s_["blade"],
            impulse=s_["impulse"],
            offset=s_["offset"],
            paddle_mass=s_["paddle_mass"],
            blade_p=s_["blade_p"],
            blade_v=s_["blade_v"],
            blade_q=s_["blade_q"],
            blade_w=s_["blade_w"],
            compliance=s_["compliance"],
            robot_hit=s_["touched"],
            opponent_hit=s_["opponent_hit"],
            table_side=s_["table_side"],
            robot_edge_hit=s_["edge_hit"],
        )
        """This step's contacts (views of fixed buffers: read them before the next :meth:`step`)."""
        self.reaction = Reaction(s_["force"], s_["torque"], s_["point_w"], s_["touched"])
        """The wrench applied to the paddle over the coming physics step (fixed buffers)."""
        self.impulse = s_["impulse_sum"]
        """``sum J`` of the robot blades' contacts in the last step, ``(N, 3)``."""
        self.moment = s_["moment"]
        """``sum r_c x J`` about the table origin, ``(N, 3)``."""
        self.osi = torch.full((n,), math.nan, dtype=dtype, device=device)
        """The robot blade's effective mass at its centre along the face normal, for environments whose ball was within
        reach at their last step (articulated), kg."""
        self.blade = (s_["now_p"], s_["now_v"], s_["now_q"], s_["now_w"])
        """The robot blade at the last :meth:`step` (no PhysX read): centre, velocity, ``(w, x, y, z)`` game axes,
        angular velocity, table frame; the velocities are its track's at the step's end, the ones the law used (the
        chord's under ``track="line"``; zero on a parked step)."""
        self.warm_up_s: dict[str, Any] = {}
        """:meth:`warm_up`'s report (empty until it runs)."""
        if config.warm_up:
            self.warm_up()

    def warm_up(self) -> dict[str, Any]:
        """
        Compile and capture every phase now: the coupling's ``prepare`` and ``collect``, and each of the ball batch's
        phases and pair tiers (:meth:`lib.physics.vectorized.dense.DenseBallBatch.warm_up`).

        Without this, a phase compiles on its first use, which on CUDA means about 240 s inside the first rally pass's
        steps (208 ms per physics step there against 15 ms afterwards).

        Each phase runs once out of turn on whatever the buffers hold, and every buffer is put back after it, so the
        coupling's state and every later result are unchanged. It is idempotent: a captured phase is not redone. It
        reads nothing from PhysX, so it may run before :meth:`start`. Without CUDA it does nothing.

        :return: ``{"wall_s": ..., "ball_phases_s": DenseBallBatch.setup_s}`` (and the cache's restore and publication
            under :attr:`CouplingConfig.compile_cache`), also kept in :attr:`warm_up_s`.
        """

        if not self.graphs:
            return {}
        from lib.physics.vectorized import compile_cache

        report: dict[str, Any] = {}
        mode = self.config.compile_cache
        if mode != "off" and not self._graphs and not self.batch.setup_s:
            report["cache_restore"] = compile_cache.restore()
        torch.cuda.synchronize()
        began = time.perf_counter()
        saved = {key: value.clone() for key, value in self._s.items()}
        self._phase("prepare", self._prepare)
        ball = self.batch.warm_up()
        self._phase("collect", self._collect)
        for key, value in saved.items():
            self._s[key].copy_(value)
        torch.cuda.synchronize()
        report.update(wall_s=round(time.perf_counter() - began, 3), ball_phases_s=ball)
        if mode == "restore+publish" and report["wall_s"] > 30.0:
            # Seconds of compiling, not loading: publish what this run compiled for the next ones.
            report["cache_publish"] = compile_cache.publish(time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}")
        self.warm_up_s = report
        return report

    # ---- graphs over the fixed buffers --------------------------------------------------------------------------------

    def _phase(self, name: str, body: Any) -> None:
        """Run ``body`` (which reads and writes only fixed buffers), captured as a CUDA graph after warming up."""

        if not self.graphs:
            body()
            return
        graph = self._graphs.get(name)
        if graph is None:
            saved = {key: value.clone() for key, value in self._s.items()}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    body()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            for key, value in saved.items():
                self._s[key].copy_(value)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()
            torch.cuda.synchronize()
            for key, value in saved.items():
                self._s[key].copy_(value)
            self._graphs[name] = graph
        graph.replay()

    def _prepare(self) -> None:
        s_ = self._s
        config = self.config
        opponent = None
        if config.opponent:
            opponent = (s_["opp_prev_p"], s_["opp_prev_v"], s_["opp_prev_q"], s_["opp_now_p"], s_["opp_now_v"], s_["opp_now_q"], s_["opp_teleport"])
        out = self._prepare_fn(
            (s_["raw_pos"], s_["raw_lin"], s_["raw_quat"]),
            (self.origins, self.table_origin, self.paddle_to_game),
            (s_["hist_p"], s_["hist_v"], s_["hist_q"]),
            s_["skip"],
            opponent,
            (self.batch.state, self.batch.active),
            self.h,
            self.parts,
            config.blade_lag_steps,
            config.track == "line",
            self.reach,
        )
        names = ["now_p", "now_v", "now_q", "now_w", "hist_p", "hist_v", "hist_q", "sub_p", "sub_v", "sub_q", "near"]
        if config.opponent:
            names += ["opp_prev_p", "opp_prev_v", "opp_prev_q"]
        for name, value in zip(names, out, strict=True):
            s_[name].copy_(value)
        s_["count_skipped"].add_(s_["skip"].sum())
        s_["skipped"].copy_(s_["skip"])
        s_["skip"].zero_()
        s_["opp_teleport"].zero_()

    def _collect(self) -> None:
        s_ = self._s
        params = self.physics.params
        view = self.batch.log_view()
        out = self._collect_fn(
            view["count"],
            view["time"],
            view["surface"],
            view["point"],
            view["normal"],
            view["before"],
            view["after"],
            (s_["sub_p"], s_["sub_v"], s_["sub_q"]),
            s_["t_start"],
            self.batch.dt,
            self.h,
            params.mass,
            params.inertia,
            self._blocks,
            (params.paddle_mass, params.paddle_inertia_xx, params.paddle_inertia_yy, params.paddle_inertia_zz),
            self.robot_blades,
            self.config.angular_torque_term,
        )
        names = (
            "blade",
            "impulse",
            "offset",
            "paddle_mass",
            "impulse_sum",
            "moment",
            "force",
            "torque",
            "point",
            "touched",
            "opponent_hit",
            "table_side",
            "blade_p",
            "blade_v",
            "blade_q",
            "blade_w",
            "compliance",
        )
        for name, value in zip(names, out, strict=True):
            s_[name].copy_(value)
        s_["point_w"].copy_(s_["point"] + self.origins + self.table_origin)
        s_["count_contacts"].add_(s_["touched"].sum())
        # The QA count of edge strikes: a robot-blade contact on the rounded edge (region blade_edge).
        robot_slot = (s_["blade"] >= 0) & (s_["blade"] < self.robot_blades)
        s_["edge_hit"].copy_((robot_slot & (view["region"] == BLADE_EDGE_REGION)).any(1))
        s_["count_edge"].add_(s_["edge_hit"].sum())
        s_["count_skipped_hit"].add_((s_["skipped"] & s_["touched"]).sum())

    # ---- inputs -------------------------------------------------------------------------------------------------------

    def _read_paddle(self) -> None:
        """
        The paddle link's state into the fixed buffers (world frame, Lab's ``(x, y, z, w)``): position and orientation,
        and the linear velocity only for the Hermite track (no angular velocity: the slerp needs none).
        """

        data = self.robot.data
        pid = self.paddle_id
        s_ = self._s
        s_["raw_pos"].copy_(_data(data.body_link_pos_w)[:, pid])
        s_["raw_quat"].copy_(_data(data.body_link_quat_w)[:, pid])
        if self.config.track == "hermite":
            s_["raw_lin"].copy_(_data(data.body_link_lin_vel_w)[:, pid])

    def blade_now(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        The robot blade's centre now, read from PhysX, table frame: position, velocity, ``(w, x, y, z)`` game-axes
        orientation, spin. Each call reads both velocities (costly every physics step: use :attr:`blade`).
        """

        data = self.robot.data
        pid = self.paddle_id
        p = _data(data.body_link_pos_w)[:, pid].to(self.dtype) - self.origins - self.table_origin
        v = _data(data.body_link_lin_vel_w)[:, pid].to(self.dtype)
        w = _data(data.body_link_ang_vel_w)[:, pid].to(self.dtype)
        rotation = quat_xyzw_to_matrix(_data(data.body_link_quat_w)[:, pid].to(self.dtype)) @ self.paddle_to_game
        return p, v, matrix_to_quat_wxyz(rotation), w

    def _update_compliance(self) -> None:
        """
        ``J M^-1 J^T`` of the paddle about the blade centre, armature included, for the environments whose ball is
        within reach (the others cannot touch the blade this step and keep their last blocks).
        """

        rows = torch.nonzero(self._s["near"]).flatten()
        if rows.numel() == 0:
            return
        robot = self.robot
        data = robot.data
        fixed = bool(getattr(robot, "is_fixed_base", False))
        if self.kinematics is not None:
            jacobian = self.kinematics(_data(data.body_link_quat_w)[rows, 0], _data(data.joint_pos)[rows])
        else:
            jacobian = _data(data.body_link_jacobian_w)[rows, self.paddle_id - 1 if fixed else self.paddle_id].to(torch.float64)
        mass_matrix = _data(data.mass_matrix)[rows].to(torch.float64)
        armature = torch.zeros(rows.numel(), mass_matrix.shape[1], dtype=torch.float64, device=mass_matrix.device)
        base = 0 if fixed else 6
        for actuator in robot.actuators.values():
            ids = actuator.joint_indices
            ids = (
                torch.arange(robot.num_joints, device=mass_matrix.device) if isinstance(ids, slice) else torch.as_tensor(ids, device=mass_matrix.device).long()
            )
            values = _data(actuator.armature).to(torch.float64)
            values = values.expand(self.n, -1)[rows] if values.shape[0] in (1, self.n) else values
            armature[:, base + ids] = values
        a, b, c = compliance_blocks(jacobian, mass_matrix, armature)
        a, b, c = (x.to(self.dtype) for x in (a, b, c))
        blocks_a, blocks_b, blocks_c, on = self._blocks
        blocks_a[rows, 0], blocks_b[rows, 0], blocks_c[rows, 0] = a, b, c
        on[rows, 0] = True
        self.batch.set_compliance(blocks_a, blocks_b, blocks_c, on)
        face = quat_axes(self._s["now_q"][rows])[:, :, 2]
        self.osi[rows] = 1.0 / (face * mat_vec(a, face)).sum(-1)

    def verify_jacobian(self, env_ids: Tensor | None = None) -> dict[str, float]:
        """
        The torch Jacobian against PhysX's (``body_link_jacobian_w``, one refresh) at the current state: the largest
        absolute difference of the linear rows (m/s per unit) and of the angular rows, and the largest entry.
        """

        if self.kinematics is None:
            raise ValueError("no torch Jacobian to verify (jacobian='physx')")
        data = self.robot.data
        rows = torch.arange(self.n, device=self.device) if env_ids is None else env_ids
        fixed = bool(getattr(self.robot, "is_fixed_base", False))
        physx = _data(data.body_link_jacobian_w)[rows, self.paddle_id - 1 if fixed else self.paddle_id].to(torch.float64)
        mine = self.kinematics(_data(data.body_link_quat_w)[rows, 0], _data(data.joint_pos)[rows])
        diff = (mine - physx).abs()
        return {
            "linear_max": float(diff[:, :3].max()),
            "angular_max": float(diff[:, 3:].max()),
            "entry_max": float(physx.abs().max()),
        }

    def _check_jacobian(self) -> None:
        """The startup check (:attr:`CouplingConfig.verify_jacobian`): raise if any environment's torch Jacobian
        differs from PhysX's by more than ``2e-5 max(1, |origin| / 16 m)``."""

        assert self.kinematics is not None
        data = self.robot.data
        fixed = bool(getattr(self.robot, "is_fixed_base", False))
        physx = _data(data.body_link_jacobian_w)[:, self.paddle_id - 1 if fixed else self.paddle_id].to(torch.float64)
        mine = self.kinematics(_data(data.body_link_quat_w)[:, 0], _data(data.joint_pos))
        error = (mine - physx).abs().amax(dim=(1, 2))
        tolerance = 2e-5 * torch.clamp(self.origins.to(torch.float64).norm(dim=-1) / 16.0, min=1.0)
        bad = error > tolerance
        if bool(bad.any()):
            worst = int(torch.argmax(error / tolerance))
            raise ValueError(
                f"the torch paddle Jacobian differs from PhysX's in {int(bad.sum())} environments (env {worst}: "
                f"{float(error[worst]):.2e} against {float(tolerance[worst]):.2e}); check the joint chain"
            )

    def set_opponent(self, p: Tensor, v: Tensor, q: Tensor, teleport: Tensor | None = None) -> None:
        """
        The opponent's blade centre now (the end of the physics step the next :meth:`step` flies), table frame,
        ``(N, 3)``, ``(N, 3)``, ``(N, 4)`` (game axes, ``(w, x, y, z)``); ``teleport`` ``(N,)`` parks it at this pose for
        that step. Call it before :meth:`step`.
        """

        s_ = self._s
        s_["opp_now_p"].copy_(p)
        s_["opp_now_v"].copy_(v)
        s_["opp_now_q"].copy_(q)
        if teleport is not None:
            s_["opp_teleport"].copy_(teleport)

    # ---- the step -----------------------------------------------------------------------------------------------------

    def start(self) -> None:
        """Take the blade's current state as its whole history (after a build, or a whole-scene reset), at rest in
        orientation (and in position too under ``track="line"``, which reads no velocity)."""

        self._read_paddle()
        s_ = self._s
        p = s_["raw_pos"] - self.origins - self.table_origin
        q = matrix_to_quat_wxyz(quat_xyzw_to_matrix(s_["raw_quat"]) @ self.paddle_to_game)
        v = s_["raw_lin"] if self.config.track == "hermite" else torch.zeros_like(p)
        s_["hist_p"][:] = p
        s_["hist_v"][:] = v
        s_["hist_q"][:] = q
        for name, value in (("now_p", p), ("now_v", v), ("now_q", q), ("now_w", torch.zeros_like(p))):
            s_[name].copy_(value)
        self.started = True

    def step(self) -> StepContacts | None:
        """
        The ball over the physics step that just ended, and the robot's reaction over the coming one (see the module
        notes). Returns that step's contacts; the first call after a build only records the blade (``None``).
        """

        if not self.started:
            self.start()
            return None
        if self._pending:
            raise RuntimeError(
                "wrench='direct': the last step's reaction was never written; the env must call the articulation's "
                "write_data_to_sim every physics step (Lab's DirectRLEnv does)"
            )
        if self._verify_pending:
            self._verify_pending = False
            self._check_jacobian()
        config = self.config
        cuda = self.graphs
        if config.timing and cuda:
            torch.cuda.synchronize()
        began = time.perf_counter()
        self._read_paddle()
        self._phase("prepare", self._prepare)
        if config.compliance == "articulated":
            compliance_began = time.perf_counter()
            self._update_compliance()
            self._counters.compliance_s += time.perf_counter() - compliance_began
        s_ = self._s
        batch = self.batch
        s_["t_start"].fill_(batch.t)
        batch.clear_log()
        for k in range(self.parts):
            batch.set_blades(s_["sub_p"][k], s_["sub_v"][k], s_["sub_q"][k], s_["sub_p"][k + 1], s_["sub_v"][k + 1], s_["sub_q"][k + 1])
            batch.step()
        self._phase("collect", self._collect)
        self._apply(self.reaction)
        self._counters.intervals += 1
        self._counters.substeps += self.parts
        self.t = batch.t
        if config.timing and cuda:
            torch.cuda.synchronize()
        self._counters.wall_s += time.perf_counter() - began
        return self.contacts

    @property
    def counters(self) -> _Counters:
        """``intervals``, ``substeps``, ``contacts`` (robot-blade contact steps), ``edge_contacts`` (those with an edge
        strike: the QA count), ``skipped_intervals``, ``skipped_with_contact``, ``wall_s``, ``compliance_s`` and
        ``direct_writes`` (reads the device)."""

        s_ = self._s
        contacts, skipped, skipped_hit, edge = torch.stack([s_["count_contacts"], s_["count_skipped"], s_["count_skipped_hit"], s_["count_edge"]]).tolist()
        counters = self._counters
        counters.contacts, counters.skipped_intervals, counters.skipped_with_contact = int(contacts), int(skipped), int(skipped_hit)
        counters.edge_contacts = int(edge)
        return counters

    def _apply(self, reaction: Reaction) -> None:
        """
        The reaction onto the paddle body only, for every environment (zero where nothing touched): the force at the
        paddle's centre of mass and the torque about it, both computed in the environment's frame in float64 (the
        composer's own lever, a world position minus the centre of mass in float32, loses about 3e-5 m at 360 m from
        the world origin).
        """

        s_ = self._s
        com = s_["now_p"] + (quat_xyzw_to_matrix(s_["raw_quat"]) @ self.com_b[:, None])[..., 0]
        reaction.torque_w.add_(torch.cross(s_["point"] - com, reaction.force_w, dim=-1))
        reaction.point_w.copy_(com + self.origins + self.table_origin)
        if self.robot is None:
            return
        wrench = self.config.wrench
        if wrench == "composer":
            if self._to_composer(reaction.force_w, reaction.torque_w):
                self.route = "composer"
        elif wrench == "direct":
            self._pending = True  # the step's single write happens in the articulation's write_data_to_sim
        else:
            self._write_view(reaction.force_w, reaction.torque_w)
            self.route = "unguarded"

    def add_external(self, force_w: Tensor, torque_w: Tensor) -> None:
        """
        An extra wrench on the paddle over the coming physics step, for every environment: a world-frame force at the
        paddle's centre of mass and a world-frame torque about it, ``(N, 3)`` each (float64; zeros where none). Call it
        any time before the step's write (in ``_apply_action``, before or after :meth:`step`); several calls add up.
        It reaches PhysX with the ball's reaction, by the same route: added to the composer, or merged into the
        bypass's single write. :attr:`reaction` stays the ball's alone.
        """

        if self.config.wrench == "composer":
            self._to_composer(force_w, torque_w)
            return
        if self.config.wrench != "direct":
            raise ValueError(f"wrench={self.config.wrench!r} (a negative control) takes no external wrench")
        self._external_force.add_(force_w.to(self._external_force.dtype))
        self._external_torque.add_(torque_w.to(self._external_torque.dtype))
        self._external_pending = True

    def _to_composer(self, force_w: Tensor, torque_w: Tensor) -> bool:
        """A paddle wrench added to the instantaneous composer (Lab clears it after every write, so additions sum)."""

        composer = getattr(self.robot, "instantaneous_wrench_composer", None)
        if composer is None:
            return False
        composer.add_forces_and_torques_index(
            forces=force_w.to(torch.float32)[:, None, :],
            torques=torque_w.to(torch.float32)[:, None, :],
            positions=None,
            body_ids=torch.tensor([self.paddle_id], dtype=torch.int32, device=self.device),
            env_ids=torch.arange(self.n, dtype=torch.int32, device=self.device),
            is_global=True,
        )
        return True

    def _write_view(self, force_w: Tensor, torque_w: Tensor) -> None:
        """One write of the paddle's wrench to ``robot.root_view`` (looked up now: a wrapper on it sees the call)."""

        self._direct_force[:, self.paddle_id] = force_w.to(torch.float32)
        self._direct_torque[:, self.paddle_id] = torque_w.to(torch.float32)
        self.robot.root_view.apply_forces_and_torques_at_position(
            force_data=_view_array(self._direct_force.reshape(-1)),
            torque_data=_view_array(self._direct_torque.reshape(-1)),
            position_data=None,
            indices=self.robot._ALL_INDICES,
            is_global=True,
        )

    def _write_step(self, lab_write: Any, *args: Any, **kwargs: Any) -> Any:
        """
        The articulation's ``write_data_to_sim`` under ``wrench="direct"``: Lab's write (joint commands; no force, as
        the composer must be unused), then the step's single force write, the reaction plus every
        :meth:`add_external`. PhysX applies only the last force write of a step, so a composer write would erase the
        reaction or be erased by it: that raises.
        """

        robot = self.robot
        used = [name for name in ("instantaneous_wrench_composer", "permanent_wrench_composer") if bool(getattr(getattr(robot, name, None), "active", False))]
        if used and (self._pending or self._external_pending):
            raise RuntimeError(
                f"wrench='direct': {', '.join(used)} was written this step, and PhysX would keep only one of its write and the "
                "coupling's; put other paddle wrenches through RobotBallCoupling.add_external"
            )
        out = lab_write(*args, **kwargs)
        if self._pending or self._external_pending:
            force = self.reaction.force_w + self._external_force if self._pending else self._external_force.clone()
            torque = self.reaction.torque_w + self._external_torque if self._pending else self._external_torque.clone()
            self._write_view(force, torque)
            self.route = "direct"
            self._counters.direct_writes += 1
        self._pending = False
        self._external_pending = False
        self._external_force.zero_()
        self._external_torque.zero_()
        return out

    # ---- resets and serves --------------------------------------------------------------------------------------------

    def reset(self, env_ids: Tensor) -> None:
        """Environments reset this step: their robot blade is parked over the step (no sweep, no wrench), ball off."""

        rows = torch.as_tensor(env_ids, device=self.device).long()
        self._s["skip"][rows] = True
        self.batch.active[rows] = False

    def launch(self, env_ids: Tensor, states: Tensor) -> None:
        """Serve or place balls: table-frame states ``(K, 9)`` (position, velocity, spin), live from now."""

        self.batch.reset(torch.as_tensor(env_ids, device=self.device).long(), states)

    def park(self, env_ids: Tensor) -> None:
        """Take some environments' balls out of play."""

        self.batch.active[torch.as_tensor(env_ids, device=self.device).long()] = False

    # ---- readouts -----------------------------------------------------------------------------------------------------

    @property
    def ball(self) -> Tensor:
        """Every ball's state, table frame, ``(N, 9)``."""

        return self.batch.state

    @property
    def active(self) -> Tensor:
        return self.batch.active

    @property
    def ended(self) -> Tensor:
        return self.batch.ended

    def table_to_world(self, points: Tensor) -> Tensor:
        """Table-frame points ``(N, 3)`` to world points."""

        return points + self.origins + self.table_origin


__all__ = [
    "BLADE_EDGE_REGION",
    "PADDLE_TO_GAME",
    "PARKED_OPPONENT",
    "CouplingConfig",
    "PaddleJacobian",
    "Reaction",
    "RobotBallCoupling",
    "StepContacts",
    "collect_fn",
    "compliance_blocks",
    "contact_compliance",
    "matrix_to_quat_wxyz",
    "prepare_fn",
    "quat_xyzw_to_matrix",
    "sample_track_fn",
]
