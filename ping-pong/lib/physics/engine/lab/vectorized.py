"""
The reference model's per-step physics, batched and array-generic.

Every function takes the array namespace ``xp`` (``numpy`` in the tests,
``torch`` in the engine) and arrays whose last axis is the vector axis, so
one implementation runs on a GPU batch of thousands of balls and is checked,
element for element, against the scalar oracle in :mod:`lib.physics.reference`:

- :func:`aero_wrench` is :meth:`lib.physics.reference.flight.FlightModel.wrench`.
- :func:`resolve_impacts` is :func:`lib.physics.reference.impact.resolve_impact`.
- :func:`sweep_blades` and :func:`sweep_table` are the hybrid contact sweeps of
  :mod:`lib.physics.engine.contacts` (the game adapter), over one physics step.
- :func:`script_pose` is :func:`lib.physics.reference.surfaces.constant_twist_paddle`.

Only operations numpy and torch spell the same way are used: arithmetic,
comparisons, ``xp.where``, ``xp.stack``, ``xp.sin``/``xp.cos``, ``sum(-1)`` and
``@``. Quaternions are ``(x, y, z, w)``, as Isaac Lab writes them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Array = Any
"""A numpy array or a torch tensor; the vector axis is last."""

SMALL_ANGLE = 1e-8
"""Rotation angle (rad) below which the Rodrigues coefficients use their series."""


def dot(a: Array, b: Array) -> Array:
    """Dot product over the last axis."""

    return (a * b).sum(-1)


def norm(a: Array) -> Array:
    """Euclidean norm over the last axis."""

    return dot(a, a) ** 0.5


def cross(xp: Any, a: Array, b: Array) -> Array:
    """Cross product over the last axis."""

    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], -1)


def positive(xp: Any, x: Array) -> Array:
    """``max(x, 0)`` elementwise."""

    return xp.where(x > 0, x, x * 0)


def clamp(xp: Any, x: Array, low: float, high: float) -> Array:
    """``x`` clipped to ``[low, high]``."""

    x = xp.where(x < low, x * 0 + low, x)
    return xp.where(x > high, x * 0 + high, x)


# --------------------------------------------------------------------------------------
# Aerodynamics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AeroArrays:
    """
    Per-ball flight coefficients, each of shape ``(N,)``; the fields of :class:`FlightModel`.

    :param mass: Ball mass, kg.
    :param inertia: Ball moment of inertia, kg m^2.
    :param radius: Ball radius, m.
    :param aero_gain: ``rho A / (2 m)``, 1/m; zero switches aerodynamics off for that ball.
    :param drag_coefficient: ``C_D0``.
    :param drag_spin_slope: ``dC_D/dS``.
    :param lift_slope: ``dC_L/dS`` at ``S = 0``.
    :param lift_c_max: Lift saturation (rational), cap (linear) or constant.
    :param spin_decay_gain: ``C_tau rho A R^2 / (2 I)``, 1/m.
    :param lift_form: ``rational``, ``linear`` or ``constant``; one form per batch.
    """

    mass: Array
    inertia: Array
    radius: Array
    aero_gain: Array
    drag_coefficient: Array
    drag_spin_slope: Array
    lift_slope: Array
    lift_c_max: Array
    spin_decay_gain: Array
    lift_form: str = "rational"


def lift_coefficient(xp: Any, s: Array, slope: Array, c_max: Array, form: str) -> Array:
    """
    ``C_L(S)`` of :class:`lib.physics.params.LiftModel`, batched.

    :param s: Spin parameter, ``>= 0``.
    :param slope: Initial slope.
    :param c_max: Saturation, cap or constant.
    :param form: ``rational``, ``linear`` or ``constant``.
    :return: ``C_L``, zero where ``S <= 0``.
    """

    s = positive(xp, s)
    if form == "rational":
        out = slope * s / (1.0 + slope * s / c_max)
    elif form == "linear":
        linear = slope * s
        out = xp.where(linear < c_max, linear, c_max + 0 * s)
    elif form == "constant":
        out = c_max + 0 * s
    else:
        raise ValueError(f"unknown lift form {form!r}")
    return xp.where(s > 0, out, 0 * s)


def aero_wrench(xp: Any, velocity: Array, spin: Array, coefficients: AeroArrays) -> tuple[Array, Array]:
    """
    Aerodynamic force and torque on each ball, excluding gravity (batched :meth:`FlightModel.wrench`).

    :param velocity: Velocity relative to still air, m/s, shape ``(N, 3)``.
    :param spin: Angular velocity, rad/s, shape ``(N, 3)``.
    :param coefficients: Per-ball coefficients.
    :return: Force (N) and torque (N m), each ``(N, 3)``, table frame.
    """

    c = coefficients
    v2 = dot(velocity, velocity)
    moving = v2 > 0
    speed = v2**0.5
    safe_v2 = xp.where(moving, v2, 0 * v2 + 1.0)
    safe_speed = xp.where(moving, speed, 0 * speed + 1.0)
    wv = dot(spin, velocity)
    wperp2 = dot(spin, spin) - wv * wv / safe_v2
    wperp = xp.where(moving & (wperp2 > 0), positive(xp, wperp2) ** 0.5, 0 * wperp2)
    s = c.radius * wperp / safe_speed
    k = c.aero_gain
    drag = xp.where(moving, k * (c.drag_coefficient + c.drag_spin_slope * s) * speed, 0 * speed)
    acceleration = -drag[..., None] * velocity
    spinning = wperp > 0
    lift = lift_coefficient(xp, s, c.lift_slope, c.lift_c_max, c.lift_form)
    magnus = xp.where(spinning, k * lift * speed / xp.where(spinning, wperp, 0 * wperp + 1.0), 0 * wperp)
    acceleration = acceleration + magnus[..., None] * cross(xp, spin, velocity)
    spin_rate = -(c.spin_decay_gain * speed)[..., None] * spin
    return c.mass[..., None] * acceleration, c.inertia[..., None] * spin_rate


# --------------------------------------------------------------------------------------
# Impact
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveSpec:
    """
    A coefficient against normal approach speed, piecewise linear and held constant past both ends.

    :param speeds: Knot speeds, m/s, strictly increasing.
    :param values: Coefficient at each knot.
    :param reference: The scalar the curve was tabulated for; ``None`` makes the curve replace the scalar.
    """

    speeds: tuple[float, ...]
    values: tuple[float, ...]
    reference: float | None


def curve_value(xp: Any, x: Array, curve: CurveSpec) -> Array:
    """
    ``numpy.interp(x, speeds, values)`` with fixed knots, written with clamps so torch runs it unchanged.

    :param x: Query speeds, m/s.
    :param curve: The curve.
    :return: Values at ``x``.
    """

    out = 0 * x + curve.values[0]
    for i in range(len(curve.speeds) - 1):
        span = curve.speeds[i + 1] - curve.speeds[i]
        slope = (curve.values[i + 1] - curve.values[i]) / span
        out = out + slope * clamp(xp, x - curve.speeds[i], 0.0, span)
    return out


@dataclass(frozen=True)
class MaterialArrays:
    """
    Per-ball impact law for one surface kind, each array of shape ``(N,)`` (batched :class:`ContactMaterial`).

    :param restitution: Normal restitution scalar.
    :param friction: Coulomb friction scalar.
    :param tangential_restitution: ``e_t``.
    :param normal_spin_retention: ``s_n``.
    :param restitution_curve: Optional speed curve, scaled per ball by ``restitution / reference``.
    :param friction_curve: Optional speed curve for friction, scaled the same way.
    :param tangential_damping: Catch law (the net mesh): fraction ``d_t`` of the tangential velocity removed.
    :param spin_damping: Catch law: fraction ``d_w`` of the spin removed.
    """

    restitution: Array
    friction: Array
    tangential_restitution: Array
    normal_spin_retention: Array
    restitution_curve: CurveSpec | None = None
    friction_curve: CurveSpec | None = None
    tangential_damping: Array | None = None
    spin_damping: Array | None = None


REST_SPEED_MPS: float = 0.05
"""A static-surface touch whose rebound would separate slower than this while gravity presses the ball on is a sustained
(resting or rolling) contact, not a bounce: the reference simulation's ``rest_speed`` and the game adapter's
``REST_SPEED_MPS``."""


def sustained_material(xp: Any, material: MaterialArrays) -> MaterialArrays:
    """
    The law of a sustained contact (batched :func:`lib.physics.engine.contacts.sustained_material`): no normal rebound,
    Coulomb friction without tangential restitution, and spin about the normal kept.
    """

    zero = 0 * material.restitution
    return MaterialArrays(
        restitution=zero,
        friction=material.friction,
        tangential_restitution=zero,
        normal_spin_retention=zero + 1.0,
        restitution_curve=None,
        friction_curve=material.friction_curve,
    )


def restitution_at(xp: Any, material: MaterialArrays, approach: Array) -> Array:
    """Batched :meth:`ContactMaterial.restitution_at`: clipped to [0, 1]."""

    curve = material.restitution_curve
    if curve is None:
        return material.restitution + 0 * approach
    scale = 1.0 if curve.reference is None else material.restitution / curve.reference
    return clamp(xp, curve_value(xp, approach, curve) * scale, 0.0, 1.0)


def friction_at(xp: Any, material: MaterialArrays, approach: Array) -> Array:
    """Batched :meth:`ContactMaterial.friction_at`: never negative."""

    curve = material.friction_curve
    if curve is None:
        return material.friction + 0 * approach
    scale = 1.0 if curve.reference is None else material.friction / curve.reference
    return positive(xp, curve_value(xp, approach, curve) * scale)


@dataclass(frozen=True)
class BallArrays:
    """
    Per-ball rigid-body constants, each ``(N,)``.

    :param mass: kg.
    :param radius: m.
    :param inertia: kg m^2.
    """

    mass: Array
    radius: Array
    inertia: Array

    @property
    def inertia_ratio(self) -> Array:
        """``I / (m R^2)``."""

        return self.inertia / (self.mass * self.radius * self.radius)


@dataclass(frozen=True)
class ImpactBatch:
    """
    Outcome of :func:`resolve_impacts`.

    :param velocity: Post-impact velocity, ``(N, 3)``; unchanged where not approaching.
    :param spin: Post-impact spin, ``(N, 3)``.
    :param approaching: Whether the contact was approaching (an impulse was applied), ``(N,)``.
    :param stick: Whether the tangential target lay inside the friction cone, ``(N,)``.
    :param approach_speed: ``-u_n`` before, m/s, ``(N,)``.
    :param restitution: The ``e`` used, ``(N,)``.
    :param degenerate: Rigid-body sliding contacts with no consistent impulse (Painleve), resolved as a point mass, ``(N,)``.
    :param separation_speed: Normal speed of the contact point away from the surface after the impact, m/s, ``(N,)``.
    """

    velocity: Array
    spin: Array
    approaching: Array
    stick: Array
    approach_speed: Array
    restitution: Array
    degenerate: Array | None = None
    separation_speed: Array | None = None


def resolve_impacts(
    xp: Any,
    velocity: Array,
    spin: Array,
    ball: BallArrays,
    normal: Array,
    material: MaterialArrays,
    surface_velocity: Array,
    inverse_surface_mass: Array,
    contact_offset: Array | None = None,
    inverse_surface_inertia: Array | None = None,
    catch: Array | None = None,
) -> ImpactBatch:
    """
    Batched :func:`lib.physics.reference.impact.resolve_impact` (Brach's rigid-ball impact with tangential compliance).

    With ``contact_offset`` and ``inverse_surface_inertia`` the surface is a free rigid body (the reference's
    off-centre paddle): the contact-point compliance is ``K_s = (1/M) 1 + [r]x^T I^-1 [r]x``, the stick impulse lands
    ``u_n' = -e u_n`` and ``u_t' = -e_t u_t`` exactly through the full 3x3 compliance, and the slip impulse opposes
    the initial slip with ``P_n`` landing ``u_n' = -e u_n``. A row with zero offset or zero inverse inertia is the
    point-mass law exactly, so table and net rows can share the call.

    :param velocity: Ball centre velocity before impact, m/s, ``(N, 3)``.
    :param spin: Ball angular velocity before impact, rad/s, ``(N, 3)``.
    :param ball: Ball constants.
    :param normal: Unit normal from the surface into the ball, ``(N, 3)``.
    :param material: The pair's impact law.
    :param surface_velocity: Velocity of the surface's material point at the contact, m/s, ``(N, 3)``.
    :param inverse_surface_mass: ``1 / M``, 1/kg, ``(N,)``; zero for a fixed or perfectly driven surface.
    :param contact_offset: Contact point relative to the surface body's centre of mass, m, ``(N, 3)``.
    :param inverse_surface_inertia: The surface body's inverse inertia tensor in the same frame, ``(N, 3, 3)``;
        zero for a point-mass or immovable surface.
    :param catch: Rows resolved by the reference's ``catch`` law (the net mesh, an immovable surface):
        ``v_r' = -e (v_r . n) n + (1 - d_t) v_t`` and ``w' = (1 - d_w) w``, with no friction, ``(N,)``.
    :return: The post-impact state; balls whose contact is not approaching keep their state.
    """

    m = ball.mass
    k = ball.inertia_ratio
    m_n = 1.0 / (1.0 / m + inverse_surface_mass)
    m_t = 1.0 / (1.0 / m + inverse_surface_mass + 1.0 / (k * m))
    r_c = -ball.radius[..., None] * normal
    u = velocity + cross(xp, spin, r_c) - surface_velocity
    u_n = dot(u, normal)
    u_t = u - u_n[..., None] * normal
    slip = norm(u_t)
    approach = positive(xp, -u_n)
    e = restitution_at(xp, material, approach)
    mu = friction_at(xp, material, approach)
    e_t = material.tangential_restitution
    approaching = u_n < 0
    p_n = -(1.0 + e) * m_n * u_n
    p_stick = -((1.0 + e_t) * m_t)[..., None] * u_t
    stick = (1.0 + e_t) * m_t * slip <= mu * p_n
    safe_slip = xp.where(slip > 0, slip, 0 * slip + 1.0)
    p_slip = -(mu * p_n / safe_slip)[..., None] * u_t
    p_t = xp.where(stick[..., None], p_stick, p_slip)
    degenerate = approaching & False
    if contact_offset is not None and inverse_surface_inertia is not None:
        p_n, p_t, stick, degenerate = _rigid_surface_impulse(
            xp, normal, u_n, u_t, slip, e, mu, e_t, m, k, inverse_surface_mass, contact_offset, inverse_surface_inertia, p_n, p_t, stick
        )
    impulse = p_n[..., None] * normal + p_t
    v_out = velocity + impulse / m[..., None]
    w_out = spin + cross(xp, r_c, p_t) / ball.inertia[..., None]
    w_normal = dot(spin, normal)
    w_out = w_out - ((1.0 - material.normal_spin_retention) * w_normal)[..., None] * normal
    if catch is not None and material.tangential_damping is not None and material.spin_damping is not None:
        relative = velocity - surface_velocity
        normal_part = dot(relative, normal)
        tangential = relative - normal_part[..., None] * normal
        v_catch = surface_velocity - (e * normal_part)[..., None] * normal + (1.0 - material.tangential_damping)[..., None] * tangential
        w_catch = (1.0 - material.spin_damping)[..., None] * spin
        v_out = xp.where(catch[..., None], v_catch, v_out)
        w_out = xp.where(catch[..., None], w_catch, w_out)
        stick = stick & ~catch
    v_out = xp.where(approaching[..., None], v_out, velocity)
    w_out = xp.where(approaching[..., None], w_out, spin)
    separation = dot(v_out + cross(xp, w_out, r_c) - surface_velocity, normal)
    return ImpactBatch(
        velocity=v_out,
        spin=w_out,
        approaching=approaching,
        stick=stick & approaching,
        approach_speed=approach,
        restitution=e,
        degenerate=degenerate & approaching,
        separation_speed=separation,
    )


def net_mesh(xp: Any, centre: Array, normal: Array, radius: Array, half_span: float, height: float, tape_height: float) -> Array:
    """
    Batched :meth:`lib.physics.reference.surfaces.Net.region` == ``mesh``: a touch on a face, within the span, below the tape.

    A face touch is the one whose normal is the face's (``|n_x| = 1``: the closest point lies on the face, not an edge
    or an end); its touching point ``c - R n`` must lie within the span and below the rigid top tape.

    :param centre: Ball centres at the touch, ``(N, 3)``.
    :param normal: Touch normals from the net into the ball, ``(N, 3)``.
    :param radius: Ball radii, ``(N,)``.
    :return: ``(N,)`` booleans.
    """

    point = centre - radius[..., None] * normal
    on_face = abs(normal[..., 0]) >= 1.0 - 1e-5
    return on_face & (abs(point[..., 1]) < half_span) & (point[..., 2] < height - tape_height)


def _rigid_surface_impulse(
    xp: Any,
    normal: Array,
    u_n: Array,
    u_t: Array,
    slip: Array,
    e: Array,
    mu: Array,
    e_t: Array,
    m: Array,
    k: Array,
    inverse_surface_mass: Array,
    offset: Array,
    inverse_inertia: Array,
    point_p_n: Array,
    point_p_t: Array,
    point_stick: Array,
) -> tuple[Array, Array, Array, Array]:
    """The reference's rigid-body branch, batched: the normal and tangential impulses, stick, and Painleve rows."""

    eye = 0 * inverse_inertia
    eye[..., 0, 0] = 1.0
    eye[..., 1, 1] = 1.0
    eye[..., 2, 2] = 1.0
    r = skew(xp, offset)
    k_surface = inverse_surface_mass[..., None, None] * eye + xp.swapaxes(r, -1, -2) @ inverse_inertia @ r
    outer = normal[..., :, None] * normal[..., None, :]
    k_ball = eye / m[..., None, None] + (eye - outer) / (k * m)[..., None, None]
    k_total = k_ball + k_surface
    target = -((1.0 + e) * u_n)[..., None] * normal - (1.0 + e_t)[..., None] * u_t
    p_stick = xp.linalg.solve(k_total, target[..., None])[..., 0]
    p_stick_n = dot(p_stick, normal)
    p_stick_t = p_stick - p_stick_n[..., None] * normal
    p_stick_t_mag = norm(p_stick_t)
    stick = (p_stick_n > 0) & (p_stick_t_mag <= mu * p_stick_n)
    safe_slip = xp.where(slip > 0, slip, 0 * slip + 1.0)
    safe_stick_t = xp.where(p_stick_t_mag > 0, p_stick_t_mag, 0 * p_stick_t_mag + 1.0)
    slip_dir = xp.where(
        (slip > 0)[..., None], u_t / safe_slip[..., None], xp.where((p_stick_t_mag > 0)[..., None], -p_stick_t / safe_stick_t[..., None], 0 * u_t)
    )
    direction = normal - mu[..., None] * slip_dir
    denominator = dot(normal, (k_total @ direction[..., None])[..., 0])
    degenerate = ~stick & (denominator <= 0)
    safe_denominator = xp.where(denominator > 0, denominator, 0 * denominator + 1.0)
    p_slip_n = -(1.0 + e) * u_n / safe_denominator
    p_slip_t = -(mu * p_slip_n)[..., None] * slip_dir
    p_n = xp.where(stick, p_stick_n, xp.where(degenerate, point_p_n, p_slip_n))
    p_t = xp.where(stick[..., None], p_stick_t, xp.where(degenerate[..., None], point_p_t, p_slip_t))
    return p_n, p_t, stick | (degenerate & point_stick), degenerate


# --------------------------------------------------------------------------------------
# Rotations and the scripted paddle
# --------------------------------------------------------------------------------------


def skew(xp: Any, a: Array) -> Array:
    """The cross-product matrix ``[a]x`` of each vector, ``(..., 3, 3)``."""

    zero = 0 * a[..., 0]
    rows = [
        xp.stack([zero, -a[..., 2], a[..., 1]], -1),
        xp.stack([a[..., 2], zero, -a[..., 0]], -1),
        xp.stack([-a[..., 1], a[..., 0], zero], -1),
    ]
    return xp.stack(rows, -2)


def rotation_from_rotvec(xp: Any, rotvec: Array) -> Array:
    """
    Rodrigues' formula, batched: ``R = I + A [r]x + B [r]x^2``.

    :param rotvec: Rotation vectors (axis times angle, rad), ``(..., 3)``.
    :return: Rotation matrices, ``(..., 3, 3)``.
    """

    theta = norm(rotvec)
    small = theta < SMALL_ANGLE
    safe = xp.where(small, 0 * theta + 1.0, theta)
    a = xp.where(small, 1.0 - theta * theta / 6.0, xp.sin(safe) / safe)
    b = xp.where(small, 0.5 - theta * theta / 24.0, (1.0 - xp.cos(safe)) / (safe * safe))
    k = skew(xp, rotvec)
    eye = 0 * k
    eye[..., 0, 0] = 1.0
    eye[..., 1, 1] = 1.0
    eye[..., 2, 2] = 1.0
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def quat_from_rotvec(xp: Any, rotvec: Array) -> Array:
    """Unit quaternions ``(x, y, z, w)`` of rotation vectors, ``(..., 4)``."""

    theta = norm(rotvec)
    small = theta < SMALL_ANGLE
    safe = xp.where(small, 0 * theta + 1.0, theta)
    s = xp.where(small, 0.5 - theta * theta / 48.0, xp.sin(0.5 * safe) / safe)
    w = xp.cos(0.5 * theta)
    return xp.stack([rotvec[..., 0] * s, rotvec[..., 1] * s, rotvec[..., 2] * s, w], -1)


def quat_multiply(xp: Any, a: Array, b: Array) -> Array:
    """Hamilton product ``a b`` of ``(x, y, z, w)`` quaternions: apply ``b``, then ``a``."""

    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return xp.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


@dataclass(frozen=True)
class ScriptArrays:
    """
    Constant-twist paddle scripts, one per environment (batched :func:`constant_twist_paddle`).

    :param position: Blade centre at ``t = 0``, m, ``(N, 3)``.
    :param rotation: Blade rotation at ``t = 0`` (columns: X towards the handle, Y, Z the forehand normal), ``(N, 3, 3)``.
    :param quaternion: The same rotation as ``(x, y, z, w)``, ``(N, 4)``.
    :param velocity: Linear velocity, m/s, ``(N, 3)``.
    :param angular_velocity: Angular velocity, rad/s, ``(N, 3)``.
    """

    position: Array
    rotation: Array
    quaternion: Array
    velocity: Array
    angular_velocity: Array


@dataclass(frozen=True)
class BladeState:
    """
    Every blade at one instant.

    :param position: Blade centres, ``(N, 3)``.
    :param rotation: Rotations, ``(N, 3, 3)``.
    :param quaternion: Rotations as ``(x, y, z, w)``, ``(N, 4)``.
    :param velocity: Linear velocities, ``(N, 3)``.
    :param angular_velocity: Angular velocities, ``(N, 3)``.
    """

    position: Array
    rotation: Array
    quaternion: Array
    velocity: Array
    angular_velocity: Array


def script_pose(xp: Any, script: ScriptArrays, t: float) -> BladeState:
    """
    Every scripted blade at time ``t``: ``p0 + v t`` and ``exp([w] t) R0``.

    :param script: The scripts.
    :param t: Time since the scripts' reference, s.
    :return: The blades' poses and twists.
    """

    rotvec = script.angular_velocity * t
    return BladeState(
        position=script.position + script.velocity * t,
        rotation=rotation_from_rotvec(xp, rotvec) @ script.rotation,
        quaternion=quat_multiply(xp, quat_from_rotvec(xp, rotvec), script.quaternion),
        velocity=script.velocity,
        angular_velocity=script.angular_velocity,
    )


# --------------------------------------------------------------------------------------
# Hybrid contact sweeps over one step
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BladeGeometry:
    """
    The blade outline every environment shares.

    :param half_thickness: Half the blade thickness with its rubbers, m.
    :param half_length: Elliptical outline semi-axis along local X (towards the handle), m.
    :param half_width: Semi-axis along local Y, m.
    """

    half_thickness: float
    half_length: float
    half_width: float


@dataclass(frozen=True)
class SweepBatch:
    """
    The first touch of each ball on a surface within one step.

    :param contact: Whether the ball touches, approaching and inside the outline, ``(N,)``.
    :param tunnelled: Whether the centre crossed the mid-plane inside the outline without a contact, ``(N,)``.
    :param fraction: Contact time as a fraction of the step, ``(N,)``.
    :param centre: Ball centre at contact, ``(N, 3)``.
    :param normal: Unit normal from the surface into the ball, ``(N, 3)``.
    :param surface_velocity: Velocity of the surface point at the contact, ``(N, 3)``.
    :param offset: Blade sweeps: the touching point relative to the blade centre, ``(N, 3)``.
    :param axes: Blade sweeps: the blade's orthonormal axes at the contact as columns (X handle, Y, Z normal), ``(N, 3, 3)``.
    :param depth: Table and net sweeps: how far a ball that started the step inside the surface is inside it, m, ``(N,)``;
        the contact centre is moved out along the normal by this much (the game adapter's push-out).
    """

    contact: Array
    tunnelled: Array
    fraction: Array
    centre: Array
    normal: Array
    surface_velocity: Array
    offset: Array | None = None
    axes: Array | None = None
    depth: Array | None = None


def _blade_at(xp: Any, start: BladeState, end: BladeState, fraction: Array) -> tuple[Array, Array]:
    position = start.position + fraction[..., None] * (end.position - start.position)
    rotation = start.rotation + fraction[..., None, None] * (end.rotation - start.rotation)
    return position, rotation


def sweep_blades(
    xp: Any, position: Array, velocity: Array, radius: Array, start: BladeState, end: BladeState, dt: float, geometry: BladeGeometry
) -> SweepBatch:
    """
    Sweep each ball against both faces of its moving blade over one step (batched :func:`lib.physics.engine.contacts.sweep_blade`).

    The ball moves at constant velocity over the step; the blade's centre and
    axes move linearly between ``start`` and ``end``. A contact is the first
    time the ball's surface reaches a face from outside, with the touching
    point inside the elliptical outline and the ball approaching the face. A
    ball that starts within reach of a face is tested at the step start.

    :param position: Ball centres at the step start, ``(N, 3)``.
    :param velocity: Ball velocities, ``(N, 3)``.
    :param radius: Ball radii, ``(N,)``.
    :param start: Blades at the step start.
    :param end: Blades at the step end.
    :param dt: Step, s.
    :param geometry: Blade outline.
    :return: The contacts and tunnelling flags.
    """

    n0 = start.rotation[..., :, 2]
    n1 = end.rotation[..., :, 2]
    d0 = dot(n0, position - start.position)
    d1 = dot(n1, position + velocity * dt - end.position)
    reach = radius + geometry.half_thickness
    sign = xp.where(d0 >= 0, 0 * d0 + 1.0, 0 * d0 - 1.0)
    outside = abs(d0) > reach
    crossing = sign * d1 <= reach
    span = d0 - d1
    safe_span = xp.where(abs(span) > 1e-15, span, 0 * span + 1.0)
    fraction = xp.where(outside, (d0 - sign * reach) / safe_span, 0 * d0)
    fraction = clamp(xp, fraction, 0.0, 1.0)

    def touch(f: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
        blade_position, blade_rotation = _blade_at(xp, start, end, f)
        axis_x = blade_rotation[..., :, 0]
        axis_y = blade_rotation[..., :, 1]
        axis_z = blade_rotation[..., :, 2]
        normal = axis_z / norm(axis_z)[..., None]
        centre = position + velocity * (f * dt)[..., None]
        point = centre - (sign * radius)[..., None] * normal
        offset = point - blade_position
        along_x = dot(offset, axis_x) / norm(axis_x)
        along_y = dot(offset, axis_y) / norm(axis_y)
        within = (along_x / geometry.half_length) ** 2 + (along_y / geometry.half_width) ** 2 <= 1.0
        blade_velocity = start.velocity + f[..., None] * (end.velocity - start.velocity)
        blade_spin = start.angular_velocity + f[..., None] * (end.angular_velocity - start.angular_velocity)
        surface_velocity = blade_velocity + cross(xp, blade_spin, point - blade_position)
        # The interpolated rotation is orthonormalised for the inertia tensor: Z the face normal, X in the face.
        unit_x = axis_x - dot(axis_x, normal)[..., None] * normal
        unit_x = unit_x / norm(unit_x)[..., None]
        axes = xp.stack([unit_x, cross(xp, normal, unit_x), normal], -1)
        return within, sign[..., None] * normal, surface_velocity, centre, offset, axes

    within, normal, surface_velocity, centre, offset, axes = touch(fraction)
    approaching = dot(velocity - surface_velocity, normal) < 0
    candidate = (outside & crossing) | ~outside
    contact = candidate & within & approaching
    mid = clamp(xp, d0 / safe_span, 0.0, 1.0)
    mid_within = touch(mid)[0]
    sides_differ = (d0 >= 0) != (d1 >= 0)
    tunnelled = ~contact & outside & sides_differ & mid_within
    return SweepBatch(
        contact=contact, tunnelled=tunnelled, fraction=fraction, centre=centre, normal=normal, surface_velocity=surface_velocity, offset=offset, axes=axes
    )


def surface_inverse_inertia(xp: Any, axes: Array, inverse_moments: Array) -> Array:
    """
    ``R diag(1/I) R^T``: the inverse of the world-frame inertia ``R diag(I) R^T`` of each blade.

    :param axes: Orthonormal blade axes as columns, ``(N, 3, 3)``.
    :param inverse_moments: ``1/I_xx, 1/I_yy, 1/I_zz`` per row, ``(N, 3)``; zero rows give a point mass.
    :return: ``(N, 3, 3)``.
    """

    return (axes * inverse_moments[..., None, :]) @ xp.swapaxes(axes, -1, -2)


TABLE_START_SLOP_M: float = 2e-4
"""A ball that starts a step up to this far below the playing surface, falling and over the table, touches it at once.

The sweep predicts the step's end from the state it reads, and the engine's actual end can lie slightly lower: the
float32 read-back of a lattice environment metres from the origin is good to about 1e-6 m, and an integrator that
applies the step's velocity kick before moving the ball ends up to ``|a| dt^2`` lower (1.6e-6 m at 1/4000 s). A ball
whose predicted end sat in that sliver above the plane started the next step just below it and was never swept
again: it fell through the table (sensitivity v3, ``spin_decay_coefficient=0.031`` loop, run eed6888b, an
environment at (6, -18, -2) m; about one bounce in a thousand). The game's physics, with the table at the world
origin, let none of 3000 bounces through at 960 or 1920 Hz (runs 0982e8df, ae4b720f), so the read-back is the likely
cause here. 0.2 mm covers either and is far below anything the ball tracking resolves; deeper than that the ball can
only be under the table, which this sweep must not touch.
"""


def sweep_table(xp: Any, position: Array, velocity: Array, radius: Array, dt: float, half_length: float, half_width: float) -> SweepBatch:
    """
    Sweep each ball against the playing surface ``z = 0`` from above (batched :func:`lib.physics.engine.contacts.sweep_table`).

    :param position: Ball centres at the step start, ``(N, 3)``.
    :param velocity: Ball velocities, ``(N, 3)``.
    :param radius: Ball radii, ``(N,)``.
    :param dt: Step, s.
    :param half_length: Half the table length, m.
    :param half_width: Half the table width, m.
    :return: The contacts (``tunnelled`` is always false: the table is a plane, crossed only through a contact). Unlike the
        game adapter, a falling ball that starts up to :data:`TABLE_START_SLOP_M` below the plane touches at fraction 0.
    """

    z0 = position[..., 2] - radius
    vz = velocity[..., 2]
    z1 = z0 + vz * dt
    falling = (z0 >= -TABLE_START_SLOP_M) & (z1 <= 0) & (vz < 0)
    span = z0 - z1
    fraction = clamp(xp, xp.where(falling & (span > 0), z0 / xp.where(span > 0, span, 0 * span + 1.0), 0 * z0), 0.0, 1.0)
    centre = position + velocity * (fraction * dt)[..., None]
    inside = (abs(centre[..., 0]) <= half_length) & (abs(centre[..., 1]) <= half_width)
    up = 0 * velocity
    up[..., 2] = 1.0
    return SweepBatch(
        contact=falling & inside,
        tunnelled=falling & ~falling,
        fraction=fraction,
        centre=centre,
        normal=up,
        surface_velocity=0 * velocity,
        depth=positive(xp, -z0),
    )


def box_gap(xp: Any, centre: Array, radius: Array, half_thickness: float, half_span: float, height: float) -> tuple[Array, Array]:
    """
    Distance from each ball's surface to the net box, and the unit normal from its closest point to the centre.

    The box is ``|x| <= half_thickness``, ``|y| <= half_span``, ``0 <= z <= height`` (the reference model's :class:`Net`).

    :param centre: Ball centres, ``(N, 3)``.
    :param radius: Ball radii, ``(N,)``.
    :return: ``(gap, normal)``; the normal is ``+x`` or ``-x`` for a centre inside the box (never in play).
    """

    x = clamp(xp, centre[..., 0], -half_thickness, half_thickness)
    y = clamp(xp, centre[..., 1], -half_span, half_span)
    z = clamp(xp, centre[..., 2], 0.0, height)
    closest = xp.stack([x, y, z], -1)
    offset = centre - closest
    distance = norm(offset)
    inside = distance < 1e-9
    fallback = 0 * offset
    fallback[..., 0] = xp.where(centre[..., 0] >= 0, 0 * distance + 1.0, 0 * distance - 1.0)
    normal = xp.where(inside[..., None], fallback, offset / xp.where(inside, 0 * distance + 1.0, distance)[..., None])
    return distance - radius, normal


NET_SWEEP_SAMPLES: int = 16
"""Sub-step samples of the net sweep; a crossing is then bracketed by bisection."""

NET_SWEEP_BISECTIONS: int = 8
"""Bisections of the bracketed crossing: the touch is located to ``dt / (16 * 256)``."""


def sweep_net(xp: Any, position: Array, velocity: Array, radius: Array, dt: float, half_thickness: float, half_span: float, height: float) -> SweepBatch:
    """
    Sweep each ball against the net box over one step (the hybrid net).

    The gap to a convex box along a straight path is convex in time, so its
    first root in the step is found by sampling the step and bisecting the
    first bracket. The normal is the reference model's: from the box's closest
    point to the ball centre at the touch, so a ball clipping the top edge is
    deflected by the edge's geometry, not by where it was at the step start.

    :param position: Ball centres at the step start, ``(N, 3)``.
    :param velocity: Ball velocities, ``(N, 3)``.
    :param radius: Ball radii, ``(N,)``.
    :param dt: Step, s.
    :return: The touches (``tunnelled`` reports a centre that ends the step inside the box).
    """

    gap0, normal0 = box_gap(xp, position, radius, half_thickness, half_span, height)
    lo = 0 * gap0
    hi = 0 * gap0 + 1.0
    found = gap0 <= 0
    for k in range(1, NET_SWEEP_SAMPLES + 1):
        f = k / NET_SWEEP_SAMPLES
        gap, _ = box_gap(xp, position + velocity * (f * dt), radius, half_thickness, half_span, height)
        first = ~found & (gap <= 0)
        lo = xp.where(first, 0 * gap + (k - 1) / NET_SWEEP_SAMPLES, lo)
        hi = xp.where(first, 0 * gap + f, hi)
        found = found | first
    started_inside = gap0 <= 0
    for _ in range(NET_SWEEP_BISECTIONS):
        mid = 0.5 * (lo + hi)
        gap, _ = box_gap(xp, position + velocity * (mid * dt)[..., None], radius, half_thickness, half_span, height)
        lo = xp.where(gap > 0, mid, lo)
        hi = xp.where(gap > 0, hi, mid)
    fraction = xp.where(started_inside, 0 * lo, lo)
    centre = position + velocity * (fraction * dt)[..., None]
    _, normal = box_gap(xp, centre, radius, half_thickness, half_span, height)
    normal = xp.where(started_inside[..., None], normal0, normal)
    approaching = dot(velocity, normal) < 0
    end_gap, _ = box_gap(xp, position + velocity * dt, 0 * radius, half_thickness, half_span, height)
    return SweepBatch(
        contact=found & approaching,
        tunnelled=~found & (end_gap <= 0),
        fraction=fraction,
        centre=centre,
        normal=normal,
        surface_velocity=0 * velocity,
        depth=xp.where(started_inside, positive(xp, -gap0), 0 * gap0),
    )


# --------------------------------------------------------------------------------------
# Static shapes: the table board, net posts and clamps
# --------------------------------------------------------------------------------------

Shape = tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float]]
"""A static shape: ``("box", low, high, _)``, ``("cylinder", (axis_x, axis_y, _), (rho, _, _), (z_bottom, z_top))`` (vertical),
``("cylinder_x", a, b, (rho, _))`` (axis along X from ``a`` to ``b``) or ``("capsule", a, b, (rho, _))``."""


def shape_gap(xp: Any, centre: Array, radius: Array, shape: Shape) -> tuple[Array, Array, Array]:
    """
    Distance from each ball's surface to a static shape, the unit normal from its closest point, and whether the centre is inside.

    Boxes are axis-aligned (the reference's ``TableBoard`` and clamp boxes); cylinders are vertical and capped (the
    posts). A centre inside the shape gets the normal of the nearest face and a negative gap.

    :param centre: Ball centres, ``(N, 3)``.
    :param radius: Ball radii, ``(N,)``.
    :param shape: The shape.
    :return: ``(gap, normal, inside)``.
    """

    kind, a, b, z_range = shape
    if kind == "capsule":
        return _capsule_gap(xp, centre, radius, a, b, z_range[0])
    if kind == "cylinder_x":
        # A cylinder along X: the vertical cylinder's algebra with the axes permuted (x <-> z).
        swapped = xp.stack([centre[..., 2], centre[..., 1], centre[..., 0]], -1)
        gap, normal, inside = shape_gap(xp, swapped, radius, ("cylinder", (a[2], a[1], 0.0), (z_range[0], 0.0, 0.0), (a[0], b[0])))
        return gap, xp.stack([normal[..., 2], normal[..., 1], normal[..., 0]], -1), inside
    x, y, z = centre[..., 0], centre[..., 1], centre[..., 2]
    if kind == "box":
        low, high = a, b
        cx, cy, cz = clamp(xp, x, low[0], high[0]), clamp(xp, y, low[1], high[1]), clamp(xp, z, low[2], high[2])
        inside_depths = [x - low[0], high[0] - x, y - low[1], high[1] - y, z - low[2], high[2] - z]
        face_normals = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]
        radial_normal = None
    else:
        ax, ay, rho = a[0], a[1], b[0]
        dx, dy = x - ax, y - ay
        radial = xp.sqrt(dx * dx + dy * dy)
        safe = xp.where(radial > 1e-12, radial, 0 * radial + 1.0)
        outside = radial > rho
        cx = xp.where(outside, ax + dx * rho / safe, x)
        cy = xp.where(outside, ay + dy * rho / safe, y)
        cz = clamp(xp, z, z_range[0], z_range[1])
        inside_depths = [rho - radial, z - z_range[0], z_range[1] - z]
        face_normals = [None, (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]
        radial_normal = xp.stack([xp.where(radial > 1e-12, dx / safe, 0 * dx + 1.0), dy / safe, 0 * dx], -1)
    closest = xp.stack([cx, cy, cz], -1)
    offset = centre - closest
    distance = norm(offset)
    inside = distance < 1e-12
    normal = offset / xp.where(inside, 0 * distance + 1.0, distance)[..., None]
    depth = inside_depths[0]
    fallback = 0 * centre
    for k, d in enumerate(inside_depths):
        nearer = d <= depth if k else d == d
        depth = xp.where(nearer, d, depth)
        normal_k = face_normals[k]
        if normal_k is None:
            assert radial_normal is not None
            fallback = xp.where(nearer[..., None], radial_normal, fallback)
        else:
            fixed = 0 * centre
            for axis in range(3):
                fixed[..., axis] = normal_k[axis]
            fallback = xp.where(nearer[..., None], fixed, fallback)
    normal = xp.where(inside[..., None], fallback, normal)
    gap = xp.where(inside, -depth, distance) - radius
    return gap, normal, inside


def _capsule_gap(xp: Any, centre: Array, radius: Array, a: tuple[float, float, float], b: tuple[float, float, float], rho: float) -> tuple[Array, Array, Array]:
    """A capsule (segment ``a``-``b`` swept by radius ``rho``): gap, normal from its surface, and whether the centre is inside."""

    start = 0 * centre
    axis = 0 * centre
    for k in range(3):
        start[..., k] = a[k]
        axis[..., k] = b[k] - a[k]
    length2 = float(sum((b[k] - a[k]) ** 2 for k in range(3)))
    t = clamp(xp, dot(centre - start, axis) / max(length2, 1e-24), 0.0, 1.0)
    offset = centre - (start + axis * t[..., None])
    distance = norm(offset)
    degenerate = distance < 1e-12
    fallback = 0 * centre
    fallback[..., 2] = 1.0
    normal = xp.where(degenerate[..., None], fallback, offset / xp.where(degenerate, 0 * distance + 1.0, distance)[..., None])
    return distance - rho - radius, normal, distance < rho


def sweep_shape(xp: Any, position: Array, velocity: Array, radius: Array, dt: float, shape: Shape) -> SweepBatch:
    """
    Sweep each ball against a static shape over one step (batched :func:`lib.physics.engine.contacts.sweep_box`).

    The ball moves at constant velocity; the gap is sampled at :data:`NET_SWEEP_SAMPLES` points and the first
    bracketed root bisected :data:`NET_SWEEP_BISECTIONS` times, so the touch is located to ``dt / 4096``. The normal
    runs from the shape's closest point to the centre at the touch (edges and corners deflect by their geometry). A
    ball that starts overlapping the shape (centre outside) touches at once and carries its depth; a centre inside the
    shape is a tunnel that happened earlier and is reported, not resolved.

    :return: The touches; ``tunnelled`` when the centre starts inside, or ends inside without a touch.
    """

    gap0, normal0, inside0 = shape_gap(xp, position, radius, shape)
    lo = 0 * gap0
    hi = 0 * gap0 + 1.0
    found = gap0 <= 0
    for k in range(1, NET_SWEEP_SAMPLES + 1):
        f = k / NET_SWEEP_SAMPLES
        gap, _n, _i = shape_gap(xp, position + velocity * (f * dt), radius, shape)
        first = ~found & (gap <= 0)
        lo = xp.where(first, 0 * gap + (k - 1) / NET_SWEEP_SAMPLES, lo)
        hi = xp.where(first, 0 * gap + f, hi)
        found = found | first
    started = gap0 <= 0
    for _ in range(NET_SWEEP_BISECTIONS):
        mid = 0.5 * (lo + hi)
        gap, _n, _i = shape_gap(xp, position + velocity * (mid * dt)[..., None], radius, shape)
        lo = xp.where(gap > 0, mid, lo)
        hi = xp.where(gap > 0, hi, mid)
    fraction = xp.where(started, 0 * lo, lo)
    centre = position + velocity * (fraction * dt)[..., None]
    _g, normal, _i = shape_gap(xp, centre, radius, shape)
    normal = xp.where(started[..., None], normal0, normal)
    approaching = dot(velocity, normal) < 0
    _g_end, _n_end, inside_end = shape_gap(xp, position + velocity * dt, 0 * radius, shape)
    return SweepBatch(
        contact=found & approaching & ~inside0,
        tunnelled=inside0 | (~found & inside_end),
        fraction=fraction,
        centre=centre,
        normal=normal,
        surface_velocity=0 * velocity,
        depth=xp.where(started & ~inside0, positive(xp, -gap0), 0 * gap0),
    )


def first_touch(xp: Any, sweeps: Sequence[SweepBatch]) -> tuple[SweepBatch, Array]:
    """
    Per ball, the earliest touching sweep of several, and its index (``-1`` where none touches).

    Fields are taken from the chosen sweep; ``tunnelled`` is any sweep's.
    """

    fractions = xp.stack([xp.where(s.contact, s.fraction, 0 * s.fraction + 2.0) for s in sweeps], -1)
    index = fractions.argmin(-1)
    chosen = sweeps[0]
    contact, fraction, centre, normal, surface_velocity, depth = (
        chosen.contact & (index == 0),
        chosen.fraction,
        chosen.centre,
        chosen.normal,
        chosen.surface_velocity,
        chosen.depth if chosen.depth is not None else 0 * chosen.fraction,
    )
    tunnelled = chosen.tunnelled
    for k, s in enumerate(sweeps[1:], start=1):
        pick = s.contact & (index == k)
        contact = contact | pick
        fraction = xp.where(pick, s.fraction, fraction)
        centre = xp.where(pick[..., None], s.centre, centre)
        normal = xp.where(pick[..., None], s.normal, normal)
        surface_velocity = xp.where(pick[..., None], s.surface_velocity, surface_velocity)
        depth = xp.where(pick, s.depth if s.depth is not None else 0 * s.fraction, depth)
        tunnelled = tunnelled | s.tunnelled
    which = xp.where(contact, index, 0 * index - 1)
    return SweepBatch(
        contact=contact, tunnelled=tunnelled, fraction=fraction, centre=centre, normal=normal, surface_velocity=surface_velocity, depth=depth
    ), which


def part_shape(part: Any) -> Shape:
    """A reference undercarriage :class:`lib.physics.reference.undercarriage.Part` as a :data:`Shape`."""

    if part.shape == "box":
        return ("box", tuple(part.a), tuple(part.b), (0.0, 0.0))
    if part.shape == "cylinder_z":
        return ("cylinder", tuple(part.a), (part.radius, 0.0, 0.0), (part.a[2], part.b[2]))
    if part.shape == "cylinder_x":
        return ("cylinder_x", tuple(part.a), tuple(part.b), (part.radius, 0.0))
    return ("capsule", tuple(part.a), tuple(part.b), (part.radius, 0.0))


def table_board_shape(half_length: float, half_width: float, thickness: float) -> Shape:
    """The reference's ``TableBoard`` box: the playing surface on top, ``thickness`` deep."""

    return ("box", (-half_length, -half_width, -thickness), (half_length, half_width, 0.0), (0.0, 0.0))


def sweep_table_box(xp: Any, position: Array, velocity: Array, radius: Array, dt: float, half_length: float, half_width: float, thickness: float) -> SweepBatch:
    """
    The whole table board (batched :func:`lib.physics.engine.contacts.sweep_table_box`): the playing-surface plane
    (:func:`sweep_table`, unchanged) first, and the board box (ends, sides, edges, corners, underside) where the plane
    finds no touch.
    """

    top = sweep_table(xp, position, velocity, radius, dt, half_length, half_width)
    board = sweep_shape(xp, position, velocity, radius, dt, table_board_shape(half_length, half_width, thickness))
    use_top = top.contact
    return SweepBatch(
        contact=top.contact | board.contact,
        tunnelled=~top.contact & board.tunnelled,
        fraction=xp.where(use_top, top.fraction, board.fraction),
        centre=xp.where(use_top[..., None], top.centre, board.centre),
        normal=xp.where(use_top[..., None], top.normal, board.normal),
        surface_velocity=0 * velocity,
        depth=xp.where(use_top, top.depth if top.depth is not None else 0 * top.fraction, board.depth if board.depth is not None else 0 * top.fraction),
    )


def classify_table_touch(xp: Any, normal: Array, tolerance: float = 1e-6) -> Array:
    """
    Batched :func:`lib.physics.reference.surfaces.classify_table_touch` as codes: 0 ``top``, 1 ``edge``, 2 ``side``, 3 ``underside``.

    The tolerance is wider than the reference's 1e-9 for float32 normals.
    """

    nz = normal[..., 2]
    zero = 0 * nz
    return xp.where(nz >= 1.0 - tolerance, zero, xp.where(nz > tolerance, zero + 1, xp.where(nz >= -tolerance, zero + 2, zero + 3)))


def imposed_start(contact_centre: Array, post_velocity: Array, fraction: Array, dt: float) -> Array:
    """
    The position to write at the step start so the engine's step ends where the event-driven solution is.

    The engine integrates the whole step with the post-impact velocity, so a
    ball written at ``c - v' f dt`` passes the contact point ``c`` at the contact
    time and ends the step at ``c + v' (1 - f) dt`` (gravity and the air force
    add ``a dt^2``, under a micrometre at ``dt = 1/4000``).

    :param contact_centre: Ball centre at contact, ``(N, 3)``.
    :param post_velocity: Post-impact velocity, ``(N, 3)``.
    :param fraction: Contact time as a fraction of the step, ``(N,)``.
    :param dt: Step, s.
    :return: Start positions, ``(N, 3)``.
    """

    return contact_centre - post_velocity * (fraction * dt)[..., None]


def imposed_state(
    contact_centre: Array, post_velocity: Array, post_spin: Array, acceleration: Array, angular_acceleration: Array, fraction: Array, dt: float
) -> tuple[Array, Array, Array]:
    """
    The state to write at the step start so the engine's semi-implicit step ends on the event-driven solution.

    The ball leaves the contact at ``t_c = f dt`` with ``v'`` and ``w'`` and then accelerates at ``a'`` (gravity and
    the air force at the post-impact state) for the rest of the step. The engine integrates the whole step,
    ``v_end = v_s + a' dt`` and ``x_end = x_s + v_end dt``; landing ``v_end = v' + a' (1 - f) dt`` and
    ``x_end = c + v' (1 - f) dt + a' ((1 - f) dt)^2 / 2`` needs ``v_s = v' - a' f dt`` and
    ``x_s = c - v' f dt - a' (1 - f)(1 + f) dt^2 / 2``. Without the ``a' f dt`` term the step after every contact
    carries an extra ``a' f dt``: an error proportional to ``dt`` whose size depends on the contact's sub-step phase,
    which made C8's step-halving differences irregular (``dt`` 1/4000-1/24000, runs 152df3b2 and 4bfc4ad1).

    :param contact_centre: Ball centre at contact, ``(N, 3)``.
    :param post_velocity: Post-impact velocity, ``(N, 3)``.
    :param post_spin: Post-impact spin, ``(N, 3)``.
    :param acceleration: Linear acceleration after the contact, ``(N, 3)``.
    :param angular_acceleration: Angular acceleration after the contact, ``(N, 3)``.
    :param fraction: Contact time as a fraction of the step, ``(N,)``.
    :param dt: Step, s.
    :return: ``(position, velocity, spin)`` to write.
    """

    lead = (fraction * dt)[..., None]
    rest = dt - lead
    position = contact_centre - post_velocity * lead - 0.5 * acceleration * rest * (dt + lead)
    return position, post_velocity - acceleration * lead, post_spin - angular_acceleration * lead


__all__ = [
    "NET_SWEEP_BISECTIONS",
    "NET_SWEEP_SAMPLES",
    "REST_SPEED_MPS",
    "TABLE_START_SLOP_M",
    "AeroArrays",
    "BallArrays",
    "BladeGeometry",
    "BladeState",
    "CurveSpec",
    "ImpactBatch",
    "MaterialArrays",
    "ScriptArrays",
    "Shape",
    "SweepBatch",
    "aero_wrench",
    "box_gap",
    "classify_table_touch",
    "cross",
    "curve_value",
    "dot",
    "first_touch",
    "friction_at",
    "imposed_start",
    "imposed_state",
    "lift_coefficient",
    "net_mesh",
    "norm",
    "part_shape",
    "quat_from_rotvec",
    "quat_multiply",
    "resolve_impacts",
    "restitution_at",
    "rotation_from_rotvec",
    "script_pose",
    "shape_gap",
    "skew",
    "surface_inverse_inertia",
    "sustained_material",
    "sweep_blades",
    "sweep_net",
    "sweep_shape",
    "sweep_table",
    "sweep_table_box",
    "table_board_shape",
]
