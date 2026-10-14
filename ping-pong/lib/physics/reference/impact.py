"""
Rigid-ball impact on a flat surface with any orientation and velocity.

The ball is a rigid sphere of mass ``m``, radius ``R`` and moment of inertia
``I = k m R^2``. The surface is a plane with unit normal ``n`` (pointing from
the surface towards the ball), moving at ``u_s`` at the contact point, and
behaving as a point of effective mass ``M`` (infinite for the table and floor,
finite for a paddle held in a hand). Notation, all vectors in one frame:

- contact arm ``r_c = -R n``
- contact-point velocity ``v_c = v + omega x r_c``
- relative contact velocity ``u = v_c - u_s``, with normal part ``u_n = u . n``
  (negative when approaching) and tangential part ``u_t = u - u_n n``.

An impulse ``P`` at the contact changes the ball by ``dv = P / m`` and
``d omega = (r_c x P) / I``, and the surface by ``-P / M``. Because the contact
normal passes through the centre of a sphere, the normal and tangential
problems decouple::

    du_n = P_n / m_n,   1/m_n = 1/m + 1/M
    du_t = P_t / m_t,   1/m_t = 1/m + 1/M + 1/(k m)

**Normal.** Newton's restitution, ``u_n' = -e u_n``, so ``P_n = -(1 + e) m_n u_n``.
For a sphere on a plane Newton's, Poisson's and Stronge's definitions agree.

**Tangential** (Brach 1989, 1991; ``e_t = 0`` is the rigid Coulomb model of Routh
and the rolling limit of Garwin 1969). The impulse that would leave the
contact point at ``u_t' = -e_t u_t`` is ``P_stick = -(1 + e_t) m_t u_t``. If
``|P_stick| <= mu P_n`` the contact sticks and takes it (with ``e_t = 0`` the
ball leaves rolling; with ``e_t > 0`` the slip reverses, the overspin a
compliant rubber returns). Otherwise the ball slides for the whole impact and
``P_t = -mu P_n u_t / |u_t|``. Slip direction cannot change during the impact
because the tangential dynamics are isotropic, so the 3D problem is 1D along
``u_t``.

**Off-centre impacts on a rigid paddle.** Given the contact offset ``r`` from
the surface body's centre of mass and its inertia tensor ``I_s``, the
surface's contact-point compliance is ``K_s = (1/M) 1 + [r]x^T I_s^-1 [r]x``
in place of ``(1/M) 1``. Along the normal this is the effective mass
``1/M_eff = 1/M + (r x n)^T I_s^-1 (r x n)``: 5 cm off-centre a 0.17 kg blade
meets the ball with about 0.075 kg. An in-plane offset also couples normal
and tangential impulse, so this branch solves with the full 3x3 compliance:
the stick impulse lands ``u_n' = -e u_n`` and ``u_t' = -e_t u_t`` exactly, and
the slip impulse keeps friction opposed to the initial slip (Brach's
algebraic model) with ``u_n' = -e u_n``. The surface takes ``-P`` at ``r``:
``dV = -P/M`` and ``d omega_s = -I_s^-1 (r x P)``. With ``r = 0`` it is the
point-mass law above. Against the 2.7 g ball the coupling is tiny, so energy
never rises in practice; the tests check it over paddle-like cases.

**Catch law (the net mesh).** A compliant mesh is not a rigid plane: it
wraps around the ball and absorbs its motion along the face instead of
gripping its surface. Under ``ContactMaterial.law == "catch"``, with the
velocity ``v_r = v - u_s`` relative to an immovable surface::

    v_r'  = -e (v_r . n) n + (1 - d_t) (v_r - (v_r . n) n)
    omega' = (1 - d_w) omega

There is no contact-point friction, so spin is never converted into
tangential velocity: a ball with backspin that hits the mesh gains no lift.
Each factor lies in [0, 1], so the kinetic energy change
``1/2 m (e^2 - 1) v_n^2 + 1/2 m ((1 - d_t)^2 - 1) |v_t|^2 + 1/2 I ((1 - d_w)^2 - 1) |omega|^2``
is never positive. The law is phenomenological: momentum and angular
momentum go into the net. Ball-net measurements do not exist, so
``d_t`` and ``d_w`` are judgement.

**Spin about the normal.** No impulse above has a moment about ``n``, so the
rigid model keeps that spin component; ``normal_spin_retention`` ``s_n``
scales it to ``s_n (omega . n)`` as a stand-in for pivoting friction.

Energy: the two-body kinetic energy changes by
``1/2 m_n (u_n'^2 - u_n^2) + 1/2 m_t (|u_t'|^2 - |u_t|^2) - 1/2 I (1 - s_n^2) (omega . n)^2``,
which is never positive for ``e, e_t, s_n`` in [0, 1] and ``mu >= 0``.

References: R. M. Brach, "Rigid body collisions", J. Appl. Mech. 56 (1989)
133-138, and "Mechanical Impact Dynamics" (Wiley, 1991); R. Garwin,
"Kinematics of an ultraelastic rough ball", Am. J. Phys. 37 (1969) 88-92;
R. Cross, "Grip-slip behavior of a bouncing ball", Am. J. Phys. 70 (2002)
1093-1102.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.physics.params import BallParams, ContactMaterial

Regime = Literal["slip", "stick", "catch", "separating"]
"""``slip``: friction saturated throughout. ``stick``: the tangential target was
reached inside the friction cone. ``catch``: a compliant surface (the net mesh)
caught the ball. ``separating``: the contact was not approaching, so no impulse
was applied."""


@dataclass(frozen=True)
class ImpactResult:
    """
    Outcome of one impact.

    :param velocity: Ball centre velocity after impact, m/s.
    :param spin: Ball angular velocity after impact, rad/s.
    :param normal_impulse: Normal impulse on the ball, N s (a vector along ``n``).
    :param tangential_impulse: Friction impulse on the ball, N s (a vector in the plane).
    :param regime: Which branch of the friction law applied.
    :param normal: Unit contact normal used, from the surface into the ball.
    :param approach_speed: ``-u_n`` before impact, m/s (positive when approaching).
    :param separation_speed: ``u_n'`` after impact, m/s.
    :param slip_before: Tangential contact velocity relative to the surface before impact, m/s.
    :param slip_after: Tangential contact velocity relative to the surface after impact, m/s.
    :param restitution: The ``e`` used (after any speed curve).
    :param friction: The ``mu`` used (after any speed curve).
    :param tangential_restitution: The ``e_t`` used.
    :param friction_required: ``|P_stick| / P_n``: the smallest ``mu`` that would stick; 0 with no slip.
    :param surface_velocity_change: Velocity change of the surface's centre of mass, m/s (zero when infinite).
    :param energy_change: Change in two-body kinetic energy, J (never positive in the point-mass model).
    :param surface_angular_velocity_change: Angular velocity change of a rigid-body surface,
        ``-I_s^-1 (r x P)``, rad/s; zero for a point mass.
    :param surface_effective_mass: The surface mass the ball meets along the normal, kg:
        ``1/(1/M + (r x n)^T I_s^-1 (r x n))`` for a rigid body, ``M`` for a point mass.
    """

    velocity: NDArray[np.float64]
    spin: NDArray[np.float64]
    normal_impulse: NDArray[np.float64]
    tangential_impulse: NDArray[np.float64]
    regime: Regime
    normal: NDArray[np.float64]
    approach_speed: float
    separation_speed: float
    slip_before: NDArray[np.float64]
    slip_after: NDArray[np.float64]
    restitution: float
    friction: float
    tangential_restitution: float
    friction_required: float
    surface_velocity_change: NDArray[np.float64]
    energy_change: float
    surface_angular_velocity_change: NDArray[np.float64]
    surface_effective_mass: float

    @property
    def impulse(self) -> NDArray[np.float64]:
        """Total impulse on the ball, N s."""

        return self.normal_impulse + self.tangential_impulse


def _vec(value: ArrayLike) -> NDArray[np.float64]:
    out = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"non-finite vector {out}")
    return out


def contact_velocity(velocity: ArrayLike, spin: ArrayLike, normal: ArrayLike, radius: float) -> NDArray[np.float64]:
    """
    Velocity of the ball's material point touching a surface with unit normal ``normal``.

    :param velocity: Ball centre velocity, m/s.
    :param spin: Ball angular velocity, rad/s.
    :param normal: Unit normal from the surface into the ball.
    :param radius: Ball radius, m.
    :return: ``v + omega x (-R n)``, m/s.
    """

    return _vec(velocity) + np.cross(_vec(spin), -radius * _vec(normal))


def point_effective_mass(mass: float, inertia: ArrayLike, offset: ArrayLike, direction: ArrayLike) -> float:
    """
    Effective mass of a free rigid body at a point, along a direction.

    ``1 / M_eff = 1/mass + (r x d)^T I^-1 (r x d)``. For a paddle with ``d`` the
    face normal this is how much of the paddle an off-centre impact meets.

    :param mass: Body mass, kg (``math.inf`` gives the rotational term alone).
    :param inertia: Inertia tensor about the centre of mass, kg m^2: 3x3, in the frame of ``offset`` and ``direction``.
    :param offset: Point relative to the centre of mass, m.
    :param direction: Direction (normalised here).
    :return: Effective mass, kg.
    """

    d = _vec(direction)
    d = d / np.linalg.norm(d)
    arm = np.cross(_vec(offset), d)
    tensor = np.asarray(inertia, dtype=float).reshape(3, 3)
    inverse = (0.0 if math.isinf(mass) else 1.0 / mass) + float(arm @ np.linalg.solve(tensor, arm))
    return math.inf if inverse == 0.0 else 1.0 / inverse


def _skew(r: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def resolve_impact(
    velocity: ArrayLike,
    spin: ArrayLike,
    ball: BallParams,
    normal: ArrayLike,
    material: ContactMaterial,
    *,
    surface_velocity: ArrayLike = (0.0, 0.0, 0.0),
    effective_mass: float = math.inf,
    contact_offset: ArrayLike | None = None,
    surface_inertia: ArrayLike | None = None,
) -> ImpactResult:
    """
    Resolve one rigid-ball impact against a flat surface.

    :param velocity: Ball centre velocity before impact, m/s.
    :param spin: Ball angular velocity before impact, rad/s.
    :param ball: Ball radius, mass and inertia.
    :param normal: Contact normal from the surface into the ball (normalised here).
    :param material: Restitution, friction and tangential restitution of the pair.
        Speed curves are evaluated at the normal approach speed.
    :param surface_velocity: Velocity of the surface's material point at the contact, m/s.
        For a paddle this is ``v_p + omega_p x (q - p)``.
    :param effective_mass: Surface mass, kg; ``math.inf`` for a fixed or perfectly driven surface.
    :param contact_offset: Contact point relative to the surface body's centre of mass, m,
        in the frame of ``normal``. With ``surface_inertia`` it makes the surface a free
        rigid body (see the module notes); without it the surface is a point mass.
    :param surface_inertia: The surface body's inertia tensor about its centre of mass, kg m^2,
        3x3 in the frame of ``normal``. Required with ``contact_offset``.
    :return: The post-impact state and the impulse breakdown. A contact that is not
        approaching returns the input state unchanged with regime ``separating``.
    """

    if not effective_mass > 0.0:
        raise ValueError(f"effective mass must be positive, got {effective_mass}")
    if (contact_offset is None) != (surface_inertia is None):
        raise ValueError("contact_offset and surface_inertia go together")
    if material.law == "catch" and (contact_offset is not None or not math.isinf(effective_mass)):
        raise ValueError("the catch law treats the surface as immovable: no finite mass or contact offset")
    v = _vec(velocity)
    w = _vec(spin)
    n = _vec(normal)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        raise ValueError("contact normal has zero length")
    n = n / n_norm
    u_s = _vec(surface_velocity)

    m = ball.mass
    radius = ball.radius
    inertia = ball.inertia
    k = ball.inertia_ratio
    inv_m_surface = 0.0 if math.isinf(effective_mass) else 1.0 / effective_mass

    rigid = contact_offset is not None and surface_inertia is not None
    offset = _vec(contact_offset) if contact_offset is not None else np.zeros(3)
    tensor = np.zeros((3, 3))
    inv_tensor = np.zeros((3, 3))
    if surface_inertia is not None:
        tensor = np.asarray(surface_inertia, dtype=float).reshape(3, 3)
        if not (np.all(np.isfinite(tensor)) and np.allclose(tensor, tensor.T, rtol=1e-9, atol=1e-15)):
            raise ValueError("surface inertia must be a finite symmetric 3x3 tensor")
        if np.any(np.linalg.eigvalsh(tensor) <= 0.0):
            raise ValueError("surface inertia must be positive definite")
        inv_tensor = np.linalg.inv(tensor)
        if math.isinf(effective_mass):
            raise ValueError("a rigid-body surface needs a finite effective mass")

    # Contact-point compliance (velocity change per unit impulse): the ball's, and the surface's
    # K_s = (1/M) I + [r]x^T I_s^-1 [r]x, which is isotropic (the point-mass model) when r = 0.
    skew = _skew(offset)
    k_surface = inv_m_surface * np.eye(3) + skew.T @ inv_tensor @ skew
    k_ball = np.eye(3) / m + (np.eye(3) - np.outer(n, n)) / (k * m)
    k_total = k_ball + k_surface
    surface_mass_normal = point_effective_mass(effective_mass, tensor, offset, n) if rigid else effective_mass

    r_c = -radius * n
    u = v + np.cross(w, r_c) - u_s
    u_n = float(u @ n)
    u_t = u - u_n * n
    slip_speed = float(np.linalg.norm(u_t))
    approach = -u_n
    e = material.restitution_at(max(approach, 0.0))
    mu = material.friction_at(max(approach, 0.0))
    e_t = material.tangential_restitution
    zero = np.zeros(3)

    if u_n >= 0.0:
        return ImpactResult(
            velocity=v,
            spin=w,
            normal_impulse=zero,
            tangential_impulse=zero.copy(),
            regime="separating",
            normal=n,
            approach_speed=approach,
            separation_speed=u_n,
            slip_before=u_t,
            slip_after=u_t.copy(),
            restitution=e,
            friction=mu,
            tangential_restitution=e_t,
            friction_required=0.0,
            surface_velocity_change=zero.copy(),
            energy_change=0.0,
            surface_angular_velocity_change=zero.copy(),
            surface_effective_mass=surface_mass_normal,
        )

    if material.law == "catch":
        relative = v - u_s
        normal_part = float(relative @ n)
        tangential = relative - normal_part * n
        v_out = u_s - e * normal_part * n + (1.0 - material.tangential_damping) * tangential
        w_out = (1.0 - material.spin_damping) * w
        impulse = m * (v_out - v)
        p_n_catch = float(impulse @ n)
        u_out = v_out + np.cross(w_out, r_c) - u_s
        u_n_out = float(u_out @ n)
        relative_out = v_out - u_s
        energy_change = 0.5 * m * (float(relative_out @ relative_out) - float(relative @ relative)) + 0.5 * inertia * (float(w_out @ w_out) - float(w @ w))
        return ImpactResult(
            velocity=v_out,
            spin=w_out,
            normal_impulse=p_n_catch * n,
            tangential_impulse=impulse - p_n_catch * n,
            regime="catch",
            normal=n,
            approach_speed=approach,
            separation_speed=u_n_out,
            slip_before=u_t,
            slip_after=u_out - u_n_out * n,
            restitution=e,
            friction=mu,
            tangential_restitution=e_t,
            friction_required=0.0,
            surface_velocity_change=zero.copy(),
            energy_change=energy_change,
            surface_angular_velocity_change=zero.copy(),
            surface_effective_mass=math.inf,
        )

    regime: Regime
    if not rigid:
        # Point-mass surface: the normal and tangential problems decouple and solve in closed form.
        m_n = 1.0 / (1.0 / m + inv_m_surface)
        m_t = 1.0 / (1.0 / m + inv_m_surface + 1.0 / (k * m))
        p_n = -(1.0 + e) * m_n * u_n
        p_stick_mag = (1.0 + e_t) * m_t * slip_speed
        friction_required = p_stick_mag / p_n if p_n > 0.0 else 0.0
        if p_stick_mag <= mu * p_n:
            p_t = -(1.0 + e_t) * m_t * u_t
            regime = "stick"
        else:
            p_t = -mu * p_n * (u_t / slip_speed)
            regime = "slip"
    else:
        # Rigid-body surface: an off-centre offset couples the directions, so solve with the full
        # compliance. Stick: the impulse that lands u_n' = -e u_n and u_t' = -e_t u_t exactly. Slip:
        # friction opposes the initial slip (Brach's algebraic model) and P_n lands u_n' = -e u_n.
        target = -(1.0 + e) * u_n * n - (1.0 + e_t) * u_t
        p_stick = np.linalg.solve(k_total, target)
        p_stick_n = float(p_stick @ n)
        p_stick_t = p_stick - p_stick_n * n
        p_stick_t_mag = float(np.linalg.norm(p_stick_t))
        friction_required = p_stick_t_mag / p_stick_n if p_stick_n > 0.0 else math.inf
        if p_stick_n > 0.0 and p_stick_t_mag <= mu * p_stick_n:
            p_n, p_t = p_stick_n, p_stick_t
            regime = "stick"
        else:
            if slip_speed > 0.0:
                slip_dir = u_t / slip_speed
            elif p_stick_t_mag > 0.0:
                slip_dir = -p_stick_t / p_stick_t_mag
            else:
                slip_dir = np.zeros(3)
            direction = n - mu * slip_dir
            denominator = float(n @ k_total @ direction)
            if denominator <= 0.0:
                raise ValueError("friction and the off-centre coupling admit no consistent sliding impulse (Painleve case)")
            p_n = -(1.0 + e) * u_n / denominator
            p_t = -mu * p_n * slip_dir
            regime = "slip"

    impulse = p_n * n + p_t
    v_out = v + impulse / m
    # r_c is parallel to n, so only the friction impulse carries a moment, and none of it about n.
    w_out = w + np.cross(r_c, p_t) / inertia
    s_n = material.normal_spin_retention
    w_normal = float(w @ n)
    w_out = w_out - (1.0 - s_n) * w_normal * n
    surface_dv = -impulse * inv_m_surface
    surface_dw = -inv_tensor @ np.cross(offset, impulse)
    u_s_out = u_s + surface_dv + np.cross(surface_dw, offset)
    u_out = v_out + np.cross(w_out, r_c) - u_s_out
    u_n_out = float(u_out @ n)
    u_t_out = u_out - u_n_out * n
    # Work of a contact impulse on two bodies is P . (u + u') / 2; the normal-spin damping is extra.
    energy_change = 0.5 * float(impulse @ (u + u_out)) - 0.5 * inertia * (1.0 - s_n**2) * w_normal**2

    return ImpactResult(
        velocity=v_out,
        spin=w_out,
        normal_impulse=p_n * n,
        tangential_impulse=p_t,
        regime=regime,
        normal=n,
        approach_speed=approach,
        separation_speed=u_n_out,
        slip_before=u_t,
        slip_after=u_t_out,
        restitution=e,
        friction=mu,
        tangential_restitution=e_t,
        friction_required=friction_required,
        surface_velocity_change=surface_dv,
        energy_change=energy_change,
        surface_angular_velocity_change=surface_dw,
        surface_effective_mass=surface_mass_normal,
    )
