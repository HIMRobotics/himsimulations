"""
The paddle impact inverted: the face velocity that sends a ball off with a wanted velocity.

:func:`lib.physics.reference.impact.resolve_impact` maps an incoming ball and a
face (normal, velocity at the contact point, effective mass, and optionally the
paddle as a free rigid body) to an outgoing ball. Inside the friction cone (the
``stick`` regime) that map is linear in the relative contact velocity, so for a
given normal it inverts in closed form:

- the impulse is fixed by the wanted change, ``P = m (v_out - v_in)``;
- with the contact compliance ``K`` (the ball's ``1/m`` and ``1/(k m)`` across
  the normal, plus the surface's ``(1/M) 1 + [r]x^T I_s^-1 [r]x``), the stick law
  ``K P = -(1 + e) u_n n - (1 + e_t) u_t`` gives the relative contact velocity
  ``u``: its normal part ``u_n = -a`` solves ``a (1 + e(a)) = (K P) . n`` (one
  monotone root when ``e`` falls with speed), its tangential part is
  ``u_t = -(K P)_t / (1 + e_t)``;
- the face velocity at the contact point is ``u_s = v_c - u``, with ``v_c = v_in +
  omega_in x (-R n)`` the ball's contact-point velocity.

Impulses outside the friction cone (``|P_t| > mu P_n``) cannot be produced: a
sliding impulse lies on the cone. Those, and impulses that do not push along the
normal, have no face velocity (``None``).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.physics.params import BallTableParams, ContactMaterial

Array = NDArray[np.float64]


def _skew(r: Array) -> Array:
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def contact_compliance(
    params: BallTableParams, normal: ArrayLike, *, effective_mass: float, contact_offset: ArrayLike | None = None, surface_inertia: ArrayLike | None = None
) -> Array:
    """
    The contact-point compliance ``K`` (velocity change per unit impulse) of :func:`resolve_impact`, 3x3.

    :param params: Ball mass, radius and inertia.
    :param normal: Unit contact normal.
    :param effective_mass: Surface mass, kg (``math.inf`` for an immovable surface).
    :param contact_offset: Contact point from the surface body's centre of mass, m (with ``surface_inertia``).
    :param surface_inertia: The surface body's inertia tensor, kg m^2, in the frame of ``normal``.
    """

    if (contact_offset is None) != (surface_inertia is None):
        raise ValueError("contact_offset and surface_inertia go together")
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    ball = params.ball
    k_ball = np.eye(3) / ball.mass + (np.eye(3) - np.outer(n, n)) / (ball.inertia_ratio * ball.mass)
    k_surface = (0.0 if math.isinf(effective_mass) else 1.0 / effective_mass) * np.eye(3)
    if contact_offset is not None and surface_inertia is not None:
        skew = _skew(np.asarray(contact_offset, dtype=np.float64).reshape(3))
        k_surface = k_surface + skew.T @ np.linalg.inv(np.asarray(surface_inertia, dtype=np.float64).reshape(3, 3)) @ skew
    return k_ball + k_surface


def _approach_speed(target: float, material: ContactMaterial) -> float:
    """The approach speed ``a`` with ``a (1 + e(a)) = target`` (bisection to machine precision; ``e`` in [0, 1])."""

    lo, hi = 0.5 * target, target
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if mid * (1.0 + material.restitution_at(mid)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paddle_velocity_for(
    params: BallTableParams,
    v_in: ArrayLike,
    spin_in: ArrayLike,
    v_out: ArrayLike,
    normal: ArrayLike,
    *,
    effective_mass: float | None = None,
    contact_offset: ArrayLike | None = None,
    surface_inertia: ArrayLike | None = None,
    material: ContactMaterial | None = None,
) -> Array | None:
    """
    The face velocity at the contact point that turns ``v_in`` into ``v_out`` through the reference impact law.

    :param params: The physics (ball, the rubber law and the paddle's effective mass by default).
    :param v_in: Ball velocity before the hit, m/s.
    :param spin_in: Ball spin before the hit, rad/s.
    :param v_out: Wanted ball velocity after the hit, m/s.
    :param normal: Face normal from the face into the ball (normalised here).
    :param effective_mass: Paddle effective mass, kg (default ``params.paddle.effective_mass``).
    :param contact_offset: Contact point from the blade centre, m: the paddle as a free rigid body (the game's
        hybrid paddle). With ``surface_inertia``; both ``None`` is the point-mass paddle.
    :param surface_inertia: The paddle's inertia tensor in the table frame, kg m^2 (``R diag(I) R^T``).
    :param material: The impact law (default ``params.rubber_contact``); the ``catch`` law is not invertible here.
    :return: The face velocity ``u_s``, m/s (pass it to ``resolve_impact`` as ``surface_velocity``), or ``None``
        when the impulse does not push the ball out along the normal or lies outside the friction cone.
    """

    law = material if material is not None else params.rubber_contact
    if law.law != "rigid":
        raise ValueError("only the rigid impact law can be inverted for a face velocity")
    mass = params.paddle.effective_mass if effective_mass is None else effective_mass
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    v0 = np.asarray(v_in, dtype=np.float64).reshape(3)
    w0 = np.asarray(spin_in, dtype=np.float64).reshape(3)
    impulse = params.ball.mass * (np.asarray(v_out, dtype=np.float64).reshape(3) - v0)
    p_n = float(impulse @ n)
    if p_n <= 0.0:
        return None
    target = contact_compliance(params, n, effective_mass=mass, contact_offset=contact_offset, surface_inertia=surface_inertia) @ impulse
    target_n = float(target @ n)
    if target_n <= 0.0:
        return None
    approach = _approach_speed(target_n, law)
    p_t = impulse - p_n * n
    if float(np.linalg.norm(p_t)) > law.friction_at(approach) * p_n * (1.0 + 1e-12):
        return None
    u_t = -(target - target_n * n) / (1.0 + law.tangential_restitution)
    relative = -approach * n + u_t
    contact_point_velocity = v0 + np.cross(w0, -params.ball.radius * n)
    return contact_point_velocity - relative


__all__ = ["contact_compliance", "paddle_velocity_for"]
