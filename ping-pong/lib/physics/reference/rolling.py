"""
Sustained rolling on a flat horizontal surface: the floor or the playing surface.

The impact law describes bounces. A ball that rebounds from a flat surface
slower than ``rest_speed`` is in sustained contact, and :func:`simulate`
hands it to this regime instead of stopping. The ball stays on the plane
(``z`` fixed at the surface plus ``R``, ``v_z = 0``), and the normal force
per unit mass is ``N/m = g - a_air,z``, so Magnus lift or downforce is
carried. There are two phases.

**Sliding** (contact slip ``u_t = v + omega x (-R n)`` not yet zero): Coulomb
friction ``F = -mu N u_t/|u_t|`` acts at the contact point, plus the air force
and torque of :class:`lib.physics.reference.flight.FlightModel`. The slip
falls towards zero; within a few integration steps of zero sliding ends (an
event), and the rigid stick impulse removes the residual slip. That never adds
energy and, because friction acts at the contact point, does not change the
rolling state reached.

**Rolling** (no slip): the surface's ``rolling_resistance`` ``C_rr`` gives a
resisting couple ``C_rr N R`` against the rolling. With the air force ``F_air``
and torque ``tau_air`` the centre obeys::

    m (1 + k) dv/dt = F_air,h - C_rr N v_hat + (tau_air,h x n) / R
    domega_h/dt = (n x dv/dt) / R,   domega_n/dt = tau_air,n / I

where ``k = I / (m R^2)``. With no air, the ball decelerates at
``C_rr g / (1 + k)`` and stops after ``v0^2 (1 + k) / (2 C_rr g)``. It has
stopped (an event) below ``stop_speed``.

On the table, the regime ends when the centre leaves the playing surface:
the ball returns to flight and goes over the board's edge. Any contact with
another surface (a wall, a leg, the net) also returns it to flight through
the impact law.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from lib.physics.params import BallParams, ContactMaterial
from lib.physics.reference.flight import FlightModel

SLIP_EPSILON_MPS = 1e-6
"""Contact slip below which sliding has ended, m/s."""

STOP_SPEED_MPS = 1e-3
"""Centre speed below which a rolling ball has stopped, m/s."""


@dataclass(frozen=True)
class RollingSurface:
    """
    A flat, horizontal surface a ball can roll on.

    :param kind: ``floor`` or ``table``.
    :param name: Its name on events.
    :param height: The surface's ``z`` in the table frame, m.
    :param material: Its law: ``friction`` for sliding, ``rolling_resistance`` for rolling.
    :param half_length: Half extent along X, m; ``math.inf`` for the floor.
    :param half_width: Half extent along Y, m; ``math.inf`` for the floor.
    """

    kind: str
    name: str
    height: float
    material: ContactMaterial
    half_length: float = math.inf
    half_width: float = math.inf


@dataclass(frozen=True)
class RollingModel:
    """
    The derivative of a ball on a :class:`RollingSurface`, sliding or rolling.

    :param flight: The air forces and torques (gravity is replaced by the support).
    :param ball: Ball radius, mass and inertia.
    :param surface: The surface.
    :param gravity: Gravitational acceleration, m/s^2.
    :param sliding: True while the contact slips.
    """

    flight: FlightModel
    ball: BallParams
    surface: RollingSurface
    gravity: float
    sliding: bool

    def derivative(self, state: Sequence[float]) -> list[float]:
        """Time derivative of a 9-float state on the surface (``dz/dt = dv_z/dt = 0``)."""

        free = self.flight.derivative(state)
        ax_air, ay_air, az_air = free[3], free[4], free[5] + self.gravity
        dwx_air, dwy_air, dwz_air = free[6], free[7], free[8]
        support = max(self.gravity - az_air, 0.0)
        radius = self.ball.radius
        k = self.ball.inertia_ratio
        vx, vy = state[3], state[4]
        wx, wy = state[6], state[7]
        if self.sliding:
            ux, uy = vx - radius * wy, vy + radius * wx
            slip = math.hypot(ux, uy)
            mu = self.surface.material.friction_at(0.0)
            fx, fy = (-mu * support * ux / slip, -mu * support * uy / slip) if slip > 0.0 else (0.0, 0.0)
            # The friction force acts at r_c = -R n: its moment about the centre is (R F_y, -R F_x, 0).
            return [vx, vy, 0.0, ax_air + fx, ay_air + fy, 0.0, dwx_air + fy / (k * radius), dwy_air - fx / (k * radius), dwz_air]
        speed = math.hypot(vx, vy)
        resist = self.surface.material.rolling_resistance * support
        hx, hy = (vx / speed, vy / speed) if speed > 0.0 else (0.0, 0.0)
        ax = (ax_air - resist * hx + k * radius * dwy_air) / (1.0 + k)
        ay = (ay_air - resist * hy - k * radius * dwx_air) / (1.0 + k)
        return [vx, vy, 0.0, ax, ay, 0.0, -ay / radius, ax / radius, dwz_air]


def contact_slip(state: Sequence[float], radius: float) -> tuple[float, float]:
    """The horizontal velocity of the ball's lowest point, m/s."""

    return state[3] - radius * state[7], state[4] + radius * state[6]


def stick(state: Sequence[float], ball: BallParams) -> list[float]:
    """
    Remove any slip at the lowest point with the rigid stick impulse ``-m_t u_t``.

    The kinetic energy changes by ``-1/2 m_t |u_t|^2``, never positive.
    """

    radius, k = ball.radius, ball.inertia_ratio
    ux, uy = contact_slip(state, radius)
    out = list(state)
    out[3] -= k / (1.0 + k) * ux
    out[4] -= k / (1.0 + k) * uy
    out[6] -= uy / ((1.0 + k) * radius)
    out[7] += ux / ((1.0 + k) * radius)
    return out


def settle_on(state: Sequence[float], surface: RollingSurface, ball: BallParams) -> list[float]:
    """Put a ball that rebounded slower than ``rest_speed`` on the surface: ``z`` on it, ``v_z = 0`` (energy never rises)."""

    out = list(state)
    out[2] = surface.height + ball.radius
    out[5] = 0.0
    return out


@dataclass(frozen=True)
class SlipEnds:
    """
    Event: sliding has ended.

    Kinetic friction reduces the slip along its own direction at about
    ``mu g (1 + 1/k)``. Near zero, a fixed-step integrator chatters across the
    friction's discontinuity, so sliding is ended ``margin`` before zero (a
    few steps' worth of slip). The stick impulse then removes the rest.
    Friction acts at the contact point, so the ball's angular momentum about
    that point is conserved through sliding and through the stick impulse:
    the rolling state reached does not depend on when the stick is applied.

    :param radius: Ball radius, m.
    :param direction: Unit slip direction when sliding began, ``(x, y)``.
    :param margin: Slip along that direction at which sliding is ended, m/s.
    """

    radius: float
    direction: tuple[float, float]
    margin: float

    @classmethod
    def starting(cls, state: Sequence[float], radius: float, margin: float) -> SlipEnds:
        """The event for a ball sliding from ``state``."""

        ux, uy = contact_slip(state, radius)
        slip = math.hypot(ux, uy)
        return cls(radius, (ux / slip, uy / slip), margin)

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Slip along the initial slip direction above the margin, m/s."""

        ux, uy = contact_slip(state, self.radius)
        return ux * self.direction[0] + uy * self.direction[1] - self.margin


@dataclass(frozen=True)
class Stops:
    """Event: the centre speed has fallen below :data:`STOP_SPEED_MPS`."""

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Centre speed above the threshold, m/s."""

        return math.hypot(state[3], state[4]) - STOP_SPEED_MPS


@dataclass(frozen=True)
class LeavesSurface:
    """Event: the centre has rolled off the surface's edge."""

    half_length: float
    half_width: float

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Distance from the centre to the nearest edge, m (negative once off)."""

        return min(self.half_length - abs(state[0]), self.half_width - abs(state[1]))

    def clearance(self, state: Sequence[float], gap: float) -> float:
        """How far the centre can move without the gap changing sign, m (1-Lipschitz)."""

        return abs(gap)


def pivot_off_edge(
    state: Sequence[float], surface: RollingSurface, ball: BallParams, gravity: float, *, step: float = 2e-5
) -> tuple[float, list[float]] | None:
    """
    Roll a slow ball over the surface's edge until it leaves it, with Coulomb friction on the edge.

    A ball rolling off an edge slower than ``sqrt(g R)`` (across the edge)
    pivots about it before leaving. Its centre turns on a circle of radius
    ``R`` about the edge line; ``theta`` is measured from the vertical, and
    ``Omega`` is its spin about the edge's direction (rolling over the edge
    is ``Omega = theta'``). With ``N = m (g cos(theta) - R theta'^2)``, the
    friction ``F`` on the edge obeys::

        m R theta'' = m g sin(theta) - F,      I Omega' = F R

    It sticks (``F = k m g sin(theta) / (1 + k)``) while that needs less
    than ``mu N``, and otherwise slides (``F = mu N``). The ball leaves when
    ``N`` reaches zero. Without slip that happens at
    ``cos(theta_r) = ((1 + k) v^2 / (g R) + 2) / (3 + k)`` (``10/17`` for a
    solid sphere at rest); the friction cone usually makes it slide off
    sooner. The phase is integrated with classical RK4 at ``step``; friction
    never adds energy. The motion along the edge carries on unchanged. Air
    and rolling resistance are left out of this short phase.

    :param state: The ball as its centre crosses the edge line (on the surface, rolling).
    :param surface: The bounded surface it is leaving; its ``friction`` acts on the edge.
    :param ball: Ball parameters.
    :param gravity: m/s^2.
    :param step: Integration step, s.
    :return: ``(pivot duration s, state at release)``, or ``None`` when the ball is fast enough to leave at once.
    """

    radius, k = ball.radius, ball.inertia_ratio
    x, y = state[0], state[1]
    if surface.half_length - abs(x) <= surface.half_width - abs(y):
        out = (math.copysign(1.0, x), 0.0)
        edge = (math.copysign(surface.half_length, x), y)
    else:
        out = (0.0, math.copysign(1.0, y))
        edge = (x, math.copysign(surface.half_width, y))
    vx, vy = state[3], state[4]
    across = vx * out[0] + vy * out[1]
    if across <= 0.0 or across * across >= gravity * radius:
        return None
    along = (vx - across * out[0], vy - across * out[1])
    mu = surface.material.friction_at(0.0)

    def support(theta: float, rate: float) -> float:
        return gravity * math.cos(theta) - radius * rate * rate

    def derivative(y: tuple[float, float, float]) -> tuple[float, float, float]:
        theta, rate, spin = y
        normal = max(support(theta, rate), 0.0)
        needed = k * gravity * math.sin(theta) / (1.0 + k)
        slip = rate - spin
        if abs(slip) < 1e-9 and needed <= mu * normal:
            friction = needed
        else:
            friction = math.copysign(mu * normal, slip if abs(slip) >= 1e-9 else 1.0)
        return rate, (gravity * math.sin(theta) - friction) / radius, friction / (k * radius)

    def rk4(y: tuple[float, float, float], h: float) -> tuple[float, float, float]:
        k1 = derivative(y)
        k2 = derivative((y[0] + 0.5 * h * k1[0], y[1] + 0.5 * h * k1[1], y[2] + 0.5 * h * k1[2]))
        k3 = derivative((y[0] + 0.5 * h * k2[0], y[1] + 0.5 * h * k2[1], y[2] + 0.5 * h * k2[2]))
        k4 = derivative((y[0] + h * k3[0], y[1] + h * k3[1], y[2] + h * k3[2]))
        out = tuple(y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(3))
        nxt = (out[0], out[1], out[2])
        # Friction cannot reverse the slip within a step: a slip that crosses zero sticks, keeping the angular
        # momentum about the edge (on which the friction has no moment), m R^2 theta' + I Omega.
        if (y[1] - y[2]) * (nxt[1] - nxt[2]) < 0.0:
            common = (nxt[1] + k * nxt[2]) / (1.0 + k)
            nxt = (nxt[0], common, common)
        return nxt

    y = (0.0, across / radius, across / radius)
    elapsed = 0.0
    while elapsed < 10.0:
        nxt = rk4(y, step)
        n0, n1 = support(y[0], y[1]), support(nxt[0], nxt[1])
        if n1 <= 0.0:
            # Leave where the support vanishes: re-take the step up to the interpolated crossing.
            fraction = n0 / (n0 - n1) if n0 > n1 else 1.0
            y = rk4(y, fraction * step)
            elapsed += fraction * step
            break
        y = nxt
        elapsed += step
    theta, rate, spin = y
    # The release point is nudged 1e-9 R outward horizontally so the board's edge sees the ball clear of it.
    reach = radius * math.sin(theta) * (1.0 + 1e-9)
    tangential = rate * radius
    return elapsed, [
        edge[0] + reach * out[0] + along[0] * elapsed,
        edge[1] + reach * out[1] + along[1] * elapsed,
        surface.height + radius * math.cos(theta),
        tangential * math.cos(theta) * out[0] + along[0],
        tangential * math.cos(theta) * out[1] + along[1],
        -tangential * math.sin(theta),
        -spin * out[1] - along[1] / radius,
        spin * out[0] + along[0] / radius,
        state[8],
    ]
