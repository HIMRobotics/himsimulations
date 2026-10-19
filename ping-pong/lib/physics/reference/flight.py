"""
Ball flight in still air: gravity, quadratic drag, Magnus lift and spin decay.

State vector (9 floats, table frame, SI): position ``p`` (m), velocity ``v``
(m/s), angular velocity ``omega`` (rad/s). With ``A = pi R^2`` and
``k = rho A / (2 m)`` (1/m)::

    dp/dt     = v
    dv/dt     = -g z_hat
                - k C_D(S) |v| v
                + k C_L(S) |v|^2 (omega_perp x v_hat) / |omega_perp|
    domega/dt = -(C_tau rho A R^2 / (2 I)) |v| omega

where ``omega_perp = omega - (omega . v_hat) v_hat`` is the spin component
perpendicular to the velocity, ``S = R |omega_perp| / |v|`` is the spin
parameter, ``C_D(S) = C_D0 + (dC_D/dS) S`` and ``C_L(S)`` is
:class:`lib.physics.params.LiftModel`.

So the Magnus force is ``F_M = 1/2 rho A C_L(S) |v|^2 e_M`` with the unit
direction ``e_M = omega_perp x v_hat / |omega_perp|`` (equal to
``omega_hat x v_hat`` when spin is perpendicular to velocity). Using the
perpendicular spin means spin about the direction of travel neither lifts nor
raises ``S``; for a linear ``C_L = C_M S`` it reduces exactly to the
table-tennis-robotics form ``F_M = 1/2 rho A R C_M (omega x v)``. Topspin on a
ball moving along +X is ``omega`` along +Y, and ``omega x v`` then points down:
the ball dips.

This lift model is the source of truth for Magnus. Engine-native fluid
models use other conventions: MuJoCo's ``F = C_M rho V (omega x v)`` is
``C_L = (8/3) C_M S``, so its default ``C_M = 1`` is 2.7 to 10 times stronger
than measured table-tennis balls. An adapter switches those off and applies
:meth:`FlightModel.wrench`.

The integrator is classical fourth-order Runge-Kutta at a fixed step (default
1e-4 s, 10 kHz). Events are signed gap functions (see
:mod:`lib.physics.reference.surfaces`): a gap that goes from ``>= 0`` to
``< 0`` across a step is bracketed by bisection on the step length, re-taking a
single RK4 step from the step start for each trial, so the event state is
exactly what the integrator would produce at that time. The returned event
state is the last trial still on the non-negative side.

Two implementations of the derivative exist: a scalar one on Python floats
(about 3 us per RK4 step), used by the event integrator and the trajectory
fits, and a vectorised numpy one (about 120 us per step whatever the batch
size, so it only pays above a few dozen balls) for large batches such as
parameter sweeps. A test holds them equal.

**Speed without changing answers.** :meth:`FlightModel.rk4_step` is the RK4
step with the derivative inlined: the same floating-point operations in the
same order as :func:`rk4_step`, so bit for bit the same state, about twice as
fast. :func:`integrate_until_event` skips evaluating a *static* surface's gap
while the ball provably cannot change its sign: a static gap is 1-Lipschitz in
the ball's position (a distance), so after an evaluation ``g`` its sign at the
following step ends cannot change while the ball's summed step-end travel stays
below ``|g|`` (each surface's :meth:`clearance` says how far). The event logic
only reads the sign at step ends, so the steps, the roots and every state are
bit for bit those of evaluating every gap at every step. The bounding box is
handled the same way.

**Adaptive steps (opt-in, ``max_step``).** Near anything the ball could touch,
steps are ``dt``; a longer step, up to ``max_step``, is taken only when no
watched surface or blade can be reached within it (every gap's clearance exceeds
the ball's reach over the step plus ``clear_margin``). No contact can therefore
start inside a long step: grazing touches are resolved at ``dt``. With
``graze`` (opt-in, with or without ``max_step``) a step that ends clear of a surface but came within a small
distance of it is searched for a dip below zero inside the step (a touch that
enters and leaves between two step ends), which a fixed step misses when the ball
penetrates less than about ``(v dt)^2 / (8 R)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.physics.params import BallTableParams, LiftModel
from lib.physics.reference.surfaces import Floor, Net, NetFixture, TableBoard, TableTop, net_fixtures
from lib.physics.reference.undercarriage import TableFrame

STATE_SIZE = 9
"""Position (3), velocity (3), angular velocity (3)."""

DEFAULT_DT = 1e-4
"""Default flight step, s (10 kHz)."""

StateLike = Sequence[float] | NDArray[np.float64]
"""One 9-float state; numpy rows are converted to Python floats for the scalar path."""


def _floats(state: StateLike) -> list[float]:
    values = [float(x) for x in state]
    if len(values) != STATE_SIZE:
        raise ValueError(f"a state has {STATE_SIZE} values, got {len(values)}")
    return values


class Dynamics(Protocol):
    """Anything the scalar integrator can step: a 9-state derivative (flight, or rolling on a surface)."""

    def derivative(self, state: Sequence[float]) -> list[float]:
        """Time derivative of one 9-float state."""
        ...


class GapEvent(Protocol):
    """Anything the integrator can watch: a signed gap that fires when it turns negative."""

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Signed clearance at time ``t`` (s) for ``state``; any unit."""
        ...


GapFunction = Callable[[float, Sequence[float]], float]


@dataclass(frozen=True)
class FlightModel:
    """
    Flight coefficients precomputed from a parameter set.

    :param gravity: Gravitational acceleration along -Z, m/s^2.
    :param radius: Ball radius, m.
    :param mass: Ball mass, kg.
    :param inertia: Ball moment of inertia, kg m^2.
    :param aero_gain: ``k = rho A / (2 m)``, 1/m.
    :param drag_coefficient: ``C_D0``, dimensionless.
    :param drag_spin_slope: ``dC_D/dS``, dimensionless.
    :param lift: ``C_L(S)``.
    :param spin_decay_gain: ``C_tau rho A R^2 / (2 I)``, 1/m; spin decays at this times ``|v|`` per second.
    """

    gravity: float
    radius: float
    mass: float
    inertia: float
    aero_gain: float
    drag_coefficient: float
    drag_spin_slope: float
    lift: LiftModel
    spin_decay_gain: float

    @classmethod
    def from_params(cls, params: BallTableParams) -> FlightModel:
        """The flight model of ``params``."""

        ball = params.ball
        aero = params.aero
        q = 0.5 * aero.air_density * ball.area
        return cls(
            gravity=aero.gravity,
            radius=ball.radius,
            mass=ball.mass,
            inertia=ball.inertia,
            aero_gain=q / ball.mass,
            drag_coefficient=aero.drag_coefficient,
            drag_spin_slope=aero.drag_spin_slope,
            lift=aero.lift,
            spin_decay_gain=aero.spin_decay_coefficient * q * ball.radius**2 / ball.inertia,
        )

    def derivative(self, state: Sequence[float]) -> list[float]:
        """
        Time derivative of one state (scalar fast path).

        :param state: 9 floats: position (m), velocity (m/s), angular velocity (rad/s).
        :return: 9 floats: velocity (m/s), acceleration (m/s^2), angular acceleration (rad/s^2).
        """

        vx, vy, vz = state[3], state[4], state[5]
        wx, wy, wz = state[6], state[7], state[8]
        ax = 0.0
        ay = 0.0
        az = -self.gravity
        dwx = 0.0
        dwy = 0.0
        dwz = 0.0
        v2 = vx * vx + vy * vy + vz * vz
        if v2 > 0.0:
            speed = math.sqrt(v2)
            wv = wx * vx + wy * vy + wz * vz
            wperp2 = wx * wx + wy * wy + wz * wz - wv * wv / v2
            wperp = math.sqrt(wperp2) if wperp2 > 0.0 else 0.0
            s = self.radius * wperp / speed
            k = self.aero_gain
            drag = k * (self.drag_coefficient + self.drag_spin_slope * s) * speed
            ax -= drag * vx
            ay -= drag * vy
            az -= drag * vz
            if wperp > 0.0:
                magnus = k * self.lift.coefficient(s) * speed / wperp
                ax += magnus * (wy * vz - wz * vy)
                ay += magnus * (wz * vx - wx * vz)
                az += magnus * (wx * vy - wy * vx)
            decay = self.spin_decay_gain * speed
            dwx = -decay * wx
            dwy = -decay * wy
            dwz = -decay * wz
        return [vx, vy, vz, ax, ay, az, dwx, dwy, dwz]

    @cached_property
    def _acceleration(self) -> Callable[[float, float, float, float, float, float], tuple[float, float, float, float, float, float]]:
        """
        :meth:`derivative`'s acceleration and angular acceleration as a closure over the coefficients.

        The expressions and their evaluation order are :meth:`derivative`'s (and :meth:`LiftModel.coefficient`'s),
        so every result is bit for bit the same; only attribute lookups and list building are saved.
        """

        gravity = self.gravity
        radius = self.radius
        gain = self.aero_gain
        drag0 = self.drag_coefficient
        drag_slope = self.drag_spin_slope
        decay_gain = self.spin_decay_gain
        form = self.lift.form
        slope = self.lift.slope
        c_max = self.lift.c_max
        sqrt = math.sqrt

        def acceleration(vx: float, vy: float, vz: float, wx: float, wy: float, wz: float) -> tuple[float, float, float, float, float, float]:
            v2 = vx * vx + vy * vy + vz * vz
            if not v2 > 0.0:
                return 0.0, 0.0, -gravity, 0.0, 0.0, 0.0
            speed = sqrt(v2)
            wv = wx * vx + wy * vy + wz * vz
            wperp2 = wx * wx + wy * wy + wz * wz - wv * wv / v2
            wperp = sqrt(wperp2) if wperp2 > 0.0 else 0.0
            s = radius * wperp / speed
            drag = gain * (drag0 + drag_slope * s) * speed
            ax = 0.0 - drag * vx
            ay = 0.0 - drag * vy
            az = -gravity - drag * vz
            if wperp > 0.0:
                if s <= 0.0:
                    lift = 0.0
                elif form == "rational":
                    lift = slope * s / (1.0 + slope * s / c_max)
                elif form == "linear":
                    lift = min(slope * s, c_max)
                else:
                    lift = c_max
                magnus = gain * lift * speed / wperp
                ax += magnus * (wy * vz - wz * vy)
                ay += magnus * (wz * vx - wx * vz)
                az += magnus * (wx * vy - wy * vx)
            decay = decay_gain * speed
            return ax, ay, az, -decay * wx, -decay * wy, -decay * wz

        return acceleration

    def rk4_step(self, state: Sequence[float], h: float) -> list[float]:
        """
        One classical RK4 step of length ``h`` (s): :func:`rk4_step` for this model, bit for bit, about twice as fast.

        :param state: 9 floats.
        :param h: Step, s (may be negative).
        :return: The state after the step.
        """

        acc = self._acceleration
        x, y, z, vx, vy, vz, wx, wy, wz = state
        half = 0.5 * h
        a1 = acc(vx, vy, vz, wx, wy, wz)
        v2x, v2y, v2z = vx + half * a1[0], vy + half * a1[1], vz + half * a1[2]
        a2 = acc(v2x, v2y, v2z, wx + half * a1[3], wy + half * a1[4], wz + half * a1[5])
        v3x, v3y, v3z = vx + half * a2[0], vy + half * a2[1], vz + half * a2[2]
        a3 = acc(v3x, v3y, v3z, wx + half * a2[3], wy + half * a2[4], wz + half * a2[5])
        v4x, v4y, v4z = vx + h * a3[0], vy + h * a3[1], vz + h * a3[2]
        a4 = acc(v4x, v4y, v4z, wx + h * a3[3], wy + h * a3[4], wz + h * a3[5])
        sixth = h / 6.0
        return [
            x + sixth * (vx + 2.0 * v2x + 2.0 * v3x + v4x),
            y + sixth * (vy + 2.0 * v2y + 2.0 * v3y + v4y),
            z + sixth * (vz + 2.0 * v2z + 2.0 * v3z + v4z),
            vx + sixth * (a1[0] + 2.0 * a2[0] + 2.0 * a3[0] + a4[0]),
            vy + sixth * (a1[1] + 2.0 * a2[1] + 2.0 * a3[1] + a4[1]),
            vz + sixth * (a1[2] + 2.0 * a2[2] + 2.0 * a3[2] + a4[2]),
            wx + sixth * (a1[3] + 2.0 * a2[3] + 2.0 * a3[3] + a4[3]),
            wy + sixth * (a1[4] + 2.0 * a2[4] + 2.0 * a3[4] + a4[4]),
            wz + sixth * (a1[5] + 2.0 * a2[5] + 2.0 * a3[5] + a4[5]),
        ]

    def derivative_array(self, states: ArrayLike) -> NDArray[np.float64]:
        """
        Vectorised :meth:`derivative` over any leading shape.

        :param states: Array of shape ``(..., 9)``.
        :return: Array of shape ``(..., 9)``.
        """

        y = np.asarray(states, dtype=float)
        v = y[..., 3:6]
        w = y[..., 6:9]
        v2 = np.sum(v * v, axis=-1)
        moving = v2 > 0.0
        speed = np.sqrt(v2)
        safe_v2 = np.where(moving, v2, 1.0)
        safe_speed = np.where(moving, speed, 1.0)
        wv = np.sum(w * v, axis=-1)
        wperp2 = np.sum(w * w, axis=-1) - wv * wv / safe_v2
        wperp = np.where(moving & (wperp2 > 0.0), np.sqrt(np.maximum(wperp2, 0.0)), 0.0)
        s = self.radius * wperp / safe_speed
        k = self.aero_gain
        drag = np.where(moving, k * (self.drag_coefficient + self.drag_spin_slope * s) * speed, 0.0)
        acc = -drag[..., None] * v
        spinning = wperp > 0.0
        magnus = np.where(spinning, k * self.lift.coefficient_array(s) * speed / np.where(spinning, wperp, 1.0), 0.0)
        acc = acc + magnus[..., None] * np.cross(w, v)
        acc[..., 2] -= self.gravity
        dw = -(self.spin_decay_gain * speed)[..., None] * w
        return np.concatenate([v, acc, dw], axis=-1)

    def wrench(self, velocity: ArrayLike, spin: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        The aerodynamic force and torque on the ball, excluding gravity.

        This is what an engine adapter applies every physics step so that the
        engine's flight matches this model.

        :param velocity: Ball velocity relative to still air, m/s.
        :param spin: Ball angular velocity, rad/s.
        :return: Force (N) and torque (N m), table frame.
        """

        state = np.concatenate([np.zeros(3), np.asarray(velocity, dtype=float).reshape(3), np.asarray(spin, dtype=float).reshape(3)])
        d = self.derivative_array(state)
        acc = d[3:6].copy()
        acc[2] += self.gravity
        return self.mass * acc, self.inertia * d[6:9]


def rk4_step(model: Dynamics, state: Sequence[float], h: float) -> list[float]:
    """
    One classical RK4 step of length ``h`` (s) for one state (scalar fast path).

    :param model: Flight model.
    :param state: 9 floats.
    :param h: Step, s (may be negative).
    :return: The state after the step.
    """

    if type(model) is FlightModel:
        return model.rk4_step(state, h)
    f = model.derivative
    half = 0.5 * h
    k1 = f(state)
    k2 = f([state[i] + half * k1[i] for i in range(STATE_SIZE)])
    k3 = f([state[i] + half * k2[i] for i in range(STATE_SIZE)])
    k4 = f([state[i] + h * k3[i] for i in range(STATE_SIZE)])
    sixth = h / 6.0
    return [state[i] + sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(STATE_SIZE)]


def rk4_step_array(model: FlightModel, states: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """
    One RK4 step of length ``h`` (s) for a batch of states of shape ``(..., 9)``.
    """

    f = model.derivative_array
    k1 = f(states)
    k2 = f(states + 0.5 * h * k1)
    k3 = f(states + 0.5 * h * k2)
    k4 = f(states + h * k3)
    return states + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass(frozen=True)
class EventHit:
    """
    The first root of a watched gap within a step.

    :param index: Index of the event in the list passed to the integrator.
    :param time: Last bisection time on the non-negative side, s.
    :param state: State at ``time`` (gap ``>= 0``).
    :param time_after: First bisection time on the negative side, s.
    :param state_after: State at ``time_after`` (gap ``< 0``).
    """

    index: int
    time: float
    state: tuple[float, ...]
    time_after: float
    state_after: tuple[float, ...]


@dataclass(frozen=True)
class FlightSegment:
    """
    A stretch of free flight.

    :param times: Sample times after the start, s; the last is the event or end time.
    :param states: States at ``times``.
    :param hit: The event that ended the segment, if any.
    :param stopped: True when the ``stop`` predicate (or the ``bounds``) ended the segment.
    :param steps: Integration steps taken.
    :param coarse_steps: Of those, the steps longer than ``dt`` (adaptive mode).
    :param grazes: Events found inside a step by the graze search (``graze``).
    """

    times: list[float]
    states: list[list[float]]
    hit: EventHit | None
    stopped: bool
    steps: int = 0
    coarse_steps: int = 0
    grazes: int = 0


CLEAR_MARGIN_M: float = 1e-3
"""Adaptive steps: clearance kept beyond the ball's reach over a long step, m."""

GRAZE_SPEED_MPS: float = 40.0
"""Graze search: the speed assumed for a moving surface that states no bound of its own, m/s."""

_STEP_FN = Callable[[Sequence[float], float], list[float]]


def _step_function(model: Dynamics) -> _STEP_FN:
    if type(model) is FlightModel:
        return model.rk4_step

    def step(state: Sequence[float], h: float) -> list[float]:
        return rk4_step(model, state, h)

    return step


def _due(travel: float, clearance: float) -> float:
    """The summed travel at which a gap must be evaluated again (with a margin for rounding)."""

    if not clearance > 0.0:
        return travel
    return travel + clearance * (1.0 - 1e-9) - 1e-12


def _bisect(step: _STEP_FN, gap: GapFunction, t0: float, y0: Sequence[float], h: float, tolerance: float) -> tuple[float, list[float], float, list[float]]:
    """Bracket the root of ``gap`` within ``(t0, t0 + h]`` by re-stepping from ``y0``."""

    lo, hi = 0.0, h
    y_lo = list(y0)
    y_hi = step(y0, h)
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        y_mid = step(y0, mid)
        if gap(t0 + mid, y_mid) >= 0.0:
            lo, y_lo = mid, y_mid
        else:
            hi, y_hi = mid, y_mid
    return t0 + lo, y_lo, t0 + hi, y_hi


def _dip(step: _STEP_FN, gap: GapFunction, t: float, y: Sequence[float], h: float, g0: float, g1: float) -> float | None:
    """
    The time into a step at which a gap that is non-negative at both ends is negative, if the search finds one.

    The gap of a distance to a convex shape is convex along a straight path; three-point parabolic steps towards the
    minimum (two refinements after the midpoint) find a dip that a step-end check misses.
    """

    a, fa, c, fc = 0.0, g0, h, g1
    b = 0.5 * h
    fb = gap(t + b, step(y, b))
    for _ in range(3):
        if fb < 0.0:
            return b
        denominator = (b - a) * (fb - fc) - (b - c) * (fb - fa)
        if denominator == 0.0:
            return None
        x = b - 0.5 * ((b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)) / denominator
        if not a < x < c or abs(x - b) < 1e-12 * h:
            return None
        fx = gap(t + x, step(y, x))
        if fx < 0.0:
            return x
        if fx < fb:
            if x < b:
                c, fc = b, fb
            else:
                a, fa = b, fb
            b, fb = x, fx
        elif x < b:
            a, fa = x, fx
        else:
            c, fc = x, fx
    return None


def integrate_until_event(
    model: Dynamics,
    state0: StateLike,
    t0: float,
    t_end: float,
    events: Sequence[GapEvent] = (),
    *,
    dt: float = DEFAULT_DT,
    record_every: int = 1,
    stop: Callable[[float, Sequence[float]], bool] | None = None,
    time_tolerance: float = 1e-10,
    bounds: tuple[float, float, float] | None = None,
    max_step: float | None = None,
    clear_margin: float = CLEAR_MARGIN_M,
    graze: bool = False,
) -> FlightSegment:
    """
    Integrate one ball from ``t0`` until the first event, a stop, or ``t_end``.

    An event fires when its gap goes from ``>= 0`` at a step's start to ``< 0``
    at its end; a gap already negative at the start is not armed until it
    becomes non-negative. When several fire in one step the earliest root wins.

    A gap whose event has a ``clearance`` method (:class:`StaticClearance`) is
    evaluated only when the ball's summed step-end travel since its last
    evaluation could have changed its sign; every result is bit for bit that of
    evaluating it every step (see the module notes).

    :param model: Flight model.
    :param state0: Initial state (9 floats).
    :param t0: Start time, s.
    :param t_end: End time, s.
    :param events: Objects with ``gap(t, state) -> float`` (metres, or any signed quantity).
    :param dt: Step, s, on a grid anchored at ``t0``; the last step is shortened to land on ``t_end``. With
        ``max_step``, the step near anything the ball could touch.
    :param record_every: Keep every n-th step in the output (the final sample is always kept).
    :param stop: Optional predicate checked after each step; true ends the segment.
    :param time_tolerance: Bisection bracket width, s.
    :param bounds: ``(bx, by, bz)``, m: leaving ``|x| <= bx``, ``|y| <= by``, ``z <= bz`` at a step end ends the
        segment as ``stop`` does (checked only when the ball could have left).
    :param max_step: Adaptive steps (opt-in): the longest step, s, taken only where no event can fire within it
        (every event needs a ``clearance`` or ``clearance_between``; otherwise every step is ``dt``). ``None`` keeps
        the fixed grid.
    :param clear_margin: Adaptive steps: clearance kept beyond the ball's reach over a long step, m.
    :param graze: Search each ``dt`` step that ends clear of a surface, but came within its travel of it, for a touch
        inside it (opt-in; with or without ``max_step``).
    :return: The samples after ``t0`` and the hit, if any.
    """

    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if record_every < 1:
        raise ValueError("record_every must be at least 1")
    step_fn = _step_function(model)
    adaptive = max_step is not None and max_step > dt
    t = t0
    y = _floats(state0)
    count = len(events)
    gap_fns = [event.gap for event in events]
    clearances: list[Callable[[Sequence[float], float], float] | None] = [getattr(event, "clearance", None) for event in events]
    windows: list[Callable[[float, float, Sequence[float]], float] | None] = [getattr(event, "clearance_between", None) for event in events]
    static = tuple(i for i in range(count) if clearances[i] is not None)
    dynamic = tuple(i for i in range(count) if clearances[i] is None)
    gaps = [gap(t, y) for gap in gap_fns]
    travel = 0.0
    due = [math.inf] * count
    for i in static:
        clearance = clearances[i]
        assert clearance is not None
        due[i] = _due(0.0, clearance(y, gaps[i]))
    next_due = min((due[i] for i in static), default=math.inf)
    bx, by, bz = bounds if bounds is not None else (math.inf, math.inf, math.inf)
    bounds_due = 0.0 if bounds is not None else math.inf
    coarse_ok = adaptive and type(model) is FlightModel and all(clearances[i] is not None or windows[i] is not None for i in range(count))
    grazing = graze
    speed_bounds = [float(getattr(event, "speed_bound", GRAZE_SPEED_MPS)) for event in events]
    times: list[float] = []
    states: list[list[float]] = []
    step = 0
    coarse_steps = 0
    grazes = 0
    sqrt = math.sqrt
    while t < t_end:
        if adaptive:
            h = dt
            if coarse_ok:
                h = _long_step(model, y, t, dt, float(max_step), clear_margin, next_due - travel, bounds_due - travel, dynamic, windows)  # type: ignore[arg-type]
            t1 = t + h
        else:
            t1 = t0 + (step + 1) * dt
        if t1 >= t_end - 1e-12:
            t1 = t_end
        h = t1 - t
        coarse = adaptive and h > dt * (1.0 + 1e-9)
        y1 = step_fn(y, h)
        dx, dy, dz = y1[0] - y[0], y1[1] - y[1], y1[2] - y[2]
        chord = sqrt(dx * dx + dy * dy + dz * dz)
        travel += chord
        firing: list[int] = []
        searched: list[tuple[int, float, float]] = []
        for i in dynamic:
            g_start = gaps[i]
            g_end = gap_fns[i](t1, y1)
            if g_start >= 0.0 > g_end:
                firing.append(i)
            elif grazing and not coarse and g_start >= 0.0 and g_end >= 0.0 and g_start + g_end < chord + speed_bounds[i] * h + 1e-9:
                searched.append((i, g_start, g_end))
            gaps[i] = g_end
        if travel >= next_due:
            for i in static:
                if due[i] <= travel:
                    g_start = gaps[i]
                    g_end = gap_fns[i](t1, y1)
                    if g_start >= 0.0 > g_end:
                        firing.append(i)
                    elif grazing and not coarse and g_start >= 0.0 and g_end >= 0.0 and g_start + g_end < chord + 1e-9:
                        searched.append((i, g_start, g_end))
                    gaps[i] = g_end
                    clearance = clearances[i]
                    assert clearance is not None
                    due[i] = _due(travel, clearance(y1, g_end))
            next_due = min(due[i] for i in static)
        dips: dict[int, float] = {}
        for i, g_start, g_end in searched:
            inside = _dip(step_fn, gap_fns[i], t, y, h, g_start, g_end)
            if inside is not None:
                # A touch that entered and left between the step's ends: bracket its root before the dip.
                dips[i] = inside
                firing.append(i)
        if firing:
            best: EventHit | None = None
            for index in sorted(firing):
                t_hit, y_hit, t_after, y_after = _bisect(step_fn, gap_fns[index], t, y, dips.get(index, h), time_tolerance)
                if best is None or t_hit < best.time:
                    best = EventHit(index, t_hit, tuple(y_hit), t_after, tuple(y_after))
            assert best is not None
            grazes += int(best.index in dips)
            times.append(best.time)
            states.append(list(best.state))
            return FlightSegment(times, states, best, False, step + 1, coarse_steps + int(coarse), grazes)
        step += 1
        coarse_steps += int(coarse)
        t, y = t1, y1
        last = t == t_end
        stopped = False
        if travel >= bounds_due:
            margin = min(bx - abs(y[0]), by - abs(y[1]), bz - y[2])
            if margin < 0.0:
                stopped = True
            else:
                bounds_due = _due(travel, margin)
        if not stopped and stop is not None:
            stopped = stop(t, y)
        if step % record_every == 0 or last or stopped:
            times.append(t)
            states.append(y)
        if stopped:
            return FlightSegment(times, states, None, True, step, coarse_steps, grazes)
    return FlightSegment(times, states, None, False, step, coarse_steps, grazes)


def _long_step(
    model: FlightModel,
    y: Sequence[float],
    t: float,
    dt: float,
    max_step: float,
    margin: float,
    static_clearance: float,
    bounds_clearance: float,
    dynamic: Sequence[int],
    windows: Sequence[Callable[[float, float, Sequence[float]], float] | None],
) -> float:
    """The longest step (``max_step`` halved until it is at most twice ``dt``) in which the ball reaches nothing."""

    clearance = min(static_clearance, bounds_clearance)
    if not clearance > margin:
        return dt
    vx, vy, vz = y[3], y[4], y[5]
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    shortest = max_step
    while 0.5 * shortest > 1.5 * dt:
        shortest *= 0.5
    if speed * shortest + margin >= clearance:
        # Even the shortest long step reaches something on its velocity alone (most steps near a surface).
        return dt
    a = model._acceleration(vx, vy, vz, y[6], y[7], y[8])
    # The acceleration changes little over a few milliseconds; twice its start value plus 2 m/s^2 bounds it.
    accel = 2.0 * math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) + 2.0
    h = max_step
    while h > 1.5 * dt:
        reach = speed * h + 0.5 * accel * h * h + margin
        if reach < clearance:
            moving = math.inf
            for i in dynamic:
                window = windows[i]
                moving = min(moving, window(t, t + h, y) if window is not None else 0.0)
                if not moving > reach:
                    break
            if moving > reach:
                return h
        h *= 0.5
    return dt


def integrate(
    model: FlightModel, state0: StateLike, duration: float, *, dt: float = DEFAULT_DT, record_every: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Free flight with no surfaces, for a fixed duration.

    :param model: Flight model.
    :param state0: Initial state (9 floats).
    :param duration: Seconds to integrate.
    :param dt: Step, s.
    :param record_every: Keep every n-th step.
    :return: Times (s, starting at 0) and states, shape ``(N,)`` and ``(N, 9)``.
    """

    segment = integrate_until_event(model, state0, 0.0, duration, dt=dt, record_every=record_every)
    times = np.array([0.0, *segment.times])
    states = np.array([_floats(state0), *segment.states], dtype=float)
    return times, states


def _step_plan(t0: float, times: NDArray[np.float64], dt: float) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Step lengths covering ``[t0, max(times)]`` with no step longer than ``dt``, landing on every requested time."""

    if np.any(times < t0 - 1e-12):
        raise ValueError("requested times must not precede t0")
    end = float(times.max()) if times.size else t0
    grid = t0 + dt * np.arange(1, math.floor((end - t0) / dt + 1e-9) + 1)
    knots = np.unique(np.concatenate([[t0], grid, times]))
    knots = knots[knots >= t0]
    steps = np.diff(knots)
    index = np.searchsorted(knots, times)
    return steps, index.astype(np.int64)


def propagate(model: FlightModel, state0: StateLike, t0: float, times: ArrayLike, *, dt: float = DEFAULT_DT) -> NDArray[np.float64]:
    """
    States of one ball at the requested times (free flight, no surfaces).

    Steps never exceed ``dt`` and land exactly on every requested time.

    :param model: Flight model.
    :param state0: State at ``t0`` (9 floats).
    :param t0: Start time, s.
    :param times: Requested times, s, each ``>= t0``, any order.
    :param dt: Maximum step, s.
    :return: Array of shape ``(len(times), 9)``.
    """

    requested = np.asarray(times, dtype=float).reshape(-1)
    steps, index = _step_plan(t0, requested, dt)
    out = np.empty((steps.size + 1, STATE_SIZE))
    y = _floats(state0)
    out[0] = y
    for i, h in enumerate(steps):
        y = rk4_step(model, y, float(h))
        out[i + 1] = y
    return out[index]


def flight_events(params: BallTableParams) -> tuple[TableTop, Net, Floor]:
    """The main static surfaces a flight is watched against: table top, net and floor."""

    return TableTop.from_params(params), Net.from_params(params), Floor.from_params(params)


def fixture_events(params: BallTableParams, *, undercarriage: bool = True) -> tuple[TableBoard | NetFixture | TableFrame, ...]:
    """
    The rest of the static geometry: the table board (ends, sides, edges, underside), the net posts and
    clamps, and (unless ``undercarriage`` is false) the table's apron, legs and frame.
    """

    frame = (TableFrame.from_params(params),) if undercarriage else ()
    return (TableBoard.from_params(params), *net_fixtures(params), *frame)
