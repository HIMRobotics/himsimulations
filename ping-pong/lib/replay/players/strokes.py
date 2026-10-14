"""
The stroke controller: bounded adaptations that let a recorded stroke meet the live ball.

The ball is always live physics, so the simulated ball never arrives exactly
where the recorded one did. For each scheduled stroke the controller predicts
the incoming ball with the reference model (:mod:`lib.physics.reference`) from
the live simulator state, then chooses one bounded adaptation of the recorded
paddle motion:

- a **time warp** ``dt`` (|dt| <= 60 ms): the motion around the stroke plays
  ``dt`` later, so the recorded contact pose is reached at ``t_s + dt``;
- a **paddle offset** ``dp`` (|dp| <= 8 cm): the blade centre moves so the ball
  meets it;
- a **blade rotation** (<= 10 deg): the face tilts so the reference impact
  model (moving surface, effective mass, rubber tangential restitution) sends
  the ball toward the stroke's landing zone.

The time warp picks which point of the incoming flight the recorded contact
pose meets (the smallest offset), the rotation is solved by damped
Gauss-Newton on the predicted landing point, and the two are iterated. The
character animator applies the result through two-bone arm IK blended over
at least 150 ms (:mod:`lib.replay.players.playback`). The physics
decides what then happens: a clamped adaptation can miss.

Everything here is pure numpy over the reference model.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.physics.params import BallTableParams
from lib.physics.reference.impact import resolve_impact
from lib.physics.reference.simulate import Trajectory, simulate
from lib.replay.players import rotations as rot

Array = NDArray[Any]

PREDICTION_DT_S: float = 1e-3
"""Reference-model step for predictions inside the replay loop (RK4; the error at 1 ms is below 1e-9 m)."""

ARENA_BOUNDS_M: tuple[float, float, float] = (4.5, 4.0, 4.0)
"""A predicted ball leaving ``|x|, |y|, z`` beyond these is out of play."""


@dataclass(frozen=True)
class AdaptationBounds:
    """
    How far the controller may bend a recorded stroke.

    :ivar max_time_shift_s: Largest time warp, s.
    :ivar max_offset_m: Largest blade-centre offset, m.
    :ivar max_rotation_rad: Largest blade tilt, rad.
    :ivar blend_s: Ramp time of the IK blend into the adaptation before contact, s (at least 0.15).
    :ivar hold_s: Half-width of the full-strength plateau around contact, s.
    :ivar follow_through_s: The plateau continues this long after contact, so the follow-through, the fastest part of a
        stroke, plays at its recorded speed and the time warp unwinds during the recovery.
    :ivar release_s: Ramp time out of the adaptation after the follow-through, s; longer than the ramp in, because
        unwinding a time warp replays the follow-through faster by ``1 + 1.5 * dt / release_s``.
    """

    max_time_shift_s: float = 0.060
    max_offset_m: float = 0.080
    max_rotation_rad: float = math.radians(10.0)
    blend_s: float = 0.20
    hold_s: float = 0.05
    follow_through_s: float = 0.25
    release_s: float = 0.50

    def __post_init__(self) -> None:
        if self.blend_s < 0.15:
            raise ValueError(f"the IK blend must last at least 150 ms, got {self.blend_s * 1e3:.0f} ms")
        if self.max_time_shift_s * 1.5 / self.blend_s >= 1.0:
            raise ValueError("the time warp must stay monotonic: max_time_shift_s * 1.5 / blend_s must be below 1")


FIXED_V1_BOUNDS = AdaptationBounds()
"""The fallback bounds for a package that carries no reference uncertainty: 60 ms, 8 cm and 10 degrees."""
EVIDENCE_K: float = 2.0
"""The evidence bounds allow ``EVIDENCE_K`` sigma of a stroke's reference uncertainty per axis."""


PRESWING_SEEN_S: float = 0.33
"""A live ball's deviation seen at least this long before contact can be met by re-planning the swing itself."""
PRESWING_ROTATION_DEG: float = 10.0
"""Pre-swing face re-orientation allowance (a judgement; the priors give none): hand orientation follows a
slant change at about 30-45 deg/s after its 0.15 s latency (van Mierlo 2009, Voudouris 2013), about 6-9 deg in the
0.18 s a pre-swing re-plan leaves, rounded to the 10 deg the controller used before."""
LATE_ROTATION_RATE_DEG_S: float = 30.0
"""Face re-orientation speed after the orientation latency (the slow end of the 30-45 deg/s proxies)."""


@dataclass(frozen=True)
class HumanLimits:
    """
    The ``priors.json`` ``adaptation_*`` values the human re-plan allowance uses. Defaults are the nominals.
    """

    latency_s: float = 0.11
    timing_latency_s: float = 0.17
    orientation_latency_s: float = 0.15
    time_shift_late_s: float = 0.02
    time_shift_preswing_s: float = 0.06
    offset_late_m: float = 0.02
    offset_preswing_m: float = 0.10
    rotation_late_deg: float = 3.0
    accel_max_mps2: float = 10.0
    speed_max_mps: float = 0.4
    source: str = "defaults"


@dataclass(frozen=True)
class ReplanAllowance:
    """
    How far a human could re-plan a stroke against a live ball whose deviation became visible ``seen_s`` before contact.

    :ivar seen_s: Time before contact the deviation exceeded the reference envelope, s (``None``: it never did).
    :ivar regime: ``none`` (no deviation to answer), ``late`` or ``preswing``.
    """

    time_shift_s: float
    offset_m: float
    rotation_rad: float
    seen_s: float | None
    regime: str


def replan_allowance(seen_s: float | None, limits: HumanLimits) -> ReplanAllowance:
    """
    The human re-plan allowance for a deviation seen ``seen_s`` before contact.

    - Never seen (the live ball stays inside the reference envelope): nothing to re-plan.
    - Seen at least :data:`PRESWING_SEEN_S` before contact: the pre-swing limits (about 60 ms and 10 cm) and
      :data:`PRESWING_ROTATION_DEG`.
    - Later: nothing before the latency, then a minimum-jerk shift over ``T = seen - latency`` bounded by peak
      acceleration and speed, ``min(a T^2 / 5.77, v T / 1.875)``, under the late ceiling (2 cm); timing 20 ms once past
      the timing latency; the face at :data:`LATE_ROTATION_RATE_DEG_S` after the orientation latency, under 3 deg.
    """

    if seen_s is None or seen_s <= 0.0:
        return ReplanAllowance(0.0, 0.0, 0.0, seen_s, "none")
    if seen_s >= PRESWING_SEEN_S:
        return ReplanAllowance(limits.time_shift_preswing_s, limits.offset_preswing_m, math.radians(PRESWING_ROTATION_DEG), seen_s, "preswing")
    window = seen_s - limits.latency_s
    offset = 0.0 if window <= 0.0 else min(limits.accel_max_mps2 * window**2 / 5.77, limits.speed_max_mps * window / 1.875, limits.offset_late_m)
    time_shift = limits.time_shift_late_s if seen_s > limits.timing_latency_s else 0.0
    turn = max(0.0, seen_s - limits.orientation_latency_s) * LATE_ROTATION_RATE_DEG_S
    return ReplanAllowance(time_shift, offset, math.radians(min(turn, limits.rotation_late_deg)), seen_s, "late")


def evidence_v3_bounds(time_sd_s: float, position_sd_m: float, normal_sd_deg: float, allowance: ReplanAllowance, *, k: float = EVIDENCE_K) -> AdaptationBounds:
    """
    The evidence bounds: per axis, ``k`` sigma of the reference's reconstruction and registration uncertainty (never
    capped by human limits: it is not a human correction) plus the human re-plan allowance for the live ball's deviation.
    """

    time_shift = k * time_sd_s + allowance.time_shift_s
    return AdaptationBounds(
        max_time_shift_s=time_shift,
        max_offset_m=k * position_sd_m + allowance.offset_m,
        max_rotation_rad=math.radians(k * normal_sd_deg) + allowance.rotation_rad,
        blend_s=max(FIXED_V1_BOUNDS.blend_s, 1.7 * time_shift),
        hold_s=FIXED_V1_BOUNDS.hold_s,
        follow_through_s=FIXED_V1_BOUNDS.follow_through_s,
        release_s=max(FIXED_V1_BOUNDS.release_s, 2.5 * time_shift),
    )


@dataclass(frozen=True)
class PaddleState:
    """
    A paddle blade's pose and twist.

    :ivar position: Blade centre (mid-thickness), table frame, m.
    :ivar rotation: 3x3 matrix whose columns are the blade's X (width), Y (towards the tip) and Z (forehand normal).
    :ivar velocity: Blade-centre velocity, m/s.
    :ivar angular_velocity: rad/s.
    """

    position: Array
    rotation: Array
    velocity: Array
    angular_velocity: Array

    @property
    def normal(self) -> Array:
        """The forehand normal (+Z)."""

        return self.rotation[:, 2]


def predict(params: BallTableParams, state: ArrayLike, t0: float, *, horizon_s: float = 2.0, dt: float = PREDICTION_DT_S) -> Trajectory:
    """
    The reference-model flight of a ball from a state, through table, net and floor contacts.

    :param params: The controller's physics belief.
    :param state: Position, velocity, spin (9 floats), table frame.
    :param t0: Its time, s.
    :param horizon_s: Longest prediction, s.
    :param dt: Integration step, s.
    :return: The trajectory, ending at the floor, out of bounds, rest or the horizon.
    """

    return simulate(state, params, t0=t0, dt=dt, max_time=horizon_s, terminal_events=("floor",), bounds=ARENA_BOUNDS_M, rest_speed=0.3)


def state_at(trajectory: Trajectory, time: float) -> Array | None:
    """
    The trajectory's state at a time, linearly interpolated (post-impact at an impact time).

    :return: 9 floats, or ``None`` outside the trajectory.
    """

    times = trajectory.times
    if time < times[0] or time > times[-1]:
        return None
    index = int(np.searchsorted(times, time, side="right"))
    if index >= len(times):
        return trajectory.states[-1].copy()
    if index == 0:
        return trajectory.states[0].copy()
    t_a, t_b = times[index - 1], times[index]
    if t_b - t_a < 1e-12:
        return trajectory.states[index].copy()
    weight = (time - t_a) / (t_b - t_a)
    return (1.0 - weight) * trajectory.states[index - 1] + weight * trajectory.states[index]


@dataclass(frozen=True)
class Landing:
    """
    Where a ball leaving a paddle comes down.

    :ivar point: ``(x, y)`` of the first table contact, or of the terminal event when there is none, m.
    :ivar time: Its time, s.
    :ivar on_table: Whether the first contact after the paddle is the table.
    :ivar net_touched: Whether the ball touched the net first.
    :ivar net_clearance_m: Ball bottom above the net top where it crossed ``x = 0`` (``-inf`` when it never crossed).
    :ivar height_m: Ball-centre height at the landing (or net) contact, m.
    :ivar obstacle: The first contact was the board's side or underside, or a net post or clamp: no landing.
    """

    point: tuple[float, float]
    time: float
    on_table: bool
    net_touched: bool
    net_clearance_m: float
    height_m: float
    obstacle: bool = False


def event_class(event: Any) -> str:
    """
    What a reference-model event means for a stroke's landing:

    - ``landing``: the playing surface, its top or its edge (an edge ball counts:
      :func:`lib.physics.reference.surfaces.is_edge_ball`);
    - ``net``: the net's mesh or cord;
    - ``obstacle``: the board's side, end or underside, a net post or clamp: never a landing;
    - ``outside``: the table plane crossed beyond the table; otherwise the event's own kind (``floor``...).
    """

    from lib.physics.reference.surfaces import is_edge_ball

    region = getattr(event.contact, "region", "") if event.contact is not None else ""
    if event.kind == "table":
        touch = region or "top"
        return "landing" if touch in ("top", "edge", "side", "underside") and is_edge_ball(touch) else "obstacle"
    if event.kind == "net":
        return "obstacle" if region in ("post", "clamp") else "net"
    if event.kind == "table_plane_outside":
        return "outside"
    return str(event.kind)


def landing_of(params: BallTableParams, state: ArrayLike, t0: float, *, dt: float = PREDICTION_DT_S) -> Landing:
    """
    The first landing of a ball just struck (or served past its own-half bounce).

    :param params: Physics belief.
    :param state: Ball state right after the stroke.
    :param t0: Stroke time, s.
    :param dt: Integration step, s.
    :return: The landing.
    """

    trajectory = simulate(
        state, params, t0=t0, dt=dt, max_time=2.0, terminal_events=("table", "floor", "net", "table_plane_outside"), bounds=ARENA_BOUNDS_M, rest_speed=0.0
    )
    terminal = trajectory.terminal
    clearance = _net_clearance(trajectory, params)
    position = terminal.position
    kind = event_class(terminal)
    return Landing(
        point=(float(position[0]), float(position[1])),
        time=float(terminal.time),
        on_table=kind == "landing",
        net_touched=kind == "net",
        net_clearance_m=clearance,
        height_m=float(position[2]),
        obstacle=kind == "obstacle",
    )


def _net_clearance(trajectory: Trajectory, params: BallTableParams) -> float:
    x = trajectory.positions[:, 0]
    crossing = np.nonzero(np.signbit(x[:-1]) != np.signbit(x[1:]))[0]
    if crossing.size == 0:
        return -math.inf
    index = int(crossing[0])
    weight = x[index] / (x[index] - x[index + 1]) if x[index] != x[index + 1] else 0.0
    z = (1.0 - weight) * trajectory.positions[index, 2] + weight * trajectory.positions[index + 1, 2]
    return float(z - params.ball.radius - params.net.height)


def strike(params: BallTableParams, ball: ArrayLike, paddle: PaddleState, face_normal: ArrayLike) -> Array:
    """
    The ball state right after a paddle contact, by the reference impact model with the paddle as a free rigid
    body about the blade centre (the replay adapter's hybrid paddle contact).

    :param params: Physics belief (rubber law, effective mass, ball).
    :param ball: Ball state at contact (9 floats).
    :param paddle: Paddle pose and twist at contact.
    :param face_normal: Unit normal of the struck face, from the blade towards the ball.
    :return: The post-impact state (position unchanged).
    """

    state = np.asarray(ball, dtype=np.float64)
    normal = np.asarray(face_normal, dtype=np.float64)
    contact_point = state[:3] - params.ball.radius * normal
    surface_velocity = paddle.velocity + np.cross(paddle.angular_velocity, contact_point - paddle.position)
    blade = params.paddle
    # The blade's axes are (width, length, normal); lib.physics's paddle frame is (towards the handle, across, normal).
    inertia = paddle.rotation @ np.diag([blade.inertia_yy, blade.inertia_xx, blade.inertia_zz]) @ paddle.rotation.T
    impact = resolve_impact(
        state[3:6],
        state[6:9],
        params.ball,
        normal,
        params.rubber_contact,
        surface_velocity=surface_velocity,
        effective_mass=blade.effective_mass,
        contact_offset=contact_point - paddle.position,
        surface_inertia=inertia,
    )
    return np.concatenate([state[:3], impact.velocity, impact.spin])


def solve_through(
    params: BallTableParams,
    position: ArrayLike,
    spin: ArrayLike,
    target: ArrayLike,
    flight_time_s: float,
    initial_velocity: ArrayLike,
    *,
    iterations: int = 6,
    tolerance_m: float = 0.005,
) -> tuple[Array, float]:
    """
    The launch velocity whose reference-model flight (table bounces included) passes ``target`` at ``flight_time_s``.

    Damped Gauss-Newton on the three velocity components from ``initial_velocity`` (a recorded flight's fit), so the
    solution stays the nearest flight to the recording that reaches the next contact.

    :return: ``(velocity, residual_m)``: the best velocity found and its miss distance.
    """

    start = np.asarray(position, dtype=np.float64)
    w = np.asarray(spin, dtype=np.float64)
    goal = np.asarray(target, dtype=np.float64)

    def miss(velocity: Array) -> Array:
        trajectory = predict(params, np.concatenate([start, velocity, w]), 0.0, horizon_s=flight_time_s + 0.02)
        state = state_at(trajectory, flight_time_s)
        return (state[:3] if state is not None else trajectory.states[-1][:3]) - goal

    velocity = np.asarray(initial_velocity, dtype=np.float64).copy()
    residual = miss(velocity)
    best = (float(np.linalg.norm(residual)), velocity.copy())
    damping = 1e-3
    step = 1e-3
    for _ in range(iterations):
        if best[0] < tolerance_m:
            break
        jacobian = np.zeros((3, 3))
        for column in range(3):
            probe = velocity.copy()
            probe[column] += step
            jacobian[:, column] = (miss(probe) - residual) / step
        delta = -np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(3), jacobian.T @ residual)
        candidate = velocity + delta
        residual_c = miss(candidate)
        if float(np.linalg.norm(residual_c)) < best[0]:
            velocity, residual = candidate, residual_c
            best = (float(np.linalg.norm(residual_c)), candidate.copy())
            damping *= 0.3
        else:
            damping *= 10.0
    return best[1], best[0]


FACE_CONTACT_SHARE: float = 0.5
"""A ball meeting the face within this share of the outline needs no in-plane move of the blade. Half the outline (3.75
cm): at 0.8 the planner aims balls at the blade's rim, which leaves no margin for the execution's error (891 recorded
strokes replayed through the planner realised 62 % at 0.8 against 77 % at 0.5, and 77 % at 0.0 and 0.3). 0.5 keeps the
realised share and needs fewer offset clamps than a centre hit (15 % against 20 %)."""


def face_offset(centre_move: Array, face: Array, rotation: Array, params: BallTableParams) -> Array:
    """
    The blade move a hit needs: all of ``centre_move`` along the face normal (the face must reach the ball), and in the
    face's plane only what would otherwise leave the contact outside :data:`FACE_CONTACT_SHARE` of the outline (the ball
    may meet the blade off centre).

    :param centre_move: The move that would bring the blade centre to a centre hit, m.
    :param face: Unit normal of the struck face.
    :param rotation: The blade's axes (width, length, normal).
    """

    along = float(centre_move @ face)
    plane = centre_move - along * face
    u = float(plane @ rotation[:, 0]) / params.paddle.blade_half_width
    v = float(plane @ rotation[:, 1]) / params.paddle.blade_half_length
    reach = math.hypot(u, v)
    excess = plane * (1.0 - FACE_CONTACT_SHARE / reach) if reach > FACE_CONTACT_SHARE else np.zeros(3)
    return along * face + excess


def tilt_axes(normal: ArrayLike) -> tuple[Array, Array]:
    """Two unit axes orthogonal to a face normal (horizontal first) that parameterise a tilt."""

    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    horizontal = np.cross((0.0, 0.0, 1.0), n)
    if np.linalg.norm(horizontal) < 1e-6:
        horizontal = np.cross((1.0, 0.0, 0.0), n)
    horizontal /= np.linalg.norm(horizontal)
    return horizontal, np.cross(n, horizontal)


@dataclass(frozen=True)
class AimSolution:
    """
    A blade tilt that sends the ball toward a target.

    :ivar rotation: World-frame tilt applied to the blade, rotation vector (rad).
    :ivar normal: The tilted face normal.
    :ivar post_state: Ball state after contact with the tilted face.
    :ivar landing: The predicted landing.
    :ivar error_m: Distance from the landing to the target, m.
    :ivar required_rad: The tilt the solver wanted before the bound, rad.
    :ivar clamped: Whether the bound cut the tilt.
    :ivar next_miss_m: With a next contact: the predicted path's closest approach to it within the receiver's time
        window, m (``nan`` without one).
    :ivar next_dt_s: When that closest approach comes, relative to the next contact's recorded time, s.
    """

    rotation: Array
    normal: Array
    post_state: Array
    landing: Landing
    error_m: float
    required_rad: float
    clamped: bool
    next_miss_m: float = math.nan
    next_dt_s: float = math.nan


MAX_AIM_TILT_RAD: float = math.pi / 2
"""The aim never searches beyond a quarter turn of the face: past it the reported requirement is meaningless."""


NEXT_TIME_SCALE_S: float = 0.15
"""How much a next-contact closest approach that comes early or late counts: 0.15 s weighs like ``contact_scale_m``."""
OBSTACLE_PENALTY_M: float = 0.15
"""How far outside the far half a first contact with the board's side, a post or a clamp counts in the aim."""
AIM_GRID: int = 9
"""Start points per tilt axis of the aim's grid over its bound."""


def _outside_half(point: tuple[float, float], sign: float, params: BallTableParams) -> float:
    """How far a table-plane point lies outside the far half ``sign * x in [0, L/2]``, ``|y| <= W/2`` (0 inside), m."""

    x, y = point[0] * sign, point[1]
    half_length, half_width = 0.5 * params.table.length, 0.5 * params.table.width
    dx = max(0.0 - x, 0.0, x - half_length)
    dy = max(abs(y) - half_width, 0.0)
    return math.hypot(dx, dy)


def aim_face(
    params: BallTableParams,
    ball: ArrayLike,
    paddle: PaddleState,
    face_normal: ArrayLike,
    target_xy: ArrayLike,
    *,
    time_s: float,
    max_rotation_rad: float,
    iterations: int = 10,
    grid: int = AIM_GRID,
    initial: ArrayLike = (0.0, 0.0),
    next_contact: tuple[float, ArrayLike] | None = None,
    next_window_s: float = 0.0,
    zone_m: float = 0.30,
    contact_scale_m: float = 0.05,
    cache: dict[bytes, Any] | None = None,
) -> AimSolution:
    """
    Tilt a face (within a bound) so the struck ball lands on a target, or meets the other player's next contact.

    Without ``next_contact`` the solver puts the landing on ``target_xy`` (least squares in units of ``zone_m``).

    With ``next_contact`` it minimises how far the struck ball's predicted path passes from the next contact: the
    closest approach within ``next_window_s`` of the contact's recorded time (the receiver may warp that far), with
    a soft term for how early or late it comes (:data:`NEXT_TIME_SCALE_S`). The landing is then a constraint, not a
    target: a penalty for a first bounce outside the far half and, continuously, for a net ball's deficit under the
    tape. The miss term is the path's distance whatever its first event, so the objective has no plateaus.

    Damped Gauss-Newton from the best point of a ``grid`` x ``grid`` start grid over the bound.

    :param params: Physics belief.
    :param ball: Ball state at contact.
    :param paddle: Paddle pose and twist at contact.
    :param face_normal: The struck face's nominal normal (towards the ball).
    :param target_xy: Landing target ``(x, y)``, m (its sign of ``x`` names the far half).
    :param time_s: Contact time, s.
    :param max_rotation_rad: Bound on the tilt.
    :param iterations: Gauss-Newton iterations.
    :param grid: Points per axis of the coarse starting grid over the bound.
    :param initial: Starting tilt angles about :func:`tilt_axes`, rad.
    :param next_contact: ``(time s, ball-centre point m)`` of the other player's next recorded contact, or ``None``.
    :param next_window_s: The receiver's time bound around the next contact, s.
    :param zone_m: Landing scale, m.
    :param contact_scale_m: Next-contact miss scale, m.
    :param cache: Evaluations already made for this ball, paddle, target and next contact (``plan_stroke`` shares one
        across its passes): an evaluation is a pure function of the struck ball, so a repeated one (the grid's centre,
        a rejected Gauss-Newton step's Jacobian probes, a second pass meeting the same ball) is looked up, not re-flown.
    :return: The solution (the best one found, bounded).
    """

    target = np.asarray(target_xy, dtype=np.float64)
    sign = 1.0 if target[0] >= 0.0 else -1.0
    nominal = np.asarray(face_normal, dtype=np.float64)
    nominal = nominal / np.linalg.norm(nominal)
    axes = tilt_axes(nominal)

    meet = None if next_contact is None else (float(next_contact[0]), np.asarray(next_contact[1], dtype=np.float64))
    window = max(0.0, float(next_window_s))

    flights: dict[bytes, Any] = {} if cache is None else cache
    context = b"".join(
        np.asarray(value, dtype=np.float64).tobytes()
        for value in (time_s, target, *(meet if meet is not None else (math.nan,)), window, zone_m, contact_scale_m)
    )

    def evaluate(angles: Array) -> tuple[Array, Array, Landing, Array, float, float]:
        rotation_vector = angles[0] * axes[0] + angles[1] * axes[1]
        normal = rot.rotate(rot.from_rotvec(rotation_vector), nominal)
        post = strike(params, ball, paddle, normal)
        key = context + post.tobytes()
        known = flights.get(key)
        if known is not None:
            residual, landing, miss, dt = known
            return residual.copy(), normal, landing, post, miss, dt
        out = _evaluate_post(post, normal)
        flights[key] = (out[0].copy(), out[2], out[4], out[5])
        return out

    def _evaluate_post(post: Array, normal: Array) -> tuple[Array, Array, Landing, Array, float, float]:
        if meet is None:
            landing = landing_of(params, post, time_s)
            return np.asarray(landing.point) - target, normal, landing, post, math.nan, math.nan
        trajectory = simulate(
            post,
            params,
            t0=time_s,
            dt=2.0 * PREDICTION_DT_S,
            max_time=max(meet[0] + window - time_s, 0.0) + 0.05,
            terminal_events=("floor", "net"),
            bounds=ARENA_BOUNDS_M,
            rest_speed=0.0,
        )
        first = next((event for event in trajectory.events if event.kind in ("table", "table_plane_outside", "net", "floor")), trajectory.terminal)
        kind = event_class(first)
        landing = Landing(
            point=(float(first.position[0]), float(first.position[1])),
            time=float(first.time),
            on_table=kind == "landing",
            net_touched=kind == "net",
            net_clearance_m=_net_clearance(trajectory, params),
            height_m=float(first.position[2]),
            obstacle=kind == "obstacle",
        )
        # The path's closest approach to the next contact within the receiver's window (the last state when the
        # flight ends before it: still continuous, and the time term says how early).
        times = trajectory.times
        positions = trajectory.positions
        inside = (times >= meet[0] - window - 1e-9) & (times <= meet[0] + window + 1e-9)
        # No sample inside a narrow window (none at all for a zero window): the one nearest the contact's time, which is
        # the last when the flight ended before it.
        candidates = np.flatnonzero(inside) if np.any(inside) else np.array([int(np.argmin(np.abs(times - meet[0])))])
        distances = np.linalg.norm(positions[candidates] - meet[1], axis=1)
        best = int(candidates[int(np.argmin(distances))])
        miss = float(np.linalg.norm(positions[best] - meet[1]))
        dt = float(times[best] - meet[0])
        outside = _outside_half(landing.point, sign, params) if not landing.net_touched else 0.0
        if landing.obstacle:
            # The board's side or underside, a post or a clamp: no landing at all, whatever its position.
            outside = max(outside, OBSTACLE_PENALTY_M)
        deficit = max(params.net.height + params.ball.radius - landing.height_m, 0.0) + 0.05 if landing.net_touched else 0.0
        residual = np.array([miss / contact_scale_m, dt / NEXT_TIME_SCALE_S, outside / (0.1 * zone_m), 10.0 * deficit / zone_m])
        return residual, normal, landing, post, miss, dt

    # The objective is not monotonic in the tilt (it peaks, and the net cuts it), so Gauss-Newton starts from the
    # best point of a grid over the bound.
    angles = np.asarray(initial, dtype=np.float64).copy()
    residual, normal, landing, post, miss, dt = evaluate(angles)
    best = (float(np.linalg.norm(residual)), angles.copy(), normal, landing, post, miss, dt)
    ticks = np.linspace(-max_rotation_rad, max_rotation_rad, grid)
    for pitch in ticks:
        for yaw in ticks:
            if math.hypot(pitch, yaw) > max_rotation_rad + 1e-9:
                continue
            probe = np.array([pitch, yaw])
            probe_residual, probe_normal, probe_landing, probe_post, probe_miss, probe_dt = evaluate(probe)
            if float(np.linalg.norm(probe_residual)) < best[0]:
                best = (float(np.linalg.norm(probe_residual)), probe, probe_normal, probe_landing, probe_post, probe_miss, probe_dt)
                angles, residual = probe, probe_residual
    damping = 1e-3
    step = math.radians(0.5)
    for _ in range(iterations):
        jacobian = np.zeros((residual.size, 2))
        for column in range(2):
            probe = angles.copy()
            probe[column] += step
            jacobian[:, column] = (evaluate(probe)[0] - residual) / step
        lhs = jacobian.T @ jacobian + damping * np.eye(2)
        delta = -np.linalg.solve(lhs, jacobian.T @ residual)
        candidate = angles + delta
        size = float(np.linalg.norm(candidate))
        if size > MAX_AIM_TILT_RAD:
            candidate *= MAX_AIM_TILT_RAD / size
        residual_c, normal_c, landing_c, post_c, miss_c, dt_c = evaluate(candidate)
        if float(np.linalg.norm(residual_c)) < best[0]:
            angles, residual = candidate, residual_c
            best = (float(np.linalg.norm(residual_c)), candidate.copy(), normal_c, landing_c, post_c, miss_c, dt_c)
            damping *= 0.3
        else:
            damping *= 10.0
        if best[0] < 0.01:
            break
    required = float(np.linalg.norm(best[1]))
    clamped = required > max_rotation_rad
    if clamped:
        angles = best[1] * (max_rotation_rad / required)
        residual, normal, landing, post, miss, dt = evaluate(angles)
        best = (float(np.linalg.norm(residual)), angles, normal, landing, post, miss, dt)
    rotation_vector = best[1][0] * axes[0] + best[1][1] * axes[1]
    error = float(np.hypot(best[3].point[0] - target[0], best[3].point[1] - target[1]))
    return AimSolution(
        rotation=rotation_vector,
        normal=best[2],
        post_state=best[4],
        landing=best[3],
        error_m=error,
        required_rad=required,
        clamped=clamped,
        next_miss_m=best[5],
        next_dt_s=best[6],
    )


@dataclass(frozen=True)
class StrokePlan:
    """
    The adaptation chosen for one stroke.

    :ivar stroke: Stroke index.
    :ivar scheduled_s: Recorded contact time, session s.
    :ivar time_shift_s: Applied time warp, s.
    :ivar offset_m: Applied blade-centre offset, m (world).
    :ivar rotation: Applied blade tilt, world rotation vector (rad).
    :ivar contact_s: Planned contact time, session s.
    :ivar ball_at_contact: Predicted ball state at contact.
    :ivar landing: Predicted landing of the struck ball.
    :ivar target_xy: The landing target, m.
    :ivar required_time_shift_s: Time warp before clamping, s.
    :ivar required_offset_m: Offset magnitude before clamping, m.
    :ivar required_rotation_rad: Tilt magnitude before clamping, rad.
    :ivar within_bounds: Whether nothing had to be clamped.
    :ivar reachable: Whether the predicted ball reaches the stroke window at all.
    :ivar planned_at_s: Session time the plan was made, s.
    :ivar prediction: Predicted incoming ball positions for telemetry, ``(N, 3)``.
    :ivar predicted_next_miss_m: The struck ball's predicted closest approach to the other player's next contact
        (within the receiver's time window), m; ``nan`` without a next contact.
    :ivar predicted_next_dt_s: When it comes, relative to that contact's recorded time, s.
    :ivar clamp_reach: The arm could not hold the planned blade and the held one misses the ball's contact on the face
        (a clamp with cause ``reach``, not an IK miss; :func:`project_plan`).
    :ivar reach_fraction: The share of the planned offset the held blade keeps.
    :ivar reach_shortfall_m: How far the planned blade lay beyond the arm's reach, m.
    :ivar reach_tilt_shortfall_deg: How far the planned face tilt lay beyond what the joints' ranges hold, degrees (the
        held tilt is the plan's; :func:`project_plan`).
    """

    stroke: int
    scheduled_s: float
    time_shift_s: float
    offset_m: Array
    rotation: Array
    contact_s: float
    ball_at_contact: Array | None
    landing: Landing | None
    target_xy: tuple[float, float]
    required_time_shift_s: float
    required_offset_m: float
    required_rotation_rad: float
    within_bounds: bool
    reachable: bool
    planned_at_s: float
    prediction: Array = field(repr=False)
    bounds: AdaptationBounds = field(default_factory=lambda: AdaptationBounds())
    predicted_next_miss_m: float = math.nan
    predicted_next_dt_s: float = math.nan
    clamp_reach: bool = False
    reach_fraction: float = 1.0
    reach_shortfall_m: float = 0.0
    reach_tilt_shortfall_deg: float = 0.0

    @property
    def clamp_time(self) -> bool:
        """The time warp sat at its bound."""

        return self.reachable and abs(self.time_shift_s) >= self.bounds.max_time_shift_s - 1e-9

    @property
    def clamp_offset(self) -> bool:
        """The blade-centre offset needed more than its bound."""

        return self.reachable and self.required_offset_m > self.bounds.max_offset_m

    @property
    def clamp_tilt(self) -> bool:
        """The face tilt needed more than its bound."""

        return self.reachable and self.required_rotation_rad > self.bounds.max_rotation_rad + 1e-9


NominalPaddle = Callable[[float], PaddleState]
"""Recorded (unadapted) paddle pose and twist at a motion time."""


def plan_stroke(
    params: BallTableParams,
    *,
    stroke: int,
    scheduled_s: float,
    ball_state: ArrayLike,
    now_s: float,
    nominal_paddle: NominalPaddle,
    target_xy: ArrayLike,
    bounds: AdaptationBounds,
    passes: int = 2,
    next_contact: tuple[float, ArrayLike] | None = None,
    next_window_s: float = 0.0,
) -> StrokePlan:
    """
    Choose the bounded adaptation that lets a recorded stroke meet the predicted ball.

    :param params: Physics belief (nominal parameters, whatever the simulator runs).
    :param stroke: Stroke index.
    :param scheduled_s: Recorded contact time, session s.
    :param ball_state: Live ball state (9 floats) at ``now_s``.
    :param now_s: Session time of the state, s.
    :param nominal_paddle: The recorded paddle at a motion time.
    :param target_xy: Landing target on the far half, m.
    :param bounds: Adaptation bounds.
    :param passes: Alternations between the time/offset choice and the aim.
    :param next_contact: ``(time s, ball-centre point m)`` of the other player's next recorded contact, to keep the rally on the recording.
    :param next_window_s: The receiver's time bound around that contact, s (:func:`aim_face`).
    :return: The plan; ``reachable`` is false when the predicted ball never enters the stroke window.
    """

    target = np.asarray(target_xy, dtype=np.float64)
    horizon = max(0.2, scheduled_s + bounds.max_time_shift_s + 0.1 - now_s)
    trajectory = predict(params, ball_state, now_s, horizon_s=horizon)
    prediction = trajectory.positions[:: max(1, len(trajectory.positions) // 120)]
    recorded = nominal_paddle(scheduled_s)
    shifts = np.linspace(-bounds.max_time_shift_s, bounds.max_time_shift_s, 25)
    candidates = [(float(shift), state_at(trajectory, scheduled_s + float(shift))) for shift in shifts]
    candidates = [(shift, state) for shift, state in candidates if state is not None and state[2] > -0.05]
    empty = np.zeros(3)
    if not candidates:
        return StrokePlan(
            stroke=stroke,
            scheduled_s=scheduled_s,
            time_shift_s=0.0,
            offset_m=empty,
            rotation=empty.copy(),
            contact_s=scheduled_s,
            ball_at_contact=None,
            landing=None,
            target_xy=(float(target[0]), float(target[1])),
            required_time_shift_s=0.0,
            required_offset_m=math.inf,
            required_rotation_rad=0.0,
            within_bounds=False,
            reachable=False,
            planned_at_s=now_s,
            prediction=prediction,
            bounds=bounds,
        )
    face_gap = params.ball.radius + 0.5 * params.paddle.thickness
    tilt = np.zeros(3)
    best_shift, best_state = candidates[len(candidates) // 2]
    aim: AimSolution | None = None
    flights: dict[bytes, Any] = {}
    for _ in range(passes):
        normal = rot.rotate(rot.from_rotvec(tilt), recorded.normal)

        def face_for(state: Array, normal: Array = normal) -> Array:
            # The struck face is the one the ball approaches (relative to the moving blade),
            # whichever side of the blade it happens to be on at a candidate time.
            return normal if float(normal @ (state[3:6] - recorded.velocity)) < 0.0 else -normal

        def offset_for(state: Array) -> Array:
            return face_offset(state[:3] - face_for(state) * face_gap - recorded.position, face_for(state), recorded.rotation, params)

        best_shift, best_state = min(candidates, key=lambda item: float(np.linalg.norm(offset_for(item[1]))))
        face = face_for(best_state)
        nominal_face = recorded.normal if float(recorded.normal @ face) >= 0.0 else -recorded.normal
        # The blade moves only as far as the hit needs; the ball may meet it off centre (the impact is the rigid paddle's).
        moved = PaddleState(
            position=recorded.position + offset_for(best_state),
            rotation=recorded.rotation,
            velocity=recorded.velocity,
            angular_velocity=recorded.angular_velocity,
        )
        aim = aim_face(
            params,
            best_state,
            moved,
            nominal_face,
            target,
            time_s=scheduled_s + best_shift,
            max_rotation_rad=bounds.max_rotation_rad,
            next_contact=next_contact,
            next_window_s=next_window_s,
            cache=flights,
        )
        tilt = aim.rotation
    assert aim is not None
    face = aim.normal
    required_offset = face_offset(best_state[:3] - face * face_gap - recorded.position, face, recorded.rotation, params)
    required_norm = float(np.linalg.norm(required_offset))
    offset = required_offset if required_norm <= bounds.max_offset_m else required_offset * (bounds.max_offset_m / required_norm)
    within = required_norm <= bounds.max_offset_m and not aim.clamped and abs(best_shift) < bounds.max_time_shift_s - 1e-9
    return StrokePlan(
        stroke=stroke,
        scheduled_s=scheduled_s,
        time_shift_s=best_shift,
        offset_m=offset,
        rotation=aim.rotation,
        contact_s=scheduled_s + best_shift,
        ball_at_contact=best_state,
        landing=aim.landing,
        target_xy=(float(target[0]), float(target[1])),
        required_time_shift_s=best_shift,
        required_offset_m=required_norm,
        required_rotation_rad=aim.required_rad,
        within_bounds=within,
        reachable=True,
        planned_at_s=now_s,
        prediction=prediction,
        bounds=bounds,
        predicted_next_miss_m=aim.next_miss_m,
        predicted_next_dt_s=aim.next_dt_s,
    )


REACH_FACE_NORMAL_M: float = 0.025
"""How far the held face may stand off the planned ball along its normal and still count as meeting it on the face: the
contact then comes a little earlier or later, and the hybrid contact (not the plan) decides where."""


def contact_on_face(params: BallTableParams, ball: ArrayLike, blade_position: ArrayLike, blade_rotation: ArrayLike, face: ArrayLike) -> bool:
    """
    Whether a ball meets a blade's face inside its outline less a ball-radius margin.

    The physics-honest test for a blade the arm holds short of the planned one: the contact is wherever the ball meets
    the face, and the off-centre rigid-blade impact law resolves it, so a residual shortfall is acceptable while the
    ball still lands on the face (in-plane inside the outline's semi-axes less the ball radius; along the normal within
    :data:`REACH_FACE_NORMAL_M` of touching).

    :param ball: Ball state at the planned contact.
    :param blade_position: Blade centre the arm holds, m.
    :param blade_rotation: The blade's axes (width, length, normal).
    :param face: Unit normal of the struck face (toward the ball).
    """

    relative = np.asarray(ball, dtype=np.float64)[:3] - np.asarray(blade_position, dtype=np.float64)
    axes = np.asarray(blade_rotation, dtype=np.float64)
    normal = np.asarray(face, dtype=np.float64)
    gap = params.ball.radius + 0.5 * params.paddle.thickness
    along = float(relative @ normal) - gap
    in_plane = relative - float(relative @ normal) * normal
    u = float(in_plane @ axes[:, 0]) / max(params.paddle.blade_half_width - params.ball.radius, 1e-6)
    v = float(in_plane @ axes[:, 1]) / max(params.paddle.blade_half_length - params.ball.radius, 1e-6)
    return abs(along) <= REACH_FACE_NORMAL_M and u * u + v * v <= 1.0


def project_plan(
    params: BallTableParams,
    plan: StrokePlan,
    nominal_paddle: NominalPaddle,
    held_offset_m: ArrayLike,
    shortfall_m: ArrayLike,
    *,
    next_contact: tuple[float, ArrayLike] | None = None,
    next_window_s: float = 0.0,
    held_rotation_rad: ArrayLike | None = None,
    tilt_shortfall_deg: float = 0.0,
) -> StrokePlan:
    """
    Project a plan the arm cannot hold onto the nearest one it can: the blade where the arm holds it at contact (as
    much of the adaptation as reaches, on every axis) and, when the joints' ranges cannot turn the hand to the planned
    tilt, the tilt they hold (``held_rotation_rad``; without it the planned tilt is kept).

    The held plan stays within bounds when the planned ball still meets the held face (:func:`contact_on_face`): the
    shortfall then only moves the contact across the face. Otherwise it is a clamp with cause ``reach``. The predicted
    landing and next-contact miss are those of the held blade.

    :param plan: The plan, with ``ball_at_contact``.
    :param nominal_paddle: The paddle the plan adapted (its reference).
    :param held_offset_m: The held blade's offset from the reference at contact, m.
    :param shortfall_m: The planned blade less the held one at contact, m.
    :param held_rotation_rad: The face tilt the arm holds (world rotation vector, rad), when it is not the plan's
        (:attr:`~lib.replay.players.playback.PlanReach.held_rotation_rad`).
    :param tilt_shortfall_deg: The planned tilt less the held one, degrees (reported).
    :return: The projected plan.
    """

    offset = np.asarray(held_offset_m, dtype=np.float64)
    rotation = np.asarray(plan.rotation if held_rotation_rad is None else held_rotation_rad, dtype=np.float64)
    landing, miss, dt = plan.landing, math.nan, math.nan
    on_face = False
    if plan.ball_at_contact is not None:
        recorded = nominal_paddle(plan.scheduled_s)
        ball = plan.ball_at_contact
        face = recorded.normal if float(recorded.normal @ (ball[3:6] - recorded.velocity)) < 0.0 else -recorded.normal
        tilted = rot.rotate(rot.from_rotvec(rotation), face)
        held_rotation = rot.to_matrix(rot.from_rotvec(rotation)) @ recorded.rotation
        moved = PaddleState(position=recorded.position + offset, rotation=held_rotation, velocity=recorded.velocity, angular_velocity=recorded.angular_velocity)
        on_face = contact_on_face(params, ball, moved.position, held_rotation, tilted)
        aim = aim_face(
            params,
            ball,
            moved,
            tilted,
            plan.target_xy,
            time_s=plan.contact_s,
            max_rotation_rad=0.0,
            grid=1,
            iterations=0,
            next_contact=next_contact,
            next_window_s=next_window_s,
        )
        landing, miss, dt = aim.landing, aim.next_miss_m, aim.next_dt_s
    planned = float(np.linalg.norm(plan.offset_m))
    return dataclasses.replace(
        plan,
        offset_m=offset,
        rotation=rotation,
        landing=landing,
        within_bounds=plan.within_bounds and on_face,
        clamp_reach=not on_face,
        reach_fraction=float(min(1.0, np.linalg.norm(offset) / planned)) if planned > 1e-9 else 1.0,
        reach_shortfall_m=float(np.linalg.norm(shortfall_m)),
        reach_tilt_shortfall_deg=float(tilt_shortfall_deg) if held_rotation_rad is not None else 0.0,
        predicted_next_miss_m=miss,
        predicted_next_dt_s=dt,
    )


def solve_launch(
    params: BallTableParams,
    position: ArrayLike,
    spin: ArrayLike,
    *,
    targets: Sequence[tuple[float, float]],
    flight_time_s: float | None,
    initial_velocity: ArrayLike,
    t0: float = 0.0,
    iterations: int = 12,
) -> tuple[Array, float]:
    """
    Solve a launch velocity whose table bounces land on given points (serves and continuity balls).

    With one target the residual is the first table contact's ``(x, y)`` and,
    when ``flight_time_s`` is given, its time. With two targets (a serve) the
    residual is the first bounce's ``x`` (its ``y`` is free) and the second
    bounce's ``(x, y)``.

    :param params: Physics belief.
    :param position: Launch point, m.
    :param spin: Launch spin, rad/s.
    :param targets: One or two ``(x, y)`` table points.
    :param flight_time_s: Time to the first bounce for a one-target launch, or ``None``.
    :param initial_velocity: Starting guess, m/s.
    :param t0: Launch time, s.
    :param iterations: Gauss-Newton iterations.
    :return: The velocity and the residual norm (m, s mixed; small is good).
    """

    p = np.asarray(position, dtype=np.float64)
    w = np.asarray(spin, dtype=np.float64)

    def residual(velocity: Array) -> Array:
        state = np.concatenate([p, velocity, w])
        trajectory = simulate(state, params, t0=t0, dt=PREDICTION_DT_S, max_time=2.5, terminal_events=("floor", "net"), bounds=ARENA_BOUNDS_M, rest_speed=0.0)
        bounces = [event for event in trajectory.events if event_class(event) in ("landing", "outside")]
        terminal = trajectory.terminal
        # A net ball stops at the tape: score it by how far it fell short, so the solver lifts it over.
        net_penalty = 0.0
        if terminal.kind == "net":
            net_penalty = 0.3 + 3.0 * max(params.net.height + params.ball.radius - float(terminal.position[2]), 0.0)
        last = targets[-1]
        beyond = -float(np.sign(last[0])) * net_penalty
        if len(targets) == 1:
            first = bounces[0] if bounces else terminal
            out = [first.position[0] - targets[0][0] + (beyond if not bounces else 0.0), first.position[1] - targets[0][1]]
            if flight_time_s is not None:
                out.append((first.time - t0 - flight_time_s) * 5.0)
            return np.asarray(out)
        first_position = bounces[0].position if bounces else terminal.position
        second_position = bounces[1].position if len(bounces) > 1 else terminal.position
        second_x = second_position[0] - targets[1][0] + (beyond if len(bounces) < 2 else 0.0)
        return np.asarray([first_position[0] - targets[0][0], second_x, second_position[1] - targets[1][1]])

    velocity = np.asarray(initial_velocity, dtype=np.float64).copy()
    current = residual(velocity)
    damping = 1e-3
    for _ in range(iterations):
        jacobian = np.zeros((current.size, 3))
        for column in range(3):
            probe = velocity.copy()
            probe[column] += 0.02
            jacobian[:, column] = (residual(probe) - current) / 0.02
        delta = -np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(3), jacobian.T @ current)
        candidate = velocity + delta
        candidate_residual = residual(candidate)
        if float(np.linalg.norm(candidate_residual)) < float(np.linalg.norm(current)):
            velocity, current = candidate, candidate_residual
            damping *= 0.3
        else:
            damping *= 10.0
        if float(np.linalg.norm(current)) < 0.005:
            break
    return velocity, float(np.linalg.norm(current))


__all__ = [
    "ARENA_BOUNDS_M",
    "EVIDENCE_K",
    "FACE_CONTACT_SHARE",
    "FIXED_V1_BOUNDS",
    "PREDICTION_DT_S",
    "PRESWING_SEEN_S",
    "AdaptationBounds",
    "AimSolution",
    "HumanLimits",
    "Landing",
    "NominalPaddle",
    "PaddleState",
    "ReplanAllowance",
    "StrokePlan",
    "aim_face",
    "contact_on_face",
    "event_class",
    "evidence_v3_bounds",
    "face_offset",
    "landing_of",
    "plan_stroke",
    "predict",
    "project_plan",
    "replan_allowance",
    "solve_launch",
    "solve_through",
    "state_at",
    "strike",
    "tilt_axes",
]
