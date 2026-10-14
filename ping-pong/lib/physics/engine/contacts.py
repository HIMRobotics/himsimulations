"""
Contact detection and the hybrid contacts, pure numpy over one physics step.

Two kinds of contact reach the game:

``hybrid``
    The engine never sees the pair (the paddle has no ball collision; in
    hybrid table mode the table-ball pair is filtered). Each physics step the
    adapter sweeps the ball against the surface over the coming step, and at
    the first touch resolves the impact with
    :func:`lib.physics.reference.impact.resolve_impact` (moving surface,
    effective mass, rubber tangential restitution and normal-spin retention).
    The new state is written so that, after the engine integrates the step,
    the ball sits where the event-driven solution puts it: it leaves the
    contact point at the contact time with the post-impact velocity.

``engine``
    PhysX resolves table, net and floor contacts with the surface materials.
    The adapter recognises them afterwards from the step's velocity change
    and records the states either side.

Every contact becomes a :class:`ContactEvent` with its pre- and post-impact
state, so a run's evidence lists each bounce and hit with the numbers needed
to check it (normal restitution, energy, impact speed).

**Sustained contact.** The impact law describes a bounce. When a ball reaches a
static surface so slowly that its rebound would leave it with less than
:data:`REST_SPEED_MPS` of separation while gravity presses it on (a ball landing
on the net cord, rolling on the table or resting on a forearm), gravity closes
the gap again within the step, and resolving every step as a new impact gave a
chain of hundreds of micro-impacts (games 3, 6 and 7 of the v3 runs: 250-650 net
"contacts" in 0.09-0.18 s at 2.0 mm/s approach, ``g cos(theta) dt / (1 + e)``).
The reference simulation stops at this point ("sustained rolling contact is
outside the impact model", ``lib.physics.reference.simulate``'s ``rest_speed``,
the same 0.05 m/s), so the adapter resolves it as the constraint it is
(:func:`impose_contact` with ``rest_speed``): no normal rebound, Coulomb friction
without the impact-only tangential restitution, the ball on the surface (a
started-inside touch is moved out by its depth), and one event for the whole
episode (:class:`ContactEpisodes`).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.physics.engine.colliders import ROOM_BOXES, ROOM_HEIGHT_M, ROOM_X_M, ROOM_Y_M, SURFACE_HEIGHT_M, TABLE_BOARD_THICKNESS_M, room_gap
from lib.physics.params import BallTableParams, ContactMaterial
from lib.physics.reference.impact import resolve_impact
from lib.physics.reference.undercarriage import Part, undercarriage_parts

Array = NDArray[np.float64]
Surface = Literal["paddle", "table", "net", "floor", "wall", "body", "frame", "other"]

NATIVE_CONTACT_DV_MPS: float = 0.15
"""A step whose velocity changes by more than this beyond gravity and the applied air force held a contact."""
NATIVE_MERGE_S: float = 0.0035
"""Engine contacts on the same surface within this long of each other are one contact (3 steps at 960 Hz)."""
STOP_SPEED_MPS: float = 1e-3
"""A ball rolling on the floor or the playing surface slower than this has stopped (the reference model's
``lib.physics.reference.rolling.STOP_SPEED_MPS``)."""
FLAT_NORMAL_Z: float = 1.0 - 1e-9
"""A sustained touch whose normal's vertical component exceeds this is on a flat top (the floor or the playing surface),
where rolling resistance acts and a ball can stop; the board's edge and the net cord are never flat and never hold a ball."""
REST_SPEED_MPS: float = 0.05
"""A static-surface touch whose rebound would separate slower than this, with gravity pressing the ball on, is a sustained
(rolling or resting) contact, not a bounce: the reference simulation's ``rest_speed``. Its ballistic return time,
``2 u / g``, is at most 10 ms, so a bounce this weak is invisible at 24 fps."""


@dataclass(frozen=True)
class ContactEvent:
    """
    One ball contact, with the states either side.

    :ivar index: Event number in the run.
    :ivar time_s: Game time of the contact, s.
    :ivar step: Physics step index.
    :ivar surface: What the ball hit.
    :ivar mode: ``hybrid`` (imposed by the reference impact model) or ``engine`` (resolved by PhysX).
    :ivar actor: The paddle's owner for paddle contacts, else ``None``.
    :ivar position: Ball centre at contact, m.
    :ivar normal: Contact normal from the surface into the ball.
    :ivar before: Ball state (position, velocity, spin) just before.
    :ivar after: Ball state just after.
    :ivar approach_speed_mps: Normal approach speed relative to the surface, m/s.
    :ivar normal_restitution: ``-u_n' / u_n`` relative to the surface (nan when not approaching).
    :ivar energy_before_j: Ball kinetic energy (translation and spin) before, J.
    :ivar energy_after_j: The same after, J.
    :ivar regime: Impact-model branch for hybrid contacts (``slip``, ``stick``), empty for engine contacts.
    :ivar blade_offset_m: Contact point from the blade centre in blade axes ``(width, length)``, paddle contacts only.
    :ivar surface_velocity: Velocity of the surface point at the contact, m/s.
    :ivar duration_s: How long a sustained contact lasted (0 for a bounce), s.
    :ivar steps: Physics steps the contact spanned (1 for a bounce).
    :ivar end: A sustained contact's ball state after its latest step (``None`` for a bounce); ``before``, ``after`` and
        the energies stay those of the first touch, so the restitution and energy checks keep their meaning (over a
        roll gravity does work on the ball).
    """

    index: int
    time_s: float
    step: int
    surface: Surface
    mode: Literal["hybrid", "engine"]
    actor: str | None
    position: Array
    normal: Array
    before: Array
    after: Array
    approach_speed_mps: float
    normal_restitution: float
    energy_before_j: float
    energy_after_j: float
    regime: str = ""
    blade_offset_m: tuple[float, float] | None = None
    surface_velocity: Array = field(default_factory=lambda: np.zeros(3))
    duration_s: float = 0.0
    steps: int = 1
    end: Array | None = None

    @property
    def side(self) -> str:
        """``8E`` for a contact at ``x >= 0``, else ``8F``."""

        return "8E" if self.position[0] >= 0.0 else "8F"

    @property
    def energy_gain_j(self) -> float:
        """Kinetic energy gained by the ball, J (positive is a violation for fixed surfaces)."""

        return self.energy_after_j - self.energy_before_j


def kinetic_energy(state: ArrayLike, mass: float, inertia: float) -> float:
    """Ball kinetic energy of a 9-float state (translation plus spin), J."""

    s = np.asarray(state, dtype=np.float64)
    return 0.5 * mass * float(s[3:6] @ s[3:6]) + 0.5 * inertia * float(s[6:9] @ s[6:9])


@dataclass(frozen=True)
class BladePose:
    """
    A paddle blade at one instant.

    :ivar position: Blade centre, m.
    :ivar rotation: Columns X (width), Y (towards the tip), Z (forehand normal).
    :ivar velocity: Blade-centre velocity, m/s.
    :ivar angular_velocity: rad/s.
    """

    position: Array
    rotation: Array
    velocity: Array
    angular_velocity: Array


@dataclass(frozen=True)
class SweptContact:
    """
    The first touch of the ball on a surface within a step.

    :ivar fraction: Time of contact as a fraction of the step, in [0, 1].
    :ivar ball: Ball state at contact.
    :ivar normal: Unit normal from the surface into the ball.
    :ivar surface_velocity: Velocity of the surface point at the contact, m/s.
    :ivar blade_offset_m: Contact point from the blade centre in blade axes (paddles).
    :ivar contact_offset: Contact point minus the blade centre (its centre of mass), table frame, m (paddles).
    :ivar surface_rotation: The blade's orthonormal axes at contact (columns: width, length, face normal) (paddles).
    :ivar depth: How far the ball's surface was inside the surface at the touch (a step that started inside), m; the
        imposed state moves the centre out along the normal by this much, so the touching point is on the surface.
    :ivar edge: A touch on a blade's rounded edge (resolved with ``rim_contact``; region ``blade_edge``).
    """

    fraction: float
    ball: Array
    normal: Array
    surface_velocity: Array
    blade_offset_m: tuple[float, float] | None
    contact_offset: Array | None = None
    surface_rotation: Array | None = None
    depth: float = 0.0
    edge: bool = False


def sweep_blade(
    ball: ArrayLike, start: BladePose, end: BladePose, dt: float, *, radius: float, half_thickness: float, half_width: float, half_length: float
) -> tuple[SweptContact | None, bool]:
    """
    Sweep the ball against both faces of a moving blade over one step.

    The ball moves at constant velocity and the blade's centre and normal move
    linearly between ``start`` and ``end`` (gravity changes the ball's position
    by at most 5 micrometres in a millisecond step). A contact is the first time
    the ball's surface reaches a face from outside, with the touching point inside
    the elliptical outline and the ball approaching the face.

    :param ball: Ball state at the step start.
    :param start: Blade at the step start.
    :param end: Blade at the step end.
    :param dt: Step, s.
    :param radius: Ball radius, m.
    :param half_thickness: Half the blade thickness with rubbers, m.
    :param half_width: Outline semi-axis along X, m.
    :param half_length: Outline semi-axis along Y, m.
    :return: ``(contact or None, tunnelled)``; ``tunnelled`` is true when the ball's centre crossed
        the blade's mid-plane inside the outline without a contact (a missed collision).
    """

    state = np.asarray(ball, dtype=np.float64)
    c0 = state[:3]
    v = state[3:6]
    c1 = c0 + v * dt
    n0, n1 = start.rotation[:, 2], end.rotation[:, 2]
    d0 = float(n0 @ (c0 - start.position))
    d1 = float(n1 @ (c1 - end.position))
    gap = radius + half_thickness

    def inside(fraction: float, sign: float) -> tuple[bool, tuple[float, float], Array, Array, Array]:
        position = start.position + fraction * (end.position - start.position)
        rotation = start.rotation + fraction * (end.rotation - start.rotation)
        normal = rotation[:, 2] / np.linalg.norm(rotation[:, 2])
        centre = c0 + v * fraction * dt
        point = centre - sign * normal * radius
        offset = point - position
        along_width = float(offset @ rotation[:, 0]) / max(float(np.linalg.norm(rotation[:, 0])), 1e-12)
        along_length = float(offset @ rotation[:, 1]) / max(float(np.linalg.norm(rotation[:, 1])), 1e-12)
        within = (along_width / half_width) ** 2 + (along_length / half_length) ** 2 <= 1.0
        velocity = start.velocity + fraction * (end.velocity - start.velocity)
        spin = start.angular_velocity + fraction * (end.angular_velocity - start.angular_velocity)
        surface_velocity = velocity + np.cross(spin, point - position)
        return within, (along_width, along_length), sign * normal, surface_velocity, centre

    def frame(fraction: float, sign: float) -> tuple[Array, Array]:
        """The contact point from the blade centre, and the blade's orthonormal axes, at ``fraction``."""

        position = start.position + fraction * (end.position - start.position)
        rotation = start.rotation + fraction * (end.rotation - start.rotation)
        u, _s, vt = np.linalg.svd(rotation)
        axes = u @ vt
        point = c0 + v * fraction * dt - sign * axes[:, 2] * radius
        return point - position, axes

    sign = 1.0 if d0 >= 0.0 else -1.0
    tunnelled = False
    if abs(d0) > gap:
        crossing = sign * d1 <= gap
        if not crossing:
            return None, False
        span = d0 - d1
        fraction = (d0 - sign * gap) / span if abs(span) > 1e-15 else 0.0
        fraction = min(max(fraction, 0.0), 1.0)
    else:
        fraction = 0.0
    within, offset, normal, surface_velocity, centre = inside(fraction, sign)
    approaching = float((v - surface_velocity) @ normal) < 0.0
    if not within or not approaching:
        if np.sign(d0) != np.sign(d1) and abs(d0) > gap:
            mid_within, *_ = inside(min(max(d0 / (d0 - d1), 0.0), 1.0), sign)
            tunnelled = bool(mid_within)
        return None, tunnelled
    contact_state = np.concatenate([centre, v, state[6:9]])
    contact_offset, axes = frame(fraction, sign)
    return (
        SweptContact(
            fraction=fraction,
            ball=contact_state,
            normal=normal,
            surface_velocity=surface_velocity,
            blade_offset_m=offset,
            contact_offset=contact_offset,
            surface_rotation=axes,
        ),
        False,
    )


TABLE_START_SLOP_M: float = 2e-4
"""A ball that starts a step up to this far below the playing surface, falling and over the table, touches it at once.

The sweep predicts the step's end from the start velocity, but PhysX integrates semi-implicitly (the step's gravity
and drag kick come first), so the step ends up to ``|a| dt^2`` lower (about 1e-5 m at 960 Hz), and the float32
read-back adds about 1e-6 m. A ball whose predicted end sat in that sliver above the plane started the next step just
below it and was never swept again: it fell through the table (found by the calibration harness on production,
about 0.5-1% of bounces at 960 Hz). The calibration harness's batched sweep carries the same slop
(:data:`lib.physics.engine.lab.vectorized.TABLE_START_SLOP_M`); deeper than it the ball can only be under the table.
"""


def sweep_table(ball: ArrayLike, dt: float, *, radius: float, half_length: float, half_width: float) -> SweptContact | None:
    """
    Sweep the ball against the playing surface (the hybrid table).

    :return: The first touch of the plane ``z = radius`` from above inside the rectangle, or ``None``; a falling ball
        that starts up to :data:`TABLE_START_SLOP_M` below the plane touches at fraction 0.
    """

    state = np.asarray(ball, dtype=np.float64)
    z0 = float(state[2]) - radius
    vz = float(state[5])
    z1 = z0 + vz * dt
    if z0 < -TABLE_START_SLOP_M or z1 > 0.0 or vz >= 0.0:
        return None
    fraction = min(max(z0 / (z0 - z1), 0.0), 1.0) if z0 != z1 else 0.0
    centre = state[:3] + state[3:6] * fraction * dt
    if abs(centre[0]) > half_length or abs(centre[1]) > half_width:
        return None
    return SweptContact(
        fraction=fraction,
        ball=np.concatenate([centre, state[3:9]]),
        normal=np.array([0.0, 0.0, 1.0]),
        surface_velocity=np.zeros(3),
        blade_offset_m=None,
        depth=max(-z0, 0.0),
    )


def sweep_net(ball: ArrayLike, dt: float, *, radius: float, half_thickness: float, half_span: float, height: float) -> tuple[SweptContact | None, bool]:
    """
    Sweep the ball against the net box over one step (the hybrid net): the calibration harness's batched sweep
    (:func:`lib.physics.engine.lab.vectorized.sweep_net`) for one ball, with the depth of a touch that starts inside.

    :param ball: Ball state at the step start.
    :param dt: Step, s.
    :return: ``(contact or None, tunnelled)``; ``tunnelled`` is true when the centre ends the step inside the box without a touch.
    """

    from lib.physics.engine.lab.vectorized import box_gap
    from lib.physics.engine.lab.vectorized import sweep_net as batched

    state = np.asarray(ball, dtype=np.float64)
    batch = batched(np, state[None, :3], state[None, 3:6], np.array([radius]), dt, half_thickness, half_span, height)
    tunnelled = bool(batch.tunnelled[0])
    if not bool(batch.contact[0]):
        return None, tunnelled
    fraction = float(batch.fraction[0])
    centre = np.asarray(batch.centre[0], dtype=np.float64)
    depth = 0.0
    if fraction == 0.0:
        gap, _normal = box_gap(np, centre[None, :], np.array([radius]), half_thickness, half_span, height)
        depth = max(-float(gap[0]), 0.0)
    contact = SweptContact(
        fraction=fraction,
        ball=np.concatenate([centre, state[3:9]]),
        normal=np.asarray(batch.normal[0], dtype=np.float64),
        surface_velocity=np.zeros(3),
        blade_offset_m=None,
        depth=depth,
    )
    return contact, tunnelled


BOX_SWEEP_SAMPLES: int = 16
"""Sub-step samples of :func:`sweep_box` (the net sweep's count); a crossing is then bracketed by bisection."""
BOX_SWEEP_BISECTIONS: int = 8
"""Bisections of the bracketed crossing: the touch is located to ``dt / (16 * 256)``, as in the net sweep."""

Box = tuple[tuple[float, float, float], tuple[float, float, float]]
"""An axis-aligned box as ``(low corner, high corner)``, table frame, m."""


def box_distance(position: ArrayLike, box: Box) -> tuple[float, tuple[float, float, float], bool]:
    """
    Distance from a point to an axis-aligned box, the unit normal from the box's closest point to the point, and
    whether the point is inside (then the distance is minus the depth to the nearest face, whose outward normal is given).

    Scalar (no numpy per call): it runs every physics step near the table.
    """

    x, y, z = (float(v) for v in np.asarray(position, dtype=np.float64)[:3])
    (lx, ly, lz), (hx, hy, hz) = box
    cx, cy, cz = min(max(x, lx), hx), min(max(y, ly), hy), min(max(z, lz), hz)
    dx, dy, dz = x - cx, y - cy, z - cz
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance > 1e-12:
        return distance, (dx / distance, dy / distance, dz / distance), False
    depths = (x - lx, hx - x, y - ly, hy - y, z - lz, hz - z)
    k = min(range(6), key=depths.__getitem__)
    normals = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
    return -depths[k], normals[k], True


def sweep_box(ball: ArrayLike, dt: float, *, radius: float, box: Box) -> tuple[SweptContact | None, bool]:
    """
    Sweep the ball against a static axis-aligned box over one step: faces, edges and corners.

    The algorithm is the net sweep's (:func:`lib.physics.engine.lab.vectorized.sweep_net`): the ball moves at
    constant velocity; the gap (distance to the box less the radius) is sampled at :data:`BOX_SWEEP_SAMPLES` points of
    the step and the first bracketed root bisected; the normal runs from the box's closest point to the centre at the
    touch, so an edge or a corner deflects the ball by its geometry. A ball that starts the step overlapping the box
    (centre outside it) touches at once and is moved out by its depth. A centre already inside the box is a tunnel that
    happened earlier: nothing physical is left to resolve, and it is reported, not pushed out.

    :param ball: Ball state at the step start.
    :param dt: Step, s.
    :param radius: Ball radius, m.
    :param box: The box.
    :return: ``(contact or None, tunnelled)``; ``tunnelled`` when the centre starts inside, or ends inside without a touch.
    """

    state = np.asarray(ball, dtype=np.float64)
    x, y, z, vx, vy, vz = (float(v) for v in state[:6])
    distance, normal, inside = box_distance((x, y, z), box)
    if inside:
        return None, True
    gap0 = distance - radius
    travel = math.sqrt(vx * vx + vy * vy + vz * vz) * dt
    if gap0 > travel + 1e-9:
        return None, False

    def gap_at(f: float) -> float:
        return box_distance((x + vx * f * dt, y + vy * f * dt, z + vz * f * dt), box)[0] - radius

    depth = 0.0
    if gap0 <= 0.0:
        fraction = 0.0
        depth = -gap0
    else:
        lo = hi = -1.0
        for k in range(1, BOX_SWEEP_SAMPLES + 1):
            f = k / BOX_SWEEP_SAMPLES
            if gap_at(f) <= 0.0:
                lo, hi = (k - 1) / BOX_SWEEP_SAMPLES, f
                break
        if hi < 0.0:
            return None, box_distance((x + vx * dt, y + vy * dt, z + vz * dt), box)[2]
        for _ in range(BOX_SWEEP_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if gap_at(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        fraction = lo
        _distance, normal, _inside = box_distance((x + vx * lo * dt, y + vy * lo * dt, z + vz * lo * dt), box)
    if vx * normal[0] + vy * normal[1] + vz * normal[2] >= 0.0:
        return None, False
    centre = state[:3] + state[3:6] * (fraction * dt)
    contact = SweptContact(
        fraction=fraction,
        ball=np.concatenate([centre, state[3:9]]),
        normal=np.array(normal),
        surface_velocity=np.zeros(3),
        blade_offset_m=None,
        depth=depth,
    )
    return contact, False


TABLE_TUNNEL_DEPTH_M: float = 0.005
"""A ball whose surface is more than this deep in the table board has gone through it (:func:`crossed_table`)."""


def table_board(half_length: float, half_width: float, thickness: float = TABLE_BOARD_THICKNESS_M) -> Box:
    """The table-top board as a box: the playing surface ``z = 0`` on top, ``thickness`` deep."""

    return ((-half_length, -half_width, -thickness), (half_length, half_width, 0.0))


def sweep_table_box(
    ball: ArrayLike, dt: float, *, radius: float, half_length: float, half_width: float, thickness: float = TABLE_BOARD_THICKNESS_M
) -> SweptContact | None:
    """
    Sweep the ball against the whole table-top board (the hybrid table): the playing surface, its edges and corners,
    the side and end faces and the underside.

    The playing surface is :func:`sweep_table`, unchanged (the reference model's and the calibration harness's table
    plane, slop included), so every bounce on the top is what it was. Only when it finds no touch is the board swept as
    a box (:func:`sweep_box`): an edge ball deflected by the edge's geometry, a ball reaching an end or side face (a
    dead ball coming back off a player: game 5, iteration 5, run b21788e2, entered the 8E end face and fell through the
    board, since the top plane was the table's only hybrid geometry and its native collider is filtered for the ball),
    or one rising into the underside off the floor. Every face takes the table's impact law (the stepper passes
    ``params.table_contact``; see :mod:`lib.physics.engine.colliders` for why the unmeasured faces share it).

    :return: The first touch, or ``None``.
    """

    top = sweep_table(ball, dt, radius=radius, half_length=half_length, half_width=half_width)
    if top is not None:
        return top
    contact, _tunnelled = sweep_box(ball, dt, radius=radius, box=table_board(half_length, half_width, thickness))
    return contact


@dataclass(frozen=True, eq=False)
class FrameGeometry:
    """
    The table's undercarriage for the hybrid sweep: the reference model's 38 parts (apron rails, legs, feet, casters,
    stretchers, spine, struts; :func:`lib.physics.reference.undercarriage.undercarriage_parts`, the table asset's frame)
    with their bounding boxes, so the adapter sweeps exactly the oracle's geometry. Every part takes the contract's
    ``fixture_contact`` (painted steel; judgement), as in the oracle.

    :ivar parts: The primitives.
    :ivar boxes: Each part's exact axis-aligned bounding box ``(x_lo, y_lo, z_lo, x_hi, y_hi, z_hi)``, m.
    :ivar bounds: The box around every part, m.
    """

    parts: tuple[Part, ...]
    boxes: tuple[tuple[float, float, float, float, float, float], ...]
    bounds: Box

    @classmethod
    def from_params(cls, params: BallTableParams) -> FrameGeometry:
        """The undercarriage of ``params``."""

        parts = undercarriage_parts(params)
        boxes = tuple(part.aabb() for part in parts)
        low = (min(b[0] for b in boxes), min(b[1] for b in boxes), min(b[2] for b in boxes))
        high = (max(b[3] for b in boxes), max(b[4] for b in boxes), max(b[5] for b in boxes))
        return cls(parts=parts, boxes=boxes, bounds=(low, high))

    def nearest(self, x: float, y: float, z: float) -> tuple[int, tuple[float, float, float, float]]:
        """The nearest part's index and its closest point and distance (0 inside it), m; bounding boxes prune exactly."""

        best = -1
        found: tuple[float, float, float, float] = (0.0, 0.0, 0.0, math.inf)
        for k, (lx, ly, lz, hx, hy, hz) in enumerate(self.boxes):
            dx, dy, dz = max(lx - x, 0.0, x - hx), max(ly - y, 0.0, y - hy), max(lz - z, 0.0, z - hz)
            if dx * dx + dy * dy + dz * dz >= found[3] * found[3]:
                continue
            candidate = self.parts[k].closest(x, y, z)
            if candidate[3] < found[3]:
                best, found = k, candidate
        return best, found


def sweep_frame(ball: ArrayLike, dt: float, *, radius: float, frame: FrameGeometry) -> tuple[SweptContact | None, str | None, bool]:
    """
    Sweep the ball against the table's undercarriage over one step (the hybrid frame).

    The algorithm is :func:`sweep_box`'s on the gap to the nearest part: a cheap rejection against the frame's bounding
    box, then (only within one step's travel of a part) :data:`BOX_SWEEP_SAMPLES` samples and
    :data:`BOX_SWEEP_BISECTIONS` bisections; the normal runs from the part's closest point to the centre at the touch.
    A centre already inside a part is a tunnel that happened earlier and is reported, not pushed out.

    :param ball: Ball state at the step start.
    :param dt: Step, s.
    :param radius: Ball radius, m.
    :param frame: The undercarriage (:meth:`FrameGeometry.from_params`).
    :return: ``(contact or None, the part's name or None, tunnelled)``.
    """

    state = np.asarray(ball, dtype=np.float64)
    x, y, z, vx, vy, vz = (float(v) for v in state[:6])
    travel = math.sqrt(vx * vx + vy * vy + vz * vz) * dt
    (lx, ly, lz), (hx, hy, hz) = frame.bounds
    dx, dy, dz = max(lx - x, 0.0, x - hx), max(ly - y, 0.0, y - hy), max(lz - z, 0.0, z - hz)
    if math.sqrt(dx * dx + dy * dy + dz * dz) - radius > travel + 1e-9:
        return None, None, False
    index, (_cx, _cy, _cz, distance) = frame.nearest(x, y, z)
    if distance <= 0.0:
        return None, None, True
    gap0 = distance - radius
    if gap0 > travel + 1e-9:
        return None, None, False

    def gap_at(f: float) -> float:
        return frame.nearest(x + vx * f * dt, y + vy * f * dt, z + vz * f * dt)[1][3] - radius

    depth = 0.0
    if gap0 <= 0.0:
        fraction = 0.0
        depth = -gap0
    else:
        lo = hi = -1.0
        for k in range(1, BOX_SWEEP_SAMPLES + 1):
            f = k / BOX_SWEEP_SAMPLES
            if gap_at(f) <= 0.0:
                lo, hi = (k - 1) / BOX_SWEEP_SAMPLES, f
                break
        if hi < 0.0:
            end = frame.nearest(x + vx * dt, y + vy * dt, z + vz * dt)[1][3]
            return None, None, end <= 0.0
        for _ in range(BOX_SWEEP_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if gap_at(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        fraction = lo
    px, py, pz = x + vx * fraction * dt, y + vy * fraction * dt, z + vz * fraction * dt
    index, (cx, cy, cz, distance) = frame.nearest(px, py, pz)
    if distance <= 1e-12:
        return None, None, True
    normal = ((px - cx) / distance, (py - cy) / distance, (pz - cz) / distance)
    if vx * normal[0] + vy * normal[1] + vz * normal[2] >= 0.0:
        return None, None, False
    contact = SweptContact(
        fraction=fraction,
        ball=np.concatenate([[px, py, pz], state[3:9]]),
        normal=np.array(normal),
        surface_velocity=np.zeros(3),
        blade_offset_m=None,
        depth=depth,
    )
    return contact, frame.parts[index].name, False


@dataclass(frozen=True, eq=False)
class FixtureGeometry:
    """
    The net posts and clamps as hybrid static shapes: the reference model's ``NetFixture``s
    (:func:`lib.physics.reference.surfaces.net_fixtures`), each as the calibration harness's :data:`Shape` (a capped
    vertical cylinder per post, an axis-aligned box per clamp), with the contract's law (``post_contact``, steel, for the
    posts; ``fixture_contact`` for the clamps) and one bounding box per side of the table for a cheap rejection.

    :ivar names: Fixture names (``net_post_neg_y``...).
    :ivar regions: ``post`` or ``clamp``.
    :ivar shapes: The shapes, for :func:`lib.physics.engine.lab.vectorized.sweep_shape`.
    :ivar materials: Each fixture's impact law.
    :ivar sides: Per side of the table, ``(bounding box, indices of its fixtures)``.
    """

    names: tuple[str, ...]
    regions: tuple[str, ...]
    shapes: tuple[Any, ...]
    materials: tuple[ContactMaterial, ...]
    sides: tuple[tuple[Box, tuple[int, ...]], ...]

    @classmethod
    def from_params(cls, params: BallTableParams) -> FixtureGeometry:
        """The posts and clamps of ``params``, exactly as the reference model and the calibration harness build them."""

        from lib.physics.reference.surfaces import net_fixtures

        names, regions, shapes, materials, lows, highs = [], [], [], [], [], []
        for fixture in net_fixtures(params):
            c, h = fixture.centre, fixture.size
            if fixture.shape == "box":
                low, high = (c[0] - h[0], c[1] - h[1], c[2] - h[2]), (c[0] + h[0], c[1] + h[1], c[2] + h[2])
                shape: Any = ("box", low, high, (0.0, 0.0))
            else:
                low = (c[0] - h[0], c[1] - h[0], fixture.z_range[0])
                high = (c[0] + h[0], c[1] + h[0], fixture.z_range[1])
                shape = ("cylinder", tuple(c), tuple(h), tuple(fixture.z_range))
            names.append(fixture.name)
            regions.append(fixture.region)
            shapes.append(shape)
            materials.append(fixture.material)
            lows.append(low)
            highs.append(high)
        sides = []
        for sign in (-1.0, 1.0):
            members = tuple(k for k in range(len(shapes)) if (lows[k][1] + highs[k][1]) * sign > 0.0)
            box_low = tuple(min(lows[k][i] for k in members) for i in range(3))
            box_high = tuple(max(highs[k][i] for k in members) for i in range(3))
            sides.append((((box_low[0], box_low[1], box_low[2]), (box_high[0], box_high[1], box_high[2])), members))
        return cls(names=tuple(names), regions=tuple(regions), shapes=tuple(shapes), materials=tuple(materials), sides=tuple(sides))


def sweep_fixtures(ball: ArrayLike, dt: float, *, radius: float, fixtures: FixtureGeometry) -> tuple[SweptContact | None, int | None, bool]:
    """
    Sweep the ball against the net posts and clamps over one step (hybrid): the calibration harness's
    :func:`lib.physics.engine.lab.vectorized.sweep_shape` on each fixture within one step's reach of the ball (a
    scalar bounding-box test per side first, so a ball away from the side lines costs two distance checks), the
    earliest touch winning.

    :return: ``(contact or None, the fixture's index or None, tunnelled)``.
    """

    from lib.physics.engine.lab.vectorized import sweep_shape

    state = np.asarray(ball, dtype=np.float64)
    vx, vy, vz = (float(v) for v in state[3:6])
    reach = radius + math.sqrt(vx * vx + vy * vy + vz * vz) * dt + 1e-9
    candidates = [k for box, members in fixtures.sides if box_distance(state[:3], box)[0] <= reach for k in members]
    if not candidates:
        return None, None, False
    best: tuple[float, int, Any] | None = None
    tunnelled = False
    for k in candidates:
        batch = sweep_shape(np, state[None, :3], state[None, 3:6], np.array([radius]), dt, fixtures.shapes[k])
        tunnelled = tunnelled or bool(batch.tunnelled[0])
        if bool(batch.contact[0]) and (best is None or float(batch.fraction[0]) < best[0]):
            best = (float(batch.fraction[0]), k, batch)
    if best is None:
        return None, None, tunnelled
    fraction, k, batch = best
    contact = SweptContact(
        fraction=fraction,
        ball=np.concatenate([np.asarray(batch.centre[0], dtype=np.float64), state[3:9]]),
        normal=np.asarray(batch.normal[0], dtype=np.float64),
        surface_velocity=np.zeros(3),
        blade_offset_m=None,
        depth=float(batch.depth[0]) if batch.depth is not None else 0.0,
    )
    return contact, k, tunnelled


def sweep_floor(ball: ArrayLike, dt: float, *, radius: float, floor_z: float) -> SweptContact | None:
    """
    Sweep the ball against the floor plane ``z = floor_z`` from above (the hybrid floor): the table plane's sweep
    (:func:`sweep_table`) without a boundary, slop included, so a ball rolling on the floor touches every step at
    fraction 0 and is resolved as a sustained contact (:func:`impose_contact` with ``rest_speed``).

    Native PhysX friction on the floor turned a rolling ball's spin about the vertical into sideways velocity and
    gained 14 % kinetic energy (game 6, iteration 6, run 0640a75a, event 632); the reference law applies no torque
    about the normal and never gains energy.

    :return: The touch, or ``None``.
    """

    state = np.asarray(ball, dtype=np.float64)
    z0 = float(state[2]) - radius - floor_z
    vz = float(state[5])
    z1 = z0 + vz * dt
    if z0 < -TABLE_START_SLOP_M or z1 > 0.0 or vz >= 0.0:
        return None
    fraction = min(max(z0 / (z0 - z1), 0.0), 1.0) if z0 != z1 else 0.0
    centre = state[:3] + state[3:6] * fraction * dt
    return SweptContact(
        fraction=fraction,
        ball=np.concatenate([centre, state[3:9]]),
        normal=np.array([0.0, 0.0, 1.0]),
        surface_velocity=np.zeros(3),
        blade_offset_m=None,
        depth=max(-z0, 0.0),
    )


ROOM_CLEAR_Y_M: float = min(box.low[1] for box in ROOM_BOXES if not box.name.startswith(("wall_", "ceiling")))
"""Everything in the room beyond the walls and ceiling (the furniture) stands at ``y`` above this, m."""


def sweep_room(ball: ArrayLike, dt: float, *, radius: float) -> tuple[SweptContact | None, str | None, bool]:
    """
    Sweep the ball against the room's walls, ceiling and furniture (:data:`lib.physics.engine.colliders.ROOM_BOXES`,
    the hybrid room): one scalar test while the ball is clear of all of them, else :func:`sweep_box` per box, the
    earliest touch winning.

    :return: ``(contact or None, the box's name or None, tunnelled)``.
    """

    state = np.asarray(ball, dtype=np.float64)
    x, y, z, vx, vy, vz = (float(v) for v in state[:6])
    reach = radius + math.sqrt(vx * vx + vy * vy + vz * vz) * dt + 1e-9
    ceiling = ROOM_HEIGHT_M - SURFACE_HEIGHT_M
    if ROOM_X_M[0] + reach < x < ROOM_X_M[1] - reach and ROOM_Y_M[0] + reach < y < ROOM_CLEAR_Y_M - reach and z < ceiling - reach:
        return None, None, False
    best: tuple[SweptContact, str] | None = None
    tunnelled = False
    for box in ROOM_BOXES:
        contact, inside = sweep_box(state, dt, radius=radius, box=(box.low, box.high))
        tunnelled = tunnelled or inside
        if contact is not None and (best is None or contact.fraction < best[0].fraction):
            best = (contact, box.name)
    if best is None:
        return None, None, tunnelled
    return best[0], best[1], tunnelled


def crossed_frame(before: ArrayLike, after: ArrayLike, frame: FrameGeometry, radius: float) -> bool:
    """Whether the ball went into the undercarriage within one step: its surface more than :data:`TABLE_TUNNEL_DEPTH_M` deep in a part."""

    s0 = np.asarray(before, dtype=np.float64)
    s1 = np.asarray(after, dtype=np.float64)
    (lx, ly, lz), (hx, hy, hz) = frame.bounds
    x, y, z = float(s1[0]), float(s1[1]), float(s1[2])
    if max(lx - x, 0.0, x - hx, ly - y, y - hy, lz - z, z - hz) > radius:
        return False
    deep = radius - TABLE_TUNNEL_DEPTH_M
    return frame.nearest(x, y, z)[1][3] < deep <= frame.nearest(float(s0[0]), float(s0[1]), float(s0[2]))[1][3]


@dataclass(frozen=True)
class ImposedState:
    """
    What to write into the engine for a hybrid contact.

    :ivar start_position: Ball position to write at the step start, m, so the engine's step ends where the event-driven solution is.
    :ivar velocity: Velocity to write at the step start, m/s (the post-impact velocity less the rest of the step's
        acceleration it will receive before the contact time).
    :ivar spin: Angular velocity to write, rad/s (likewise).
    :ivar after: Post-impact state at the contact: centre, velocity, spin.
    :ivar before: Pre-impact state at the contact: centre and the step-start velocity and spin carried to the contact time.
    :ivar regime: Impact branch (``slip`` or ``stick``; ``rolling`` for a sustained contact).
    :ivar sustained: The touch was a sustained (rolling or resting) contact, not a bounce (see :data:`REST_SPEED_MPS`).
    """

    start_position: Array
    velocity: Array
    spin: Array
    after: Array
    regime: str
    before: Array
    sustained: bool = False


def sustained_material(material: ContactMaterial) -> ContactMaterial:
    """
    The law of a sustained contact: no normal rebound and Coulomb friction without tangential restitution.

    Restitution and tangential restitution describe the compression and recovery of one impact; a ball resting or
    rolling on a surface has neither, and re-applying them every step would make it chatter. Friction (and its speed
    curve) is kept. Spin about the normal is kept too (``normal_spin_retention`` 1), since a per-impact fraction applied
    every step would remove it within milliseconds.
    """

    return material.model_copy(update={"restitution": 0.0, "restitution_curve": None, "tangential_restitution": 0.0, "normal_spin_retention": 1.0})


def blade_inertia(params: BallTableParams, axes: Array) -> Array:
    """
    The paddle's inertia tensor in the table frame, ``R diag(I) R^T``, for blade axes in this adapter's convention.

    The adapter's blade axes are (width, length, face normal); ``lib.physics``'s paddle frame is (towards the handle,
    across, normal), so the principal moments swap their first two places.
    """

    paddle = params.paddle
    return axes @ np.diag([paddle.inertia_yy, paddle.inertia_xx, paddle.inertia_zz]) @ axes.T


def impose_contact(
    contact: SweptContact,
    material: ContactMaterial,
    params: BallTableParams,
    dt: float,
    *,
    effective_mass: float,
    rigid_body: bool = False,
    flight: Any = None,
    rest_speed: float | None = None,
    rolling: bool = False,
) -> ImposedState | None:
    """
    Resolve a swept contact with the reference impact model at the contact time, and the state to write.

    The ball reaches the contact ``f dt`` into the step, having accelerated at ``a`` (gravity and the air force of the
    step-start state) from the step-start velocity, so the impact meets ``v + a f dt`` (and ``w + alpha f dt``). It
    leaves with ``v'`` and accelerates at ``a'`` for the rest of the step. The engine integrates the whole step
    semi-implicitly from what is written (``v_end = v_s + a' dt``, ``x_end = x_s + v_end dt``), so landing on the
    event-driven ``v' + a' (1 - f) dt`` and ``c + v' (1 - f) dt + a' ((1 - f) dt)^2 / 2`` needs
    ``v_s = v' - a' f dt`` and ``x_s = c - v' f dt - a' (1 - f)(1 + f) dt^2 / 2`` (the calibration harness's
    :func:`lib.physics.engine.lab.vectorized.imposed_state`). Without these the step after every contact carried an
    extra ``a f dt`` that depended on the contact's phase within the step.

    :param contact: The detected touch (its ball state carries the step-start velocity and spin).
    :param material: The pair's impact law.
    :param params: Ball geometry and mass (and the paddle's inertia).
    :param dt: Step, s.
    :param effective_mass: Surface effective mass, kg (``math.inf`` for fixed surfaces).
    :param rigid_body: Meet the ball as a free rigid paddle (effective mass plus the paddle inertia about the blade
        centre, ``lib.physics.reference.impact``'s off-centre law) when the contact carries its offset and axes.
    :param flight: The flight model whose ``wrench(velocity, spin)`` gives the air force and torque (the stepper's);
        ``None`` treats the step as unaccelerated (gravity included only through ``flight``).
    :param rest_speed: For static and body surfaces, :data:`REST_SPEED_MPS`: a touch whose impact would separate slower
        than this while gravity presses the ball on is resolved as a sustained contact (:func:`sustained_material`) and
        flagged ``sustained``. ``None`` (paddles, and the calibration-compatible behaviour) always resolves an impact.
    :param rolling: The floor and the playing surface (:func:`roll_on_flat`): a sustained touch on their flat top that leaves
        the ball rolling loses the surface's rolling resistance over the step, and stops below :data:`STOP_SPEED_MPS`.
    :return: The state to write, or ``None`` if the impact model finds the contact separating.
    """

    state = contact.ball
    centre = state[:3] + contact.normal * contact.depth
    lead = contact.fraction * dt
    rest = dt - lead
    gravity = np.array([0.0, 0.0, -params.aero.gravity])

    def accelerations(velocity: Array, spin: Array) -> tuple[Array, Array]:
        if flight is None:
            return np.zeros(3), np.zeros(3)
        force, torque = flight.wrench(velocity, spin)
        return gravity + np.asarray(force) / params.ball.mass, np.asarray(torque) / params.ball.inertia

    acceleration, angular_acceleration = accelerations(state[3:6], state[6:9])
    velocity = state[3:6] + acceleration * lead
    spin = state[6:9] + angular_acceleration * lead
    rigid = rigid_body and math.isfinite(effective_mass) and contact.contact_offset is not None and contact.surface_rotation is not None
    impact = resolve_impact(
        velocity,
        spin,
        params.ball,
        contact.normal,
        material,
        surface_velocity=contact.surface_velocity,
        effective_mass=effective_mass,
        contact_offset=contact.contact_offset if rigid else None,
        surface_inertia=blade_inertia(params, contact.surface_rotation) if rigid and contact.surface_rotation is not None else None,
    )
    if impact.regime == "separating":
        return None
    sustained = False
    if rest_speed is not None and not rigid and impact.separation_speed < rest_speed and float(gravity @ contact.normal) < 0.0:
        sustained = True
        impact = resolve_impact(
            velocity,
            spin,
            params.ball,
            contact.normal,
            sustained_material(material),
            surface_velocity=contact.surface_velocity,
            effective_mass=effective_mass,
        )
    after_velocity, after_spin = impact.velocity, impact.spin
    if sustained and rolling and impact.regime == "stick":
        support = max(-float(acceleration @ contact.normal), 0.0) if flight is not None else params.aero.gravity * float(contact.normal[2])
        after_velocity, after_spin = roll_on_flat(after_velocity, after_spin, contact.normal, material, support, dt, params)
    after_acceleration, after_angular = accelerations(after_velocity, after_spin)
    return ImposedState(
        start_position=centre - after_velocity * lead - 0.5 * after_acceleration * rest * (dt + lead),
        velocity=after_velocity - after_acceleration * lead,
        spin=after_spin - after_angular * lead,
        after=np.concatenate([centre, after_velocity, after_spin]),
        regime="rolling" if sustained else impact.regime,
        before=np.concatenate([centre, velocity, spin]),
        sustained=sustained,
    )


def roll_on_flat(
    velocity: Array, spin: Array, normal: Array, material: ContactMaterial, support: float, dt: float, params: BallTableParams
) -> tuple[Array, Array]:
    """
    One step of rolling resistance on a flat top (the reference model's rolling regime,
    :mod:`lib.physics.reference.rolling`): a ball rolling without slip loses ``C_rr N dt / (m (1 + k))`` of horizontal
    speed along its motion, never reversing it, with its spin kept consistent with rolling, ``dw = (n x dv) / R``; slower
    than :data:`STOP_SPEED_MPS` it has stopped (horizontal velocity and spin zero; spin about the normal kept). Only
    flat tops (:data:`FLAT_NORMAL_Z`) take it, so the board's edge never holds a ball and the cord stays sustained.

    :param velocity: The ball's velocity after the sustained touch, m/s.
    :param spin: Its spin, rad/s.
    :param normal: The contact normal.
    :param material: The surface's law (``rolling_resistance`` is ``C_rr``).
    :param support: The normal force per unit mass, ``g - a_air,n``, m/s^2.
    :param dt: Step, s.
    :param params: Ball radius and inertia ratio.
    :return: The velocity and spin after the step's resistance.
    """

    n = np.asarray(normal, dtype=np.float64)
    if float(n[2]) <= FLAT_NORMAL_Z:
        return velocity, spin
    v = np.asarray(velocity, dtype=np.float64).copy()
    w = np.asarray(spin, dtype=np.float64).copy()
    radius = params.ball.radius
    horizontal = v - float(v @ n) * n
    speed = float(np.linalg.norm(horizontal))
    if speed < STOP_SPEED_MPS:
        w_normal = float(w @ n)
        return v - horizontal, w_normal * n
    loss = min(material.rolling_resistance * support * dt / (1.0 + params.ball.inertia_ratio), speed)
    delta = -loss * horizontal / speed
    return v + delta, w + np.cross(n, delta) / radius


class ContactEpisodes:
    """
    Folds the per-step touches of a sustained contact into one :class:`ContactEvent`.

    The stepper asks :meth:`touch` before recording a hybrid contact. A bounce, or the first step of a sustained
    contact, is recorded as a new event; the following sustained steps on the same surface (no more than
    ``max_gap_steps`` apart) extend that event instead (:func:`extend_event`): its ``end`` state becomes the latest,
    its ``duration_s`` and ``steps`` grow. A bounce on the surface closes its episode.

    :param max_gap_steps: Steps a sustained contact may skip and still be the same episode.
    """

    def __init__(self, max_gap_steps: int = 2) -> None:
        self.max_gap_steps = max_gap_steps
        self._open: dict[str, tuple[int, int]] = {}

    def touch(self, key: str, step: int, sustained: bool, next_index: int) -> int | None:
        """
        Classify one touch.

        :param key: The surface (``table``, ``net``, ``body:<role>``...).
        :param step: Physics step index.
        :param sustained: Whether :func:`impose_contact` resolved it as sustained.
        :param next_index: The index a new event would take (``len(events)``).
        :return: The index of the open event this touch extends, or ``None`` when the caller records a new event.
        """

        current = self._open.get(key)
        if sustained and current is not None and step - current[1] <= self.max_gap_steps:
            self._open[key] = (current[0], step)
            return current[0]
        if sustained:
            self._open[key] = (next_index, step)
        else:
            self._open.pop(key, None)
        return None

    def close(self, key: str | None = None) -> None:
        """End the open episode on one surface (or on every surface): the ball was placed or parked."""

        if key is None:
            self._open.clear()
        else:
            self._open.pop(key, None)


def extend_event(event: ContactEvent, after: ArrayLike, time_s: float) -> ContactEvent:
    """
    A sustained contact's event carried to a later step: the latest ball state (``end``), the duration and the step count.

    :param event: The episode's event.
    :param after: The ball state after this step's touch.
    :param time_s: This touch's time, s.
    """

    return dataclasses.replace(event, end=np.asarray(after, dtype=np.float64).copy(), duration_s=max(time_s - event.time_s, 0.0), steps=event.steps + 1)


@dataclass(frozen=True)
class NativeContact:
    """
    An engine-resolved contact recognised from one step's velocity change.

    :ivar surface: Classified surface.
    :ivar normal: Its normal into the ball.
    :ivar approach_speed_mps: Normal approach speed before, m/s.
    :ivar normal_restitution: ``-v_n' / v_n``.
    """

    surface: Surface
    normal: Array
    approach_speed_mps: float
    normal_restitution: float


def classify_native(
    before: ArrayLike, after: ArrayLike, dt: float, params: BallTableParams, *, applied_dv: ArrayLike = (0.0, 0.0, 0.0)
) -> NativeContact | None:
    """
    Recognise an engine contact between two consecutive step states.

    :param before: State at the step start.
    :param after: State at the step end.
    :param dt: Step, s.
    :param params: Geometry.
    :param applied_dv: Velocity change the adapter itself applied this step (air force), m/s.
    :return: The contact, or ``None`` when the step was free flight.
    """

    s0 = np.asarray(before, dtype=np.float64)
    s1 = np.asarray(after, dtype=np.float64)
    gravity = np.array([0.0, 0.0, -params.aero.gravity])
    dv = s1[3:6] - s0[3:6] - gravity * dt - np.asarray(applied_dv, dtype=np.float64)
    if float(np.linalg.norm(dv)) < NATIVE_CONTACT_DV_MPS:
        return None
    radius = params.ball.radius
    half_length, half_width = 0.5 * params.table.length, 0.5 * params.table.width
    position = 0.5 * (s0[:3] + s1[:3])
    margin = 0.015
    surface: Surface
    if abs(position[0]) <= half_length + margin and abs(position[1]) <= half_width + margin and abs(position[2] - radius) < radius + margin:
        surface, normal = "table", np.array([0.0, 0.0, 1.0])
    elif (
        abs(position[0]) <= 0.5 * params.net.thickness + radius + margin
        and abs(position[1]) <= params.net_half_span + radius
        and -(radius + margin) <= position[2] <= params.net.height + radius + margin
    ):
        # The net box (0 <= z <= height) grown by the radius and the margin, below as well as above: a ball must reach
        # above the table plane to touch it. (Unbounded below, game 7's dead ball bouncing on the floor under the net,
        # run 2529abad events 325-329 at z = -0.74, was labelled ``net``.)
        surface, normal = "net", np.array([math.copysign(1.0, s0[0] if s0[0] != 0.0 else -s0[3]), 0.0, 0.0])
    elif (room := room_gap(position, radius))[0] < margin and (
        position[2] - (-params.table.height) >= radius + margin or abs(float(room[1] @ dv)) > abs(float(dv[2]))
    ):
        # A wall, the ceiling or a cabinet; at floor level only when the kick is along the box's normal rather than up
        # (a ball rolling along the floor into a wall; measured on production, probe c9aab2fb).
        surface, normal = "wall", room[1]
    elif position[2] - (-params.table.height) < radius + margin:
        surface, normal = "floor", np.array([0.0, 0.0, 1.0])
    else:
        surface, normal = "other", dv / float(np.linalg.norm(dv))
    vn0 = float(s0[3:6] @ normal)
    vn1 = float(s1[3:6] @ normal)
    restitution = -vn1 / vn0 if vn0 < -1e-6 else math.nan
    return NativeContact(surface=surface, normal=normal, approach_speed_mps=max(-vn0, 0.0), normal_restitution=restitution)


def crossed_table(before: ArrayLike, after: ArrayLike, params: BallTableParams) -> bool:
    """
    Whether the ball's centre went into the table within one step (tunnelling).

    A ball that touches the table has its centre a radius above the surface, so a centre crossing ``z = 0`` inside
    the rectangle passed through the top, whether in one step or sinking slowly over several (a missed hybrid touch).
    A ball whose surface reaches more than :data:`TABLE_TUNNEL_DEPTH_M` into the board (:func:`table_board`) through an
    edge, a side or end face or the underside is a tunnel too: the hybrid board leaves it micrometres deep at most. (Run
    b21788e2's ball cut through the bottom corner of the 8E end with its centre never inside the board.)
    """

    s0 = np.asarray(before, dtype=np.float64)
    s1 = np.asarray(after, dtype=np.float64)
    half_length, half_width = 0.5 * params.table.length, 0.5 * params.table.width
    inside = abs(s1[0]) <= half_length and abs(s1[1]) <= half_width
    if inside and s0[2] >= 0.0 > s1[2]:
        return True
    board = table_board(half_length, half_width)
    deep = params.ball.radius - TABLE_TUNNEL_DEPTH_M
    distance, normal, inside_board = box_distance(s1[:3], board)
    if inside_board:
        return not box_distance(s0[:3], board)[2]
    through_top = normal[2] > 1.0 - 1e-9  # the playing surface keeps the centre-crossing test above
    return not through_top and distance < deep <= box_distance(s0[:3], board)[0]


def crossed_net(before: ArrayLike, after: ArrayLike, params: BallTableParams) -> bool:
    """Whether the ball's centre passed through the net below its tape within one step (tunnelling)."""

    s0 = np.asarray(before, dtype=np.float64)
    s1 = np.asarray(after, dtype=np.float64)
    if np.sign(s0[0]) == np.sign(s1[0]) or s0[0] == 0.0:
        return False
    fraction = s0[0] / (s0[0] - s1[0])
    crossing = s0[:3] + fraction * (s1[:3] - s0[:3])
    return bool(abs(crossing[1]) < params.net_half_span - params.ball.radius and 0.0 < crossing[2] < params.net.height - params.ball.radius)


def crossed_floor(after: ArrayLike, params: BallTableParams, *, tolerance: float = 0.005) -> bool:
    """Whether the ball's centre is below the floor (tunnelling)."""

    return bool(float(np.asarray(after)[2]) < -params.table.height - tolerance)


__all__ = [
    "BOX_SWEEP_BISECTIONS",
    "BOX_SWEEP_SAMPLES",
    "FLAT_NORMAL_Z",
    "NATIVE_CONTACT_DV_MPS",
    "NATIVE_MERGE_S",
    "REST_SPEED_MPS",
    "STOP_SPEED_MPS",
    "TABLE_START_SLOP_M",
    "TABLE_TUNNEL_DEPTH_M",
    "BladePose",
    "Box",
    "ContactEpisodes",
    "ContactEvent",
    "FixtureGeometry",
    "FrameGeometry",
    "ImposedState",
    "NativeContact",
    "Surface",
    "SweptContact",
    "blade_inertia",
    "box_distance",
    "classify_native",
    "crossed_floor",
    "crossed_frame",
    "crossed_net",
    "crossed_table",
    "extend_event",
    "impose_contact",
    "kinetic_energy",
    "roll_on_flat",
    "sustained_material",
    "sweep_blade",
    "sweep_box",
    "sweep_fixtures",
    "sweep_floor",
    "sweep_frame",
    "sweep_net",
    "sweep_room",
    "sweep_table",
    "sweep_table_box",
    "table_board",
]
