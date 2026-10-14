"""
Contact geometry for the reference model: the table top, the net, the floor and scripted paddles.

Each surface exposes a signed ``gap`` (metres; positive while the ball is
clear) that the flight integrator watches for a sign change, and a
``contact`` that, at the root, says whether the crossing is a real contact and
returns its frame: point, normal, surface velocity, material and effective
mass.

Everything is in the table frame: Z-up, origin at the centre of the playing
surface, +X along the table length, the net in the plane ``x = 0`` and the
floor at ``z = -table.height``.

The table is two surfaces. :class:`TableTop` is the playing surface, the
plane the reference has always used, and its behaviour is unchanged.
:class:`TableBoard` is the rest of the top board: its ends, sides, edges,
corners and underside. That is the box the game adapter sweeps
(``engine/contacts.py`` ``table_board``, 25 mm deep). The net assembly is the
net box (:class:`Net`), two post cylinders and two clamp boxes
(:class:`NetFixture`), matching ``scene/colliders.py``. The undercarriage is in
:mod:`lib.physics.reference.undercarriage`.

Deliberately not modelled (each is a documented limitation of the oracle, not
of the simulator under test):

- the paddle edge: only the two elliptical faces collide;
- the paddle's own response beyond the impulse: its script carries on unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.physics.params import BallTableParams, ContactMaterial
from lib.physics.reference.blade import blade_axes, blade_contact, blade_gap, blade_solid_contact, blade_solid_gap

SurfaceKind = Literal["table", "net", "floor", "paddle", "frame", "wall"]


@dataclass(frozen=True)
class Contact:
    """
    The frame of one contact, evaluated at the root of a surface's gap.

    :param kind: Which kind of surface.
    :param name: The surface's name (a paddle's own name, otherwise the kind).
    :param point: Contact point on the surface, m (table frame).
    :param normal: Unit normal from the surface into the ball.
    :param surface_velocity: Velocity of the surface's material point at the contact, m/s.
    :param material: Ball-surface impact law.
    :param effective_mass: Surface mass, kg (``math.inf`` for fixed surfaces).
    :param offset: Contact point relative to a rigid surface body's centre of mass, m; ``None`` for a point mass.
    :param inertia: That body's inertia tensor about its centre of mass in the table frame, kg m^2; ``None`` for a point mass.
    :param region: For the net, ``mesh`` (the compliant face below the top tape) or ``cord`` (the
        tape, top edge and ends); empty for other surfaces.
    """

    kind: SurfaceKind
    name: str
    point: NDArray[np.float64]
    normal: NDArray[np.float64]
    surface_velocity: NDArray[np.float64]
    material: ContactMaterial
    effective_mass: float
    offset: NDArray[np.float64] | None = None
    inertia: NDArray[np.float64] | None = None
    region: str = ""


class Surface(Protocol):
    """A surface the flight integrator watches."""

    @property
    def kind(self) -> SurfaceKind: ...

    @property
    def name(self) -> str: ...

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Signed clearance between ball and surface, m; negative once crossed."""
        ...

    def contact(self, t: float, state: Sequence[float]) -> Contact | None:
        """The contact frame at a root of :meth:`gap`, or ``None`` when the crossing is not a contact."""
        ...


_ZERO3 = np.zeros(3)
_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class TableTop:
    """
    The playing surface: the plane ``z = 0`` bounded by the table rectangle.

    The gap is ``z - R`` everywhere, so crossing the plane outside the
    rectangle is still a root; :meth:`contact` returns ``None`` there and the
    caller records a ``table_plane_outside`` event.

    :param radius: Ball radius, m.
    :param half_length: Half the table length, m.
    :param half_width: Half the table width, m.
    :param material: Ball-table impact law.
    """

    radius: float
    half_length: float
    half_width: float
    material: ContactMaterial
    kind: SurfaceKind = "table"
    name: str = "table"

    @classmethod
    def from_params(cls, params: BallTableParams) -> TableTop:
        """The table top of ``params``."""

        return cls(params.ball.radius, 0.5 * params.table.length, 0.5 * params.table.width, params.table_contact)

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Height of the ball's lowest point above the table plane, m."""

        return state[2] - self.radius

    def clearance(self, state: Sequence[float], gap: float) -> float:
        """How far the centre can move without the gap changing sign, m (the gap is 1-Lipschitz)."""

        return abs(gap)

    def contains(self, x: float, y: float) -> bool:
        """Whether a point's ``(x, y)`` lies on the playing surface, edges included."""

        return abs(x) <= self.half_length and abs(y) <= self.half_width

    def overhangs(self, x: float, y: float) -> bool:
        """Whether a centre at ``(x, y)`` off the table is still within one radius of it, over the board's top edge."""

        dx = max(abs(x) - self.half_length, 0.0)
        dy = max(abs(y) - self.half_width, 0.0)
        return dx * dx + dy * dy < self.radius * self.radius

    def contact(self, t: float, state: Sequence[float]) -> Contact | None:
        """A contact when the point below the ball is on the table, else ``None``."""

        x, y = state[0], state[1]
        if not self.contains(x, y):
            return None
        return Contact(self.kind, self.name, np.array([x, y, 0.0]), _UP.copy(), _ZERO3.copy(), self.material, math.inf, region="top")


@dataclass(frozen=True)
class Net:
    """
    The net: the box ``|x| <= thickness/2``, ``|y| <= half_span``, ``0 <= z <= height``.

    The gap is the distance from the ball centre to the box minus the radius,
    so the face, the top cord and the ends are all covered, and the contact
    normal points from the closest point of the box to the ball centre.

    A contact on a face below the top tape is the ``mesh``: it uses
    ``material`` under the ``catch`` law, so the net catches the ball instead
    of acting as a rigid wall. Everything else (the tape band, the top edge
    and the ends) is the ``cord`` and uses ``material`` as a rigid contact.

    :param radius: Ball radius, m.
    :param half_thickness: Half the net thickness along X, m.
    :param half_span: Half the net length along Y, m.
    :param height: Net top above the table, m.
    :param material: Ball-net law; its ``law`` field is overridden per region.
    :param tape_height: Height of the rigid top tape below the net top, m.
    """

    radius: float
    half_thickness: float
    half_span: float
    height: float
    material: ContactMaterial
    kind: SurfaceKind = "net"
    name: str = "net"
    tape_height: float = 0.0

    @classmethod
    def from_params(cls, params: BallTableParams) -> Net:
        """The net of ``params``."""

        return cls(
            params.ball.radius, 0.5 * params.net.thickness, params.net_half_span, params.net.height, params.net_contact, tape_height=params.net.tape_height
        )

    def region(self, point: ArrayLike) -> Literal["mesh", "cord"]:
        """
        Which part of the net a contact point (m) on the box belongs to.

        ``mesh``: on a face (``|x|`` at the half thickness), within the span and
        below the tape. ``cord``: the tape, the top edge or an end.
        """

        x, y, z = (float(c) for c in np.asarray(point, dtype=float).reshape(3))
        on_face = abs(x) >= self.half_thickness - 1e-12 and abs(y) < self.half_span and z < self.height - self.tape_height
        return "mesh" if on_face else "cord"

    def material_at(self, point: ArrayLike) -> ContactMaterial:
        """The law for a contact point: ``catch`` on the mesh, ``rigid`` on the cord."""

        if self.region(point) == "mesh":
            return self.material.model_copy(update={"law": "catch"})
        return self.material.model_copy(update={"law": "rigid"})

    def _closest(self, state: Sequence[float]) -> tuple[float, float, float, float]:
        x, y, z = state[0], state[1], state[2]
        cx = min(max(x, -self.half_thickness), self.half_thickness)
        cy = min(max(y, -self.half_span), self.half_span)
        cz = min(max(z, 0.0), self.height)
        return cx, cy, cz, math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Distance from the ball's surface to the net box, m."""

        return self._closest(state)[3] - self.radius

    def clearance(self, state: Sequence[float], gap: float) -> float:
        """How far the centre can move without the gap changing sign, m (a distance: 1-Lipschitz)."""

        return abs(gap)

    def contact(self, t: float, state: Sequence[float]) -> Contact | None:
        """The contact at the closest point of the net box; ``None`` if the centre is inside the box."""

        cx, cy, cz, distance = self._closest(state)
        if distance < 1e-12:
            return None
        point = np.array([cx, cy, cz])
        normal = (np.asarray(state[:3], dtype=float) - point) / distance
        return Contact(self.kind, self.name, point, normal, _ZERO3.copy(), self.material_at(point), math.inf, region=self.region(point))


@dataclass(frozen=True)
class Floor:
    """
    The floor: the unbounded plane ``z = -table_height``.

    :param radius: Ball radius, m.
    :param table_height: Height of the playing surface above the floor, m.
    :param material: Ball-floor impact law.
    """

    radius: float
    table_height: float
    material: ContactMaterial
    kind: SurfaceKind = "floor"
    name: str = "floor"

    @classmethod
    def from_params(cls, params: BallTableParams) -> Floor:
        """The floor of ``params``."""

        return cls(params.ball.radius, params.table.height, params.floor_contact)

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Height of the ball's lowest point above the floor, m."""

        return state[2] + self.table_height - self.radius

    def clearance(self, state: Sequence[float], gap: float) -> float:
        """How far the centre can move without the gap changing sign, m (1-Lipschitz)."""

        return abs(gap)

    def contact(self, t: float, state: Sequence[float]) -> Contact | None:
        """Always a contact."""

        point = np.array([state[0], state[1], -self.table_height])
        return Contact(self.kind, self.name, point, _UP.copy(), _ZERO3.copy(), self.material, math.inf)


TableTouch = Literal["top", "edge", "side", "underside"]
"""Where a ball touched the table board.

- ``top``: the playing surface, normal straight up.
- ``edge``: the top edge, with the normal tilted up from the horizontal.
- ``side``: an end or side face, with a horizontal normal.
- ``underside``: the underside or a bottom edge, with the normal pointing down.
"""


def classify_table_touch(normal: ArrayLike, *, tolerance: float = 1e-9) -> TableTouch:
    """
    Classify a table contact by its normal, from the board into the ball.

    The board's faces are axis-aligned, so the top face gives exactly ``+z``,
    an end or side face a horizontal normal, and only the edges and corners
    give anything in between.

    :param normal: Contact normal (unit).
    :param tolerance: Slack on the vertical component.
    :return: ``top``, ``edge``, ``side`` or ``underside``.
    """

    nz = float(np.asarray(normal, dtype=float).reshape(3)[2])
    if nz >= 1.0 - tolerance:
        return "top"
    if nz > tolerance:
        return "edge"
    if nz >= -tolerance:
        return "side"
    return "underside"


def is_edge_ball(touch: TableTouch) -> bool:
    """
    Whether a table touch counts as the ball touching the playing surface.

    Under the Laws, the top edge counts and a side or end face does not. ``top``
    and ``edge`` count; ``side`` and ``underside`` do not.
    """

    return touch in ("top", "edge")


def _box_closest(x: float, y: float, z: float, low: tuple[float, float, float], high: tuple[float, float, float]) -> tuple[float, float, float, float]:
    cx = min(max(x, low[0]), high[0])
    cy = min(max(y, low[1]), high[1])
    cz = min(max(z, low[2]), high[2])
    return cx, cy, cz, math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)


@dataclass(frozen=True)
class TableBoard:
    """
    The top board below and around the playing surface: ends, sides, edges, corners and the underside.

    The board is the box ``|x| <= half_length``, ``|y| <= half_width``,
    ``-thickness <= z <= 0``, the game adapter's ``table_board``. Its gap is
    the distance from the ball centre to the box minus the radius, so an edge
    or corner deflects the ball by its geometry. A centre above the playing
    surface's footprint belongs to :class:`TableTop`, whose plane gap there is
    the same distance, so the top face never gives two contacts and the top
    plane's behaviour is unchanged. Contacts carry ``region`` from
    :func:`classify_table_touch`.

    :param radius: Ball radius, m.
    :param half_length: Half the table length, m.
    :param half_width: Half the table width, m.
    :param thickness: Board depth below the playing surface, m.
    :param material: The table's impact law (the board faces are the same board; judgement).
    """

    radius: float
    half_length: float
    half_width: float
    thickness: float
    material: ContactMaterial
    kind: SurfaceKind = "table"
    name: str = "table"

    @classmethod
    def from_params(cls, params: BallTableParams) -> TableBoard:
        """The board of ``params``."""

        table = params.table
        return cls(params.ball.radius, 0.5 * table.length, 0.5 * table.width, table.board_thickness, params.table_contact)

    @property
    def box(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """``((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi))``, m."""

        return ((-self.half_length, -self.half_width, -self.thickness), (self.half_length, self.half_width, 0.0))

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Distance from the ball's surface to the board, m; infinite over the playing surface (the top's domain)."""

        x, y, z = state[0], state[1], state[2]
        if z >= 0.0 and abs(x) <= self.half_length and abs(y) <= self.half_width:
            return math.inf
        low, high = self.box
        return _box_closest(x, y, z, low, high)[3] - self.radius

    def clearance(self, state: Sequence[float], gap: float) -> float:
        """
        How far the centre can move without the gap changing sign, m.

        Over the playing surface (an infinite gap) the board is ``z`` below the centre, so the gap stays
        non-negative while the centre moves less than ``z - R``; elsewhere the gap is a distance. A negative gap
        also turns (infinite) where the centre reaches the playing surface's domain (``z >= 0`` over the table),
        so its clearance is also bounded by the distance to that domain.
        """

        if math.isinf(gap):
            return max(state[2] - self.radius, 0.0)
        if gap < 0.0:
            dx = max(abs(state[0]) - self.half_length, 0.0)
            dy = max(abs(state[1]) - self.half_width, 0.0)
            dz = max(-state[2], 0.0)
            return min(-gap, math.sqrt(dx * dx + dy * dy + dz * dz))
        return gap

    def contact(self, t: float, state: Sequence[float]) -> Contact | None:
        """The contact at the board's closest point; ``None`` if the centre is inside the board (a tunnel)."""

        low, high = self.box
        cx, cy, cz, distance = _box_closest(state[0], state[1], state[2], low, high)
        if distance < 1e-12:
            return None
        point = np.array([cx, cy, cz])
        normal = (np.asarray(state[:3], dtype=float) - point) / distance
        return Contact(self.kind, self.name, point, normal, _ZERO3.copy(), self.material, math.inf, region=classify_table_touch(normal))


NET_POST_RADIUS_M = 0.0075
"""Net post radius, m (net asset, as ``scene/colliders.py``)."""

NET_POST_OUTSET_M = 0.0045
"""Post axis beyond each end of the net box along Y, m."""

NET_POST_Z_M = (-0.03, 0.1585)
"""Post span in the table frame, m: from the clamp to the cap above the tape."""

NET_CLAMP_HALF_X_M = 0.016
NET_CLAMP_Y_M = (0.7275, 0.927)
NET_CLAMP_Z_M = (-0.089, 0.016)
"""Each net clamp's box in the table frame, m (``|y|`` range; mirrored either side). It grips the table's side
edge and stands 16 mm above the playing surface over the last 35 mm inside the side line."""


@dataclass(frozen=True)
class NetFixture:
    """
    A net post (a vertical cylinder) or clamp (a box), as a rigid surface.

    Both belong to the net assembly under the Laws, so contacts are ``kind
    = "net"`` with ``region`` ``post`` or ``clamp``. Posts take ``post_contact``
    (steel: the ITTF steel-block restitution, drag-corrected); clamps take ``fixture_contact``.

    :param radius: Ball radius, m.
    :param shape: ``cylinder`` (vertical axis) or ``box``.
    :param centre: Cylinder axis point ``(x, y, _)`` or box centre, m.
    :param size: Cylinder ``(radius, _, _)`` or box half extents, m.
    :param z_range: Cylinder ``(bottom, top)``, m (ignored for a box).
    :param material: Ball-fixture law.
    :param name: Event name, e.g. ``net_post_pos_y``.
    :param region: ``post`` or ``clamp``.
    """

    radius: float
    shape: Literal["cylinder", "box"]
    centre: tuple[float, float, float]
    size: tuple[float, float, float]
    z_range: tuple[float, float]
    material: ContactMaterial
    name: str
    region: str
    kind: SurfaceKind = "net"

    def _closest(self, x: float, y: float, z: float) -> tuple[float, float, float, float]:
        if self.shape == "box":
            cx, cy, cz = self.centre
            hx, hy, hz = self.size
            return _box_closest(x, y, z, (cx - hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz + hz))
        ax, ay = self.centre[0], self.centre[1]
        rho = self.size[0]
        dx, dy = x - ax, y - ay
        radial = math.hypot(dx, dy)
        px, py = (ax + dx * rho / radial, ay + dy * rho / radial) if radial > rho else (x, y)
        pz = min(max(z, self.z_range[0]), self.z_range[1])
        return px, py, pz, math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Distance from the ball's surface to the fixture, m."""

        x, y, z = state[0], state[1], state[2]
        # Cheap and sign-preserving far away: the fixtures all sit within 5 cm of x = 0.
        if abs(x) > 0.25:
            return abs(x) - 0.1
        return self._closest(x, y, z)[3] - self.radius

    def clearance(self, state: Sequence[float], gap: float) -> float:
        """
        How far the centre can move without the gap changing sign, m: ``|gap|``. Far away the gap is a lower
        bound of the distance (the fixtures lie within 4 cm of ``x = 0``), near it is the distance.
        """

        return abs(gap)

    def contact(self, t: float, state: Sequence[float]) -> Contact | None:
        """The contact at the fixture's closest point; ``None`` if the centre is inside it."""

        cx, cy, cz, distance = self._closest(state[0], state[1], state[2])
        if distance < 1e-12:
            return None
        point = np.array([cx, cy, cz])
        normal = (np.asarray(state[:3], dtype=float) - point) / distance
        return Contact(self.kind, self.name, point, normal, _ZERO3.copy(), self.material, math.inf, region=self.region)


def net_fixtures(params: BallTableParams) -> tuple[NetFixture, ...]:
    """The two steel net posts (``post_contact``) and the two clamps (``fixture_contact``)."""

    radius = params.ball.radius
    post_y = params.net_half_span + NET_POST_OUTSET_M
    post_material = params.post_contact
    clamp_y = 0.5 * (NET_CLAMP_Y_M[0] + NET_CLAMP_Y_M[1])
    clamp_half = (NET_CLAMP_HALF_X_M, 0.5 * (NET_CLAMP_Y_M[1] - NET_CLAMP_Y_M[0]), 0.5 * (NET_CLAMP_Z_M[1] - NET_CLAMP_Z_M[0]))
    clamp_z = 0.5 * (NET_CLAMP_Z_M[0] + NET_CLAMP_Z_M[1])
    fixtures: list[NetFixture] = []
    for side, sign in (("neg_y", -1.0), ("pos_y", 1.0)):
        fixtures.append(
            NetFixture(radius, "cylinder", (0.0, sign * post_y, 0.0), (NET_POST_RADIUS_M, 0.0, 0.0), NET_POST_Z_M, post_material, f"net_post_{side}", "post")
        )
    for side, sign in (("neg_y", -1.0), ("pos_y", 1.0)):
        fixtures.append(NetFixture(radius, "box", (0.0, sign * clamp_y, clamp_z), clamp_half, (0.0, 0.0), params.fixture_contact, f"net_clamp_{side}", "clamp"))
    return tuple(fixtures)


@dataclass(frozen=True)
class PaddlePose:
    """
    The paddle blade's pose and twist at one instant, in the table frame.

    :param position: Blade centre (mid-plane), m.
    :param rotation: 3x3 matrix whose columns are the blade's local X (towards
        the handle), Y and Z (the forehand face normal) axes.
    :param velocity: Linear velocity of the blade centre, m/s.
    :param angular_velocity: Angular velocity, rad/s.
    """

    position: NDArray[np.float64]
    rotation: NDArray[np.float64]
    velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    angular_velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    def point_velocity(self, point: ArrayLike) -> NDArray[np.float64]:
        """Velocity of the blade's material point at ``point`` (m), m/s."""

        return self.velocity + np.cross(self.angular_velocity, np.asarray(point, dtype=float) - self.position)


PaddleTrajectory = Callable[[float], PaddlePose]
"""A scripted paddle: time (s) to pose and twist."""


@dataclass(frozen=True)
class ScriptedPaddle:
    """
    A paddle whose motion is prescribed, not simulated.

    The impact still sees the paddle's finite effective mass; the script then
    carries on unchanged, as a stiffly driven paddle would.

    :param trajectory: Pose and twist as a function of time. A
        :class:`lib.physics.reference.paddle_track.PaddleTrack` (the game's
        per-tick segments) is also watched cheaply: see :class:`PaddleFaces`.
    :param name: Name recorded on its events.
    :param rigid_body: Meet the ball as a rigid body (effective mass plus the
        paddle inertia, centre of mass at the blade centre), so an off-centre
        hit meets less mass. False keeps the point-mass model.
    :param windows: Opt-in active windows ``((start, end), ...)``, s: outside
        them the paddle is not watched at all, so a touch there is not detected
        (:func:`lib.physics.reference.paddle_track.contact_windows` builds them
        around planned contacts). Empty (the default): always watched.
    """

    trajectory: PaddleTrajectory
    name: str = "paddle"
    rigid_body: bool = True
    windows: tuple[tuple[float, float], ...] = ()

    def active(self, t: float) -> bool:
        """Whether the paddle is watched at ``t``."""

        windows = self.windows
        if not windows:
            return True
        return any(start <= t <= end for start, end in windows)

    def active_between(self, t0: float, t1: float) -> bool:
        """Whether any window meets ``[t0, t1]``."""

        windows = self.windows
        if not windows:
            return True
        return any(start <= t1 and t0 <= end for start, end in windows)


PADDLE_NEAR_MARGIN_M: float = 1e-3
""":class:`PaddleFaces`: margin added to the blade's reach and the ball radius when deciding the ball is far from it, m."""


@dataclass(frozen=True)
class PaddleFaces:
    """
    Both faces of a scripted paddle as one watched surface.

    The ball touches a face when it is within ``R + thickness/2`` of the face
    plane (the *band*) **and** its touching point (the centre less ``R`` along
    the face normal) lies inside the elliptical blade outline. The gap is
    therefore ``max(|n . (c - p)| - (R + thickness/2), (e - 1) min(a, b))``,
    with ``e`` the touching point's elliptical radius in the outline's
    semi-axes ``a`` (``half_length``) and ``b``: negative exactly when both
    hold. It turns negative either through the band (a face hit) or through the
    outline while the ball is in the band (a hit at the rim, by a ball that
    reached the face plane beside the blade). The game adapter's sweep
    (``engine/contacts.py`` ``sweep_blade``) takes both: a step that starts in
    the band with the touching point inside the outline and approaching is a
    touch at the step's start. Until 2026-10-07 the gap was the band alone: a
    rim hit was never armed and the ball passed through the blade, and every
    crossing of the face's infinite plane beside the blade cost a bisection
    that found no contact (``CHANGELOG`` in ``reports/physics-ref.md``).

    **Far from the blade.** When the trajectory can bound where the blade is
    (a :class:`lib.physics.reference.paddle_track.PaddleTrack`: a sphere around
    the blade centre's path in the segment in force), a ball farther than that
    sphere plus the blade's reach plus ``R`` gets a positive gap (its distance
    beyond that) without evaluating the pose. The gap above is positive there
    too, and the integrator reads only signs, so this changes nothing; the
    pose is evaluated only near the blade, and then only its centre and axes.

    **Windows** (opt-in, :attr:`ScriptedPaddle.windows`): outside them the gap
    is infinite and no root is a contact.

    **The rounded edge** (with :attr:`edge_material`, which :meth:`from_params` sets to the contract's
    ``rim_contact``): the gap is :func:`lib.physics.reference.blade.blade_solid_gap`, so a ball reaching the blade's
    edge touches it (an ``edge`` touch, region ``blade_edge``, resolved with :attr:`edge_material` and the rigid
    paddle's mass at the edge point) instead of passing into the blade. Where the face governs, the gap is the
    face-only gap's own value, so face touches are unchanged bit for bit. Without it (a direct construction), the
    blade is the face-only one above. The reach and the point-speed bound grow to the rim (``max(a, b) + h``).

    :param paddle: The scripted paddle.
    :param radius: Ball radius, m.
    :param half_thickness: Half the blade thickness including rubbers, m.
    :param half_length: Outline semi-axis along local X, m.
    :param half_width: Outline semi-axis along local Y, m.
    :param material: Ball-rubber impact law, both faces.
    :param effective_mass: Paddle effective mass, kg.
    :param inertia: Principal moments about the blade's local X, Y and Z through its centre, kg m^2.
    :param edge_material: Ball-edge impact law (the rounded edge), or ``None`` for the face-only blade.
    """

    paddle: ScriptedPaddle
    radius: float
    half_thickness: float
    half_length: float
    half_width: float
    material: ContactMaterial
    effective_mass: float
    inertia: tuple[float, float, float] = (1.0, 1.0, 1.0)
    kind: SurfaceKind = "paddle"
    edge_material: ContactMaterial | None = None

    @property
    def name(self) -> str:
        """The paddle's name."""

        return self.paddle.name

    @classmethod
    def from_params(cls, paddle: ScriptedPaddle, params: BallTableParams, *, edge: bool = True) -> PaddleFaces:
        """The faces of ``paddle`` with the geometry, rubber and effective mass of ``params``, and its rounded edge
        (``rim_contact``) unless ``edge`` is false."""

        blade = params.paddle
        return cls(
            paddle=paddle,
            radius=params.ball.radius,
            half_thickness=0.5 * blade.thickness,
            half_length=blade.blade_half_length,
            half_width=blade.blade_half_width,
            material=params.rubber_contact,
            effective_mass=blade.effective_mass,
            inertia=blade.inertia,
            edge_material=params.rim_contact if edge else None,
        )

    @cached_property
    def _track(self) -> Any:
        """The trajectory when it can bound the blade (``sphere``, ``centre_axes``, ``clearance``), else ``None``."""

        trajectory = self.paddle.trajectory
        if all(callable(getattr(trajectory, name, None)) for name in ("sphere", "centre_axes", "state", "clearance", "point_speed_bound")):
            return trajectory
        return None

    @cached_property
    def reach(self) -> float:
        """Farthest a touching ball's centre can be from the blade centre, plus a margin, m."""

        if self.edge_material is not None:
            return max(self.half_length, self.half_width) + self.half_thickness + self.radius + PADDLE_NEAR_MARGIN_M
        blade = math.hypot(max(self.half_length, self.half_width), self.half_thickness)
        return blade + self.radius + PADDLE_NEAR_MARGIN_M

    @cached_property
    def speed_bound(self) -> float:
        """A bound on the speed of the blade's points, m/s (for the integrator's graze search); 40 m/s when unknown."""

        track = self._track
        if track is None:
            return 40.0
        extent = max(self.half_length, self.half_width)
        extent = extent + self.half_thickness if self.edge_material is not None else math.hypot(extent, self.half_thickness)
        return float(track.point_speed_bound(extent))

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Distance from the ball's surface to the nearer face plane, m (positive far from a bounded blade)."""

        if self.paddle.windows and not self.paddle.active(t):
            return math.inf
        track = self._track
        if track is not None:
            mx, my, mz, spread = track.sphere(t)
            dx, dy, dz = state[0] - mx, state[1] - my, state[2] - mz
            far = math.sqrt(dx * dx + dy * dy + dz * dz) - spread - self.reach
            if far > 0.0:
                return far
            px, py, pz, x_axis, y_axis, z_axis = track.centre_axes(t)
        else:
            pose = self.paddle.trajectory(t)
            rot = pose.rotation
            px, py, pz = (float(c) for c in pose.position)
            # The game's blade axes of a reference rotation [-Y, X, Z]: X = column 1, Y = -column 0.
            x_axis = (float(rot[0, 1]), float(rot[1, 1]), float(rot[2, 1]))
            y_axis = (-float(rot[0, 0]), -float(rot[1, 0]), -float(rot[2, 0]))
            z_axis = (float(rot[0, 2]), float(rot[1, 2]), float(rot[2, 2]))
        gap = blade_gap if self.edge_material is None else blade_solid_gap
        return gap(
            state[0] - px,
            state[1] - py,
            state[2] - pz,
            x_axis,
            y_axis,
            z_axis,
            radius=self.radius,
            half_thickness=self.half_thickness,
            half_length=self.half_length,
            half_width=self.half_width,
        )

    def clearance_between(self, t0: float, t1: float, state: Sequence[float]) -> float:
        """How far the ball's centre can move during ``[t0, t1]`` without reaching the blade, m (0 when unknown)."""

        if not self.paddle.active_between(t0, t1):
            return math.inf
        track = self._track
        if track is None:
            return 0.0
        return max(float(track.clearance(t0, t1, state[0], state[1], state[2])) - self.reach, 0.0)

    def contact(self, t: float, state: Sequence[float]) -> Contact | None:
        """The face contact, or ``None`` when the touching point is off the blade outline (or outside the windows)."""

        if self.paddle.windows and not self.paddle.active(t):
            return None
        track = self._track
        if track is not None:
            position, orientation, velocity, angular_velocity = track.state(t)
            axes = blade_axes(orientation)
        else:
            pose = self.paddle.trajectory(t)
            rot = np.asarray(pose.rotation, dtype=float)
            axes = np.column_stack([rot[:, 1], -rot[:, 0], rot[:, 2]])
            position, velocity, angular_velocity = pose.position, pose.velocity, pose.angular_velocity
        # The contact state the game adapter resolves its touch with too (lib.physics.reference.blade).
        frame = blade_contact if self.edge_material is None else blade_solid_contact
        touch = frame(
            state[:3],
            axes,
            position,
            velocity,
            angular_velocity,
            radius=self.radius,
            half_thickness=self.half_thickness,
            half_length=self.half_length,
            half_width=self.half_width,
        )
        # A rim root is bracketed from outside the outline, a bisection width (about 1e-10 s) away from it.
        if touch is None or not touch.on_blade:
            return None
        edge = touch.touch == "edge"
        material = self.edge_material if edge and self.edge_material is not None else self.material
        region = "blade_edge" if edge else ""
        if not self.paddle.rigid_body or math.isinf(self.effective_mass):
            return Contact(self.kind, self.name, touch.point, touch.normal, touch.surface_velocity, material, self.effective_mass, region=region)
        inertia = touch.inertia(*self.inertia)
        return Contact(self.kind, self.name, touch.point, touch.normal, touch.surface_velocity, material, self.effective_mass, touch.offset, inertia, region)


def blade_rotation(face_normal: ArrayLike, handle_direction: ArrayLike = (0.0, 0.0, -1.0)) -> NDArray[np.float64]:
    """
    A blade rotation matrix with the given forehand normal.

    :param face_normal: Desired local +Z (forehand face normal), any length.
    :param handle_direction: Rough direction of local +X (towards the handle);
        it is orthogonalised against the normal.
    :return: 3x3 rotation matrix with columns (X, Y, Z).
    """

    z_axis = np.asarray(face_normal, dtype=float)
    z_axis = z_axis / np.linalg.norm(z_axis)
    hint = np.asarray(handle_direction, dtype=float)
    x_axis = hint - (hint @ z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-9:
        fallback = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = fallback - (fallback @ z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.asarray(np.column_stack([x_axis, y_axis, z_axis]), dtype=np.float64)


def constant_twist_paddle(
    position: ArrayLike,
    rotation: ArrayLike,
    velocity: ArrayLike = (0.0, 0.0, 0.0),
    angular_velocity: ArrayLike = (0.0, 0.0, 0.0),
    *,
    reference_time: float = 0.0,
    name: str = "paddle",
    rigid_body: bool = True,
) -> ScriptedPaddle:
    """
    A paddle moving with constant linear and angular velocity, met as a rigid body.

    At time ``t`` the blade centre is ``position + velocity (t - t_ref)`` and
    the orientation is ``exp([angular_velocity] (t - t_ref)) rotation``.

    :param position: Blade centre at the reference time, m.
    :param rotation: Blade rotation at the reference time (columns X, Y, Z).
    :param velocity: Linear velocity, m/s.
    :param angular_velocity: Angular velocity, rad/s (table frame).
    :param reference_time: Time at which ``position`` and ``rotation`` hold, s.
    :param name: Paddle name.
    :param rigid_body: See :class:`ScriptedPaddle`.
    """

    from scipy.spatial.transform import Rotation

    p0 = np.asarray(position, dtype=float).reshape(3)
    r0 = np.asarray(rotation, dtype=float).reshape(3, 3)
    v = np.asarray(velocity, dtype=float).reshape(3)
    w = np.asarray(angular_velocity, dtype=float).reshape(3)
    spinning = bool(np.any(w != 0.0))

    def trajectory(t: float) -> PaddlePose:
        dt = t - reference_time
        rot = Rotation.from_rotvec(w * dt).as_matrix() @ r0 if spinning else r0
        return PaddlePose(p0 + v * dt, rot, v, w)

    return ScriptedPaddle(trajectory, name, rigid_body)
