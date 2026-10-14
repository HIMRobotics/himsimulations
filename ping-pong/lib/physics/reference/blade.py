"""
The blade at a contact: one geometry and contact state for the reference oracle and the replay's adapter.

A paddle touch is resolved by :func:`lib.physics.reference.impact.resolve_impact`
with the blade's state at the touch: the struck face's normal, the touching point,
its offset from the blade centre (the rigid paddle's lever arm), the velocity of the
blade's material point there, and the blade's axes (its inertia in the table frame).
The oracle and the adapter must hand it the *same* state, or the same touch leaves
at different velocities: interpolating the blade linearly inside a step instead of
evaluating its segment at the root moves the material point's velocity by
millimetres per second (inside a step the Hermite velocity is quadratic).

This module is that state, from the segment itself:

- :func:`blade_gap` is the touch condition both use: ``max(|n . (c - p)| -
  (R + h), (e - 1) min(a, b))``, negative exactly when the ball is within reach of
  the face plane (the band) **and** its touching point (the centre less ``R``
  along the normal) is inside the elliptical outline, ``e`` being that point's
  elliptical radius. It turns negative through the band (a ``face`` touch) or
  through the outline while in the band (a ``rim`` touch);
- :func:`blade_touch_time` finds when, inside a step, that gap first turns
  negative, with the blade on its segment (``PaddleSegment.at``) and the ball on a
  constant-acceleration path from the step's start;
- :func:`blade_contact` (and :func:`segment_contact`, which evaluates the segment)
  gives the contact frame at that time.

**The rounded edge** (the oracle's and the GPU batch's blade; the replay's adapter keeps the face-only gap above).
A real blade is a solid: a ball whose centre passes up to one radius beyond the outline strikes its edge, which the
face-only gap lets through (about 3 per 1,000 shots). The edge is modelled as a torus of tube radius ``h`` (half the blade thickness) around the outline
in the mid-plane, so the blade is the outline's disc with a rounded rim:

- :func:`ellipse_closest` is the outline's point nearest a point of its plane, by a fixed number of iterations (the
  batched engine runs the same ones);
- :func:`blade_edge_gap` is the ball's distance to that rim tube: ``sqrt(D^2 + d^2) - (R + h)``, ``D`` the in-plane
  distance from the centre's projection to the outline;
- :func:`blade_solid_gap` is the touch condition with the edge: :func:`blade_gap` itself wherever it is negative
  or the projection is strictly inside the outline (there the edge can never be nearer), otherwise the smaller of
  the two. Every value the face governs is therefore the face-only gap's own float, so a face touch, its root and
  the integrator's graze search are bit-for-bit those of the face-only blade; only near the edge does it differ;
- :func:`blade_solid_contact` gives the frame there: the face's (``face`` or ``rim``) where the face gap governs, else
  an ``edge`` touch whose normal points from the outline's nearest point to the ball centre.

Axes are the game's (``PaddleSegment`` / ``quat_matrix``): columns X across the
blade (``half_width``), Y towards the tip (``half_length``), Z the forehand face.
The reference model's ``PaddlePose`` rotation is ``[-Y, X, Z]`` of them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

EDGE_ITERATIONS: int = 3
"""
Iterations of :func:`ellipse_closest`, the same in the batched engine. Each one roughly cubes the error.

Measured on the contract's outline (semi-axes 0.075 and 0.0785 m) at 200,000 points within the edge's reach (in-plane
distance from the outline up to R + h = 0.02685 m), as the largest error in the distance to the outline against 12
iterations:

- 1 iteration: 3.3e-5 m;
- 2 iterations: 6.5e-15 m outside the outline (3.7e-13 m inside);
- 3 iterations: 2.8e-17 m, rounding;
- 4 iterations: 2.8e-17 m.

Three reach rounding. A fourth only lengthens the engine's float64 root-loop kernels (about +50 % of the ball's
cost at 8,192 environments).
"""

OUTLINE_TOLERANCE: float = 1e-6
"""Slack on the touching point's squared elliptical radius for a touch to count as on the blade (a rim touch is
found from just outside the outline, within a root bracket of about 1e-10 s)."""

BladeTouch = Literal["face", "rim", "edge"]


"""``face``: the ball crossed the face plane over the blade. ``rim``: the ball was already within reach of the face
plane beside the blade and the outline met it (the face-only blade's resolution of a side approach). ``edge``: the
ball struck the blade's rounded edge (:func:`blade_solid_gap`), with the normal from the outline into the ball."""


def blade_axes(orientation: ArrayLike) -> Array:
    """The blade's axes (columns X across, Y towards the tip, Z the forehand face) of a ``(w, x, y, z)`` orientation,
    normalised as the game's ``quat_matrix`` does."""

    w, x, y, z = (float(c) for c in np.asarray(orientation, dtype=np.float64).reshape(4))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class BladeContact:
    """
    The blade's contact state with a ball, at one instant.

    :ivar touch: ``face`` or ``rim`` (see :data:`BladeTouch`).
    :ivar normal: Unit normal of the struck face, from the blade into the ball.
    :ivar point: The touching point, the ball centre less the radius along ``normal``, m.
    :ivar offset: ``point`` from the blade centre, table frame, m: the lever arm of the rigid paddle.
    :ivar blade_offset: ``point`` in blade axes ``(across, along the length)``, m (the adapter's ``blade_offset_m``).
    :ivar outline: The touching point's elliptical radius in the outline's semi-axes (1 on the rim).
    :ivar signed_distance: The ball centre's distance from the blade's mid-plane along the forehand normal, m.
    :ivar surface_velocity: Velocity of the blade's material point at ``point``, m/s.
    :ivar axes: The blade's axes (columns X across, Y towards the tip, Z the forehand face).
    :ivar position: Blade centre, m.
    :ivar velocity: Blade-centre velocity, m/s.
    :ivar angular_velocity: Blade angular velocity, rad/s.
    """

    touch: BladeTouch
    normal: Array
    point: Array
    offset: Array
    blade_offset: tuple[float, float]
    outline: float
    signed_distance: float
    surface_velocity: Array
    axes: Array
    position: Array
    velocity: Array
    angular_velocity: Array

    @property
    def on_blade(self) -> bool:
        """Whether the touch is on the blade: an ``edge`` touch, or a touching point inside the outline (within
        :data:`OUTLINE_TOLERANCE`)."""

        return self.touch == "edge" or self.outline * self.outline <= 1.0 + OUTLINE_TOLERANCE

    def inertia(self, inertia_xx: float, inertia_yy: float, inertia_zz: float) -> Array:
        """
        The paddle's inertia tensor in the table frame, kg m^2.

        :param inertia_xx: About the blade's length (the contract's ``paddle_inertia_xx``).
        :param inertia_yy: About its width (``paddle_inertia_yy``).
        :param inertia_zz: About the face normal (``paddle_inertia_zz``).
        """

        axes = self.axes
        return np.asarray(axes @ np.diag([inertia_yy, inertia_xx, inertia_zz]) @ axes.T, dtype=np.float64)


def blade_gap(
    rx: float,
    ry: float,
    rz: float,
    x_axis: Sequence[float],
    y_axis: Sequence[float],
    z_axis: Sequence[float],
    *,
    radius: float,
    half_thickness: float,
    half_length: float,
    half_width: float,
) -> float:
    """
    The touch condition of a ball against a blade, m: negative exactly when the ball touches a face.

    ``max(|d| - (R + h), (e - 1) min(a, b))`` with ``d`` the centre's distance from the mid-plane and ``e`` the touching
    point's elliptical radius (the touching point differs from the centre only along the normal, so its in-plane
    coordinates are the centre's).

    :param rx: Ball centre less blade centre, x, m.
    :param ry: The same, y.
    :param rz: The same, z.
    :param x_axis: Blade X (across), unit.
    :param y_axis: Blade Y (towards the tip), unit.
    :param z_axis: Blade Z (forehand normal), unit.
    """

    band = abs(z_axis[0] * rx + z_axis[1] * ry + z_axis[2] * rz) - (radius + half_thickness)
    across = (x_axis[0] * rx + x_axis[1] * ry + x_axis[2] * rz) / half_width
    along = (y_axis[0] * rx + y_axis[1] * ry + y_axis[2] * rz) / half_length
    outline = (math.sqrt(across * across + along * along) - 1.0) * min(half_length, half_width)
    return band if band > outline else outline


def blade_contact(
    centre: ArrayLike,
    axes: ArrayLike,
    position: ArrayLike,
    velocity: ArrayLike,
    angular_velocity: ArrayLike,
    *,
    radius: float,
    half_thickness: float,
    half_length: float,
    half_width: float,
    side: float | None = None,
) -> BladeContact | None:
    """
    The contact frame of a ball centred at ``centre`` against a blade in the given state.

    :param centre: Ball centre, m.
    :param axes: Blade axes (columns X across, Y towards the tip, Z the forehand face): :func:`blade_axes`.
    :param position: Blade centre, m.
    :param velocity: Blade-centre velocity, m/s.
    :param angular_velocity: Blade angular velocity, rad/s.
    :param radius: Ball radius, m.
    :param half_thickness: Half the blade thickness with rubbers, m.
    :param half_length: Outline semi-axis along Y, m.
    :param half_width: Outline semi-axis along X, m.
    :param side: The struck face, ``+1`` (forehand) or ``-1``; ``None`` takes the side the centre is on.
    :return: The frame (whether or not the point is on the blade: see :attr:`BladeContact.on_blade`), or ``None`` when
        the centre lies on the mid-plane and no side is given.
    """

    c = np.asarray(centre, dtype=np.float64).reshape(3)
    frame = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    p = np.asarray(position, dtype=np.float64).reshape(3)
    face = frame[:, 2]
    rx, ry, rz = float(c[0] - p[0]), float(c[1] - p[1]), float(c[2] - p[2])
    d = float(face[0]) * rx + float(face[1]) * ry + float(face[2]) * rz
    if side is None:
        if d == 0.0:
            return None
        side = math.copysign(1.0, d)
    normal = side * face
    point = c - radius * normal
    offset = point - p
    across = float(offset @ frame[:, 0])
    along = float(offset @ frame[:, 1])
    outline = math.hypot(across / half_width, along / half_length)
    band = abs(d) - (radius + half_thickness)
    rim = (outline - 1.0) * min(half_length, half_width)
    v = np.asarray(velocity, dtype=np.float64).reshape(3)
    w = np.asarray(angular_velocity, dtype=np.float64).reshape(3)
    return BladeContact(
        touch="rim" if rim > band else "face",
        normal=normal,
        point=point,
        offset=offset,
        blade_offset=(across, along),
        outline=outline,
        signed_distance=d,
        surface_velocity=v + np.cross(w, offset),
        axes=frame.copy(),
        position=p.copy(),
        velocity=v.copy(),
        angular_velocity=w.copy(),
    )


def ellipse_closest(x: float, y: float, semi_x: float, semi_y: float) -> tuple[float, float]:
    """
    The point of the ellipse ``(X / semi_x)^2 + (Y / semi_y)^2 = 1`` nearest ``(x, y)`` (a point outside it), by
    :data:`EDGE_ITERATIONS` fixed-point steps of the evolute construction (each step moves along the ellipse by the
    arc its curvature circle subtends; the batched engine runs the same steps).
    """

    px, py = abs(x), abs(y)
    tx = ty = 0.7071067811865476
    a, b = semi_x, semi_y
    c2 = a * a - b * b
    for _ in range(EDGE_ITERATIONS):
        ex = c2 * tx * tx * tx / a
        ey = -c2 * ty * ty * ty / b
        rx, ry = a * tx - ex, b * ty - ey
        qx, qy = px - ex, py - ey
        r = math.hypot(rx, ry)
        q = max(math.hypot(qx, qy), 1e-300)
        tx = min(1.0, max(0.0, (qx * r / q + ex) / a))
        ty = min(1.0, max(0.0, (qy * r / q + ey) / b))
        norm = math.hypot(tx, ty)
        tx, ty = tx / norm, ty / norm
    return math.copysign(a * tx, x), math.copysign(b * ty, y)


def blade_edge_gap(
    across: float,
    along: float,
    distance: float,
    *,
    radius: float,
    half_thickness: float,
    half_length: float,
    half_width: float,
) -> float:
    """
    The ball's clearance from the blade's rounded edge, m: ``sqrt(D^2 + d^2) - (R + h)``.

    :param across: The centre's coordinate along the blade's X (across), m.
    :param along: Along its Y (towards the tip), m.
    :param distance: Its distance ``d`` from the mid-plane along the forehand normal, m.
    """

    qx, qy = ellipse_closest(across, along, half_width, half_length)
    dx, dy = across - qx, along - qy
    return math.sqrt(dx * dx + dy * dy + distance * distance) - (radius + half_thickness)


def blade_solid_gap(
    rx: float,
    ry: float,
    rz: float,
    x_axis: Sequence[float],
    y_axis: Sequence[float],
    z_axis: Sequence[float],
    *,
    radius: float,
    half_thickness: float,
    half_length: float,
    half_width: float,
) -> float:
    """
    The touch condition of a ball against a blade with its rounded edge, m: negative exactly when the ball touches a
    face or the edge.

    :func:`blade_gap`'s own value wherever that is negative or the centre's projection is strictly inside the outline
    (the edge can never be nearer there), else the smaller of it and :func:`blade_edge_gap`. On the outline itself
    the face gap is never negative, and the edge is what a ball there touches.
    """

    gap = blade_gap(rx, ry, rz, x_axis, y_axis, z_axis, radius=radius, half_thickness=half_thickness, half_length=half_length, half_width=half_width)
    if gap < 0.0:
        return gap
    across = x_axis[0] * rx + x_axis[1] * ry + x_axis[2] * rz
    along = y_axis[0] * rx + y_axis[1] * ry + y_axis[2] * rz
    # blade_gap's own outline term, strictly inside: the face governs (on the outline itself the face gap is never
    # negative, and the edge is what a ball there touches)
    a_n, b_n = across / half_width, along / half_length
    if (math.sqrt(a_n * a_n + b_n * b_n) - 1.0) * min(half_length, half_width) < 0.0:
        return gap
    distance = z_axis[0] * rx + z_axis[1] * ry + z_axis[2] * rz
    edge = blade_edge_gap(across, along, distance, radius=radius, half_thickness=half_thickness, half_length=half_length, half_width=half_width)
    return edge if edge < gap else gap


def blade_solid_contact(
    centre: ArrayLike,
    axes: ArrayLike,
    position: ArrayLike,
    velocity: ArrayLike,
    angular_velocity: ArrayLike,
    *,
    radius: float,
    half_thickness: float,
    half_length: float,
    half_width: float,
) -> BladeContact | None:
    """
    The contact frame of a ball against the blade with its rounded edge (at a root of :func:`blade_solid_gap`).

    Where the face gap governs (it is non-positive, or the centre's projection is inside the outline, or it is not
    larger than the edge gap) this is :func:`blade_contact`'s frame. Otherwise it is an ``edge`` touch: the normal from
    the outline's nearest point to the ball centre, the touching point the centre less ``R`` along it.
    """

    c = np.asarray(centre, dtype=np.float64).reshape(3)
    frame = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    p = np.asarray(position, dtype=np.float64).reshape(3)
    r = c - p
    x_axis, y_axis, z_axis = (tuple(float(v) for v in frame[:, k]) for k in range(3))
    sizes = {"radius": radius, "half_thickness": half_thickness, "half_length": half_length, "half_width": half_width}
    face_gap = blade_gap(float(r[0]), float(r[1]), float(r[2]), x_axis, y_axis, z_axis, **sizes)
    solid = blade_solid_gap(float(r[0]), float(r[1]), float(r[2]), x_axis, y_axis, z_axis, **sizes)
    if solid == face_gap:
        return blade_contact(c, frame, p, velocity, angular_velocity, **sizes)
    across = float(frame[:, 0] @ r)
    along = float(frame[:, 1] @ r)
    d = float(frame[:, 2] @ r)
    qx, qy = ellipse_closest(across, along, half_width, half_length)
    rim = p + qx * frame[:, 0] + qy * frame[:, 1]
    gap = c - rim
    normal = gap / np.linalg.norm(gap)
    point = c - radius * normal
    offset = point - p
    o_across = float(offset @ frame[:, 0])
    o_along = float(offset @ frame[:, 1])
    v = np.asarray(velocity, dtype=np.float64).reshape(3)
    w = np.asarray(angular_velocity, dtype=np.float64).reshape(3)
    return BladeContact(
        touch="edge",
        normal=normal,
        point=point,
        offset=offset,
        blade_offset=(o_across, o_along),
        outline=math.hypot(o_across / half_width, o_along / half_length),
        signed_distance=d,
        surface_velocity=v + np.cross(w, offset),
        axes=frame.copy(),
        position=p.copy(),
        velocity=v.copy(),
        angular_velocity=w.copy(),
    )


def _blade_state(blade: Any, t: float) -> tuple[Any, Any, Any, Any]:
    """``PaddleSegment.at(t)``, or any blade with a ``state(t)``."""

    at: Any = getattr(blade, "at", None)
    state: Any = at(t) if callable(at) else blade.state(t)
    position, orientation, velocity, angular = state
    return position, orientation, velocity, angular


def segment_contact(
    blade: Any,
    t: float,
    centre: ArrayLike,
    *,
    radius: float,
    half_thickness: float,
    half_length: float,
    half_width: float,
    side: float | None = None,
    edge: bool = False,
) -> BladeContact | None:
    """
    :func:`blade_contact` with the blade evaluated exactly at ``t`` on its segment.

    :param blade: A ``PaddleSegment`` (its ``at``) or any blade with a ``state(t)``.
    :param t: Contact time, s.
    :param centre: Ball centre at ``t``, m.
    :param edge: The blade with its rounded edge: :func:`blade_solid_contact` (``side`` must then be ``None``).
    """

    position, orientation, velocity, angular = _blade_state(blade, t)
    if edge:
        if side is not None:
            raise ValueError("the rounded edge's contact takes the side from the geometry; pass side=None")
        return blade_solid_contact(
            centre,
            blade_axes(orientation),
            position,
            velocity,
            angular,
            radius=radius,
            half_thickness=half_thickness,
            half_length=half_length,
            half_width=half_width,
        )
    return blade_contact(
        centre,
        blade_axes(orientation),
        position,
        velocity,
        angular,
        radius=radius,
        half_thickness=half_thickness,
        half_length=half_length,
        half_width=half_width,
        side=side,
    )


def blade_touch_time(
    blade: Any,
    t0: float,
    h: float,
    ball: ArrayLike,
    *,
    radius: float,
    half_thickness: float,
    half_length: float,
    half_width: float,
    acceleration: ArrayLike = (0.0, 0.0, 0.0),
    tolerance: float = 1e-12,
    edge: bool = False,
) -> float | None:
    """
    When, inside ``[t0, t0 + h]``, the ball first touches the blade (:func:`blade_gap` turns negative; with ``edge``,
    :func:`blade_solid_gap`, the blade with its rounded edge).

    The blade is on its segment, evaluated exactly; the ball moves from its state at ``t0`` with constant
    ``acceleration`` (gravity and the air force at the step's start: the path the adapter integrates over a step). The
    gap is bisected to ``tolerance`` seconds; the result is the first time on the touching side, the oracle's root
    to within ``tolerance``. A touch that enters and leaves between the two ends is not searched for.

    :param blade: A ``PaddleSegment`` or any blade with a ``state(t)``.
    :param t0: Step start, s.
    :param h: Step, s.
    :param ball: Ball state at ``t0`` (position, velocity; anything after them is ignored).
    :param radius: Ball radius, m.
    :param half_thickness: Half the blade thickness, m.
    :param half_length: Outline semi-axis along the blade's Y, m.
    :param half_width: Outline semi-axis along its X, m.
    :param acceleration: Ball acceleration over the step, m/s^2.
    :param tolerance: Bisection bracket, s.
    :param edge: The blade with its rounded edge (the oracle's ``simulate(..., blade_edge=True)``); False: face-only.
    :return: The touch time; ``t0`` when the ball already touches at the step's start (the adapter's started-inside
        touch; the oracle never arms a gap that starts negative); ``None`` when there is no touch at the step's end.
    """

    state = np.asarray(ball, dtype=np.float64).reshape(-1)
    c0, v = state[:3], state[3:6]
    a = np.asarray(acceleration, dtype=np.float64).reshape(3)
    solid = blade_solid_gap if edge else blade_gap

    def gap(tau: float) -> float:
        position, orientation, _velocity, _angular = _blade_state(blade, t0 + tau)
        axes = blade_axes(orientation)
        r = c0 + v * tau + 0.5 * a * tau * tau - np.asarray(position, dtype=np.float64)
        return solid(
            float(r[0]),
            float(r[1]),
            float(r[2]),
            tuple(axes[:, 0].tolist()),
            tuple(axes[:, 1].tolist()),
            tuple(axes[:, 2].tolist()),
            radius=radius,
            half_thickness=half_thickness,
            half_length=half_length,
            half_width=half_width,
        )

    if gap(0.0) < 0.0:
        return t0
    if gap(h) >= 0.0:
        return None
    lo, hi = 0.0, h
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        if gap(mid) >= 0.0:
            lo = mid
        else:
            hi = mid
    return t0 + lo


def blade_touch(
    blade: Any,
    t0: float,
    h: float,
    ball: ArrayLike,
    *,
    radius: float,
    half_thickness: float,
    half_length: float,
    half_width: float,
    acceleration: ArrayLike = (0.0, 0.0, 0.0),
    tolerance: float = 1e-12,
    edge: bool = False,
) -> tuple[float, Array, BladeContact] | None:
    """
    The first touch of the ball on the blade inside one step, with its contact state: what an adapter resolves.

    :func:`blade_touch_time`, then the ball's centre on its constant-acceleration path at that time and
    :func:`segment_contact` there. The caller checks that the ball approaches the face
    (``(v - contact.surface_velocity) . contact.normal < 0``) before resolving the impact. With ``edge`` the blade has
    its rounded edge (the oracle's default): the touch may be ``contact.touch == "edge"``, which the
    caller resolves with ``rim_contact``.

    :return: ``(time, ball centre, contact)``, or ``None`` when the ball does not touch the blade by the step's end.
    """

    t = blade_touch_time(
        blade,
        t0,
        h,
        ball,
        radius=radius,
        half_thickness=half_thickness,
        half_length=half_length,
        half_width=half_width,
        acceleration=acceleration,
        tolerance=tolerance,
        edge=edge,
    )
    if t is None:
        return None
    state = np.asarray(ball, dtype=np.float64).reshape(-1)
    a = np.asarray(acceleration, dtype=np.float64).reshape(3)
    tau = t - t0
    centre = state[:3] + state[3:6] * tau + 0.5 * a * tau * tau
    contact = segment_contact(blade, t, centre, radius=radius, half_thickness=half_thickness, half_length=half_length, half_width=half_width, edge=edge)
    if contact is None or not contact.on_blade:
        return None
    return t, centre, contact


__all__ = [
    "EDGE_ITERATIONS",
    "OUTLINE_TOLERANCE",
    "BladeContact",
    "BladeTouch",
    "blade_axes",
    "blade_contact",
    "blade_edge_gap",
    "blade_gap",
    "blade_solid_contact",
    "blade_solid_gap",
    "blade_touch",
    "blade_touch_time",
    "ellipse_closest",
    "segment_contact",
]
