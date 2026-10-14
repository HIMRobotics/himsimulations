"""
A paddle driven by the game's per-tick segments, as a reference-model trajectory.

The game drives each blade along one segment per render tick
(``lib/physics/engine/stepper.py`` ``PaddleSegment``): a cubic Hermite
position from ``(p0, v0)`` at ``t0`` to ``(p1, v1)`` at ``t1``, an orientation
slerped from ``q0`` to ``q1`` (``(w, x, y, z)``) with the constant angular
velocity ``rotvec(q1 q0*) / (t1 - t0)``, and optionally one interior knot
``(t, position, velocity, orientation)`` at a planned contact, which splits the
segment in two. Outside its span a segment is clamped: the position and
orientation hold their end values, and the velocity is the end velocity (as the
game evaluates it).

:class:`PaddleTrack` holds a sequence of such segments (the generator's
canonical paddle, for both data tiers) and evaluates them with the game's
expressions on Python floats, so the oracle meets the blade the game sweeps.
The reference model does not import the adapter: a segment is read by its
attributes (or mapping keys) ``t0, t1, p0, v0, p1, v1, q0, q1, knot``. A test
holds the track equal to ``PaddleSegment.at`` (position, velocity and angular
velocity to 1e-12; orientation to 1e-12 per matrix entry).

**Blade axes.** The game's blade axes are the columns of its rotation matrix:
X across the blade (width), Y towards the tip (length), Z the forehand normal.
The reference model's (:class:`lib.physics.reference.surfaces.PaddlePose`) are X
towards the handle (length), Y across (width), Z the forehand normal, so a pose
here is ``[-Y_game, X_game, Z_game]``: the same face, outline and inertia.

**Cheap gaps.** The oracle watches a blade every step. Besides the full pose,
the track gives the blade centre and face normal alone (:meth:`centre_normal`),
and per segment a sphere that contains the blade centre's whole path
(:meth:`sphere`; the Hermite curve lies in the convex hull of its Bezier control
points), so :class:`lib.physics.reference.surfaces.PaddleFaces` skips the pose
while the ball is far from every place the blade can be in that segment.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.physics.reference.surfaces import PaddlePose

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def _vec(value: Any) -> Vec3:
    x, y, z = (float(c) for c in np.asarray(value, dtype=float).reshape(3))
    return (x, y, z)


def _quat(value: Any) -> Quat:
    w, x, y, z = (float(c) for c in np.asarray(value, dtype=float).reshape(4))
    return (w, x, y, z)


def _field(segment: Any, name: str) -> Any:
    if isinstance(segment, Mapping):
        return segment[name]
    return getattr(segment, name)


def _rotvec(q: Quat) -> Vec3:
    """The game's ``_quat_rotvec``: normalise, take the short way round, then the rotation vector."""

    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    w, x, y, z = q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return (2.0 * x, 2.0 * y, 2.0 * z)
    angle = 2.0 * math.atan2(s, w)
    return (x / s * angle, y / s * angle, z / s * angle)


def _product(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


@dataclass(frozen=True)
class TrackPiece:
    """
    One Hermite piece of a track (a segment, or one half of a segment split at its knot), precomputed.

    :ivar t0: Piece start, s.
    :ivar t1: Piece end, s.
    :ivar p0: Blade centre at ``t0``, m.
    :ivar v0: Its velocity at ``t0``, m/s.
    :ivar p1: Blade centre at ``t1``, m.
    :ivar v1: Its velocity at ``t1``, m/s.
    :ivar q0: Orientation at ``t0`` (game axes, ``(w, x, y, z)``).
    :ivar q1: Orientation at ``t1``, made to lie on ``q0``'s side (as the game's slerp does).
    :ivar theta: Slerp angle, rad (``nan`` for the game's normalised-lerp branch, ``q0 . q1 > 0.9995``).
    :ivar angular_velocity: The piece's constant angular velocity, rad/s (the game's ``rotvec(q1 q0*) / h``).
    :ivar centre: Centre of a sphere containing the blade centre over the piece, m.
    :ivar radius: Its radius, m.
    :ivar speed: A bound on the blade centre's speed over the piece, m/s.
    """

    t0: float
    t1: float
    p0: Vec3
    v0: Vec3
    p1: Vec3
    v1: Vec3
    q0: Quat
    q1: Quat
    theta: float
    angular_velocity: Vec3
    centre: Vec3
    radius: float
    speed: float

    @classmethod
    def build(cls, t0: float, t1: float, p0: Vec3, v0: Vec3, p1: Vec3, v1: Vec3, q0: Quat, q1: Quat) -> TrackPiece:
        """A piece from its end states (the game's ``PaddleSegment`` without a knot)."""

        h = t1 - t0
        dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
        b = q1
        if dot < 0.0:
            b, dot = (-q1[0], -q1[1], -q1[2], -q1[3]), -dot
        theta = math.nan if dot > 0.9995 else math.acos(min(dot, 1.0))
        rate = _rotvec(_product(q1, (q0[0], -q0[1], -q0[2], -q0[3])))
        angular = (rate[0] / h, rate[1] / h, rate[2] / h)
        third = h / 3.0
        control = (
            p0,
            (p0[0] + third * v0[0], p0[1] + third * v0[1], p0[2] + third * v0[2]),
            (p1[0] - third * v1[0], p1[1] - third * v1[1], p1[2] - third * v1[2]),
            p1,
        )
        centre = tuple(sum(point[i] for point in control) / 4.0 for i in range(3))
        radius = max(math.dist(point, centre) for point in control) * (1.0 + 1e-9) + 1e-9
        speed = max(math.dist(control[i + 1], control[i]) * 3.0 / h for i in range(3)) * (1.0 + 1e-9)
        return cls(t0, t1, p0, v0, p1, v1, q0, b, theta, angular, (centre[0], centre[1], centre[2]), radius, speed)

    def _u(self, t: float) -> tuple[float, float]:
        h = self.t1 - self.t0
        return min(max((t - self.t0) / h, 0.0), 1.0), h

    def position(self, t: float) -> Vec3:
        """Blade centre at ``t``, m (the game's Hermite expression)."""

        u, h = self._u(t)
        u2, u3 = u * u, u * u * u
        c0, c1, c2, c3 = 2 * u3 - 3 * u2 + 1, (u3 - 2 * u2 + u) * h, -2 * u3 + 3 * u2, (u3 - u2) * h
        p0, v0, p1, v1 = self.p0, self.v0, self.p1, self.v1
        return (
            c0 * p0[0] + c1 * v0[0] + c2 * p1[0] + c3 * v1[0],
            c0 * p0[1] + c1 * v0[1] + c2 * p1[1] + c3 * v1[1],
            c0 * p0[2] + c1 * v0[2] + c2 * p1[2] + c3 * v1[2],
        )

    def velocity(self, t: float) -> Vec3:
        """Blade-centre velocity at ``t``, m/s (the game's derivative of the Hermite, at the clamped parameter)."""

        u, h = self._u(t)
        u2 = u * u
        c0, c1, c2, c3 = 6 * u2 - 6 * u, (3 * u2 - 4 * u + 1) * h, -6 * u2 + 6 * u, (3 * u2 - 2 * u) * h
        p0, v0, p1, v1 = self.p0, self.v0, self.p1, self.v1
        return (
            (c0 * p0[0] + c1 * v0[0] + c2 * p1[0] + c3 * v1[0]) / h,
            (c0 * p0[1] + c1 * v0[1] + c2 * p1[1] + c3 * v1[1]) / h,
            (c0 * p0[2] + c1 * v0[2] + c2 * p1[2] + c3 * v1[2]) / h,
        )

    def orientation(self, t: float) -> Quat:
        """Orientation at ``t`` in the game's axes, unnormalised as the game's slerp leaves it."""

        u, _h = self._u(t)
        a, b = self.q0, self.q1
        if math.isnan(self.theta):
            out = (a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), a[3] + u * (b[3] - a[3]))
            norm = math.sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3])
            return (out[0] / norm, out[1] / norm, out[2] / norm, out[3] / norm)
        theta = self.theta
        s = math.sin(theta)
        ka, kb = math.sin((1.0 - u) * theta), math.sin(u * theta)
        return ((ka * a[0] + kb * b[0]) / s, (ka * a[1] + kb * b[1]) / s, (ka * a[2] + kb * b[2]) / s, (ka * a[3] + kb * b[3]) / s)


def _game_axes(q: Quat) -> tuple[Vec3, Vec3, Vec3]:
    """The columns X, Y, Z of the game's ``quat_matrix(q)`` (it normalises ``q``)."""

    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    w, x, y, z = q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
        (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
        (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)),
    )


class PaddleTrack:
    """
    A blade driven by the game's per-tick segments: a :data:`lib.physics.reference.surfaces.PaddleTrajectory`.

    At time ``t`` the segment in force is the last one starting at or before ``t`` (the first one before the
    track starts), evaluated as the game does, clamped to its span. Segments with the same start keep the last.

    :param segments: The game's ``PaddleSegment`` objects, or anything with their attributes or keys.
    """

    def __init__(self, segments: Iterable[Any]) -> None:
        by_start: dict[float, Any] = {}
        for segment in segments:
            by_start[float(_field(segment, "t0"))] = segment
        if not by_start:
            raise ValueError("a paddle track needs at least one segment")
        starts = sorted(by_start)
        self.segment_starts: tuple[float, ...] = tuple(starts)
        """Each segment's start, s, in order."""
        self.segment_ends: tuple[float, ...] = tuple(float(_field(by_start[t], "t1")) for t in starts)
        """Each segment's end, s."""
        self.pieces: tuple[tuple[TrackPiece, ...], ...] = tuple(self._pieces(by_start[t]) for t in starts)
        """Each segment's Hermite pieces (two when its knot lies inside it)."""
        self.knots: tuple[float | None, ...] = tuple(piece[0].t1 if len(piece) == 2 else None for piece in self.pieces)
        """Each segment's interior knot time, s, if any."""
        self.speed_bound: float = max(piece.speed for pieces in self.pieces for piece in pieces)
        """A bound on the blade centre's speed over the whole track, m/s."""
        self.spin_bound: float = max(math.hypot(*piece.angular_velocity) for pieces in self.pieces for piece in pieces)
        """The largest angular speed over the whole track, rad/s."""
        self._last = 0

    @staticmethod
    def _pieces(segment: Any) -> tuple[TrackPiece, ...]:
        t0, t1 = float(_field(segment, "t0")), float(_field(segment, "t1"))
        p0, v0, p1, v1 = (_vec(_field(segment, name)) for name in ("p0", "v0", "p1", "v1"))
        q0, q1 = _quat(_field(segment, "q0")), _quat(_field(segment, "q1"))
        knot = _field(segment, "knot") if not isinstance(segment, Mapping) or "knot" in segment else None
        if knot is not None:
            tm = float(knot[0])
            if t0 < tm < t1:
                pm, vm, qm = _vec(knot[1]), _vec(knot[2]), _quat(knot[3])
                return (TrackPiece.build(t0, tm, p0, v0, pm, vm, q0, qm), TrackPiece.build(tm, t1, pm, vm, p1, v1, qm, q1))
        return (TrackPiece.build(t0, t1, p0, v0, p1, v1, q0, q1),)

    def index(self, t: float) -> int:
        """The segment in force at ``t``."""

        starts = self.segment_starts
        last = self._last
        if starts[last] <= t and (last + 1 == len(starts) or t < starts[last + 1]):
            return last
        found = max(bisect.bisect_right(starts, t) - 1, 0)
        self._last = found
        return found

    def piece(self, t: float) -> TrackPiece:
        """The Hermite piece in force at ``t`` (the knot belongs to the piece before it, as in the game)."""

        pieces = self.pieces[self.index(t)]
        if len(pieces) == 2 and t > pieces[0].t1:
            return pieces[1]
        return pieces[0]

    def sphere(self, t: float) -> tuple[float, float, float, float]:
        """A sphere ``(x, y, z, r)`` (m) containing the blade centre wherever the segment in force at ``t`` puts it."""

        pieces = self.pieces[self.index(t)]
        piece = pieces[0] if len(pieces) == 1 or t <= pieces[0].t1 else pieces[1]
        c = piece.centre
        return c[0], c[1], c[2], piece.radius

    def clearance(self, t0: float, t1: float, x: float, y: float, z: float) -> float:
        """The least distance (m) from a point to the spheres of every piece in force during ``[t0, t1]``."""

        first, last = self.index(t0), self.index(t1)
        best = math.inf
        for k in range(first, last + 1):
            for piece in self.pieces[k]:
                c = piece.centre
                best = min(best, math.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) - piece.radius)
        return best

    def point_speed_bound(self, reach: float) -> float:
        """A bound on the speed of any blade point within ``reach`` (m) of the centre, m/s."""

        return self.speed_bound + self.spin_bound * reach

    def centre_normal(self, t: float) -> tuple[float, float, float, float, float, float]:
        """The blade centre (m) and the forehand face normal at ``t``: what a face gap needs, without the full pose."""

        piece = self.piece(t)
        px, py, pz = piece.position(t)
        w, x, y, z = piece.orientation(t)
        norm2 = w * w + x * x + y * y + z * z
        w, x, y, z = (c / math.sqrt(norm2) for c in (w, x, y, z))
        return px, py, pz, 2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)

    def centre_axes(self, t: float) -> tuple[float, float, float, Vec3, Vec3, Vec3]:
        """The blade centre (m) and the game's blade axes at ``t``: X across, Y towards the tip, Z the forehand face."""

        piece = self.piece(t)
        px, py, pz = piece.position(t)
        x_axis, y_axis, z_axis = _game_axes(piece.orientation(t))
        return px, py, pz, x_axis, y_axis, z_axis

    def state(self, t: float) -> tuple[Vec3, Quat, Vec3, Vec3]:
        """The game's ``PaddleSegment.at(t)``: position, orientation ``(w, x, y, z)`` (game axes), velocity, angular velocity."""

        piece = self.piece(t)
        return piece.position(t), piece.orientation(t), piece.velocity(t), piece.angular_velocity

    def __call__(self, t: float) -> PaddlePose:
        """The blade's pose and twist at ``t`` in the reference model's axes."""

        position, orientation, velocity, angular = self.state(t)
        x_axis, y_axis, z_axis = _game_axes(orientation)
        rotation = np.array(
            [
                [-y_axis[0], x_axis[0], z_axis[0]],
                [-y_axis[1], x_axis[1], z_axis[1]],
                [-y_axis[2], x_axis[2], z_axis[2]],
            ]
        )
        return PaddlePose(np.array(position), rotation, np.array(velocity), np.array(angular))


def contact_windows(times: Sequence[float], *, before_s: float = 0.15, after_s: float = 0.05) -> tuple[tuple[float, float], ...]:
    """
    Active windows ``[t_c - before_s, t_c + after_s]`` around planned contact times, merged where they overlap.

    For :class:`lib.physics.reference.surfaces.ScriptedPaddle`'s ``windows``: outside them the paddle is not
    watched at all (an opt-in saving: a touch outside every window is not detected).
    """

    spans = sorted((float(t) - before_s, float(t) + after_s) for t in times)
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def paddle_rotation_from_game(q: ArrayLike) -> NDArray[np.float64]:
    """The reference model's blade rotation (columns: towards the handle, across, forehand normal) of a game quaternion."""

    x_axis, y_axis, z_axis = _game_axes(_quat(q))
    return np.array([[-y_axis[i], x_axis[i], z_axis[i]] for i in range(3)])


__all__ = ["PaddleTrack", "TrackPiece", "contact_windows", "paddle_rotation_from_game"]
