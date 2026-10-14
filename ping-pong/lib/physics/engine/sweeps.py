"""
The hybrid contact sweeps over one physics step, batched and array-generic.

Every function takes the array namespace ``xp`` (``numpy`` here, ``torch`` on a
GPU) and arrays whose last axis is the vector axis, so one implementation runs
on one ball or a batch of thousands. :mod:`lib.physics.engine.contacts` (the
replay's adapter) calls :func:`sweep_net`, :func:`box_gap` and
:func:`sweep_shape` for its net, posts and clamps; :func:`sweep_table` and
:func:`sweep_table_box` are the table's.

Only operations numpy and torch spell the same way are used: arithmetic,
comparisons, ``xp.where``, ``xp.stack``, ``xp.sin``/``xp.cos``, ``sum(-1)`` and
``@``. Quaternions are ``(x, y, z, w)``, as Isaac Lab writes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Array = Any
"""A numpy array or a torch tensor; the vector axis is last."""


def dot(a: Array, b: Array) -> Array:
    """Dot product over the last axis."""

    return (a * b).sum(-1)


def norm(a: Array) -> Array:
    """Euclidean norm over the last axis."""

    return dot(a, a) ** 0.5


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


# --------------------------------------------------------------------------------------
# Impact
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Rotations and the scripted paddle
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Hybrid contact sweeps over one step
# --------------------------------------------------------------------------------------


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


TABLE_START_SLOP_M: float = 2e-4
"""A ball that starts a step up to this far below the playing surface, falling and over the table, touches it at once.

The sweep predicts the step's end from the state it reads, and the engine's actual end can lie slightly lower: the
float32 read-back of an environment metres from the origin is good to about 1e-6 m, and an integrator that applies
the step's velocity kick before moving the ball ends up to ``|a| dt^2`` lower (1.6e-6 m at 1/4000 s). A ball whose
predicted end sat in that sliver above the plane started the next step just below it and was never swept again: it
fell through the table (about one bounce in a thousand, far from the origin; none of 3000 at the origin). 0.2 mm
covers either and is far below anything the ball tracking resolves; deeper than that the ball can only be under the
table, which this sweep must not touch.
"""


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


__all__ = [
    "NET_SWEEP_BISECTIONS",
    "NET_SWEEP_SAMPLES",
    "TABLE_START_SLOP_M",
    "Shape",
    "SweepBatch",
    "box_gap",
    "dot",
    "norm",
    "shape_gap",
    "sweep_net",
    "sweep_shape",
]
