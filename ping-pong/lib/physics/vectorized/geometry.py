"""
Contact geometry, batched: the signed gaps and contact frames of the oracle's surfaces, and the blades.

Static surfaces, in the oracle's order (``simulate`` watches them in this order, and the earliest root wins with ties
to the lower index): the playing surface (:class:`lib.physics.reference.surfaces.TableTop`), the net box
(:class:`~lib.physics.reference.surfaces.Net`), the floor, the table board (:class:`~lib.physics.reference.surfaces.TableBoard`),
the two posts and the two clamps (:func:`~lib.physics.reference.surfaces.net_fixtures`). The undercarriage and the
room are not watched: a training episode ends when the ball leaves play.

A blade moves over a step as the game's ``PaddleSegment`` does (cubic Hermite position, slerped orientation with the
constant angular velocity ``rotvec(q1 q0*) / h``): :class:`BladeTrack`. Its touch condition is the oracle's: with the
rounded edge (:attr:`VecParams.blade_edge`, the default) :func:`lib.physics.reference.blade.blade_solid_gap` (the
face-only gap's own value wherever the face governs, so face touches are those of the face-only blade, bit for bit),
else :func:`lib.physics.reference.blade.blade_gap`. Quaternions are ``(w, x, y, z)`` and blade axes are the game's (X
across, Y towards the tip, Z the forehand face).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from lib.physics.canonical_events import region_code
from lib.physics.reference.blade import EDGE_ITERATIONS
from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.params import VecParams

Tensor = Any

TABLE_TOP, NET, FLOOR, BOARD, POST_NEG, POST_POS, CLAMP_NEG, CLAMP_POS = range(8)
"""Static surface indices, in the oracle's order."""
STATIC_COUNT = 8

MAT_TABLE, MAT_RUBBER, MAT_NET_CORD, MAT_NET_MESH, MAT_FLOOR, MAT_POST, MAT_FIXTURE, MAT_RIM = range(8)
"""Material codes: the contract's ``table_contact``, ``rubber_contact``, ``net_contact`` (rigid at the cord, the catch
law on the mesh), ``floor_contact``, ``post_contact``, ``fixture_contact`` and ``rim_contact`` (the blade's edge)."""

REGION = {name: region_code(name) for name in ("", "top", "edge", "side", "underside", "mesh", "cord", "post", "clamp", "blade_edge")}
"""Region codes (:data:`lib.physics.canonical_events.REGION_NAMES` indices)."""

OUTLINE_TOLERANCE = 1e-6
"""As :data:`lib.physics.reference.blade.OUTLINE_TOLERANCE`."""


def _take(value: Tensor, env: Tensor | None) -> Tensor:
    return value if env is None else value[env]


def _box_closest(pos: Tensor, low: Tensor, high: Tensor) -> tuple[Tensor, Tensor]:
    closest = torch.minimum(torch.maximum(pos, low), high)
    offset = pos - closest
    return closest, torch.sqrt((offset * offset).sum(-1))


def _cylinder_closest(pos: Tensor, ax: Tensor, ay: Tensor, rho: float, z0: float, z1: float) -> tuple[Tensor, Tensor]:
    dx, dy = pos[:, 0] - ax, pos[:, 1] - ay
    radial = torch.sqrt(dx * dx + dy * dy)
    outside = radial > rho
    scale = torch.where(outside, rho / torch.where(outside, radial, torch.ones_like(radial)), torch.ones_like(radial))
    px = torch.where(outside, ax + dx * scale, pos[:, 0])
    py = torch.where(outside, ay + dy * scale, pos[:, 1])
    pz = pos[:, 2].clamp(z0, z1)
    closest = torch.stack([px, py, pz], -1)
    offset = pos - closest
    return closest, torch.sqrt((offset * offset).sum(-1))


def _boxes(params: VecParams, env: Tensor | None, like: Tensor) -> dict[str, tuple[Tensor, Tensor]]:
    """The boxes of the board, the net and the clamps per row, ``(M, 3)`` corners."""

    zero = torch.zeros_like(like[:, 0])
    hl, hw = _take(params.table_half_length, env), _take(params.table_half_width, env)
    thick = _take(params.board_thickness, env)
    ht, hs, height = _take(params.net_half_thickness, env), _take(params.net_half_span, env), _take(params.net_height, env)
    cx, (cy0, cy1), (cz0, cz1) = params.clamps
    out = {
        "board": (torch.stack([-hl, -hw, -thick], -1), torch.stack([hl, hw, zero], -1)),
        "net": (torch.stack([-ht, -hs, zero], -1), torch.stack([ht, hs, height], -1)),
    }
    for side, sign in (("clamp_neg", -1.0), ("clamp_pos", 1.0)):
        y_low, y_high = (cy0, cy1) if sign > 0 else (-cy1, -cy0)
        out[side] = (torch.stack([zero - cx, zero + y_low, zero + cz0], -1), torch.stack([zero + cx, zero + y_high, zero + cz1], -1))
    return out


def static_gaps(pos: Tensor, params: VecParams, env: Tensor | None = None, *, continuous: bool = False) -> Tensor:
    """
    The static surfaces' gaps (m), ``(M, 8)``, the oracle's sign for sign but one: a centre over the playing surface
    less than a radius above it.

    The oracle's board gap is infinite over the playing surface (the top face is the top's). A centre that crosses into
    that domain below one radius has passed through the board's top edge; the oracle sees the touch only if one of its
    step ends falls inside that crossing (its grid restarts at the top's off-table root, so it usually does), the batch's
    global grid often does not. Here the board's gap stays finite there (``z - R``, the box distance, which meets the
    edge's distance continuously), so the touch fires whatever the grid; the root and the contact are the oracle's.

    :param pos: Ball centres ``(M, 3)``.
    :param params: The batch's physics.
    :param env: Environment of each row.
    :param continuous: The board's gap as the plain box distance even over the playing surface (where the oracle's is
        infinite, the top face being the playing surface's). Root finding uses it; detection does not.
    """

    radius = _take(params.radius, env)
    boxes = _boxes(params, env, pos)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    top = z - radius
    _c, net_d = _box_closest(pos, *boxes["net"])
    floor = z + _take(params.table_height, env) - radius
    _c, board_d = _box_closest(pos, *boxes["board"])
    board = board_d - radius
    if not continuous:
        over = (z >= 0.0) & (x.abs() <= _take(params.table_half_length, env)) & (y.abs() <= _take(params.table_half_width, env))
        board = torch.where(over & (top >= 0.0), torch.full_like(board, math.inf), board)
    posts_y, rho, (z0, z1) = params.posts
    zero = torch.zeros_like(x)
    py = _take(posts_y, env)
    _c, post_neg = _cylinder_closest(pos, zero, py[:, 0], rho, z0, z1)
    _c, post_pos = _cylinder_closest(pos, zero, py[:, 1], rho, z0, z1)
    _c, clamp_neg = _box_closest(pos, *boxes["clamp_neg"])
    _c, clamp_pos = _box_closest(pos, *boxes["clamp_pos"])
    return torch.stack(
        [top, net_d - radius, floor, board, post_neg - radius, post_pos - radius, clamp_neg - radius, clamp_pos - radius],
        -1,
    )


@dataclass(frozen=True)
class StaticContact:
    """
    Contact frames on static surfaces, one per row.

    :ivar valid: A contact (the oracle's ``contact`` is not ``None``), ``(M,)``.
    :ivar normal: From the surface into the ball, ``(M, 3)``.
    :ivar point: Contact point, ``(M, 3)``.
    :ivar region: Region code, ``(M,)``.
    :ivar material: Material code, ``(M,)``.
    """

    valid: Tensor
    normal: Tensor
    point: Tensor
    region: Tensor
    material: Tensor


def static_contact(surface: Tensor, pos: Tensor, params: VecParams, env: Tensor) -> StaticContact:
    """
    The contact frame of each row's static surface at its ball centre (:meth:`Surface.contact` of the oracle).

    :param surface: Static surface index per row, ``(M,)``.
    :param pos: Ball centres ``(M, 3)``.
    :param params: The batch's physics.
    :param env: Environment of each row.
    """

    m = pos.shape[0]
    zero = torch.zeros(m, dtype=pos.dtype, device=pos.device)
    up = torch.stack([zero, zero, zero + 1.0], -1)
    boxes = _boxes(params, env, pos)
    hl, hw = params.table_half_length[env], params.table_half_width[env]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    long = torch.full((m,), 0, dtype=torch.long, device=pos.device)

    def frame(closest: Tensor, distance: Tensor) -> tuple[Tensor, Tensor]:
        ok = distance >= 1e-12
        normal = (pos - closest) / torch.where(ok, distance, torch.ones_like(distance))[:, None]
        return ok, normal

    # Defaults: the playing surface.
    valid = (x.abs() <= hl) & (y.abs() <= hw)
    normal = up.clone()
    point = torch.stack([x, y, zero], -1)
    region = long + REGION["top"]
    material = long + MAT_TABLE

    def put(mask: Tensor, ok: Tensor, n: Tensor, p: Tensor, reg: Tensor, mat: Tensor) -> None:
        nonlocal valid, normal, point, region, material
        valid = torch.where(mask, ok, valid)
        normal = torch.where(mask[:, None], n, normal)
        point = torch.where(mask[:, None], p, point)
        region = torch.where(mask, reg, region)
        material = torch.where(mask, mat, material)

    # Net box: mesh below the tape on a face, cord elsewhere.
    closest, distance = _box_closest(pos, *boxes["net"])
    ok, n = frame(closest, distance)
    ht, hs, height, tape = params.net_half_thickness[env], params.net_half_span[env], params.net_height[env], params.net_tape_height[env]
    mesh = (closest[:, 0].abs() >= ht - 1e-12) & (closest[:, 1].abs() < hs) & (closest[:, 2] < height - tape)
    put(
        surface == NET,
        ok,
        n,
        closest,
        torch.where(mesh, long + REGION["mesh"], long + REGION["cord"]),
        torch.where(mesh, long + MAT_NET_MESH, long + MAT_NET_CORD),
    )
    # Floor.
    floor_point = torch.stack([x, y, -params.table_height[env]], -1)
    put(surface == FLOOR, torch.ones_like(valid), up, floor_point, long + REGION[""], long + MAT_FLOOR)
    # Board: its region from the normal; a centre over the playing surface belongs to the top.
    closest, distance = _box_closest(pos, *boxes["board"])
    ok, n = frame(closest, distance)
    over = (z >= 0.0) & (x.abs() <= hl) & (y.abs() <= hw)
    nz = n[:, 2]
    touch = torch.where(
        nz >= 1.0 - 1e-9,
        long + REGION["top"],
        torch.where(nz > 1e-9, long + REGION["edge"], torch.where(nz >= -1e-9, long + REGION["side"], long + REGION["underside"])),
    )
    put(surface == BOARD, ok & ~over, n, closest, touch, long + MAT_TABLE)
    # Posts.
    posts_y, rho, (z0, z1) = params.posts
    for index, column in ((POST_NEG, 0), (POST_POS, 1)):
        closest, distance = _cylinder_closest(pos, zero, posts_y[env, column], rho, z0, z1)
        ok, n = frame(closest, distance)
        put(surface == index, ok, n, closest, long + REGION["post"], long + MAT_POST)
    # Clamps.
    for index, key in ((CLAMP_NEG, "clamp_neg"), (CLAMP_POS, "clamp_pos")):
        closest, distance = _box_closest(pos, *boxes[key])
        ok, n = frame(closest, distance)
        put(surface == index, ok, n, closest, long + REGION["clamp"], long + MAT_FIXTURE)
    return StaticContact(valid, normal, point, region, material)


# ---- blades ---------------------------------------------------------------------------------------------------------


def quat_axes(q: Tensor) -> Tensor:
    """The game's ``quat_matrix`` of ``(w, x, y, z)`` quaternions ``(..., 4)``: ``(..., 3, 3)``, columns X, Y, Z."""

    q = q / torch.sqrt((q * q).sum(-1, keepdim=True))
    w, x, y, z = q.unbind(-1)
    rows = [
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ]
    return torch.stack(rows, -2)


def _quat_product(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def _rotvec(q: Tensor) -> Tensor:
    """The game's ``_quat_rotvec``: normalise, take the short way round, then the rotation vector."""

    q = q / torch.sqrt((q * q).sum(-1, keepdim=True))
    q = torch.where(q[..., :1] < 0.0, -q, q)
    vec = q[..., 1:]
    s = torch.sqrt((vec * vec).sum(-1))
    small = s < 1e-12
    angle = 2.0 * torch.atan2(s, q[..., 0])
    scale = torch.where(small, torch.full_like(s, 2.0), angle / torch.where(small, torch.ones_like(s), s))
    return vec * scale[..., None]


@dataclass(frozen=True)
class BladeTrack:
    """
    Every blade's motion over one step, as the game's ``PaddleSegment`` (no knot): ``(N, B, ...)``.

    :ivar p0: Blade centre at the step's start, m.
    :ivar v0: Its velocity, m/s.
    :ivar q0: Orientation, ``(w, x, y, z)``, game axes.
    :ivar p1: Blade centre at the step's end.
    :ivar v1: Its velocity.
    :ivar q1: Orientation at the end, on ``q0``'s side.
    :ivar theta: Slerp angle, rad (unused where ``nlerp``).
    :ivar nlerp: The game's normalised-lerp branch (``q0 . q1 > 0.9995``).
    :ivar omega: The constant angular velocity ``rotvec(q1 q0*) / h``, rad/s.
    :ivar h: The step, s.
    """

    p0: Tensor
    v0: Tensor
    q0: Tensor
    p1: Tensor
    v1: Tensor
    q1: Tensor
    theta: Tensor
    nlerp: Tensor
    omega: Tensor
    h: float

    @classmethod
    def from_states(cls, p0: Tensor, v0: Tensor, q0: Tensor, p1: Tensor, v1: Tensor, q1: Tensor, h: float) -> BladeTrack:
        """A step's blade motion from its states at the step's ends (as the game builds a ``PaddleSegment``)."""

        dot = (q0 * q1).sum(-1)
        flip = dot < 0.0
        q1s = torch.where(flip[..., None], -q1, q1)
        dot = torch.where(flip, -dot, dot)
        nlerp = dot > 0.9995
        theta = torch.acos(dot.clamp(max=1.0))
        conj = q0 * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q0.dtype, device=q0.device)
        omega = _rotvec(_quat_product(q1, conj)) / h
        return cls(p0, v0, q0, p1, v1, q1s, theta, nlerp, omega, float(h))

    @property
    def count(self) -> int:
        """Blades per environment."""

        return int(self.p0.shape[1])

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        """``(p0, v0, q0, p1, v1, q1, theta, nlerp, omega)``: what :func:`blade_state` reads."""

        return (self.p0, self.v0, self.q0, self.p1, self.v1, self.q1, self.theta, self.nlerp, self.omega)

    def state(self, u: Tensor, env: Tensor, blade: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Blade centre, orientation, velocity and angular velocity at fractions ``u`` of the step.

        :param u: Step fraction per row, ``(M,)``, clamped to ``[0, 1]``.
        :param env: Environment per row.
        :param blade: Blade per row.
        :return: ``(M, 3)``, ``(M, 4)``, ``(M, 3)``, ``(M, 3)``.
        """

        return blade_state(self.tensors, self.h, u, env, blade)

    def split(self, parts: int) -> list[BladeTrack]:
        """This motion as ``parts`` consecutive sub-steps (each the exact restriction of the cubic and the slerp)."""

        n, b = self.p0.shape[0], self.p0.shape[1]
        env = torch.arange(n, device=self.p0.device).repeat_interleave(b)
        blade = torch.arange(b, device=self.p0.device).repeat(n)
        states = []
        for k in range(parts + 1):
            u = torch.full((n * b,), k / parts, dtype=self.p0.dtype, device=self.p0.device)
            pos, quat, vel, _omega = self.state(u, env, blade)
            states.append((pos.reshape(n, b, 3), quat.reshape(n, b, 4), vel.reshape(n, b, 3)))
        h = self.h / parts
        out = []
        for k in range(parts):
            (pa, qa, va), (pb, qb, vb) = states[k], states[k + 1]
            out.append(BladeTrack.from_states(pa, va, qa, pb, vb, qb, h))
        return out


def blade_state(tensors: tuple[Tensor, ...], h: float, u: Tensor, env: Tensor, blade: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Blade centre, orientation, velocity and angular velocity at step fractions ``u`` (:meth:`BladeTrack.state`).

    :param tensors: :attr:`BladeTrack.tensors`.
    :param h: The step, s.
    :param u: Step fraction per row ``(M,)``, clamped to ``[0, 1]``.
    :param env: Environment per row.
    :param blade: Blade per row.
    """

    p0_all, v0_all, q0_all, p1_all, v1_all, q1_all, theta_all, nlerp_all, omega_all = tensors
    u = u.clamp(0.0, 1.0)
    p0, v0, p1, v1 = p0_all[env, blade], v0_all[env, blade], p1_all[env, blade], v1_all[env, blade]
    u2, u3 = u * u, u * u * u
    c0, c1, c2, c3 = 2 * u3 - 3 * u2 + 1, (u3 - 2 * u2 + u) * h, -2 * u3 + 3 * u2, (u3 - u2) * h
    position = c0[:, None] * p0 + c1[:, None] * v0 + c2[:, None] * p1 + c3[:, None] * v1
    d0, d1, d2, d3 = 6 * u2 - 6 * u, (3 * u2 - 4 * u + 1) * h, -6 * u2 + 6 * u, (3 * u2 - 2 * u) * h
    velocity = (d0[:, None] * p0 + d1[:, None] * v0 + d2[:, None] * p1 + d3[:, None] * v1) / h
    a, b = q0_all[env, blade], q1_all[env, blade]
    theta = theta_all[env, blade]
    lerped = a + u[:, None] * (b - a)
    lerped = lerped / torch.sqrt((lerped * lerped).sum(-1, keepdim=True))
    s = torch.sin(theta)
    safe = torch.where(s > 0.0, s, torch.ones_like(s))
    slerped = (torch.sin((1.0 - u) * theta)[:, None] * a + torch.sin(u * theta)[:, None] * b) / safe[:, None]
    orientation = torch.where(nlerp_all[env, blade][:, None], lerped, slerped)
    return position, orientation, velocity, omega_all[env, blade]


def blade_gaps_of(pos: Tensor, tensors: tuple[Tensor, ...], h: float, u: Tensor, params: VecParams, env: Tensor) -> Tensor:
    """:func:`blade_gaps` from :attr:`BladeTrack.tensors`."""

    m, b = pos.shape[0], tensors[0].shape[1]
    rows_env = env.repeat_interleave(b)
    rows_blade = torch.arange(b, device=pos.device).repeat(m)
    position, orientation, _v, _w = blade_state(tensors, h, u.repeat_interleave(b), rows_env, rows_blade)
    gap = blade_gap(pos.repeat_interleave(b, 0), position, quat_axes(orientation), params, rows_env)
    return gap.reshape(m, b)


def ellipse_closest(x: Tensor, y: Tensor, semi_x: Tensor, semi_y: Tensor) -> tuple[Tensor, Tensor]:
    """
    :func:`lib.physics.reference.blade.ellipse_closest` per element: the outline's point nearest ``(x, y)``, by the
    same :data:`~lib.physics.reference.blade.EDGE_ITERATIONS` fixed steps (element-wise, so it compiles and captures).
    """

    px, py = x.abs(), y.abs()
    tx = torch.full_like(px, 0.7071067811865476)
    ty = torch.full_like(py, 0.7071067811865476)
    a, b = semi_x, semi_y
    c2 = a * a - b * b
    for _ in range(EDGE_ITERATIONS):
        ex = c2 * tx * tx * tx / a
        ey = -c2 * ty * ty * ty / b
        rx, ry = a * tx - ex, b * ty - ey
        qx, qy = px - ex, py - ey
        r = torch.sqrt(rx * rx + ry * ry)
        q = torch.sqrt(qx * qx + qy * qy)
        q = torch.where(q > 0.0, q, torch.ones_like(q))
        tx = ((qx * r / q + ex) / a).clamp(0.0, 1.0)
        ty = ((qy * r / q + ey) / b).clamp(0.0, 1.0)
        norm = torch.sqrt(tx * tx + ty * ty)
        tx, ty = tx / norm, ty / norm
    return torch.copysign(a * tx, x), torch.copysign(b * ty, y)


def edge_gap(across_m: Tensor, along_m: Tensor, distance: Tensor, radius: Tensor, half_thickness: Tensor, hl: Tensor, hw: Tensor) -> Tensor:
    """:func:`lib.physics.reference.blade.blade_edge_gap` per element (coordinates in metres in the blade axes)."""

    qx, qy = ellipse_closest(across_m, along_m, hw, hl)
    dx, dy = across_m - qx, along_m - qy
    return torch.sqrt(dx * dx + dy * dy + distance * distance) - (radius + half_thickness)


def blade_gap(pos: Tensor, blade_position: Tensor, axes: Tensor, params: VecParams, env: Tensor) -> Tensor:
    """
    The touch condition per row: ``max(|n . r| - (R + h), (e - 1) min(a, b))``
    (:func:`lib.physics.reference.blade.blade_gap`), and with :attr:`VecParams.blade_edge` the rounded edge as in
    :func:`lib.physics.reference.blade.blade_solid_gap` (that value wherever it is negative or the centre is strictly
    inside the outline, else the smaller of it and the edge gap).
    """

    r = pos - blade_position
    radius, half_thickness = params.radius[env], params.blade_half_thickness[env]
    hl, hw = params.blade_half_length[env], params.blade_half_width[env]
    distance = (r * axes[:, :, 2]).sum(-1)
    band = distance.abs() - (radius + half_thickness)
    across_m = (r * axes[:, :, 0]).sum(-1)
    along_m = (r * axes[:, :, 1]).sum(-1)
    across = across_m / hw
    along = along_m / hl
    outline = (torch.sqrt(across * across + along * along) - 1.0) * torch.minimum(hl, hw)
    face = torch.maximum(band, outline)
    if not params.blade_edge:
        return face
    edge = edge_gap(across_m, along_m, distance, radius, half_thickness, hl, hw)
    governs = (face < 0.0) | (outline < 0.0)
    return torch.where(governs, face, torch.minimum(face, edge))


@dataclass(frozen=True)
class BladeContact:
    """
    The blade's contact state per row (:class:`lib.physics.reference.blade.BladeContact`).

    :ivar valid: On the blade (outline within :data:`OUTLINE_TOLERANCE`) and off the mid-plane, ``(M,)``.
    :ivar normal: Struck face's normal into the ball.
    :ivar point: Touching point.
    :ivar offset: ``point`` from the blade centre.
    :ivar surface_velocity: Velocity of the blade's material point at ``point``.
    :ivar inertia: The paddle's inertia tensor in the table frame, ``(M, 3, 3)``.
    :ivar rim: The touch came through the outline (``rim``) rather than the face plane.
    :ivar edge: The touch is on the blade's rounded edge (:attr:`VecParams.blade_edge`): the normal from the outline's
        nearest point into the ball, resolved with ``rim_contact``, region ``blade_edge``.
    """

    valid: Tensor
    normal: Tensor
    point: Tensor
    offset: Tensor
    surface_velocity: Tensor
    inertia: Tensor
    rim: Tensor
    edge: Tensor


def blade_contact(pos: Tensor, blade_position: Tensor, orientation: Tensor, velocity: Tensor, omega: Tensor, params: VecParams, env: Tensor) -> BladeContact:
    """The contact frame of balls centred at ``pos`` against blades in the given states, one per row."""

    axes = quat_axes(orientation)
    radius = params.radius[env]
    face = axes[:, :, 2]
    r = pos - blade_position
    d = (face * r).sum(-1)
    side = torch.where(d >= 0.0, torch.ones_like(d), -torch.ones_like(d))
    normal = side[:, None] * face
    point = pos - radius[:, None] * normal
    offset = point - blade_position
    hl, hw = params.blade_half_length[env], params.blade_half_width[env]
    across = (offset * axes[:, :, 0]).sum(-1) / hw
    along = (offset * axes[:, :, 1]).sum(-1) / hl
    outline2 = across * across + along * along
    band = d.abs() - (radius + params.blade_half_thickness[env])
    rim = (torch.sqrt(outline2) - 1.0) * torch.minimum(hl, hw) > band
    valid = (d != 0.0) & (outline2 <= 1.0 + OUTLINE_TOLERANCE)
    edge = torch.zeros_like(valid)
    if params.blade_edge:
        # The edge's frame where it governs the gap (lib.physics.reference.blade.blade_solid_contact).
        half_thickness = params.blade_half_thickness[env]
        across_m = (r * axes[:, :, 0]).sum(-1)
        along_m = (r * axes[:, :, 1]).sum(-1)
        c_outline = (torch.sqrt((across_m / hw) ** 2 + (along_m / hl) ** 2) - 1.0) * torch.minimum(hl, hw)
        face_gap = torch.maximum(band, c_outline)
        edge_g = edge_gap(across_m, along_m, d, radius, half_thickness, hl, hw)
        edge = (face_gap >= 0.0) & (c_outline >= 0.0) & (edge_g < face_gap)
        qx, qy = ellipse_closest(across_m, along_m, hw, hl)
        rim_point = blade_position + qx[:, None] * axes[:, :, 0] + qy[:, None] * axes[:, :, 1]
        gap = pos - rim_point
        length = torch.sqrt((gap * gap).sum(-1))
        edge_normal = gap / torch.where(length > 0.0, length, torch.ones_like(length))[:, None]
        normal = torch.where(edge[:, None], edge_normal, normal)
        point = torch.where(edge[:, None], pos - radius[:, None] * edge_normal, point)
        offset = point - blade_position
        valid = torch.where(edge, length > 0.0, valid)
    surface_velocity = velocity + torch.cross(omega, offset, dim=-1)
    diag = torch.stack([params.paddle_inertia_yy[env], params.paddle_inertia_xx[env], params.paddle_inertia_zz[env]], -1)
    inertia = ((axes * diag[:, None, :])[:, :, :, None] * axes.transpose(-1, -2)[:, None, :, :]).sum(-2)
    return BladeContact(valid, normal, point, offset, surface_velocity, inertia, rim, edge)


__all__ = [
    "BOARD",
    "CLAMP_NEG",
    "CLAMP_POS",
    "FLOOR",
    "MAT_FIXTURE",
    "MAT_FLOOR",
    "MAT_NET_CORD",
    "MAT_NET_MESH",
    "MAT_POST",
    "MAT_RIM",
    "MAT_RUBBER",
    "MAT_TABLE",
    "NET",
    "POST_NEG",
    "POST_POS",
    "REGION",
    "STATIC_COUNT",
    "TABLE_TOP",
    "BladeContact",
    "BladeTrack",
    "StaticContact",
    "blade_contact",
    "blade_gap",
    "blade_gaps_of",
    "blade_state",
    "edge_gap",
    "ellipse_closest",
    "quat_axes",
    "static_contact",
    "static_gaps",
]
