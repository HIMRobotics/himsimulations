"""
The root finder's inner loop written per component, for :class:`lib.physics.vectorized.dense.DenseBallBatch`.

An Illinois step evaluates one gap on the RK4 path. Written with ``(M, 3)`` vectors and stacked gap tables
(:func:`lib.physics.vectorized.sim.pair_gap_fn`) it mixes ``(M, 3)``, ``(M, 8)`` and ``(M,)`` shapes, and a compiler
splits it into about twenty kernels per step (measured on an RTX PRO 6000: 19 kernels, 68 us per step). Here every
quantity is an ``(M,)`` column: the pair's constants are gathered once (:func:`pair_constants`), and a step is pure
element-wise arithmetic, which ``torch.compile`` fuses into a few kernels. The expressions are those of
:func:`lib.physics.vectorized.flight.rk4_step`, :func:`lib.physics.vectorized.geometry.static_gaps` (continuous form)
and :func:`lib.physics.vectorized.geometry.blade_gap`, in the same order, so the values agree to rounding. ``edge``
(:attr:`VecParams.blade_edge`, a constant of the compiled step) gives the blade its rounded edge.
"""

from __future__ import annotations

from typing import Any

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.geometry import CLAMP_NEG, CLAMP_POS, FLOOR, NET, POST_NEG, POST_POS, STATIC_COUNT, TABLE_TOP, _boxes, edge_gap
from lib.physics.vectorized.params import VecParams

Tensor = Any

# The pair's gap kind.
KIND_TOP, KIND_BOX, KIND_FLOOR, KIND_CYLINDER, KIND_BLADE = range(5)

# Columns of the constants table.
_FLIGHT = ("gravity", "radius", "gain", "drag0", "drag_slope", "slope", "c_max", "decay_gain")
_GEOMETRY = ("floor_offset", "low_x", "low_y", "low_z", "high_x", "high_y", "high_z", "axis_y")
_BLADE = (
    "p0x", "p0y", "p0z", "v0x", "v0y", "v0z", "p1x", "p1y", "p1z", "v1x", "v1y", "v1z",
    "q0w", "q0x", "q0y", "q0z", "q1w", "q1x", "q1y", "q1z", "theta", "sin_theta", "half_thickness", "half_length",
    "half_width",
)  # fmt: skip
_START = ("a1x", "a1y", "a1z", "b1x", "b1y", "b1z")
COLUMNS = _FLIGHT + _GEOMETRY + _BLADE + _START
_AT = {name: i for i, name in enumerate(COLUMNS)}


def pair_constants(params: VecParams, ys: Tensor, row: Tensor, surface: Tensor, blades: tuple[Tensor, ...] | None) -> tuple[Tensor, Tensor, Tensor]:
    """
    Each pair's constants: ``(table (M, len(COLUMNS)), kind (M,), nlerp (M,))``, including the RK4 path's first
    stage (the acceleration at the pair's start state, the same for every trial step).

    :param ys: Each pair's state at its round's start, ``(M, 9)``.
    :param row: Environment of each pair.
    :param surface: Surface of each pair (static index, or ``STATIC_COUNT`` + blade).
    :param blades: :attr:`lib.physics.vectorized.geometry.BladeTrack.tensors`, or ``None``.
    """

    zero = torch.zeros(row.shape[0], dtype=params.dtype, device=row.device)
    columns = [
        params.gravity[row],
        params.radius[row],
        params.aero_gain[row],
        params.drag_coefficient[row],
        params.drag_spin_slope[row],
        params.lift_slope[row],
        params.lift_c_max[row],
        params.spin_decay_gain[row],
    ]
    boxes = _boxes(params, row, zero[:, None])
    low = torch.where((surface == NET)[:, None], boxes["net"][0], boxes["board"][0])
    high = torch.where((surface == NET)[:, None], boxes["net"][1], boxes["board"][1])
    low = torch.where((surface == CLAMP_NEG)[:, None], boxes["clamp_neg"][0], torch.where((surface == CLAMP_POS)[:, None], boxes["clamp_pos"][0], low))
    high = torch.where((surface == CLAMP_NEG)[:, None], boxes["clamp_neg"][1], torch.where((surface == CLAMP_POS)[:, None], boxes["clamp_pos"][1], high))
    posts_y = params.posts[0][row]
    axis_y = torch.where(surface == POST_NEG, posts_y[:, 0], posts_y[:, 1])
    columns += [params.table_height[row], low[:, 0], low[:, 1], low[:, 2], high[:, 0], high[:, 1], high[:, 2], axis_y]
    kind = torch.full_like(surface, KIND_BOX)
    kind = torch.where(surface == TABLE_TOP, torch.full_like(kind, KIND_TOP), kind)
    kind = torch.where(surface == FLOOR, torch.full_like(kind, KIND_FLOOR), kind)
    kind = torch.where((surface == POST_NEG) | (surface == POST_POS), torch.full_like(kind, KIND_CYLINDER), kind)
    kind = torch.where(surface >= STATIC_COUNT, torch.full_like(kind, KIND_BLADE), kind)
    if blades is None:
        columns += [zero] * len(_BLADE)
        nlerp = torch.zeros_like(surface, dtype=torch.bool)
    else:
        p0, v0, q0, p1, v1, q1, theta, nlerp_all, _omega = blades
        blade = (surface - STATIC_COUNT).clamp(min=0)
        for vector in (p0, v0, p1, v1):
            picked = vector[row, blade]
            columns += [picked[:, 0], picked[:, 1], picked[:, 2]]
        for quat in (q0, q1):
            picked = quat[row, blade]
            columns += [picked[:, 0], picked[:, 1], picked[:, 2], picked[:, 3]]
        picked_theta = theta[row, blade]
        columns += [picked_theta, torch.sin(picked_theta), params.blade_half_thickness[row], params.blade_half_length[row], params.blade_half_width[row]]
        nlerp = nlerp_all[row, blade]
    c = {name: column for name, column in zip(_FLIGHT, columns[: len(_FLIGHT)], strict=True)}
    columns += list(_acceleration(c, params.lift_form, ys[:, 3], ys[:, 4], ys[:, 5], ys[:, 6], ys[:, 7], ys[:, 8]))
    return torch.stack(columns, -1), kind, nlerp


def _acceleration(c: dict[str, Tensor], lift_form: str, vx: Tensor, vy: Tensor, vz: Tensor, wx: Tensor, wy: Tensor, wz: Tensor) -> tuple[Tensor, ...]:
    """:func:`lib.physics.vectorized.flight.acceleration`, per component."""

    v2 = vx * vx + vy * vy + vz * vz
    moving = v2 > 0.0
    speed = torch.sqrt(v2)
    safe_v2 = torch.where(moving, v2, torch.ones_like(v2))
    wv = wx * vx + wy * vy + wz * vz
    wperp2 = wx * wx + wy * wy + wz * wz - wv * wv / safe_v2
    wperp = torch.where(moving & (wperp2 > 0.0), torch.sqrt(wperp2.clamp_min(0.0)), torch.zeros_like(wperp2))
    s = c["radius"] * wperp / torch.where(moving, speed, torch.ones_like(speed))
    drag = c["gain"] * (c["drag0"] + c["drag_slope"] * s) * speed
    ax = 0.0 - drag * vx
    ay = 0.0 - drag * vy
    az = -c["gravity"] - drag * vz
    spinning = wperp > 0.0
    slope, c_max = c["slope"], c["c_max"]
    if lift_form == "rational":
        lift = slope * s / (1.0 + slope * s / c_max)
    elif lift_form == "linear":
        lift = torch.minimum(slope * s, c_max)
    else:
        lift = c_max + torch.zeros_like(s)
    lift = torch.where(s > 0.0, lift, torch.zeros_like(lift))
    magnus = torch.where(spinning, c["gain"] * lift * speed / torch.where(spinning, wperp, torch.ones_like(wperp)), torch.zeros_like(wperp))
    ax = ax + magnus * (wy * vz - wz * vy)
    ay = ay + magnus * (wz * vx - wx * vz)
    az = az + magnus * (wx * vy - wy * vx)
    decay = c["decay_gain"] * speed
    return ax, ay, az, -decay * wx, -decay * wy, -decay * wz


def rk4_position(c: dict[str, Tensor], lift_form: str, y: tuple[Tensor, ...], h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The position after one RK4 step of ``h`` (:func:`lib.physics.vectorized.flight.rk4_step`), per component; the
    first stage comes from the constants (columns ``a1*``, ``b1*``)."""

    px, py, pz, vx, vy, vz, wx, wy, wz = y
    half = 0.5 * h
    a1 = (c["a1x"], c["a1y"], c["a1z"], c["b1x"], c["b1y"], c["b1z"])
    v2 = (vx + half * a1[0], vy + half * a1[1], vz + half * a1[2])
    w2 = (wx + half * a1[3], wy + half * a1[4], wz + half * a1[5])
    a2 = _acceleration(c, lift_form, *v2, *w2)
    v3 = (vx + half * a2[0], vy + half * a2[1], vz + half * a2[2])
    w3 = (wx + half * a2[3], wy + half * a2[4], wz + half * a2[5])
    a3 = _acceleration(c, lift_form, *v3, *w3)
    v4 = (vx + h * a3[0], vy + h * a3[1], vz + h * a3[2])
    sixth = h / 6.0
    return (
        px + sixth * (vx + 2.0 * v2[0] + 2.0 * v3[0] + v4[0]),
        py + sixth * (vy + 2.0 * v2[1] + 2.0 * v3[1] + v4[1]),
        pz + sixth * (vz + 2.0 * v2[2] + 2.0 * v3[2] + v4[2]),
    )


def _blade_gap(c: dict[str, Tensor], nlerp: Tensor, h: float, u: Tensor, x: Tensor, y: Tensor, z: Tensor, edge: bool) -> Tensor:
    """:func:`lib.physics.vectorized.geometry.blade_state` then :func:`blade_gap`, per component."""

    u = u.clamp(0.0, 1.0)
    u2, u3 = u * u, u * u * u
    c0, c1, c2, c3 = 2 * u3 - 3 * u2 + 1, (u3 - 2 * u2 + u) * h, -2 * u3 + 3 * u2, (u3 - u2) * h
    bx = c0 * c["p0x"] + c1 * c["v0x"] + c2 * c["p1x"] + c3 * c["v1x"]
    by = c0 * c["p0y"] + c1 * c["v0y"] + c2 * c["p1y"] + c3 * c["v1y"]
    bz = c0 * c["p0z"] + c1 * c["v0z"] + c2 * c["p1z"] + c3 * c["v1z"]
    a = (c["q0w"], c["q0x"], c["q0y"], c["q0z"])
    b = (c["q1w"], c["q1x"], c["q1y"], c["q1z"])
    lerped = tuple(ai + u * (bi - ai) for ai, bi in zip(a, b, strict=True))
    norm = torch.sqrt(lerped[0] * lerped[0] + lerped[1] * lerped[1] + lerped[2] * lerped[2] + lerped[3] * lerped[3])
    lerped = tuple(q / norm for q in lerped)
    theta = c["theta"]
    s = c["sin_theta"]
    safe = torch.where(s > 0.0, s, torch.ones_like(s))
    wa, wb = torch.sin((1.0 - u) * theta), torch.sin(u * theta)
    slerped = tuple((wa * ai + wb * bi) / safe for ai, bi in zip(a, b, strict=True))
    q = tuple(torch.where(nlerp, li, si) for li, si in zip(lerped, slerped, strict=True))
    # quat_axes: normalise, then the columns X, Y, Z of the rotation.
    norm = torch.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    qw, qx, qy, qz = (component / norm for component in q)
    x_axis = (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw))
    y_axis = (2 * (qx * qy - qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qx * qw))
    z_axis = (2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx * qx + qy * qy))
    rx, ry, rz = x - bx, y - by, z - bz
    hl, hw = c["half_length"], c["half_width"]
    distance = rx * z_axis[0] + ry * z_axis[1] + rz * z_axis[2]
    band = distance.abs() - (c["radius"] + c["half_thickness"])
    across_m = rx * x_axis[0] + ry * x_axis[1] + rz * x_axis[2]
    along_m = rx * y_axis[0] + ry * y_axis[1] + rz * y_axis[2]
    across = across_m / hw
    along = along_m / hl
    outline = (torch.sqrt(across * across + along * along) - 1.0) * torch.minimum(hl, hw)
    face = torch.maximum(band, outline)
    if not edge:
        return face
    rim = edge_gap(across_m, along_m, distance, c["radius"], c["half_thickness"], hl, hw)
    governs = (face < 0.0) | (outline < 0.0)
    return torch.where(governs, face, torch.minimum(face, rim))


def pair_gap(
    table: Tensor,
    kind: Tensor,
    nlerp: Tensor,
    lift_form: str,
    rho: float,
    z_span: tuple[float, float],
    y: tuple[Tensor, ...],
    x: Tensor,
    start: Tensor,
    h: float,
    edge: bool = True,
) -> Tensor:
    """Each pair's gap (continuous form) at ``x`` into its step on the RK4 path from ``y`` (components)."""

    c = {name: table[:, i] for i, name in enumerate(COLUMNS)}
    px, py, pz = rk4_position(c, lift_form, y, x)
    radius = c["radius"]
    top = pz - radius
    floor = pz + c["floor_offset"] - radius
    ox = px - torch.minimum(torch.maximum(px, c["low_x"]), c["high_x"])
    oy = py - torch.minimum(torch.maximum(py, c["low_y"]), c["high_y"])
    oz = pz - torch.minimum(torch.maximum(pz, c["low_z"]), c["high_z"])
    box = torch.sqrt(ox * ox + oy * oy + oz * oz) - radius
    dx, dy = px - 0.0, py - c["axis_y"]
    radial = torch.sqrt(dx * dx + dy * dy)
    outside = radial > rho
    scale = torch.where(outside, rho / torch.where(outside, radial, torch.ones_like(radial)), torch.ones_like(radial))
    cx = torch.where(outside, 0.0 + dx * scale, px)
    cy = torch.where(outside, c["axis_y"] + dy * scale, py)
    cz = pz.clamp(z_span[0], z_span[1])
    ex, ey, ez = px - cx, py - cy, pz - cz
    cylinder = torch.sqrt(ex * ex + ey * ey + ez * ez) - radius
    blade = _blade_gap(c, nlerp, h, (start + x) / h, px, py, pz, edge)
    gap = torch.where(kind == KIND_TOP, top, box)
    gap = torch.where(kind == KIND_FLOOR, floor, gap)
    gap = torch.where(kind == KIND_CYLINDER, cylinder, gap)
    return torch.where(kind == KIND_BLADE, blade, gap)


def illinois_columns_fn(
    table: Tensor,
    kind: Tensor,
    nlerp: Tensor,
    lift_form: str,
    rho: float,
    z_span: tuple[float, float],
    ys: Tensor,
    start: Tensor,
    used: Tensor,
    h: float,
    lo: Tensor,
    hi: Tensor,
    g_lo: Tensor,
    g_hi: Tensor,
    side: Tensor,
    steps: int,
    tolerance: float,
    edge: bool = True,
) -> tuple[Tensor, ...]:
    """``steps`` Illinois steps (every fourth a bisection) of :func:`lib.physics.vectorized.sim.illinois_fn`, per
    component, then which used brackets are still wider than ``tolerance``; ``edge``: the blades' rounded edge."""

    y = tuple(ys[:, i] for i in range(9))
    for iteration in range(steps):
        width = hi - lo
        bisect = 0.5 * (lo + hi)
        if iteration % 4 != 3:
            guess = hi - g_hi * width / (g_hi - g_lo)
            ok = torch.isfinite(guess) & (guess > lo) & (guess < hi)
            x = torch.where(ok, guess, bisect)
        else:
            x = bisect
        g = pair_gap(table, kind, nlerp, lift_form, rho, z_span, y, x, start, h, edge)
        left = g >= 0.0
        g_hi = torch.where(left & (side > 0.0), 0.5 * g_hi, g_hi)
        g_lo = torch.where(~left & (side < 0.0), 0.5 * g_lo, g_lo)
        done = width <= 0.0
        lo, hi, g_lo, g_hi = (
            torch.where(done, lo, torch.where(left, x, lo)),
            torch.where(done, hi, torch.where(left, hi, x)),
            torch.where(done, g_lo, torch.where(left, g, g_lo)),
            torch.where(done, g_hi, torch.where(left, g_hi, g)),
        )
        side = torch.where(left, torch.ones_like(side), -torch.ones_like(side))
    # Per slot; reduced in a separate call, so that the steps above stay one element-wise kernel each (fused into the
    # reduction they ran on a few blocks, 8x slower).
    open_ = used & ((hi - lo) > tolerance)
    return lo, hi, g_lo, g_hi, side, open_


def converged_fn(open_: Tensor) -> Tensor:
    """Whether no bracket is still open."""

    return ~open_.any()


__all__ = ["COLUMNS", "converged_fn", "illinois_columns_fn", "pair_constants", "pair_gap", "rk4_position"]
