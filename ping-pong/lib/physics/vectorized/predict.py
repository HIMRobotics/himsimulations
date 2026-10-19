"""
Where will the ball be? A batched flight-plus-one-table-bounce predictor for decisions (a policy's forecast, the opponent).

:class:`BouncePredictor` takes each environment's ball state now and ``K`` candidate delays ``tau`` and returns, per
environment, the first contact (it must be a table-top bounce on the receiver's half, ``side``), the ball just after
it, and the ball at ``bounce + tau_k`` for every candidate. It is one fixed-iteration computation, captured as one
CUDA graph (:class:`lib.physics.vectorized.graphs.GraphedFunction`): no host sync, one launch per call.

The physics is the batch's, on the same grid as :class:`lib.physics.vectorized.dense.DenseBallBatch` with no blades:

1. **flight** from now in steps of ``dt`` (RK4, :func:`lib.physics.vectorized.flight.rk4_step`), watching every static
   gap (:func:`lib.physics.vectorized.geometry.static_gaps`, through quantities of the same sign that need no square
   root, :func:`contact_indicators`) for the first one to fire (``>= 0`` to ``< 0``); a row stops at the step its
   first gap fires; ``horizon`` bounds the search;
2. **the bounce**: if the table top fired (the board may fire with it: over the playing surface its gap is the top's)
   and nothing else did, its root on the RK4 path (``root_iterations`` Illinois steps,
   :func:`lib.physics.vectorized.pairs.illinois_columns_fn`); it must be a contact (over the playing surface),
   approaching, on the receiver's half; then the table's impact law;
3. **after the bounce**, steps of ``dt`` from the bounce (the oracle restarts its grid there) up to ``tau_max``,
   keeping each candidate's state at its last whole step, then one partial step to ``bounce + tau_k``; every static
   gap is watched, and a candidate is ``clean`` only if no gap fired before it (a step that both fires and contains
   ``tau_k`` counts as fired: conservative).

Steps 1 and 3 are written one column per quantity (as :mod:`lib.physics.vectorized.pairs`), so a compiled block of
steps is a few element-wise kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.flight import rk4_step
from lib.physics.vectorized.geometry import BOARD, TABLE_TOP, static_contact
from lib.physics.vectorized.graphs import GraphedFunction, compiler
from lib.physics.vectorized.impact import resolve
from lib.physics.vectorized.pairs import _acceleration, illinois_columns_fn, pair_constants
from lib.physics.vectorized.params import VecParams

Tensor = Any

NEVER = 1 << 30
"""``first`` of a row whose gaps never fired."""

REASON_OK, REASON_NO_CONTACT, REASON_NOT_TOP, REASON_MISSED_TOP, REASON_WRONG_HALF = range(5)
"""Why a prediction is not valid: nothing within the horizon; the first contact was not the table top (net, posts,
clamps, board edge or side, floor); the top's plane was crossed off the table (or not approaching); the bounce is on
the other half."""


@dataclass(frozen=True)
class Prediction:
    """
    Per environment ``(N, ...)``; per candidate ``(N, K, ...)``.

    :ivar valid: The first contact is a table-top bounce on the receiver's half within the horizon.
    :ivar reason: ``REASON_*`` code (``REASON_OK`` where valid).
    :ivar bounce_time: Time from now to the bounce, s.
    :ivar bounce_point: Contact point on the table, m.
    :ivar after_bounce: The ball just after the bounce (position, velocity, spin).
    :ivar time: ``bounce_time + tau``, s.
    :ivar state: The ball at ``time``.
    :ivar clean: No other contact between the bounce and ``time`` (and ``tau`` within ``tau_max``).
    """

    valid: Tensor
    reason: Tensor
    bounce_time: Tensor
    bounce_point: Tensor
    after_bounce: Tensor
    time: Tensor
    state: Tensor
    clean: Tensor


def _flight_columns(params: VecParams) -> dict[str, Tensor]:
    return {
        "gravity": params.gravity,
        "radius": params.radius,
        "gain": params.aero_gain,
        "drag0": params.drag_coefficient,
        "drag_slope": params.drag_spin_slope,
        "slope": params.lift_slope,
        "c_max": params.lift_c_max,
        "decay_gain": params.spin_decay_gain,
    }


def rk4_columns(c: dict[str, Tensor], lift_form: str, y: tuple[Tensor, ...], h: Any) -> tuple[Tensor, ...]:
    """:func:`lib.physics.vectorized.flight.rk4_step` per component: the whole state after one step of ``h``."""

    px, py, pz, vx, vy, vz, wx, wy, wz = y
    half = 0.5 * h
    a1 = _acceleration(c, lift_form, vx, vy, vz, wx, wy, wz)
    v2 = (vx + half * a1[0], vy + half * a1[1], vz + half * a1[2])
    w2 = (wx + half * a1[3], wy + half * a1[4], wz + half * a1[5])
    a2 = _acceleration(c, lift_form, *v2, *w2)
    v3 = (vx + half * a2[0], vy + half * a2[1], vz + half * a2[2])
    w3 = (wx + half * a2[3], wy + half * a2[4], wz + half * a2[5])
    a3 = _acceleration(c, lift_form, *v3, *w3)
    v4 = (vx + h * a3[0], vy + h * a3[1], vz + h * a3[2])
    w4 = (wx + h * a3[3], wy + h * a3[4], wz + h * a3[5])
    a4 = _acceleration(c, lift_form, *v4, *w4)
    sixth = h / 6.0
    return (
        px + sixth * (vx + 2.0 * v2[0] + 2.0 * v3[0] + v4[0]),
        py + sixth * (vy + 2.0 * v2[1] + 2.0 * v3[1] + v4[1]),
        pz + sixth * (vz + 2.0 * v2[2] + 2.0 * v3[2] + v4[2]),
        vx + sixth * (a1[0] + 2.0 * a2[0] + 2.0 * a3[0] + a4[0]),
        vy + sixth * (a1[1] + 2.0 * a2[1] + 2.0 * a3[1] + a4[1]),
        vz + sixth * (a1[2] + 2.0 * a2[2] + 2.0 * a3[2] + a4[2]),
        wx + sixth * (a1[3] + 2.0 * a2[3] + 2.0 * a3[3] + a4[3]),
        wy + sixth * (a1[4] + 2.0 * a2[4] + 2.0 * a3[4] + a4[4]),
        wz + sixth * (a1[5] + 2.0 * a2[5] + 2.0 * a3[5] + a4[5]),
    )


def _box(x: Tensor, y: Tensor, z: Tensor, low: tuple[Any, Any, Any], high: tuple[Any, Any, Any]) -> Tensor:
    ox = x - torch.minimum(torch.maximum(x, low[0]), high[0])
    oy = y - torch.minimum(torch.maximum(y, low[1]), high[1])
    oz = z - torch.minimum(torch.maximum(z, low[2]), high[2])
    return torch.sqrt(ox * ox + oy * oy + oz * oz)


def _cylinder(x: Tensor, y: Tensor, z: Tensor, ay: Tensor, rho: float, z0: float, z1: float) -> Tensor:
    dx, dy = x - 0.0, y - ay
    radial = torch.sqrt(dx * dx + dy * dy)
    outside = radial > rho
    scale = torch.where(outside, rho / torch.where(outside, radial, torch.ones_like(radial)), torch.ones_like(radial))
    cx = torch.where(outside, 0.0 + dx * scale, x)
    cy = torch.where(outside, ay + dy * scale, y)
    cz = z.clamp(z0, z1)
    ex, ey, ez = x - cx, y - cy, z - cz
    return torch.sqrt(ex * ex + ey * ey + ez * ez)


def _box2(x: Tensor, y: Tensor, z: Tensor, low: tuple[Any, Any, Any], high: tuple[Any, Any, Any]) -> Tensor:
    ox = x - torch.minimum(torch.maximum(x, low[0]), high[0])
    oy = y - torch.minimum(torch.maximum(y, low[1]), high[1])
    oz = z - torch.minimum(torch.maximum(z, low[2]), high[2])
    return ox * ox + oy * oy + oz * oz


def contact_indicators(params: VecParams, x: Tensor, y: Tensor, z: Tensor) -> tuple[Tensor, ...]:
    """
    Quantities with the signs of the 8 static gaps (:func:`static_gap_columns`): the top's and the floor's gaps, and
    ``distance^2 - R^2`` for the boxes and posts, so a step's fired test needs no square root for a box. A gap is
    ``>= 0`` exactly when its indicator is (up to rounding at a tangency).
    """

    radius = params.radius
    r2 = radius * radius
    zero = torch.zeros_like(x)
    hl, hw, thick = params.table_half_length, params.table_half_width, params.board_thickness
    ht, hs, height = params.net_half_thickness, params.net_half_span, params.net_height
    top = z - radius
    net = _box2(x, y, z, (-ht, -hs, zero), (ht, hs, height)) - r2
    floor = z + params.table_height - radius
    board = _box2(x, y, z, (-hl, -hw, -thick), (hl, hw, zero)) - r2
    over = (z >= 0.0) & (x.abs() <= hl) & (y.abs() <= hw)
    board = torch.where(over & (top >= 0.0), torch.full_like(board, math.inf), board)
    posts_y, rho, (z0, z1) = params.posts
    post_neg = _cylinder(x, y, z, posts_y[:, 0], rho, z0, z1)
    post_pos = _cylinder(x, y, z, posts_y[:, 1], rho, z0, z1)
    cx, (cy0, cy1), (cz0, cz1) = params.clamps
    clamp_neg = _box2(x, y, z, (zero - cx, zero - cy1, zero + cz0), (zero + cx, zero - cy0, zero + cz1)) - r2
    clamp_pos = _box2(x, y, z, (zero - cx, zero + cy0, zero + cz0), (zero + cx, zero + cy1, zero + cz1)) - r2
    return (top, net, floor, board, post_neg * post_neg - r2, post_pos * post_pos - r2, clamp_neg, clamp_pos)


def indicators_fn(params: VecParams, y: Tensor) -> Tensor:
    """:func:`contact_indicators` of states ``(N, 9)``, ``(N, 8)``."""

    return torch.stack(contact_indicators(params, y[:, 0], y[:, 1], y[:, 2]), -1)


def flight_block_fn(
    params: VecParams, y: Tensor, g: Tensor, live: Tensor, pre: Tensor, g_pre: Tensor, g_post: Tensor, first: Tensor, j: Tensor, dt: float, steps: int
) -> tuple[Tensor, ...]:
    """
    ``steps`` steps of the search for each row's first contact (``g``: :func:`contact_indicators`). A row whose
    indicator fires keeps its step's start state (``pre``) and its indicators either side (``g_pre``, ``g_post``; the
    top's is its gap), records the step's number (``first``, 1-based) and stops.
    """

    c = _flight_columns(params)
    state = y.unbind(-1)
    gaps = g.unbind(-1)
    pre_c, g_pre_c, g_post_c = pre.unbind(-1), g_pre.unbind(-1), g_post.unbind(-1)
    for _ in range(steps):
        j = j + 1
        moved = rk4_columns(c, params.lift_form, state, dt)
        after = contact_indicators(params, moved[0], moved[1], moved[2])
        fired = torch.zeros_like(live)
        for before_k, after_k in zip(gaps, after, strict=True):
            fired = fired | ((before_k >= 0.0) & (after_k < 0.0))
        now = live & fired
        pre_c = tuple(torch.where(now, a, b) for a, b in zip(state, pre_c, strict=True))
        g_pre_c = tuple(torch.where(now, a, b) for a, b in zip(gaps, g_pre_c, strict=True))
        g_post_c = tuple(torch.where(now, a, b) for a, b in zip(after, g_post_c, strict=True))
        first = torch.where(now, j, first)
        live = live & ~fired
        state = tuple(torch.where(live, a, b) for a, b in zip(moved, state, strict=True))
        gaps = tuple(torch.where(live, a, b) for a, b in zip(after, gaps, strict=True))
    return torch.stack(state, -1), torch.stack(gaps, -1), live, torch.stack(pre_c, -1), torch.stack(g_pre_c, -1), torch.stack(g_post_c, -1), first, j


def bounce_fn(params: VecParams, pre: Tensor, tau: Tensor, side: Tensor) -> tuple[Tensor, ...]:
    """The top's contact at the root: ``(contact, approaching, on_side, point, y_hit, after)``."""

    n = pre.shape[0]
    rows = torch.arange(n, device=pre.device)
    y_hit = rk4_step(pre, tau, params, rows)
    frame = static_contact(torch.full_like(rows, TABLE_TOP), y_hit[:, :3], params, rows)
    impact = resolve(
        y_hit[:, 3:6],
        y_hit[:, 6:9],
        frame.normal,
        torch.zeros_like(frame.normal),
        frame.material,
        rows,
        params,
        effective_mass=torch.full_like(tau, math.inf),
    )
    on_side = frame.point[:, 0] * side > 0.0
    after = torch.cat([y_hit[:, :3], impact.velocity, impact.spin], -1)
    return frame.valid, impact.approaching, on_side, frame.point, y_hit, after


def after_block_fn(
    params: VecParams, y: Tensor, g: Tensor, event: Tensor, captured: Tensor, whole: Tensor, j: Tensor, dt: float, steps: int
) -> tuple[Tensor, ...]:
    """
    ``steps`` steps after the bounce: each candidate keeps the state after its ``whole`` steps; ``event`` records the
    first step (1-based) in which any static gap fired.
    """

    c = _flight_columns(params)
    k_count = captured.shape[1]
    state = y.unbind(-1)
    gaps = g.unbind(-1)
    caught = [captured[:, k].unbind(-1) for k in range(k_count)]
    marks = whole.unbind(-1)
    for _ in range(steps):
        j = j + 1
        moved = rk4_columns(c, params.lift_form, state, dt)
        after = contact_indicators(params, moved[0], moved[1], moved[2])
        fired = torch.zeros_like(event, dtype=torch.bool)
        for before_k, after_k in zip(gaps, after, strict=True):
            fired = fired | ((before_k >= 0.0) & (after_k < 0.0))
        event = torch.where(fired & (event == NEVER), j, event)
        caught = [tuple(torch.where(marks[k] == j, a, b) for a, b in zip(moved, caught[k], strict=True)) for k in range(k_count)]
        state, gaps = moved, after
    return torch.stack(state, -1), torch.stack(gaps, -1), event, torch.stack([torch.stack(k, -1) for k in caught], 1), j


def final_fn(params: VecParams, captured: Tensor, rest: Tensor) -> Tensor:
    """Each candidate's last partial step, of ``rest`` (0: none), ``(N, K, 9)``."""

    n, k = captured.shape[0], captured.shape[1]
    rows = torch.arange(n, device=captured.device).repeat_interleave(k)
    return rk4_step(captured.reshape(n * k, 9), rest.reshape(n * k), params, rows).reshape(n, k, 9)


class BouncePredictor:
    """
    Flight plus one table bounce for every environment, ``K`` candidate delays each, one graph per call.

    :param params: The environments' physics (``N`` = ``params.n``).
    :param candidates: Delays per environment, ``K``.
    :param dt: The step, s (the ball batch's).
    :param horizon: How far ahead the bounce is searched for, s.
    :param tau_max: The longest delay after the bounce, s.
    :param root_iterations: Illinois steps for the bounce's root (a multiple of 12).
    :param block: Steps per compiled block.
    :param compile: ``torch.compile`` the blocks (default: on CUDA).
    :param graphs: Capture the call as one CUDA graph (default: on CUDA).
    """

    def __init__(
        self,
        params: VecParams,
        *,
        candidates: int = 5,
        dt: float = 1e-3,
        horizon: float = 1.0,
        tau_max: float = 0.7,
        root_iterations: int = 60,
        block: int = 10,
        compile: bool | None = None,
        graphs: bool | None = None,
    ) -> None:
        cuda = str(params.device).startswith("cuda")
        self.params = params
        self.candidates = int(candidates)
        self.dt = float(dt)
        self.search_steps = math.ceil(horizon / dt - 1e-9)
        self.after_steps = math.ceil(tau_max / dt - 1e-9)
        self.tau_max = float(tau_max)
        if root_iterations % 12:
            raise ValueError("root_iterations must be a multiple of 12")
        self.root_iterations = int(root_iterations)
        self.block = int(block)
        wrap = compiler(cuda if compile is None else bool(compile))
        self._flight = wrap(flight_block_fn)
        self._after = wrap(after_block_fn)
        self._bounce = wrap(bounce_fn)
        self._final = wrap(final_fn)
        self._indicators = wrap(indicators_fn)
        self._constants = wrap(pair_constants)
        self._illinois = wrap(illinois_columns_fn)
        n, k = params.n, self.candidates
        examples = (
            torch.zeros(n, 9, dtype=params.dtype, device=params.device),
            torch.zeros(n, k, dtype=params.dtype, device=params.device),
            torch.ones(n, dtype=params.dtype, device=params.device),
        )
        self._call = GraphedFunction(self._predict, examples, graphs=cuda if graphs is None else graphs)

    @property
    def setup_s(self) -> tuple[float, float] | None:
        """The first call's (compile and warm-up, capture) times, s."""

        return self._call.setup_s

    def predict(self, state: Tensor, tau: Tensor, side: Tensor | float = 1.0) -> Prediction:
        """
        :param state: Each environment's ball now, ``(N, 9)``.
        :param tau: Candidate delays after the bounce, ``(N, K)``, s, in ``[0, tau_max]``.
        :param side: The receiver's half, ``+1`` (``x > 0``) or ``-1``, per environment or for all.
        :return: The prediction; its tensors are the graph's buffers (copy them to keep them past the next call).
        """

        params = self.params
        side_t = torch.as_tensor(side, dtype=params.dtype, device=params.device).expand(params.n)
        out = self._call(
            torch.as_tensor(state, dtype=params.dtype, device=params.device), torch.as_tensor(tau, dtype=params.dtype, device=params.device), side_t
        )
        return Prediction(*out)

    def _predict(self, state: Tensor, tau: Tensor, side: Tensor) -> tuple[Tensor, ...]:
        params, dt = self.params, self.dt
        n = state.shape[0]
        device = state.device
        rows = torch.arange(n, device=device)
        # 1. The first contact.
        g = self._indicators(params, state)
        live = torch.ones(n, dtype=torch.bool, device=device)
        pre = state.clone()
        g_pre = g.clone()
        g_post = g.clone()
        first = torch.full((n,), NEVER, dtype=torch.long, device=device)
        j = torch.zeros((), dtype=torch.long, device=device)
        y = state
        done = 0
        while done < self.search_steps:
            steps = min(self.block, self.search_steps - done)
            y, g, live, pre, g_pre, g_post, first, j = self._flight(params, y, g, live, pre, g_pre, g_post, first, j, dt, steps)
            done += steps
        fired = (g_pre >= 0.0) & (g_post < 0.0)
        top = fired[:, TABLE_TOP]
        others = fired.clone()
        others[:, TABLE_TOP] = False
        others[:, BOARD] = False
        found = first < NEVER
        # 2. The bounce: the top's root, its contact and the table's law.
        zeros = torch.zeros(n, dtype=torch.long, device=device)
        table, kind, nlerp = self._constants(params, pre, rows, zeros, None)
        lo = torch.zeros(n, dtype=params.dtype, device=device)
        g_lo = g_pre[:, TABLE_TOP]
        hi = torch.where(g_lo < 0.0, lo, torch.full_like(lo, dt))
        g_lo = torch.where(torch.isfinite(g_lo) & (g_lo >= 0.0), g_lo, torch.ones_like(g_lo))
        g_hi = g_post[:, TABLE_TOP]
        side_mark = torch.zeros_like(lo)
        start = torch.zeros_like(lo)
        _posts, rho, z_span = params.posts
        for _ in range(self.root_iterations // 12):
            lo, hi, g_lo, g_hi, side_mark, _open = self._illinois(
                table,
                kind,
                nlerp,
                params.lift_form,
                float(rho),
                (float(z_span[0]), float(z_span[1])),
                pre,
                start,
                top,
                dt,
                lo,
                hi,
                g_lo,
                g_hi,
                side_mark,
                12,
                0.0,
                False,  # no blades are watched: the face-only blade compiles as before
            )
        tau_bounce = lo
        contact, approaching, on_side, point, _y_hit, after = self._bounce(params, pre, tau_bounce, side)
        reason = torch.full((n,), REASON_OK, dtype=torch.long, device=device)
        reason = torch.where(on_side, reason, torch.full_like(reason, REASON_WRONG_HALF))
        reason = torch.where(contact & approaching, reason, torch.full_like(reason, REASON_MISSED_TOP))
        reason = torch.where(top & ~others.any(1), reason, torch.full_like(reason, REASON_NOT_TOP))
        reason = torch.where(found, reason, torch.full_like(reason, REASON_NO_CONTACT))
        valid = reason == REASON_OK
        bounce_time = (first - 1).to(params.dtype) * dt + tau_bounce
        # 3. After the bounce.
        k = tau.shape[1]
        whole = torch.floor(tau / dt).long().clamp(0, self.after_steps)
        rest = tau - whole.to(params.dtype) * dt
        captured = after[:, None, :].expand(n, k, 9).clone()
        g = self._indicators(params, after)
        event = torch.full((n,), NEVER, dtype=torch.long, device=device)
        j = torch.zeros((), dtype=torch.long, device=device)
        y = after
        done = 0
        while done < self.after_steps:
            steps = min(self.block, self.after_steps - done)
            y, g, event, captured, j = self._after(params, y, g, event, captured, whole, j, dt, steps)
            done += steps
        state_k = self._final(params, captured, rest)
        clean = valid[:, None] & (tau <= self.tau_max) & ((event[:, None] > whole + 1) | ((rest == 0.0) & (event[:, None] > whole)))
        return valid, reason, bounce_time, point, after, bounce_time[:, None] + tau, state_k, clean


__all__ = [
    "NEVER",
    "REASON_MISSED_TOP",
    "REASON_NOT_TOP",
    "REASON_NO_CONTACT",
    "REASON_OK",
    "REASON_WRONG_HALF",
    "BouncePredictor",
    "Prediction",
    "after_block_fn",
    "bounce_fn",
    "contact_indicators",
    "final_fn",
    "flight_block_fn",
    "indicators_fn",
    "rk4_columns",
]
