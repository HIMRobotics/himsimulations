"""
The batched step without host round trips: :class:`DenseBallBatch`, the same physics as
:class:`lib.physics.vectorized.sim.BallBatch`, laid out for CUDA graphs.

``BallBatch.step`` compacts the live rows and the fired gaps with ``nonzero`` and checks convergence on the host, so
every step pays dozens of host syncs and hundreds of kernel launches; on a server CPU that costs 2.5 ms per quiet step
and 14 to 29 ms per step with a contact, whatever the batch size. Here every array keeps a fixed shape and lives in
persistent buffers, so each phase of a step is captured once as a CUDA graph (after ``torch.compile`` has fused it) and
replayed:

1. **prologue** (once per step): the blades' derived motion, every gap at the step's start;
2. **detect** (once per round): the RK4 step of every row to the step's end, every gap there, the fired gaps (from
   ``>= 0`` to ``< 0``), and their ranks (a running count, so the fired pairs pack into slots without ``nonzero``);
   rows with nothing fired commit their end state;
3. **roots** (once per ``pair_capacity`` fired pairs): each fired pair's root on the RK4 path, in blocks of
   ``root_block`` Illinois steps until every bracket is within ``root_tolerance`` (``BallBatch``'s test, read once per
   block) or ``root_iterations`` ran; then its contact and impact, and each row's earliest valid contact so far (ties
   to the lower surface);
4. **apply** (once per round with a contact): the chosen contact of each row, logged; terminal contacts and slow
   rebounds stop the ball; the others carry on from the touch, and the next round starts there.

The host reads one small tensor per round (how many pairs fired; how full the event log is) and replays the next
phase. The rules are ``BallBatch``'s: the oracle's arming, roots, contact validity, ordering, terminal kinds, rest and
overflow. The one difference is in the root's last digits: the convergence test runs every ``root_block`` steps
rather than every 4, so a root may be refined a few steps further (all within ``root_tolerance``).

Not supported here: ``graze``. Without CUDA (or with ``graphs=False``) the phases run as plain functions, which is
how the CPU tests check this class against the oracle.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Collection, Sequence
from functools import partial
from typing import Any

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.flight import rk4_step
from lib.physics.vectorized.geometry import STATIC_COUNT, BladeTrack, _quat_product, _rotvec, blade_gap, blade_state, quat_axes, static_gaps
from lib.physics.vectorized.pairs import COLUMNS, converged_fn, illinois_columns_fn, pair_constants
from lib.physics.vectorized.params import VecParams
from lib.physics.vectorized.sim import ENDED_OVERFLOW, ENDED_REST, ENDED_TERMINAL, KIND_OF_STATIC, EventLog, contact_fn, gaps_fn

Tensor = Any

_FAR = 1 << 30


def blade_motion(p0: Tensor, v0: Tensor, q0: Tensor, p1: Tensor, v1: Tensor, q1: Tensor, h: float) -> tuple[Tensor, ...]:
    """:attr:`BladeTrack.tensors` of :meth:`BladeTrack.from_states` (the same operations, inside a graph)."""

    dot = (q0 * q1).sum(-1)
    flip = dot < 0.0
    q1s = torch.where(flip[..., None], -q1, q1)
    dot = torch.where(flip, -dot, dot)
    nlerp = dot > 0.9995
    theta = torch.acos(dot.clamp(max=1.0))
    sign = torch.ones_like(q0)
    sign[..., 1:] = -1.0
    omega = _rotvec(_quat_product(q1, q0 * sign)) / h
    return (p0, v0, q0, p1, v1, q1s, theta, nlerp, omega)


def prologue_fn(params: VecParams, state: Tensor, static_cache: Tensor, active: Tensor, raw: tuple[Tensor, ...] | None, h: float) -> tuple[Tensor, ...]:
    """The step's start: blades' motion, ``y0``, ``start``, every gap ``g0`` and ``pending``."""

    n = state.shape[0]
    env = torch.arange(n, device=state.device)
    start = torch.zeros(n, dtype=state.dtype, device=state.device)
    if raw is None:
        return (state.clone(), start, static_cache.clone(), active.clone())
    p0, v0, q0, p1, v1, q1 = raw
    blades = blade_motion(p0, v0, q0, p1, v1, q1, h)
    g0 = gaps_fn(params, state, env, start, blades, h, False)
    return (state.clone(), start, g0, active.clone(), *blades)


def detect_fn(
    params: VecParams,
    y0: Tensor,
    start: Tensor,
    g0: Tensor,
    pending: Tensor,
    state: Tensor,
    static_cache: Tensor,
    active: Tensor,
    ended: Tensor,
    blades: tuple[Tensor, ...] | None,
    h: float,
    arm_tolerance: float,
    last: bool,
) -> tuple[Tensor, ...]:
    """One round's RK4 step to the step's end and its fired gaps; calm rows commit."""

    n = y0.shape[0]
    env = torch.arange(n, device=y0.device)
    step = h - start
    y1 = rk4_step(y0, step, params, env)
    g1 = gaps_fn(params, y1, env, torch.ones_like(step), blades, h, False)
    fired = pending[:, None] & (g0 >= -arm_tolerance) & (g1 < 0.0)
    hit = fired.any(1)
    calm = pending & ~hit
    state = torch.where(calm[:, None], y1, state)
    static_cache = torch.where(calm[:, None], g1[:, :STATIC_COUNT], static_cache)
    if last:
        state = torch.where(hit[:, None], y1, state)
        active = active & ~hit
        ended = torch.where(hit, torch.full_like(ended, ENDED_OVERFLOW), ended)
        fired = torch.zeros_like(fired)
        hit = torch.zeros_like(hit)
    return (y1, g1, hit, fired.reshape(-1), state, static_cache, active, ended)


def pair_gap_one(params: VecParams, ys: Tensor, x: Tensor, row: Tensor, start: Tensor, surface: Tensor, blades: tuple[Tensor, ...] | None, h: float) -> Tensor:
    """
    Each pair's gap (continuous form) at ``x`` into its step on the RK4 path from ``ys``: the value of
    :func:`lib.physics.vectorized.sim.pair_gap_fn`, with only the pair's own blade evaluated, so that every array
    keeps the pairs' length and a compiler fuses a whole Illinois step into a few kernels.
    """

    y = rk4_step(ys, x, params, row)
    static = static_gaps(y[:, :3], params, row, continuous=True).gather(1, surface.clamp(max=STATIC_COUNT - 1)[:, None])[:, 0]
    if blades is None:
        return static
    blade = (surface - STATIC_COUNT).clamp(min=0)
    position, orientation, _velocity, _omega = blade_state(blades, h, (start + x) / h, row, blade)
    on_blade = blade_gap(y[:, :3], position, quat_axes(orientation), params, row)
    return torch.where(surface >= STATIC_COUNT, on_blade, static)


def illinois_block_fn(
    params: VecParams,
    ys: Tensor,
    row: Tensor,
    start: Tensor,
    surface: Tensor,
    used: Tensor,
    blades: tuple[Tensor, ...] | None,
    h: float,
    lo: Tensor,
    hi: Tensor,
    g_lo: Tensor,
    g_hi: Tensor,
    side: Tensor,
    steps: int,
    tolerance: float,
) -> tuple[Tensor, ...]:
    """
    ``steps`` (a multiple of 4) Illinois steps of :func:`lib.physics.vectorized.sim.illinois_fn` (every fourth a
    bisection), then whether every used bracket is within ``tolerance`` (``BallBatch``'s exit test).
    """

    for iteration in range(steps):
        width = hi - lo
        bisect = 0.5 * (lo + hi)
        if iteration % 4 != 3:
            guess = hi - g_hi * width / (g_hi - g_lo)
            ok = torch.isfinite(guess) & (guess > lo) & (guess < hi)
            x = torch.where(ok, guess, bisect)
        else:
            x = bisect
        g = pair_gap_one(params, ys, x, row, start, surface, blades, h)
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
    converged = (~used | ((hi - lo) <= tolerance)).all()
    return lo, hi, g_lo, g_hi, side, converged


def roots_prep_fn(g0: Tensor, g1: Tensor, start: Tensor, flat: Tensor, rank: Tensor, base: Tensor, h: float, capacity: int) -> tuple[Tensor, ...]:
    """The fired pairs ranked ``[base, base + capacity)`` in slots, and each one's starting bracket
    (``BallBatch._root``): ``(used, row, surface, lo, hi, g_lo, g_hi, side)``."""

    n, s = g1.shape
    device = g1.device
    picked = flat & (rank >= base) & (rank < base + capacity)
    slot = torch.where(picked, rank - base, torch.full_like(rank, capacity)).long()
    index = torch.arange(n * s, device=device)
    pair = torch.full((capacity + 1,), -1, dtype=torch.long, device=device).scatter_reduce(
        0, slot, torch.where(picked, index, torch.full_like(index, -1)), reduce="amax"
    )[:capacity]
    used = pair >= 0
    pair = pair.clamp(min=0)
    row = pair // s
    surface = pair % s
    width = h - start[row]
    g_lo = g0[row, surface]
    g_hi = g1[row, surface]
    lo = torch.zeros_like(width)
    hi = torch.where(g_lo < 0.0, torch.zeros_like(width), width)
    g_lo = torch.where(torch.isfinite(g_lo) & (g_lo >= 0.0), g_lo, torch.ones_like(g_lo))
    return used, row, surface, lo, hi, g_lo, g_hi, torch.zeros_like(width)


def roots_slots_fn(y0: Tensor, start: Tensor, row: Tensor) -> tuple[Tensor, Tensor]:
    """The slots' round-start states and times."""

    return y0[row], start[row]


def roots_finish_fn(
    params: VecParams,
    y0: Tensor,
    start: Tensor,
    used: Tensor,
    row: Tensor,
    surface: Tensor,
    tau: Tensor,
    best: tuple[Tensor, ...],
    blades: tuple[Tensor, ...] | None,
    h: float,
    compliance: tuple[Tensor, ...] | None = None,
) -> tuple[Tensor, ...]:
    """
    The slots' contacts and impacts at their roots, merged into each row's earliest valid contact so far, ``best`` =
    ``(when, surface, y_hit, velocity, spin, normal, point, region, separation)``; ``compliance`` as
    :func:`lib.physics.vectorized.sim.contact_fn`.
    """

    n = y0.shape[0]
    capacity = row.shape[0]
    device = y0.device
    ys, ts = y0[row], start[row]
    y_hit = rk4_step(ys, tau, params, row)
    when = ts + tau
    valid, normal, point, region, _material, velocity, spin, separation = contact_fn(params, y_hit, row, surface, when, blades, h, compliance)
    valid = valid & used
    # This batch's earliest valid contact of each row, ties to the lower surface.
    target = torch.where(used, row, torch.full_like(row, n))
    key = torch.where(valid, when, torch.full_like(when, math.inf))
    first = torch.full((n + 1,), math.inf, dtype=when.dtype, device=device).scatter_reduce(0, target, key, reduce="amin")
    at_first = valid & (key == first[target])
    surface_key = torch.where(at_first, surface, torch.full_like(surface, _FAR))
    lowest = torch.full((n + 1,), _FAR, dtype=surface.dtype, device=device).scatter_reduce(0, target, surface_key, reduce="amin")
    chosen = at_first & (surface == lowest[target])
    slots = torch.arange(capacity, device=device)
    of_row = torch.full((n + 1,), -1, dtype=torch.long, device=device).scatter_reduce(
        0, torch.where(chosen, target, torch.full_like(target, n)), torch.where(chosen, slots, torch.full_like(slots, -1)), reduce="amax"
    )[:n]
    has = of_row >= 0
    k = of_row.clamp(min=0)
    b_when, b_surface, b_y, b_v, b_w, b_normal, b_point, b_region, b_sep = best
    c_when = torch.where(has, when[k], torch.full_like(b_when, math.inf))
    c_surface = surface[k]
    better = has & ((c_when < b_when) | ((c_when == b_when) & (c_surface < b_surface)))
    return (
        torch.where(better, c_when, b_when),
        torch.where(better, c_surface, b_surface),
        torch.where(better[:, None], y_hit[k], b_y),
        torch.where(better[:, None], velocity[k], b_v),
        torch.where(better[:, None], spin[k], b_w),
        torch.where(better[:, None], normal[k], b_normal),
        torch.where(better[:, None], point[k], b_point),
        torch.where(better, region[k], b_region),
        torch.where(better, separation[k], b_sep),
    )


def apply_fn(
    params: VecParams,
    y1: Tensor,
    g1: Tensor,
    hit: Tensor,
    best: tuple[Tensor, ...],
    state: Tensor,
    static_cache: Tensor,
    active: Tensor,
    ended: Tensor,
    events: Tensor,
    terminal_static: Tensor,
    blades: tuple[Tensor, ...] | None,
    h: float,
    rest_speed: float,
) -> tuple[Tensor, ...]:
    """Each row's chosen contact: the after-state, stops, and the next round's start for the rest."""

    n = y1.shape[0]
    env = torch.arange(n, device=y1.device)
    when, surface, y_hit, velocity, spin, normal, _point, _region, separation = best
    has = hit & torch.isfinite(when)
    quiet = hit & ~has
    state = torch.where(quiet[:, None], y1, state)
    static_cache = torch.where(quiet[:, None], g1[:, :STATIC_COUNT], static_cache)
    after = torch.cat([y_hit[:, :3], velocity, spin], -1)
    static = surface < STATIC_COUNT
    terminal = static & terminal_static[surface.clamp(0, STATIC_COUNT - 1)]
    rest = static & (separation < rest_speed) & (normal[:, 2] > 0.0)
    stop = has & (terminal | rest)
    state = torch.where(stop[:, None], after, state)
    static_cache = torch.where(stop[:, None], static_gaps(after[:, :3], params, env), static_cache)
    active = active & ~stop
    ended = torch.where(stop, torch.where(terminal, torch.full_like(ended, ENDED_TERMINAL), torch.full_like(ended, ENDED_REST)), ended)
    carry = has & ~stop
    start = torch.where(carry, when, torch.zeros_like(when))
    g0 = gaps_fn(params, after, env, start / h, blades, h, False)
    return (after, start, g0, carry, has, state, static_cache, active, ended, events + has.sum())


def log_fn(has: Tensor, t_step: Tensor, best: tuple[Tensor, ...], after: Tensor, count: Tensor, columns: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Write each row's contact (if any) at its next log slot."""

    when, surface, y_hit, _v, _w, normal, point, region, _sep = best
    capacity = columns[0].shape[1]
    at = count.clamp(max=capacity - 1)[:, None]
    values = (t_step + when.double(), surface, region, point, normal, y_hit, after)
    out = []
    for column, value in zip(columns, values, strict=True):
        shape = (-1, 1, *column.shape[2:])
        index = at.reshape(-1, 1, *([1] * (column.dim() - 2))).expand(shape[0], 1, *column.shape[2:])
        old = column.gather(1, index)
        new = value.to(column.dtype).reshape(shape)
        mask = has.reshape(-1, 1, *([1] * (column.dim() - 2)))
        out.append(column.scatter(1, index, torch.where(mask, new, old)))
    return (*out, count + (has & (count < capacity)).long(), (has & (count >= capacity)).sum())


class _Phase:
    """One phase: a pure function compiled once, wrapped to write its results into buffers, then graph-captured."""

    def __init__(self, owner: DenseBallBatch, name: str, body: Callable[[], None]) -> None:
        self.owner, self.name, self.body = owner, name, body
        self.graph: Any = None

    def __call__(self) -> None:
        owner = self.owner
        if not owner.graphs:
            self.body()
            return
        if self.graph is None:
            began = time.perf_counter()
            saved = owner._snapshot()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.body()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            warmed = time.perf_counter()
            owner._restore(saved)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.body()
            owner._restore(saved)
            torch.cuda.synchronize()
            owner.setup_s[self.name] = (round(warmed - began, 3), round(time.perf_counter() - warmed, 3))
        self.graph.replay()


class DenseBallBatch:
    """
    :class:`lib.physics.vectorized.sim.BallBatch` as CUDA graphs: one live ball per environment, a fixed step.

    :param params: The environments' physics.
    :param state: Initial states ``(N, 9)``.
    :param dt: The step, s (captured into the graphs; fixed).
    :param blades: Blades per environment (0: none).
    :param t0: Start time, s.
    :param terminal_kinds: Contact kinds that stop a ball.
    :param rest_speed: A static-surface rebound slower than this from a surface gravity presses the ball onto stops it.
    :param root_iterations: Illinois steps per fired gap at most (``BallBatch``'s).
    :param root_tolerance: Root bracket, s: the steps stop, in blocks, once every bracket of the batch is this narrow.
    :param root_block: Illinois steps per replay between those checks (a multiple of 4).
    :param max_events_per_step: Contacts one ball may have in one step before it is stopped as ``overflow``.
    :param arm_tolerance: As ``BallBatch``.
    :param pair_capacity: Fired gaps rooted per graph replay (more take more replays). Default: two batch sizes,
        ``max(64, N // 64)`` (a rally fires about one gap per 250 balls per 1 ms step) and ``max(256, N // 4)`` (a burst:
        synchronised strokes or resets put thousands of contacts in one step); each round takes the smallest that
        holds its pairs.
    :param log: Keep an :class:`EventLog`.
    :param log_capacity: Contacts each environment's on-device log holds before the host empties it.
    :param compile: ``torch.compile`` each phase (default: on CUDA).
    :param graphs: Capture each phase as a CUDA graph (default: on CUDA).
    :param log_history: Move the on-device log into :attr:`log` whenever it fills. ``False``: the caller reads it
        per step (:meth:`log_view`) and empties it (:meth:`clear_log`); rows past ``log_capacity`` are counted as lost.
    :param compliance: Blades may carry an articulated contact compliance (:meth:`set_compliance`).
    :param dynamic: Compile size-generic kernels, so batches of any size share one compilation (the shapes are fixed
        anyway once a phase is captured).
    """

    def __init__(
        self,
        params: VecParams,
        state: Tensor,
        *,
        dt: float,
        blades: int = 0,
        t0: float = 0.0,
        terminal_kinds: Collection[str] = ("floor",),
        rest_speed: float = 0.05,
        root_iterations: int = 60,
        root_tolerance: float = 1e-10,
        root_block: int = 12,
        max_events_per_step: int = 4,
        arm_tolerance: float = 0.0,
        pair_capacity: int | Sequence[int] | None = None,
        log: bool = True,
        log_capacity: int = 32,
        log_history: bool = True,
        compliance: bool = False,
        compile: bool | None = None,
        graphs: bool | None = None,
        dynamic: bool = True,
    ) -> None:
        cuda = str(params.device).startswith("cuda")
        self.params = params
        self.dt = float(dt)
        self.blades = int(blades)
        self.t = float(t0)
        self.rest_speed = float(rest_speed)
        self.root_iterations = int(root_iterations)
        self.root_tolerance = float(root_tolerance)
        if root_block % 4 or root_block <= 0:
            raise ValueError("root_block must be a positive multiple of 4 (every fourth Illinois step bisects)")
        self.root_block = int(root_block)
        self.max_events_per_step = int(max_events_per_step)
        self.arm_tolerance = float(arm_tolerance)
        self.graphs = cuda if graphs is None else bool(graphs)
        use_compile = cuda if compile is None else bool(compile)
        n, s = params.n, STATIC_COUNT + self.blades
        if isinstance(pair_capacity, int):
            tiers = [pair_capacity]
        elif pair_capacity is not None:
            tiers = [int(t) for t in pair_capacity]
        else:
            tiers = [max(64, n // 64), max(256, n // 4)]
        self.tiers = sorted({min(t, n * s) for t in tiers})
        self.capacity = self.tiers[-1]
        dtype, device = params.dtype, params.device
        self._log = EventLog() if log else None
        self.logging = log
        self.log_history = bool(log_history)
        self.compliance = bool(compliance) and blades > 0
        self.events_total = 0
        self.log_lost = 0
        self.setup_s: dict[str, tuple[float, float]] = {}
        """Each phase's first use: (compile and warm-up, capture), s."""

        def zeros(*shape: int, kind: Any = None) -> Tensor:
            return torch.zeros(shape, dtype=kind if kind is not None else dtype, device=device)

        b = self.blades
        self._buf: dict[str, Tensor] = {
            "state": torch.as_tensor(state, dtype=dtype, device=device).reshape(n, 9).clone(),
            "active": torch.ones(n, dtype=torch.bool, device=device),
            "ended": zeros(n, kind=torch.long),
            "static_cache": zeros(n, STATIC_COUNT),
            "events": zeros(kind=torch.long),
            "t_now": torch.full((), float(t0), dtype=torch.float64, device=device),
            "t_step": torch.zeros((), dtype=torch.float64, device=device),
            "y0": zeros(n, 9),
            "start": zeros(n),
            "g0": zeros(n, s),
            "pending": torch.zeros(n, dtype=torch.bool, device=device),
            "y1": zeros(n, 9),
            "g1": zeros(n, s),
            "hit": torch.zeros(n, dtype=torch.bool, device=device),
            "flat": torch.zeros(n * s, dtype=torch.bool, device=device),
            "rank": torch.zeros(n * s, dtype=torch.int32, device=device),
            "count": zeros(kind=torch.long),
            "base": zeros(kind=torch.int32),
            "after": zeros(n, 9),
            "has": torch.zeros(n, dtype=torch.bool, device=device),
            "best_when": zeros(n),
            "best_surface": zeros(n, kind=torch.long),
            "best_y": zeros(n, 9),
            "best_v": zeros(n, 3),
            "best_w": zeros(n, 3),
            "best_normal": zeros(n, 3),
            "best_point": zeros(n, 3),
            "best_region": zeros(n, kind=torch.long),
            "best_sep": zeros(n),
            "report": zeros(3, kind=torch.long),
        }

        for tier, capacity in enumerate(self.tiers):
            slots = {
                "used": torch.zeros(capacity, dtype=torch.bool, device=device),
                "row": zeros(capacity, kind=torch.long),
                "surface": zeros(capacity, kind=torch.long),
                "lo": zeros(capacity),
                "hi": zeros(capacity),
                "g_lo": zeros(capacity),
                "g_hi": zeros(capacity),
                "side": zeros(capacity),
                "y": zeros(capacity, 9),
                "t": zeros(capacity),
                "table": zeros(capacity, len(COLUMNS)),
                "kind": zeros(capacity, kind=torch.long),
                "nlerp": torch.zeros(capacity, dtype=torch.bool, device=device),
            }
            self._buf.update({f"slot{tier}_{name}": value for name, value in slots.items()})
        if b:
            for name, width in (("p0", 3), ("v0", 3), ("q0", 4), ("p1", 3), ("v1", 3), ("q1", 4)):
                self._buf[f"raw_{name}"] = zeros(n, b, width)
            self._buf["raw_q0"][..., 0] = 1.0
            self._buf["raw_q1"][..., 0] = 1.0
            for name, shape in (("p0", (3,)), ("v0", (3,)), ("q0", (4,)), ("p1", (3,)), ("v1", (3,)), ("q1", (4,)), ("theta", ()), ("omega", (3,))):
                self._buf[f"blade_{name}"] = zeros(n, b, *shape)
            self._buf["blade_nlerp"] = torch.zeros(n, b, dtype=torch.bool, device=device)
            if self.compliance:
                for name in ("comp_a", "comp_b", "comp_c"):
                    self._buf[name] = zeros(n, b, 3, 3)
                self._buf["comp_on"] = torch.zeros(n, b, dtype=torch.bool, device=device)
        if log:
            self._buf.update(
                log_time=torch.zeros(n, log_capacity, dtype=torch.float64, device=device),
                log_surface=zeros(n, log_capacity, kind=torch.long),
                log_region=zeros(n, log_capacity, kind=torch.long),
                log_point=zeros(n, log_capacity, 3),
                log_normal=zeros(n, log_capacity, 3),
                log_before=zeros(n, log_capacity, 9),
                log_after=zeros(n, log_capacity, 9),
                log_count=zeros(n, kind=torch.long),
                log_lost=zeros(kind=torch.long),
            )
        self._buf["static_cache"].copy_(static_gaps(self._buf["state"][:, :3], params))
        self._terminal_static = torch.tensor([kind in frozenset(terminal_kinds) for kind in KIND_OF_STATIC], device=device)
        wrap = self._compiler(use_compile, dynamic)
        self._prologue_fn = wrap(prologue_fn)
        self._detect_fn = wrap(detect_fn)
        self._roots_prep_fn = wrap(roots_prep_fn)
        self._roots_slots_fn = wrap(roots_slots_fn)
        self._pair_constants_fn = wrap(pair_constants)
        self._illinois_fn = wrap(illinois_columns_fn)
        self._converged_fn = wrap(converged_fn)
        self._roots_finish_fn = wrap(roots_finish_fn)
        self._apply_fn = wrap(apply_fn)
        self._log_fn = wrap(log_fn)
        self._prologue = _Phase(self, "prologue", self._run_prologue)
        self._detect = _Phase(self, "detect", lambda: self._run_detect(False))
        self._detect_last = _Phase(self, "detect_last", lambda: self._run_detect(True))
        self._roots = [
            (
                _Phase(self, f"roots{suffix}", partial(self._run_roots_start, tier)),
                _Phase(self, f"roots_more{suffix}", partial(self._run_roots_block, tier)),
                _Phase(self, f"roots_finish{suffix}", partial(self._run_roots_finish, tier)),
            )
            for tier in range(len(self.tiers))
            for suffix in ["" if tier == 0 else f"_{self.tiers[tier]}"]
        ]
        self._apply = _Phase(self, "apply", self._run_apply)

    # ---- plumbing -------------------------------------------------------------------------------------------------

    @staticmethod
    def _compiler(enabled: bool, dynamic: bool) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if not enabled:
            return lambda fn: fn

        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            compiled = torch.compile(fn, dynamic=dynamic, fullgraph=True)
            state = {"eager": False}

            def call(*args: Any) -> Any:
                if state["eager"]:
                    return fn(*args)
                try:
                    return compiled(*args)
                except Exception as error:
                    from torch._dynamo.exc import BackendCompilerFailed  # pyright: ignore[reportMissingImports]
                    from torch._inductor.exc import InductorError  # pyright: ignore[reportMissingImports]

                    if not isinstance(error, (BackendCompilerFailed, InductorError)):
                        raise
                    state["eager"] = True
                    warnings.warn(f"torch.compile failed for {fn.__name__}; running it eagerly: {error}", RuntimeWarning, stacklevel=2)
                    return fn(*args)

            return call

        return wrap

    def _snapshot(self) -> dict[str, Tensor]:
        return {name: value.clone() for name, value in self._buf.items()}

    def _restore(self, saved: dict[str, Tensor]) -> None:
        for name, value in saved.items():
            self._buf[name].copy_(value)

    def _write(self, names: Sequence[str], values: Sequence[Tensor]) -> None:
        for name, value in zip(names, values, strict=True):
            self._buf[name].copy_(value)

    def _blades(self) -> tuple[Tensor, ...] | None:
        if not self.blades:
            return None
        b = self._buf
        return (b["blade_p0"], b["blade_v0"], b["blade_q0"], b["blade_p1"], b["blade_v1"], b["blade_q1"], b["blade_theta"], b["blade_nlerp"], b["blade_omega"])

    def _best(self) -> tuple[Tensor, ...]:
        b = self._buf
        return (b["best_when"], b["best_surface"], b["best_y"], b["best_v"], b["best_w"], b["best_normal"], b["best_point"], b["best_region"], b["best_sep"])

    _BEST = ("best_when", "best_surface", "best_y", "best_v", "best_w", "best_normal", "best_point", "best_region", "best_sep")
    _LOG = ("log_time", "log_surface", "log_region", "log_point", "log_normal", "log_before", "log_after")

    def _run_prologue(self) -> None:
        b = self._buf
        raw = tuple(b[f"raw_{k}"] for k in ("p0", "v0", "q0", "p1", "v1", "q1")) if self.blades else None
        out = self._prologue_fn(self.params, b["state"], b["static_cache"], b["active"], raw, self.dt)
        names = ["y0", "start", "g0", "pending"]
        if self.blades:
            names += [f"blade_{k}" for k in ("p0", "v0", "q0", "p1", "v1", "q1", "theta", "nlerp", "omega")]
        self._write(names, out)
        b["t_step"].copy_(b["t_now"])
        b["t_now"].add_(self.dt)

    def _run_detect(self, last: bool) -> None:
        b = self._buf
        out = self._detect_fn(
            self.params,
            b["y0"],
            b["start"],
            b["g0"],
            b["pending"],
            b["state"],
            b["static_cache"],
            b["active"],
            b["ended"],
            self._blades(),
            self.dt,
            self.arm_tolerance,
            last,
        )
        self._write(("y1", "g1", "hit", "flat", "state", "static_cache", "active", "ended"), out)
        # The fired pairs' ranks: the library scan (a compiled one ran on a few blocks, 0.16 ms at 82k pairs).
        torch.cumsum(b["flat"], 0, dtype=torch.int32, out=b["rank"])
        b["count"].copy_(b["rank"][-1])
        b["rank"].sub_(1)
        b["best_when"].fill_(math.inf)
        b["best_surface"].fill_(_FAR)
        b["report"][0].copy_(b["count"])
        if self.logging:
            b["report"][1].copy_(b["log_count"].max())

    _SLOTS = ("used", "row", "surface", "lo", "hi", "g_lo", "g_hi", "side")

    def _slot(self, tier: int, name: str) -> Tensor:
        return self._buf[f"slot{tier}_{name}"]

    def _run_roots_start(self, tier: int) -> None:
        # The fired pairs of this batch into slots, their brackets, and the first block of Illinois steps.
        b = self._buf
        capacity = self.tiers[tier]
        out = self._roots_prep_fn(b["g0"], b["g1"], b["start"], b["flat"], b["rank"], b["base"], self.dt, capacity)
        self._write([f"slot{tier}_{name}" for name in self._SLOTS], out)
        self._write((f"slot{tier}_y", f"slot{tier}_t"), self._roots_slots_fn(b["y0"], b["start"], self._slot(tier, "row")))
        self._write(
            (f"slot{tier}_table", f"slot{tier}_kind", f"slot{tier}_nlerp"),
            self._pair_constants_fn(self.params, self._slot(tier, "y"), self._slot(tier, "row"), self._slot(tier, "surface"), self._blades()),
        )
        self._run_roots_block(tier)

    def _run_roots_block(self, tier: int) -> None:
        b = self._buf
        _posts_y, rho, z_span = self.params.posts
        out = self._illinois_fn(
            self._slot(tier, "table"),
            self._slot(tier, "kind"),
            self._slot(tier, "nlerp"),
            self.params.lift_form,
            float(rho),
            (float(z_span[0]), float(z_span[1])),
            self._slot(tier, "y"),
            self._slot(tier, "t"),
            self._slot(tier, "used"),
            self.dt,
            self._slot(tier, "lo"),
            self._slot(tier, "hi"),
            self._slot(tier, "g_lo"),
            self._slot(tier, "g_hi"),
            self._slot(tier, "side"),
            self.root_block,
            self.root_tolerance,
            self.params.blade_edge,
        )
        self._write([f"slot{tier}_{name}" for name in ("lo", "hi", "g_lo", "g_hi", "side")], out[:5])
        b["report"][2].copy_(self._converged_fn(out[5]))

    def _run_roots_finish(self, tier: int) -> None:
        b = self._buf
        compliance = (b["comp_a"], b["comp_b"], b["comp_c"], b["comp_on"]) if self.compliance else None
        out = self._roots_finish_fn(
            self.params,
            b["y0"],
            b["start"],
            self._slot(tier, "used"),
            self._slot(tier, "row"),
            self._slot(tier, "surface"),
            self._slot(tier, "lo"),
            self._best(),
            self._blades(),
            self.dt,
            compliance,
        )
        self._write(self._BEST, out)
        b["base"].add_(self.tiers[tier])

    def _run_apply(self) -> None:
        b = self._buf
        out = self._apply_fn(
            self.params,
            b["y1"],
            b["g1"],
            b["hit"],
            self._best(),
            b["state"],
            b["static_cache"],
            b["active"],
            b["ended"],
            b["events"],
            self._terminal_static,
            self._blades(),
            self.dt,
            self.rest_speed,
        )
        self._write(("after", "start", "g0", "pending", "has", "state", "static_cache", "active", "ended", "events"), out)
        b["y0"].copy_(b["after"])
        if self.logging:
            columns = tuple(b[name] for name in self._LOG)
            logged = self._log_fn(b["has"], b["t_step"], self._best(), b["after"], b["log_count"], columns)
            self._write((*self._LOG, "log_count"), logged[:-1])
            b["log_lost"].add_(logged[-1])

    # ---- the public step ------------------------------------------------------------------------------------------

    def set_blades(self, p0: Tensor, v0: Tensor, q0: Tensor, p1: Tensor, v1: Tensor, q1: Tensor) -> None:
        """The blades' states at this step's two ends, ``(N, B, 3)`` / ``(N, B, 4)`` (game axes, ``(w, x, y, z)``)."""

        b = self._buf
        for name, value in (("p0", p0), ("v0", v0), ("q0", q0), ("p1", p1), ("v1", v1), ("q1", q1)):
            b[f"raw_{name}"].copy_(value)

    def set_compliance(self, a: Tensor, b: Tensor, c: Tensor, on: Tensor) -> None:
        """
        Articulated blades: the blocks of ``J M^-1 J^T`` about each blade centre, in the table frame, ``(N, B, 3, 3)``
        each, and which blades use them, ``(N, B)`` (:func:`lib.physics.vectorized.sim.contact_fn`).
        """

        if not self.compliance:
            raise ValueError("this batch was built without compliance=True")
        buf = self._buf
        for name, value in (("comp_a", a), ("comp_b", b), ("comp_c", c), ("comp_on", on)):
            buf[name].copy_(value)

    def log_view(self) -> dict[str, Tensor]:
        """
        The on-device log since it was last emptied: ``count (N,)`` and, per environment and slot, ``time (N, L)``
        (absolute, float64), ``surface``, ``region``, ``point``, ``normal``, ``before`` and ``after``; slots
        ``>= count`` are stale. Views of the buffers: read them before the next step.
        """

        if not self.logging:
            raise ValueError("this batch keeps no log")
        b = self._buf
        names = {
            "time": "log_time",
            "surface": "log_surface",
            "region": "log_region",
            "point": "log_point",
            "normal": "log_normal",
            "before": "log_before",
            "after": "log_after",
        }
        out = {key: b[name] for key, name in names.items()}
        out["count"] = b["log_count"]
        return out

    def clear_log(self) -> None:
        """Empty the on-device log (without moving it into :attr:`log`)."""

        if self.logging:
            self._buf["log_count"].zero_()

    def warm_up(self) -> dict[str, tuple[float, float]]:
        """
        Compile and capture every phase now: the prologue, both detects, each pair tier's roots, roots_more and
        roots_finish, and apply. On their first use they would otherwise pay that inside some later step. A phase
        compiles on its first call, which on CUDA takes tens of seconds for the root loop; a robot-ball coupling's first
        rally pass spent about 240 s here (``reports/physics-ref.md`` Addendum 17).

        Each phase runs once out of turn, and the batch's buffers are put back after it, so the batch's state and every
        later result are what they would have been without this call. Already captured phases are skipped (idempotent).
        Without CUDA graphs it does nothing.

        :return: :attr:`setup_s`, each phase's (compile and warm-up, capture) seconds.
        """

        if not self.graphs:
            return dict(self.setup_s)
        phases = [self._prologue, self._detect, self._detect_last, *(phase for tier in self._roots for phase in tier), self._apply]
        saved = self._snapshot()
        for phase in phases:
            if phase.graph is None:
                phase()
                self._restore(saved)
        torch.cuda.synchronize()
        return dict(self.setup_s)

    def step(self, paddles: BladeTrack | None = None) -> None:
        """
        Advance every live ball by ``dt``, resolving every contact inside the step.

        :param paddles: The blades over this step (or set them with :meth:`set_blades` first and pass ``None``).
        """

        if paddles is not None:
            if abs(paddles.h - self.dt) > 1e-15 * self.dt:
                raise ValueError("the blades' step is not this batch's dt")
            self.set_blades(paddles.p0, paddles.v0, paddles.q0, paddles.p1, paddles.v1, paddles.q1)
        self._prologue()
        for round_ in range(self.max_events_per_step + 1):
            last = round_ == self.max_events_per_step
            (self._detect_last if last else self._detect)()
            count, log_used, _ = self._buf["report"].tolist()
            if self.logging and self.log_history and log_used >= self._buf["log_time"].shape[1] - self.max_events_per_step - 1:
                self.flush()
            if count == 0 or last:
                break
            self._buf["base"].zero_()
            remaining = count
            while remaining > 0:
                # The smallest batch that takes every remaining pair, else the largest.
                tier = next((t for t, capacity in enumerate(self.tiers) if capacity >= remaining), len(self.tiers) - 1)
                start, more, finish = self._roots[tier]
                start()
                done = self.root_block
                while done < self.root_iterations and not self._buf["report"][2].item():
                    more()
                    done += self.root_block
                finish()
                remaining -= self.tiers[tier]
            self._apply()
        self.t += self.dt

    def run(self, steps: int, paddles: Sequence[BladeTrack] | None = None) -> None:
        """``steps`` steps (the ``k``-th with ``paddles[k]`` when given)."""

        for k in range(steps):
            self.step(None if paddles is None else paddles[k])

    def reset(self, rows: Tensor, state: Tensor) -> None:
        """Restart some environments' balls from new states, live again (an RL auto-reset)."""

        b = self._buf
        rows = torch.as_tensor(rows, device=self.params.device)
        if rows.dtype == torch.bool:
            rows = torch.nonzero(rows).flatten()
        state = torch.as_tensor(state, dtype=self.params.dtype, device=self.params.device).reshape(-1, 9)
        b["state"][rows] = state
        b["active"][rows] = True
        b["ended"][rows] = 0
        b["static_cache"][rows] = static_gaps(state[:, :3], self.params, rows)

    # ---- state and the event log ------------------------------------------------------------------------------------

    @property
    def state(self) -> Tensor:
        return self._buf["state"]

    @property
    def active(self) -> Tensor:
        return self._buf["active"]

    @active.setter
    def active(self, value: Tensor) -> None:
        if value is not self._buf["active"]:
            self._buf["active"].copy_(value)

    @property
    def ended(self) -> Tensor:
        return self._buf["ended"]

    @property
    def events(self) -> int:
        """Contacts resolved so far (reads the device)."""

        return int(self._buf["events"])

    def flush(self) -> None:
        """Move the on-device log into :attr:`log` and empty it."""

        if not self.logging:
            return
        b = self._buf
        counts = b["log_count"]
        slots = torch.arange(b["log_time"].shape[1], device=counts.device)
        mask = slots[None, :] < counts[:, None]
        rows, columns = torch.nonzero(mask, as_tuple=True)
        if rows.numel():
            assert self._log is not None
            self._log.append(
                env=rows,
                time=b["log_time"][rows, columns],
                surface=b["log_surface"][rows, columns],
                region=b["log_region"][rows, columns],
                point=b["log_point"][rows, columns],
                normal=b["log_normal"][rows, columns],
                before=b["log_before"][rows, columns],
                after=b["log_after"][rows, columns],
            )
        counts.zero_()
        self.log_lost = int(b["log_lost"])

    @property
    def log(self) -> EventLog | None:
        """Every logged contact (empties the device log first)."""

        self.flush()
        return self._log


__all__ = [
    "DenseBallBatch",
    "apply_fn",
    "blade_motion",
    "detect_fn",
    "illinois_block_fn",
    "log_fn",
    "pair_gap_one",
    "prologue_fn",
    "roots_finish_fn",
    "roots_prep_fn",
]
