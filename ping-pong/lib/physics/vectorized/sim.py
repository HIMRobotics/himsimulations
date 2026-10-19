"""
The batched event-driven step: every environment's ball over one step, with every touch resolved where it happens.

:meth:`BallBatch.step` advances all live balls by ``dt`` the way
:func:`lib.physics.reference.simulate.simulate` does, in lockstep:

1. an RK4 step to the step's end (:func:`lib.physics.vectorized.flight.rk4_step`);
2. every watched gap at the end; a gap that went from ``>= 0`` to ``< 0`` fired (a gap negative at the start is not
   armed), as in :func:`lib.physics.reference.flight.integrate_until_event`;
3. each fired gap's root inside the step, on the RK4 path (re-stepping from the step's start), to ``root_tolerance``
   (Illinois steps, then the bracket's non-negative end, the oracle's event state);
4. the contact frame there (:mod:`.geometry`); a root that is not a contact (the table plane beside the table, a
   blade beside its outline, a separating touch) changes nothing; the earliest contact of each ball, ties to the
   lower surface index, is resolved (:mod:`.impact`) and logged;
5. a ball whose contact is terminal (``terminal_kinds``), or that rebounds slower than ``rest_speed`` from a surface
   gravity presses it onto, stops (the oracle with ``rolling=False``); the others carry on from the touch to the
   step's end, and steps 1-4 repeat on that remainder.

Blades move over a step as :class:`lib.physics.vectorized.geometry.BladeTrack` (the game's ``PaddleSegment``), built
from the robot's paddle states at the step's two ends. ``graze=True`` also searches a step that ends clear of a
surface, but came within its travel of it, for a touch inside it (the oracle's ``graze``).

The step grid is global (the oracle restarts its grid after each event): the two differ by the RK4 truncation error,
about 1e-12 m at a 1 ms step.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

from lib.physics.canonical_events import REGION_NAMES, CanonicalEvent, RawTouch, merge
from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.flight import rk4_step
from lib.physics.vectorized.geometry import (
    MAT_RIM,
    MAT_RUBBER,
    REGION,
    STATIC_COUNT,
    BladeTrack,
    blade_contact,
    blade_gaps_of,
    blade_state,
    static_contact,
    static_gaps,
)
from lib.physics.vectorized.impact import _skew, mat_mul, resolve
from lib.physics.vectorized.params import VecParams

Tensor = Any


KIND_OF_STATIC = ("table", "net", "floor", "table", "net", "net", "net", "net")
"""Each static surface's contact kind (posts and clamps belong to the net assembly)."""

ENDED_TERMINAL, ENDED_REST, ENDED_OVERFLOW = 1, 2, 3
"""Why a ball stopped: a terminal contact, a slow rebound (``rest``), or more contacts in one step than allowed."""


@dataclass
class EventLog:
    """
    Every resolved contact, appended per step on the batch's device.

    Rows: ``env``, ``time`` (s), ``surface`` (static index, or 8 + blade), ``region`` (code), ``point``, ``normal``,
    ``before`` and ``after`` (ball states).
    """

    env: list[Tensor] = field(default_factory=list)
    time: list[Tensor] = field(default_factory=list)
    surface: list[Tensor] = field(default_factory=list)
    region: list[Tensor] = field(default_factory=list)
    point: list[Tensor] = field(default_factory=list)
    normal: list[Tensor] = field(default_factory=list)
    before: list[Tensor] = field(default_factory=list)
    after: list[Tensor] = field(default_factory=list)

    def append(self, **rows: Tensor) -> None:
        """Add one batch of rows."""

        for name, value in rows.items():
            getattr(self, name).append(value)

    def tensors(self) -> dict[str, Tensor]:
        """Every row so far, concatenated, on the CPU, in time order per environment."""

        if not self.env:
            return {}
        out = {name: torch.cat(getattr(self, name)).cpu() for name in ("env", "time", "surface", "region", "point", "normal", "before", "after")}
        order = torch.argsort(out["time"], stable=True)
        return {name: value[order] for name, value in out.items()}

    def touches(self, env: int, players: Sequence[str]) -> list[RawTouch]:
        """One environment's contacts as :class:`lib.physics.canonical_events.RawTouch` rows."""

        rows = self.tensors()
        if not rows:
            return []
        mine = torch.nonzero(rows["env"] == env).flatten().tolist()
        out = []
        for position, i in enumerate(mine):
            surface = int(rows["surface"][i])
            kind = KIND_OF_STATIC[surface] if surface < STATIC_COUNT else "paddle"
            player = players[surface - STATIC_COUNT] if surface >= STATIC_COUNT else None
            region = REGION_NAMES[int(rows["region"][i])]
            if kind == "floor" or (kind == "paddle" and region != "blade_edge"):
                region = ""
            after = rows["after"][i].double().numpy()
            t = float(rows["time"][i])
            out.append(
                RawTouch(
                    position,
                    kind,
                    region,
                    player,
                    t,
                    rows["point"][i].double().numpy(),
                    rows["normal"][i].double().numpy(),
                    rows["before"][i].double().numpy(),
                    after,
                    t,
                    after,
                )
            )
        return out

    def canonical(self, env: int, players: Sequence[str] = ("A", "B")) -> tuple[CanonicalEvent, ...]:
        """One environment's canonical contact sequence (:func:`lib.physics.canonical_events.merge`)."""

        return merge(self.touches(env, players))


# ---- the pure step functions (compiled with torch.compile on a GPU) ----------------------------------------------


def gaps_fn(params: VecParams, y: Tensor, env: Tensor, u: Tensor, blades: tuple[Tensor, ...] | None, h: float, continuous: bool) -> Tensor:
    """Every watched gap of each row, ``(M, 8 + B)``: the static surfaces, then the blades at step fraction ``u``."""

    static = static_gaps(y[:, :3], params, env, continuous=continuous)
    if blades is None:
        return static
    return torch.cat([static, blade_gaps_of(y[:, :3], blades, h, u, params, env)], -1)


def advance_fn(params: VecParams, ys: Tensor, step: Tensor, env: Tensor, blades: tuple[Tensor, ...] | None, h: float) -> tuple[Tensor, Tensor]:
    """The RK4 step of each row over ``step`` and every gap at its end (the step's end, ``u = 1``)."""

    y1 = rk4_step(ys, step, params, env)
    return y1, gaps_fn(params, y1, env, torch.ones_like(step), blades, h, False)


def pair_gap_fn(params: VecParams, ys: Tensor, x: Tensor, env: Tensor, start: Tensor, surface: Tensor, blades: tuple[Tensor, ...] | None, h: float) -> Tensor:
    """Each pair's gap (continuous form) at ``x`` into its step, on the RK4 path from ``ys``."""

    y = rk4_step(ys, x, params, env)
    gaps = gaps_fn(params, y, env, (start + x) / h, blades, h, True)
    return gaps.gather(1, surface[:, None])[:, 0]


def illinois_fn(
    params: VecParams,
    ys: Tensor,
    env: Tensor,
    start: Tensor,
    surface: Tensor,
    blades: tuple[Tensor, ...] | None,
    h: float,
    lo: Tensor,
    hi: Tensor,
    g_lo: Tensor,
    g_hi: Tensor,
    side: Tensor,
    secant: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """One Illinois (or, with ``secant`` false, bisection) step on each pair's bracket ``[lo, hi]``."""

    width = hi - lo
    bisect = 0.5 * (lo + hi)
    if secant:
        guess = hi - g_hi * width / (g_hi - g_lo)
        ok = torch.isfinite(guess) & (guess > lo) & (guess < hi)
        x = torch.where(ok, guess, bisect)
    else:
        x = bisect
    g = pair_gap_fn(params, ys, x, env, start, surface, blades, h)
    left = g >= 0.0
    # Illinois: halve the stale end's value when the same end moves twice.
    g_hi = torch.where(left & (side > 0.0), 0.5 * g_hi, g_hi)
    g_lo = torch.where(~left & (side < 0.0), 0.5 * g_lo, g_lo)
    lo_new = torch.where(left, x, lo)
    g_lo_new = torch.where(left, g, g_lo)
    hi_new = torch.where(left, hi, x)
    g_hi_new = torch.where(left, g_hi, g)
    done = width <= 0.0
    return (
        torch.where(done, lo, lo_new),
        torch.where(done, hi, hi_new),
        torch.where(done, g_lo, g_lo_new),
        torch.where(done, g_hi, g_hi_new),
        torch.where(left, torch.ones_like(side), -torch.ones_like(side)),
    )


def contact_fn(
    params: VecParams,
    y_hit: Tensor,
    env: Tensor,
    surface: Tensor,
    when: Tensor,
    blades: tuple[Tensor, ...] | None,
    h: float,
    compliance: tuple[Tensor, ...] | None = None,
) -> tuple[Tensor, ...]:
    """
    Each pair's contact frame at its root and the impact there: ``(valid, normal, point, region, material, velocity,
    spin, separation)``. A pair is valid when its root is a contact and the ball approaches the surface.

    ``compliance`` = ``(A, B, C, on)``, each ``(N, B, ...)``: per environment and blade, the blocks of ``J M^-1 J^T``
    about the blade centre (``A`` translational, ``B`` coupling, ``C`` rotational, ``(3, 3)`` each) and whether to use
    them (``on``). A blade contact with ``on`` takes the surface compliance at its contact point,
    ``K = A + B [r]x - [r]x B^T - [r]x C [r]x`` (``r`` from the blade centre), in place of the contract's paddle mass and
    free-paddle inertia: an articulated paddle (the robot's arm, motor armature included).
    """

    static = surface < STATIC_COUNT
    frame = static_contact(surface.clamp(max=STATIC_COUNT - 1), y_hit[:, :3], params, env)
    valid = torch.where(static, frame.valid, torch.zeros_like(frame.valid))
    normal, point, region, material = frame.normal, frame.point, frame.region, frame.material
    surface_velocity = torch.zeros_like(normal)
    mass = torch.full_like(when, float("inf"))
    offset = torch.zeros_like(normal)
    inertia = torch.eye(3, dtype=y_hit.dtype, device=y_hit.device).expand(when.shape[0], 3, 3)
    k_surface = compliant = None
    if blades is not None:
        blade = (surface - STATIC_COUNT).clamp(min=0)
        position, orientation, velocity, omega = blade_state(blades, h, when / h, env, blade)
        touch = blade_contact(y_hit[:, :3], position, orientation, velocity, omega, params, env)
        rows = ~static
        valid = torch.where(rows, touch.valid, valid)
        normal = torch.where(rows[:, None], touch.normal, normal)
        point = torch.where(rows[:, None], touch.point, point)
        region = torch.where(rows, torch.where(touch.edge, torch.full_like(region, REGION["blade_edge"]), torch.zeros_like(region)), region)
        material = torch.where(rows, torch.where(touch.edge, torch.full_like(material, MAT_RIM), torch.full_like(material, MAT_RUBBER)), material)
        surface_velocity = torch.where(rows[:, None], touch.surface_velocity, surface_velocity)
        mass = torch.where(rows, params.paddle_mass[env], mass)
        offset = torch.where(rows[:, None], touch.offset, offset)
        inertia = torch.where(rows[:, None, None], touch.inertia, inertia)
        if compliance is not None:
            a_all, b_all, c_all, on_all = compliance
            a_m, b_m, c_m = a_all[env, blade], b_all[env, blade], c_all[env, blade]
            skew = _skew(touch.offset)
            k_surface = a_m + mat_mul(b_m, skew) - mat_mul(skew, b_m.transpose(-1, -2)) - mat_mul(mat_mul(skew, c_m), skew)
            compliant = rows & on_all[env, blade]
    impact = resolve(
        y_hit[:, 3:6],
        y_hit[:, 6:9],
        normal,
        surface_velocity,
        material,
        env,
        params,
        effective_mass=mass,
        offset=offset,
        inertia=inertia,
        compliance=k_surface,
        compliant=compliant,
    )
    return valid & impact.approaching, normal, point, region, material, impact.velocity, impact.spin, impact.separation_speed


def rk4_fn(params: VecParams, y: Tensor, step: Tensor, env: Tensor) -> Tensor:
    """:func:`lib.physics.vectorized.flight.rk4_step` (a separate graph for the compiled path)."""

    return rk4_step(y, step, params, env)


def _compiled(fn: Any) -> Any:
    """
    ``fn`` compiled with dynamic shapes, called eagerly on a single row (inductor's size-1 kernels failed to launch
    on an sm_120 GPU with torch 2.11).

    If compilation fails, ``fn`` runs eagerly from then on, with one warning, instead of failing every call: the
    Isaac Lab 3.0 engine image has no C compiler, which triton needs to build its launcher.
    """

    compiled = torch.compile(fn, dynamic=True)
    eager = False

    def call(*args: Any) -> Any:
        nonlocal eager
        if eager or args[1].shape[0] < 2:
            return fn(*args)
        try:
            return compiled(*args)
        except Exception as error:
            from torch._dynamo.exc import BackendCompilerFailed  # pyright: ignore[reportMissingImports]
            from torch._inductor.exc import InductorError  # pyright: ignore[reportMissingImports]

            if not isinstance(error, (BackendCompilerFailed, InductorError)):
                raise
            eager = True
            warnings.warn(f"torch.compile failed for {fn.__name__}; running it eagerly: {error}", RuntimeWarning, stacklevel=2)
            return fn(*args)

    return call


class BallBatch:
    """
    One live ball per environment, stepped in lockstep with the oracle's physics.

    :param params: The environments' physics.
    :param state: Initial states ``(N, 9)``: position (m), velocity (m/s), spin (rad/s).
    :param t0: Start time, s.
    :param terminal_kinds: Contact kinds that stop a ball (default: the floor).
    :param rest_speed: A static-surface rebound slower than this from a surface gravity presses the ball onto stops it.
    :param graze: Search steps that end clear of a surface for a touch inside them.
    :param root_iterations: Root-finding iterations per fired gap (at most).
    :param root_tolerance: Root bracket, s.
    :param max_events_per_step: Contacts one ball may have in one step before it is stopped as ``overflow``.
    :param arm_tolerance: A gap this far below zero at a step's start still counts as armed, m. The default, 0, is the
        oracle's rule; on 1,358 calibration trials 1e-9 (float64) and 1e-5 (float32) changed no event.
    :param compile: Compile the step's pure functions with ``torch.compile`` (default: on a CUDA device). The first
        steps then pay the compilation (seconds); every later step launches a few fused kernels instead of hundreds.
    :param log: Keep an :class:`EventLog`.
    """

    def __init__(
        self,
        params: VecParams,
        state: Tensor,
        *,
        t0: float = 0.0,
        terminal_kinds: Collection[str] = ("floor",),
        rest_speed: float = 0.05,
        graze: bool = False,
        root_iterations: int = 60,
        root_tolerance: float = 1e-10,
        max_events_per_step: int = 4,
        arm_tolerance: float = 0.0,
        compile: bool | None = None,
        log: bool = True,
    ) -> None:
        self.params = params
        self.state = torch.as_tensor(state, dtype=params.dtype, device=params.device).reshape(params.n, 9).clone()
        self.t = float(t0)
        self.terminal_kinds = frozenset(terminal_kinds)
        self.rest_speed = float(rest_speed)
        self.graze = graze
        self.root_iterations = int(root_iterations)
        self.root_tolerance = float(root_tolerance)
        self.max_events_per_step = int(max_events_per_step)
        self.arm_tolerance = float(arm_tolerance)
        self.active = torch.ones(params.n, dtype=torch.bool, device=params.device)
        self.ended = torch.zeros(params.n, dtype=torch.long, device=params.device)
        self.log = EventLog() if log else None
        self._terminal_static = torch.tensor([kind in self.terminal_kinds for kind in KIND_OF_STATIC], device=params.device)
        self.events = 0
        use_compile = (str(params.device).startswith("cuda")) if compile is None else compile
        wrap = _compiled if use_compile else (lambda fn: fn)
        self._advance = wrap(advance_fn)
        self._gaps_fn = wrap(gaps_fn)
        self._illinois = wrap(illinois_fn)
        self._pair_gap = wrap(pair_gap_fn)
        self._contact = wrap(contact_fn)
        self._rk4 = wrap(rk4_fn)
        self._static = static_gaps(self.state[:, :3], params)

    def _root(
        self, ys: Tensor, env: Tensor, start: Tensor, h: Tensor, surface: Tensor, g_lo: Tensor, g_hi: Tensor, blades: tuple[Tensor, ...] | None, dt: float
    ) -> Tensor:
        """The last time (from ``start``) on the non-negative side of each pair's gap root inside ``(0, h]``."""

        lo = torch.zeros_like(h)
        hi = torch.where(g_lo < 0.0, torch.zeros_like(h), h)
        g_lo = torch.where(torch.isfinite(g_lo) & (g_lo >= 0.0), g_lo, torch.full_like(g_lo, 1.0))
        side = torch.zeros_like(h)
        for iteration in range(self.root_iterations):
            if iteration % 4 == 0 and bool(((hi - lo) <= self.root_tolerance).all()):
                break
            lo, hi, g_lo, g_hi, side = self._illinois(self.params, ys, env, start, surface, blades, dt, lo, hi, g_lo, g_hi, side, iteration % 4 != 3)
        return lo

    def _dip(
        self, ys: Tensor, env: Tensor, start: Tensor, h: Tensor, surface: Tensor, g0: Tensor, g1: Tensor, blades: tuple[Tensor, ...] | None, dt: float
    ) -> Tensor:
        """For gaps non-negative at both ends: a time inside the step where the gap is negative, else ``nan``."""

        def gap(x: Tensor) -> Tensor:
            return self._pair_gap(self.params, ys, x, env, start, surface, blades, dt)

        a, fa, c, fc = torch.zeros_like(h), g0, h, g1
        b = 0.5 * h
        fb = gap(b)
        found = torch.where(fb < 0.0, b, torch.full_like(b, float("nan")))
        for _ in range(3):
            denominator = (b - a) * (fb - fc) - (b - c) * (fb - fa)
            x = b - 0.5 * ((b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)) / torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
            usable = (denominator != 0.0) & (x > a) & (x < c)
            x = torch.where(usable, x, b)
            fx = gap(x)
            found = torch.where(torch.isnan(found) & usable & (fx < 0.0), x, found)
            better = usable & (fx < fb)
            right_of = x > b
            a, fa = (
                torch.where(better & right_of, b, torch.where(~better & ~right_of & usable, x, a)),
                torch.where(better & right_of, fb, torch.where(~better & ~right_of & usable, fx, fa)),
            )
            c, fc = (
                torch.where(better & ~right_of, b, torch.where(~better & right_of & usable, x, c)),
                torch.where(better & ~right_of, fb, torch.where(~better & right_of & usable, fx, fc)),
            )
            b, fb = torch.where(better, x, b), torch.where(better, fx, fb)
        return found

    def step(self, dt: float, paddles: BladeTrack | None = None) -> None:
        """
        Advance every live ball by ``dt`` (s), resolving every contact inside the step.

        :param dt: Step, s.
        :param paddles: The blades' motion over this step (``None``: no blades). Its step must be ``dt``.
        """

        params = self.params
        rows = torch.nonzero(self.active).flatten()
        if rows.numel() == 0:
            self.t += dt
            return
        blades = paddles.tensors if paddles is not None and paddles.count > 0 else None
        m = rows.numel()
        y0 = self.state[rows].clone()
        start = torch.zeros(m, dtype=params.dtype, device=params.device)
        g0 = self._static[rows] if blades is None else self._gaps_fn(params, y0, rows, start, blades, dt, False)
        pending = torch.arange(m, device=params.device)
        for iteration in range(self.max_events_per_step + 1):
            env = rows[pending]
            ys, ts = y0[pending], start[pending]
            h = dt - ts
            y1, g1 = self._advance(params, ys, h, env, blades, dt)
            gs = g0[pending]
            # A gap fires when it goes from at or above zero at the start to below zero at the end (the oracle's rule).
            fired = (gs >= -self.arm_tolerance) & (g1 < 0.0)
            dips = None
            if self.graze:
                dx = y1[:, :3] - ys[:, :3]
                reach = torch.sqrt((dx * dx).sum(-1))[:, None] + torch.zeros_like(g1)
                if g1.shape[1] > STATIC_COUNT:
                    reach[:, STATIC_COUNT:] += 40.0 * h[:, None]
                near = (gs >= 0.0) & (g1 >= 0.0) & (gs + g1 < reach + 1e-9)
                if bool(near.any()):
                    pr, ps = torch.nonzero(near, as_tuple=True)
                    inside = self._dip(ys[pr], env[pr], ts[pr], h[pr], ps, gs[pr, ps], g1[pr, ps], blades, dt)
                    hit_dip = ~torch.isnan(inside)
                    if bool(hit_dip.any()):
                        dips = torch.full_like(g1, float("nan"))
                        dips[pr[hit_dip], ps[hit_dip]] = inside[hit_dip]
                        fired = fired | ~torch.isnan(dips)
            hit = fired.any(1)
            calm = ~hit
            done = pending[calm]
            self.state[rows[done]] = y1[calm]
            self._static[rows[done]] = g1[calm, :STATIC_COUNT]
            if not bool(hit.any()):
                break
            if iteration == self.max_events_per_step:
                over = pending[hit]
                self.state[rows[over]] = y1[hit]
                self.active[rows[over]] = False
                self.ended[rows[over]] = ENDED_OVERFLOW
                break
            # The root of every fired gap.
            hit_index = torch.nonzero(hit).flatten()
            pr, ps = torch.nonzero(fired[hit_index], as_tuple=True)
            k = hit_index[pr]
            pair_env = env[k]
            pair_h = h[k]
            g_hi = g1[k, ps]
            if dips is not None:
                window = dips[k, ps]
                inside = ~torch.isnan(window)
                pair_h = torch.where(inside, window, pair_h)
                g_hi = torch.where(inside, self._pair_gap(params, ys[k], pair_h, pair_env, ts[k], ps, blades, dt), g_hi)
            g_lo = gs[k, ps]
            tau = self._root(ys[k], pair_env, ts[k], pair_h, ps, g_lo, g_hi, blades, dt)
            y_hit = self._rk4(params, ys[k], tau, pair_env)
            when = ts[k] + tau
            valid, normal, point, region, _material, v_out, w_out, separation = self._contact(params, y_hit, pair_env, ps, when, blades, dt)
            # Each ball's earliest valid contact, ties to the lower surface index.
            key = torch.where(valid, when, torch.full_like(when, float("inf")))
            first = torch.full((hit_index.numel(),), float("inf"), dtype=when.dtype, device=when.device).scatter_reduce(0, pr, key, reduce="amin")
            at_first = valid & (key == first[pr])
            surface_key = torch.where(at_first, ps, torch.full_like(ps, 1 << 30))
            lowest = torch.full((hit_index.numel(),), 1 << 30, dtype=ps.dtype, device=ps.device).scatter_reduce(0, pr, surface_key, reduce="amin")
            chosen = at_first & (ps == lowest[pr])
            has_event = torch.zeros(hit_index.numel(), dtype=torch.bool, device=params.device)
            has_event[pr[chosen]] = True
            # A ball whose fired gaps were not contacts finishes the step untouched.
            quiet = hit_index[~has_event]
            self.state[rows[pending[quiet]]] = y1[quiet]
            self._static[rows[pending[quiet]]] = g1[quiet, :STATIC_COUNT]
            c = torch.nonzero(chosen).flatten()
            if c.numel() == 0:
                break
            ball = pending[k[c]]
            ball_env = pair_env[c]
            after = torch.cat([y_hit[c, :3], v_out[c], w_out[c]], -1)
            surface = ps[c]
            if self.log is not None:
                self.log.append(
                    env=ball_env, time=self.t + when[c], surface=surface, region=region[c], point=point[c], normal=normal[c], before=y_hit[c], after=after
                )
            self.events += int(c.numel())
            # Terminal contacts and slow rebounds end the ball.
            static_c = surface < STATIC_COUNT
            terminal = static_c & self._terminal_static[surface.clamp(max=STATIC_COUNT - 1)]
            rest = static_c & (separation[c] < self.rest_speed) & (normal[c, 2] > 0.0)
            stop = terminal | rest
            self.state[ball_env[stop]] = after[stop]
            self._static[ball_env[stop]] = static_gaps(after[stop, :3], params, ball_env[stop])
            self.active[ball_env[stop]] = False
            self.ended[ball_env[stop]] = torch.where(terminal[stop], torch.full_like(surface[stop], ENDED_TERMINAL), torch.full_like(surface[stop], ENDED_REST))
            go = ~stop
            carry = ball[go]
            if carry.numel() == 0:
                break
            y0[carry] = after[go]
            start[carry] = when[c][go]
            g0[carry] = self._gaps_fn(params, after[go], ball_env[go], when[c][go] / dt, blades, dt, False)
            pending = carry
        self.t += dt

    def run(self, duration: float, dt: float, paddles: Sequence[BladeTrack] | None = None) -> None:
        """Step ``duration / dt`` times (the ``k``-th step with ``paddles[k]`` when given)."""

        steps = round(duration / dt)
        for k in range(steps):
            self.step(dt, None if paddles is None else paddles[k])

    def reset(self, rows: Tensor, state: Tensor) -> None:
        """
        Restart some environments' balls from new states, live again (an RL auto-reset). Their logged events stay.

        :param rows: Environment indices ``(K,)``, or a boolean mask ``(N,)``.
        :param state: The new states ``(K, 9)``: position (m), velocity (m/s), spin (rad/s).
        """

        params = self.params
        rows = torch.as_tensor(rows, device=params.device)
        if rows.dtype == torch.bool:
            rows = torch.nonzero(rows).flatten()
        state = torch.as_tensor(state, dtype=params.dtype, device=params.device).reshape(-1, 9)
        self.state[rows] = state
        self.active[rows] = True
        self.ended[rows] = 0
        self._static[rows] = static_gaps(state[:, :3], params, rows)


__all__ = ["ENDED_OVERFLOW", "ENDED_REST", "ENDED_TERMINAL", "BallBatch", "EventLog", "advance_fn", "contact_fn", "gaps_fn", "illinois_fn", "pair_gap_fn"]
