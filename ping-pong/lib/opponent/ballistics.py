"""
Batched free flight for the opponent's planning: the bounce predictor, the post-bounce path, launch solves and shot
descriptions.

Every flight here is the calibrated model's free flight (:func:`lib.physics.vectorized.flight.rk4_step`, the oracle's
RK4 batched) with each row's own physics, and every surface event is a root of a smooth free-flight coordinate found
by marching coarse RK4 steps and closing the bracket with Illinois steps on RK4 sub-steps from the bracket's start. A
fixed number of iterations per call, no per-row control flow, so a call is one graph over all rows.

What the generator does with the reference oracle (``Rally.predict_bounce``: ``simulate`` at 4 ms against the table,
the net and the floor, no fixtures; ``flight.solve_launch``: Gauss-Newton on the landing) this does on the same
free-flight path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.geometry import MAT_TABLE
from lib.physics.vectorized.impact import inv3, mat_vec, resolve
from lib.physics.vectorized.params import VecParams

Tensor = Any


@dataclass(frozen=True)
class Coeffs:
    """
    Each row's flight coefficients, gathered once from :class:`VecParams` (``env`` rows), so a RK4 step reads no
    per-environment tables. :func:`rk4` is :func:`lib.physics.vectorized.flight.rk4_step` term for term.
    """

    gravity: Tensor
    radius: Tensor
    gain: Tensor
    drag0: Tensor
    drag_slope: Tensor
    slope: Tensor
    c_max: Tensor
    decay_gain: Tensor
    lift_form: str

    @classmethod
    def of(cls, params: VecParams, env: Tensor) -> Coeffs:
        return cls(
            params.gravity[env],
            params.radius[env],
            params.aero_gain[env],
            params.drag_coefficient[env],
            params.drag_spin_slope[env],
            params.lift_slope[env],
            params.lift_c_max[env],
            params.spin_decay_gain[env],
            params.lift_form,
        )

    def _map(self, fn: Any) -> Coeffs:
        return Coeffs(
            fn(self.gravity),
            fn(self.radius),
            fn(self.gain),
            fn(self.drag0),
            fn(self.drag_slope),
            fn(self.slope),
            fn(self.c_max),
            fn(self.decay_gain),
            self.lift_form,
        )

    def repeat(self, k: int) -> Coeffs:
        return self._map(lambda v: v.repeat(k))

    def take(self, index: Tensor) -> Coeffs:
        return self._map(lambda v: v[index])


def _acceleration(v: Tensor, w: Tensor, c: tuple[Tensor, ...], lift_form: str) -> tuple[Tensor, Tensor]:
    gravity, radius, gain, drag0, drag_slope, slope, c_max, decay_gain = c
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    v2 = vx * vx + vy * vy + vz * vz
    moving = v2 > 0.0
    speed = torch.sqrt(v2)
    safe_v2 = torch.where(moving, v2, torch.ones_like(v2))
    wv = wx * vx + wy * vy + wz * vz
    wperp2 = wx * wx + wy * wy + wz * wz - wv * wv / safe_v2
    wperp = torch.where(moving & (wperp2 > 0.0), torch.sqrt(wperp2.clamp_min(0.0)), torch.zeros_like(wperp2))
    s = radius * wperp / torch.where(moving, speed, torch.ones_like(speed))
    drag = gain * (drag0 + drag_slope * s) * speed
    ax = 0.0 - drag * vx
    ay = 0.0 - drag * vy
    az = -gravity - drag * vz
    spinning = wperp > 0.0
    if lift_form == "rational":
        lift = slope * s / (1.0 + slope * s / c_max)
    elif lift_form == "linear":
        lift = torch.minimum(slope * s, c_max)
    else:
        lift = c_max + torch.zeros_like(s)
    lift = torch.where(s > 0.0, lift, torch.zeros_like(lift))
    magnus = torch.where(spinning, gain * lift * speed / torch.where(spinning, wperp, torch.ones_like(wperp)), torch.zeros_like(wperp))
    ax = ax + magnus * (wy * vz - wz * vy)
    ay = ay + magnus * (wz * vx - wx * vz)
    az = az + magnus * (wx * vy - wy * vx)
    decay = decay_gain * speed
    return torch.stack([ax, ay, az], -1), torch.stack([-decay * wx, -decay * wy, -decay * wz], -1)


def _rk4_eager(y: Tensor, h: Tensor, c: tuple[Tensor, ...], lift_form: str) -> Tensor:
    hh = h[:, None]
    p, v, w = y[:, 0:3], y[:, 3:6], y[:, 6:9]
    half = 0.5 * hh
    a1, b1 = _acceleration(v, w, c, lift_form)
    v2, w2 = v + half * a1, w + half * b1
    a2, b2 = _acceleration(v2, w2, c, lift_form)
    v3, w3 = v + half * a2, w + half * b2
    a3, b3 = _acceleration(v3, w3, c, lift_form)
    v4, w4 = v + hh * a3, w + hh * b3
    a4, b4 = _acceleration(v4, w4, c, lift_form)
    sixth = hh / 6.0
    return torch.cat([p + sixth * (v + 2.0 * v2 + 2.0 * v3 + v4), v + sixth * (a1 + 2.0 * a2 + 2.0 * a3 + a4), w + sixth * (b1 + 2.0 * b2 + 2.0 * b3 + b4)], -1)


_COMPILED: dict[Any, Any] = {}
COMPILE = True
"""Compile the hot pure-tensor bodies with ``torch.compile`` on CUDA (eager, with one warning, if that fails: the
Isaac Lab image has no C compiler unless the phase-two image adds one)."""


def compiled(fn: Any) -> Any:
    """``fn`` compiled (dynamic shapes) when the first tensor argument is on CUDA, else eager; one shared fallback."""

    def call(*args: Any) -> Any:
        first = next(a for a in args if torch.is_tensor(a))
        if COMPILE and first.is_cuda and first.shape[0] > 1 and not _COMPILED.get("failed"):
            fast = _COMPILED.get(fn)
            if fast is None:
                fast = torch.compile(fn, dynamic=True)
                _COMPILED[fn] = fast
            try:
                return fast(*args)
            except Exception as error:
                import warnings

                _COMPILED["failed"] = True
                _COMPILED["error"] = f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"
                warnings.warn(f"torch.compile failed in the opponent; running eagerly: {error}", RuntimeWarning, stacklevel=2)
        return fn(*args)

    return call


def compile_status() -> dict[str, Any]:
    """
    Whether the opponent's hot bodies run compiled: ``{"enabled", "fallback", "error"}``. ``fallback`` is set once a
    compile failed (no C compiler for Triton, for one) and every body runs eagerly from then on, with the same results;
    record it with a run's results.
    """

    return {"enabled": bool(COMPILE), "fallback": bool(_COMPILED.get("failed", False)), "error": _COMPILED.get("error")}


def _coeff_args(c: Coeffs) -> tuple[Tensor, ...]:
    return (c.gravity, c.radius, c.gain, c.drag0, c.drag_slope, c.slope, c.c_max, c.decay_gain)


def _rk4_body(y: Tensor, h: Tensor, c: tuple[Tensor, ...], lift_form: str) -> Tensor:
    return _rk4_eager(y, h, c, lift_form)


_rk4_fast = compiled(_rk4_body)


def rk4(y: Tensor, h: Tensor | float, c: Coeffs) -> Tensor:
    """One RK4 step per row with the rows' coefficients (``h`` per row or one for all)."""

    if not torch.is_tensor(h):
        h = torch.full((y.shape[0],), float(h), dtype=y.dtype, device=y.device)
    return _rk4_fast(y, h, _coeff_args(c), c.lift_form)


def _march_body(
    y: Tensor,
    g: Tensor,
    t: Tensor,
    h: Tensor,
    found: Tensor,
    base: Tensor,
    base_t: Tensor,
    g_lo: Tensor,
    g_hi: Tensor,
    gvec: Tensor,
    gconst: Tensor,
    c: tuple[Tensor, ...],
    lift_form: str,
) -> tuple[Tensor, ...]:
    y1 = _rk4_eager(y, h, c, lift_form)
    g1 = (y1[:, 0:3] * gvec).sum(-1) + gconst
    hit = (g >= 0.0) & ~found & (g1 < 0.0)
    base = torch.where(hit[:, None], y, base)
    base_t = torch.where(hit, t, base_t)
    g_lo = torch.where(hit, g, g_lo)
    g_hi = torch.where(hit, g1, g_hi)
    return y1, g1, t + h, found | hit, base, base_t, g_lo, g_hi


def _illinois_body(
    base: Tensor, lo: Tensor, hi: Tensor, f_lo: Tensor, f_hi: Tensor, side: Tensor, gvec: Tensor, gconst: Tensor, c: tuple[Tensor, ...], lift_form: str
) -> tuple[Tensor, ...]:
    width = hi - lo
    guess = hi - f_hi * width / (f_hi - f_lo)
    ok = torch.isfinite(guess) & (guess > lo) & (guess < hi)
    x = torch.where(ok, guess, 0.5 * (lo + hi))
    fx = (_rk4_eager(base, x, c, lift_form)[:, 0:3] * gvec).sum(-1) + gconst
    left = fx >= 0.0
    f_hi = torch.where(left & (side > 0.0), 0.5 * f_hi, f_hi)
    f_lo = torch.where(~left & (side < 0.0), 0.5 * f_lo, f_lo)
    lo, f_lo2 = torch.where(left, x, lo), torch.where(left, fx, f_lo)
    hi, f_hi2 = torch.where(left, hi, x), torch.where(left, f_hi, fx)
    side = torch.where(left, torch.ones_like(side), -torch.ones_like(side))
    return lo, hi, f_lo2, f_hi2, side


_march_fast = compiled(_march_body)
_illinois_fast = compiled(_illinois_body)


COARSE_S = 0.04
"""Marching step of a root search, s (RK4 at 40 ms is within about 0.1 mm of the 1 ms path over a 1 s flight)."""
ILLINOIS_STEPS = 14
SUBSTEP_S = 0.01
"""Largest RK4 sub-step when propagating to an arbitrary time, s."""


def propagate(params: VecParams, y: Tensor, env: Tensor, duration: Tensor, max_step: float = SUBSTEP_S, *, steps: int | None = None) -> Tensor:
    """
    Free flight of each row for its own ``duration`` (s, either sign), in equal RK4 steps of at most ``max_step``.

    :param steps: A fixed step count (no host synchronisation; graph-capturable). Each row's step is then
        ``duration / steps``.
    """

    c = Coeffs.of(params, env)

    if y.shape[0] == 0:
        return y
    if steps is None:
        longest = float(duration.abs().max())
        steps = max(1, int(-(-longest // max_step)))
    h = duration / steps
    for _ in range(steps):
        y = rk4(y, h, c)
    return y


@dataclass(frozen=True)
class Root:
    """
    The first descending root of a free-flight coordinate.

    :ivar found: A root was found within the horizon.
    :ivar t: Its time from the start, s (``inf`` where not found).
    :ivar state: The state there (the last marched state where not found).
    """

    found: Tensor
    t: Tensor
    state: Tensor


def first_root(params: VecParams, y0: Tensor, env: Tensor, gap: Any, horizon: float | Tensor, *, coarse: float = COARSE_S, steps: int | None = None) -> Root:
    """
    The first time each row's ``gap`` goes from ``>= 0`` to ``< 0`` along its free flight, within ``horizon``.

    :param gap: A linear function of the ball's position, ``(gvec (M, 3), gconst (M,))``: ``gvec . p + gconst``
        (:func:`table_gap`, :func:`plane_gap`).
    :param steps: A fixed marching step count (no host synchronisation; graph-capturable); each row marches
        ``horizon / steps``. Default: enough steps of at most ``coarse`` for the longest horizon.
    """

    c = Coeffs.of(params, env)
    gvec, gconst = gap
    m = y0.shape[0]
    dtype, device = y0.dtype, y0.device
    horizon_t: Tensor = horizon if torch.is_tensor(horizon) else torch.full((m,), float(horizon), dtype=dtype, device=device)
    if m == 0:
        return Root(torch.zeros(0, dtype=torch.bool, device=device), horizon_t, y0)
    if steps is None:
        steps = max(1, int(-(-float(horizon_t.max()) // coarse)))
    h = horizon_t / steps
    args = _coeff_args(c)
    y = y0
    g = (y[:, 0:3] * gvec).sum(-1) + gconst
    found = torch.zeros(m, dtype=torch.bool, device=device)
    base = y0.clone()
    base_t = torch.zeros(m, dtype=dtype, device=device)
    g_lo = g.clone()
    g_hi = torch.zeros_like(g)
    t = torch.zeros(m, dtype=dtype, device=device)
    for _ in range(steps):
        y, g, t, found, base, base_t, g_lo, g_hi = _march_fast(y, g, t, h, found, base, base_t, g_lo, g_hi, gvec, gconst, args, c.lift_form)
    lo = torch.zeros(m, dtype=dtype, device=device)
    hi = h.clone()
    side = torch.zeros(m, dtype=dtype, device=device)
    f_lo, f_hi = g_lo, g_hi
    for _ in range(ILLINOIS_STEPS):
        lo, hi, f_lo, f_hi, side = _illinois_fast(base, lo, hi, f_lo, f_hi, side, gvec, gconst, args, c.lift_form)
    tau = torch.where((f_lo - f_hi).abs() > 0.0, lo + f_lo * (hi - lo) / (f_lo - f_hi), lo)
    tau = torch.minimum(torch.maximum(tau, lo), hi)
    state = rk4(base, tau, c)
    t_root = torch.where(found, base_t + tau, torch.full_like(tau, float("inf")))
    return Root(found, t_root, torch.where(found[:, None], state, y))


def _det3(a: Tensor) -> Tensor:
    return (
        a[:, 0, 0] * (a[:, 1, 1] * a[:, 2, 2] - a[:, 1, 2] * a[:, 2, 1])
        - a[:, 0, 1] * (a[:, 1, 0] * a[:, 2, 2] - a[:, 1, 2] * a[:, 2, 0])
        + a[:, 0, 2] * (a[:, 1, 0] * a[:, 2, 1] - a[:, 1, 1] * a[:, 2, 0])
    )


def table_gap(params: VecParams, env: Tensor) -> tuple[Tensor, Tensor]:
    """``z - R``: the ball's bottom over the table plane."""

    radius = params.radius[env]
    gvec = torch.zeros((*radius.shape, 3), dtype=radius.dtype, device=radius.device)
    gvec[:, 2] = 1.0
    return gvec, -radius


def plane_gap(direction: Tensor) -> tuple[Tensor, Tensor]:
    """``-x * direction``: non-negative before the ball crosses the net plane moving along ``direction``."""

    gvec = torch.zeros((*direction.shape, 3), dtype=direction.dtype, device=direction.device)
    gvec[:, 0] = -direction
    return gvec, torch.zeros_like(direction)


@dataclass(frozen=True)
class Bounce:
    """
    The predicted own-half bounce (``Rally.predict_bounce``).

    :ivar ok: The ball's next table touch is on the receiver's half, before the net or the floor stops it.
    :ivar t: Time of the bounce from the prediction's start, s.
    :ivar after: Ball state just after the bounce (the table impact law).
    """

    ok: Tensor
    t: Tensor
    after: Tensor


def table_bounce(params: VecParams, state: Tensor, env: Tensor) -> Tensor:
    """The ball state after a table-top touch at ``state`` (normal +Z, the table's impact law)."""

    n = torch.zeros_like(state[:, 0:3])
    n[:, 2] = 1.0
    impact = resolve(
        state[:, 3:6],
        state[:, 6:9],
        n,
        torch.zeros_like(n),
        torch.full((state.shape[0],), MAT_TABLE, dtype=torch.long, device=state.device),
        env,
        params,
        effective_mass=torch.full_like(state[:, 0], float("inf")),
    )
    return torch.cat([state[:, 0:3], impact.velocity, impact.spin], -1)


def net_crossing(params: VecParams, y0: Tensor, env: Tensor, horizon: float | Tensor, *, steps: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """
    Where each ball crosses the net plane within ``horizon``: ``(crosses, t, state)``; ``crosses`` is false for a
    ball already past it or never reaching it.
    """

    direction = torch.sign(y0[:, 3])
    direction = torch.where(direction == 0.0, torch.ones_like(direction), direction)
    root = first_root(params, y0, env, plane_gap(direction), horizon, steps=steps)
    return root.found, root.t, root.state


def net_blocks(params: VecParams, state: Tensor, env: Tensor) -> Tensor:
    """A ball crossing the net plane at ``state`` meets the net (its bottom below the net top, within the span)."""

    radius = params.radius[env]
    return (state[:, 2] < params.net_height[env] + radius) & (state[:, 2] > -radius) & (state[:, 1].abs() <= params.net_half_span[env] + radius)


def predict_bounce(params: VecParams, y: Tensor, env: Tensor, end: Tensor, horizon: float = 1.5, *, steps: int | None = None) -> Bounce:
    """
    Under each row's physics, the ball's next table touch and the state after it, if it is on the receiver's half
    (``end``, the receiver end's x-sign) and nothing (the net, the floor, the table's outside) stops it first.
    """

    land = first_root(params, y, env, table_gap(params, env), horizon, steps=steps)
    crosses, t_net, at_net = net_crossing(params, y, env, horizon, steps=steps)
    blocked = crosses & (t_net < land.t) & net_blocks(params, at_net, env)
    p = land.state
    on_table = (p[:, 0].abs() <= params.table_half_length[env]) & (p[:, 1].abs() <= params.table_half_width[env])
    own = torch.sign(p[:, 0]) == end
    ok = land.found & ~blocked & on_table & own
    after = table_bounce(params, p, env)
    return Bounce(ok, land.t, after)


def path_grid(params: VecParams, start: Tensor, env: Tensor, first: float, step: float, count: int) -> Tensor:
    """States at ``first + k * step`` (``k < count``) after ``start``: ``(M, count, 9)`` (RK4 at ``step``)."""

    c = Coeffs.of(params, env)

    sub = max(1, round(first / step))
    y = start
    for _ in range(sub):
        y = rk4(y, first / sub, c)
    out = [y]
    for _ in range(count - 1):
        y = rk4(y, step, c)
        out.append(y)
    return torch.stack(out, 1)


def flight_at(params: VecParams, start: Tensor, env: Tensor, duration: Tensor, steps: int) -> Tensor:
    """Free flight of each row for its ``duration`` in ``steps`` equal RK4 steps."""

    c = Coeffs.of(params, env)

    h = duration / steps
    y = start
    for _ in range(steps):
        y = rk4(y, h, c)
    return y


@dataclass(frozen=True)
class Launch:
    """
    A launch solve (``flight.solve_launch``): the velocity that lands on the target after the flight time.

    :ivar velocity: ``(M, 3)``.
    :ivar residual: ``|p(T) - target|`` with the target on the table plane, m.
    :ivar ok: Converged, clears the net, lands on the far half.
    :ivar clearance: Ball bottom over the net top where it crosses, m.
    :ivar speed_at_net: m/s.
    """

    velocity: Tensor
    residual: Tensor
    ok: Tensor
    clearance: Tensor
    speed_at_net: Tensor


SHOOT_STEPS = 12
SHOOT_ITERATIONS = 5
SHOOT_PROBE = 1e-4


def solve_launch(
    params: VecParams,
    env: Tensor,
    position: Tensor,
    spin: Tensor,
    target: Tensor,
    flight_s: Tensor,
    guess: Tensor,
    *,
    tolerance: float = 0.05,
    steps: int | None = None,
) -> Launch:
    """
    The launch velocity whose free flight from ``position`` with ``spin`` puts the ball's bottom on the table plane
    at ``target`` (x, y) after ``flight_s``: Newton on the 3-vector ``p(T) - (x*, y*, R)`` with a forward-difference
    Jacobian (the generator's damped Gauss-Newton on landing and time solves the same equations).
    """

    m = position.shape[0]
    radius = params.radius[env]
    goal = torch.stack([target[:, 0], target[:, 1], radius], -1)
    v = guess.clone()
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    env4 = env.repeat(4)
    for _ in range(SHOOT_ITERATIONS):
        probes = torch.cat([v, v + SHOOT_PROBE * eye[0], v + SHOOT_PROBE * eye[1], v + SHOOT_PROBE * eye[2]], 0)
        start = torch.cat([position.repeat(4, 1), probes, spin.repeat(4, 1)], -1)
        end = flight_at(params, start, env4, flight_s.repeat(4), SHOOT_STEPS)
        p = end[:, 0:3].reshape(4, m, 3)
        f = p[0] - goal
        jac = torch.stack([(p[1] - p[0]), (p[2] - p[0]), (p[3] - p[0])], -1) / SHOOT_PROBE
        finite = torch.isfinite(jac).all(-1).all(-1)
        clean = torch.where(finite[:, None, None], jac, eye.expand(m, 3, 3))
        good = finite & (_det3(clean).abs() > 1e-12)
        safe = torch.where(good[:, None, None], clean, eye.expand(m, 3, 3))
        step = mat_vec(inv3(safe), f)
        v = torch.where(good[:, None], v - step, v)
    start = torch.cat([position, v, spin], -1)
    end = flight_at(params, start, env, flight_s, SHOOT_STEPS)
    residual = torch.sqrt(((end[:, 0:3] - goal) ** 2).sum(-1))
    direction = torch.sign(target[:, 0] - position[:, 0])
    root = first_root(params, start, env, plane_gap(direction), flight_s, steps=steps)
    at = root.state
    clearance = torch.where(root.found, at[:, 2] - radius - params.net_height[env], torch.full_like(residual, float("nan")))
    speed = torch.where(root.found, torch.sqrt((at[:, 3:6] ** 2).sum(-1)), torch.full_like(residual, float("nan")))
    blocked = root.found & net_blocks(params, at, env)
    far = torch.sign(target[:, 0]) == direction
    ok = torch.isfinite(residual) & (residual <= tolerance) & ~blocked & far & torch.isfinite(v).all(-1)
    return Launch(v, residual, ok, clearance, speed)


@dataclass(frozen=True)
class Landing:
    """A ball's flight to its first touch of the table plane (``flight.fly``, for intended targets and descriptions)."""

    landing: Tensor
    t: Tensor
    found: Tensor
    on_far_half: Tensor
    net: Tensor
    clearance: Tensor
    speed_at_net: Tensor


def fly(params: VecParams, start: Tensor, env: Tensor, horizon: float = 1.6, *, steps: int | None = None) -> Landing:
    """Fly to the first touch of the table plane, noting the net crossing (the far half is the one ``v_x`` points to)."""

    radius = params.radius[env]
    land = first_root(params, start, env, table_gap(params, env), horizon, steps=steps)
    direction = torch.sign(start[:, 3])
    direction = torch.where(direction == 0.0, torch.ones_like(direction), direction)
    net = first_root(params, start, env, plane_gap(direction), horizon, steps=steps)
    before = net.found & (net.t < land.t)
    blocked = before & net_blocks(params, net.state, env)
    p = land.state
    on_table = (p[:, 0].abs() <= params.table_half_length[env]) & (p[:, 1].abs() <= params.table_half_width[env])
    far = land.found & on_table & (p[:, 0] * direction > 0.0) & ~blocked
    clearance = torch.where(before, net.state[:, 2] - radius - params.net_height[env], torch.full_like(land.t, float("nan")))
    speed = torch.where(before, torch.sqrt((net.state[:, 3:6] ** 2).sum(-1)), torch.full_like(land.t, float("nan")))
    return Landing(p[:, 0:2], land.t, land.found, far, blocked, clearance, speed)


__all__ = [
    "Bounce",
    "Landing",
    "Launch",
    "Root",
    "compile_status",
    "first_root",
    "flight_at",
    "fly",
    "net_blocks",
    "net_crossing",
    "path_grid",
    "predict_bounce",
    "propagate",
    "solve_launch",
    "table_bounce",
]
