"""
The strike, batched: the paddle state at contact that turns an incoming ball into the chosen outgoing one.

Ports of the offline data generator's strike and the reference inverse
(:mod:`lib.physics.reference.inverse`), row by row the same equations:

- :func:`paddle_velocity_for`: the stick-regime inverse of the reference impact law (point-mass or rigid blade);
- :func:`normal_for_spin`: damped Newton on the face normal's two angles about the impulse so the outgoing ball
  carries the drawn topspin and sidespin;
- :func:`plan_strike`: the rigid-blade strike (blade centre, orientation with the donor's grip roll, arm-pivot
  angular velocity) and its forward check through :func:`lib.physics.vectorized.impact.resolve`;
- :func:`execute`: the generator's execution noise (timing, face tilt, blade speed, position on the face).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.geometry import MAT_RUBBER
from lib.physics.vectorized.impact import inv3, mat_vec, resolve
from lib.physics.vectorized.params import VecParams

from .ops import (
    blade_rotation,
    cross,
    dot,
    frame_exp,
    frame_log,
    norm,
    quat_from_matrix,
    rotate_about,
    rotation_from_rotvec,
    strike_frame,
    unit,
)

Tensor = Any

APPROACH_BISECTIONS = 48


def _interp(x: Tensor, speeds: Tensor, values: Tensor) -> Tensor:
    """``numpy.interp`` on one shared curve (constant beyond the ends), rows of ``x``."""

    count = speeds.shape[0]
    j = ((speeds[None, :] <= x[:, None]).sum(1) - 1).clamp(0, count - 2)
    s0, s1, v0, v1 = speeds[j], speeds[j + 1], values[j], values[j + 1]
    w = ((x - s0) / (s1 - s0)).clamp(0.0, 1.0)
    return v0 + w * (v1 - v0)


def approach_speed(target: Tensor, scale: Tensor, speeds: Tensor, values: Tensor) -> Tensor:
    """
    The approach speed ``a`` with ``a (1 + scale * c(a)) = target`` for a piecewise-linear ``c`` held constant beyond
    its knots (the rubber's restitution curve, scaled per environment): exact, per segment a quadratic.
    """

    count = speeds.shape[0]
    f = speeds[None, :] * (1.0 + scale[:, None] * values[None, :])
    j = (f <= target[:, None]).sum(1)
    seg = (j - 1).clamp(0, count - 2)
    s0, s1, c0, c1 = speeds[seg], speeds[seg + 1], values[seg], values[seg + 1]
    k = (c1 - c0) / (s1 - s0)
    a_coef = scale * k
    b_coef = 1.0 + scale * (c0 - k * s0)
    mid = 2.0 * target / (b_coef + torch.sqrt((b_coef * b_coef + 4.0 * a_coef * target).clamp_min(0.0)))
    low = target / (1.0 + scale * values[0])
    high = target / (1.0 + scale * values[-1])
    return torch.where(j == 0, low, torch.where(j >= count, high, mid))


def _approach(law: Any, env: Tensor, target_n: Tensor) -> Tensor:
    curve = law.restitution_curve
    if curve is None:
        return target_n / (1.0 + law.restitution[env])
    scale = law.restitution[env] / curve.reference if curve.reference is not None else torch.ones_like(target_n)
    return approach_speed(target_n, scale, curve.speeds, curve.values)


def _skew(r: Tensor) -> Tensor:
    zero = torch.zeros_like(r[:, 0])
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    return torch.stack([torch.stack([zero, -z, y], -1), torch.stack([z, zero, -x], -1), torch.stack([-y, x, zero], -1)], -2)


def paddle_velocity_for(
    params: VecParams, env: Tensor, v_in: Tensor, spin_in: Tensor, v_out: Tensor, normal: Tensor, offset: Tensor | None = None, inertia: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """
    The face velocity at the contact point that turns ``v_in`` into ``v_out`` under the rubber's stick law
    (:func:`lib.physics.reference.inverse.paddle_velocity_for`): ``(u_s, ok)``; ``ok`` is false when the impulse does
    not push along the normal or lies outside the friction cone.
    """

    law = params.rubber
    m = params.mass[env]
    k = params.inertia_ratio[env]
    radius = params.radius[env]
    mass = params.paddle_mass[env]
    n = unit(normal)
    impulse = m[:, None] * (v_out - v_in)
    p_n = dot(impulse, n)
    eye = torch.eye(3, dtype=n.dtype, device=n.device).expand(n.shape[0], 3, 3)
    nn = n[:, :, None] * n[:, None, :]
    compliance = eye / m[:, None, None] + (eye - nn) / (k * m)[:, None, None] + eye / mass[:, None, None]
    if offset is not None and inertia is not None:
        skew = _skew(offset)
        compliance = compliance + skew.transpose(-1, -2) @ inv3(inertia) @ skew
    target = mat_vec(compliance, impulse)
    target_n = dot(target, n)
    # a (1 + e(a)) = target_n: exact on the piecewise-linear restitution curve.
    approach = _approach(law, env, target_n.clamp_min(0.0))
    p_t = impulse - p_n[:, None] * n
    mu = law.friction_at(approach.clamp_min(0.0), env)
    ok = (p_n > 0.0) & (target_n > 0.0) & (norm(p_t) <= mu * p_n * (1.0 + 1e-12))
    u_t = -(target - target_n[:, None] * n) / (1.0 + law.tangential_restitution[env])[:, None]
    relative = -approach[:, None] * n + u_t
    contact_point = v_in + cross(spin_in, -radius[:, None] * n)
    return contact_point - relative, ok


def _spin_core(
    angles: Tensor,
    d: Tensor,
    e1: Tensor,
    e2: Tensor,
    ball_in: Tensor,
    v_out: Tensor,
    target: Tensor,
    m: Tensor,
    k: Tensor,
    radius: Tensor,
    mass: Tensor,
    inertia: Tensor,
    e_t: Tensor,
    s_n: Tensor,
    mu: Tensor,
    scale: Tensor,
    speeds: Tensor,
    values: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    One evaluation of the normal solve's residual: the normal at ``angles``, the point-mass stick inverse, the
    point-mass rubber impact (``resolve``'s point branch), and the outgoing spin's components less the target.
    """

    angle = torch.sqrt(angles[:, 0] ** 2 + angles[:, 1] ** 2)
    small = angle < 1e-12
    safe = torch.where(small, torch.ones_like(angle), angle)
    axis = (angles[:, 0:1] * e1 + angles[:, 1:2] * e2) / safe[:, None]
    n = torch.where(small[:, None], d, torch.cos(angle)[:, None] * d + torch.sin(angle)[:, None] * axis)
    v_in, w_in = ball_in[:, 3:6], ball_in[:, 6:9]
    impulse = m[:, None] * (v_out - v_in)
    p_n = (impulse * n).sum(-1)
    kp = impulse / m[:, None] + (impulse - p_n[:, None] * n) / (k * m)[:, None] + impulse / mass[:, None]
    target_n = (kp * n).sum(-1)
    a = approach_speed(target_n.clamp_min(0.0), scale, speeds, values)
    p_t = impulse - p_n[:, None] * n
    ok = (p_n > 0.0) & (target_n > 0.0) & (torch.sqrt((p_t * p_t).sum(-1)) <= mu * p_n * (1.0 + 1e-12))
    u_t = -(kp - target_n[:, None] * n) / (1.0 + e_t)[:, None]
    r_c = -radius[:, None] * n
    u_s = v_in + torch.cross(w_in, r_c, dim=-1) - (-a[:, None] * n + u_t)
    # The forward point-mass impact with that face velocity.
    u = v_in + torch.cross(w_in, r_c, dim=-1) - u_s
    u_n = (u * n).sum(-1)
    u_tan = u - u_n[:, None] * n
    slip = torch.sqrt((u_tan * u_tan).sum(-1))
    approach = (-u_n).clamp_min(0.0)
    e = (_interp(approach, speeds, values) * scale).clamp(0.0, 1.0)
    m_n = 1.0 / (1.0 / m + 1.0 / mass)
    m_t = 1.0 / (1.0 / m + 1.0 / mass + 1.0 / (k * m))
    pn = -(1.0 + e) * m_n * u_n
    stick = (1.0 + e_t) * m_t * slip <= mu * pn
    safe_slip = torch.where(slip > 0.0, slip, torch.ones_like(slip))
    pt = torch.where(stick[:, None], -(1.0 + e_t)[:, None] * m_t[:, None] * u_tan, -(mu * pn)[:, None] * (u_tan / safe_slip[:, None]))
    w_out = w_in + torch.cross(r_c, pt, dim=-1) / inertia[:, None]
    w_out = w_out - ((1.0 - s_n) * (w_in * n).sum(-1))[:, None] * n
    w_out = torch.where((u_n < 0.0)[:, None], w_out, w_in)
    vhat = v_out / torch.sqrt((v_out * v_out).sum(-1, keepdim=True))
    ez = torch.zeros_like(vhat)
    ez[:, 2] = 1.0
    s1 = torch.cross(ez, vhat, dim=-1)
    s1 = s1 / torch.sqrt((s1 * s1).sum(-1, keepdim=True)).clamp_min(1e-12)
    s2 = torch.cross(vhat, s1, dim=-1)
    residual = torch.stack([(w_out * s1).sum(-1), (w_out * s2).sum(-1)], -1) - target
    return residual, ok & ((n * d).sum(-1) > 0.05)


def _rubber_rows(params: VecParams, env: Tensor) -> tuple[Tensor, ...] | None:
    """The point-mass rubber law's per-row coefficients for :func:`_spin_core` (``None``: a law it does not cover)."""

    law = params.rubber
    curve = law.restitution_curve
    if law.friction_curve is not None or law.law != "rigid":
        return None
    if curve is None:
        speeds = torch.arange(2, dtype=params.dtype, device=params.device)
        values = torch.ones(2, dtype=params.dtype, device=params.device)
        scale = law.restitution[env]
    else:
        speeds, values = curve.speeds, curve.values
        scale = law.restitution[env] / curve.reference if curve.reference is not None else torch.ones_like(law.restitution[env])
    return (
        params.mass[env],
        params.inertia_ratio[env],
        params.radius[env],
        params.paddle_mass[env],
        params.inertia[env],
        law.tangential_restitution[env],
        law.normal_spin_retention[env],
        law.friction[env],
        scale,
        speeds,
        values,
    )


NORMAL_ITERATIONS = 8
NORMAL_PROBE = 1e-3


def _newton_body(
    ab: Tensor,
    current: Tensor,
    ok: Tensor,
    active: Tensor,
    d: Tensor,
    e1: Tensor,
    e2: Tensor,
    ball_in: Tensor,
    v_out: Tensor,
    target: Tensor,
    rubber: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One damped Newton step of :func:`normal_for_spin` (forward-difference Jacobian, step capped at 0.3 rad)."""

    active = active & (torch.sqrt((current * current).sum(-1)) >= 0.5)
    m = ab.shape[0]
    probe = torch.cat([ab, ab], 0)
    bump = torch.zeros_like(probe)
    bump[:m, 0] = NORMAL_PROBE
    bump[m:, 1] = NORMAL_PROBE
    rubber2 = tuple(torch.cat([r, r], 0) for r in rubber[:9]) + tuple(rubber[9:])
    moved, valid2 = _spin_core(
        probe + bump,
        torch.cat([d, d], 0),
        torch.cat([e1, e1], 0),
        torch.cat([e2, e2], 0),
        torch.cat([ball_in, ball_in], 0),
        torch.cat([v_out, v_out], 0),
        torch.cat([target, target], 0),
        *rubber2,
    )
    j0 = (moved[:m] - current) / NORMAL_PROBE
    j1 = (moved[m:] - current) / NORMAL_PROBE
    probe_ok = valid2[:m] & valid2[m:]
    ok = ok & ~(active & ~probe_ok)
    active = active & probe_ok
    a = (j0 * j0).sum(-1) + 1e-6
    b = (j0 * j1).sum(-1)
    c = (j1 * j1).sum(-1) + 1e-6
    r0 = (j0 * current).sum(-1)
    r1 = (j1 * current).sum(-1)
    det = a * c - b * b
    det = torch.where(active & (det != 0.0), det, torch.ones_like(det))
    step = -torch.stack([(c * r0 - b * r1) / det, (a * r1 - b * r0) / det], -1)
    size = torch.sqrt((step * step).sum(-1)).clamp_min(1e-12)
    step = step * torch.clamp(0.3 / size, max=1.0)[:, None]
    candidate = ab + step
    moved_c, valid_c = _spin_core(candidate, d, e1, e2, ball_in, v_out, target, *rubber)
    better = active & valid_c & (torch.sqrt((moved_c * moved_c).sum(-1)) < torch.sqrt((current * current).sum(-1)))
    ab = torch.where(better[:, None], candidate, ab)
    current = torch.where(better[:, None], moved_c, current)
    return ab, current, ok, active & better


def normal_for_spin(params: VecParams, env: Tensor, ball_in: Tensor, v_out: Tensor, spin_target: Tensor, start: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    ``strike.normal_for_spin``: the face normal giving the outgoing ball ``spin_target`` (topspin, sidespin about
    ``v_out``). Damped Newton on its angles about the impulse from ``start``, steps capped at 0.3 rad, a step kept only
    when it lowers the residual.

    :return: ``(normal, residual rad/s, ok)``; ``ok`` false where the generator returns ``None``.
    """

    from .ballistics import compiled

    frame = strike_frame(v_out - ball_in[:, 3:6])
    d, e1, e2 = frame
    rubber = _rubber_rows(params, env)
    if rubber is None:
        raise NotImplementedError("the batched normal solve covers the rigid rubber law without a friction curve")
    ab = frame_log(frame, start)
    if ab.shape[0] == 0:
        return frame_exp(frame, ab), torch.zeros_like(ab[:, 0]), torch.zeros_like(ab[:, 0], dtype=torch.bool)
    core = compiled(_spin_core)
    step = compiled(_newton_body)
    current, ok = core(ab, d, e1, e2, ball_in, v_out, spin_target, *rubber)
    active = ok.clone()
    for _ in range(NORMAL_ITERATIONS):
        ab, current, ok, active = step(ab, current, ok, active, d, e1, e2, ball_in, v_out, spin_target, rubber)
    return frame_exp(frame, ab), norm(current), ok


@dataclass
class Strike:
    """A planned paddle state at contact and the ball it produces (``strike.Strike``), rows first."""

    time: Tensor
    ball_in: Tensor
    position: Tensor
    orientation: Tensor
    """``(w, x, y, z)``, the game's blade axes."""
    rotation: Tensor
    """The reference blade rotation (columns: handle, across, forehand normal)."""
    velocity: Tensor
    angular_velocity: Tensor
    normal: Tensor
    face: Tensor
    v_out: Tensor
    spin_out: Tensor
    face_velocity: Tensor
    ok: Tensor

    def take(self, index: Tensor) -> Strike:
        return Strike(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def where(self, mask: Tensor, other: Strike) -> Strike:
        out = {}
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            out[f.name] = torch.where(mask.reshape(mask.shape + (1,) * (a.dim() - mask.dim())), a, b)
        return Strike(**out)


def paddle_inertia(params: VecParams, env: Tensor, rotation: Tensor) -> Tensor:
    """The paddle's inertia tensor in the table frame for reference rotations (``R diag(I) R^T``)."""

    diag = torch.stack([params.paddle_inertia_xx[env], params.paddle_inertia_yy[env], params.paddle_inertia_zz[env]], -1)
    return (rotation * diag[:, None, :]) @ rotation.transpose(-1, -2)


def forward(
    params: VecParams, env: Tensor, ball_in: Tensor, position: Tensor, rotation: Tensor, velocity: Tensor, omega: Tensor, normal: Tensor
) -> tuple[Tensor, Tensor]:
    """``strike.forward``: the oracle's rigid-blade impact for a blade touching the ball."""

    radius = params.radius[env]
    point = ball_in[:, 0:3] - radius[:, None] * normal
    offset = point - position
    impact = resolve(
        ball_in[:, 3:6],
        ball_in[:, 6:9],
        normal,
        velocity + cross(omega, offset),
        torch.full((ball_in.shape[0],), MAT_RUBBER, dtype=torch.long, device=ball_in.device),
        env,
        params,
        effective_mass=params.paddle_mass[env],
        offset=offset,
        inertia=paddle_inertia(params, env, rotation),
    )
    return impact.velocity, impact.spin


def plan_strike(
    params: VecParams, env: Tensor, time: Tensor, ball_in: Tensor, v_out: Tensor, normal: Tensor, face: Tensor, shoulder: Tensor, tip: Tensor
) -> Strike:
    """``strike.plan_strike`` with the donor's grip roll (``tip``) and a centre contact."""

    n = unit(normal)
    z_axis = torch.where((face == 0)[:, None], n, -n)
    rotation = blade_rotation(z_axis, -tip)
    radius = params.radius[env]
    point = ball_in[:, 0:3] - radius[:, None] * n
    position = point - params.blade_half_thickness[env][:, None] * n
    offset = point - position
    inertia = paddle_inertia(params, env, rotation)
    u, ok = paddle_velocity_for(params, env, ball_in[:, 3:6], ball_in[:, 6:9], v_out, n, offset, inertia)
    r = position - shoulder
    omega = cross(r, u) / dot(r, r).clamp_min(1e-6)[:, None]
    velocity = u - cross(omega, offset)
    out_v, out_w = forward(params, env, ball_in, position, rotation, velocity, omega, n)
    game = torch.stack([rotation[:, :, 1], -rotation[:, :, 0], rotation[:, :, 2]], -1)
    return Strike(
        time=time,
        ball_in=ball_in,
        position=position,
        orientation=quat_from_matrix(game),
        rotation=rotation,
        velocity=velocity,
        angular_velocity=omega,
        normal=n,
        face=face,
        v_out=out_v,
        spin_out=out_w,
        face_velocity=velocity + cross(omega, offset),
        ok=ok & torch.isfinite(out_v).all(-1),
    )


@dataclass
class ContactCommand:
    """
    The opponent's blade at its planned (executed) contact: the kinematic command the env flies.

    :ivar t: Contact time, s (absolute episode time).
    :ivar position: Blade centre, m.
    :ivar orientation: ``(w, x, y, z)``, game blade axes (``PaddleSegment``).
    :ivar velocity: Blade-centre velocity, m/s.
    :ivar angular_velocity: rad/s.
    :ivar face: The rubber presented (0 forehand, 1 backhand).
    :ivar active: The row swings.
    :ivar tick0, tick1: Body-first realisation (or ``None``): the blade's states ``(t, position, velocity)`` at the
        render ticks around the contact, through which the blade moves as the oracle's hand does (cubic Hermite
        sub-segments through the knot, :mod:`lib.opponent.blade`); rows with a NaN time keep the
        constant twist.
    """

    t: Tensor
    position: Tensor
    orientation: Tensor
    velocity: Tensor
    angular_velocity: Tensor
    face: Tensor
    active: Tensor
    tick0: tuple[Tensor, Tensor, Tensor] | None = None
    tick1: tuple[Tensor, Tensor, Tensor] | None = None


def execute(strike: Strike, radius: Tensor, sd: Tensor, generator: Any = None, paddle_speed_scale: float = 1.0) -> tuple[ContactCommand, Tensor]:
    """
    ``Rally.execute``: the final plan with execution noise. ``sd`` is ``(M, 4)`` (timing s, face direction rad, blade
    speed fraction, position on the face m): the anchors times the noise scale.

    :return: ``(command, noise draws (M, 4) as the shards store them: dt, |tilt|, dv, |dc|)``.
    """

    m = strike.time.shape[0]
    z = torch.randn((m, 6), dtype=strike.time.dtype, device=strike.time.device, generator=generator)
    dt = z[:, 0] * sd[:, 0]
    tilt = z[:, 1:3] * sd[:, 1:2]
    dv = z[:, 3] * sd[:, 2]
    dc = z[:, 4:6] * sd[:, 3:4]
    rotation = strike.rotation
    axis = tilt[:, 0:1] * rotation[:, :, 0] + tilt[:, 1:2] * rotation[:, :, 1]
    point = strike.ball_in[:, 0:3] - radius[:, None] * strike.normal
    turned = rotation_from_rotvec(axis) @ rotation
    position = point + rotate_about(axis, strike.position - point)
    position = position + dc[:, 0:1] * turned[:, :, 0] + dc[:, 1:2] * turned[:, :, 1]
    velocity = strike.velocity * (1.0 + dv)[:, None] * paddle_speed_scale
    game = torch.stack([turned[:, :, 1], -turned[:, :, 0], turned[:, :, 2]], -1)
    command = ContactCommand(
        t=strike.time + dt,
        position=position,
        orientation=quat_from_matrix(game),
        velocity=velocity,
        angular_velocity=strike.angular_velocity * paddle_speed_scale,
        face=strike.face,
        active=strike.ok,
    )
    return command, torch.stack([dt, norm(tilt), dv, norm(dc)], -1)


__all__ = ["ContactCommand", "Strike", "execute", "forward", "normal_for_spin", "paddle_inertia", "paddle_velocity_for", "plan_strike"]
