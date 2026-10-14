"""
The impact law, batched: :func:`lib.physics.reference.impact.resolve_impact` over rows.

Each row is one ball against one surface: a static one (infinite mass; the point-mass branch), a paddle (finite
effective mass with the contact offset and the paddle's inertia: the off-centre rigid-body branch, the 3x3 compliance
solved per row), or the net mesh (the catch law). Materials are coded as in :mod:`lib.physics.vectorized.geometry`
and evaluated per environment, speed curves included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.geometry import MAT_FIXTURE, MAT_FLOOR, MAT_NET_CORD, MAT_NET_MESH, MAT_POST, MAT_RIM, MAT_RUBBER, MAT_TABLE
from lib.physics.vectorized.params import Material, VecParams

Tensor = Any


@dataclass(frozen=True)
class ImpactBatch:
    """
    Impact outcomes, one per row.

    :ivar velocity: Ball velocity after, ``(M, 3)`` (unchanged where not approaching).
    :ivar spin: Ball spin after, ``(M, 3)``.
    :ivar approaching: The contact was approaching, so an impulse applied (the oracle's ``separating`` otherwise).
    :ivar separation_speed: Normal separation speed after, relative to the surface after, m/s.
    :ivar stick: The tangential target lay inside the friction cone.
    """

    velocity: Tensor
    spin: Tensor
    approaching: Tensor
    separation_speed: Tensor
    stick: Tensor


def _materials(params: VecParams) -> dict[int, Material]:
    return {
        MAT_TABLE: params.table,
        MAT_RUBBER: params.rubber,
        MAT_NET_CORD: params.net,
        MAT_NET_MESH: params.net,
        MAT_FLOOR: params.floor,
        MAT_POST: params.post,
        MAT_FIXTURE: params.fixture,
        MAT_RIM: params.rim,
    }


def mat_vec(a: Tensor, x: Tensor) -> Tensor:
    """``a @ x`` for ``(M, 3, 3)`` by ``(M, 3)``, written out so that a compiler fuses it (no batched GEMM)."""

    return (a * x[:, None, :]).sum(-1)


def mat_mul(a: Tensor, b: Tensor) -> Tensor:
    """``a @ b`` for ``(M, 3, 3)`` pairs, written out (no batched GEMM)."""

    return (a[:, :, :, None] * b[:, None, :, :]).sum(-2)


def inv3(a: Tensor) -> Tensor:
    """
    The inverse of each ``(3, 3)`` matrix by its adjugate: no solver library and no error check, so it runs inside a
    captured CUDA graph without a host sync. Every matrix here (inertia tensors, contact compliances) is symmetric
    positive definite and well conditioned.
    """

    a00, a01, a02 = a[:, 0, 0], a[:, 0, 1], a[:, 0, 2]
    a10, a11, a12 = a[:, 1, 0], a[:, 1, 1], a[:, 1, 2]
    a20, a21, a22 = a[:, 2, 0], a[:, 2, 1], a[:, 2, 2]
    c00, c01, c02 = a11 * a22 - a12 * a21, a12 * a20 - a10 * a22, a10 * a21 - a11 * a20
    c10, c11, c12 = a02 * a21 - a01 * a22, a00 * a22 - a02 * a20, a01 * a20 - a00 * a21
    c20, c21, c22 = a01 * a12 - a02 * a11, a02 * a10 - a00 * a12, a00 * a11 - a01 * a10
    det = a00 * c00 + a01 * c01 + a02 * c02
    adjugate = torch.stack([torch.stack([c00, c10, c20], -1), torch.stack([c01, c11, c21], -1), torch.stack([c02, c12, c22], -1)], -2)
    return adjugate / det[:, None, None]


def _skew(r: Tensor) -> Tensor:
    zero = torch.zeros_like(r[:, 0])
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    return torch.stack([torch.stack([zero, -z, y], -1), torch.stack([z, zero, -x], -1), torch.stack([-y, x, zero], -1)], -2)


def resolve(
    v: Tensor,
    w: Tensor,
    normal: Tensor,
    surface_velocity: Tensor,
    material: Tensor,
    env: Tensor,
    params: VecParams,
    *,
    effective_mass: Tensor,
    offset: Tensor | None = None,
    inertia: Tensor | None = None,
    compliance: Tensor | None = None,
    compliant: Tensor | None = None,
) -> ImpactBatch:
    """
    Resolve one impact per row.

    :param v: Ball velocity before, ``(M, 3)``.
    :param w: Ball spin before, ``(M, 3)``.
    :param normal: Unit normal from the surface into the ball, ``(M, 3)``.
    :param surface_velocity: Velocity of the surface's material point at the contact, ``(M, 3)``.
    :param material: Material code per row, ``(M,)``.
    :param env: Environment per row.
    :param params: The batch's physics.
    :param effective_mass: Surface mass per row, kg (``inf`` for static surfaces), ``(M,)``.
    :param offset: Contact point from a rigid surface's centre of mass, ``(M, 3)``; with ``inertia`` (rows with an
        infinite mass ignore both).
    :param inertia: That body's inertia tensor in the table frame, ``(M, 3, 3)``.
    :param compliance: The surface's own contact-point compliance (velocity change of its material point per unit
        impulse), ``(M, 3, 3)``, used instead of ``1/M`` and the free body's rotation on the ``compliant`` rows (which
        need a finite ``effective_mass`` and ``offset``/``inertia`` given, any values): an articulated paddle's
        ``J_p M^-1 J_p^T`` at the contact point. With ``K = 1/M + [r]x^T I^-1 [r]x`` it is the rigid branch exactly.
    :param compliant: Rows that use ``compliance``, ``(M,)``.
    """

    m_ball = params.mass[env]
    radius = params.radius[env]
    inertia_ball = params.inertia[env]
    k = params.inertia_ratio[env]
    n = normal
    r_c = -radius[:, None] * n
    u = v + torch.cross(w, r_c, dim=-1) - surface_velocity
    u_n = (u * n).sum(-1)
    u_t = u - u_n[:, None] * n
    slip = torch.sqrt((u_t * u_t).sum(-1))
    approach = (-u_n).clamp_min(0.0)
    approaching = u_n < 0.0

    e = torch.zeros_like(u_n)
    mu = torch.zeros_like(u_n)
    e_t = torch.zeros_like(u_n)
    s_n = torch.ones_like(u_n)
    d_t = torch.zeros_like(u_n)
    d_w = torch.zeros_like(u_n)
    for code, law in _materials(params).items():
        here = material == code
        e = torch.where(here, law.restitution_at(approach, env), e)
        mu = torch.where(here, law.friction_at(approach, env), mu)
        e_t = torch.where(here, law.tangential_restitution[env], e_t)
        s_n = torch.where(here, law.normal_spin_retention[env], s_n)
        d_t = torch.where(here, law.tangential_damping[env], d_t)
        d_w = torch.where(here, law.spin_damping[env], d_w)

    finite = torch.isfinite(effective_mass)
    inv_mass = torch.where(finite, 1.0 / torch.where(finite, effective_mass, torch.ones_like(effective_mass)), torch.zeros_like(effective_mass))
    rigid = finite & (offset is not None and inertia is not None)
    safe_slip = torch.where(slip > 0.0, slip, torch.ones_like(slip))

    # Point-mass surface: closed form.
    m_n = 1.0 / (1.0 / m_ball + inv_mass)
    m_t = 1.0 / (1.0 / m_ball + inv_mass + 1.0 / (k * m_ball))
    p_n_point = -(1.0 + e) * m_n * u_n
    p_stick_mag = (1.0 + e_t) * m_t * slip
    stick_point = p_stick_mag <= mu * p_n_point
    p_t_point = torch.where(stick_point[:, None], -(1.0 + e_t)[:, None] * m_t[:, None] * u_t, -(mu * p_n_point)[:, None] * (u_t / safe_slip[:, None]))

    p_n, p_t, stick = p_n_point, p_t_point, stick_point
    surface_dv = torch.zeros_like(v)
    surface_dw = torch.zeros_like(v)
    if offset is not None and inertia is not None:
        # Rigid-body surface: the full 3x3 compliance.
        eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(v.shape[0], 3, 3)
        safe_inertia = torch.where(rigid[:, None, None], inertia, eye)
        inv_tensor = inv3(safe_inertia)
        skew = _skew(torch.where(rigid[:, None], offset, torch.zeros_like(offset)))
        k_surface = inv_mass[:, None, None] * eye + mat_mul(mat_mul(skew.transpose(-1, -2), inv_tensor), skew)
        if compliance is not None and compliant is not None:
            k_surface = torch.where(compliant[:, None, None], compliance, k_surface)
        nn = n[:, :, None] * n[:, None, :]
        k_ball = eye / m_ball[:, None, None] + (eye - nn) / (k * m_ball)[:, None, None]
        k_total = k_ball + k_surface
        target = -(1.0 + e)[:, None] * u_n[:, None] * n - (1.0 + e_t)[:, None] * u_t
        p_stick = mat_vec(inv3(k_total), target)
        p_stick_n = (p_stick * n).sum(-1)
        p_stick_t = p_stick - p_stick_n[:, None] * n
        p_stick_t_mag = torch.sqrt((p_stick_t * p_stick_t).sum(-1))
        stick_rigid = (p_stick_n > 0.0) & (p_stick_t_mag <= mu * p_stick_n)
        fallback = torch.where(
            (p_stick_t_mag > 0.0)[:, None],
            -p_stick_t / torch.where(p_stick_t_mag > 0.0, p_stick_t_mag, torch.ones_like(p_stick_t_mag))[:, None],
            torch.zeros_like(p_stick_t),
        )
        slip_dir = torch.where((slip > 0.0)[:, None], u_t / safe_slip[:, None], fallback)
        direction = n - mu[:, None] * slip_dir
        denominator = (n * mat_vec(k_total, direction)).sum(-1)
        p_n_slip = -(1.0 + e) * u_n / torch.where(denominator > 0.0, denominator, torch.ones_like(denominator))
        p_n_rigid = torch.where(stick_rigid, p_stick_n, p_n_slip)
        p_t_rigid = torch.where(stick_rigid[:, None], p_stick_t, -(mu * p_n_slip)[:, None] * slip_dir)
        p_n = torch.where(rigid, p_n_rigid, p_n)
        p_t = torch.where(rigid[:, None], p_t_rigid, p_t)
        stick = torch.where(rigid, stick_rigid, stick)
        impulse_rigid = p_n_rigid[:, None] * n + p_t_rigid
        surface_dw = torch.where(rigid[:, None], -mat_vec(inv_tensor, torch.cross(offset, impulse_rigid, dim=-1)), surface_dw)

    impulse = p_n[:, None] * n + p_t
    v_out = v + impulse / m_ball[:, None]
    w_out = w + torch.cross(r_c, p_t, dim=-1) / inertia_ball[:, None]
    w_normal = (w * n).sum(-1)
    w_out = w_out - ((1.0 - s_n) * w_normal)[:, None] * n
    surface_dv = -impulse * inv_mass[:, None]
    u_s_out = surface_velocity + surface_dv
    if offset is not None:
        u_s_out = u_s_out + torch.where(rigid[:, None], torch.cross(surface_dw, offset, dim=-1), torch.zeros_like(v))

    # The catch law (the net mesh): no contact friction; the net absorbs momentum and spin.
    catch = material == MAT_NET_MESH
    relative = v - surface_velocity
    normal_part = (relative * n).sum(-1)
    tangential = relative - normal_part[:, None] * n
    v_catch = surface_velocity - (e * normal_part)[:, None] * n + (1.0 - d_t)[:, None] * tangential
    w_catch = (1.0 - d_w)[:, None] * w
    v_out = torch.where(catch[:, None], v_catch, v_out)
    w_out = torch.where(catch[:, None], w_catch, w_out)
    if compliance is not None and compliant is not None:
        u_s_out = torch.where(compliant[:, None], surface_velocity - mat_vec(compliance, impulse), u_s_out)
    u_s_out = torch.where(catch[:, None], surface_velocity, u_s_out)

    u_out = v_out + torch.cross(w_out, r_c, dim=-1) - u_s_out
    separation = (u_out * n).sum(-1)
    v_out = torch.where(approaching[:, None], v_out, v)
    w_out = torch.where(approaching[:, None], w_out, w)
    return ImpactBatch(v_out, w_out, approaching, torch.where(approaching, separation, u_n), stick)


__all__ = ["ImpactBatch", "inv3", "mat_mul", "mat_vec", "resolve"]
