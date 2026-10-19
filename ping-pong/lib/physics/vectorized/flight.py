"""
Flight, batched: :meth:`lib.physics.reference.flight.FlightModel.derivative` and the RK4 step, over rows.

Each function takes states of shape ``(M, 9)`` and the environment index of every row (``env``; ``None`` when row
``i`` is environment ``i``). The expressions and their order are the scalar oracle's, so in float64 a step agrees
with :meth:`lib.physics.reference.flight.FlightModel.rk4_step` to rounding.
"""

from __future__ import annotations

from typing import Any

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.params import VecParams

Tensor = Any


def _take(value: Tensor, env: Tensor | None) -> Tensor:
    return value if env is None else value[env]


def acceleration(v: Tensor, w: Tensor, params: VecParams, env: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """
    The acceleration (m/s^2) and angular acceleration (rad/s^2) of balls with velocities ``v`` and spins ``w``.

    :param v: ``(M, 3)``.
    :param w: ``(M, 3)``.
    :param params: The batch's physics.
    :param env: Environment of each row, ``(M,)``; ``None`` for rows ``0..N-1``.
    :return: ``(M, 3)`` and ``(M, 3)``.
    """

    gravity = _take(params.gravity, env)
    radius = _take(params.radius, env)
    gain = _take(params.aero_gain, env)
    drag0 = _take(params.drag_coefficient, env)
    drag_slope = _take(params.drag_spin_slope, env)
    slope = _take(params.lift_slope, env)
    c_max = _take(params.lift_c_max, env)
    decay_gain = _take(params.spin_decay_gain, env)
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
    if params.lift_form == "rational":
        lift = slope * s / (1.0 + slope * s / c_max)
    elif params.lift_form == "linear":
        lift = torch.minimum(slope * s, c_max)
    else:
        lift = c_max + torch.zeros_like(s)
    lift = torch.where(s > 0.0, lift, torch.zeros_like(lift))
    magnus = torch.where(spinning, gain * lift * speed / torch.where(spinning, wperp, torch.ones_like(wperp)), torch.zeros_like(wperp))
    ax = ax + magnus * (wy * vz - wz * vy)
    ay = ay + magnus * (wz * vx - wx * vz)
    az = az + magnus * (wx * vy - wy * vx)
    decay = decay_gain * speed
    a = torch.stack([ax, ay, az], -1)
    b = torch.stack([-decay * wx, -decay * wy, -decay * wz], -1)
    return a, b


def rk4_step(y: Tensor, h: Tensor | float, params: VecParams, env: Tensor | None = None) -> Tensor:
    """
    One classical RK4 step per row: :func:`lib.physics.reference.flight.rk4_step` batched.

    :param y: States ``(M, 9)``: position, velocity, spin.
    :param h: Step per row ``(M,)``, or one step for all, s.
    :param params: The batch's physics.
    :param env: Environment of each row.
    :return: ``(M, 9)``.
    """

    steps: Tensor = h if torch.is_tensor(h) else torch.full((y.shape[0],), float(h), dtype=y.dtype, device=y.device)
    h = steps[:, None]
    p, v, w = y[:, 0:3], y[:, 3:6], y[:, 6:9]
    half = 0.5 * h
    a1, b1 = acceleration(v, w, params, env)
    v2, w2 = v + half * a1, w + half * b1
    a2, b2 = acceleration(v2, w2, params, env)
    v3, w3 = v + half * a2, w + half * b2
    a3, b3 = acceleration(v3, w3, params, env)
    v4, w4 = v + h * a3, w + h * b3
    a4, b4 = acceleration(v4, w4, params, env)
    sixth = h / 6.0
    return torch.cat(
        [
            p + sixth * (v + 2.0 * v2 + 2.0 * v3 + v4),
            v + sixth * (a1 + 2.0 * a2 + 2.0 * a3 + a4),
            w + sixth * (b1 + 2.0 * b2 + 2.0 * b3 + b4),
        ],
        -1,
    )


__all__ = ["acceleration", "rk4_step"]
