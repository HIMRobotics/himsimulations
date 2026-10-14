"""
The opponent's kinematic blade: its contact command flown through :class:`lib.physics.vectorized.BallBatch`.

Within a window around its contact (default +-50 ms, design section 3.4) the blade moves with **constant linear and
angular velocity through the contact state**: ``p(t) = p_c + v (t - t_c)``, ``q(t) = exp(w (t - t_c)) q_c``. A
:class:`lib.physics.vectorized.geometry.BladeTrack` built from the states at any step's two ends reproduces that
motion exactly (a cubic Hermite through two points of a line with the line's slope is the line; the slerp between
two orientations of a constant twist is the twist), so the ball meets the blade at the commanded state whatever the
step grid, and a contact inside a step needs no knot: the ``PaddleSegment`` knot's two sub-segments are the same
line. Outside the window the blade is parked out of reach (below the floor, behind its end) with zero velocity,
teleported between steps (``p0 = p1``), never swept.

Body-first, the oracle's paddle is the hand's: a cubic Hermite per 1/24 s render tick, with the held knot inside its
tick. A command carrying that tick's two states (``ContactCommand.tick0``, ``tick1``) moves its blade centre through
them: cubic Hermite from the first tick state to the knot and from the knot to the second, and on at those states'
velocities outside the tick; the orientation keeps the constant twist. A step's ``BladeTrack`` reproduces each cubic
exactly; across the knot (a change of acceleration) inside one 5 ms step it is a cubic through the step's ends.
"""

from __future__ import annotations

from typing import Any

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.geometry import BladeTrack

from .ops import quat_from_rotvec, quat_multiply
from .strike import ContactCommand

Tensor = Any

WINDOW_S = 0.05


def _hermite(ta: Tensor, pa: Tensor, va: Tensor, tb: Tensor, pb: Tensor, vb: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Position and velocity at ``t`` on the cubic Hermite from ``(ta, pa, va)`` to ``(tb, pb, vb)`` (rows)."""

    h = (tb - ta).clamp_min(1e-9)[..., None]
    u = ((t - ta)[..., None] / h).clamp(0.0, 1.0)
    u2, u3 = u * u, u * u * u
    x = (2 * u3 - 3 * u2 + 1) * pa + (u3 - 2 * u2 + u) * h * va + (-2 * u3 + 3 * u2) * pb + (u3 - u2) * h * vb
    v = ((6 * u2 - 6 * u) * pa + (3 * u2 - 4 * u + 1) * h * va + (-6 * u2 + 6 * u) * pb + (3 * u2 - 2 * u) * h * vb) / h
    return x, v


def twist_state(command: ContactCommand, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The commanded blade at times ``t`` (rows): position, orientation (wxyz), velocity, angular velocity."""

    dt = (t - command.t)[..., None]
    position = command.position + command.velocity * dt
    velocity = command.velocity.expand_as(position)
    orientation = quat_multiply(quat_from_rotvec(command.angular_velocity * dt), command.orientation)
    if command.tick0 is not None and command.tick1 is not None:
        t0, p0, v0 = command.tick0
        t1, p1, v1 = command.tick1
        real = torch.isfinite(t0) & torch.isfinite(command.t)
        tc = torch.where(real, command.t, t)
        t0 = torch.where(real, t0, tc - 1.0)
        t1 = torch.where(real, t1, tc + 1.0)
        pl, vl = _hermite(t0, p0, v0, tc, command.position, command.velocity, t)
        pr, vr = _hermite(tc, command.position, command.velocity, t1, p1, v1, t)
        before = (t < t0)[..., None]
        after = (t >= t1)[..., None]
        left = (t < tc)[..., None]
        pos = torch.where(before, p0 + v0 * (t - t0)[..., None], torch.where(after, p1 + v1 * (t - t1)[..., None], torch.where(left, pl, pr)))
        vel = torch.where(before, v0, torch.where(after, v1, torch.where(left, vl, vr)))
        position = torch.where(real[..., None], pos, position)
        velocity = torch.where(real[..., None], vel, velocity)
    return position, orientation, velocity, command.angular_velocity


def parked_pose(end: Tensor, table_half_length: Tensor, table_height: Tensor) -> tuple[Tensor, Tensor]:
    """A blade out of play: 3 m behind its end and 1 m below the floor, flat."""

    position = torch.stack([end * (table_half_length + 3.0), torch.zeros_like(end), -table_height - 1.0], -1)
    orientation = torch.zeros((*end.shape, 4), dtype=end.dtype, device=end.device)
    orientation[..., 0] = 1.0
    return position, orientation


def step_states(
    command: ContactCommand, t0: float, h: float, park_position: Tensor, park_orientation: Tensor, window: float = WINDOW_S
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    The blade's states at the ends of the step ``[t0, t0 + h]``: the twist where the step overlaps the active
    command's window, else the parked pose (both ends, zero velocity).
    """

    rows = command.t.shape
    a = torch.full(rows, t0, dtype=command.t.dtype, device=command.t.device)
    b = a + h
    live = command.active & (b >= command.t - window) & (a <= command.t + window)
    p0, q0, v0, _w = twist_state(command, a)
    p1, q1, v1, _w = twist_state(command, b)
    zero = torch.zeros_like(v0)
    m = live[..., None]
    return (
        torch.where(m, p0, park_position),
        torch.where(m, v0, zero),
        torch.where(m, q0, park_orientation),
        torch.where(m, p1, park_position),
        torch.where(m, v1, zero),
        torch.where(m, q1, park_orientation),
    )


def blade_track(commands: list[ContactCommand], t0: float, h: float, parks: list[tuple[Tensor, Tensor]], window: float = WINDOW_S) -> BladeTrack:
    """Every environment's blades (one per command set, in order) over one step: a ``BladeTrack`` of ``(N, B, ...)``."""

    parts = [step_states(c, t0, h, p, q, window) for c, (p, q) in zip(commands, parks, strict=True)]
    stacked = [torch.stack([part[i] for part in parts], 1) for i in range(6)]
    p0, v0, q0, p1, v1, q1 = stacked
    return BladeTrack.from_states(p0, v0, q0, p1, v1, q1, h)


def idle_command(n: int, dtype: Any, device: Any) -> ContactCommand:
    """No swing: every blade parked."""

    nan = torch.full((n,), float("nan"), dtype=dtype, device=device)
    return ContactCommand(
        t=nan.clone(),
        position=torch.zeros((n, 3), dtype=dtype, device=device),
        orientation=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device).expand(n, 4).clone(),
        velocity=torch.zeros((n, 3), dtype=dtype, device=device),
        angular_velocity=torch.zeros((n, 3), dtype=dtype, device=device),
        face=torch.zeros(n, dtype=torch.long, device=device),
        active=torch.zeros(n, dtype=torch.bool, device=device),
    )


__all__ = ["WINDOW_S", "blade_track", "idle_command", "parked_pose", "step_states", "twist_state"]
