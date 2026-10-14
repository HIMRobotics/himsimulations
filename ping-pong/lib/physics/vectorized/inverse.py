"""
The paddle impact inverted, batched: :mod:`lib.physics.reference.inverse` over rows.

From an intent (the ball's wanted velocity after the hit) to the face velocity at the contact point, in the stick
regime, where the impact law (:func:`lib.physics.vectorized.impact.resolve`) is linear in the relative contact
velocity:

- the impulse is ``P = m (v_out - v_in)``;
- with the contact compliance ``K`` (:func:`contact_compliance`), the stick law ``K P = -(1 + e) u_n n - (1 + e_t)
  u_t`` gives the relative contact velocity: its normal part ``u_n = -a`` solves ``a (1 + e(a)) = (K P) . n``
  (bisection, as the reference: from ``[target / 2, target]`` until the midpoint stops moving), its tangential part
  is ``u_t = -(K P)_t / (1 + e_t)``;
- the face velocity is ``u_s = v_c - u``, ``v_c = v_in + w_in x (-R n)``.

A row is ``valid`` when the impulse pushes the ball out along the normal and lies inside the friction cone (the
reference returns ``None`` otherwise). Everything is fixed-shape and sync-free; :class:`GraphedInverse` replays it as
one CUDA graph.
"""

from __future__ import annotations

from typing import Any

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.graphs import GraphedFunction, compiler
from lib.physics.vectorized.impact import _skew, inv3, mat_mul, mat_vec
from lib.physics.vectorized.params import VecParams

Tensor = Any


def contact_compliance(
    params: VecParams, env: Tensor, normal: Tensor, effective_mass: Tensor, offset: Tensor | None = None, inertia: Tensor | None = None
) -> Tensor:
    """
    The contact-point compliance ``K`` of the impact law, ``(M, 3, 3)`` (:func:`lib.physics.reference.inverse.contact_compliance`).

    :param env: Environment of each row.
    :param normal: Unit contact normals, ``(M, 3)``.
    :param effective_mass: Surface masses, kg (``inf``: immovable), ``(M,)``.
    :param offset: Contact points from the paddle's centre of mass, ``(M, 3)`` (with ``inertia``).
    :param inertia: The paddle's inertia tensors in the table frame, ``(M, 3, 3)``.
    """

    if (offset is None) != (inertia is None):
        raise ValueError("offset and inertia go together")
    m = params.mass[env]
    k = params.inertia_ratio[env]
    eye = torch.eye(3, dtype=normal.dtype, device=normal.device).expand(normal.shape[0], 3, 3)
    nn = normal[:, :, None] * normal[:, None, :]
    k_ball = eye / m[:, None, None] + (eye - nn) / (k * m)[:, None, None]
    finite = torch.isfinite(effective_mass)
    inv_mass = torch.where(finite, 1.0 / torch.where(finite, effective_mass, torch.ones_like(effective_mass)), torch.zeros_like(effective_mass))
    k_surface = inv_mass[:, None, None] * eye
    if offset is not None and inertia is not None:
        skew = _skew(offset)
        k_surface = k_surface + mat_mul(mat_mul(skew.transpose(-1, -2), inv3(inertia)), skew)
    return k_ball + k_surface


def paddle_velocity_for(
    params: VecParams,
    env: Tensor,
    v_in: Tensor,
    spin_in: Tensor,
    v_out: Tensor,
    normal: Tensor,
    effective_mass: Tensor,
    offset: Tensor | None = None,
    inertia: Tensor | None = None,
    iterations: int = 64,
) -> tuple[Tensor, Tensor]:
    """
    The face velocity at the contact point that turns ``v_in`` into ``v_out`` through the rubber's impact law.

    :param env: Environment of each row (its ball and rubber).
    :param v_in: Ball velocities before the hit, ``(M, 3)``.
    :param spin_in: Ball spins before the hit, ``(M, 3)``.
    :param v_out: Wanted ball velocities after the hit, ``(M, 3)``.
    :param normal: Face normals from the face into the ball, ``(M, 3)`` (normalised here).
    :param effective_mass: Paddle effective masses, kg, ``(M,)``.
    :param offset: Contact points from the blade centre, ``(M, 3)``: the paddle as a free rigid body (with ``inertia``).
    :param inertia: The paddle's inertia tensors in the table frame, ``(M, 3, 3)``.
    :param iterations: Bisection steps for the approach speed (the midpoint stops moving after about 53 in float64).
    :return: ``(face_velocity (M, 3), valid (M,))``; invalid rows hold no meaningful velocity.
    """

    if params.rubber.law != "rigid":
        raise ValueError("only the rigid impact law can be inverted for a face velocity")
    n = normal / torch.sqrt((normal * normal).sum(-1, keepdim=True))
    impulse = params.mass[env][:, None] * (v_out - v_in)
    p_n = (impulse * n).sum(-1)
    target = mat_vec(contact_compliance(params, env, n, effective_mass, offset, inertia), impulse)
    target_n = (target * n).sum(-1)
    positive = (p_n > 0.0) & (target_n > 0.0)
    safe_target = torch.where(positive, target_n, torch.ones_like(target_n))
    # The reference's bisection: a (1 + e(a)) = target on [target / 2, target], until the midpoint stops moving.
    lo, hi = 0.5 * safe_target, safe_target
    moving = torch.ones_like(positive)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        moving = moving & (mid > lo) & (mid < hi)
        below = mid * (1.0 + params.rubber.restitution_at(mid, env)) < safe_target
        lo = torch.where(moving & below, mid, lo)
        hi = torch.where(moving & ~below, mid, hi)
    approach = 0.5 * (lo + hi)
    p_t = impulse - p_n[:, None] * n
    inside = torch.sqrt((p_t * p_t).sum(-1)) <= params.rubber.friction_at(approach, env) * p_n * (1.0 + 1e-12)
    u_t = -(target - target_n[:, None] * n) / (1.0 + params.rubber.tangential_restitution[env])[:, None]
    relative = -approach[:, None] * n + u_t
    contact_point_velocity = v_in + torch.cross(spin_in, -params.radius[env][:, None] * n, dim=-1)
    return contact_point_velocity - relative, positive & inside


def blade_velocity_for(face_velocity: Tensor, angular_velocity: Tensor, offset: Tensor) -> Tensor:
    """The blade-centre velocities whose material points at ``offset`` (from the centre) move at ``face_velocity``."""

    return face_velocity - torch.cross(angular_velocity, offset, dim=-1)


class GraphedInverse:
    """
    :func:`paddle_velocity_for` for ``M`` fixed rows (rigid paddle: offsets and inertias given), one graph per call.

    :param params: The environments' physics.
    :param rows: ``M``, the rows per call.
    :param rigid: Take contact offsets and inertia tensors (the paddle as a free rigid body).
    :param compile: ``torch.compile`` it (default: on CUDA).
    :param graphs: Capture it (default: on CUDA).
    """

    def __init__(self, params: VecParams, rows: int, *, rigid: bool = True, compile: bool | None = None, graphs: bool | None = None) -> None:
        cuda = str(params.device).startswith("cuda")
        self.params = params
        self.rigid = rigid
        self._fn = compiler(cuda if compile is None else bool(compile))(paddle_velocity_for)

        def zeros(*shape: int) -> Tensor:
            return torch.zeros(shape, dtype=params.dtype, device=params.device)

        examples = [torch.zeros(rows, dtype=torch.long, device=params.device), zeros(rows, 3), zeros(rows, 3), zeros(rows, 3), zeros(rows, 3), zeros(rows)]
        examples[4][:, 0] = 1.0
        examples[5] += 1.0
        if rigid:
            examples += [zeros(rows, 3), torch.eye(3, dtype=params.dtype, device=params.device).expand(rows, 3, 3).clone()]
        self._call = GraphedFunction(self._run, examples, graphs=cuda if graphs is None else graphs)

    def _run(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        return self._fn(self.params, *inputs)

    @property
    def setup_s(self) -> tuple[float, float] | None:
        return self._call.setup_s

    def __call__(
        self,
        env: Tensor,
        v_in: Tensor,
        spin_in: Tensor,
        v_out: Tensor,
        normal: Tensor,
        effective_mass: Tensor,
        offset: Tensor | None = None,
        inertia: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """As :func:`paddle_velocity_for`; the results are the graph's buffers."""

        inputs = [env, v_in, spin_in, v_out, normal, effective_mass]
        if self.rigid:
            if offset is None or inertia is None:
                raise ValueError("this inverse was built for the rigid paddle: give offset and inertia")
            inputs += [offset, inertia]
        return self._call(*inputs)


__all__ = ["GraphedInverse", "blade_velocity_for", "contact_compliance", "paddle_velocity_for"]
