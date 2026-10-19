"""
The parameter contract as per-environment tensors.

:class:`VecParams` holds every quantity the batched physics reads, one row per
environment, built from :class:`lib.physics.params.BallTableParams` objects
(one shared set, or one per environment for domain randomisation). Derived
quantities (the ball's moment of inertia, the aerodynamic gain, the spin-decay
gain) come from :class:`lib.physics.reference.flight.FlightModel`, so they are
the oracle's numbers. A speed curve's shape is shared by every environment; its
scale follows each environment's scalar, as
:meth:`lib.physics.params.ContactMaterial.restitution_at` does.

:meth:`VecParams.with_values` replaces flat contract scalars
(:data:`lib.physics.params.PARAMETER_SPECS` names) by per-environment tensors
without building parameter objects, for fast resets.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lib.physics.params import BallTableParams, ContactMaterial, SpeedCurve
from lib.physics.reference.flight import FlightModel
from lib.physics.reference.surfaces import NET_CLAMP_HALF_X_M, NET_CLAMP_Y_M, NET_CLAMP_Z_M, NET_POST_OUTSET_M, NET_POST_RADIUS_M, NET_POST_Z_M
from lib.physics.vectorized._torch import torch

Tensor = Any
"""A torch tensor."""


@dataclass(frozen=True)
class Curve:
    """
    A piecewise-linear coefficient of the normal approach speed, held constant beyond its ends (``numpy.interp``).

    :ivar speeds: Knot speeds, m/s, ``(K,)``.
    :ivar values: Values at the knots, ``(K,)``.
    :ivar reference: The scalar the curve was tabulated for (``None``: the curve replaces the scalar).
    """

    speeds: Tensor
    values: Tensor
    reference: float | None

    def __call__(self, x: Tensor) -> Tensor:
        """The curve at speeds ``x``."""

        count = int(self.speeds.shape[0])
        if count == 1:
            return torch.zeros_like(x) + self.values[0]
        upper = torch.searchsorted(self.speeds, x.contiguous(), right=True).clamp(1, count - 1)
        lower = upper - 1
        x0, x1 = self.speeds[lower], self.speeds[upper]
        y0, y1 = self.values[lower], self.values[upper]
        weight = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
        return y0 + weight * (y1 - y0)


@dataclass(frozen=True)
class Material:
    """
    One ball-surface impact law for every environment (:class:`lib.physics.params.ContactMaterial`).

    :ivar restitution: Normal restitution scalar, ``(N,)``.
    :ivar friction: Coulomb friction scalar, ``(N,)``.
    :ivar tangential_restitution: ``e_t``, ``(N,)``.
    :ivar normal_spin_retention: ``s_n``, ``(N,)``.
    :ivar tangential_damping: Catch law ``d_t``, ``(N,)``.
    :ivar spin_damping: Catch law ``d_w``, ``(N,)``.
    :ivar restitution_curve: Shared speed curve, scaled per environment by ``restitution / reference``.
    :ivar friction_curve: Shared speed curve for friction, scaled the same way.
    :ivar law: ``rigid`` or ``catch``.
    """

    restitution: Tensor
    friction: Tensor
    tangential_restitution: Tensor
    normal_spin_retention: Tensor
    tangential_damping: Tensor
    spin_damping: Tensor
    restitution_curve: Curve | None
    friction_curve: Curve | None
    law: str

    def restitution_at(self, approach: Tensor, env: Tensor) -> Tensor:
        """Restitution at approach speeds ``approach`` (m/s) for environments ``env`` (index tensor)."""

        scalar = self.restitution[env]
        curve = self.restitution_curve
        if curve is None:
            return scalar
        scale = scalar / curve.reference if curve.reference is not None else torch.ones_like(scalar)
        return (curve(approach) * scale).clamp(0.0, 1.0)

    def friction_at(self, approach: Tensor, env: Tensor) -> Tensor:
        """Friction at approach speeds ``approach`` (m/s) for environments ``env``."""

        scalar = self.friction[env]
        curve = self.friction_curve
        if curve is None:
            return scalar
        scale = scalar / curve.reference if curve.reference is not None else torch.ones_like(scalar)
        return (curve(approach) * scale).clamp_min(0.0)


MATERIAL_FIELDS = ("table_contact", "rubber_contact", "net_contact", "floor_contact", "post_contact", "fixture_contact", "rim_contact")
"""The contract materials the batched physics uses (no room: the training scene has no walls). ``rim_contact`` is the
blade's rounded edge (an unvalidated judgement prior; :mod:`lib.physics.reference.blade`)."""

_MATERIAL_SCALARS: dict[str, dict[str, str]] = {
    "table_contact": {"restitution": "table_restitution", "friction": "table_friction", "tangential_restitution": "table_tangential_restitution"},
    "rubber_contact": {
        "restitution": "rubber_restitution",
        "friction": "rubber_friction",
        "tangential_restitution": "rubber_tangential_restitution",
        "normal_spin_retention": "rubber_normal_spin_retention",
    },
    "net_contact": {
        "restitution": "net_restitution",
        "friction": "net_friction",
        "tangential_damping": "net_tangential_damping",
        "spin_damping": "net_spin_damping",
    },
    "floor_contact": {"restitution": "floor_restitution", "friction": "floor_friction"},
    "post_contact": {"restitution": "post_restitution", "friction": "post_friction"},
    "fixture_contact": {"restitution": "fixture_restitution", "friction": "fixture_friction"},
    "rim_contact": {"restitution": "rim_restitution", "friction": "rim_friction"},
}
"""Which flat contract names (``PARAMETER_SPECS``) set which material fields."""


def _curve(curve: SpeedCurve | None, device: Any, dtype: Any) -> Curve | None:
    if curve is None:
        return None
    return Curve(torch.tensor(curve.speeds, dtype=dtype, device=device), torch.tensor(curve.values, dtype=dtype, device=device), curve.reference)


def _same_curve(a: SpeedCurve | None, b: SpeedCurve | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.speeds == b.speeds and a.values == b.values and a.reference == b.reference


@dataclass(frozen=True)
class VecParams:
    """
    The physics of ``n`` environments, as tensors on one device and dtype.

    Every attribute below is ``(N,)`` unless stated; the fixtures' geometry (posts and clamps, as in
    :func:`lib.physics.reference.surfaces.net_fixtures`) is fixed. ``blade_edge`` (a model switch, not a tensor) gives
    the blades their rounded edge (:func:`lib.physics.reference.blade.blade_solid_gap`, the oracle's default); False is
    the face-only blade, an A/B and negative control (``dataclasses.replace(params, blade_edge=False)``).
    """

    n: int
    device: Any
    dtype: Any
    lift_form: str
    # Ball and air.
    radius: Tensor
    mass: Tensor
    inertia: Tensor
    inertia_ratio: Tensor
    gravity: Tensor
    aero_gain: Tensor
    drag_coefficient: Tensor
    drag_spin_slope: Tensor
    lift_slope: Tensor
    lift_c_max: Tensor
    spin_decay_gain: Tensor
    # Table, net and paddle geometry.
    table_half_length: Tensor
    table_half_width: Tensor
    table_height: Tensor
    board_thickness: Tensor
    net_half_thickness: Tensor
    net_half_span: Tensor
    net_height: Tensor
    net_tape_height: Tensor
    blade_half_length: Tensor
    blade_half_width: Tensor
    blade_half_thickness: Tensor
    paddle_mass: Tensor
    paddle_inertia_xx: Tensor
    paddle_inertia_yy: Tensor
    paddle_inertia_zz: Tensor
    # Materials.
    table: Material
    rubber: Material
    net: Material
    floor: Material
    post: Material
    fixture: Material
    rim: Material
    base: BallTableParams
    blade_edge: bool = True

    @classmethod
    def from_params(cls, params: BallTableParams | Sequence[BallTableParams], n: int | None = None, *, device: Any = "cpu", dtype: Any = None) -> VecParams:
        """
        Tensors for ``n`` environments sharing ``params``, or one environment per element of a sequence.

        :param params: One parameter set, or one per environment (their speed-curve shapes and lift form must agree).
        :param n: Environments, with a single parameter set.
        :param device: Torch device.
        :param dtype: ``torch.float64`` (the default) or ``torch.float32``.
        :raises ValueError: When per-environment sets disagree on a curve's shape or the lift form
        """

        dtype = torch.float64 if dtype is None else dtype
        sets = [params] * int(n if n is not None else 1) if isinstance(params, BallTableParams) else list(params)
        if not sets:
            raise ValueError("at least one environment is needed")
        base = sets[0]
        forms = {p.aero.lift.form for p in sets}
        if len(forms) != 1:
            raise ValueError(f"one lift form per batch, got {sorted(forms)}")
        for name in MATERIAL_FIELDS:
            first: ContactMaterial = getattr(base, name)
            for p in sets[1:]:
                other: ContactMaterial = getattr(p, name)
                if not (_same_curve(first.restitution_curve, other.restitution_curve) and _same_curve(first.friction_curve, other.friction_curve)):
                    raise ValueError(f"{name}: per-environment speed curves must share their shape (randomise the scalars instead)")
        models = [FlightModel.from_params(p) for p in sets]

        def column(values: Iterable[float]) -> Tensor:
            return torch.tensor(list(values), dtype=dtype, device=device)

        def material(name: str, law: str) -> Material:
            mats: list[ContactMaterial] = [getattr(p, name) for p in sets]
            first = mats[0]
            return Material(
                restitution=column(m.restitution for m in mats),
                friction=column(m.friction for m in mats),
                tangential_restitution=column(m.tangential_restitution for m in mats),
                normal_spin_retention=column(m.normal_spin_retention for m in mats),
                tangential_damping=column(m.tangential_damping for m in mats),
                spin_damping=column(m.spin_damping for m in mats),
                restitution_curve=_curve(first.restitution_curve, device, dtype),
                friction_curve=_curve(first.friction_curve, device, dtype),
                law=law,
            )

        return cls(
            n=len(sets),
            device=device,
            dtype=dtype,
            lift_form=base.aero.lift.form,
            radius=column(m.radius for m in models),
            mass=column(m.mass for m in models),
            inertia=column(m.inertia for m in models),
            inertia_ratio=column(p.ball.inertia_ratio for p in sets),
            gravity=column(m.gravity for m in models),
            aero_gain=column(m.aero_gain for m in models),
            drag_coefficient=column(m.drag_coefficient for m in models),
            drag_spin_slope=column(m.drag_spin_slope for m in models),
            lift_slope=column(m.lift.slope for m in models),
            lift_c_max=column(m.lift.c_max for m in models),
            spin_decay_gain=column(m.spin_decay_gain for m in models),
            table_half_length=column(0.5 * p.table.length for p in sets),
            table_half_width=column(0.5 * p.table.width for p in sets),
            table_height=column(p.table.height for p in sets),
            board_thickness=column(p.table.board_thickness for p in sets),
            net_half_thickness=column(0.5 * p.net.thickness for p in sets),
            net_half_span=column(p.net_half_span for p in sets),
            net_height=column(p.net.height for p in sets),
            net_tape_height=column(p.net.tape_height for p in sets),
            blade_half_length=column(p.paddle.blade_half_length for p in sets),
            blade_half_width=column(p.paddle.blade_half_width for p in sets),
            blade_half_thickness=column(0.5 * p.paddle.thickness for p in sets),
            paddle_mass=column(p.paddle.effective_mass for p in sets),
            paddle_inertia_xx=column(p.paddle.inertia_xx for p in sets),
            paddle_inertia_yy=column(p.paddle.inertia_yy for p in sets),
            paddle_inertia_zz=column(p.paddle.inertia_zz for p in sets),
            table=material("table_contact", "rigid"),
            rubber=material("rubber_contact", "rigid"),
            net=material("net_contact", "rigid"),
            floor=material("floor_contact", "rigid"),
            post=material("post_contact", "rigid"),
            fixture=material("fixture_contact", "rigid"),
            rim=material("rim_contact", "rigid"),
            base=base,
        )

    def with_values(self, values: Mapping[str, Any]) -> VecParams:
        """
        A copy with flat contract scalars replaced per environment (domain randomisation without parameter objects).

        Supported: the ball (``ball_radius``, ``ball_mass``, ``ball_shell_thickness``), the air and the flight
        coefficients, the paddle's effective mass and inertia, and every material scalar in :data:`_MATERIAL_SCALARS`.
        The ball's inertia, the aerodynamic gain and the spin-decay gain are recomputed as the contract does.

        :param values: Flat name to a scalar or an ``(N,)`` tensor.
        :raises KeyError: For a name this batch does not randomise
        """

        def full(value: Any) -> Tensor:
            out = torch.as_tensor(value, dtype=self.dtype, device=self.device)
            return out.expand(self.n).clone() if out.dim() == 0 else out

        flat = self.base.to_flat()
        known = {
            "ball_radius",
            "ball_mass",
            "ball_shell_thickness",
            "air_density",
            "gravity",
            "drag_coefficient",
            "drag_spin_slope",
            "lift_slope",
            "lift_max",
            "spin_decay_coefficient",
            "paddle_effective_mass",
            "paddle_inertia_xx",
            "paddle_inertia_yy",
            "paddle_inertia_zz",
        } | {name for fields in _MATERIAL_SCALARS.values() for name in fields.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise KeyError(f"not randomisable in the batch: {unknown}")
        updates: dict[str, Any] = {}
        ball_names = {"ball_radius", "ball_mass", "ball_shell_thickness", "air_density", "spin_decay_coefficient"}
        if ball_names & set(values):
            # Recompute the derived ball and air quantities as BallParams and FlightModel do.
            get = lambda name: full(values[name]) if name in values else full(flat[name])  # noqa: E731
            radius, mass, shell = get("ball_radius"), get("ball_mass"), get("ball_shell_thickness")
            small = radius - shell
            ratio = (
                0.4
                * (radius**4 + radius**3 * small + radius**2 * small**2 + radius * small**3 + small**4)
                / (radius**2 + radius * small + small**2)
                / radius**2
            )
            inertia = ratio * mass * radius**2
            q = 0.5 * get("air_density") * (math.pi * radius**2)
            updates.update(
                radius=radius,
                mass=mass,
                inertia=inertia,
                inertia_ratio=ratio,
                aero_gain=q / mass,
                spin_decay_gain=get("spin_decay_coefficient") * q * radius**2 / inertia,
            )
        simple = {
            "gravity": "gravity",
            "drag_coefficient": "drag_coefficient",
            "drag_spin_slope": "drag_spin_slope",
            "lift_slope": "lift_slope",
            "lift_max": "lift_c_max",
            "paddle_effective_mass": "paddle_mass",
            "paddle_inertia_xx": "paddle_inertia_xx",
            "paddle_inertia_yy": "paddle_inertia_yy",
            "paddle_inertia_zz": "paddle_inertia_zz",
        }
        for name, attribute in simple.items():
            if name in values:
                updates[attribute] = full(values[name])
        short = {
            "table_contact": "table",
            "rubber_contact": "rubber",
            "net_contact": "net",
            "floor_contact": "floor",
            "post_contact": "post",
            "fixture_contact": "fixture",
            "rim_contact": "rim",
        }
        for field_name, names in _MATERIAL_SCALARS.items():
            changed = {attribute: full(values[name]) for attribute, name in names.items() if name in values}
            if changed:
                updates[short[field_name]] = dataclasses.replace(getattr(self, short[field_name]), **changed)
        return dataclasses.replace(self, **updates)

    @property
    def posts(self) -> tuple[Tensor, float, tuple[float, float]]:
        """The two posts: axis ``y`` per environment ``(N, 2)`` (negative then positive side), radius, ``z`` span."""

        y = self.net_half_span + NET_POST_OUTSET_M
        return torch.stack([-y, y], -1), NET_POST_RADIUS_M, NET_POST_Z_M

    @property
    def clamps(self) -> tuple[float, tuple[float, float], tuple[float, float]]:
        """The two clamps (mirrored in ``y``): half extent in ``x``, ``|y|`` range, ``z`` range."""

        return NET_CLAMP_HALF_X_M, NET_CLAMP_Y_M, NET_CLAMP_Z_M


__all__ = ["MATERIAL_FIELDS", "Curve", "Material", "VecParams"]
