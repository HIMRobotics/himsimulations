"""
Physics parameter contract for the table-tennis ball, table, net, floor and paddle.

Every quantity is SI: metres, kilograms, seconds, radians. Every model is a
frozen pydantic model, so a parameter set is a value that can be hashed into a
run record, compared and serialised.

Defaults come from ``lib/physics/data/priors.json`` when that file is present,
keyed by the flat names in :data:`PARAMETER_SPECS`. A name the file does not
carry falls back to a provisional default defined here; its provenance says
``provisional`` so no report can mistake it for a measured or cited value.

Frames: the table frame is Z-up with its origin at the centre of the table-top
surface and +X along the table length. The floor is the plane
``z = -table.height`` and the net is the plane ``x = 0``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

PRIORS_PATH = Path(__file__).resolve().parent / "data" / "priors.json"
"""Where the cited priors live; written by the calibration work, read here."""

PROVISIONAL = "provisional"
"""Provenance label for a value that is an engineering placeholder, not a citation or a measurement."""

_FROZEN = ConfigDict(extra="forbid", frozen=True)

LiftForm = Literal["rational", "linear", "constant"]
LIFT_FORMS: tuple[str, ...] = ("rational", "linear", "constant")


class SpeedCurve(BaseModel):
    """
    A coefficient as a piecewise-linear function of the normal approach speed.

    The value is held constant beyond the first and last points.

    :param speeds: Normal approach speeds, m/s, non-negative and strictly increasing.
    :param values: Coefficient at each speed, dimensionless.
    :param reference: The scalar coefficient the curve was tabulated for, dimensionless.
        When set, the owning :class:`ContactMaterial` multiplies the curve by its
        scalar over this reference, so randomising the scalar moves the whole curve
        and the nominal scalar reproduces the curve exactly. When ``None`` the
        curve replaces the scalar.
    """

    model_config = _FROZEN

    speeds: tuple[float, ...] = Field(min_length=1)
    values: tuple[float, ...] = Field(min_length=1)
    reference: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check_shape(self) -> SpeedCurve:
        if len(self.speeds) != len(self.values):
            raise ValueError(f"speed curve has {len(self.speeds)} speeds but {len(self.values)} values")
        if self.speeds[0] < 0.0:
            raise ValueError("speed curve speeds must be non-negative")
        if any(b <= a for a, b in zip(self.speeds, self.speeds[1:], strict=False)):
            raise ValueError("speed curve speeds must be strictly increasing")
        return self

    def __call__(self, speed: float) -> float:
        """Coefficient at ``speed`` (m/s)."""

        return float(np.interp(speed, self.speeds, self.values))


class BallParams(BaseModel):
    """
    The ball: a thin uniform spherical shell.

    :param radius: Outer radius, m.
    :param mass: Mass, kg.
    :param shell_thickness: Wall thickness, m. Sets the moment of inertia through
        the exact hollow-shell formula; it must not exceed the radius (equal is a
        solid sphere).
    """

    model_config = _FROZEN

    radius: float = Field(gt=0.0)
    mass: float = Field(gt=0.0)
    shell_thickness: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_shell(self) -> BallParams:
        if self.shell_thickness > self.radius:
            raise ValueError(f"shell thickness {self.shell_thickness} m exceeds the radius {self.radius} m")
        return self

    @property
    def inner_radius(self) -> float:
        """Inner radius of the shell, m."""

        return self.radius - self.shell_thickness

    @property
    def inertia_ratio(self) -> float:
        """
        Moment of inertia over ``mass * radius**2``, dimensionless.

        Exact for a uniform shell of outer radius ``R`` and inner radius ``a``::

            I = (2/5) m (R^5 - a^5) / (R^3 - a^3)

        evaluated in the factored form ``(R^4 + R^3 a + R^2 a^2 + R a^3 + a^4) /
        (R^2 + R a + a^2)`` so a very thin shell does not lose precision. It is
        2/3 in the thin-shell limit and 2/5 for a solid sphere.
        """

        big = self.radius
        small = self.inner_radius
        numerator = big**4 + big**3 * small + big**2 * small**2 + big * small**3 + small**4
        denominator = big**2 + big * small + small**2
        return 0.4 * numerator / denominator / big**2

    @property
    def inertia(self) -> float:
        """Moment of inertia about any diameter, kg m^2."""

        return self.inertia_ratio * self.mass * self.radius**2

    @property
    def area(self) -> float:
        """Frontal (cross-section) area ``pi R^2``, m^2."""

        return math.pi * self.radius**2


class LiftModel(BaseModel):
    """
    Lift (Magnus) coefficient ``C_L`` as a function of the spin parameter ``S``.

    ``S = R |omega_perp| / |v|`` uses the spin component perpendicular to the
    velocity, because spin about the direction of travel produces no lift.

    - ``rational``: ``C_L = slope S / (1 + slope S / c_max)``. Linear at small
      ``S`` with slope ``slope``, saturating at ``c_max``. ``slope = 1`` and
      ``c_max = 0.5`` gives ``S / (1 + 2 S)``, an uncited placeholder shape.
    - ``linear``: ``C_L = min(slope S, c_max)``. With a large ``c_max`` this is
      the constant-Magnus-coefficient model ``F = 1/2 rho A R C_M (omega x v)``
      common in table-tennis robotics, with ``C_M = slope``.
    - ``constant``: ``C_L = c_max`` whenever there is perpendicular spin.

    :param form: Functional form, one of ``rational``, ``linear`` or ``constant``.
    :param slope: ``dC_L/dS`` at ``S = 0``, dimensionless (unused by ``constant``).
    :param c_max: Saturation value (``rational``), cap (``linear``) or the constant, dimensionless.
    """

    model_config = _FROZEN

    form: LiftForm = "rational"
    slope: float = Field(default=1.0, ge=0.0)
    c_max: float = Field(default=0.5, gt=0.0)

    def coefficient(self, spin_parameter: float) -> float:
        """``C_L`` at spin parameter ``S`` (dimensionless, ``S >= 0``)."""

        s = spin_parameter
        if s <= 0.0:
            return 0.0
        if self.form == "rational":
            return self.slope * s / (1.0 + self.slope * s / self.c_max)
        if self.form == "linear":
            return min(self.slope * s, self.c_max)
        return self.c_max

    def coefficient_array(self, spin_parameter: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`coefficient`."""

        s = np.maximum(np.asarray(spin_parameter, dtype=float), 0.0)
        if self.form == "rational":
            out = self.slope * s / (1.0 + self.slope * s / self.c_max)
        elif self.form == "linear":
            out = np.minimum(self.slope * s, self.c_max)
        else:
            out = np.full_like(s, self.c_max)
        return np.where(s > 0.0, out, 0.0)


class AeroParams(BaseModel):
    """
    Air and gravity.

    The flight equations these feed are documented in
    :mod:`lib.physics.reference.flight`.

    :param air_density: rho, kg/m^3. Zero gives a vacuum.
    :param gravity: Magnitude of gravitational acceleration along -Z, m/s^2.
    :param drag_coefficient: ``C_D`` at zero spin, dimensionless.
    :param drag_spin_slope: ``dC_D/dS``, dimensionless; ``C_D = drag_coefficient + drag_spin_slope * S``.
    :param lift: Lift-coefficient model ``C_L(S)``.
    :param spin_decay_coefficient: ``C_tau``, dimensionless. The air torque is
        ``tau = -C_tau * (1/2) rho A R^2 |v| omega``, so spin decays at the rate
        ``C_tau rho A R^2 |v| / (2 I)`` per second.
    """

    model_config = _FROZEN

    air_density: float = Field(ge=0.0)
    gravity: float = Field(ge=0.0)
    drag_coefficient: float = Field(ge=0.0)
    drag_spin_slope: float = Field(default=0.0, ge=0.0)
    lift: LiftModel = LiftModel()
    spin_decay_coefficient: float = Field(default=0.0, ge=0.0)


class ContactMaterial(BaseModel):
    """
    Impact law for one ball-surface pair.

    :param restitution: Normal coefficient of restitution ``e`` in [0, 1]:
        the normal contact velocity reverses to ``-e`` times itself.
    :param friction: Coulomb friction coefficient ``mu``, dimensionless, >= 0.
    :param tangential_restitution: Tangential coefficient of restitution
        ``e_t`` in [0, 1]. Zero is the rigid Coulomb model (a sticking contact
        leaves rolling). Positive values model tangential compliance: a
        sticking contact point reverses its slip to ``-e_t`` times itself, which
        is how a grippy rubber returns overspin.
    :param normal_spin_retention: Fraction ``s_n`` in [0, 1] of the spin about
        the contact normal that survives the impact. The rigid model has no
        torque about the normal, so 1 is its natural value; less than 1 stands
        in for pivoting friction in a grippy rubber.
    :param restitution_curve: Optional ``e`` against normal approach speed
        (m/s). With a ``reference`` it is scaled by ``restitution / reference``
        (and clipped to [0, 1]); without one it replaces ``restitution``.
    :param friction_curve: Optional ``mu`` against normal approach speed (m/s),
        scaled by ``friction / reference`` in the same way.
    :param law: ``rigid`` (the default) is the rigid-ball impact with Coulomb
        friction. ``catch`` is a compliant surface that catches the ball, the
        net mesh: the normal velocity reverses at ``e``, while
        ``tangential_damping`` and ``spin_damping`` scale down the tangential
        velocity and the spin directly, so spin is never converted into
        tangential velocity. See :mod:`lib.physics.reference.impact`.
    :param tangential_damping: Fraction ``d_t`` in [0, 1] of the ball's
        tangential velocity (relative to the surface) that a catch removes.
        Used only by the ``catch`` law.
    :param spin_damping: Fraction ``d_w`` in [0, 1] of the ball's spin that a
        catch removes. Used only by the ``catch`` law.
    :param rolling_resistance: Rolling-resistance coefficient ``C_rr`` for a ball
        rolling on this surface, dimensionless: the resisting couple is
        ``C_rr N R``, equivalently the centre force ``C_rr N`` that would keep
        the ball rolling at constant speed. A freely rolling ball then
        decelerates at ``C_rr g / (1 + k)``. Used by the reference model's
        sustained-rolling regime.
    """

    model_config = _FROZEN

    restitution: float = Field(ge=0.0, le=1.0)
    friction: float = Field(ge=0.0)
    tangential_restitution: float = Field(default=0.0, ge=0.0, le=1.0)
    normal_spin_retention: float = Field(default=1.0, ge=0.0, le=1.0)
    restitution_curve: SpeedCurve | None = None
    friction_curve: SpeedCurve | None = None
    law: Literal["rigid", "catch"] = "rigid"
    tangential_damping: float = Field(default=0.0, ge=0.0, le=1.0)
    spin_damping: float = Field(default=0.0, ge=0.0, le=1.0)
    rolling_resistance: float = Field(default=0.0, ge=0.0)

    def as_catch(self) -> ContactMaterial:
        """This material under the ``catch`` law (same restitution and damping)."""

        return self.model_copy(update={"law": "catch"})

    @model_validator(mode="after")
    def _check_curves(self) -> ContactMaterial:
        if self.restitution_curve is not None and not all(0.0 <= v <= 1.0 for v in self.restitution_curve.values):
            raise ValueError("restitution curve values must lie in [0, 1]")
        if self.friction_curve is not None and not all(v >= 0.0 for v in self.friction_curve.values):
            raise ValueError("friction curve values must be non-negative")
        return self

    def restitution_at(self, normal_speed: float) -> float:
        """Normal restitution at an approach speed of ``normal_speed`` m/s."""

        curve = self.restitution_curve
        if curve is None:
            return self.restitution
        scale = 1.0 if curve.reference is None else self.restitution / curve.reference
        return min(max(curve(normal_speed) * scale, 0.0), 1.0)

    def friction_at(self, normal_speed: float) -> float:
        """Friction coefficient at an approach speed of ``normal_speed`` m/s."""

        curve = self.friction_curve
        if curve is None:
            return self.friction
        scale = 1.0 if curve.reference is None else self.friction / curve.reference
        return max(curve(normal_speed) * scale, 0.0)


class TableGeometry(BaseModel):
    """
    The table top: the playing surface and the board under it.

    :param length: Along +X, m.
    :param width: Along +Y, m.
    :param height: Playing surface above the floor, m.
    :param board_thickness: Depth of the top board under the playing surface, m: the
        board is the box ``|x| <= length/2``, ``|y| <= width/2``, ``-board_thickness <= z <= 0``.
    """

    model_config = _FROZEN

    length: float = Field(gt=0.0)
    width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)
    board_thickness: float = Field(default=0.025, gt=0.0)


class NetGeometry(BaseModel):
    """
    The net, modelled as a thin vertical box in the plane ``x = 0`` from the table top up.

    Its face below the top tape is a compliant mesh that catches the ball
    (``net_contact`` under the ``catch`` law). The top tape, the top edge
    and the ends are rigid (``net_contact`` under the ``rigid`` law).

    :param height: Top of the net above the playing surface, m.
    :param overhang: How far the net extends beyond each side line, m.
    :param thickness: Along X, m.
    :param tape_height: Height of the stiff top tape below the net top, m.
        A face contact whose point lies within it is rigid, not mesh.
    """

    model_config = _FROZEN

    height: float = Field(gt=0.0)
    overhang: float = Field(ge=0.0)
    thickness: float = Field(ge=0.0)
    tape_height: float = Field(default=0.015, ge=0.0)

    @model_validator(mode="after")
    def _check_tape(self) -> NetGeometry:
        if self.tape_height > self.height:
            raise ValueError(f"net tape height {self.tape_height} m exceeds the net height {self.height} m")
        return self


PADDLE_INERTIA_DERIVATION = (
    "derived: uniform slab of the cited 157 x 150 x 13.6 mm head at the cited 0.17 kg effective mass, "
    "about the blade centre (I = m (a^2 + b^2) / 12); hand coupling not included; range is judgement"
)
"""How the paddle inertia defaults were obtained."""

PADDLE_INERTIA_XX = 3.214e-4
"""0.17 (0.150^2 + 0.0136^2) / 12, kg m^2: about the handle axis."""

PADDLE_INERTIA_YY = 3.518e-4
"""0.17 (0.157^2 + 0.0136^2) / 12, kg m^2: about the in-plane axis across the blade."""

PADDLE_INERTIA_ZZ = 6.679e-4
"""0.17 (0.157^2 + 0.150^2) / 12, kg m^2: about the face normal."""


class PaddleParams(BaseModel):
    """
    The paddle blade with its rubbers, as an elliptical slab.

    :param blade_half_length: Semi-axis along the blade's local +X (towards the handle), m.
    :param blade_half_width: Semi-axis along the blade's local +Y, m.
    :param thickness: Total thickness, wood plus both rubbers, m. The local +Z
        axis is the forehand face normal.
    :param effective_mass: Mass the ball meets at contact, kg: the paddle plus
        whatever part of the hand and forearm couples into a sub-millisecond
        impact. Infinite for a perfectly driven paddle.
    :param inertia_xx: Principal moment about the blade's local X (the handle
        axis) through its centre of mass, kg m^2. Resists an impact offset
        across the blade (local Y).
    :param inertia_yy: Principal moment about local Y, kg m^2. Resists an
        impact offset along the blade (local X).
    :param inertia_zz: Principal moment about the face normal, local Z, kg m^2.

    The centre of mass is taken at the blade centre. With an impact point
    offset ``r`` from it, a rigid paddle meets the ball with
    ``1/M_eff = 1/effective_mass + (r x n)^T I^-1 (r x n)``.
    """

    model_config = _FROZEN

    blade_half_length: float = Field(gt=0.0)
    blade_half_width: float = Field(gt=0.0)
    thickness: float = Field(gt=0.0)
    effective_mass: float = Field(gt=0.0)
    inertia_xx: float = Field(default=PADDLE_INERTIA_XX, gt=0.0)
    inertia_yy: float = Field(default=PADDLE_INERTIA_YY, gt=0.0)
    inertia_zz: float = Field(default=PADDLE_INERTIA_ZZ, gt=0.0)

    @property
    def inertia(self) -> tuple[float, float, float]:
        """Principal moments ``(I_xx, I_yy, I_zz)`` in the blade frame, kg m^2."""

        return (self.inertia_xx, self.inertia_yy, self.inertia_zz)


class BallTableParams(BaseModel):
    """
    The complete physics parameter set for one simulated world.

    :param ball: Ball geometry, mass and inertia.
    :param aero: Air density, gravity, drag, lift and spin decay.
    :param table: Table geometry.
    :param net: Net geometry.
    :param paddle: Paddle geometry and effective mass.
    :param table_contact: Ball-table impact law.
    :param rubber_contact: Ball-rubber impact law, for either paddle face.
    :param net_contact: Ball-net law: rigid for the top tape, top edge and ends; its restitution and damping
        under the ``catch`` law for the mesh face (:attr:`net_mesh_contact`).
    :param floor_contact: Ball-floor impact law.
    :param fixture_contact: Ball against the net clamps and the table's undercarriage (painted metal or hard plastic).
    :param post_contact: Ball against the steel net posts.
    :param wall_contact: Ball against the room's walls, ceiling and solid furniture.
    :param rim_contact: Ball against the paddle blade's rounded edge (wood or edge tape; rigid Coulomb, no rubber
        law). **An unvalidated judgement prior:** no measurement exists, and the capture cannot see an edge hit.
    """

    model_config = _FROZEN

    ball: BallParams
    aero: AeroParams
    table: TableGeometry
    net: NetGeometry
    paddle: PaddleParams
    table_contact: ContactMaterial
    rubber_contact: ContactMaterial
    net_contact: ContactMaterial
    floor_contact: ContactMaterial
    fixture_contact: ContactMaterial = ContactMaterial(restitution=0.85, friction=0.25)
    post_contact: ContactMaterial = ContactMaterial(restitution=0.946, friction=0.20)
    wall_contact: ContactMaterial = ContactMaterial(restitution=0.80, friction=0.35)
    rim_contact: ContactMaterial = ContactMaterial(restitution=0.50, friction=0.30)

    @property
    def net_mesh_contact(self) -> ContactMaterial:
        """The ball-net law for the mesh face below the top tape: ``net_contact`` under the ``catch`` law."""

        return self.net_contact.as_catch()

    @property
    def net_half_span(self) -> float:
        """Half the net length along Y, m: half the table width plus the overhang."""

        return 0.5 * self.table.width + self.net.overhang

    def to_flat(self) -> dict[str, float]:
        """Every scalar in :data:`PARAMETER_SPECS`, keyed by its flat name."""

        dumped = self.model_dump()
        out: dict[str, float] = {}
        for spec in PARAMETER_SPECS:
            node: Any = dumped
            for key in spec.path:
                node = node[key]
            out[spec.name] = float(node)
        return out

    def with_values(self, values: Mapping[str, float]) -> BallTableParams:
        """A copy with the named scalars replaced; names come from :data:`PARAMETER_SPECS`."""

        unknown = sorted(set(values) - set(_SPEC_BY_NAME))
        if unknown:
            raise KeyError(f"unknown physics parameter names: {unknown}")
        dumped = self.model_dump()
        for name, value in values.items():
            node: Any = dumped
            path = _SPEC_BY_NAME[name].path
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = float(value)
        return BallTableParams.model_validate(dumped)


@dataclass(frozen=True)
class ParameterSpec:
    """
    One flat, named scalar of the contract with its provisional default.

    :param name: Flat key, as used in ``priors.json`` and run records.
    :param path: Field path inside :class:`BallTableParams`.
    :param unit: SI unit; ``1`` for dimensionless.
    :param default: Provisional nominal value, used only when the priors file lacks the name.
    :param low: Provisional lower bound of the randomisation range.
    :param high: Provisional upper bound; ``low == high`` means not randomised.
    :param note: Why the provisional value is what it is.
    :param source: Provenance of the fallback value: ``provisional`` for a placeholder,
        or a ``derived: ...`` statement for a value computed from cited inputs.
    """

    name: str
    path: tuple[str, ...]
    unit: str
    default: float
    low: float
    high: float
    note: str
    source: str = PROVISIONAL


PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("ball_radius", ("ball", "radius"), "m", 0.0200, 0.0200, 0.0202, "40 mm ball; 40+ balls measure up to about 40.4 mm"),
    ParameterSpec("ball_mass", ("ball", "mass"), "kg", 0.0027, 0.00267, 0.00277, "2.7 g nominal, 2.67-2.77 g tolerance band"),
    ParameterSpec(
        "ball_shell_thickness", ("ball", "shell_thickness"), "m", 0.0004, 0.0003, 0.0006, "placeholder wall thickness; moves I/(mR^2) between 0.65 and 0.66"
    ),
    ParameterSpec("air_density", ("aero", "air_density"), "kg/m^3", 1.20, 1.12, 1.25, "indoor air, 15-30 C at 0-600 m altitude"),
    ParameterSpec("gravity", ("aero", "gravity"), "m/s^2", 9.81, 9.81, 9.81, "standard gravity; not randomised"),
    ParameterSpec("drag_coefficient", ("aero", "drag_coefficient"), "1", 0.45, 0.35, 0.60, "subcritical smooth sphere; placeholder until fitted"),
    ParameterSpec("drag_spin_slope", ("aero", "drag_spin_slope"), "1", 0.0, 0.0, 0.0, "spin-dependent drag off until a prior exists"),
    ParameterSpec("lift_slope", ("aero", "lift", "slope"), "1", 1.0, 0.5, 1.5, "S/(1+2S) form; placeholder"),
    ParameterSpec("lift_max", ("aero", "lift", "c_max"), "1", 0.5, 0.3, 0.6, "S/(1+2S) form; placeholder"),
    ParameterSpec(
        "spin_decay_coefficient",
        ("aero", "spin_decay_coefficient"),
        "1",
        0.02,
        0.0,
        0.1,
        "about 8 % spin loss per second at 10 m/s; placeholder, unobservable here",
    ),
    ParameterSpec("table_length", ("table", "length"), "m", 2.74, 2.74, 2.74, "regulation table"),
    ParameterSpec("table_width", ("table", "width"), "m", 1.525, 1.525, 1.525, "regulation table"),
    ParameterSpec("table_height", ("table", "height"), "m", 0.76, 0.76, 0.76, "regulation table"),
    ParameterSpec(
        "table_board_thickness",
        ("table", "board_thickness"),
        "m",
        0.025,
        0.025,
        0.025,
        "the table asset's top board",
        "derived: the table asset's top mesh spans 0.735-0.76 m above the floor",
    ),
    ParameterSpec("net_height", ("net", "height"), "m", 0.1525, 0.1525, 0.1525, "regulation net"),
    ParameterSpec("net_overhang", ("net", "overhang"), "m", 0.1525, 0.1525, 0.1525, "regulation net"),
    ParameterSpec("net_thickness", ("net", "thickness"), "m", 0.003, 0.003, 0.003, "placeholder mesh-plus-cord thickness"),
    ParameterSpec("net_tape_height", ("net", "tape_height"), "m", 0.015, 0.010, 0.015, "judgement: ITTF T2 allows a top tape up to 15 mm (per priors notes)"),
    ParameterSpec("paddle_blade_half_length", ("paddle", "blade_half_length"), "m", 0.0785, 0.0785, 0.0785, "placeholder 157 mm blade"),
    ParameterSpec("paddle_blade_half_width", ("paddle", "blade_half_width"), "m", 0.0750, 0.0750, 0.0750, "placeholder 150 mm blade"),
    ParameterSpec("paddle_thickness", ("paddle", "thickness"), "m", 0.014, 0.014, 0.014, "placeholder: about 6 mm wood plus two 4 mm rubbers"),
    ParameterSpec("paddle_effective_mass", ("paddle", "effective_mass"), "kg", 0.30, 0.15, 1.00, "placeholder: 0.17-0.19 kg paddle plus partial hand coupling"),
    ParameterSpec(
        "paddle_inertia_xx", ("paddle", "inertia_xx"), "kg m^2", PADDLE_INERTIA_XX, 2.5e-4, 6.5e-4, "slab about the handle axis", PADDLE_INERTIA_DERIVATION
    ),
    ParameterSpec(
        "paddle_inertia_yy", ("paddle", "inertia_yy"), "kg m^2", PADDLE_INERTIA_YY, 2.7e-4, 7.0e-4, "slab about the cross-blade axis", PADDLE_INERTIA_DERIVATION
    ),
    ParameterSpec(
        "paddle_inertia_zz", ("paddle", "inertia_zz"), "kg m^2", PADDLE_INERTIA_ZZ, 5.0e-4, 1.3e-3, "slab about the face normal", PADDLE_INERTIA_DERIVATION
    ),
    ParameterSpec(
        "table_restitution", ("table_contact", "restitution"), "1", 0.94, 0.92, 0.955, "as priors.json; drag-corrected drop test (0.876 ignores drag)"
    ),
    ParameterSpec("table_friction", ("table_contact", "friction"), "1", 0.25, 0.10, 0.50, "placeholder"),
    ParameterSpec("table_tangential_restitution", ("table_contact", "tangential_restitution"), "1", 0.0, 0.0, 0.0, "rigid Coulomb table"),
    ParameterSpec("rubber_restitution", ("rubber_contact", "restitution"), "1", 0.85, 0.70, 0.92, "placeholder"),
    ParameterSpec("rubber_friction", ("rubber_contact", "friction"), "1", 0.80, 0.40, 1.20, "placeholder: inverted rubber is grippy"),
    ParameterSpec(
        "rubber_tangential_restitution", ("rubber_contact", "tangential_restitution"), "1", 0.70, 0.50, 0.85, "as priors.json: rubber reverses contact slip"
    ),
    ParameterSpec(
        "rubber_normal_spin_retention", ("rubber_contact", "normal_spin_retention"), "1", 1.0, 1.0, 1.0, "rigid model: no torque about the face normal"
    ),
    ParameterSpec("net_restitution", ("net_contact", "restitution"), "1", 0.20, 0.05, 0.40, "placeholder: the net is soft"),
    ParameterSpec("net_friction", ("net_contact", "friction"), "1", 0.30, 0.10, 0.60, "placeholder"),
    ParameterSpec(
        "net_tangential_damping",
        ("net_contact", "tangential_damping"),
        "1",
        0.80,
        0.50,
        0.95,
        "judgement: no ball-net measurement exists; the mesh catches the ball",
    ),
    ParameterSpec("net_spin_damping", ("net_contact", "spin_damping"), "1", 0.70, 0.40, 0.95, "judgement: no ball-net measurement exists"),
    ParameterSpec("floor_restitution", ("floor_contact", "restitution"), "1", 0.85, 0.70, 0.92, "placeholder: concrete"),
    ParameterSpec("floor_friction", ("floor_contact", "friction"), "1", 0.30, 0.15, 0.60, "placeholder"),
    ParameterSpec(
        "fixture_restitution",
        ("fixture_contact", "restitution"),
        "1",
        0.85,
        0.75,
        0.92,
        "judgement: no measurement; below the ball's 0.92-0.95 rigid-target restitution (Remond 2022)",
    ),
    ParameterSpec("fixture_friction", ("fixture_contact", "friction"), "1", 0.25, 0.15, 0.40, "judgement: painted metal or hard plastic"),
    ParameterSpec(
        "post_restitution",
        ("post_contact", "restitution"),
        "1",
        0.946,
        0.921,
        0.970,
        "ITTF T3 steel-block bounce 240-265 mm from 305 mm, drag-corrected through this model (C_D 0.50, rho 1.19)",
    ),
    ParameterSpec("post_friction", ("post_contact", "friction"), "1", 0.20, 0.10, 0.35, "judgement: ABS on painted or polished steel"),
    ParameterSpec(
        "wall_restitution",
        ("wall_contact", "restitution"),
        "1",
        0.80,
        0.70,
        0.90,
        "judgement: below the ball's rigid-target bound (Remond 2022); a stud wall and sheet-metal cabinets dissipate more",
    ),
    ParameterSpec("wall_friction", ("wall_contact", "friction"), "1", 0.35, 0.20, 0.50, "judgement: painted or coated surface"),
    ParameterSpec(
        "floor_rolling_resistance",
        ("floor_contact", "rolling_resistance"),
        "1",
        0.010,
        0.003,
        0.030,
        "judgement: no measurement for a hollow plastic ball on concrete; high-bounce balls roll easily (Cross 2015)",
    ),
    ParameterSpec(
        "table_rolling_resistance", ("table_contact", "rolling_resistance"), "1", 0.005, 0.002, 0.015, "judgement: smoother and stiffer than the floor"
    ),
    # The paddle's rounded edge (appended last, so the earlier specs keep their order).
    ParameterSpec(
        "rim_restitution",
        ("rim_contact", "restitution"),
        "1",
        0.50,
        0.30,
        0.70,
        "UNVALIDATED judgement prior: a thin wooden edge or edge tape, well below the ball's 0.92-0.95 rigid-target"
        " restitution (Remond 2022); no measurement, and the capture cannot see an edge hit",
    ),
    ParameterSpec(
        "rim_friction",
        ("rim_contact", "friction"),
        "1",
        0.30,
        0.15,
        0.50,
        "UNVALIDATED judgement prior: plastic ball on wood or edge tape; no measurement",
    ),
)
"""The flat parameter names, their field paths, units and provisional defaults and ranges."""

_SPEC_BY_NAME: dict[str, ParameterSpec] = {spec.name: spec for spec in PARAMETER_SPECS}

LIFT_FORM_KEY = "lift_form"
"""Priors key holding the lift-model form as a string."""

CURVE_KEYS: dict[str, tuple[str, str]] = {
    f"{surface}_{coefficient}_{suffix}": (f"{surface}_contact", f"{coefficient}_curve")
    for surface in ("table", "rubber", "net", "floor")
    for coefficient in ("restitution", "friction")
    for suffix in ("curve", "vs_speed")
}
"""Priors keys that may hold a ``{"speeds": [...], "values": [...]}`` speed curve, in either spelling.

Each curve's ``reference`` is set to the nominal value of its scalar (for
example ``table_restitution``), so the scalar scales the curve: the nominal
scalar reproduces the curve and a randomised scalar moves all of it. A curve
therefore never silently overrides, and freezes, the scalar beside it.
``speed_curves=False`` drops the curves for a constant-coefficient run."""

METADATA_KEYS = frozenset({"sources", "references", "citations", "meta", "metadata", "schema", "version", "notes"})
"""Top-level priors keys that describe the file rather than a parameter."""

_DIMENSIONLESS = {"", "1", "-", "dimensionless", "none", "unitless"}
_UNIT_ALIASES: dict[str, str] = {
    "kg/m3": "kg/m^3",
    "kg m^-3": "kg/m^3",
    "kg·m^-3": "kg/m^3",
    "kg/m³": "kg/m^3",
    "m/s2": "m/s^2",
    "m s^-2": "m/s^2",
    "m/s²": "m/s^2",
    "kg*m^2": "kg m^2",
    "kg.m^2": "kg m^2",
    "kg·m^2": "kg m^2",
    "kg m2": "kg m^2",
    "kg m²": "kg m^2",
    "kgm^2": "kg m^2",
}
_UNIT_SCALES: dict[tuple[str, str], float] = {("mm", "m"): 1e-3, ("cm", "m"): 1e-2, ("g", "kg"): 1e-3}


def _canonical_unit(unit: str | None) -> str:
    text = (unit or "").strip()
    if text.lower() in _DIMENSIONLESS:
        return "1"
    return _UNIT_ALIASES.get(text, text)


@dataclass(frozen=True)
class ParameterPrior:
    """
    The nominal value and randomisation range of one flat parameter, with its provenance.

    :param name: Flat parameter name.
    :param value: Nominal value in SI units.
    :param low: Lower bound of the randomisation range, SI.
    :param high: Upper bound, SI; ``low == high`` means fixed.
    :param unit: SI unit.
    :param source: Citation or provenance text from the priors file, or ``provisional``.
    """

    name: str
    value: float
    low: float
    high: float
    unit: str
    source: str

    @property
    def provisional(self) -> bool:
        """True when the value is this module's placeholder rather than a priors-file entry."""

        return self.source == PROVISIONAL

    @property
    def randomised(self) -> bool:
        """True when the range has width."""

        return self.high > self.low


@dataclass(frozen=True)
class PriorTable:
    """
    Nominal values and ranges for every flat parameter.

    :param priors: One :class:`ParameterPrior` per name in :data:`PARAMETER_SPECS`.
    :param lift_form: The lift-model form.
    :param curves: Speed curves by ``(section, field)`` target, e.g. ``("table_contact", "restitution_curve")``,
        each with its ``reference`` set to the nominal scalar (see :data:`CURVE_KEYS`).
    :param path: The priors file read, or ``None`` when none was present.
    :param unrecognised: Keys in the priors file this contract does not know.
    """

    priors: Mapping[str, ParameterPrior]
    lift_form: LiftForm
    curves: Mapping[tuple[str, str], SpeedCurve]
    path: Path | None
    unrecognised: tuple[str, ...]

    def nominal(self, *, speed_curves: bool = True) -> BallTableParams:
        """
        The nominal parameter set.

        :param speed_curves: Attach the speed curves in :attr:`curves`, each scaled by
            its scalar; false gives constant coefficients (for an engine that has no
            speed-dependent restitution).
        """

        return _build({name: prior.value for name, prior in self.priors.items()}, self.lift_form, self.curves if speed_curves else {})

    def ranges(self) -> dict[str, tuple[float, float]]:
        """``(low, high)`` per flat name, SI."""

        return {name: (prior.low, prior.high) for name, prior in self.priors.items()}

    def provisional_names(self) -> tuple[str, ...]:
        """Names whose value is a placeholder, not a priors-file entry."""

        return tuple(name for name, prior in self.priors.items() if prior.provisional)

    def sample(self, rng: np.random.Generator, names: Iterable[str] | None = None, *, speed_curves: bool = True) -> BallTableParams:
        """
        A parameter set with each named parameter drawn uniformly from its range.

        :param rng: Random generator; the draw is reproducible from its seed.
        :param names: Parameters to randomise; default every parameter whose range has width.
            The rest stay nominal.
        :param speed_curves: Attach the speed curves, each scaled by its (randomised) scalar.
        :return: A validated parameter set.
        """

        chosen = [name for name, prior in self.priors.items() if prior.randomised] if names is None else list(names)
        unknown = sorted(set(chosen) - set(self.priors))
        if unknown:
            raise KeyError(f"unknown physics parameter names: {unknown}")
        values = {name: prior.value for name, prior in self.priors.items()}
        for name in chosen:
            prior = self.priors[name]
            values[name] = float(rng.uniform(prior.low, prior.high)) if prior.randomised else prior.value
        return _build(values, self.lift_form, self.curves if speed_curves else {})


def _build(values: Mapping[str, float], lift_form: str, curves: Mapping[tuple[str, str], SpeedCurve]) -> BallTableParams:
    tree: dict[str, Any] = {}
    for spec in PARAMETER_SPECS:
        node = tree
        for key in spec.path[:-1]:
            node = node.setdefault(key, {})
        node[spec.path[-1]] = float(values[spec.name])
    tree["aero"]["lift"]["form"] = lift_form
    for (section, field), curve in curves.items():
        tree[section][field] = curve.model_dump()
    return BallTableParams.model_validate(tree)


def provisional_priors() -> PriorTable:
    """The fallback table (provisional placeholders and derived values), ignoring any priors file."""

    priors = {spec.name: ParameterPrior(spec.name, spec.default, spec.low, spec.high, spec.unit, spec.source) for spec in PARAMETER_SPECS}
    return PriorTable(priors=priors, lift_form="rational", curves={}, path=None, unrecognised=())


def _entries(document: Any) -> dict[str, Any]:
    """The name-to-entry mapping inside a priors document, tolerating a wrapper key or a list."""

    if isinstance(document, Mapping):
        for wrapper in ("parameters", "priors", "params"):
            inner = document.get(wrapper)
            if isinstance(inner, Mapping | list):
                document = inner
                break
    if isinstance(document, list):
        return {str(item["name"]): item for item in document if isinstance(item, Mapping) and "name" in item}
    if isinstance(document, Mapping):
        return {str(key): value for key, value in document.items() if not str(key).startswith(("_", "$"))}
    raise ValueError("priors document must be a mapping or a list of named entries")


def _first(entry: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in entry and entry[key] is not None:
            return entry[key]
    return None


def _source_text(entry: Mapping[str, Any]) -> str:
    raw = _first(entry, ("source", "citation", "citations", "reference", "references", "sources"))
    if raw is None:
        return "priors.json"
    if isinstance(raw, list):
        return "; ".join(str(item) for item in raw)
    if isinstance(raw, Mapping):
        return "; ".join(f"{key}: {value}" for key, value in raw.items())
    return str(raw)


def _parse_prior(spec: ParameterSpec, entry: Any) -> ParameterPrior:
    if isinstance(entry, int | float) and not isinstance(entry, bool):
        value = float(entry)
        return ParameterPrior(spec.name, value, value, value, spec.unit, "priors.json")
    if not isinstance(entry, Mapping):
        raise ValueError(f"priors entry {spec.name!r} must be a number or an object, got {type(entry).__name__}")
    raw_value = _first(entry, ("value", "nominal", "default", "mean", "median"))
    if raw_value is None:
        raise ValueError(f"priors entry {spec.name!r} has no value (expected one of value, nominal, default, mean, median)")
    raw_range = _first(entry, ("range", "bounds", "randomisation_range", "randomization_range"))
    if raw_range is not None:
        if not isinstance(raw_range, list | tuple) or len(raw_range) != 2:
            raise ValueError(f"priors entry {spec.name!r} range must be a [low, high] pair")
        low, high = float(raw_range[0]), float(raw_range[1])
    else:
        lo_raw = _first(entry, ("low", "min", "lower"))
        hi_raw = _first(entry, ("high", "max", "upper"))
        low = float(raw_value) if lo_raw is None else float(lo_raw)
        high = float(raw_value) if hi_raw is None else float(hi_raw)
    expected = _canonical_unit(spec.unit)
    raw_unit = _first(entry, ("unit", "units"))
    unit = expected if raw_unit is None else _canonical_unit(str(raw_unit))
    scale = 1.0
    if unit != expected:
        if (unit, expected) not in _UNIT_SCALES:
            raise ValueError(f"priors entry {spec.name!r} is in {unit!r}; the contract needs {expected!r}")
        scale = _UNIT_SCALES[(unit, expected)]
    value = float(raw_value) * scale
    low, high = low * scale, high * scale
    if not low <= value <= high:
        raise ValueError(f"priors entry {spec.name!r}: value {value} lies outside its range [{low}, {high}]")
    return ParameterPrior(spec.name, value, low, high, spec.unit, _source_text(entry))


def _parse_curve(key: str, entry: Any) -> SpeedCurve:
    if not isinstance(entry, Mapping):
        raise ValueError(f"priors entry {key!r} must be an object with speeds and values")
    speed_unit = entry.get("speed_unit", "m/s")
    if speed_unit != "m/s":
        raise ValueError(f"priors entry {key!r} has speeds in {speed_unit!r}; the contract needs 'm/s'")
    return SpeedCurve(speeds=tuple(float(s) for s in entry["speeds"]), values=tuple(float(v) for v in entry["values"]))


def load_priors(path: Path | str | None = PRIORS_PATH) -> PriorTable:
    """
    Read the priors file and fill every name it lacks with the provisional default.

    Accepted entry shapes, keyed by the flat names of :data:`PARAMETER_SPECS`
    (optionally under a ``parameters`` or ``priors`` wrapper, or as a list of
    objects with a ``name``): a bare number, or an object with ``value`` (or
    ``nominal``/``default``/``mean``/``median``), an optional ``range`` pair
    (or ``low``/``high``, ``min``/``max``), an optional ``unit`` (``mm``,
    ``cm`` and ``g`` convert to SI; any other mismatch is an error) and an
    optional ``source`` or ``citation``. An entry with no range is fixed at
    its value. ``lift_form`` may hold the lift-model form, and the keys in
    :data:`CURVE_KEYS` may hold speed curves, which the matching scalar scales.
    Top-level :data:`METADATA_KEYS` such as ``sources`` are skipped.

    :param path: The priors file; ``None`` or a missing file gives the provisional table.
    :return: The merged table, with the keys it did not recognise listed.
    """

    table = provisional_priors()
    if path is None or not Path(path).is_file():
        return table
    resolved = Path(path)
    entries = _entries(json.loads(resolved.read_text()))
    priors = dict(table.priors)
    curves: dict[tuple[str, str], SpeedCurve] = {}
    lift_form: str = table.lift_form
    unrecognised: list[str] = []
    for key, entry in entries.items():
        if key in _SPEC_BY_NAME:
            priors[key] = _parse_prior(_SPEC_BY_NAME[key], entry)
        elif key == LIFT_FORM_KEY:
            lift_form = str(entry["value"] if isinstance(entry, Mapping) else entry)
        elif key in CURVE_KEYS:
            target = CURVE_KEYS[key]
            if target in curves:
                raise ValueError(f"priors file sets {target[0]}.{target[1]} twice")
            curves[target] = _parse_curve(key, entry)
        elif key not in METADATA_KEYS:
            unrecognised.append(key)
    if lift_form not in LIFT_FORMS:
        raise ValueError(f"priors lift_form {lift_form!r} is not one of {LIFT_FORMS}")
    for (section, field), curve in list(curves.items()):
        scalar = f"{section.removesuffix('_contact')}_{field.removesuffix('_curve')}"
        curves[(section, field)] = curve.model_copy(update={"reference": priors[scalar].value})
    merged = PriorTable(priors=priors, lift_form=cast(LiftForm, lift_form), curves=curves, path=resolved, unrecognised=tuple(sorted(unrecognised)))
    merged.nominal()
    merged.nominal(speed_curves=False)
    return merged


def nominal_params(path: Path | str | None = PRIORS_PATH, *, speed_curves: bool = True) -> BallTableParams:
    """The nominal parameter set: priors-file values where present, provisional defaults elsewhere."""

    return load_priors(path).nominal(speed_curves=speed_curves)
