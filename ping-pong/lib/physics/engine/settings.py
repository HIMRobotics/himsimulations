"""
Canonical PhysX settings for the ball, the surfaces and the paddles, derived from a parameter set.

Pure: no simulator import. The replay authors exactly these values
(:mod:`lib.physics.engine.physx`).

Why each setting (measured on Isaac Sim 6.0.1):

- **PGS, not TGS.** Under TGS a contact whose first touch falls in the first
  ~1-8% of a step loses most of its restitution (rubber 0.03-0.6, the table
  sometimes not bouncing at all), on CPU and GPU. PGS gives the material's
  restitution at every contact phase.
- **CPU dynamics with CCD.** Continuous collision detection is CPU-only in
  PhysX; with it there was no tunnelling at 20-50 m/s.
- **Stabilization off, sleep off, damping zero, angular cap raised.** The
  default cap (5729.58 deg/s = 100 rad/s) clips a 150 rev/s topspin to a
  tenth of itself.
- **Material ``min`` combine with a ball of restitution 1 and friction 2.**
  The pair then takes the surface material's values, which carry the
  ``lib.physics`` ball-pair laws.
- **Dynamic paddles.** A velocity written to a kinematic body is ignored (the
  ball returns at ``e * v_in`` whatever the paddle speed) and floods the CPU
  log; a dynamic body driven by velocity every step is exact.
- **Hybrid paddle contact.** No rigid-contact engine represents the rubber's
  tangential restitution (0.70) or normal-spin retention (0.805), so the
  paddle never collides with the ball in PhysX; the adapter detects the
  contact and imposes the reference impact model's outcome
  (:mod:`lib.physics.engine.contacts`).
- **Aerodynamics** are the reference model's :meth:`FlightModel.wrench`,
  applied every step; engine-native fluid forces stay off.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lib.physics.params import BallTableParams

_FROZEN = ConfigDict(frozen=True, extra="forbid")

ContactMode = Literal["engine", "hybrid"]
BallIntegrator = Literal["physx", "rk4"]
"""Who advances the game ball between contacts: ``physx`` (semi-implicit Euler at the game step, with the air force
applied every step) or ``rk4`` (the adapter, with the reference model's RK4 at the game step; PhysX steps only the
paddles and retired balls, at ``engine_dt_s``; :mod:`lib.physics.engine.ball_integrator`)."""
"""``engine``: PhysX resolves the contact with the surface material. ``hybrid``: PhysX ignores the pair and the
adapter imposes the reference impact model's outcome at the step the contact is detected."""


class SceneSettings(BaseModel):
    """
    The PhysX scene.

    :ivar solver: ``PGS`` (see the module docstring for why not TGS).
    :ivar gpu_dynamics: False: CCD needs CPU dynamics.
    :ivar ccd: Continuous collision detection.
    :ivar stabilization: Off: it damps slow contacts.
    :ivar bounce_threshold_mps: Relative normal speed below which PhysX applies no restitution.
    :ivar gravity_mps2: Gravity magnitude along -Z, from the contract.
    :ivar physics_dt_s: Physics step.
    """

    model_config = _FROZEN

    solver: Literal["PGS", "TGS"] = "PGS"
    gpu_dynamics: bool = False
    ccd: bool = True
    stabilization: bool = False
    bounce_threshold_mps: float = Field(default=0.1, ge=0.0)
    gravity_mps2: float = Field(gt=0.0)
    physics_dt_s: float = Field(gt=0.0)


class BallBodySettings(BaseModel):
    """
    The ball rigid body and collider.

    :ivar radius_m: Sphere radius.
    :ivar mass_kg: Mass.
    :ivar inertia_kgm2: Principal moment (shell), all three axes.
    :ivar ccd: Per-body CCD.
    :ivar max_angular_velocity_deg_s: PhysX angular-velocity cap (degrees per second, as PhysX authors it).
    :ivar linear_damping: Zero.
    :ivar angular_damping: Zero.
    :ivar sleep_threshold: Zero (never sleeps).
    :ivar stabilization_threshold: Zero.
    :ivar max_depenetration_velocity_mps: Cap on PhysX's push-out speed.
    :ivar position_iterations: Solver position iterations.
    :ivar velocity_iterations: Solver velocity iterations.
    :ivar contact_offset_m: Speculative contact distance.
    :ivar rest_offset_m: Zero: the sphere touches at its radius.
    :ivar restitution: The ball's own material restitution (1, so ``min`` picks the surface's).
    :ivar friction: The ball's own static and dynamic friction (2, so ``min`` picks the surface's).
    :ivar combine_mode: PhysX restitution and friction combine mode.
    """

    model_config = _FROZEN

    radius_m: float = Field(gt=0.0)
    mass_kg: float = Field(gt=0.0)
    inertia_kgm2: float = Field(gt=0.0)
    ccd: bool = True
    max_angular_velocity_deg_s: float = Field(default=100000.0, ge=60000.0)
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    sleep_threshold: float = 0.0
    stabilization_threshold: float = 0.0
    max_depenetration_velocity_mps: float = 100.0
    position_iterations: int = 16
    velocity_iterations: int = 4
    contact_offset_m: float = 0.01
    rest_offset_m: float = 0.0
    restitution: float = 1.0
    friction: float = 2.0
    combine_mode: Literal["min"] = "min"


class SurfaceSettings(BaseModel):
    """
    One surface's material and how its ball contacts are resolved.

    :ivar restitution: Ball-pair restitution (scalar; PhysX has no speed curve).
    :ivar friction: Ball-pair Coulomb friction.
    :ivar mode: ``engine`` or ``hybrid``.
    """

    model_config = _FROZEN

    restitution: float = Field(ge=0.0, le=1.0)
    friction: float = Field(ge=0.0)
    mode: ContactMode


class PaddleBodySettings(BaseModel):
    """
    A driven paddle body.

    :ivar mass_kg: Body mass (it collides with nothing, so this only sets what PhysX integrates).
    :ivar inertia_kgm2: Principal moments, ``(x, y, z)``.
    :ivar gravity: False: the drive alone moves it.
    :ivar kinematic: False: PhysX ignores velocities written to kinematic bodies.
    :ivar collides_with_ball: False: the paddle contact is hybrid.
    :ivar effective_mass_kg: The mass the hybrid impact model gives the ball to push against.
    :ivar rigid_body_impact: Resolve hybrid paddle hits as a free rigid paddle (the effective mass plus the paddle
        inertia about the blade centre), so an off-centre hit meets less mass and turns the blade; false: a point mass.
    :ivar max_drive_speed_mps: Clamp on the commanded linear velocity, so a bad target never flings it.
    :ivar blade_edge: The blade has the oracle's rounded edge (:mod:`lib.physics.reference.blade`): a touch on it is
        found by the solid gap and resolved with ``rim_contact``, an unvalidated prior. False: the face-only blade, through
        whose edge a ball passes, as a control.
    """

    model_config = _FROZEN

    mass_kg: float = Field(gt=0.0)
    inertia_kgm2: tuple[float, float, float]
    gravity: bool = False
    kinematic: bool = False
    collides_with_ball: bool = False
    effective_mass_kg: float = Field(gt=0.0)
    rigid_body_impact: bool = True
    max_drive_speed_mps: float = 40.0
    blade_edge: bool = True


class EngineSettings(BaseModel):
    """
    Everything the game adapter authors, from one parameter set.

    :ivar scene: The PhysX scene.
    :ivar ball: The ball body.
    :ivar table: Ball-table contact.
    :ivar net: Ball-net contact.
    :ivar floor: Ball-floor contact.
    :ivar rubber: Ball-rubber contact (always hybrid).
    :ivar paddle: The paddle bodies.
    :ivar ball_integrator: Who integrates the game ball (:data:`BallIntegrator`).
    :ivar engine_dt_s: The PhysX step under the ``rk4`` integrator (``None``: the game step, ``scene.physics_dt_s``).
        The game step stays ``scene.physics_dt_s``: the sweeps and the impact law run at it either way.
    """

    model_config = _FROZEN

    scene: SceneSettings
    ball: BallBodySettings
    table: SurfaceSettings
    net: SurfaceSettings
    floor: SurfaceSettings
    rubber: SurfaceSettings
    paddle: PaddleBodySettings
    ball_integrator: BallIntegrator = "physx"
    engine_dt_s: float | None = Field(default=None, gt=0.0)

    @property
    def physx_dt_s(self) -> float:
        """The step PhysX runs at."""

        return self.engine_dt_s if self.ball_integrator == "rk4" and self.engine_dt_s is not None else self.scene.physics_dt_s


def canonical_settings(
    params: BallTableParams,
    *,
    physics_dt_s: float,
    table_mode: ContactMode = "hybrid",
    net_mode: ContactMode = "hybrid",
    floor_mode: ContactMode = "engine",
    ball_integrator: BallIntegrator = "physx",
    engine_dt_s: float | None = None,
    blade_edge: bool = True,
) -> EngineSettings:
    """
    The adapter's settings for a parameter set.

    Table and net default to ``hybrid``, the calibrated contact model: on
    PhysX (PGS) the native table applies
    friction per axis rather than as a Coulomb cone (up to 25% excess slip on
    cross-court bounces), sticks early near the stick/slip boundary (up to
    0.47 m/s) and adds about 6% spin error through speculative contact, and
    the native net cord mis-deflects 15-25 m/s edge hits by up to 7.6 m/s;
    normal restitution and the drop test pass natively. ``engine`` keeps
    PhysX's own contact selectable for comparison.

    :param params: Physics parameters (the contract).
    :param physics_dt_s: Physics step.
    :param table_mode: How the table contact is resolved.
    :param net_mode: How the net contact is resolved (the net box; the posts stay native).
    :param floor_mode: How the floor contact is resolved: ``hybrid`` for the replay (with the room's walls, ceiling and
        furniture; native PhysX friction gains energy on a rolling ball), ``engine`` (PhysX) by default.
    :param ball_integrator: Who integrates the game ball; ``rk4`` needs every contact hybrid (PhysX never meets the ball).
    :param engine_dt_s: The PhysX step under ``rk4``: a whole number of game steps.
    :param blade_edge: The paddles' rounded edge (the oracle's default); False for face-only data and controls.
    :return: The settings.
    :raises ValueError: For ``rk4`` with an engine contact, or an engine step that is not whole game steps
    """

    if ball_integrator == "rk4":
        if "engine" in (table_mode, net_mode, floor_mode):
            raise ValueError("the rk4 ball integrator needs hybrid table, net and floor contacts: PhysX never meets the game ball")
        if engine_dt_s is not None:
            ratio = engine_dt_s / physics_dt_s
            if ratio < 1.0 - 1e-9 or abs(ratio - round(ratio)) > 1e-6:
                raise ValueError(f"the engine step {engine_dt_s} s must be a whole number of game steps of {physics_dt_s} s")
    paddle = params.paddle
    blade_mass = 0.17
    width, length, thickness = 2.0 * paddle.blade_half_width, 2.0 * paddle.blade_half_length, paddle.thickness
    inertia = (
        blade_mass * (length**2 + thickness**2) / 12.0,
        blade_mass * (width**2 + thickness**2) / 12.0,
        blade_mass * (width**2 + length**2) / 12.0,
    )
    return EngineSettings(
        scene=SceneSettings(gravity_mps2=params.aero.gravity, physics_dt_s=physics_dt_s),
        ball=BallBodySettings(radius_m=params.ball.radius, mass_kg=params.ball.mass, inertia_kgm2=params.ball.inertia),
        table=SurfaceSettings(restitution=params.table_contact.restitution, friction=params.table_contact.friction, mode=table_mode),
        net=SurfaceSettings(restitution=params.net_contact.restitution, friction=params.net_contact.friction, mode=net_mode),
        floor=SurfaceSettings(restitution=params.floor_contact.restitution, friction=params.floor_contact.friction, mode=floor_mode),
        rubber=SurfaceSettings(restitution=params.rubber_contact.restitution, friction=params.rubber_contact.friction, mode="hybrid"),
        paddle=PaddleBodySettings(mass_kg=blade_mass, inertia_kgm2=inertia, effective_mass_kg=paddle.effective_mass, blade_edge=blade_edge),
        ball_integrator=ball_integrator,
        engine_dt_s=engine_dt_s if ball_integrator == "rk4" else None,
    )


__all__ = [
    "BallBodySettings",
    "BallIntegrator",
    "ContactMode",
    "EngineSettings",
    "PaddleBodySettings",
    "SceneSettings",
    "SurfaceSettings",
    "canonical_settings",
]
