"""
Per-environment PhysX values: materials, masses, caps and which pairs collide.

Pure. PhysX combines two materials with the higher-priority combine mode of
the pair; every material here uses ``min`` (as the game adapter does), and
the ball carries restitution 1 and a friction above every surface's, so each
ball-surface pair takes exactly the surface's values, which are the
:mod:`lib.physics` pair laws. PhysX has no speed-dependent restitution, so a
native contact uses each law's scalar; the hybrid contacts use the full law.

The ball's own body and collider settings come from
:func:`lib.physics.engine.settings.canonical_settings`, shared with the game
adapter so the parity check compares like with like.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lib.physics.engine.lab.batch import BatchSpec, EnvSpec
from lib.physics.engine.lab.contact_model import apply_paddle_mass_model
from lib.physics.engine.settings import canonical_settings
from lib.physics.params import BallTableParams

COMBINE_MODE: str = "min"
"""PhysX restitution and friction combine mode of every material."""

BALL_RESTITUTION: float = 1.0
"""The ball material's own restitution, so ``min`` takes the surface's."""


@dataclass(frozen=True)
class EnvPhysics:
    """
    The USD values one environment's bodies, colliders and materials carry.

    :param ball_mass: kg.
    :param ball_radius: m.
    :param ball_inertia: Principal moment, kg m^2 (shell).
    :param max_angular_velocity_deg_s: The ball's PhysX angular-velocity cap.
    :param table_restitution: Ball-table pair restitution (scalar).
    :param table_friction: Ball-table pair friction.
    :param table_collides: False when the table contact is hybrid (the table has no ball collider).
    :param rubber_restitution: Ball-rubber pair restitution for native paddle contacts.
    :param rubber_friction: Ball-rubber pair friction.
    :param paddle_collides: True only for native paddle contacts.
    :param paddle_kinematic: Whether the paddle body is kinematic.
    :param paddle_mass: Paddle body mass, kg.
    :param paddle_inertia: Paddle body principal moments, kg m^2.
    :param net_restitution: Ball-net pair restitution.
    :param net_friction: Ball-net pair friction.
    :param net_collides: False when the net contact is hybrid (the net has no ball collider).
    """

    ball_mass: float
    ball_radius: float
    ball_inertia: float
    max_angular_velocity_deg_s: float
    table_restitution: float
    table_friction: float
    table_collides: bool
    rubber_restitution: float
    rubber_friction: float
    paddle_collides: bool
    paddle_kinematic: bool
    paddle_mass: float
    paddle_inertia: tuple[float, float, float]
    net_restitution: float
    net_friction: float
    net_collides: bool = True


def blade_box_inertia(mass: float, params: BallTableParams) -> tuple[float, float, float]:
    """
    Principal moments of the blade as a solid box: X towards the handle, Y across, Z through the faces.

    :param mass: Body mass, kg.
    :param params: Blade geometry.
    :return: ``(Ixx, Iyy, Izz)``, kg m^2.
    """

    lx, ly, lz = 2.0 * params.paddle.blade_half_length, 2.0 * params.paddle.blade_half_width, params.paddle.thickness
    return (mass * (ly**2 + lz**2) / 12.0, mass * (lx**2 + lz**2) / 12.0, mass * (lx**2 + ly**2) / 12.0)


def engine_params(spec: BatchSpec) -> dict[str, BallTableParams]:
    """
    Each parameter set as the adapter realises it: the paddle mass model applied.

    :param spec: The batch.
    :return: Converted parameter sets by key.
    """

    return {key: apply_paddle_mass_model(params, spec.engine.paddle_mass_model, spec.engine.rigid_hold_mass_kg) for key, params in spec.param_sets.items()}


def env_physics(spec: BatchSpec, env: EnvSpec, params: BallTableParams) -> EnvPhysics:
    """
    The USD values of one environment.

    :param spec: The batch (engine configuration).
    :param env: The environment.
    :param params: Its converted parameter set (:func:`engine_params`).
    :return: Values to author.
    """

    if env.paddle_mode == "pd":
        # The PD body realises the hand-held effective mass and the contract's head inertia natively, whatever the
        # hybrid mass model says: the same free rigid body the reference strikes off-centre.
        raw = spec.param_sets[env.params].paddle
        mass = raw.effective_mass
        if not math.isfinite(mass):
            raise ValueError(f"environment {env.label!r}: a PD paddle needs a finite effective mass")
        inertia = raw.inertia
    else:
        mass = spec.engine.heavy_paddle_mass_kg
        inertia = blade_box_inertia(mass, params)
    return EnvPhysics(
        ball_mass=params.ball.mass,
        ball_radius=params.ball.radius,
        ball_inertia=params.ball.inertia,
        max_angular_velocity_deg_s=env.max_angular_velocity_deg_s,
        table_restitution=params.table_contact.restitution,
        table_friction=params.table_contact.friction,
        table_collides=env.table_contact == "native",
        rubber_restitution=params.rubber_contact.restitution,
        rubber_friction=params.rubber_contact.friction,
        paddle_collides=env.paddle_mode != "none" and env.paddle_contact == "native",
        paddle_kinematic=env.paddle_mode == "kinematic",
        paddle_mass=mass,
        paddle_inertia=inertia,
        net_restitution=params.net_contact.restitution,
        net_friction=params.net_contact.friction,
        net_collides=env.net_contact == "native",
    )


def ball_friction(spec: BatchSpec) -> float:
    """
    The ball material's own friction: the shared canonical value, raised if any surface asks for more.

    :param spec: The batch.
    :return: A friction no pair's ``min`` will pick over the surface's own.
    """

    canonical = canonical_settings(spec.geometry, physics_dt_s=spec.engine.physics_dt_s).ball.friction
    highest = max(max(p.table_contact.friction, p.rubber_contact.friction, p.net_contact.friction) for p in spec.param_sets.values())
    return max(canonical, highest + 0.5)


def ball_body_settings(spec: BatchSpec) -> dict[str, float]:
    """
    The ball body and collider settings shared with the game adapter.

    :param spec: The batch.
    :return: Damping, thresholds, depenetration and iteration settings by name.
    """

    ball = canonical_settings(spec.geometry, physics_dt_s=spec.engine.physics_dt_s).ball
    return {
        "linear_damping": ball.linear_damping,
        "angular_damping": ball.angular_damping,
        "sleep_threshold": ball.sleep_threshold,
        "stabilization_threshold": ball.stabilization_threshold,
        "max_depenetration_velocity_mps": ball.max_depenetration_velocity_mps,
    }


__all__ = ["BALL_RESTITUTION", "COMBINE_MODE", "EnvPhysics", "ball_body_settings", "ball_friction", "blade_box_inertia", "engine_params", "env_physics"]
