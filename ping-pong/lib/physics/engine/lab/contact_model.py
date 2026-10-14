"""
The paddle effective-mass decision: which mass the ball pushes against, and which rubber restitution goes with it.

The cited rubber restitution curves (Sony AI 2026, Gossard 2026) were measured
on rackets held rigidly by a robot, so they are *apparent* values: the
ball's rebound relative to a racket that barely recoils. For a normal impact
on a racket of effective mass ``M`` (moving or not) the apparent restitution
measured against the racket's pre-impact velocity is::

    e_app = (e M - m) / (M + m)

with ``e`` the intrinsic restitution of the ball-rubber-blade contact and
``m`` the ball's mass. Two consistent choices follow:

``rigid_apparent``
    ``M = inf`` and ``e = e_app``: the simulated paddle never recoils and
    reproduces the measured rebounds exactly, for any hand.

``human_intrinsic`` (recommended)
    ``M = paddle_effective_mass`` (0.17 kg nominal: a 0.17-0.19 kg racket whose
    grip barely couples the hand in a sub-millisecond impact) and the intrinsic
    ``e`` re-derived so that a racket of the measurement rig's mass would show
    the cited apparent value. With a rigid rig (``rigid_hold_mass = inf``) the
    intrinsic value equals the cited one, and a hand-held paddle returns the
    ball 0.027 lower in apparent restitution, about 2 % slower off a moving
    blade, which is the physical effect of the racket's recoil.

Pure: numpy-free arithmetic on the :mod:`lib.physics` contract.
"""

from __future__ import annotations

import math

from lib.physics.engine.lab.batch import PaddleMassModel
from lib.physics.params import BallTableParams


def apparent_restitution(intrinsic: float, effective_mass: float, ball_mass: float) -> float:
    """
    The rebound restitution measured against the racket's pre-impact velocity.

    :param intrinsic: Intrinsic normal restitution of the contact.
    :param effective_mass: Racket effective mass at the impact point, kg (``inf``: rigid).
    :param ball_mass: Ball mass, kg.
    :return: ``(e M - m) / (M + m)``.
    """

    if math.isinf(effective_mass):
        return intrinsic
    return (intrinsic * effective_mass - ball_mass) / (effective_mass + ball_mass)


def intrinsic_restitution(apparent: float, rigid_hold_mass: float, ball_mass: float) -> float:
    """
    The intrinsic restitution that a rig of ``rigid_hold_mass`` would report as ``apparent``.

    :param apparent: Cited (apparent) restitution.
    :param rigid_hold_mass: Effective mass of the measurement rig, kg (``inf``: perfectly rigid).
    :param ball_mass: Ball mass, kg.
    :return: ``(e_app (M + m) + m) / M``, clipped to at most 1.
    :raises ValueError: If no restitution in [0, 1] reproduces the apparent value
    """

    if math.isinf(rigid_hold_mass):
        return apparent
    intrinsic = (apparent * (rigid_hold_mass + ball_mass) + ball_mass) / rigid_hold_mass
    if intrinsic > 1.0:
        raise ValueError(f"apparent restitution {apparent} at a {rigid_hold_mass} kg rig needs an intrinsic restitution {intrinsic} > 1")
    return intrinsic


def apply_paddle_mass_model(params: BallTableParams, model: PaddleMassModel, rigid_hold_mass: float = math.inf) -> BallTableParams:
    """
    The parameter set the hybrid paddle impact uses under a mass model.

    :param params: Parameter set whose ``rubber_restitution`` is the cited (apparent) value.
    :param model: ``human_intrinsic`` keeps ``paddle_effective_mass`` and converts the restitution;
        ``rigid_apparent`` makes the paddle infinitely heavy and keeps the cited restitution.
    :param rigid_hold_mass: Effective mass of the rig the restitution was measured on, kg.
    :return: The converted parameter set (speed curves scale with the converted scalar).
    """

    if model == "rigid_apparent":
        return params.with_values({"paddle_effective_mass": math.inf})
    intrinsic = intrinsic_restitution(params.rubber_contact.restitution, rigid_hold_mass, params.ball.mass)
    return params.with_values({"rubber_restitution": intrinsic})


def outgoing_normal_speed(ball_speed: float, paddle_speed: float, intrinsic: float, effective_mass: float, ball_mass: float) -> float:
    """
    Ball speed along the normal after a head-on hit, ball approaching at ``ball_speed`` and the paddle at ``paddle_speed``.

    :param ball_speed: Ball speed towards the paddle, m/s.
    :param paddle_speed: Paddle speed towards the ball, m/s.
    :param intrinsic: Intrinsic restitution.
    :param effective_mass: Paddle effective mass, kg.
    :param ball_mass: Ball mass, kg.
    :return: Ball speed away from the paddle after impact, m/s.
    """

    closing = ball_speed + paddle_speed
    return apparent_restitution(intrinsic, effective_mass, ball_mass) * closing + paddle_speed


__all__ = ["apparent_restitution", "apply_paddle_mass_model", "intrinsic_restitution", "outgoing_normal_speed"]
