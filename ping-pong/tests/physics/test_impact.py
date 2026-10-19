"""The rigid-ball impact law: energy, the elastic limit, the e^2 rebound, the off-centre rigid paddle and the net's catch law."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lib.physics.params import BallParams, BallTableParams, ContactMaterial, PaddleParams
from lib.physics.reference.impact import point_effective_mass, resolve_impact
from lib.physics.reference.simulate import simulate

PADDLE = PaddleParams(blade_half_length=0.0785, blade_half_width=0.075, thickness=0.0136, effective_mass=0.17)
PADDLE_INERTIA = np.diag(PADDLE.inertia)


def _kinetic(ball: BallParams, v: np.ndarray, w: np.ndarray) -> float:
    return 0.5 * ball.mass * float(v @ v) + 0.5 * ball.inertia * float(w @ w)


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(size=3)
    return n / np.linalg.norm(n)


def _approaching(rng: np.random.Generator, normal: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """A random velocity moving into the surface at 0.5 m/s or more."""

    v = rng.normal(size=3) * scale
    return v - float(v @ normal) * normal - float(rng.uniform(0.5, scale)) * normal


def test_an_elastic_frictionless_impact_only_reverses_the_normal_velocity(base_params: BallTableParams) -> None:
    ball = base_params.ball
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = _random_unit(rng)
        v, w = _approaching(rng, n), rng.normal(size=3) * 300.0
        result = resolve_impact(v, w, ball, n, ContactMaterial(restitution=1.0, friction=0.0))
        assert np.allclose(result.velocity, v - 2.0 * float(v @ n) * n, rtol=0, atol=1e-12)
        assert np.array_equal(result.spin, w)


def test_energy_never_increases_and_momentum_is_conserved_with_a_moving_surface(base_params: BallTableParams) -> None:
    ball = base_params.ball
    rng = np.random.default_rng(2)
    for _ in range(2000):
        n = _random_unit(rng)
        material = ContactMaterial(
            restitution=float(rng.uniform(0.0, 1.0)),
            friction=float(rng.choice([0.0, rng.uniform(0.0, 0.3), rng.uniform(0.3, 3.0), 50.0])),
            tangential_restitution=float(rng.choice([0.0, rng.uniform(0.0, 1.0), 1.0])),
            normal_spin_retention=float(rng.choice([1.0, rng.uniform(0.0, 1.0), 0.0])),
        )
        v = _approaching(rng, n)
        w = rng.normal(size=3) * rng.choice([0.0, 100.0, 1000.0])
        assert resolve_impact(v, w, ball, n, material).energy_change <= 1e-15
        mass, u_s = float(rng.choice([0.05, 0.3, 2.0])), rng.normal(size=3) * 5.0
        moving = resolve_impact(v + u_s, w, ball, n, material, surface_velocity=u_s, effective_mass=mass)
        u_after = u_s + moving.surface_velocity_change
        before = _kinetic(ball, v + u_s, w) + 0.5 * mass * float(u_s @ u_s)
        after = _kinetic(ball, moving.velocity, moving.spin) + 0.5 * mass * float(u_after @ u_after)
        assert after - before <= 1e-13 * before
        assert np.allclose(ball.mass * moving.velocity + mass * u_after, ball.mass * (v + u_s) + mass * u_s, rtol=0, atol=1e-13)


def test_a_normal_drop_rebounds_to_e_squared_of_its_height(vacuum_params: BallTableParams) -> None:
    for restitution in (0.5, 0.876, 0.95):
        params = vacuum_params.with_values({"table_restitution": restitution})
        radius = params.ball.radius
        trajectory = simulate([-0.7, 0.2, radius + 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], params, max_time=2.0)
        first, second = trajectory.events_of("table")[:2]
        between = (trajectory.times > first.time) & (trajectory.times < second.time)
        assert float(trajectory.positions[between, 2].max()) - radius == pytest.approx(restitution**2 * 0.30, abs=1e-7)


def test_the_paddle_inertia_gives_the_off_centre_effective_masses() -> None:
    """0.17 kg at the blade centre, about 0.077 kg 5 cm along the blade and 0.073 kg 5 cm across it."""

    normal = [0.0, 0.0, 1.0]
    for (along, across), expected in {(0.0, 0.0): 0.17, (0.05, 0.0): 0.07699, (0.0, 0.05): 0.07320, (-0.04, -0.02): 0.08565}.items():
        assert point_effective_mass(0.17, PADDLE_INERTIA, [along, across, 0.0], normal) == pytest.approx(expected, rel=2e-4)


def test_off_centre_impacts_conserve_momentum_and_angular_momentum_and_never_gain_energy(base_params: BallTableParams) -> None:
    ball = base_params.ball
    m, i_ball, radius = ball.mass, ball.inertia, ball.radius
    rng = np.random.default_rng(14)
    for _ in range(1000):
        rotation = Rotation.random(rng=rng).as_matrix()
        normal = rotation[:, 2]
        in_plane = rng.uniform(-0.08, 0.08, size=2)
        offset = in_plane[0] * rotation[:, 0] + in_plane[1] * rotation[:, 1] + 0.0068 * normal
        inertia = rotation @ np.diag(PADDLE_INERTIA.diagonal() * rng.uniform(0.75, 2.0, size=3)) @ rotation.T
        mass = float(rng.uniform(0.1, 0.4))
        material = ContactMaterial(
            restitution=float(rng.uniform(0, 1)),
            friction=float(rng.choice([0.0, rng.uniform(0, 0.5), rng.uniform(0.5, 2.0)])),
            tangential_restitution=float(rng.uniform(0, 1)),
        )
        paddle_v, paddle_w = rng.normal(size=3) * 5.0, rng.normal(size=3) * 20.0
        u_s = paddle_v + np.cross(paddle_w, offset)
        v, w = _approaching(rng, normal) + u_s, rng.normal(size=3) * 400.0
        result = resolve_impact(v, w, ball, normal, material, surface_velocity=u_s, effective_mass=mass, contact_offset=offset, surface_inertia=inertia)
        contact_point = rng.normal(size=3)
        ball_centre, paddle_centre = contact_point + radius * normal, contact_point - offset
        paddle_v2, paddle_w2 = paddle_v + result.surface_velocity_change, paddle_w + result.surface_angular_velocity_change
        assert np.allclose(m * result.velocity + mass * paddle_v2, m * v + mass * paddle_v, rtol=0, atol=1e-12)
        angular_before = m * np.cross(ball_centre, v) + i_ball * w + mass * np.cross(paddle_centre, paddle_v) + inertia @ paddle_w
        angular_after = m * np.cross(ball_centre, result.velocity) + i_ball * result.spin + mass * np.cross(paddle_centre, paddle_v2) + inertia @ paddle_w2
        assert np.allclose(angular_after, angular_before, rtol=0, atol=1e-12)
        energy_before = _kinetic(ball, v, w) + 0.5 * mass * float(paddle_v @ paddle_v) + 0.5 * float(paddle_w @ inertia @ paddle_w)
        energy_after = _kinetic(ball, result.velocity, result.spin) + 0.5 * mass * float(paddle_v2 @ paddle_v2) + 0.5 * float(paddle_w2 @ inertia @ paddle_w2)
        assert energy_after - energy_before <= 1e-12 * energy_before


def test_the_catch_law_damps_motion_along_the_mesh_and_spin_and_never_gains_energy(base_params: BallTableParams) -> None:
    ball = base_params.ball
    rng = np.random.default_rng(15)
    for _ in range(1000):
        n = _random_unit(rng)
        material = ContactMaterial(
            restitution=float(rng.uniform(0, 1)),
            friction=float(rng.uniform(0, 2)),
            law="catch",
            tangential_damping=float(rng.uniform(0, 1)),
            spin_damping=float(rng.uniform(0, 1)),
        )
        relative, w = _approaching(rng, n), rng.normal(size=3) * 400.0
        result = resolve_impact(relative, w, ball, n, material)
        assert result.regime == "catch"
        normal_part = float(relative @ n)
        expected = -material.restitution * normal_part * n + (1.0 - material.tangential_damping) * (relative - normal_part * n)
        assert np.allclose(result.velocity, expected, atol=1e-12)
        assert np.allclose(result.spin, (1.0 - material.spin_damping) * w, atol=1e-12)
        assert _kinetic(ball, result.velocity, result.spin) <= _kinetic(ball, relative, w) * (1.0 + 1e-13)


def test_the_catch_law_ignores_spin_so_backspin_cannot_lift_the_ball(base_params: BallTableParams) -> None:
    mesh = ContactMaterial(restitution=0.2, friction=0.3, law="catch", tangential_damping=0.8, spin_damping=0.7)
    face, v = np.array([1.0, 0.0, 0.0]), np.array([-5.0, 0.0, 0.79])
    for spin_y in (-200.0, 0.0, 200.0):
        assert resolve_impact(v, [0.0, spin_y, 0.0], base_params.ball, face, mesh).velocity[2] == pytest.approx(0.2 * 0.79, rel=1e-12)
    rigid = resolve_impact(v, [0.0, -200.0, 0.0], base_params.ball, face, mesh.model_copy(update={"law": "rigid"}))
    assert rigid.velocity[2] > 0.79 + 1.0, "a rigid face turns backspin into lift"
    assert math.isfinite(rigid.energy_change)
