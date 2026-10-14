"""The flight ODE and its RK4 integrator: closed forms, fourth-order convergence and the Magnus direction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lib.physics.params import BallTableParams
from lib.physics.reference.flight import FlightModel, flight_events, integrate, integrate_until_event

TABLE = 0


def _state(position: tuple[float, float, float], velocity: tuple[float, float, float], spin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[float]:
    return [*position, *velocity, *spin]


def test_vacuum_flight_is_the_closed_form_parabola(vacuum_params: BallTableParams) -> None:
    model = FlightModel.from_params(vacuum_params)
    p0, v0 = np.array([-1.2, 0.3, 0.25]), np.array([6.0, -0.5, 2.0])
    times, states = integrate(model, _state(tuple(p0), tuple(v0), (10.0, 300.0, -40.0)), 1.0, dt=1e-4)
    g = np.array([0.0, 0.0, -vacuum_params.aero.gravity])
    # RK4 is exact for a quadratic, so what remains is round-off over 10,000 steps.
    assert np.max(np.abs(states[:, 0:3] - (p0 + np.outer(times, v0) + 0.5 * np.outer(times**2, g)))) < 1e-10
    assert np.allclose(states[:, 6:9], [10.0, 300.0, -40.0], rtol=0, atol=0)


def test_a_falling_ball_follows_the_quadratic_drag_closed_form_to_terminal_velocity(base_params: BallTableParams) -> None:
    ball, aero = base_params.ball, base_params.aero
    terminal = math.sqrt(2.0 * ball.mass * aero.gravity / (aero.air_density * aero.drag_coefficient * ball.area))
    times, states = integrate(FlightModel.from_params(base_params), _state((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 2.0, dt=1e-4, record_every=100)
    g = aero.gravity
    assert np.max(np.abs(states[:, 5] + terminal * np.tanh(g * times / terminal))) < 1e-10
    assert np.max(np.abs(states[:, 2] + (terminal**2 / g) * np.log(np.cosh(g * times / terminal)))) < 1e-10
    assert 8.0 < terminal < 10.0


def test_rk4_converges_at_fourth_order_and_well_below_a_micron_at_a_millisecond(base_params: BallTableParams) -> None:
    model = FlightModel.from_params(base_params)
    start = _state((-1.3, 0.2, 0.3), (12.0, -1.0, 3.0), (-50.0, 600.0, 120.0))
    reference = integrate(model, start, 0.8, dt=1e-5, record_every=10**9)[1][-1]
    errors = [float(np.linalg.norm(integrate(model, start, 0.8, dt=dt, record_every=10**9)[1][-1][:3] - reference[:3])) for dt in (8e-3, 4e-3)]
    assert 12.0 < errors[0] / errors[1] < 20.0
    coarse = integrate(model, start, 0.8, dt=1e-3, record_every=10**9)[1][-1]
    assert float(np.linalg.norm(coarse[:3] - reference[:3])) < 1e-6


def test_topspin_dips_backspin_floats_and_sidespin_curves(base_params: BallTableParams) -> None:
    model = FlightModel.from_params(base_params)
    velocity = (10.0, 0.0, 0.0)
    plain = model.derivative(_state((0, 0, 0), velocity))
    top = model.derivative(_state((0, 0, 0), velocity, (0.0, 300.0, 0.0)))
    back = model.derivative(_state((0, 0, 0), velocity, (0.0, -300.0, 0.0)))
    side = model.derivative(_state((0, 0, 0), velocity, (0.0, 0.0, 300.0)))
    magnus = model.aero_gain * base_params.aero.lift.coefficient(base_params.ball.radius * 300.0 / 10.0) * 100.0
    assert top[5] - plain[5] == pytest.approx(-magnus, rel=1e-12)
    assert back[5] - plain[5] == pytest.approx(magnus, rel=1e-12)
    assert side[4] == pytest.approx(magnus, rel=1e-12)
    landing = {}
    for label, spin in (("top", (0.0, 300.0, 0.0)), ("none", (0.0, 0.0, 0.0)), ("back", (0.0, -300.0, 0.0))):
        segment = integrate_until_event(model, _state((-1.37, 0.0, 0.3), (5.5, 0.0, 1.5), spin), 0.0, 3.0, flight_events(base_params))
        assert segment.hit is not None and segment.hit.index == TABLE
        landing[label] = segment.hit.state[0]
    assert landing["top"] < landing["none"] < landing["back"]


def test_spin_about_the_direction_of_travel_has_no_lift(base_params: BallTableParams) -> None:
    model = FlightModel.from_params(base_params.with_values({"drag_spin_slope": 0.3}))
    plain = model.derivative(_state((0, 0, 0), (6.0, 0.0, 8.0)))
    rifled = model.derivative(_state((0, 0, 0), (6.0, 0.0, 8.0), (300.0, 0.0, 400.0)))
    assert np.allclose(rifled[3:6], plain[3:6], rtol=0, atol=1e-12)
