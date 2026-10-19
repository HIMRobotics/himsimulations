"""
The replay's ``rk4`` ball integrator: the game ball advanced by the adapter with the oracle's RK4, resumed from the
contact time after a hybrid contact. PhysX is replaced by :mod:`tests.physics.stand_in`; under ``rk4`` PhysX never
touches the game ball, so the ball's physics here is exactly the production code's. Not a simulation run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lib.physics.engine.ball_integrator import IntegratedBodies
from lib.physics.engine.settings import canonical_settings
from lib.physics.engine.stepper import GamePhysics
from lib.physics.params import nominal_params
from lib.physics.reference.flight import FlightModel, propagate
from lib.physics.reference.simulate import simulate
from tests.physics.stand_in import ReferenceBodies, StepWorld

PARAMS = nominal_params()
RADIUS = PARAMS.ball.radius
GAME_DT = 1.0 / 3840.0


def _physics(integrator: str = "rk4", engine_steps: int = 16, *, game_dt: float = GAME_DT) -> tuple[GamePhysics, StepWorld]:
    settings = canonical_settings(
        PARAMS,
        physics_dt_s=game_dt,
        table_mode="hybrid",
        net_mode="hybrid",
        floor_mode="hybrid",
        ball_integrator=integrator,  # type: ignore[arg-type]
        engine_dt_s=game_dt * engine_steps if integrator == "rk4" else None,
    )
    engine = ReferenceBodies(PARAMS, 1)
    physics = GamePhysics(PARAMS, settings, "/ball", {"suited": "/paddle"}, bodies=engine, room=False)
    physics.place_paddles({"suited": (np.array([0.0, 3.0, -0.6]), np.array([1.0, 0.0, 0.0, 0.0]))})
    world = StepWorld(engine, settings.physx_dt_s)
    physics.register(world)
    return physics, world


def _run(physics: GamePhysics, world: StepWorld, seconds: float) -> None:
    for _ in range(math.ceil(seconds / world.dt)):
        world.step()
    assert physics.error is None, physics.error


def test_free_flight_is_the_reference_models_rk4() -> None:
    model = FlightModel.from_params(PARAMS)
    state = np.array([-1.0, 0.0, 1.5, 12.0, 0.5, 3.0, 0.0, 400.0, 30.0])
    bodies = IntegratedBodies(ReferenceBodies(PARAMS, 1), model)
    bodies.write_ball(state[:3], state[3:6], state[6:9])
    steps = round(0.25 / GAME_DT)
    for _ in range(steps):
        bodies.integrate(GAME_DT)
    expected = propagate(model, state, 0.0, [steps * GAME_DT], dt=GAME_DT)[0]
    assert np.max(np.abs(bodies.ball() - expected)) < 1e-9


def test_a_contact_resumes_from_the_contact_time() -> None:
    model = FlightModel.from_params(PARAMS)
    bodies = IntegratedBodies(ReferenceBodies(PARAMS, 1), model)
    after = np.array([0.1, 0.0, RADIUS, 3.0, 0.0, 2.0, 0.0, 50.0, 0.0])
    bodies.impose(after, 0.25)
    bodies.integrate(GAME_DT)
    assert np.max(np.abs(bodies.ball() - propagate(model, after, 0.0, [0.75 * GAME_DT], dt=GAME_DT)[0])) < 1e-12


def test_the_drop_rebounds_to_the_reference_apex() -> None:
    """The 30 cm drop on the hybrid table: the replay's ball rebounds to the oracle's apex within 0.01 mm."""

    physics, world = _physics()
    physics.place_ball((-0.6, 0.25, RADIUS + 0.30), np.zeros(3), np.zeros(3), label="drop 0")
    _run(physics, world, 0.75)
    table = next(event for event in physics.events if event.surface == "table")
    reference = simulate([0.6, 0.25, RADIUS + 0.30, 0, 0, 0, 0, 0, 0], PARAMS, dt=1e-4, max_time=1.5, terminal_events=("floor",))
    bounce = next(event for event in reference.events if event.kind == "table")
    apex = float(np.max(reference.positions[reference.times > bounce.time, 2]))
    assert 0.23 <= physics.apexes[table.index] - RADIUS <= 0.26
    assert abs(physics.apexes[table.index] - apex) < 1e-5


@pytest.mark.parametrize("engine_steps", [1, 16])
def test_a_serve_gives_the_same_contacts_whatever_the_engine_step(engine_steps: int) -> None:
    reference_physics, reference_world = _physics(engine_steps=1)
    physics, world = _physics(engine_steps=engine_steps)
    for p, w in ((reference_physics, reference_world), (physics, world)):
        p.place_ball((-1.3, 0.1, 0.25), (4.5, 0.0, -0.5), (0.0, 150.0, 0.0), label="serve")
        _run(p, w, 1.0)
    ours, theirs = physics.events, reference_physics.events
    assert [e.surface for e in ours] == [e.surface for e in theirs] and len(ours) >= 2
    assert max(abs(a.time_s - b.time_s) for a, b in zip(ours, theirs, strict=True)) < 1e-8
    assert max(float(np.max(np.abs(a.after - b.after))) for a, b in zip(ours, theirs, strict=True)) < 1e-6


def test_the_integrated_flight_converges_where_semi_implicit_euler_does_not() -> None:
    """Step against half step: the second bounce agrees within 0.05 mm under rk4; PhysX's scheme differs by over 0.5 mm."""

    def landing(integrator: str, dt: float) -> np.ndarray:
        physics, world = _physics(integrator, engine_steps=1, game_dt=dt)
        physics.place_ball((-1.3, 0.1, 0.25), (4.5, 0.0, -0.5), (0.0, 150.0, 0.0), label="serve")
        _run(physics, world, 0.9)
        tables = [event for event in physics.events if event.surface == "table"]
        return np.asarray(tables[1].after[:3])

    rk4 = landing("rk4", GAME_DT) - landing("rk4", 0.5 * GAME_DT)
    euler = landing("physx", GAME_DT) - landing("physx", 0.5 * GAME_DT)
    assert 1e3 * np.max(np.abs(rk4)) < 0.05
    assert 1e3 * np.max(np.abs(euler)) > 0.5
