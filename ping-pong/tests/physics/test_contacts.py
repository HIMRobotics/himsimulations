"""The replay's hybrid contacts: the swept touch, the imposed state, the board's geometry and sustained contact."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lib.physics.engine.contacts import (
    REST_SPEED_MPS,
    TABLE_START_SLOP_M,
    BladePose,
    SweptContact,
    crossed_table,
    impose_contact,
    sustained_material,
    sweep_blade,
    sweep_box,
    sweep_table,
    sweep_table_box,
    table_board,
)
from lib.physics.engine.settings import canonical_settings
from lib.physics.params import nominal_params
from lib.physics.reference.flight import FlightModel
from lib.physics.reference.impact import resolve_impact

PARAMS = nominal_params()
DT = 1.0 / 3840.0
RADIUS = PARAMS.ball.radius
HALF = {"half_length": 0.5 * PARAMS.table.length, "half_width": 0.5 * PARAMS.table.width}
MODEL = FlightModel.from_params(PARAMS)
UP = np.array([0.0, 0.0, 1.0])


def _blade(x: float, velocity: float = 0.0) -> BladePose:
    rotation = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return BladePose(position=np.array([x, 0.0, 0.2]), rotation=rotation, velocity=np.array([velocity, 0.0, 0.0]), angular_velocity=np.zeros(3))


class _GravityOnly:
    """No air force: the step's acceleration is gravity alone, so the exact path is closed-form."""

    @staticmethod
    def wrench(velocity: np.ndarray, spin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del velocity, spin
        return np.zeros(3), np.zeros(3)


def test_the_replay_settings_are_pgs_ccd_hybrid_and_rk4() -> None:
    settings = canonical_settings(PARAMS, physics_dt_s=DT, floor_mode="hybrid", ball_integrator="rk4", engine_dt_s=10 * DT)
    assert settings.scene.solver == "PGS" and settings.scene.ccd and not settings.scene.gpu_dynamics
    assert not settings.paddle.kinematic and not settings.paddle.collides_with_ball
    assert settings.rubber.mode == settings.table.mode == settings.net.mode == settings.floor.mode == "hybrid"
    with pytest.raises(ValueError, match="hybrid"):
        canonical_settings(PARAMS, physics_dt_s=DT, table_mode="engine", floor_mode="hybrid", ball_integrator="rk4")


def test_the_blade_sweep_finds_the_first_touch_and_a_moving_blade_returns_the_ball_faster() -> None:
    gap, dt = RADIUS + 0.5 * PARAMS.paddle.thickness, 1.0 / 960.0
    sizes = {"radius": RADIUS, "half_thickness": 0.5 * PARAMS.paddle.thickness, "half_width": 0.075, "half_length": 0.0785}
    ball = np.array([gap + 0.010, 0.0, 0.2, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    contact, tunnelled = sweep_blade(ball, _blade(0.0), _blade(0.0), dt, **sizes)
    assert contact is not None and not tunnelled and math.isclose(contact.fraction, 0.010 / (20.0 * dt), rel_tol=1e-6)
    beside = ball.copy()
    beside[1] = 0.2
    assert sweep_blade(beside, _blade(0.0), _blade(0.0), dt, **sizes)[0] is None
    slow = np.array([gap + 0.005, 0.0, 0.2, -8.0, 0.0, 0.0, 0.0, -100.0, 0.0])
    contact, _ = sweep_blade(slow, _blade(0.0, 3.0), _blade(3.0 * dt, 3.0), dt, **sizes)
    assert contact is not None
    imposed = impose_contact(contact, PARAMS.rubber_contact, PARAMS, dt, effective_mass=PARAMS.paddle.effective_mass)
    reference = resolve_impact(
        contact.ball[3:6],
        contact.ball[6:9],
        PARAMS.ball,
        contact.normal,
        PARAMS.rubber_contact,
        surface_velocity=contact.surface_velocity,
        effective_mass=PARAMS.paddle.effective_mass,
    )
    assert imposed is not None and imposed.velocity[0] > 8.0
    assert np.allclose(imposed.velocity, reference.velocity) and np.allclose(imposed.spin, reference.spin)


@pytest.mark.parametrize("phase", [0.05, 0.3, 0.55, 0.8, 0.97])
def test_the_imposed_state_lands_the_step_on_the_event_solution_at_every_contact_phase(phase: float) -> None:
    g, vz = PARAMS.aero.gravity, -3.0
    start = np.array([0.6, 0.1, RADIUS - vz * phase * DT, 2.0, 0.0, vz, 0.0, 40.0, 0.0])
    contact = sweep_table(start, DT, radius=RADIUS, **HALF)
    assert contact is not None and math.isclose(contact.fraction, phase, rel_tol=1e-9)
    imposed = impose_contact(contact, PARAMS.table_contact, PARAMS, DT, effective_mass=math.inf, flight=_GravityOnly())
    assert imposed is not None
    gravity = np.array([0.0, 0.0, -g])
    lead, rest = phase * DT, (1.0 - phase) * DT
    bounce = resolve_impact(start[3:6] + gravity * lead, start[6:9], PARAMS.ball, UP, PARAMS.table_contact)
    engine_velocity = imposed.velocity + gravity * DT
    engine_position = imposed.start_position + engine_velocity * DT
    assert np.allclose(engine_velocity, bounce.velocity + gravity * rest, atol=1e-12)
    assert np.allclose(engine_position, contact.ball[:3] + bounce.velocity * rest + 0.5 * gravity * rest**2, atol=1e-12)


def test_a_top_bounce_is_exactly_the_plane_sweep_and_a_sunk_ball_still_touches() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        x, y = rng.uniform(-HALF["half_length"] + 0.01, HALF["half_length"] - 0.01), rng.uniform(-HALF["half_width"] + 0.01, HALF["half_width"] - 0.01)
        vz = -rng.uniform(0.5, 8.0)
        state = np.array([x, y, RADIUS + rng.uniform(-TABLE_START_SLOP_M, -vz * DT), rng.normal(0, 3), rng.normal(0, 2), vz, *rng.normal(0, 300, 3)])
        plane, board = sweep_table(state, DT, radius=RADIUS, **HALF), sweep_table_box(state, DT, radius=RADIUS, **HALF)
        assert (plane is None) == (board is None)
        if plane is not None and board is not None:
            assert board.fraction == plane.fraction and np.array_equal(board.ball, plane.ball)
    deep = np.array([0.7, 0.0, RADIUS - 2.0 * TABLE_START_SLOP_M, 8.0, 0.0, -2.6, 0.0, 0.0, 0.0])
    assert sweep_table(deep, DT, radius=RADIUS, **HALF) is None, "deeper than the slop the ball is under the table"


def test_edges_faces_and_the_underside_deflect_by_their_geometry() -> None:
    half_length, half_width = HALF["half_length"], HALF["half_width"]
    dx = 0.005
    edge = np.array([half_length + dx, 0.2, math.sqrt(RADIUS**2 - dx**2) + 0.0005, -0.3, 0.0, -3.0, 0.0, 0.0, 0.0])
    assert sweep_table(edge, DT, radius=RADIUS, **HALF) is None, "the top plane alone misses an edge ball"
    contact = sweep_table_box(edge, DT, radius=RADIUS, **HALF)
    assert contact is not None and 0.2 < contact.normal[0] < 0.3
    side = np.array([0.4, -half_width - RADIUS - 0.001, -0.01, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    contact = sweep_table_box(side, DT, radius=RADIUS, **HALF)
    assert contact is not None and np.allclose(contact.normal, (0.0, -1.0, 0.0))
    under = np.array([0.3, 0.1, -0.025 - RADIUS - 0.0005, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    contact = sweep_table_box(under, DT, radius=RADIUS, **HALF)
    assert contact is not None and np.allclose(contact.normal, (0.0, 0.0, -1.0))
    inside = np.array([1.0, 0.0, -0.01, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert sweep_box(inside, DT, radius=RADIUS, box=table_board(half_length, half_width)) == (None, True)
    assert crossed_table(np.array([1.4, 0.0, -0.01, -2.0, 0, 0, 0, 0, 0]), inside, PARAMS)


def _touch(velocity: tuple[float, float, float], normal: np.ndarray = UP, spin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> SweptContact:
    centre = np.array([0.3, 0.1, RADIUS])
    return SweptContact(
        fraction=0.0, ball=np.concatenate([centre, velocity, spin]), normal=normal, surface_velocity=np.zeros(3), blade_offset_m=None, depth=0.0
    )


def test_a_slow_touch_pressed_by_gravity_is_sustained_and_a_bounce_is_not() -> None:
    bounce = _touch((1.0, 0.0, -2.0), spin=(0.0, 50.0, 0.0))
    plain = impose_contact(bounce, PARAMS.table_contact, PARAMS, DT, effective_mass=math.inf, flight=MODEL)
    rested = impose_contact(bounce, PARAMS.table_contact, PARAMS, DT, effective_mass=math.inf, flight=MODEL, rest_speed=REST_SPEED_MPS)
    assert plain is not None and rested is not None and not rested.sustained and np.array_equal(plain.velocity, rested.velocity)
    slow = _touch((0.8, 0.0, -0.03), spin=(0.0, 10.0, 0.0))
    imposed = impose_contact(slow, PARAMS.table_contact, PARAMS, DT, effective_mass=math.inf, flight=MODEL, rest_speed=REST_SPEED_MPS)
    assert imposed is not None and imposed.sustained and imposed.after[5] == pytest.approx(0.0, abs=1e-12)
    reference = resolve_impact(slow.ball[3:6], slow.ball[6:9], PARAMS.ball, UP, sustained_material(PARAMS.table_contact))
    assert np.allclose(imposed.after[3:6], reference.velocity, atol=1e-6)
    face = _touch((-0.03, 0.0, -1.0), normal=np.array([1.0, 0.0, 0.0]))
    falling = impose_contact(face, PARAMS.net_contact, PARAMS, DT, effective_mass=math.inf, flight=MODEL, rest_speed=REST_SPEED_MPS)
    assert falling is not None and not falling.sustained, "gravity does not press a ball onto a vertical face"
