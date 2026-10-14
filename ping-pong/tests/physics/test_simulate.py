"""Event-driven simulation: the ITTF drop test, the net's catch law and cord, and the terminal events."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from lib.physics.params import PRIORS_PATH, BallTableParams, load_priors
from lib.physics.reference.simulate import simulate

NET_FACE_HIT = [0.08, 0.02, 0.08, -5.0, 0.0, 0.79, 0.0, -200.0, 0.0]
"""A ball rising at 0.79 m/s with -200 rad/s of backspin meets the net face from the +X side."""


def _rebound(params: BallTableParams) -> float:
    radius = params.ball.radius
    trajectory = simulate([-0.7, 0.0, radius + 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], params, max_time=1.5)
    first, second = trajectory.events_of("table")[:2]
    between = (trajectory.times > first.time) & (trajectory.times < second.time)
    return float(trajectory.positions[between, 2].max()) - radius


@pytest.mark.parametrize("speed_curves", [True, False])
def test_the_cited_table_passes_the_ittf_drop_test_with_drag_on_and_0_876_does_not(speed_curves: bool) -> None:
    """ITTF: a ball dropped from 30 cm rebounds 23-26 cm, in air. The drag-free reading e = 0.876 must fail."""

    cited = load_priors(PRIORS_PATH).nominal(speed_curves=speed_curves)
    assert cited.aero.air_density > 1.0 and cited.aero.drag_coefficient > 0.3
    assert 0.23 <= _rebound(cited) <= 0.26
    vacuum_reading = cited.model_copy(update={"table_contact": cited.table_contact.model_copy(update={"restitution": 0.876, "restitution_curve": None})})
    assert _rebound(vacuum_reading) < 0.22


def test_the_dead_table_negative_control_leaves_the_ittf_band() -> None:
    assert _rebound(load_priors(PRIORS_PATH).nominal().with_values({"table_restitution": 0.30})) < 0.05


def test_a_net_face_hit_is_caught_and_drops_beside_the_net(base_params: BallTableParams) -> None:
    net = base_params.net_contact
    trajectory = simulate(NET_FACE_HIT, base_params, max_time=3.0)
    hit = trajectory.events[0]
    assert hit.kind == "net" and hit.contact is not None and hit.impact is not None
    assert hit.contact.region == "mesh" and hit.impact.regime == "catch"
    before, after = hit.state_before, hit.state_after
    assert after[3] == pytest.approx(-net.restitution * before[3], rel=1e-12)
    assert after[5] == pytest.approx((1.0 - net.tangential_damping) * before[5], rel=1e-12)
    assert float(trajectory.positions[trajectory.times > hit.time, 2].max()) < base_params.net.height, "a caught ball must not rise to the cord"
    landing = next(event for event in trajectory.events[1:] if event.kind == "table")
    assert 0.0 < landing.position[0] < 0.3


def test_the_cord_and_tape_stay_rigid_and_the_posts_are_steel(base_params: BallTableParams) -> None:
    params = base_params.with_values({"gravity": 0.0, "air_density": 0.0})
    tape_z = params.net.height - 0.5 * params.net.tape_height
    for z, region in ((0.05, "mesh"), (tape_z, "cord"), (params.net.height + 0.5 * params.ball.radius, "cord")):
        hit = simulate([-0.5, 0.0, z, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0], params, max_time=0.5).events[0]
        assert hit.kind == "net" and hit.contact is not None and hit.impact is not None
        assert hit.contact.region == region and (hit.impact.regime == "catch") == (region == "mesh")
    post = simulate([-0.5, params.net_half_span + 0.5 * params.ball.radius, 0.05, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0], params, max_time=0.5).events[0]
    assert post.kind == "net" and post.contact is not None and post.contact.region == "post" and post.contact.material.law == "rigid"


def test_terminal_conditions(vacuum_params: BallTableParams) -> None:
    radius = vacuum_params.ball.radius
    long_ball = simulate([1.0, 0.0, 0.3, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], vacuum_params, terminal_events=("table_plane_outside",))
    assert [event.kind for event in long_ball.events] == ["table_plane_outside"]
    bouncy = vacuum_params.with_values({"table_restitution": 0.5})
    settled = simulate([-0.7, 0.0, radius + 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], bouncy, max_time=5.0, rolling=False)
    speeds = [math.sqrt(2.0 * bouncy.aero.gravity * 0.3) * 0.5**k for k in range(1, 10)]
    expected_bounces = next(k for k, speed in enumerate(speeds, start=1) if speed < 0.05)
    assert [event.kind for event in settled.events] == ["table"] * expected_bounces + ["rest"]
    away = simulate([0.0, 0.0, 1.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0], vacuum_params, bounds=(6.0, 5.0, 3.0))
    assert away.terminal.kind == "out_of_bounds"
    with pytest.raises(ValueError):
        simulate([math.nan, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vacuum_params)


def test_mechanical_energy_never_rises_over_a_rally_shot(base_params: BallTableParams) -> None:
    params = base_params
    trajectory = simulate([-1.5, 0.1, 0.25, 5.0, -0.2, -1.0, 0.0, 150.0, 0.0], params, max_time=2.0)
    ball = params.ball
    energy = [
        0.5 * ball.mass * float(s[3:6] @ s[3:6]) + 0.5 * ball.inertia * float(s[6:9] @ s[6:9]) + ball.mass * params.aero.gravity * float(s[2])
        for s in np.asarray(trajectory.states)
    ]
    assert all(b <= a + 1e-12 for a, b in pairwise(energy))
