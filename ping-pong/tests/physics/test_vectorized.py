"""
The GPU ball physics (``lib.physics.vectorized``) against the numpy oracle, on the CPU in float64: the same canonical
contact sequences (table, board, net, posts, floor) with positions within 1e-6 m, for the eager batch and the
CUDA-graph layout, and a negative control showing the comparison can fail. Not a simulation run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lib.physics.canonical_events import CanonicalEvent, RawTouch, merge  # noqa: E402
from lib.physics.params import nominal_params  # noqa: E402
from lib.physics.reference.simulate import simulate  # noqa: E402
from lib.physics.reference.surfaces import classify_table_touch  # noqa: E402
from lib.physics.vectorized import BallBatch, VecParams  # noqa: E402
from lib.physics.vectorized.dense import DenseBallBatch  # noqa: E402

DT = 5e-4
T_END = 1.2
F64 = torch.float64
BOOKKEEPING = {"roll", "pivot", "table_plane_outside", "rest", "out_of_bounds", "max_time", "max_events"}
PARAMS = nominal_params()


def _launches(n: int, seed: int) -> np.ndarray:
    """Shots from the far half towards the near half: most land, some meet the net, the posts, the board or go long."""

    rng = np.random.default_rng(seed)
    states = np.zeros((n, 9))
    states[:, 0] = rng.uniform(-1.7, -0.4, n)
    states[:, 1] = rng.uniform(-0.9, 0.9, n)
    states[:, 2] = rng.uniform(0.05, 0.45, n)
    target = np.column_stack([rng.uniform(-0.1, 1.6, n), rng.uniform(-0.85, 0.85, n), np.full(n, 0.02)])
    flight = rng.uniform(0.25, 0.65, n)
    velocity = (target - states[:, :3]) / flight[:, None]
    velocity[:, 2] += 0.5 * 9.81 * flight
    states[:, 3:6] = velocity * rng.uniform(0.9, 1.15, (n, 1))
    states[:, 6:9] = rng.normal(scale=120.0, size=(n, 3))
    edge_ball = [1.375, 0.2, 0.05, -0.3, 0.0, -1.0, 0.0, 0.0, 0.0]
    post = [-0.5, 0.925, 0.08, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    side_face = [0.4, -0.8, -0.01, 0.0, 2.0, 0.5, 0.0, 0.0, 0.0]
    return np.vstack([states, edge_ball, post, side_face])


def _oracle(state: np.ndarray, params: Any) -> tuple[CanonicalEvent, ...]:
    trajectory = simulate(
        state, params, dt=DT, max_time=T_END, terminal_events=("floor",), bounds=(60.0, 60.0, 60.0), max_events=50, rolling=False, undercarriage=False
    )
    touches = []
    for index, event in enumerate(trajectory.events):
        if event.kind in BOOKKEEPING or event.contact is None:
            continue
        region = str(event.contact.region or "")
        region = classify_table_touch(event.contact.normal) if event.kind == "table" else ("" if event.kind == "floor" else region)
        after = np.asarray(event.state_after, float)
        point, normal = np.asarray(event.contact.point, float), np.asarray(event.contact.normal, float)
        touches.append(
            RawTouch(
                index, str(event.kind), region, None, float(event.time), point, normal, np.asarray(event.state_before, float), after, float(event.time), after
            )
        )
    return merge(touches)


def _eager(states: np.ndarray, overrides: dict[str, Any] | None = None) -> list[tuple[CanonicalEvent, ...]]:
    vec = VecParams.from_params(PARAMS, len(states), dtype=F64)
    if overrides:
        vec = vec.with_values(overrides)
    ball = BallBatch(vec, torch.tensor(states, dtype=F64), terminal_kinds=("floor",), max_events_per_step=6, compile=False)
    for _ in range(round(T_END / DT)):
        if not bool(ball.active.any()):
            break
        ball.step(DT)
    assert ball.log is not None
    return [tuple(e for e in ball.log.canonical(i, ("paddle",)) if e.t <= T_END + 1e-12) for i in range(len(states))]


def _dense(states: np.ndarray) -> list[tuple[CanonicalEvent, ...]]:
    ball = DenseBallBatch(VecParams.from_params(PARAMS, len(states), dtype=F64), torch.tensor(states, dtype=F64), dt=DT, max_events_per_step=6)
    for _ in range(round(T_END / DT)):
        if not bool(ball.active.any()):
            break
        ball.step()
    assert ball.log is not None and ball.log_lost == 0
    return [tuple(e for e in ball.log.canonical(i, ("paddle",)) if e.t <= T_END + 1e-12) for i in range(len(states))]


def _agree(a: tuple[CanonicalEvent, ...], b: tuple[CanonicalEvent, ...]) -> bool:
    return [e.key for e in a] == [e.key for e in b] and all(float(np.max(np.abs(x.before[:3] - y.before[:3]))) < 1e-6 for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def shots() -> tuple[np.ndarray, list[tuple[CanonicalEvent, ...]]]:
    states = _launches(45, 7)
    return states, [_oracle(state, PARAMS) for state in states]


def test_the_batch_reproduces_the_oracles_contact_sequences(shots: tuple[np.ndarray, list[Any]]) -> None:
    states, oracle = shots
    batch = _eager(states)
    regions = {(e.kind, e.region) for sequence in oracle for e in sequence}
    assert {("table", "top"), ("table", "edge"), ("table", "side"), ("net", "mesh"), ("net", "cord"), ("net", "post"), ("floor", "")} <= regions, regions
    assert all(_agree(a, b) for a, b in zip(oracle, batch, strict=True))


def test_the_cuda_graph_layout_reproduces_the_oracle(shots: tuple[np.ndarray, list[Any]]) -> None:
    states, oracle = shots
    assert all(_agree(a, b) for a, b in zip(oracle, _dense(states), strict=True))


def test_a_changed_table_restitution_is_detected(shots: tuple[np.ndarray, list[Any]]) -> None:
    """Negative control: a 2 % lower table restitution must move the events, or the comparison proves nothing."""

    states, oracle = shots
    lowered = torch.full((len(states),), float(PARAMS.table_contact.restitution) * 0.98, dtype=F64)
    changed = _eager(states, {"table_restitution": lowered})
    assert sum(not _agree(a, b) for a, b in zip(oracle, changed, strict=True)) >= 10
