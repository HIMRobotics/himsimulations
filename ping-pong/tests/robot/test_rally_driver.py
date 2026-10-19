"""
The rally driver (:class:`lib.policy.rally.Rallies`) with HIM's real opponent, on CPU, fed synthetic contacts in the
coupling's format: who serves, which receives open, which returns are legal, and the grade's arithmetic.

Needs the opponent model: set ``PING_PONG_OPPONENT_MODEL`` to the ``ping-pong/him-opponent-model@v2`` zip, or to the
directory it unpacks to.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from lib.opponent import MODEL_FILES, load_him_opponent
from lib.physics.params import nominal_params
from lib.physics.vectorized.geometry import STATIC_COUNT, TABLE_TOP
from lib.physics.vectorized.params import VecParams
from lib.policy.grade import rally_metrics, wilson
from lib.policy.rally import OPPONENT, ROBOT, Rallies, bound_for_robot, feed_states

H = 0.005
"""The scene's physics step, s."""


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    spec = os.environ.get("PING_PONG_OPPONENT_MODEL")
    if not spec:
        pytest.skip("set PING_PONG_OPPONENT_MODEL to the ping-pong/him-opponent-model@v2 zip (or its unpacked directory)")
    path = Path(spec)
    if path.is_dir():
        return path
    out = tmp_path_factory.mktemp("opponent-model")
    with zipfile.ZipFile(path) as bundle:
        bundle.extractall(out)
    assert all((out / name).exists() for name in MODEL_FILES)
    return out


def rallies_of(model_dir: Path, n: int) -> tuple[Rallies, Any]:
    params = VecParams.from_params(nominal_params(), n, device="cpu", dtype=torch.float64)
    opponent = load_him_opponent(params, seed=0, model_dir=model_dir)
    stance = torch.as_tensor(opponent.rules.draw_stance(np.ones(n, dtype=np.int64), np.random.default_rng(0)), dtype=torch.float64)
    active = torch.ones(n, dtype=torch.bool)
    return Rallies(opponent, -stance, active, t0=0.0, h=H), params


def contacts(env: list[int], surface: list[int], time: list[float], point: list[list[float]], after: list[list[float]]) -> dict[str, Any]:
    return {
        "env": torch.tensor(env),
        "surface": torch.tensor(surface),
        "time": torch.tensor(time, dtype=torch.float64),
        "point": torch.tensor(point, dtype=torch.float64),
        "after": torch.tensor(after, dtype=torch.float64),
    }


def test_the_opponents_feeds_head_for_the_robots_half(model_dir: Path) -> None:
    rallies, params = rallies_of(model_dir, 32)
    feeds = torch.as_tensor(feed_states(rallies.opponent.rules, 32, np.random.default_rng(1)))
    assert bool((feeds[:, 3] < 0.0).all()), "every feed flies toward the robot's end (-x)"
    rows = torch.arange(32)
    bound = bound_for_robot(params, feeds, rows, torch.zeros(32, dtype=torch.bool))
    assert bound.float().mean() > 0.9, "the recorded feeds bounce on the robot's half"
    long = feeds.clone()
    long[:, 3:6] = torch.tensor([-14.0, 0.0, 3.0], dtype=torch.float64)
    assert not bool(bound_for_robot(params, long, rows, torch.zeros(32, dtype=torch.bool)).any()), "a ball flying long is let go"
    assert bool(bound_for_robot(params, long, rows, torch.ones(32, dtype=torch.bool)).all()), "a ball that already bounced is played"


def test_a_rally_counts_receives_hits_and_legal_returns(model_dir: Path) -> None:
    rallies, _params = rallies_of(model_dir, 3)
    rows = torch.arange(3)
    feeds = torch.as_tensor(feed_states(rallies.opponent.rules, 3, np.random.default_rng(2)))
    rallies.start(rows, feeds, 0.0)
    t = 0.0
    for _ in range(8):  # the receives are classified at the driver's cadence
        t += H
        rallies.ingest(None, t)
    assert rallies.robot_due().tolist() == [0, 1, 2]
    rallies.classify_receives(rows, torch.tensor([True, True, False]))
    assert rallies.robot_receives.tolist() == [1, 1, 0]
    assert bool(rallies.receive_open[:2].all()) and not bool(rallies.receive_open[2])
    # envs 0 and 1: the robot hits; env 0's return lands on the opponent's half, env 1's on its own
    out = [3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    t += H
    rallies.ingest(contacts([0, 1], [STATIC_COUNT + ROBOT] * 2, [t - 0.002] * 2, [[-1.5, 0.0, 0.2]] * 2, [out, out]), t)
    assert rallies.hits[:, ROBOT].tolist() == [1, 1, 0]
    assert not bool(rallies.receive_open[:2].any()), "a hit closes the receive"
    t += H
    rallies.ingest(contacts([0, 1], [TABLE_TOP] * 2, [t - 0.002] * 2, [[0.8, 0.1, 0.0], [-0.4, 0.1, 0.0]], [out, out]), t)
    assert rallies.legal[:, ROBOT].tolist() == [1, 0, 0]
    assert rallies.decided[:, ROBOT].tolist() == [1, 1, 0]
    # a stray touch on a closed receive (a let-go ball, a second touch) is no shot
    for _ in range(16):
        t += H
        rallies.ingest(None, t)
    rallies.ingest(contacts([1], [STATIC_COUNT + ROBOT], [t + 0.001], [[-1.0, 0.0, 0.1]], [out]), t + H)
    t += H
    assert rallies.hits[1, ROBOT] == 1 and rallies.counts["robot_stray_hits"] == 1
    # every ball goes dead: the rallies end and are recorded
    rallies.ball.active[:] = False
    t += H
    rallies.ingest(None, t)
    assert not bool(rallies.in_rally.any())
    metrics = rally_metrics({"completed": rallies.completed, "falls": 0, "env_seconds": 3.0, "seconds": 1.0})
    assert (metrics["rallies"], metrics["receives"], metrics["hits"], metrics["legal"]) == (3, 2, 2, 1)
    assert metrics["legal_per_receive"] == 0.5
    assert metrics["ended_by"]["robot_let_go"] == 1
    assert rallies.legal[:, OPPONENT].sum() == 0


def test_a_fall_cuts_the_rally_out_of_the_grade(model_dir: Path) -> None:
    rallies, _params = rallies_of(model_dir, 2)
    rows = torch.arange(2)
    rallies.start(rows, torch.as_tensor(feed_states(rallies.opponent.rules, 2, np.random.default_rng(3))), 0.0)
    rallies.abort(torch.tensor([1]))
    assert rallies.over_reason[1] == 7 and not bool(rallies.in_rally[1]) and bool(rallies.in_rally[0])
    metrics = rally_metrics({"completed": rallies.completed, "falls": 1, "env_seconds": 2.0, "seconds": 1.0})
    assert metrics["rallies"] == 0 and metrics["rallies_cut_by_a_fall"] == 1
    assert metrics["falls_per_10s"] == pytest.approx(5.0)
    assert rallies.idle(0.5).tolist() == [] and rallies.idle(1.0).tolist() == [1], "a new rally waits 1 s after the last"


def test_a_serve_opens_its_receive_after_its_own_half_bounce(model_dir: Path) -> None:
    rallies, _params = rallies_of(model_dir, 1)
    serve = torch.tensor([[1.4, 0.0, 0.2, -4.0, 0.0, -1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    rallies.start(torch.tensor([0]), serve, 0.0, serves=torch.tensor([True]))
    assert torch.isinf(rallies.t_due[0])
    rallies.ingest(contacts([0], [TABLE_TOP], [0.1], [[0.6, 0.0, 0.0]], [[0.6, 0.0, 0.02, -4.0, 0.0, 1.0, 0.0, 0.0, 0.0]]), 0.1 + H)
    assert float(rallies.t_due[0]) == pytest.approx(0.105)


def test_the_wilson_interval() -> None:
    low, high = wilson(963, 1000)
    assert low < 0.963 < high and high - low == pytest.approx(0.0234, abs=1e-3)
    assert all(np.isnan(wilson(0, 0)))
