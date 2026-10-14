"""
:class:`lib.policy.rally.CoupledRally` as a scene's hooks, on CPU, over a stand-in for the scene and its coupled ball
(the ball flies free, nothing hits it): the opponent's blade every physics step, a feed served once the robot has
settled, the receive classified, the forecast's shape and validity, and a reset cutting the rally.

Needs the opponent model: set ``PING_PONG_OPPONENT_MODEL`` to the ``ping-pong/him-opponent-model@v2`` zip, or to the
directory it unpacks to.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from lib.opponent.ballistics import flight_at
from lib.physics.params import nominal_params
from lib.physics.vectorized.params import VecParams
from lib.policy.rally import FORECAST_PRIOR_MEAN, SETTLE_S, CoupledRally

H, DECIMATION = 0.005, 4


class FreeBall:
    """The coupling's interface over a ball in free flight (the calibrated flight model; no contacts, the floor ends it)."""

    def __init__(self, n: int) -> None:
        self.physics: Any = type("Physics", (), {"params": VecParams.from_params(nominal_params(), n, device="cpu", dtype=torch.float64)})()
        self.t, self.h = 0.0, H
        self.ball = torch.zeros(n, 9, dtype=torch.float64)
        self.active = torch.zeros(n, dtype=torch.bool)
        self.contacts = None
        self.opponent: Any = None

    def launch(self, rows: Any, states: Any) -> None:
        self.ball[rows] = states.to(torch.float64)
        self.active[rows] = True

    def set_opponent(self, p: Any, v: Any, q: Any, teleport: Any) -> None:
        self.opponent = (p, v, q, teleport)

    def step(self) -> None:
        params = self.physics.params
        rows = torch.arange(self.ball.shape[0])
        self.ball = torch.where(self.active[:, None], flight_at(params, self.ball, rows, torch.full((rows.numel(),), H, dtype=torch.float64), 4), self.ball)
        self.active &= self.ball[:, 2] > -params.table_height + params.radius
        self.t += H


class Scene:
    """The table scene's interface the hooks read: the robot standing at its stance, facing the table."""

    def __init__(self, n: int) -> None:
        self.num_envs, self.step_dt = n, H * DECIMATION
        self.coupling = FreeBall(n)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.table_offset = torch.tensor([[0.0, 0.0, 0.76]], dtype=torch.float64).expand(n, 3).clone()

    def base_frame(self) -> tuple[Any, Any]:
        position = torch.tensor([[-1.92, 0.0, 0.78]]).expand(self.num_envs, 3).clone()
        return position, torch.eye(3).expand(self.num_envs, 3, 3).clone()

    def policy_step(self, rally: CoupledRally) -> Any:
        for k in range(DECIMATION):
            pose = rally.opponent_blade(self, (k + self.episode_length_buf[0].item() * DECIMATION) * H)
            self.coupling.set_opponent(*pose)
            self.coupling.step()
        self.episode_length_buf += 1
        return rally.observe(self)


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
    return out


def test_the_hooks_serve_classify_forecast_and_reset(model_dir: Path) -> None:
    scene, rally = Scene(4), CoupledRally(model_dir, seed=0)
    forecast = rally.observe(scene)  # the reset's observation: no rally, no ball
    assert forecast.shape == (4, 30)
    assert torch.allclose(forecast[:, :10], torch.tensor(FORECAST_PRIOR_MEAN[:10], dtype=forecast.dtype))
    settle = round(SETTLE_S / scene.step_dt)
    for _ in range(settle - 1):
        scene.policy_step(rally)
    assert not bool(scene.coupling.active.any()), "no feed before the robot has stood 0.5 s"
    scene.policy_step(rally)
    assert bool(scene.coupling.active.all()), "every env served once settled"
    assert bool((scene.coupling.ball[:, 3] < 0.0).all()), "the feeds fly toward the robot"
    rallies = rally.rallies
    assert rallies is not None and bool(rallies.in_rally.all())
    p, v, q, teleport = scene.coupling.opponent
    assert p.shape == (4, 3) and v.shape == (4, 3) and q.shape == (4, 4) and teleport.shape == (4,)
    forecast = scene.policy_step(rally)
    assert bool(forecast[:, 10].bool().all()), "each feed's bounce on the robot's half is forecast"
    for _ in range(3):
        scene.policy_step(rally)
    assert rallies.robot_receives.tolist() == [1, 1, 1, 1], "each feed opened one receive"
    rally.reset(scene, torch.tensor([3]))
    assert rallies.over_reason[3] == 7 and not bool(rallies.in_rally[3])
    for _ in range(100):  # the unhit balls fall to the floor: the rallies end, the receiver's; new ones start 1 s later
        scene.policy_step(rally)
    ended = [(int(env), int(code)) for part in rallies.completed for env, code in zip(part["env"], part["over_reason"], strict=True)]
    assert ended.count((3, 7)) == 1 and all(code == 9 for env, code in ended if (env, code) != (3, 7))
    assert {env for env, code in ended if code == 9} == {0, 1, 2, 3}, "the reset env plays again"
