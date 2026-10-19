"""
The opponent plays exactly as frozen: seeded rallies on CPU through :class:`lib.policy.rally.CoupledRally`, hashed
stream by stream and compared with ``data/opponent_identity.json``.

The ball flies free under the calibrated flight model; a stand-in robot returns every ball reaching its end; the
opponent's ball meets its blade when the blade comes within 6 cm. Hashed, in order:

- ``decisions``: every :meth:`~lib.opponent.Opponent.decide` plan (its fields that hold values), with the opponent's
  random state after it;
- ``refines``: every plan after :meth:`~lib.opponent.Opponent.refine`;
- ``commands``: every contact command :meth:`~lib.opponent.Opponent.execute` returns;
- ``events``: every step's contacts as the driver ingests them;
- ``blades``: the opponent's blade every physics step;
- ``forecast``: the ball forecast every policy step;
- ``rallies``: every completed rally's record.

The small case runs by default (16 envs, 10 s). The frozen full case (256 envs, 60 s; about 10 min on one CPU thread)
runs with ``PING_PONG_FULL_IDENTITY=1``. Both need the opponent model (``PING_PONG_OPPONENT_MODEL``, the
``ping-pong/him-opponent-model@v2`` zip or its directory). Refreeze, only for an intended change of the opponent's
play::

    python tests/robot/test_opponent_identity.py --write <opponent model>
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

FIXTURE = Path(__file__).with_name("data") / "opponent_identity.json"
CASES: dict[str, tuple[int, float]] = {"small": (16, 10.0), "full": (256, 60.0)}
H, DECIMATION = 0.005, 4
STREAMS = ("decisions", "refines", "commands", "events", "blades", "forecast", "rallies")


def digest(hasher: Any, value: Any) -> None:
    """Feed a value's exact bytes to ``hasher``: tensors and arrays with their dtype and shape, dataclasses field by field."""

    if value is None:
        hasher.update(b"None")
    elif isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        hasher.update(f"{array.dtype}{array.shape}".encode())
        hasher.update(array.tobytes())
    elif isinstance(value, np.ndarray):
        hasher.update(f"{value.dtype}{value.shape}".encode())
        hasher.update(np.ascontiguousarray(value).tobytes())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for item in dataclasses.fields(value):
            field_value = getattr(value, item.name)
            if field_value is not None:  # a field that is always None carries no play
                hasher.update(item.name.encode())
                digest(hasher, field_value)
    elif isinstance(value, dict):
        for key in sorted(value):
            hasher.update(str(key).encode())
            digest(hasher, value[key])
    elif isinstance(value, (tuple, list)):
        hasher.update(f"seq{len(value)}".encode())
        for item in value:
            digest(hasher, item)
    else:
        hasher.update(repr(value).encode())


class FreeBall:
    """The coupling's interface over a ball in free flight: table bounces, the stand-in robot's and the opponent's hits."""

    def __init__(self, n: int, rng: np.random.Generator) -> None:
        from lib.physics.params import nominal_params
        from lib.physics.vectorized.params import VecParams

        self.physics: Any = type("Physics", (), {"params": VecParams.from_params(nominal_params(), n, device="cpu", dtype=torch.float64)})()
        self.t, self.h, self.rng = 0.0, H, rng
        self.ball = torch.zeros(n, 9, dtype=torch.float64)
        self.active = torch.zeros(n, dtype=torch.bool)
        self.contacts: Any = None
        self.blade: Any = None

    def launch(self, rows: Any, states: Any) -> None:
        self.ball[rows] = states.to(torch.float64)
        self.active[rows] = True

    def set_opponent(self, p: Any, v: Any, q: Any, teleport: Any) -> None:
        self.blade = p

    def step(self) -> None:
        from lib.opponent.ballistics import flight_at
        from lib.physics.vectorized.geometry import STATIC_COUNT, TABLE_TOP

        params = self.physics.params
        n = self.ball.shape[0]
        before = self.ball.clone()
        self.ball = torch.where(self.active[:, None], flight_at(params, self.ball, torch.arange(n), torch.full((n,), H, dtype=torch.float64), 4), self.ball)
        radius = float(params.radius[0])
        rows: list[tuple[int, int, list[float], Any]] = []
        table = self.active & (before[:, 2] > radius) & (self.ball[:, 2] <= radius) & (self.ball[:, 0].abs() < 1.37) & (self.ball[:, 1].abs() < 0.7625)
        for env in torch.nonzero(table).flatten().tolist():
            self.ball[env, 2] = radius
            self.ball[env, 5] = -0.9 * self.ball[env, 5]
            rows.append((env, TABLE_TOP, [float(self.ball[env, 0]), float(self.ball[env, 1]), 0.0], self.ball[env].clone()))
        robot = self.active & (self.ball[:, 0] < -1.5) & (before[:, 0] >= -1.5) & (self.ball[:, 3] < 0)
        for env in torch.nonzero(robot).flatten().tolist():
            speed, side, spin = float(self.rng.uniform(4.0, 6.0)), float(self.rng.normal(0.0, 0.4)), float(self.rng.normal(-60.0, 30.0))
            self.ball[env, 3:9] = torch.tensor([speed, side, 2.2, 0.0, spin, 0.0], dtype=torch.float64)
            rows.append((env, STATIC_COUNT, self.ball[env, :3].tolist(), self.ball[env].clone()))
        if self.blade is not None:
            near = self.active & (torch.linalg.norm(self.ball[:, :3] - self.blade.to(torch.float64), dim=-1) < 0.06) & (self.ball[:, 3] > 0)
            for env in torch.nonzero(near).flatten().tolist():
                self.ball[env, 3:6] = torch.tensor([-float(self.rng.uniform(3.5, 5.5)), float(self.rng.normal(0.0, 0.4)), 2.0], dtype=torch.float64)
                rows.append((env, STATIC_COUNT + 1, self.ball[env, :3].tolist(), self.ball[env].clone()))
        self.active &= self.ball[:, 2] > -float(params.table_height[0]) + radius
        self.t += H
        self.contacts = None
        if rows:
            flat = {
                "env": torch.tensor([r[0] for r in rows]),
                "surface": torch.tensor([r[1] for r in rows]),
                "time": torch.full((len(rows),), self.t, dtype=torch.float64),
                "point": torch.tensor([r[2] for r in rows], dtype=torch.float64),
                "after": torch.stack([r[3] for r in rows]),
            }
            self.contacts = type("Contacts", (), {"flat": staticmethod(lambda flat=flat: flat)})()


class Scene:
    """The table scene's interface the hooks read: the robot standing at its stance, facing the table."""

    def __init__(self, n: int, rng: np.random.Generator) -> None:
        self.num_envs, self.step_dt = n, H * DECIMATION
        self.coupling = FreeBall(n, rng)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.table_offset = torch.tensor([[0.0, 0.0, 0.76]], dtype=torch.float64).expand(n, 3).clone()

    def base_frame(self) -> tuple[Any, Any]:
        return torch.tensor([[-1.92, 0.0, 0.78]]).expand(self.num_envs, 3).clone(), torch.eye(3).expand(self.num_envs, 3, 3).clone()


def play(model_dir: Path, num_envs: int, seconds: float) -> dict[str, Any]:
    """The seeded rallies, every stream hashed; also the counts, as a readable check."""

    from lib.policy.rally import CoupledRally

    torch.set_num_threads(1)
    hashers = {name: hashlib.sha256() for name in STREAMS}
    scene, rally = Scene(num_envs, np.random.default_rng(20261014)), CoupledRally(model_dir, seed=0)
    digest(hashers["forecast"], rally.observe(scene))
    rallies = rally.rallies
    assert rallies is not None
    opponent = rallies.opponent
    decide, refine, execute = opponent.decide, opponent.refine, opponent.execute

    def decided(*args: Any, **kwargs: Any) -> Any:
        plan = decide(*args, **kwargs)
        digest(hashers["decisions"], plan)
        digest(hashers["decisions"], opponent.generator.get_state())
        return plan

    def refined(*args: Any, **kwargs: Any) -> Any:
        out = refine(*args, **kwargs)
        digest(hashers["refines"], args[0])
        digest(hashers["refines"], opponent.generator.get_state())
        return out

    def executed(*args: Any, **kwargs: Any) -> Any:
        command = execute(*args, **kwargs)
        digest(hashers["commands"], command)
        digest(hashers["commands"], opponent.generator.get_state())
        return command

    opponent.decide, opponent.refine, opponent.execute = decided, refined, executed
    for _ in range(round(seconds / scene.step_dt)):
        for _k in range(DECIMATION):
            pose = rally.opponent_blade(scene, scene.coupling.t)
            digest(hashers["blades"], pose)
            scene.coupling.set_opponent(*pose)
            scene.coupling.step()
            if scene.coupling.contacts is not None:
                digest(hashers["events"], scene.coupling.contacts.flat())
        scene.episode_length_buf += 1
        digest(hashers["forecast"], rally.observe(scene))
    for record in rallies.completed:
        digest(hashers["rallies"], record)
    hits = rallies.hits.sum(0).tolist()
    return {
        "digests": {name: hasher.hexdigest() for name, hasher in hashers.items()},
        "counts": {**rallies.counts, "rallies": len(rallies.completed), "robot_hits": hits[0], "opponent_hits": hits[1]},
    }


def model_from(spec: str, scratch: Path) -> Path:
    path = Path(spec)
    if path.is_dir():
        return path
    with zipfile.ZipFile(path) as bundle:
        bundle.extractall(scratch)
    return scratch


@pytest.mark.parametrize("case", list(CASES))
def test_the_opponent_plays_as_frozen(case: str, tmp_path: Path) -> None:
    spec = os.environ.get("PING_PONG_OPPONENT_MODEL")
    if not spec:
        pytest.skip("set PING_PONG_OPPONENT_MODEL to the ping-pong/him-opponent-model@v2 zip (or its unpacked directory)")
    if case == "full" and os.environ.get("PING_PONG_FULL_IDENTITY") != "1":
        pytest.skip("the full case takes about 10 min on CPU: set PING_PONG_FULL_IDENTITY=1")
    frozen = json.loads(FIXTURE.read_text())[case]
    played = play(model_from(spec, tmp_path), *CASES[case])
    assert played["counts"] == frozen["counts"]
    assert played["digests"] == frozen["digests"]


if __name__ == "__main__":
    import tempfile

    if len(sys.argv) != 3 or sys.argv[1] != "--write":
        raise SystemExit(__doc__)
    with tempfile.TemporaryDirectory() as scratch:
        model = model_from(sys.argv[2], Path(scratch))
        frozen = {case: play(model, *size) for case, size in CASES.items()}
    FIXTURE.write_text(json.dumps(frozen, indent=1) + "\n")
    print(json.dumps(frozen, indent=1))
