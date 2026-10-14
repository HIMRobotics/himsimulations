"""
The ball forecast's port (:func:`lib.policy.rally.predict_flight`, :func:`lib.policy.rally.forecast_columns_from`)
against the source pipeline's own forecast on the same flights (``data/policy_observations.npz``): recorded HIM serves
and rally shots, designed sweep shots and the opponent's feeds, flown at nominal physics, with the serve fix and
without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from lib.physics.params import nominal_params
from lib.physics.vectorized.params import VecParams
from lib.policy.rally import forecast_columns_from, predict_flight

FIXTURE = Path(__file__).with_name("data") / "policy_observations.npz"


def port_forecast(data: dict[str, np.ndarray], serve_fix: bool) -> np.ndarray:
    """The port's forecast ``(T, F, 30)`` at the fixture's instants, each flight predicted once at its start."""

    f64 = torch.float64
    balls = torch.as_tensor(data["balls"], dtype=f64)
    n = balls.shape[0]
    rows = torch.arange(n)
    params = VecParams.from_params(nominal_params(), n, device="cpu", dtype=f64)
    flight = predict_flight(params, balls, rows, 0.0, torch.as_tensor(data["serve"]) if serve_fix else None)
    position, rotation, offset = (torch.as_tensor(data[name], dtype=f64) for name in ("position", "rotation", "table_offset"))
    out = []
    for k, t in enumerate(data["instants"].tolist()):
        out.append(
            forecast_columns_from(
                params,
                torch.as_tensor(data["live"][k], dtype=f64),
                torch.as_tensor(data["active"][k]),
                t,
                flight.t_bounce,
                flight.after,
                flight.ok,
                flight.toward_robot,
                position,
                rotation,
                offset,
                t_pre=flight.t_pre if serve_fix else None,
                pre_after=flight.pre_after if serve_fix else None,
            ).numpy()
        )
    return np.stack(out)


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray]:
    with np.load(FIXTURE, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


@pytest.mark.parametrize(("serve_fix", "stored"), [(True, "forecast_fixed"), (False, "forecast_unfixed")])
def test_the_port_reproduces_the_source_forecast(data: dict[str, np.ndarray], serve_fix: bool, stored: str) -> None:
    got = port_forecast(data, serve_fix)
    expected = data[stored]
    assert got.shape == expected.shape == (data["instants"].shape[0], data["balls"].shape[0], 30)
    delta = np.abs(got - expected)
    worst = np.unravel_index(np.argmax(delta), delta.shape)
    assert delta.max() <= 1e-12, f"forecast differs by {delta.max():.3e} at instant {worst[0]}, flight {worst[1]}, column {worst[2]}"


def test_the_serve_fix_changes_serves_only(data: dict[str, np.ndarray]) -> None:
    serve = data["serve"]
    changed = np.abs(data["forecast_fixed"] - data["forecast_unfixed"]).max(-1) > 0.0
    assert not changed[:, ~serve].any(), "the serve fix changed a flight that is not a serve"
    assert changed[:, serve].any(axis=1)[:-1].all(), "the serve fix changed no serve at an instant before the balls die"
    # without the fix a serve's predicted bounce on the robot's half is never valid; with it, it is
    assert (data["forecast_unfixed"][:, serve, 10] == 0.0).all()
    assert data["forecast_fixed"][0, serve, 10].mean() > 0.9


def test_the_fixture_holds_every_kind_of_flight(data: dict[str, np.ndarray]) -> None:
    kinds, counts = np.unique(data["flight_kind"], return_counts=True)
    assert dict(zip(kinds.tolist(), counts.tolist(), strict=True)) == {"feed": 8, "rally": 12, "serve": 24, "sweep": 4}
    assert (~data["active"][-1]).any(), "some ball is dead by the last instant (the forecast's prior-mean branch)"
