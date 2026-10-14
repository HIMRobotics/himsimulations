"""
The trained policy's fold (:func:`lib.policy.actor.fold_play_constants`): the 146-input actor plays exactly as the
248-input checkpoint does with its stroke reference hidden (at the normaliser's mean) and its command hidden (zeros).

- A random actor of the checkpoint's shape, folded, against the same network unfolded: exact in float64, float32
  rounding as shipped (always runs).
- The trained checkpoint, folded, against rsl_rl 5.0.1's own ``MLPModel`` on the source pipeline's observations, serves
  included (``data/policy_observations.npz``; set ``PING_PONG_G1_POLICY`` to the checkpoint's local path).
- The blind control's actor hides exactly the ball's columns and the forecast.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from lib.policy.actor import (
    ACTOR_INPUTS,
    BALL_INPUTS,
    CHECKPOINT_INPUTS,
    VELOCITY_CLIPS,
    columns,
    fold_play_constants,
    hide_ball,
    load_rally_actor,
    rally_actor_class,
    width,
)
from lib.policy.rally import FORECAST_PRIOR_MEAN, FORECAST_VALID_FLAGS

FIXTURE = Path(__file__).with_name("data") / "policy_observations.npz"
TOLERANCE = 1e-5
"""Actions, in action units: the fold changes only the first layer's summation order."""


def random_actor_state(seed: int = 0) -> dict[str, torch.Tensor]:
    """An rsl_rl ``MLPModel`` actor state of the checkpoint's shape (248-512-256-128-29), with a random normaliser."""

    generator = torch.Generator().manual_seed(seed)
    sizes = (width(CHECKPOINT_INPUTS), 512, 256, 128, 29)
    state: dict[str, torch.Tensor] = {}
    for k, (fan_in, fan_out) in enumerate(itertools.pairwise(sizes)):
        state[f"mlp.{2 * k}.weight"] = torch.randn(fan_out, fan_in, generator=generator) / fan_in**0.5
        state[f"mlp.{2 * k}.bias"] = 0.1 * torch.randn(fan_out, generator=generator)
    state["obs_normalizer._mean"] = torch.randn(1, sizes[0], generator=generator)
    state["obs_normalizer._std"] = 0.05 + torch.rand(1, sizes[0], generator=generator) * 3.0
    state["obs_normalizer._var"] = state["obs_normalizer._std"] ** 2
    state["obs_normalizer.count"] = torch.tensor(10**6)
    return state


def unfolded_actions(state: dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    """The checkpoint's network as trained: 248 inputs (the reference at its mean, the command at zeros), clipped."""

    mean, std = state["obs_normalizer._mean"][0], state["obs_normalizer._std"][0]
    reference = columns(CHECKPOINT_INPUTS, ("stroke_reference",))
    command = columns(CHECKPOINT_INPUTS, ("strike_command",))
    full = torch.cat([inputs, mean[reference].expand(inputs.shape[0], -1), torch.zeros(inputs.shape[0], len(command))], -1)
    bound = torch.full((full.shape[1],), float("inf"))
    for name, clip in VELOCITY_CLIPS.items():
        bound[columns(CHECKPOINT_INPUTS, (name,))] = clip
    x = (torch.maximum(torch.minimum(full, bound), -bound) - mean) / (std + 1e-2)
    for k in range(4):
        x = x @ state[f"mlp.{2 * k}.weight"].T + state[f"mlp.{2 * k}.bias"]
        if k < 3:
            x = torch.nn.functional.elu(x)
    return x


def random_inputs(state: dict[str, torch.Tensor], rows: int = 512) -> torch.Tensor:
    """The actor's 146 inputs about the normaliser's spread, some joint velocities far beyond their clip."""

    generator = torch.Generator().manual_seed(1)
    mean, std = state["obs_normalizer._mean"][0, :146], state["obs_normalizer._std"][0, :146]
    inputs = mean + 2.0 * std * torch.randn(rows, width(ACTOR_INPUTS), generator=generator, dtype=mean.dtype)
    inputs[:64, columns(ACTOR_INPUTS, ("joint_vel",))] *= 40.0
    return inputs


def test_the_fold_is_exact_algebra() -> None:
    """In float64 the folded network (by hand, from the folded state) equals the unfolded one to rounding."""

    state = {name: value.double() if value.is_floating_point() else value for name, value in random_actor_state().items()}
    folded = fold_play_constants(state)
    inputs = random_inputs(state)
    bound = torch.full((146,), float("inf"), dtype=torch.float64)
    for name, clip in VELOCITY_CLIPS.items():
        bound[columns(ACTOR_INPUTS, (name,))] = clip
    x = (torch.maximum(torch.minimum(inputs, bound), -bound) - folded["mean"]) / (folded["std"] + 1e-2)
    for k in range(4):
        x = x @ folded[f"mlp.{2 * k}.weight"].T + folded[f"mlp.{2 * k}.bias"]
        if k < 3:
            x = torch.nn.functional.elu(x)
    delta = (x - unfolded_actions(state, inputs)).abs().max().item()
    assert delta < 1e-10, f"the fold's algebra is off by {delta:.3e}"


def test_the_folded_actor_matches_the_unfolded_network_in_float32() -> None:
    """As shipped (float32), only the first layer's summation order differs: float32 rounding of the outputs' size."""

    state = random_actor_state()
    actor = rally_actor_class()(fold_play_constants(state)).eval()
    inputs = random_inputs(state)
    with torch.no_grad():
        expected = unfolded_actions(state, inputs)
        delta = (actor(inputs) - expected).abs().max().item()
    scale = max(1.0, expected.abs().max().item())
    assert delta < 1e-6 * scale, f"folded actor differs from the unfolded network by {delta:.3e} (outputs up to {scale:.1f})"


def test_the_fold_keeps_the_actor_inputs_and_drops_only_the_constant_blocks() -> None:
    state = random_actor_state()
    folded = fold_play_constants(state)
    assert width(ACTOR_INPUTS) == 146 and width(CHECKPOINT_INPUTS) == 248
    assert folded["mlp.0.weight"].shape == (512, 146)
    assert torch.equal(folded["mlp.0.weight"], state["mlp.0.weight"][:, :146])
    assert torch.equal(folded["mean"], state["obs_normalizer._mean"][0, :146])
    for k in (2, 4, 6):
        assert torch.equal(folded[f"mlp.{k}.weight"], state[f"mlp.{k}.weight"])
    with pytest.raises(ValueError, match="248"):
        fold_play_constants({**state, "mlp.0.weight": state["mlp.0.weight"][:, :200]})


def test_it1500_folded_plays_as_rsl_rl_on_the_source_observations() -> None:
    path = os.environ.get("PING_PONG_G1_POLICY")
    if not path:
        pytest.skip("set PING_PONG_G1_POLICY to the local path of ping-pong/g1-rally-policy@e0e0bb45-it1500 (model_final.pt)")
    from lib.robots import UNITREE_G1

    data = np.load(FIXTURE, allow_pickle=False)
    import hashlib

    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == str(data["checkpoint_sha256"]), "not the checkpoint the fixture was built with"
    actor = load_rally_actor(path, default_pose=UNITREE_G1.default_pose)
    obs = torch.as_tensor(data["obs"])
    expected = torch.as_tensor(data["actions_rsl_rl"])
    with torch.no_grad():
        delta = (actor(obs[:, :146]) - expected).abs().max().item()
    serves = np.tile(data["serve"], data["instants"].shape[0])
    assert serves.sum() > 0, "the fixture holds serves"
    assert delta < TOLERANCE, f"folded it1500 differs from rsl_rl's MLPModel by {delta:.3e} over {obs.shape[0]} rows"


def test_it1500_refuses_a_robot_in_another_default_pose() -> None:
    path = os.environ.get("PING_PONG_G1_POLICY")
    if not path:
        pytest.skip("set PING_PONG_G1_POLICY to the local path of ping-pong/g1-rally-policy@e0e0bb45-it1500 (model_final.pt)")
    from lib.robots import UNITREE_G1

    with pytest.raises(ValueError, match="default pose"):
        load_rally_actor(path, default_pose={**UNITREE_G1.default_pose, "right_elbow_joint": 0.0})


def test_the_blind_control_hides_the_ball_and_the_forecast_only() -> None:
    actor = rally_actor_class()(fold_play_constants(random_actor_state())).eval()
    blind = hide_ball(actor)
    ball = columns(ACTOR_INPUTS, BALL_INPUTS)
    forecast = columns(ACTOR_INPUTS, ("ball_forecast",))
    assert set(blind.hidden) == set(ball) | set(forecast)
    assert all(blind.hidden[c] == 0.0 for c in ball)
    for k, c in enumerate(forecast):
        assert blind.hidden[c] == (0.0 if k in FORECAST_VALID_FLAGS else FORECAST_PRIOR_MEAN[k])
    inputs = torch.randn(32, width(ACTOR_INPUTS), generator=torch.Generator().manual_seed(2))
    shown = inputs.clone()
    shown[:, list(blind.hidden)] = torch.as_tensor(list(blind.hidden.values()), dtype=shown.dtype)
    with torch.no_grad():
        assert torch.equal(blind(inputs), actor(shown))
        assert not torch.equal(blind(inputs), actor(inputs))
