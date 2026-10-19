"""
The trained G1 policy as a plain torch module: :func:`load_rally_actor` loads the checkpoint and folds it to the inputs
it actually reads at play time.

**What the network is.** An MLP, 512-256-128 with ELU, trained with PPO (rsl_rl 5.0.1). Its input is normalised by the
running mean and standard deviation it kept in training, ``z = (x - mean) / (std + 0.01)``, and its output is 29 joint
targets in units of each joint's action scale: ``q_target = default_pose + action_scale * action``.

**Why it is folded.** The checkpoint reads 248 inputs, but 102 of them are constant at play time:

- **A stroke reference, 88 columns.** It was a training-time prior, gradually hidden from the actor. A hidden
  reference reads as the normaliser's mean, which normalises to exactly 0.
- **A strike command, 14 columns.** A planner's command was shown to an earlier stage. In play it is never shown and
  reads as zeros.

A constant input adds a constant to the first layer, ``W[:, c] @ z_c``, so it folds exactly into that layer's bias.
The folded actor reads 146 inputs, laid out as in :data:`ACTOR_INPUTS`. Play needs no stroke library, planner or
command table, and loading needs no rsl_rl. ``tests/robot/test_actor_fold.py`` checks the folded actor against the
unfolded one on observations built by the source pipeline, serves included.

**The input contract** (:data:`ACTOR_INPUTS`): the robot's 116 values from :func:`lib.robots.table_scene.observe`
(``3 J + 29`` for ``J = 29`` joints), then the ball forecast's 30 (:func:`lib.policy.rally.forecast`). Velocities
are clipped first (:data:`VELOCITY_CLIPS`), as in training. Joints follow :data:`POLICY_JOINTS`, the order of the
articulation the policy trained in; play checks it against the live robot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lib.robots.table_scene import observation_terms

Tensor = Any

FORECAST_COLUMNS: int = 30
"""The ball forecast's columns (:func:`lib.policy.rally.forecast`)."""

ROBOT_INPUTS: tuple[tuple[str, int], ...] = observation_terms(29)
"""The robot's observation (:func:`lib.robots.table_scene.observe`) for the G1's 29 joints: 116 values."""

ACTOR_INPUTS: tuple[tuple[str, int], ...] = (*ROBOT_INPUTS, ("ball_forecast", FORECAST_COLUMNS))
"""What the folded actor reads, in order: 146 values."""

CHECKPOINT_INPUTS: tuple[tuple[str, int], ...] = (*ACTOR_INPUTS, ("stroke_reference", 88), ("strike_command", 14))
"""What the checkpoint's first layer reads: the actor's inputs, then the two blocks constant at play time (248)."""

VELOCITY_CLIPS: dict[str, float] = {"base_lin_vel_b": 5.0, "base_ang_vel_b": 10.0, "joint_vel": 30.0, "paddle_vel_b": 25.0, "ball_vel_b": 30.0}
"""Velocity inputs are clipped to these bounds (m/s, rad/s) before normalisation, as in training."""

BALL_INPUTS: tuple[str, ...] = ("ball_pos_b", "ball_vel_b", "ball_active")
"""The robot observation's ball columns (the blind control zeroes them)."""

POLICY_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
"""The G1's joints in its Isaac Lab articulation's order: the order of the policy's joint inputs and actions."""

NORMALISER_EPS: float = 1e-2
"""rsl_rl's ``EmpiricalNormalization`` epsilon: ``z = (x - mean) / (std + eps)``."""


def columns(terms: tuple[tuple[str, int], ...], names: tuple[str, ...]) -> list[int]:
    """The column indices of the named terms in a layout."""

    out, start = [], 0
    for name, size in terms:
        if name in names:
            out.extend(range(start, start + size))
        start += size
    return out


def width(terms: tuple[tuple[str, int], ...]) -> int:
    """A layout's total width."""

    return sum(size for _name, size in terms)


def fold_play_constants(actor_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """
    Fold the checkpoint's play-time-constant inputs into its first layer.

    The stroke reference reads the normaliser's mean, which normalises to 0 and contributes nothing. The strike command
    reads zeros, ``z = -mean / (std + eps)``, and contributes ``W[:, command] @ z``. Both are computed in float64 and
    added to the first layer's bias.

    :param actor_state: The checkpoint's ``actor_state_dict`` (rsl_rl 5.0.1 ``MLPModel``: ``obs_normalizer._mean``,
        ``obs_normalizer._std``, ``mlp.<k>.weight`` and ``mlp.<k>.bias``).
    :return: The folded actor's state: ``mean`` and ``std`` over its 146 inputs, and ``mlp.<k>`` with the first layer
        cut to 146 columns.
    :raises ValueError: If the checkpoint does not read :data:`CHECKPOINT_INPUTS`.
    """

    import torch

    weight = actor_state["mlp.0.weight"]
    if weight.shape[1] != width(CHECKPOINT_INPUTS):
        raise ValueError(f"the checkpoint's actor reads {weight.shape[1]} inputs, not the {width(CHECKPOINT_INPUTS)} of CHECKPOINT_INPUTS")
    keep = width(ACTOR_INPUTS)
    mean = actor_state["obs_normalizer._mean"][0].double()
    std = actor_state["obs_normalizer._std"][0].double()
    reference = columns(CHECKPOINT_INPUTS, ("stroke_reference",))
    command = columns(CHECKPOINT_INPUTS, ("strike_command",))
    constant = torch.zeros(weight.shape[1], dtype=torch.float64)
    constant[reference] = 0.0  # the hidden reference reads its mean: (mean - mean) / (std + eps) = 0
    constant[command] = (0.0 - mean[command]) / (std[command] + NORMALISER_EPS)  # the hidden command reads zeros
    bias = actor_state["mlp.0.bias"].double() + weight.double() @ constant
    folded = {name: value.clone() for name, value in actor_state.items() if name.startswith("mlp.")}
    folded["mlp.0.weight"] = weight[:, :keep].clone()
    folded["mlp.0.bias"] = bias.to(weight.dtype)
    folded["mean"] = actor_state["obs_normalizer._mean"][0, :keep].clone()
    folded["std"] = actor_state["obs_normalizer._std"][0, :keep].clone()
    return folded


def rally_actor_class() -> type:
    """:class:`RallyActor`'s class, defined on first use (torch is imported lazily)."""

    import torch

    class RallyActor(torch.nn.Module):
        """
        The folded policy: 146 observations in, 29 joint actions out (deterministic).

        ``forward`` clips the velocities, normalises, and runs the MLP. ``hidden`` columns (the blind control) are
        overwritten with ``hidden_values`` first.
        """

        mean: Tensor
        std: Tensor
        bound: Tensor

        def __init__(self, state: dict[str, Tensor], hidden: dict[int, float] | None = None) -> None:
            super().__init__()
            linear = sorted({int(name.split(".")[1]) for name in state if name.startswith("mlp.")})
            layers: list[Any] = []
            for k, index in enumerate(linear):
                out_features, in_features = state[f"mlp.{index}.weight"].shape
                layers.append(torch.nn.Linear(in_features, out_features))
                if k < len(linear) - 1:
                    layers.append(torch.nn.ELU())
            self.mlp = torch.nn.Sequential(*layers)
            self.mlp.load_state_dict({f"{2 * k}.{part}": state[f"mlp.{index}.{part}"] for k, index in enumerate(linear) for part in ("weight", "bias")})
            self.register_buffer("mean", state["mean"].clone())
            self.register_buffer("std", state["std"].clone())
            bound = torch.full((width(ACTOR_INPUTS),), float("inf"))
            for name, clip in VELOCITY_CLIPS.items():
                bound[columns(ACTOR_INPUTS, (name,))] = clip
            self.register_buffer("bound", bound)
            self.hidden = dict(hidden or {})
            self.state = state

        def forward(self, observation: Tensor) -> Tensor:
            x = observation.to(self.mean.dtype)
            if self.hidden:
                x = x.clone()
                index = list(self.hidden)
                x[:, index] = torch.as_tensor(list(self.hidden.values()), dtype=x.dtype, device=x.device)
            x = torch.maximum(torch.minimum(x, self.bound), -self.bound)
            return self.mlp((x - self.mean) / (self.std + NORMALISER_EPS))

    return RallyActor


def load_rally_actor(policy: str | Path, *, default_pose: dict[str, float] | None = None, device: Any = "cpu") -> Any:
    """
    The trained policy, folded to the 146 inputs it reads at play time (:func:`fold_play_constants`).

    :param policy: The checkpoint: ``asset:<name>@<version>`` (fetched from the asset store) or a local path.
    :param default_pose: The robot's default pose; the checkpoint's own must equal it (its action origin).
    :param device: Torch device.
    :return: A ``RallyActor`` module in eval mode.
    :raises ValueError: If the checkpoint was trained in another default pose.
    """

    import torch

    path = fetch_checkpoint(policy)
    saved = torch.load(str(path), map_location="cpu", weights_only=False)
    trained_pose = saved.get("infos", {}).get("pose", {}).get("default_joint_pos")
    if default_pose is not None and trained_pose is not None and dict(trained_pose) != dict(default_pose):
        raise ValueError(f"the policy trained in default pose {trained_pose}, not the robot's {default_pose}")
    actor = rally_actor_class()(fold_play_constants(saved["actor_state_dict"]))
    return actor.to(device).eval()


def hide_ball(actor: Any) -> Any:
    """
    The blind control's actor: the same weights, with the ball's columns zeroed and the forecast reading "no ball"
    (its prior mean, every valid flag 0). A policy that plays from the ball must collapse.
    """

    from lib.policy.rally import FORECAST_PRIOR_MEAN, FORECAST_VALID_FLAGS

    start = width(ROBOT_INPUTS)
    hidden = dict.fromkeys(columns(ACTOR_INPUTS, BALL_INPUTS), 0.0)
    hidden.update({start + k: (0.0 if k in FORECAST_VALID_FLAGS else value) for k, value in enumerate(FORECAST_PRIOR_MEAN)})
    blind = rally_actor_class()(actor.state, hidden=hidden)
    return blind.to(actor.mean.device).eval()


def fetch_checkpoint(policy: str | Path) -> Path:
    """A checkpoint's local path: ``asset:<name>@<version>`` fetched from the asset store, or a path as given."""

    spec = str(policy)
    if not spec.startswith("asset:"):
        return Path(spec)
    import antioch

    name, _, version = spec.removeprefix("asset:").partition("@")
    return Path(antioch.fetch_asset(name, version=version or None))


__all__ = [
    "ACTOR_INPUTS",
    "BALL_INPUTS",
    "CHECKPOINT_INPUTS",
    "POLICY_JOINTS",
    "VELOCITY_CLIPS",
    "fold_play_constants",
    "hide_ball",
    "load_rally_actor",
    "rally_actor_class",
]
