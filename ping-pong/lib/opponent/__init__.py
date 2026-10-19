"""
HIM's distilled opponent: the offline data generator's receiver, batched in torch over many environments, with one
learned part.

- **The rules, ported** (:mod:`.rules`, :mod:`.ballistics`, :mod:`.strike`): the decision at the other player's hit +
  0.15 s from a predicted bounce, the fitted 4-feature stop hazard (scaled by ``hazard_scale``, 0 in competitive
  points), the contact phase (0.06-0.70 s after the bounce, reaction >= 0.12 s, footwork 0.30 s latency at 4 m/s^2 and
  1.8 m/s, the recorded blade-offset envelope and the body-reach model), the launch and spin-matching normal solves,
  the rigid-blade inverse impact law, the plausibility density, the late correction (bounce + 0.11 s, cited caps) and
  the execution noise at the v1 release's fit (noise 0.7, hazard 0.8).
- **The stroke's feasibility, body-first** (:mod:`.body`): the body runtime's donor reach rule and the generator's
  same-outcome rule exactly, its acceptance and its nudge of the held contact (decision and execution) fitted on the
  runtime's own logged calls.
- **The intent, learned** (:mod:`.intent`): a mixture density network over the generator's shot intent (contact time
  after the bounce, landing, flight time, topspin and sidespin, rubber presented, blade offset, grip roll),
  conditioned on the incoming ball and the stance, distilled from the generator's own draws.

**How it plays** (:class:`.rally.RallySim`, driven by :class:`lib.policy.rally.Rallies` over the robot scene's ball):

1. on each paddle event in the ball's contacts, the receiver's decision is due at the hit + 0.15 s;
2. then :meth:`.opponent.Opponent.decide` runs on the ball re-propagated to the decision time from the last event
   (``RallySim.ball_at``), the receiver's stance, its hips and its hand (``RallySim._hand_at``), and the plan's held
   contact is commanded;
3. after the bounce on the receiver's half (plus 0.11 s, when that is before the contact's render tick),
   :meth:`~.opponent.Opponent.refine`; then :meth:`~.opponent.Opponent.execute`, which returns the kinematic contact
   command (:class:`.strike.ContactCommand`, the noisy contact as the body re-holds it);
4. every physics step the blade's state (:func:`.blade.step_states`: constant twist through the contact within
   +-50 ms, parked otherwise) is handed to the ball, which meets the blade at the commanded state on any step grid.

**Loading it** (:func:`load_him_opponent`): the trained intent network and the exported rules are the
``ping-pong/him-opponent-model`` asset (:func:`fetch_opponent_model`); the opponent plays profile 1 (HIM's opponent),
as the trained policy met it.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from lib.opponent.intent import INTENT_VERSION, MdnIntent
from lib.opponent.opponent import ATTEMPT, LET_GO, STATUS_NAMES, STOP, UNREACHABLE, Opponent, OpponentConfig, Plan
from lib.opponent.rules import RULES_VERSION, RuleBook
from lib.opponent.strike import ContactCommand

MODEL_FILES: tuple[str, ...] = (
    f"intent_mdn_v{INTENT_VERSION}.npz",
    f"intent_mdn_v{INTENT_VERSION}.json",
    f"opponent_rules_v{RULES_VERSION}.npz",
    f"opponent_rules_v{RULES_VERSION}.json",
)
"""The opponent model's files: the zip the ``ping-pong/him-opponent-model`` asset holds, flat."""


def fetch_opponent_model() -> Path:
    """
    The opponent model's directory: the pinned ``ping-pong/him-opponent-model`` asset fetched (or the session's cached
    copy reused) and unpacked beside it.

    :raises FileNotFoundError: If the asset lacks one of :data:`MODEL_FILES`.
    """

    import antioch

    from lib.assets import OPPONENT_MODEL

    name, _, version = OPPONENT_MODEL.removeprefix("asset:").partition("@")
    archive = Path(antioch.fetch_asset(name, version=version))
    directory = archive.with_name(f"{archive.stem}-files")
    if not all((directory / file).exists() for file in MODEL_FILES):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(directory)
    missing = [file for file in MODEL_FILES if not (directory / file).exists()]
    if missing:
        raise FileNotFoundError(f"{OPPONENT_MODEL} lacks {missing}")
    return directory


def load_him_opponent(params: Any, *, seed: int, model_dir: Path | None = None) -> Opponent:
    """
    HIM's distilled opponent for every env of ``params``: profile 1, body-first feasibility, its draws seeded.

    :param params: The envs' ball physics (:class:`lib.physics.vectorized.params.VecParams`): the opponent predicts with
        the same physics the ball flies.
    :param seed: The opponent's random draws.
    :param model_dir: The model's files (``None``: :func:`fetch_opponent_model`).
    """

    from lib.physics.vectorized._torch import torch

    directory = model_dir if model_dir is not None else fetch_opponent_model()
    rules = RuleBook.load(directory, device=params.device, dtype=params.dtype)
    intent = MdnIntent.load(directory / f"intent_mdn_v{INTENT_VERSION}.npz", device=params.device, dtype=params.dtype)
    generator = torch.Generator(device=params.device).manual_seed(int(seed) * 7919 + 5)
    return Opponent(params, intent, rules, OpponentConfig(profiles=(1,)), generator)


__all__ = [
    "ATTEMPT",
    "LET_GO",
    "MODEL_FILES",
    "STATUS_NAMES",
    "STOP",
    "UNREACHABLE",
    "ContactCommand",
    "Opponent",
    "OpponentConfig",
    "Plan",
    "RuleBook",
    "fetch_opponent_model",
    "load_him_opponent",
]
