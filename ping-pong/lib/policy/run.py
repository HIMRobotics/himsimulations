"""
The trained policy rallies against the HIM opponent in Isaac Lab, beside its blind control.

C0 interface stub: the owning builder replaces this body; the signature is the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from lib.config import PlayParams

if TYPE_CHECKING:
    import antioch


class PlayReport(BaseModel):
    """What one run measured; ``passed`` is its verdict."""

    model_config = ConfigDict(frozen=True, extra="allow")

    passed: bool


def play_ping_pong_policy(run: antioch.ScenarioRun, params: PlayParams) -> PlayReport:
    """The trained policy rallies against the HIM opponent in Isaac Lab, beside its blind control."""

    raise NotImplementedError("play_ping_pong_policy is not implemented yet")
