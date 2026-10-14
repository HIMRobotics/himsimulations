"""
Replay a recorded HIM session in Isaac Sim with live ball physics, and grade it.

C0 interface stub: the owning builder replaces this body; the signature is the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from lib.config import ReplayParams

if TYPE_CHECKING:
    import antioch


class ReplayReport(BaseModel):
    """What one run measured; ``passed`` is its verdict."""

    model_config = ConfigDict(frozen=True, extra="allow")

    passed: bool


def replay_him_session(run: antioch.ScenarioRun, params: ReplayParams) -> ReplayReport:
    """Replay a recorded HIM session in Isaac Sim with live ball physics, and grade it."""

    raise NotImplementedError("replay_him_session is not implemented yet")
