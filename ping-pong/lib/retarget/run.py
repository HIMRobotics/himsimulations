"""
Retarget HIM's mocap strokes onto a humanoid, gate them, and publish the kept strokes.

C0 interface stub: the owning builder replaces this body; the signature is the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from lib.config import RetargetParams

if TYPE_CHECKING:
    import antioch


class RetargetReport(BaseModel):
    """What one run measured; ``passed`` is its verdict."""

    model_config = ConfigDict(frozen=True, extra="allow")

    passed: bool


def retarget_him_strokes(run: antioch.ScenarioRun, params: RetargetParams) -> RetargetReport:
    """Retarget HIM's mocap strokes onto a humanoid, gate them, and publish the kept strokes."""

    raise NotImplementedError("retarget_him_strokes is not implemented yet")
