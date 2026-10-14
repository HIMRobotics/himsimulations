"""
A humanoid with its paddle at the HIM table, balls launched at it, holding its default pose.

C0 interface stub: the owning builder replaces this body; the signature is the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from lib.config import TemplateParams

if TYPE_CHECKING:
    import antioch


class TemplateReport(BaseModel):
    """What one run measured; ``passed`` is its verdict."""

    model_config = ConfigDict(frozen=True, extra="allow")

    passed: bool


def load_humanoid_at_table(run: antioch.ScenarioRun, params: TemplateParams) -> TemplateReport:
    """A humanoid with its paddle at the HIM table, balls launched at it, holding its default pose."""

    raise NotImplementedError("load_humanoid_at_table is not implemented yet")
