"""
The humanoids, as data (:class:`~lib.robots.spec.RobotSpec`), and the Isaac Lab scene that puts one at the table.

To add a humanoid, write a module like ``unitree_g1.py`` and register its spec here.
"""

from __future__ import annotations

from lib.robots.spec import RobotSpec
from lib.robots.unitree_g1 import UNITREE_G1

ROBOTS: dict[str, RobotSpec] = {UNITREE_G1.name: UNITREE_G1}
"""Every humanoid, by name."""

__all__ = ["ROBOTS", "UNITREE_G1", "RobotSpec"]
