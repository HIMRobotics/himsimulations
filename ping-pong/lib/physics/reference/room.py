"""
The lab room as rigid surfaces: four walls, the ceiling and the solid furniture along the +Y wall.

The geometry is the room asset's, as the replay's adapter uses it
(``lib/physics/engine/colliders.py`` ``room_boxes``): the interior spans
``x`` in [-5.0, 5.0] m and ``y`` in [-4.7, 3.3] m; the walls are 0.10 m thick;
the ceiling is 3.0 m above the floor. The asset's origin is on the floor, so
an asset height ``h`` is ``h - table.height`` in the table frame. The oracle
keeps its own copy, and a test pins it to the adapter's boxes.

Every box takes ``wall_contact`` (judgement; see ``priors.json``). Contacts
are ``kind = "wall"``, with the box's name as ``region``.
:func:`lib.physics.reference.simulate.simulate` includes the room only when
asked (``room=True``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lib.physics.params import BallTableParams, ContactMaterial
from lib.physics.reference.surfaces import Contact, SurfaceKind

ROOM_X_M = (-5.0, 5.0)
"""Interior extent along X, m."""
ROOM_Y_M = (-4.7, 3.3)
"""Interior extent along Y, m."""
ROOM_HEIGHT_M = 3.0
"""Floor to ceiling, m."""
WALL_THICKNESS_M = 0.10
"""Wall and ceiling thickness, m."""

FURNITURE_M: tuple[tuple[str, tuple[float, float], tuple[float, float], tuple[float, float]], ...] = (
    ("cabinet_low", (-2.9, -1.95), (2.68, ROOM_Y_M[1]), (0.0, 0.85)),
    ("cabinet_tall", (-3.6, -2.95), (2.75, ROOM_Y_M[1]), (0.0, 1.9)),
    ("lockers", (0.0, 2.4), (2.9, ROOM_Y_M[1]), (0.0, 1.0)),
    ("pedestal", (0.07, 0.47), (2.44, 2.84), (0.0, 0.85)),
    ("cart", (1.34, 1.96), (2.555, 3.005), (0.0, 0.57)),
)
"""Solid furniture against the +Y wall: name, x range, y range, height range above the floor, m."""


@dataclass(frozen=True)
class RoomBox:
    """
    One axis-aligned box of the room in the table frame.

    :param name: Box name (``wall_neg_x``, ``ceiling``, ``lockers``...).
    :param low: Minimum corner, m.
    :param high: Maximum corner, m.
    """

    name: str
    low: tuple[float, float, float]
    high: tuple[float, float, float]

    def closest(self, x: float, y: float, z: float) -> tuple[float, float, float, float]:
        """The box's closest point to ``(x, y, z)`` and the distance, m (0 inside)."""

        cx = min(max(x, self.low[0]), self.high[0])
        cy = min(max(y, self.low[1]), self.high[1])
        cz = min(max(z, self.low[2]), self.high[2])
        return cx, cy, cz, math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)


def room_boxes(params: BallTableParams) -> tuple[RoomBox, ...]:
    """Walls, ceiling and solid furniture in the table frame (the walls overlap at the corners and reach the ceiling's top)."""

    (x0, x1), (y0, y1), h, t = ROOM_X_M, ROOM_Y_M, ROOM_HEIGHT_M, WALL_THICKNESS_M
    floor = -params.table.height

    def box(name: str, x: tuple[float, float], y: tuple[float, float], z: tuple[float, float]) -> RoomBox:
        return RoomBox(name, (x[0], y[0], floor + z[0]), (x[1], y[1], floor + z[1]))

    return (
        box("wall_neg_x", (x0 - t, x0), (y0 - t, y1 + t), (0.0, h + t)),
        box("wall_pos_x", (x1, x1 + t), (y0 - t, y1 + t), (0.0, h + t)),
        box("wall_neg_y", (x0 - t, x1 + t), (y0 - t, y0), (0.0, h + t)),
        box("wall_pos_y", (x0 - t, x1 + t), (y1, y1 + t), (0.0, h + t)),
        box("ceiling", (x0 - t, x1 + t), (y0 - t, y1 + t), (h, h + t)),
        *(box(name, x, y, z) for name, x, y, z in FURNITURE_M),
    )


@dataclass(frozen=True)
class Room:
    """
    The room as one watched surface: the gap is the smallest over its boxes.

    :param radius: Ball radius, m.
    :param boxes: The boxes.
    :param material: Ball-wall law (``wall_contact``).
    """

    radius: float
    boxes: tuple[RoomBox, ...]
    material: ContactMaterial
    kind: SurfaceKind = "wall"
    name: str = "room"

    @classmethod
    def from_params(cls, params: BallTableParams) -> Room:
        """The room of ``params``."""

        return cls(params.ball.radius, room_boxes(params), params.wall_contact)

    def _nearest(self, x: float, y: float, z: float) -> tuple[RoomBox, tuple[float, float, float, float]]:
        best: tuple[RoomBox, tuple[float, float, float, float]] | None = None
        for box in self.boxes:
            found = box.closest(x, y, z)
            if best is None or found[3] < best[1][3]:
                best = (box, found)
        assert best is not None
        return best

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Distance from the ball's surface to the nearest wall, ceiling or furniture box, m."""

        return self._nearest(state[0], state[1], state[2])[1][3] - self.radius

    def clearance(self, state: Sequence[float], gap: float) -> float:
        """How far the centre can move without the gap changing sign, m (a distance: 1-Lipschitz)."""

        return abs(gap)

    def contact(self, t: float, state: Sequence[float]) -> Contact | None:
        """The contact at the nearest box; ``None`` if the centre is inside one."""

        box, (cx, cy, cz, distance) = self._nearest(state[0], state[1], state[2])
        if distance < 1e-12:
            return None
        point = np.array([cx, cy, cz])
        normal = (np.asarray(state[:3], dtype=float) - point) / distance
        return Contact(self.kind, self.name, point, normal, np.zeros(3), self.material, math.inf, region=box.name)
