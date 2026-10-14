"""
The lab room as static contact geometry for the ball: walls, ceiling and the solid furniture, in the table frame.

Pure numpy, no simulator import. :mod:`lib.environment.scene.colliders` authors
these boxes as PhysX statics (native contacts, like the floor), and the game
adapter uses the same boxes to name an engine contact ``wall``
(:func:`lib.physics.engine.contacts.classify_native`) and to count a ball that
leaves the room.

**Geometry** comes from the published room asset (``demos/table-tennis/lab-room@v2``,
built by ``reconstruction/scene/build_scene.py`` from ``reconstruction/scene/config.py``):
the interior spans ``x`` in [-5.0, 5.0] m and ``y`` in [-4.7, 3.3] m, the walls are
0.10 m thick and 3.0 m tall, and the ceiling is 3.0 m above the floor. The asset's
origin is on the floor below the table-top centre, so a height ``h`` in the asset
is ``h - 0.76`` in the table frame. The solid furniture along the ``+y`` wall
(the two cabinets, the lockers, the pedestal and the cart) is taken from the same
config as boxes; open frames, the stool, cables and flat wall clutter are left out
(a ball passes through or under them, or they lie flat on a wall).

**Material** (``WALL_CONTACT``). No measurement of a table-tennis ball on a painted
wall was found. The cited rigid-target bound is the ball's own restitution on glass,
about 0.95 at 1 m/s and 0.92 near 5 m/s (Remond et al. 2022, the floor prior's
source in ``docs/physics-priors.md``). A lab wall is a painted partition on studs
that flexes and dissipates more than a slab, and the cabinets are thin sheet panels,
so the nominal restitution is set below that bound, at 0.80 (judgement; plausible
range 0.70-0.90). Friction 0.35 is the floor prior's judgement for a painted or
coated surface (ABS on smooth glass is 0.22, Remond et al. 2023). Dead balls reach
the walls at 0.5-5 m/s, where the bound's speed dependence is small, so one constant
serves PhysX, which has no speed curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.physics.params import ContactMaterial

Array = NDArray[np.float64]

SURFACE_HEIGHT_M: float = 0.76
"""The playing surface above the floor in the room asset (the table frame's ``z = 0``)."""

ROOM_X_M: tuple[float, float] = (-5.0, 5.0)
"""Interior extent of the room along X (table length), m."""
ROOM_Y_M: tuple[float, float] = (-4.7, 3.3)
"""Interior extent along Y, m."""
ROOM_HEIGHT_M: float = 3.0
"""Floor to ceiling, m."""
WALL_THICKNESS_M: float = 0.10
"""Wall and ceiling thickness in the asset, m."""

WALL_CONTACT: ContactMaterial = ContactMaterial(restitution=0.80, friction=0.35)
"""Ball against a wall, the ceiling or the furniture (see the module docstring: rigid-target bound from Remond 2022, the rest judgement)."""

TABLE_BOARD_THICKNESS_M: float = 0.025
"""The table-top board's thickness (``z`` from -0.025 to 0 in the table frame), as built in the table asset
(``tt_table`` ``top`` mesh, 0.735-0.76 m above the floor). The hybrid table sweeps it as a box
(:func:`lib.physics.engine.contacts.sweep_table_box`).

**Its edges, side and end faces and underside take the table's impact law** (``params.table_contact``). No
measurement of a ball on a table's edge or side was found. The faces are the same chipboard board as the playing
surface, struck in its plane, where it is stiffer than the top is under a bounce; the top's restitution (0.94 at
2-4 m/s, from the drag-corrected ITTF drop test) already sits at the ball's own restitution on rigid targets (0.92-0.95
at 1-5 m/s, Remond et al. 2022), so a stiffer face cannot give more and the top's law is the nearest cited value
(judgement). The playing surface itself is unchanged: the reference model's and the calibration harness's plane."""

FIXTURE_CONTACT: ContactMaterial = ContactMaterial(restitution=0.85, friction=0.25)
"""The contract's nominal ``fixture_contact`` (``lib.physics.params``; the scene and the stepper use the contract's):
ball against the net clamps and the table's undercarriage (painted metal and hard plastic). No measurement found: the
ball's restitution on rigid targets is 0.92-0.95 at 1-5 m/s (Remond et al. 2022), so 0.85 (judgement; plausible
0.75-0.92); friction 0.25, a little above ABS on smooth glass (0.22, Remond et al. 2023; judgement). The steel posts
take ``post_contact`` (0.946, the ITTF T3 steel-block bounce, drag-corrected)."""


@dataclass(frozen=True)
class RoomBox:
    """
    One axis-aligned static box of the room, table frame.

    :ivar name: Collider name (the prim's last path element).
    :ivar low: Minimum corner, m.
    :ivar high: Maximum corner, m.
    """

    name: str
    low: tuple[float, float, float]
    high: tuple[float, float, float]

    @property
    def centre(self) -> tuple[float, float, float]:
        """Box centre, m."""

        return (0.5 * (self.low[0] + self.high[0]), 0.5 * (self.low[1] + self.high[1]), 0.5 * (self.low[2] + self.high[2]))

    @property
    def size(self) -> tuple[float, float, float]:
        """Edge lengths, m."""

        return (self.high[0] - self.low[0], self.high[1] - self.low[1], self.high[2] - self.low[2])


def _box(name: str, x: tuple[float, float], y: tuple[float, float], z_asset: tuple[float, float]) -> RoomBox:
    return RoomBox(name, (x[0], y[0], z_asset[0] - SURFACE_HEIGHT_M), (x[1], y[1], z_asset[1] - SURFACE_HEIGHT_M))


def room_boxes() -> tuple[RoomBox, ...]:
    """
    Walls, ceiling and solid furniture as boxes in the table frame.

    The walls overlap at the corners and run from the floor to the ceiling's top, so no gap is left at an edge.
    """

    (x0, x1), (y0, y1), h, t = ROOM_X_M, ROOM_Y_M, ROOM_HEIGHT_M, WALL_THICKNESS_M
    return (
        _box("wall_neg_x", (x0 - t, x0), (y0 - t, y1 + t), (0.0, h + t)),
        _box("wall_pos_x", (x1, x1 + t), (y0 - t, y1 + t), (0.0, h + t)),
        _box("wall_neg_y", (x0 - t, x1 + t), (y0 - t, y0), (0.0, h + t)),
        _box("wall_pos_y", (x0 - t, x1 + t), (y1, y1 + t), (0.0, h + t)),
        _box("ceiling", (x0 - t, x1 + t), (y0 - t, y1 + t), (h, h + t)),
        # Solid furniture against the +Y wall (reconstruction/scene/config.py "props").
        _box("cabinet_low", (-2.9, -1.95), (2.68, y1), (0.0, 0.85)),
        _box("cabinet_tall", (-3.6, -2.95), (2.75, y1), (0.0, 1.9)),
        _box("lockers", (0.0, 2.4), (2.9, y1), (0.0, 1.0)),
        _box("pedestal", (0.07, 0.47), (2.44, 2.84), (0.0, 0.85)),
        _box("cart", (1.34, 1.96), (2.555, 3.005), (0.0, 0.57)),
    )


ROOM_BOXES: tuple[RoomBox, ...] = room_boxes()
"""The room's static boxes (see :func:`room_boxes`)."""

_LOW = np.array([box.low for box in ROOM_BOXES])
_HIGH = np.array([box.high for box in ROOM_BOXES])


def room_gap(position: ArrayLike, radius: float) -> tuple[float, Array, str]:
    """
    Distance from the ball's surface to the nearest room box, with the unit normal from that box into the ball.

    :param position: Ball centre, table frame, m.
    :param radius: Ball radius, m.
    :return: ``(gap, normal, box name)``; a centre inside a box gets the normal of the nearest face out of it.
    """

    centre = np.asarray(position, dtype=np.float64)
    closest = np.clip(centre, _LOW, _HIGH)
    offset = centre - closest
    distance = np.linalg.norm(offset, axis=1)
    k = int(np.argmin(distance))
    if distance[k] > 1e-12:
        return float(distance[k]) - radius, offset[k] / distance[k], ROOM_BOXES[k].name
    # Inside a box: leave through the nearest face.
    to_low = centre - _LOW[k]
    to_high = _HIGH[k] - centre
    depths = np.concatenate([to_low, to_high])
    face = int(np.argmin(depths))
    normal = np.zeros(3)
    normal[face % 3] = -1.0 if face < 3 else 1.0
    return -float(depths[face]) - radius, normal, ROOM_BOXES[k].name


def inside_room(position: ArrayLike, *, margin: float = 0.0) -> bool:
    """Whether a ball centre is inside the room's walls and below its ceiling (``margin`` m inside each)."""

    x, y, z = (float(v) for v in np.asarray(position, dtype=np.float64)[:3])
    return (
        ROOM_X_M[0] + margin <= x <= ROOM_X_M[1] - margin
        and ROOM_Y_M[0] + margin <= y <= ROOM_Y_M[1] - margin
        and z <= ROOM_HEIGHT_M - SURFACE_HEIGHT_M - margin
    )


def left_room(before: ArrayLike, after: ArrayLike) -> bool:
    """Whether the ball's centre left the room over one step (passed through a wall or the ceiling)."""

    return inside_room(before) and not inside_room(after)


__all__ = [
    "FIXTURE_CONTACT",
    "ROOM_BOXES",
    "ROOM_HEIGHT_M",
    "ROOM_X_M",
    "ROOM_Y_M",
    "SURFACE_HEIGHT_M",
    "TABLE_BOARD_THICKNESS_M",
    "WALL_CONTACT",
    "WALL_THICKNESS_M",
    "RoomBox",
    "inside_room",
    "left_room",
    "room_boxes",
    "room_gap",
]
