"""
The table's undercarriage, the steel apron, legs, stretchers, struts, feet and casters, as rigid surfaces.

The geometry is the table asset's (``reconstruction/scene/config.py`` ``table``
and ``build_scene.py`` ``build_table``), converted from the asset's world frame
(floor at ``z = 0``, playing surface at ``0.76``) to the table frame (playing
surface at ``z = 0``). As in the asset, the apron follows the table's half
length and width; the legs, stretchers, struts and casters sit at absolute
positions; and heights follow ``params.table`` ``height`` (the floor) and
``board_thickness``. A test pins every number to the asset configuration.

The asset's cylinders are 8- to 16-sided prisms; here they are exact
cylinders, within about 1 mm. The diagonal struts are capsules: round ends
in place of flat caps, both buried in the apron and the legs. All parts take
``fixture_contact``, painted steel. That is judgement: no ball-on-frame
measurement exists.

Contacts are ``kind = "frame"``, so nothing reads a leg as a table bounce, and
``region`` names the part (``apron``, ``leg``, ``foot``, ``stretcher``,
``spine``, ``strut``, ``caster``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from lib.physics.params import BallTableParams, ContactMaterial
from lib.physics.reference.surfaces import Contact, SurfaceKind

ASSET_TOP_Z_M = 0.76
"""The playing surface's height in the asset's world frame, m (the asset's floor is ``z = 0``)."""

APRON_INSET_M = 0.06
APRON_DEPTH_M = 0.055
APRON_TUBE_M = 0.025
LEG_TUBE_M = 0.040
END_LEGS_X_M = 1.19
"""|x| of the outer leg pair, m (absolute in the asset, like the other leg positions)."""
END_LEGS_Y_M = 0.62
CENTRE_LEGS_X_M = 0.36
CENTRE_LEGS_Y_M = 0.52
STRETCHER_Z_M = 0.16
"""Stretcher and spine centre height above the floor, m."""
CASTER_RADIUS_M = 0.04
CASTER_WIDTH_M = 0.025
FOOT_RADIUS_M = 0.022
FOOT_HEIGHT_M = 0.025
STRUT_RADIUS_M = 0.012
STRUT_INSET_M = 0.45
"""The diagonal strut leaves the apron this far inside each end line, m."""
STRUT_TOP_Y_SCALE = 1.08
STRUT_LOW_Z_M = 0.35
"""Height above the floor where each strut meets its undercarriage leg, m."""
CROSS_RAIL_INNER_X_M = 0.02
"""The inner cross rail's offset from the net line, m, before its half tube."""


@dataclass(frozen=True)
class Part:
    """
    One rigid primitive of the undercarriage, in the table frame.

    :param name: Region label (``apron``, ``leg``, ``foot``, ``stretcher``, ``spine``, ``strut``, ``caster``).
    :param shape: ``box`` (``a`` low corner, ``b`` high corner), ``cylinder_z`` or ``cylinder_x``
        (``a`` axis start, ``b`` axis end, ``radius``), or ``capsule`` (segment ``a`` to ``b``, ``radius``).
    :param a: First point, m.
    :param b: Second point, m.
    :param radius: Cylinder or capsule radius, m.
    """

    name: str
    shape: Literal["box", "cylinder_z", "cylinder_x", "capsule"]
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    radius: float = 0.0

    def aabb(self) -> tuple[float, float, float, float, float, float]:
        """The part's exact axis-aligned bounding box ``(x_lo, y_lo, z_lo, x_hi, y_hi, z_hi)``, m."""

        r = self.radius
        pad = {"box": (0.0, 0.0, 0.0), "cylinder_z": (r, r, 0.0), "cylinder_x": (0.0, r, r), "capsule": (r, r, r)}[self.shape]
        low = tuple(min(self.a[i], self.b[i]) - pad[i] for i in range(3))
        high = tuple(max(self.a[i], self.b[i]) + pad[i] for i in range(3))
        return (low[0], low[1], low[2], high[0], high[1], high[2])

    def closest(self, x: float, y: float, z: float) -> tuple[float, float, float, float]:
        """The part's closest point to ``(x, y, z)`` and the distance, m (0 inside)."""

        a, b = self.a, self.b
        if self.shape == "box":
            cx, cy, cz = min(max(x, a[0]), b[0]), min(max(y, a[1]), b[1]), min(max(z, a[2]), b[2])
        elif self.shape == "cylinder_z":
            dx, dy = x - a[0], y - a[1]
            radial = math.hypot(dx, dy)
            cx, cy = (a[0] + dx * self.radius / radial, a[1] + dy * self.radius / radial) if radial > self.radius else (x, y)
            cz = min(max(z, a[2]), b[2])
        elif self.shape == "cylinder_x":
            dy, dz = y - a[1], z - a[2]
            radial = math.hypot(dy, dz)
            cy, cz = (a[1] + dy * self.radius / radial, a[2] + dz * self.radius / radial) if radial > self.radius else (y, z)
            cx = min(max(x, a[0]), b[0])
        else:
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            length2 = ux * ux + uy * uy + uz * uz
            s = min(max(((x - a[0]) * ux + (y - a[1]) * uy + (z - a[2]) * uz) / length2, 0.0), 1.0)
            px, py, pz = a[0] + s * ux, a[1] + s * uy, a[2] + s * uz
            d = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
            if d <= self.radius:
                return x, y, z, 0.0
            k = self.radius / d
            cx, cy, cz = px + (x - px) * k, py + (y - py) * k, pz + (z - pz) * k
        return cx, cy, cz, math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)


def undercarriage_parts(params: BallTableParams) -> tuple[Part, ...]:
    """The undercarriage's primitives in the table frame, as the table asset builds them."""

    table = params.table
    hl, hw = 0.5 * table.length, 0.5 * table.width
    floor = -table.height
    board_bottom = -table.board_thickness
    apron_low = board_bottom - APRON_DEPTH_M
    tube, inset, leg = APRON_TUBE_M, APRON_INSET_M, LEG_TUBE_M
    rail_y = hw - inset
    stretcher_low, stretcher_high = floor + STRETCHER_Z_M - 0.015, floor + STRETCHER_Z_M + 0.015
    parts: list[Part] = []
    for s in (-1.0, 1.0):
        xa, xb = sorted((s * CROSS_RAIL_INNER_X_M, s * (hl - inset)))
        for sy in (-1.0, 1.0):
            y = sy * rail_y
            parts.append(Part("apron", "box", (xa, y - tube / 2, apron_low), (xb, y + tube / 2, board_bottom)))
        for x in (s * (hl - inset), s * CROSS_RAIL_INNER_X_M + s * tube / 2):
            parts.append(Part("apron", "box", (x - tube / 2, -rail_y, apron_low), (x + tube / 2, rail_y, board_bottom)))
        xe = s * END_LEGS_X_M
        for sy in (-1.0, 1.0):
            y = sy * END_LEGS_Y_M
            parts.append(Part("leg", "box", (xe - leg / 2, y - leg / 2, floor + FOOT_HEIGHT_M), (xe + leg / 2, y + leg / 2, apron_low)))
            parts.append(Part("foot", "cylinder_z", (xe, y, floor), (xe, y, floor + FOOT_HEIGHT_M), FOOT_RADIUS_M))
        parts.append(Part("stretcher", "box", (xe - 0.015, -END_LEGS_Y_M, stretcher_low), (xe + 0.015, END_LEGS_Y_M, stretcher_high)))
        xc = s * CENTRE_LEGS_X_M
        rc = CASTER_RADIUS_M
        for sy in (-1.0, 1.0):
            y = sy * CENTRE_LEGS_Y_M
            parts.append(Part("leg", "box", (xc - leg / 2, y - leg / 2, floor + 2 * rc + 0.02), (xc + leg / 2, y + leg / 2, apron_low)))
            parts.append(Part("caster", "box", (xc - 0.02, y - 0.02, floor + 2 * rc), (xc + 0.02, y + 0.02, floor + 2 * rc + 0.02)))
            parts.append(Part("caster", "cylinder_x", (xc - CASTER_WIDTH_M / 2, y, floor + rc), (xc + CASTER_WIDTH_M / 2, y, floor + rc), rc))
            parts.append(Part("strut", "capsule", (s * (hl - STRUT_INSET_M), y * STRUT_TOP_Y_SCALE, apron_low), (xc, y, floor + STRUT_LOW_Z_M), STRUT_RADIUS_M))
        parts.append(Part("stretcher", "box", (xc - 0.015, -CENTRE_LEGS_Y_M, stretcher_low), (xc + 0.015, CENTRE_LEGS_Y_M, stretcher_high)))
        sx_lo, sx_hi = sorted((xc, xe))
        parts.append(Part("spine", "box", (sx_lo, -0.015, stretcher_low), (sx_hi, 0.015, stretcher_high)))
    return tuple(parts)


@dataclass(frozen=True)
class TableFrame:
    """
    The undercarriage as one watched surface: the gap is the smallest over its parts.

    Far from the frame, the gap is the distance to the frame's bounding box
    minus the radius. That is a lower bound with the same sign, and saves
    evaluating every part on every step.

    :param radius: Ball radius, m.
    :param parts: The primitives.
    :param material: Ball-frame law (``fixture_contact``; judgement).
    """

    radius: float
    parts: tuple[Part, ...]
    material: ContactMaterial
    kind: SurfaceKind = "frame"
    name: str = "table_frame"

    @classmethod
    def from_params(cls, params: BallTableParams) -> TableFrame:
        """The undercarriage of ``params``."""

        return cls(params.ball.radius, undercarriage_parts(params), params.fixture_contact)

    @cached_property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The axis-aligned box around every part, m."""

        boxes = self._part_bounds
        low = tuple(min(box[i] for box in boxes) for i in range(3))
        high = tuple(max(box[i + 3] for box in boxes) for i in range(3))
        return (low[0], low[1], low[2]), (high[0], high[1], high[2])

    @cached_property
    def _part_bounds(self) -> tuple[tuple[float, float, float, float, float, float], ...]:
        return tuple(part.aabb() for part in self.parts)

    def _nearest(self, x: float, y: float, z: float) -> tuple[Part, tuple[float, float, float, float]]:
        best: tuple[Part, tuple[float, float, float, float]] | None = None
        best_distance = math.inf
        for part, (lx, ly, lz, hx, hy, hz) in zip(self.parts, self._part_bounds, strict=True):
            # A part's bounding box is never farther than the part, so it prunes exactly.
            dx, dy, dz = max(lx - x, 0.0, x - hx), max(ly - y, 0.0, y - hy), max(lz - z, 0.0, z - hz)
            if dx * dx + dy * dy + dz * dz >= best_distance * best_distance:
                continue
            found = part.closest(x, y, z)
            if found[3] < best_distance:
                best, best_distance = (part, found), found[3]
        assert best is not None
        return best

    def gap(self, t: float, state: Sequence[float]) -> float:
        """Distance from the ball's surface to the nearest part, m (a positive lower bound when far away)."""

        x, y, z = state[0], state[1], state[2]
        (lx, ly, lz), (hx, hy, hz) = self.bounds
        dx, dy, dz = max(lx - x, 0.0, x - hx), max(ly - y, 0.0, y - hy), max(lz - z, 0.0, z - hz)
        coarse = math.sqrt(dx * dx + dy * dy + dz * dz) - self.radius
        if coarse > 0.05:
            return coarse
        return self._nearest(x, y, z)[1][3] - self.radius

    def clearance(self, state: Sequence[float], gap: float) -> float:
        """How far the centre can move without the gap changing sign, m: ``|gap|`` (a distance, or a lower bound of one)."""

        return abs(gap)

    def contact(self, t: float, state: Sequence[float]) -> Contact | None:
        """The contact at the nearest part; ``None`` if the centre is inside it."""

        part, (cx, cy, cz, distance) = self._nearest(state[0], state[1], state[2])
        if distance < 1e-12:
            return None
        point = np.array([cx, cy, cz])
        normal = (np.asarray(state[:3], dtype=float) - point) / distance
        return Contact(self.kind, self.name, point, normal, np.zeros(3), self.material, math.inf, region=part.name)
