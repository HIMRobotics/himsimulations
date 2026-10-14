"""
One canonical contact sequence from raw ball touches.

Simulators describe the same rally with different bookkeeping: one folds a sustained (resting or rolling) touch into
one event while another records each slow bounce before it settles, and one blocks a blade from touching the ball
again for :data:`PADDLE_RECONTACT_COOLDOWN_S` after it hits it. :func:`merge` maps raw touches to one
:class:`CanonicalEvent` sequence: consecutive touches of the same surface (same kind and player) within
:data:`SUSTAIN_MERGE_S` of the previous touch are one event, and a blade's touches within the recontact cooldown of
its first touch are one event.

**Codes:** :data:`KIND_CODES` ``paddle 0, table 1, net 2, floor 3, frame 4, wall 5, body 6, other 7``, and
:data:`REGION_NAMES` for the region, where index 0 is the empty region (floor, paddle and body touches).

Regions: the table's ``top``, ``edge``, ``side`` or ``underside`` (from the normal); the net's ``mesh``, ``cord``,
``post`` or ``clamp``; the undercarriage part (``apron``, ``leg``, ...); the room box (``wall_pos_x``, ``ceiling``, ...).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

SUSTAIN_MERGE_S: float = 0.020
"""Consecutive touches of the same surface within this long of the previous one are one sustained touch, s."""

PADDLE_RECONTACT_COOLDOWN_S: float = 0.03125
"""A blade's touches within this long of its first touch are one hit, s: the replay adapter's recontact cooldown
(``lib.physics.engine.stepper.PADDLE_RECONTACT_COOLDOWN_S``, 120 steps at 3840 Hz)."""


KIND_CODES: dict[str, int] = {"paddle": 0, "table": 1, "net": 2, "floor": 3, "frame": 4, "wall": 5, "body": 6, "other": 7}
"""Event kind codes (``body`` and ``other`` come only from the replay's adapter)."""

REGION_NAMES: tuple[str, ...] = (
    "",
    "top",
    "edge",
    "side",
    "underside",
    "mesh",
    "cord",
    "post",
    "clamp",
    "apron",
    "leg",
    "foot",
    "stretcher",
    "spine",
    "strut",
    "caster",
    "wall_neg_x",
    "wall_pos_x",
    "wall_neg_y",
    "wall_pos_y",
    "ceiling",
    "cabinet_low",
    "cabinet_tall",
    "lockers",
    "pedestal",
    "cart",
    "blade_edge",
)
"""``event_region`` names by index; 0 is no region. :func:`region_code` adds nothing: an unknown region raises.
``blade_edge`` (appended last, so earlier indices are unchanged) is a paddle touch on the blade's rounded edge
(:func:`lib.physics.reference.blade.blade_solid_gap`): the QA count of edge strikes is the paddle events with it."""


@dataclass(frozen=True)
class CanonicalEvent:
    """
    One canonical contact.

    :ivar kind: ``paddle``, ``table``, ``net``, ``floor``, ``frame``, ``wall``, ``body`` or ``other``.
    :ivar region: The touched part (see the module notes), ``""`` for none.
    :ivar player: The blade's owner for ``paddle`` (the oracle paddle's name, the adapter's actor), the role for
        ``body``, else ``None``.
    :ivar t: Time of the first touch, s.
    :ivar point: Contact point on the surface at the first touch, m.
    :ivar normal: Unit normal from the surface into the ball at the first touch.
    :ivar before: Ball state (position, velocity, spin) just before the first touch.
    :ivar after: Ball state just after the first touch.
    :ivar merged: Raw events folded into this one (1 for a clean bounce or hit).
    :ivar t_end: Time of the last folded touch (the end of a sustained adapter event), s.
    :ivar end: Ball state after the last folded touch.
    :ivar raw: Indices of the folded raw events in the input list.
    """

    kind: str
    region: str
    player: str | None
    t: float
    point: Array
    normal: Array
    before: Array
    after: Array
    merged: int = 1
    t_end: float = math.nan
    end: Array = field(default_factory=lambda: np.full(9, np.nan))
    raw: tuple[int, ...] = ()

    @property
    def key(self) -> tuple[str, str, str | None]:
        """``(kind, region, player)``: what the identity of two sequences compares."""

        return (self.kind, self.region, self.player)

    @property
    def kind_code(self) -> int:
        """``event_kind``."""

        return KIND_CODES[self.kind]

    @property
    def region_code(self) -> int:
        """``event_region``."""

        return region_code(self.region)


def region_code(region: str) -> int:
    """The index of a region in :data:`REGION_NAMES`."""

    try:
        return REGION_NAMES.index(region)
    except ValueError as exc:
        raise ValueError(f"unknown region {region!r}: add it to REGION_NAMES") from exc


@dataclass(frozen=True)
class RawTouch:
    """A contact from either source, before merging (``t_end``/``end`` cover an adapter's sustained event)."""

    index: int
    kind: str
    region: str
    player: str | None
    t: float
    point: Array
    normal: Array
    before: Array
    after: Array
    t_end: float
    end: Array


def _array(value: Any, size: int) -> Array:
    return np.asarray(value, dtype=np.float64).reshape(size).copy()


def merge(touches: Sequence[RawTouch], *, sustain_s: float = SUSTAIN_MERGE_S, cooldown_s: float = PADDLE_RECONTACT_COOLDOWN_S) -> tuple[CanonicalEvent, ...]:
    """
    Fold raw touches (in time order) into canonical events.

    - A paddle touch within ``cooldown_s`` of the first touch of an earlier hit by the same blade is part of that hit
      (the adapter would have blocked it).
    - Any other touch of the same kind and player as the event just before it, within ``sustain_s`` of that event's
      last touch, extends it (a sustained touch, or the oracle's slow bounces before it settles).
    """

    out: list[CanonicalEvent] = []
    for touch in touches:
        target: int | None = None
        if touch.kind == "paddle":
            for k in range(len(out) - 1, -1, -1):
                earlier = out[k]
                if earlier.kind == "paddle" and earlier.player == touch.player:
                    if touch.t - earlier.t < cooldown_s:
                        target = k
                    break
        elif out and out[-1].kind == touch.kind and out[-1].player == touch.player and touch.t - out[-1].t_end <= sustain_s:
            target = len(out) - 1
        if target is not None:
            held = out[target]
            out[target] = CanonicalEvent(
                held.kind,
                held.region,
                held.player,
                held.t,
                held.point,
                held.normal,
                held.before,
                held.after,
                held.merged + 1,
                max(held.t_end, touch.t_end),
                touch.end,
                (*held.raw, touch.index),
            )
            continue
        out.append(
            CanonicalEvent(
                touch.kind, touch.region, touch.player, touch.t, touch.point, touch.normal, touch.before, touch.after, 1, touch.t_end, touch.end, (touch.index,)
            )
        )
    return tuple(out)


__all__ = ["KIND_CODES", "PADDLE_RECONTACT_COOLDOWN_S", "REGION_NAMES", "SUSTAIN_MERGE_S", "CanonicalEvent", "RawTouch", "merge", "region_code"]
