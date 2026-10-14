"""
One canonical contact sequence for the reference oracle and the game adapter.

The oracle (:func:`lib.physics.reference.simulate.simulate`) and the game
adapter (``lib/physics/engine/stepper.py`` ``GamePhysics.events``) describe the
same rally with different bookkeeping:

- the oracle records ``roll``, ``pivot``, ``table_plane_outside``, ``rest``,
  ``out_of_bounds``, ``max_time`` and ``max_events`` events, which are not
  contacts;
- the adapter folds a sustained (resting or rolling) touch into one event, while
  the oracle records each slow bounce before it settles;
- the adapter blocks a blade from touching the ball again for
  :data:`PADDLE_RECONTACT_COOLDOWN_S` after it hits it; the oracle has no cooldown.

:func:`canonical_events` maps either list to the same :class:`CanonicalEvent`
sequence: bookkeeping events are dropped; consecutive touches of the same
surface (same kind and player) within :data:`SUSTAIN_MERGE_S` of the previous
touch are one event; a blade's touches within the recontact cooldown of its
first touch are one event. Data generation's labels, the dt-1e-4 identity gate
and the oracle-versus-adapter parity tests all use it.

**Codes** (``tt-datagen/1`` section 4.3): :data:`KIND_CODES` ``paddle 0, table 1,
net 2, floor 3, frame 4, wall 5`` (plus ``body 6`` and ``other 7``, which only the
game adapter produces), and :data:`REGION_NAMES` for ``event_region``, where
index 0 is the empty region (floor, paddle and body touches).

Regions: the table's ``top``, ``edge``, ``side`` or ``underside`` (from the
normal, :func:`lib.physics.reference.surfaces.classify_table_touch`); the net's
``mesh``, ``cord``, ``post`` or ``clamp``; the undercarriage part (``apron``,
``leg``, ...); the room box (``wall_pos_x``, ``ceiling``, ...).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lib.physics.reference.surfaces import classify_table_touch

Array = NDArray[np.float64]

SUSTAIN_MERGE_S: float = 0.020
"""Consecutive touches of the same surface within this long of the previous one are one sustained touch, s."""

PADDLE_RECONTACT_COOLDOWN_S: float = 0.03125
"""A blade's touches within this long of its first touch are one hit, s: the game adapter's recontact cooldown
(``lib.physics.engine.stepper.PADDLE_RECONTACT_COOLDOWN_S``; ``_recontact_steps`` = 120 steps at 3840 Hz). A test pins
the two equal."""

BALL_RADIUS_M: float = 0.02
"""Ball radius used to place an adapter event's contact point when none is given, m (the 40 mm ball)."""

BOOKKEEPING_KINDS: frozenset[str] = frozenset({"roll", "pivot", "table_plane_outside", "rest", "out_of_bounds", "max_time", "max_events"})
"""Oracle events that are not contacts."""

KIND_CODES: dict[str, int] = {"paddle": 0, "table": 1, "net": 2, "floor": 3, "frame": 4, "wall": 5, "body": 6, "other": 7}
"""``event_kind`` codes (``tt-datagen/1``: 0-5; ``body`` and ``other`` come only from the game adapter)."""

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


def _is_oracle(event: Any) -> bool:
    return hasattr(event, "state_before") and hasattr(event, "contact")


def oracle_touches(events: Sequence[Any]) -> list[RawTouch]:
    """
    The contacts of an oracle trajectory's events (``Trajectory.events``), bookkeeping dropped.

    :param events: :class:`lib.physics.reference.simulate.Event` objects.
    """

    touches = []
    for index, event in enumerate(events):
        if event.kind in BOOKKEEPING_KINDS or event.contact is None:
            continue
        contact = event.contact
        kind = str(event.kind)
        region = str(contact.region or "")
        if kind == "table":
            region = classify_table_touch(contact.normal)
        elif kind == "paddle":
            region = "blade_edge" if region == "blade_edge" else ""
        elif kind == "floor":
            region = ""
        player = str(event.surface) if kind == "paddle" else None
        before = _array(event.state_before, 9)
        after = _array(event.state_after, 9)
        touches.append(
            RawTouch(
                index, kind, region, player, float(event.time), _array(contact.point, 3), _array(contact.normal, 3), before, after, float(event.time), after
            )
        )
    return touches


def adapter_touches(events: Sequence[Any], *, regions: Mapping[int, str] | None = None, radius: float = BALL_RADIUS_M) -> list[RawTouch]:
    """
    The contacts of the game adapter's events (``GamePhysics.events``).

    :param events: ``lib.physics.engine.contacts.ContactEvent`` objects (or anything with their fields).
    :param regions: ``GamePhysics.event_regions`` (event index to ``mesh``, ``cord``, ``post``, ``clamp``, a frame
        part, a room box, or ``blade_edge`` for a paddle touch on the blade's rounded edge). The table's region comes
        from the normal.
    :param radius: Ball radius, m: the contact point is the ball centre less ``radius`` along the normal.
    """

    regions = regions or {}
    touches = []
    for position, event in enumerate(events):
        surface = str(event.surface)
        kind = surface if surface in KIND_CODES else "other"
        normal = _array(event.normal, 3)
        index = int(getattr(event, "index", position))
        if kind == "table":
            region = classify_table_touch(normal)
        elif kind == "paddle":
            # A touch on the blade's rounded edge (the adapter's ``blade_edge`` setting) keeps that region, as the
            # oracle's does; every other paddle touch is the face.
            region = "blade_edge" if regions.get(index) == "blade_edge" else ""
        elif kind in ("floor", "body"):
            region = ""
        else:
            region = str(regions.get(index, ""))
        player = event.actor if kind in ("paddle", "body") else None
        centre = _array(event.position, 3)
        after = _array(event.after, 9)
        end_state = getattr(event, "end", None)
        end = _array(end_state, 9) if end_state is not None else after
        t = float(event.time_s)
        touches.append(
            RawTouch(
                position,
                kind,
                region,
                player,
                t,
                centre - radius * normal,
                normal,
                _array(event.before, 9),
                after,
                t + float(getattr(event, "duration_s", 0.0)),
                end,
            )
        )
    return touches


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


def canonical_events(
    events: Sequence[Any],
    *,
    regions: Mapping[int, str] | None = None,
    radius: float = BALL_RADIUS_M,
    sustain_s: float = SUSTAIN_MERGE_S,
    cooldown_s: float = PADDLE_RECONTACT_COOLDOWN_S,
) -> tuple[CanonicalEvent, ...]:
    """
    The canonical contact sequence of an oracle event list or a game-adapter event list.

    The source is recognised by its events (an oracle ``Event`` has ``state_before`` and ``contact``). An empty list
    gives an empty sequence.

    :param events: ``Trajectory.events`` of :func:`lib.physics.reference.simulate.simulate`, or ``GamePhysics.events``.
    :param regions: Adapter only: ``GamePhysics.event_regions``.
    :param radius: Adapter only: ball radius, m (places the contact point).
    :param sustain_s: Same-surface touches within this long of the previous one are one event, s.
    :param cooldown_s: A blade's touches within this long of its first touch are one hit, s.
    :return: The canonical events in time order.
    """

    if not events:
        return ()
    if _is_oracle(events[0]):
        touches = oracle_touches(events)
    else:
        touches = adapter_touches(events, regions=regions, radius=radius)
    return merge(touches, sustain_s=sustain_s, cooldown_s=cooldown_s)


@dataclass(frozen=True)
class SequenceComparison:
    """
    How two canonical sequences agree.

    :ivar identical: The same ``(kind, region, player)`` sequence (``region`` ignored when not compared).
    :ivar first_difference: Index of the first differing event (``None`` when identical).
    :ivar max_time_s: Largest first-touch time difference over the paired events, s (paired up to the first difference).
    :ivar max_point_m: Largest ball-centre difference at the first touch, m.
    :ivar max_velocity_mps: Largest post-touch velocity difference, m/s.
    :ivar compared: Events paired.
    """

    identical: bool
    first_difference: int | None
    max_time_s: float
    max_point_m: float
    max_velocity_mps: float
    compared: int


def compare_sequences(a: Sequence[CanonicalEvent], b: Sequence[CanonicalEvent], *, regions: bool = True) -> SequenceComparison:
    """
    Compare two canonical sequences event by event.

    :param a: One sequence.
    :param b: The other.
    :param regions: Whether the region is part of an event's identity.
    """

    def key(event: CanonicalEvent) -> tuple[str, str, str | None]:
        return event.key if regions else (event.kind, "", event.player)

    first: int | None = None
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if key(x) != key(y):
            first = index
            break
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    paired = len(a) if first is None else first
    dt = dp = dv = 0.0
    for x, y in zip(a[:paired], b[:paired], strict=True):
        dt = max(dt, abs(x.t - y.t))
        dp = max(dp, float(np.max(np.abs(x.before[:3] - y.before[:3]))))
        dv = max(dv, float(np.max(np.abs(x.after[3:6] - y.after[3:6]))))
    return SequenceComparison(first is None, first, dt, dp, dv, paired)


def blade_edge_strikes(events: Sequence[CanonicalEvent]) -> int:
    """The QA count of edge strikes: the paddle events on the blade's rounded edge (region ``blade_edge``)."""

    return sum(1 for event in events if event.kind == "paddle" and event.region == "blade_edge")


__all__ = [
    "BALL_RADIUS_M",
    "BOOKKEEPING_KINDS",
    "KIND_CODES",
    "PADDLE_RECONTACT_COOLDOWN_S",
    "REGION_NAMES",
    "SUSTAIN_MERGE_S",
    "CanonicalEvent",
    "RawTouch",
    "SequenceComparison",
    "adapter_touches",
    "blade_edge_strikes",
    "canonical_events",
    "compare_sequences",
    "merge",
    "oracle_touches",
    "region_code",
]
