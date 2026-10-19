"""
The rally: serves on schedule, physical point endings, and the labelled continuity ball.

The referee reads only the ball's contacts (paddle, table, net, floor) as the
physics produced them and applies the rules a point needs:

- after a serve the ball must bounce on the server's half, then the
  receiver's; after a stroke it must bounce on the far half;
- after its bounce on a player's half that player must hit it before a second
  bounce (``double_bounce``) or before it reaches the floor (``miss``);
- a ball that comes down on the floor, or on the striker's own half, without
  reaching the far half ends the point as ``net`` (when it touched the net
  since the stroke) or ``out``.

A point therefore always ends physically. When a simulated point ends before
the recorded rally did, the next recorded stroke of that point is played with
a **continuity ball**: a new ball launched from the striker's paddle toward the
stroke's landing zone, so the scene keeps following the recording. It is
labelled in the events and the telemetry, counted, and its strokes are never
counted as realised (they are not physics evidence).

The replay ends when the serve schedule is exhausted and the last point has
ended physically; the run's ``max_sim_s`` is only a safety cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from lib.physics.engine.contacts import ContactEvent
from lib.replay.session import End, Role, ScheduledServe, SessionPackage, StrokeTarget, other_role

PointEnd = Literal["double_bounce", "net", "out", "miss", "superseded"]
Phase = Literal["waiting", "serve_own_half", "to_far_half", "await_hit", "over"]
StrokeStatus = Literal["pending", "planned", "realised", "missed", "continuity"]

STROKE_WINDOW_S: float = 0.20
"""A paddle contact within this of a stroke's planned contact counts for it."""
PLAN_LEAD_S: float = 1.5
"""Earliest a stroke is planned before its recorded time."""
REPLAN_MIN_LEAD_S: float = 0.12
"""No re-plan closer than this to contact."""
STALL_S: float = 3.0
"""An in-play ball with no contact for this long has stopped (a miss)."""


@dataclass
class StrokeRecord:
    """
    What happened to one scheduled stroke.

    :ivar stroke: The schedule entry.
    :ivar status: ``pending``, ``planned``, ``realised``, ``missed`` or ``continuity``.
    :ivar plans: Plan count.
    :ivar contact_event: Index of the paddle contact that realised it.
    :ivar contact_time_s: When that contact happened.
    :ivar volley: The contact came before the ball bounced on the striker's half.
    """

    stroke: StrokeTarget
    status: StrokeStatus = "pending"
    plans: int = 0
    contact_event: int | None = None
    contact_time_s: float | None = None
    volley: bool = False


@dataclass
class PointRecord:
    """
    One point.

    :ivar serve: The schedule entry.
    :ivar served_at_s: When the ball was put into play.
    :ivar ended_at_s: When it ended physically.
    :ivar end: Why it ended.
    :ivar continuity_balls: Continuity balls launched within it.
    :ivar contacts: Paddle contacts within it.
    :ivar endings: Every physical ending ``(time, reason)``, including those a continuity ball followed.
    """

    serve: ScheduledServe
    served_at_s: float | None = None
    ended_at_s: float | None = None
    end: PointEnd | None = None
    continuity_balls: int = 0
    contacts: int = 0
    endings: list[tuple[float, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ServeAction:
    """Put the ball into play for a serve: the replay loop places it at the server's paddle with the serve state."""

    point: int
    serve: ScheduledServe


@dataclass(frozen=True)
class PlanAction:
    """Plan (or re-plan) a stroke from the live ball."""

    stroke: StrokeTarget
    replan: bool


@dataclass(frozen=True)
class ContinuityAction:
    """Launch a labelled continuity ball from the striker's paddle for a stroke the simulated point can no longer reach."""

    stroke: StrokeTarget


Action = ServeAction | PlanAction | ContinuityAction


@dataclass
class Rally:
    """
    The referee and scheduler of one replay.

    :param package: The session package manifest (already restricted to the points to play).
    """

    package: SessionPackage
    points: list[PointRecord] = field(init=False)
    strokes: list[StrokeRecord] = field(init=False)
    log: list[tuple[float, str, str]] = field(default_factory=list)
    phase: Phase = "waiting"
    point_index: int = -1
    last_striker: Role | None = None
    awaited: Role | None = None
    net_since_stroke: bool = False
    last_contact_s: float = -math.inf

    def __post_init__(self) -> None:
        self.points = [PointRecord(serve=serve) for serve in self.package.serves]
        self.strokes = [StrokeRecord(stroke=stroke) for stroke in self.package.strokes]
        self.ends: dict[Role, End] = {track.role: track.end for track in self.package.tracks}

    # ---- helpers -------------------------------------------------------------

    def _say(self, time: float, text: str, level: str = "INFO") -> None:
        self.log.append((time, level, text))

    def _half(self, event: ContactEvent) -> End:
        return "8E" if event.position[0] >= 0.0 else "8F"

    def end_of(self, role: Role) -> End:
        """A role's end."""

        return self.ends[role]

    def _current_strokes(self) -> list[StrokeRecord]:
        return [record for record in self.strokes if record.stroke.point == self.point_index]

    def next_stroke(self) -> StrokeRecord | None:
        """The current point's first stroke not yet played."""

        return next((record for record in self._current_strokes() if record.status in ("pending", "planned")), None)

    @property
    def in_play(self) -> bool:
        """Whether a ball is live in the current point."""

        return self.phase in ("serve_own_half", "to_far_half", "await_hit")

    @property
    def finished(self) -> bool:
        """
        The terminal condition: every serve dispatched, every recorded stroke played (realised, missed or
        with a continuity ball), and the last point ended physically.
        """

        points_over = bool(self.points) and all(point.ended_at_s is not None for point in self.points)
        return points_over and all(record.status not in ("pending", "planned") for record in self.strokes)

    def _end_point(self, time: float, reason: PointEnd) -> None:
        point = self.points[self.point_index]
        point.ended_at_s = time
        point.end = reason
        point.endings.append((time, reason))
        self.phase = "over"
        self.awaited = None
        self._say(time, f"point {self.point_index} ends: {reason}")

    # ---- events ----------------------------------------------------------------

    def on_events(self, events: list[ContactEvent], now: float) -> None:
        """
        Apply contacts in time order, then check for a stalled ball.

        :param events: New contact events.
        :param now: Current session time, s.
        """

        for event in sorted(events, key=lambda item: item.time_s):
            self._on_event(event)
        if self.in_play and now - self.last_contact_s > STALL_S:
            self._end_point(now, "miss")

    def _on_event(self, event: ContactEvent) -> None:
        if not self.in_play:
            return
        self.last_contact_s = event.time_s
        half = self._half(event)
        striker = self.last_striker
        assert striker is not None
        far = self.end_of(other_role(striker))
        if event.surface == "paddle":
            actor = event.actor
            if actor is None or actor == striker:
                return
            record = self.next_stroke()
            volley = self.phase != "await_hit"
            if record is not None and record.stroke.hitter == actor:
                record.status = "realised"
                record.contact_event = event.index
                record.contact_time_s = event.time_s
                record.volley = volley
            self.points[self.point_index].contacts += 1
            self.last_striker = actor  # type: ignore[assignment]
            self.awaited = None
            self.phase = "to_far_half"
            self.net_since_stroke = False
            self._say(event.time_s, f"{actor} strikes at {event.approach_speed_mps:.1f} m/s" + (" (volley)" if volley else ""))
        elif event.surface == "net":
            self.net_since_stroke = True
            self._say(event.time_s, "net touch")
        elif event.surface == "table":
            if self.phase == "serve_own_half":
                if half == self.end_of(striker):
                    self.phase = "to_far_half"
                else:
                    self._end_point(event.time_s, "out")
            elif self.phase == "to_far_half":
                if half == far:
                    self.phase = "await_hit"
                    self.awaited = other_role(striker)
                    self._say(event.time_s, f"bounce on the {half} half")
                else:
                    self._end_point(event.time_s, "net" if self.net_since_stroke else "out")
            elif self.phase == "await_hit":
                self._end_point(event.time_s, "double_bounce" if half == far else "miss")
        elif event.surface == "floor":
            if self.phase == "await_hit":
                self._end_point(event.time_s, "miss")
            else:
                self._end_point(event.time_s, "net" if self.net_since_stroke else "out")

    # ---- the schedule ---------------------------------------------------------

    def due(self, now: float, horizon: float) -> list[Action]:
        """
        What the replay loop must do this tick.

        :param now: Game time at the tick start, s.
        :param horizon: Game time at the tick end, s.
        :return: Actions in order.
        """

        actions: list[Action] = []
        next_point = self.point_index + 1
        if next_point < len(self.points) and self.points[next_point].serve.time_s < horizon:
            if self.in_play:
                self._end_point(now, "superseded")
            self.point_index = next_point
            point = self.points[next_point]
            point.served_at_s = point.serve.time_s
            self.phase = "serve_own_half" if point.serve.kind == "serve" else "to_far_half"
            self.last_striker = point.serve.server
            self.awaited = None
            self.net_since_stroke = False
            self.last_contact_s = now
            self._say(
                point.serve.time_s,
                f"point {next_point}: {point.serve.server} " + ("serves" if point.serve.kind == "serve" else "starts the rally (recorded mid-rally)"),
            )
            actions.append(ServeAction(point=next_point, serve=point.serve))
            return actions
        if self.point_index < 0:
            return actions
        record = self.next_stroke()
        if record is None:
            return actions
        stroke = record.stroke
        if self.in_play:
            if now > stroke.time_s + STROKE_WINDOW_S + 0.15:
                record.status = "missed"
                self._say(now, f"stroke {stroke.index} by {stroke.hitter} not reached")
            elif self.last_striker != stroke.hitter and now >= stroke.time_s - PLAN_LEAD_S:
                if record.status == "pending":
                    record.status = "planned"
                    record.plans += 1
                    actions.append(PlanAction(stroke=stroke, replan=False))
                elif record.plans == 1 and self.phase == "await_hit" and stroke.time_s - now > REPLAN_MIN_LEAD_S:
                    record.plans += 1
                    actions.append(PlanAction(stroke=stroke, replan=True))
        elif self.phase == "over":
            if record.status == "planned":
                # The point ended while this stroke was being attempted (or could no longer be).
                record.status = "missed"
                self._say(now, f"stroke {stroke.index} by {stroke.hitter} not reached")
            elif stroke.time_s < horizon:
                record.status = "continuity"
                point = self.points[self.point_index]
                point.continuity_balls += 1
                point.ended_at_s = None
                point.end = None
                self.phase = "to_far_half"
                self.last_striker = stroke.hitter
                self.awaited = None
                self.net_since_stroke = False
                self.last_contact_s = now
                self._say(stroke.time_s, f"CONTINUITY ball for stroke {stroke.index} by {stroke.hitter} (not physics evidence)", "WARN")
                actions.append(ContinuityAction(stroke=stroke))
        return actions

    # ---- results ------------------------------------------------------------------

    def realised_share(self) -> float:
        """Share of scheduled strokes realised by a physical paddle contact (continuity strokes count as not realised)."""

        return sum(1 for record in self.strokes if record.status == "realised") / len(self.strokes) if self.strokes else math.nan

    def counts(self) -> dict[str, int]:
        """Stroke outcomes, point endings and continuity balls."""

        out: dict[str, int] = {f"strokes_{status}": 0 for status in ("pending", "planned", "realised", "missed", "continuity")}
        for record in self.strokes:
            out[f"strokes_{record.status}"] += 1
        for reason in ("double_bounce", "net", "out", "miss", "superseded"):
            out[f"endings_{reason}"] = sum(1 for point in self.points for _time, ending in point.endings if ending == reason)
        out["continuity_balls"] = sum(point.continuity_balls for point in self.points)
        out["volleys"] = sum(1 for record in self.strokes if record.volley)
        return out


__all__ = ["PLAN_LEAD_S", "STROKE_WINDOW_S", "Action", "ContinuityAction", "PlanAction", "PointEnd", "PointRecord", "Rally", "ServeAction", "StrokeRecord"]
