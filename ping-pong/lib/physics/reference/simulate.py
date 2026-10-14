"""
Event-driven reference simulation: flight, bounces, scripted paddles and sustained rolling.

The ball flies under :mod:`lib.physics.reference.flight` between contacts. At
each contact the impact law of :mod:`lib.physics.reference.impact` replaces the
velocity and spin instantaneously; position is continuous. The simulation ends
at the first terminal event:

- a contact whose kind (``table``, ``net``, ``floor``, ``paddle``, ``frame``,
  ``wall``) or surface name is in ``terminal_events`` (default: the floor);
- ``table_plane_outside`` if listed: the ball crossed the table plane off the
  table, the usual definition of a missed table. With ``fixtures``, a ball
  whose centre is within one radius of the board's edge at that moment is
  left to the board (an edge ball or a side hit), and no event is recorded;
- ``rest``: with ``rolling`` (the default), a ball that rebounds from the
  floor or the playing surface slower than ``rest_speed`` enters the
  sustained-rolling regime (a non-terminal ``roll`` event; see
  :mod:`lib.physics.reference.rolling`). It slides, rolls against the
  surface's rolling resistance, and is at ``rest`` once it stops. A slow
  rebound from another flat, upward-facing surface (the net's top, a frame
  rail, the net cord) is ``rest`` at once, as before. A slow rebound from the
  table board's edge is not rest, since that edge cannot hold a ball rolling
  off the table. Without ``rolling``, any slow rebound from an upward-facing
  surface is ``rest``;
- ``out_of_bounds``: the ball left the box ``|x| <= bx``, ``|y| <= by``, ``z <= bz``;
- ``max_time`` or ``max_events``.

**Cost.** The static surfaces of one parameter set are built once and reused
(they are immutable), and the integrator evaluates a static gap only when the
ball could have changed its sign (:mod:`lib.physics.reference.flight`): the
same trajectory, bit for bit, at a fraction of the cost. A paddle given as a
:class:`lib.physics.reference.paddle_track.PaddleTrack` is evaluated only near
the blade. ``max_step`` (opt-in) takes long steps where nothing can be reached
and ``dt`` steps near every surface and blade; ``graze`` (opt-in) also finds
touches that enter and leave between two step ends.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.physics.params import BallTableParams
from lib.physics.reference.flight import (
    CLEAR_MARGIN_M,
    DEFAULT_DT,
    STATE_SIZE,
    Dynamics,
    FlightModel,
    GapEvent,
    fixture_events,
    flight_events,
    integrate_until_event,
)
from lib.physics.reference.impact import ImpactResult, resolve_impact
from lib.physics.reference.rolling import (
    SLIP_EPSILON_MPS,
    STOP_SPEED_MPS,
    LeavesSurface,
    RollingModel,
    RollingSurface,
    SlipEnds,
    Stops,
    contact_slip,
    pivot_off_edge,
    settle_on,
    stick,
)
from lib.physics.reference.room import Room
from lib.physics.reference.surfaces import Contact, Floor, PaddleFaces, ScriptedPaddle, Surface, TableTop

TERMINAL_ALWAYS = ("rest", "out_of_bounds", "max_time", "max_events")
"""Event kinds that always end a simulation."""


@dataclass(frozen=True)
class Event:
    """
    One event of a simulated trajectory.

    :param kind: A contact kind (``table``, ``net``, ``floor``, ``paddle``, ``frame``, ``wall``),
        ``roll`` (entering sustained rolling), ``table_plane_outside``, ``rest``, ``out_of_bounds``,
        ``max_time`` or ``max_events``.
    :param surface: The surface's name (a paddle's own name), or empty for bookkeeping events.
    :param time: s.
    :param state_before: State just before, 9 floats.
    :param state_after: State just after (equal to ``state_before`` for non-impacts).
    :param contact: Contact frame for impacts.
    :param impact: Impact outcome for impacts.
    :param terminal: Whether this event ended the simulation.
    """

    kind: str
    surface: str
    time: float
    state_before: NDArray[np.float64]
    state_after: NDArray[np.float64]
    contact: Contact | None
    impact: ImpactResult | None
    terminal: bool

    @property
    def position(self) -> NDArray[np.float64]:
        """Ball centre at the event, m."""

        return self.state_before[:3]


@dataclass(frozen=True)
class Trajectory:
    """
    A simulated trajectory.

    Times are non-decreasing; an impact appears as two samples at the same
    time, the pre-impact state then the post-impact state.

    :param times: Sample times, s, shape ``(N,)``.
    :param states: States, shape ``(N, 9)``: position (m), velocity (m/s), spin (rad/s).
    :param events: Events in time order; the last is terminal.
    :param steps: Integration steps taken.
    :param coarse_steps: Of those, steps longer than ``dt`` (``max_step``).
    :param grazes: Contacts found inside a step by the graze search (``graze``).
    """

    times: NDArray[np.float64]
    states: NDArray[np.float64]
    events: tuple[Event, ...]
    steps: int = 0
    coarse_steps: int = 0
    grazes: int = 0

    @property
    def positions(self) -> NDArray[np.float64]:
        """Ball centre positions, m, shape ``(N, 3)``."""

        return self.states[:, 0:3]

    @property
    def velocities(self) -> NDArray[np.float64]:
        """Velocities, m/s, shape ``(N, 3)``."""

        return self.states[:, 3:6]

    @property
    def spins(self) -> NDArray[np.float64]:
        """Angular velocities, rad/s, shape ``(N, 3)``."""

        return self.states[:, 6:9]

    @property
    def terminal(self) -> Event:
        """The event that ended the simulation."""

        return self.events[-1]

    def events_of(self, kind: str) -> tuple[Event, ...]:
        """Events of one kind, in time order."""

        return tuple(event for event in self.events if event.kind == kind)


_CacheEntry = tuple[BallTableParams, tuple[Surface, ...], RollingSurface, RollingSurface, FlightModel]
_SURFACE_CACHE: dict[tuple[int, bool, bool, bool], _CacheEntry] = {}
_SURFACE_CACHE_SIZE = 16


def static_surfaces(params: BallTableParams, *, fixtures: bool = True, undercarriage: bool = True, room: bool = False) -> tuple[Surface, ...]:
    """
    The static surfaces :func:`simulate` watches for one parameter set, built once and reused.

    Parameter sets and surfaces are immutable, so a set seen again (by identity) reuses its surfaces.

    :return: Table top, net, floor, then (``fixtures``) the board, posts and clamps and (``undercarriage``) the frame,
        then (``room``) the room.
    """

    return _cached_surfaces(params, fixtures, undercarriage, room)[1]


def _cached_surfaces(params: BallTableParams, fixtures: bool, undercarriage: bool, room: bool) -> _CacheEntry:
    key = (id(params), fixtures, undercarriage, room)
    entry = _SURFACE_CACHE.get(key)
    if entry is not None and entry[0] is params:
        return entry
    surfaces: tuple[Surface, ...] = (
        *flight_events(params),
        *(fixture_events(params, undercarriage=undercarriage) if fixtures else ()),
        *((Room.from_params(params),) if room else ()),
    )
    floor_surface = RollingSurface("floor", "floor", -params.table.height, params.floor_contact)
    table_surface = RollingSurface("table", "table", 0.0, params.table_contact, 0.5 * params.table.length, 0.5 * params.table.width)
    entry = (params, surfaces, floor_surface, table_surface, FlightModel.from_params(params))
    if len(_SURFACE_CACHE) >= _SURFACE_CACHE_SIZE:
        _SURFACE_CACHE.pop(next(iter(_SURFACE_CACHE)))
    _SURFACE_CACHE[key] = entry
    return entry


def simulate(
    initial_state: ArrayLike,
    params: BallTableParams,
    *,
    t0: float = 0.0,
    paddles: Sequence[ScriptedPaddle] = (),
    dt: float = DEFAULT_DT,
    max_time: float = 5.0,
    terminal_events: Collection[str] = ("floor",),
    rest_speed: float = 0.05,
    bounds: tuple[float, float, float] = (6.0, 5.0, 6.0),
    record_every: int = 1,
    max_events: int = 1000,
    fixtures: bool = True,
    undercarriage: bool = True,
    room: bool = False,
    rolling: bool = True,
    edge_pivot: bool = True,
    max_step: float | None = None,
    clear_margin: float = CLEAR_MARGIN_M,
    graze: bool = False,
    blade_edge: bool = True,
) -> Trajectory:
    """
    Simulate one ball from an initial state until a terminal event.

    :param initial_state: Position (m), velocity (m/s) and spin (rad/s), 9 floats, table frame.
    :param params: Physics parameters.
    :param t0: Start time, s (paddle trajectories are evaluated at absolute times).
    :param paddles: Scripted paddles.
    :param dt: Flight step, s.
    :param max_time: Simulated duration cap, s.
    :param terminal_events: Contact kinds or surface names that end the run.
    :param rest_speed: Normal rebound speed below which a static-surface bounce ends the run, m/s.
    :param bounds: ``(bx, by, bz)``, m: leaving ``|x| <= bx``, ``|y| <= by``, ``z <= bz`` ends the run.
    :param record_every: Keep every n-th flight step (events are always kept).
    :param max_events: Cap on contacts.
    :param fixtures: Include the table board (ends, sides, edges, underside), the net posts and the
        clamps. False is the earlier oracle: the playing surface, the net box and the floor.
    :param undercarriage: With ``fixtures``, also include the table's apron, legs and frame (``kind = "frame"``).
    :param room: Include the room's walls, ceiling and furniture (``kind = "wall"``). Off by default, as in
        the calibration parity scene, whose long shots would otherwise reach a wall.
    :param rolling: A slow rebound on the floor or the playing surface enters the sustained-rolling regime
        (:mod:`lib.physics.reference.rolling`): the ball slides, rolls against its rolling resistance and
        stops, which is ``rest``. False is the earlier rule: any slow rebound from an upward-facing
        surface ends the run at ``rest``.
    :param edge_pivot: A ball rolling off the table slower than ``sqrt(g R)`` across the edge pivots about it
        before it leaves (:func:`lib.physics.reference.rolling.pivot_off_edge`; a ``pivot`` event). False
        releases it into flight as its centre passes the edge.
    :param max_step: Opt-in adaptive steps: the longest step, s, taken where no surface or blade can be reached within
        it; every step near one is ``dt``. ``None`` (the default) keeps the fixed ``dt`` grid. A scripted paddle that
        is not a :class:`lib.physics.reference.paddle_track.PaddleTrack` and has no windows keeps every step at ``dt``.
    :param clear_margin: Adaptive steps: clearance kept beyond the ball's reach over a long step, m.
    :param graze: Opt-in: search each ``dt`` step that ends clear of a surface, but came within its step's travel of it,
        for a touch inside the step (with or without ``max_step``).
    :param blade_edge: Paddles have their rounded edge (:class:`lib.physics.reference.surfaces.PaddleFaces`, the
        contract's ``rim_contact``). False is the earlier face-only blade, through whose edge a ball passes (an A/B and
        negative control: face touches are identical either way).
    :return: The trajectory and its events.
    """

    state0 = [float(x) for x in np.asarray(initial_state, dtype=float).reshape(STATE_SIZE)]
    if not all(np.isfinite(state0)):
        raise ValueError(f"initial state is not finite: {state0}")
    _params, statics, floor_surface, table_surface, model = _cached_surfaces(params, fixtures, undercarriage, room)
    surfaces: list[Surface] = [*statics, *(PaddleFaces.from_params(paddle, params, edge=blade_edge) for paddle in paddles)]
    terminal = frozenset(terminal_events)
    steps = coarse_steps = grazes = 0

    t = t0
    t_end = t0 + max_time
    state = state0
    times: list[float] = [t]
    states: list[list[float]] = [state]
    events: list[Event] = []
    contacts = 0
    rolling_on: RollingSurface | None = None
    sliding: SlipEnds | None = None

    def finish(kind: str, surface: str = "") -> None:
        frozen = np.array(states[-1])
        events.append(Event(kind, surface, times[-1], frozen, frozen.copy(), None, None, True))

    while True:
        dynamics: Dynamics = model
        internal: list[GapEvent] = []
        if rolling_on is not None:
            dynamics = RollingModel(model, params.ball, rolling_on, params.aero.gravity, sliding is not None)
            internal = [sliding if sliding is not None else Stops()]
            if math.isfinite(rolling_on.half_length):
                internal.append(LeavesSurface(rolling_on.half_length, rolling_on.half_width))
        segment = integrate_until_event(
            dynamics,
            state,
            t,
            t_end,
            [*surfaces, *internal],
            dt=dt,
            record_every=record_every,
            bounds=bounds,
            max_step=max_step,
            clear_margin=clear_margin,
            graze=graze,
        )
        steps += segment.steps
        coarse_steps += segment.coarse_steps
        grazes += segment.grazes
        times.extend(segment.times)
        states.extend(segment.states)
        hit = segment.hit
        if hit is None:
            finish("out_of_bounds" if segment.stopped else "max_time")
            break
        if hit.index >= len(surfaces):
            # A rolling-regime event: slip has ended, the ball has stopped, or it has rolled off the table.
            assert rolling_on is not None
            change = internal[hit.index - len(surfaces)]
            t, state = hit.time, list(hit.state)
            if isinstance(change, SlipEnds):
                state = stick(state, params.ball)
                sliding = None
                times.append(t)
                states.append(state)
                continue
            if isinstance(change, Stops):
                finish("rest", rolling_on.name)
                break
            pivot = pivot_off_edge(state, rolling_on, params.ball, params.aero.gravity) if edge_pivot else None
            if pivot is not None:
                duration, released = pivot
                events.append(Event("pivot", rolling_on.name, t, np.array(state), np.array(released), None, None, False))
                t, state = t + duration, released
                times.append(t)
                states.append(state)
            rolling_on = None
            continue
        surface = surfaces[hit.index]
        t, state = hit.time, list(hit.state)
        contact = surface.contact(t, state)
        if contact is None:
            # A centre within one radius of the board's top edge is the board's to resolve (an edge ball),
            # not a ball that passed the table.
            overhanging = fixtures and isinstance(surface, TableTop) and surface.overhangs(state[0], state[1])
            if surface.kind == "table" and isinstance(surface, TableTop) and not overhanging:
                stop = "table_plane_outside" in terminal
                before = np.array(state)
                events.append(Event("table_plane_outside", surface.name, t, before, before.copy(), None, None, stop))
                if stop:
                    break
            t, state = hit.time_after, list(hit.state_after)
            times.append(t)
            states.append(state)
            continue
        impact = resolve_impact(
            state[3:6],
            state[6:9],
            params.ball,
            contact.normal,
            contact.material,
            surface_velocity=contact.surface_velocity,
            effective_mass=contact.effective_mass,
            contact_offset=contact.offset,
            surface_inertia=contact.inertia,
        )
        if impact.regime == "separating":
            t, state = hit.time_after, list(hit.state_after)
            times.append(t)
            states.append(state)
            continue
        after = [*state[:3], *(float(x) for x in impact.velocity), *(float(x) for x in impact.spin)]
        stop = surface.kind in terminal or surface.name in terminal
        events.append(Event(surface.kind, surface.name, t, np.array(state), np.array(after), contact, impact, stop))
        times.append(t)
        states.append(after)
        state = after
        rolling_on = None
        contacts += 1
        if stop:
            break
        if contacts >= max_events:
            finish("max_events")
            break
        # A slow rebound from a surface that gravity presses the ball onto is sustained contact, not a bounce.
        # A slow rebound from a vertical face (the net mesh) is not: the ball just falls away.
        normal_z = float(contact.normal[2])
        if surface.kind == "paddle" or impact.separation_speed >= rest_speed or normal_z <= 0.0:
            continue
        flat = normal_z >= 1.0 - 1e-9
        target = None
        if rolling and flat and isinstance(surface, Floor):
            target = floor_surface
        elif rolling and flat and isinstance(surface, TableTop):
            target = table_surface
        if target is None:
            if rolling and not flat and surface.kind == "table":
                # The board's edge cannot hold a ball rolling off the table: it keeps falling off it. (The net
                # cord's slow touches stay ``rest``, the regime the game adapter calls sustained rolling.)
                continue
            finish("rest", surface.name)
            break
        settled = settle_on(state, target, params.ball)
        # Sliding ends a few steps' worth of slip before zero (see SlipEnds). A slip already inside that margin is
        # removed at once by the stick impulse, which reaches the rolling state the sliding would have reached.
        slip_rate = target.material.friction_at(0.0) * params.aero.gravity * (1.0 + 1.0 / params.ball.inertia_ratio)
        margin = max(3.0 * slip_rate * dt, SLIP_EPSILON_MPS)
        slipping = math.hypot(*contact_slip(settled, params.ball.radius)) > margin
        sliding = SlipEnds.starting(settled, params.ball.radius, margin) if slipping else None
        if not slipping:
            settled = stick(settled, params.ball)
        if not slipping and math.hypot(settled[3], settled[4]) < STOP_SPEED_MPS:
            times.append(t)
            states.append(settled)
            finish("rest", surface.name)
            break
        events.append(Event("roll", target.name, t, np.array(state), np.array(settled), contact, None, False))
        times.append(t)
        states.append(settled)
        state = settled
        rolling_on = target

    return Trajectory(np.array(times), np.array(states), tuple(events), steps, coarse_steps, grazes)


def simulate_many(initial_states: ArrayLike, params: BallTableParams, **kwargs: Any) -> list[Trajectory]:
    """
    :func:`simulate` for a batch of initial states with the same settings (a Gauss-Newton finite-difference batch, an
    aim grid): each trajectory is exactly ``simulate(state, params, **kwargs)``.

    It runs the scalar path per ball. A numpy lockstep integration (``flight.rk4_step_array``) costs about 125-210 us
    per step for any batch up to 256 balls against 2.1 us per ball for the scalar RK4 step (measured 2026-10-07), so it
    breaks even only near 70 balls before any event handling: slower for the planner's 4-8 probe batches, marginal for
    its 81-point aim grid. The batch API is here so a vectorised backend (torch, for phase two) can replace the loop.

    :param initial_states: Shape ``(B, 9)``.
    :param params: Physics parameters.
    :param kwargs: :func:`simulate`'s keyword arguments.
    :return: ``B`` trajectories.
    """

    states = np.asarray(initial_states, dtype=float).reshape(-1, STATE_SIZE)
    return [simulate(state, params, **kwargs) for state in states]


def _apex_height(model: FlightModel, launch_speed: float, radius: float, dt: float) -> float:
    """Highest point of the ball's bottom above the launch plane, launched straight up at ``launch_speed`` m/s."""

    class _Rising:
        def gap(self, t: float, state: Sequence[float]) -> float:
            return state[5]

    start = [0.0, 0.0, radius, 0.0, 0.0, launch_speed, 0.0, 0.0, 0.0]
    segment = integrate_until_event(model, start, 0.0, 10.0, (_Rising(),), dt=dt, record_every=10**9, time_tolerance=1e-12)
    if segment.hit is None:
        raise RuntimeError("ball did not reach its apex within 10 s")
    return segment.hit.state[2] - radius


def restitution_from_drop_test(drop_height: float, bounce_height: float, params: BallTableParams, *, dt: float = DEFAULT_DT) -> float:
    """
    The table restitution that reproduces a vertical drop test, with drag.

    The ball's lowest point starts ``drop_height`` above the middle of one
    half of the table, at rest, and rebounds to ``bounce_height``. Without air the answer is
    ``sqrt(bounce_height / drop_height)``; with air the drag on the way down
    and up makes the true ``e`` larger.

    :param drop_height: Release height of the ball's bottom above the table, m.
    :param bounce_height: Rebound apex of the ball's bottom, m.
    :param params: Physics parameters (the table restitution in them is ignored).
    :param dt: Flight step, s.
    :return: The normal restitution ``e``.
    """

    from scipy.optimize import brentq

    if not 0.0 < bounce_height <= drop_height:
        raise ValueError("need 0 < bounce_height <= drop_height")
    radius = params.ball.radius
    x_drop = -0.25 * params.table.length
    drop = simulate([x_drop, 0.0, radius + drop_height, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], params, dt=dt, terminal_events=("table",), max_time=10.0)
    table_hits = drop.events_of("table")
    if not table_hits:
        raise RuntimeError("drop test ball did not reach the table")
    impact_speed = -float(table_hits[0].state_before[5])
    model = FlightModel.from_params(params)
    upper = impact_speed
    if _apex_height(model, upper, radius, dt) < bounce_height:
        raise ValueError("bounce height is unreachable with restitution <= 1 under this drag")
    rebound_speed = cast(float, brentq(lambda v: _apex_height(model, v, radius, dt) - bounce_height, 1e-6, upper, xtol=1e-12))
    return rebound_speed / impact_speed
