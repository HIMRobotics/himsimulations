"""
Rallies between the robot and HIM's distilled opponent, over the robot scene's coupled ball, and the ball forecast the
policy reads.

**The rally** (:class:`CoupledRally`). The scene calls it as its hooks (:class:`lib.robots.table_scene.TableHooks`):

- **every physics step** (:meth:`CoupledRally.opponent_blade`): the step the ball just flew is ingested (its paddle,
  table, net and floor contacts), the opponent decides, refines and executes its strokes, and its blade for the next
  step is handed to the coupling;
- **every policy step** (:meth:`CoupledRally.observe`): an env with no rally on, its robot settled, gets the opponent's
  next feed (one of its recorded opening shots); a ball the opponent sent is classified (bound for the robot's half,
  or let go); and the ball forecast is returned for the policy's observation;
- **on a reset** (a fall): the env's rally ends, recorded as cut by the reset.

The robot is player 0 at the table's ``-x`` end; the opponent is player 1. The robot plays every ball as the policy
chooses; nothing plans its stroke.

**A legal return** is the driver's call: the first table contact after the robot's hit lands on the opponent's half,
before the floor or any other surface (net touches allowed). A **receive** is an incoming ball predicted to bounce on
the robot's half; balls going out are let go, and are the opponent's error.

**The forecast** (:func:`forecast_columns_from`, 30 columns): the ball's predicted bounce on the robot's half, and
its position 0.1-0.4 s ahead, from the calibrated free flight. It is predicted once per flight, when the flight starts.
A serve, which bounces first on the server's half, is predicted through that bounce (:func:`predict_flight`); without
that, the policy reads a serve as a ball with no bounce coming and returns 22 % of HIM's recorded serves instead of
98 %.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lib.opponent.rally import AWAIT, EXECUTED, OVER, RallySim

Tensor = Any

ROBOT_END: float = -1.0
"""The robot's end of the table (its x-sign, the table frame)."""

ROBOT, OPPONENT = 0, 1
"""The players: the robot's blade is the coupling's blade 0, the opponent's blade 1."""

ROBOT_DUE = 6
"""A rally phase beside :mod:`lib.opponent.rally`'s: an incoming ball waits for the robot's receive to be classified."""

FORECAST_TIMES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
"""The look-ahead times, s: four samples 0.1 s (5 policy steps) apart cover the last two-thirds of an incoming flight."""

FORECAST_PRIOR_MEAN: tuple[float, ...] = (
    (1.2, -0.2, 0.0, 0.2, -3.5, 0.0, 2.5, 0.0, -60.0, 0.0, 0.5) + (0.65, -0.27, -0.15) * 4 + (0.5,) * 4 + (4.0, -28.0, -1.0)
)
"""What each forecast column reads with no prediction (a dead ball, a flight not bound for the robot): the column's
typical value in the robot's frame. The policy trained with these."""

FORECAST_BOUNCE_VALID: int = 10
"""The bounce block's valid flag's column within the forecast."""

FORECAST_AHEAD_VALID: tuple[int, ...] = (23, 24, 25, 26)
"""The look-ahead positions' valid flags' columns within the forecast."""

FORECAST_VALID_FLAGS: tuple[int, ...] = (FORECAST_BOUNCE_VALID, *FORECAST_AHEAD_VALID)
"""Every valid flag's column within the forecast."""

SETTLE_S: float = 0.5
"""After an env reset the robot stands this long before its env's next rally."""

RESTART_GAP_S: float = 1.0
"""After a rally ends, the env waits this long before the next feed (the robot back at its ready pose)."""

SERVE_OPEN_S: float = 0.005
"""A serve's receive opens this long after the serve's own-half bounce, as the ball comes down there."""

OVER_REASONS: dict[int, str] = {
    1: "opponent_stopped",
    2: "opponent_unreachable",
    3: "opponent_let_go",
    4: "missed",
    5: "cap",
    7: "env_reset",
    8: "robot_let_go",
    9: "ball_dead",
}
"""Why a rally ended (:attr:`RallySim.over_reason` codes 1-5, then the driver's)."""


# ---- the forecast ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FlightForecast:
    """
    One flight's stored prediction for the forecast (:func:`predict_flight`), on the ball's clock.

    :ivar t_bounce: The table touch the bounce block reports, absolute time (``inf``: none within the predictor's
        horizon): the flight's next touch, or a serve's touch after its own-half bounce.
    :ivar after: The ball just after it.
    :ivar ok: It is a bounce on the receiver's half with nothing first.
    :ivar toward_robot: The flight heads to the robot's half.
    :ivar t_pre: A serve's own-half bounce before it, absolute time (``inf``: none, every flight that is not a serve).
    :ivar pre_after: The ball just after that bounce.
    """

    t_bounce: Tensor
    after: Tensor
    ok: Tensor
    toward_robot: Tensor
    t_pre: Tensor
    pre_after: Tensor


def predict_flight(params: Any, ball: Tensor, rows: Tensor, t_now: float, serve: Tensor | None = None) -> FlightForecast:
    """
    A new flight's prediction for the forecast, from its ball at the flight's start.

    A flight that is not a serve: its next table touch, a bounce on its receiver's half when nothing stops it first
    (:func:`lib.opponent.ballistics.predict_bounce`). A serve bounces first on the server's half by rule, so for a
    serve whose next touch is that own-half bounce, the prediction goes on through it (the table's impact law) to the
    following touch, which the bounce block then reports; the own-half bounce is kept for the look-ahead. Physics only:
    nothing says where the robot hits.

    :param params: The envs' ball physics (``VecParams``).
    :param ball: The flights' balls at their start ``(R, 9)``, table frame; ``rows`` their envs (rows of ``params``).
    :param t_now: The ball's clock now (the predictions' times are absolute).
    :param serve: ``(R,)``: which flights are serves (the rally's opening ball before any paddle touch); ``None``: none.
    """

    from lib.opponent.ballistics import predict_bounce
    from lib.physics.vectorized._torch import torch

    ball = ball.to(params.dtype)
    toward_robot = ball[:, 3] < 0.0
    end = torch.where(toward_robot, torch.full_like(ball[:, 3], ROBOT_END), torch.full_like(ball[:, 3], -ROBOT_END))
    bounce = predict_bounce(params, ball, rows, end)
    t_bounce = torch.where(torch.isfinite(bounce.t), t_now + bounce.t.to(params.dtype), torch.full_like(bounce.t, float("inf")))
    after, ok = bounce.after, bounce.ok
    t_pre = torch.full_like(t_bounce, float("inf"))
    pre_after = torch.zeros_like(after)
    if serve is not None and bool(serve.any()):
        served = torch.nonzero(serve.to(ok.device).bool()).flatten()
        # the serve's next touch, read as a bounce on the server's own half (the far end from its receiver)
        own = predict_bounce(params, ball[served], rows[served], -end[served])
        if bool(own.ok.any()):
            chained = served[own.ok]
            t_own = t_now + own.t[own.ok].to(params.dtype)
            second = predict_bounce(params, own.after[own.ok], rows[chained], end[chained])
            t_bounce, after, ok = t_bounce.clone(), after.clone(), ok.clone()
            t_bounce[chained] = torch.where(torch.isfinite(second.t), t_own + second.t.to(params.dtype), torch.full_like(t_own, float("inf")))
            after[chained] = second.after
            ok[chained] = second.ok
            t_pre[chained] = t_own
            pre_after[chained] = own.after[own.ok]
    return FlightForecast(t_bounce, after, ok, toward_robot, t_pre, pre_after)


def forecast_columns_from(
    params: Any,
    ball: Tensor,
    active: Tensor,
    t_now: float,
    t_bounce: Tensor,
    after: Tensor,
    ok: Tensor,
    toward_robot: Tensor,
    position: Tensor,
    rotation: Tensor,
    table_offset: Tensor,
    t_pre: Tensor | None = None,
    pre_after: Tensor | None = None,
) -> Tensor:
    """
    The 30 forecast columns, physics only, positions, velocities and spins in the robot's base frame:

    - **the predicted bounce on the robot's half** (10 + 1): its point (3), the time to it (1), the ball's velocity
      (3) and spin (3) just after it, and its valid flag;
    - **the ball 0.1, 0.2, 0.3 and 0.4 s ahead** (12 + 4): free flight from now, or from just after the bounce once
      the bounce falls inside the window, with a valid flag each;
    - **the live spin** (3).

    A column with nothing to predict reads :data:`FORECAST_PRIOR_MEAN`, its flag 0.

    :param ball: Every ball now ``(N, 9)``, table frame; ``active`` which are live.
    :param t_bounce: The flight's next table touch on its receiver's half, absolute ball-clock time (``inf``: none within
        the predictor's horizon); ``after`` the ball just after it; ``ok`` it is a bounce on that half (nothing first).
    :param toward_robot: The flight heads to the robot's half (the bounce block reports only that half's bounce).
    :param position: The base frame's origin (``rotation`` its axes) in the env's frame; ``table_offset`` the table
        frame's origin in the same frame.
    :param t_pre: A serve's own-half bounce before ``t_bounce``, absolute time (``inf``: none), and ``pre_after`` the
        ball just after it (:func:`predict_flight`); the look-ahead flies through it. ``None``: no serve is chained.
    """

    from lib.opponent.ballistics import flight_at
    from lib.physics.vectorized._torch import torch

    n = ball.shape[0]
    dtype = position.dtype
    device = position.device
    rows = torch.arange(n, device=device)
    to_base = rotation.transpose(-1, -2)
    mean = torch.as_tensor(FORECAST_PRIOR_MEAN, dtype=dtype, device=device)

    def base(vector: Tensor) -> Tensor:
        return (to_base @ vector.to(dtype)[..., None])[..., 0]

    def at(point: Tensor) -> Tensor:
        return base(point.to(dtype) + table_offset.to(dtype) - position)

    live = active.bool()
    ball = ball.to(params.dtype)
    ahead_of_bounce = t_bounce > t_now  # the flight's bounce is still to come
    # the bounce block: the robot's half only, while it is ahead
    bounce_valid = live & ok & toward_robot & ahead_of_bounce
    to_bounce = (t_bounce - t_now).clamp(min=0.0).to(dtype)
    bounce = torch.cat([at(after[:, :3]), to_bounce[:, None], base(after[:, 3:6]), base(after[:, 6:9])], -1)
    bounce = torch.where(bounce_valid[:, None], bounce, mean[:10])
    # the look-ahead: free flight from now, or from just after the bounce once the bounce falls inside the window
    floor = -params.table_height.to(params.dtype) + params.radius.to(params.dtype)
    # a serve whose own-half bounce is still to come flies from just after it once it falls inside the window
    serving = None
    if t_pre is not None and pre_after is not None:
        serving = live & torch.isfinite(t_pre) & (t_pre > t_now)
        if not bool(serving.any()):
            serving = None
    positions, flags = [], []
    for ahead in FORECAST_TIMES:
        past = ahead_of_bounce & (t_now + ahead >= t_bounce)
        dt_free = torch.full((n,), float(ahead), dtype=params.dtype, device=device)
        free = flight_at(params, ball, rows, dt_free, 8)
        dt_after = (t_now + ahead - t_bounce).clamp(min=0.0).to(params.dtype)
        dt_after = torch.where(torch.isfinite(dt_after), dt_after, torch.zeros_like(dt_after))
        bounced = flight_at(params, after.to(params.dtype), rows, dt_after, 8)
        state = torch.where((past & ok)[:, None], bounced, free)
        if serving is not None and t_pre is not None and pre_after is not None:
            between = serving & ~past & (t_now + ahead >= t_pre)
            dt_pre = (t_now + ahead - t_pre).clamp(min=0.0).to(params.dtype)
            dt_pre = torch.where(torch.isfinite(dt_pre), dt_pre, torch.zeros_like(dt_pre))
            state = torch.where(between[:, None], flight_at(params, pre_after.to(params.dtype), rows, dt_pre, 8), state)
        # a flight stopped by something other than a bounce on its receiver's half (the net, the table's outside) is not
        # forecast past that contact; nor below the floor
        valid = live & ~(past & ~ok) & (state[:, 2] > floor)
        positions.append(torch.where(valid[:, None], at(state[:, :3]), mean[11 + 3 * len(flags) : 14 + 3 * len(flags)]))
        flags.append(valid)
    spin = torch.where(live[:, None], base(ball[:, 6:9]), mean[27:30])
    return torch.cat([bounce, bounce_valid.to(dtype)[:, None], *positions, torch.stack(flags, -1).to(dtype), spin], -1)


# ---- the opponent's feeds and the robot's receives ------------------------------------------------------------------


def feed_states(rules: Any, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """
    ``n`` of the opponent's recorded feeds (profile 1, with their jitter), from its end toward the robot: the feeder's
    hitter frame turned half a turn about z (x and y of position, velocity and spin negated).
    """

    _index, local = rules.draw_feed(np.ones(n, dtype=np.int64), rng)
    state = np.asarray(local, dtype=np.float64).copy()
    if -ROBOT_END > 0:
        for start in (0, 3, 6):
            state[:, start : start + 2] *= -1.0
    return state


def bound_for_robot(params: Any, ball: Tensor, rows: Tensor, bounced: Tensor) -> Tensor:
    """
    Which incoming balls the robot must play: those predicted to bounce on its half with nothing first, and those that
    already have. A ball that will not (out, into the net) is let go.

    :param ball: The balls now ``(R, 9)``, table frame; ``rows`` their envs (rows of ``params``).
    :param bounced: ``(R,)``: the ball already bounced on the robot's half.
    """

    from lib.opponent.ballistics import predict_bounce
    from lib.physics.vectorized._torch import torch

    ball = ball.to(torch.float64)
    legal = torch.ones(ball.shape[0], dtype=torch.bool, device=ball.device)
    pending = ~bounced.to(ball.device).bool()
    if bool(pending.any()):
        index = torch.nonzero(pending).flatten()
        end = torch.full((index.numel(),), ROBOT_END, dtype=ball.dtype, device=ball.device)
        legal[index] = predict_bounce(params, ball[index], rows[index], end).ok
    return legal


# ---- the rally state machine ----------------------------------------------------------------------------------------


class _StepLog:
    """One step's contacts in :class:`lib.physics.vectorized.sim.EventLog`'s shape (one chunk), for RallySim's events."""

    def __init__(self) -> None:
        self.env: list[Tensor] = []
        self.time: list[Tensor] = []
        self.surface: list[Tensor] = []
        self.point: list[Tensor] = []
        self.after: list[Tensor] = []

    def set(self, env: Tensor, time: Tensor, surface: Tensor, point: Tensor, after: Tensor) -> None:
        self.env, self.time, self.surface, self.point, self.after = [env], [time], [surface], [point], [after]


class _ExternalBall:
    """Stands in for RallySim's own batch: the event log of the step being ingested and the balls' activity."""

    def __init__(self, active: Tensor) -> None:
        self.log = _StepLog()
        self.active = active


class Rallies(RallySim):
    """
    :class:`lib.opponent.rally.RallySim`'s state machine over an external ball (the scene's coupling): the opponent
    decides, refines and executes exactly as in the reference, and the robot's receives are the policy's.

    :param opponent: The opponent (its ``params`` are the envs' physics: the coupling's live ``VecParams``).
    :param stance: ``(N, 2)``: the opponent's ready stance, table frame.
    :param active: ``(N,)``: the external balls' activity (a view the driver reads).
    :param t0: The external ball's time now.
    :param h: The physics step.
    :param cap_time: A rally's longest duration, s.
    :param cap_shots: A rally's most shots.
    """

    def __init__(self, opponent: Any, stance: Tensor, active: Tensor, *, t0: float, h: float, cap_time: float = 30.0, cap_shots: int = 40) -> None:
        from lib.physics.vectorized._torch import torch

        params = opponent.params
        n, dev, dtype = params.n, params.device, params.dtype
        end = torch.tensor([ROBOT_END, -ROBOT_END], dtype=dtype, device=dev).expand(n, 2).clone()
        # the robot's slot never decides; profile 1 and the opponent's stance mirrored keep RallySim's per-player
        # buffers (ready blade, start hand) well defined
        profile = torch.ones((n, 2), dtype=torch.long, device=dev)
        stance = stance.to(dtype)
        super().__init__(
            opponent,
            _ExternalBall(active),
            t0=t0,
            end=end,
            profile=profile,
            stance=torch.stack([-stance, stance], 1),
            hitter=torch.ones(n, dtype=torch.long, device=dev),
            t_hit=torch.full((n,), t0 - 1.0, dtype=dtype, device=dev),
            dt=h,
            cadence=8,
            cap_time=cap_time,
            cap_shots=cap_shots,
        )
        self.phase[:] = OVER
        self.finished[:] = True
        self.in_rally = torch.zeros(n, dtype=torch.bool, device=dev)
        self.rally_start = torch.full((n,), float("inf"), dtype=dtype, device=dev)
        self.ended_at = torch.full((n,), -float("inf"), dtype=dtype, device=dev)
        self.opponent_live = torch.zeros(n, dtype=torch.bool, device=dev)
        # per rally: paddle hits and legal shots per player, and the receives the robot had to play
        self.hits = torch.zeros((n, 2), dtype=torch.long, device=dev)
        self.legal = torch.zeros((n, 2), dtype=torch.long, device=dev)
        self.decided = torch.zeros((n, 2), dtype=torch.long, device=dev)
        self.robot_receives = torch.zeros(n, dtype=torch.long, device=dev)
        # the shot in flight: who hit it (-1 none); whether the robot's receive is open (classified and not yet hit)
        self.shot_player = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.receive_open = torch.zeros(n, dtype=torch.bool, device=dev)
        # whether each rally opened with a serve, which by rule bounces first on the server's half
        self.serve = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t_origin = float(t0)
        self.completed: list[dict[str, Tensor]] = []
        self.counts = {"robot_receives": 0, "robot_let_go": 0, "robot_stray_hits": 0, "opponent_decisions": 0}

    def start(self, rows: Tensor, states: Tensor, t_now: float, *, serves: Tensor | None = None) -> None:
        """
        New rallies in ``rows``: the opponent's opening balls ``states`` (already in the external ball) in flight from
        now. The robot's receive opens at once, or for a serve (``serves``) when the ball has bounced on the server's half.
        """

        from lib.physics.vectorized._torch import torch

        dtype = self.params.dtype
        states = states.to(dtype)
        now = torch.full((rows.numel(),), t_now, dtype=dtype, device=rows.device)
        latency = self.opponent.rules.limits["arm_latency_s"]
        self.hitter[rows] = OPPONENT
        self.t_hit[rows] = now - latency  # the robot's receive is classified at once
        self.phase[rows] = AWAIT
        self.t_due[rows] = now
        self.receives[rows] = 0
        self.shots[rows] = 0
        self.static_count[rows] = 0
        self.finished[rows] = False
        self.over_reason[rows] = -1
        self.capped[rows] = False
        self.seen[rows] = False
        self.t_b[rows] = float("nan")
        self.after_b[rows] = 0.0
        self.ev_t[rows, 0] = -float("inf")
        self.ev_t[rows, 1] = now
        self.ev_after[rows] = states[:, None, :].expand(-1, 2, -1)
        self.deadline[rows] = float("inf")
        for c in self.commands:
            c.active[rows] = False
            if c.tick0 is not None and c.tick1 is not None:
                c.tick0[0][rows] = float("nan")
                c.tick1[0][rows] = float("nan")
        slots = torch.cat([rows * 2, rows * 2 + 1])
        self.pelvis.start[slots] = float("inf")
        self.pelvis.end[slots] = float("inf")
        self.pelvis.coeffs[slots] = 0.0
        self.pelvis.rest[slots] = torch.cat([self.stance[rows, 0], self.stance[rows, 1]])
        self.last_contact[slots] = -float("inf")
        self.in_rally[rows] = True
        self.rally_start[rows] = now
        self.hits[rows] = 0
        self.legal[rows] = 0
        self.decided[rows] = 0
        self.robot_receives[rows] = 0
        self.shot_player[rows] = -1
        self.receive_open[rows] = False
        self.serve[rows] = False if serves is None else serves.to(self.serve.device).bool()
        if serves is not None:
            # a serve's receive waits for its own-half bounce
            self.t_due[rows] = torch.where(self.serve[rows], torch.full_like(now, float("inf")), self.t_due[rows])

    def abort(self, rows: Tensor) -> None:
        """Rallies cut by an env reset (a fall): ended, recorded as ``env_reset``."""

        from lib.physics.vectorized._torch import torch

        live = rows[self.in_rally[rows] & ~self.finished[rows]]
        if live.numel():
            self.over_reason[live] = 7
            self.finished[live] = True
            self._record(live)
        self.in_rally[rows] = False
        self.commands[OPPONENT].active[rows] = False
        self.ended_at[rows] = torch.as_tensor(self.t, dtype=self.ended_at.dtype)

    def idle(self, t_now: float, gap_s: float = RESTART_GAP_S) -> Tensor:
        """Envs with no rally on, ended at least ``gap_s`` ago."""

        from lib.physics.vectorized._torch import torch

        return torch.nonzero(~self.in_rally & (t_now - self.ended_at >= gap_s)).flatten()

    def ingest(self, contacts: dict[str, Tensor] | None, t_now: float) -> None:
        """
        One physics step of the external ball, ended at ``t_now``: its contacts (``env``, ``time``, ``surface``,
        ``point``, ``after``; the coupling's ``StepContacts.flat``), then RallySim's step logic without the ball's own
        stepping.
        """

        from lib.physics.vectorized._torch import torch

        self.t = float(t_now)
        self.steps += 1
        was = self.finished.clone()
        if contacts is not None and contacts["env"].numel():
            self.ball.log.set(
                contacts["env"],
                contacts["time"].to(self.params.dtype),
                contacts["surface"],
                contacts["point"].to(self.params.dtype),
                contacts["after"].to(self.params.dtype),
            )
            self._log_seen = 0
            shots = self.shots.clone()
            self._shot_outcomes(contacts)
            self._events()
            new = torch.nonzero(self.shots > shots).flatten()
            if new.numel():
                # the robot's hit counts only on an open receive; a stray one (a let-go ball, a second touch) is no shot
                stray = (self.hitter[new] == ROBOT) & ~self.receive_open[new]
                if bool(stray.any()):
                    self.shot_player[new[stray]] = -1
                    self.counts["robot_stray_hits"] += int(stray.sum())
                    new = new[~stray]
            if new.numel():
                self.hits[new, self.hitter[new]] += 1
                self.shot_player[new] = self.hitter[new]
                robot = new[self.hitter[new] == ROBOT]
                self.receive_open[robot] = False
            self._serve_bounces(contacts)
        self.finished |= self.in_rally & ~self.ball.active
        self.finished |= (self.phase == OVER) & (self.static_count >= 2)
        cap = ~self.finished & self.in_rally & ((self.t - self.rally_start >= self.cap_time) | (self.shots >= self.cap_shots))
        if bool(cap.any()):
            self.capped |= cap
            self.finished |= cap
            self.over_reason = torch.where(cap & (self.over_reason < 0), torch.full_like(self.over_reason, 5), self.over_reason)
        missed = (self.phase == EXECUTED) & (self.t > self.deadline) & ~self.finished
        if bool(missed.any()):
            self.phase = torch.where(missed, torch.full_like(self.phase, OVER), self.phase)
            self.over_reason = torch.where(missed, torch.full_like(self.over_reason, 4), self.over_reason)
        ended = torch.nonzero(self.finished & ~was & self.in_rally).flatten()
        if ended.numel():
            # the ball went dead (the floor, at rest) before any decision ended the rally: the receiver's
            self.over_reason[ended] = torch.where(self.over_reason[ended] < 0, torch.full_like(ended, 9), self.over_reason[ended])
            self._record(ended)
            self.in_rally[ended] = False
            self.ended_at[ended] = self.t
            self.commands[OPPONENT].active[ended] = False
        if self.steps % self.cadence == 0:
            self._process()
        else:
            due = (
                ((self.phase == 2) & (self.t_due <= self.t))  # REFINE_DUE
                | (self.phase == 3)  # FINALIZE_DUE
                | ((self.phase == 1) & (self.t >= self.plan.first.time - 0.03))  # PRE_BOUNCE
            )
            if bool((due & (self.plan.first.time - self.t < 0.06 + self.dt)).any()):
                self._process(decisions=False)

    def _serve_bounces(self, contacts: dict[str, Tensor]) -> None:
        """A serve still waiting for its own-half bounce opens the robot's receive :data:`SERVE_OPEN_S` after the ball's
        first table touch on the server's half."""

        from lib.physics.vectorized._torch import torch
        from lib.physics.vectorized.geometry import BOARD, TABLE_TOP

        env = contacts["env"]
        waiting = self.serve[env] & (self.shots[env] == 0) & torch.isinf(self.t_due[env]) & self.in_rally[env]
        # the table's top or its board (the edge), on the server's half
        table = (contacts["surface"] == TABLE_TOP) | (contacts["surface"] == BOARD)
        server_half = table & (torch.sign(contacts["point"][:, 0].to(self.t_due.dtype)) == -ROBOT_END)
        hit = waiting & server_half
        if not bool(hit.any()):
            return
        first = torch.full((self.n,), float("inf"), dtype=self.t_due.dtype, device=env.device)
        first.scatter_reduce_(0, env[hit], contacts["time"][hit].to(self.t_due.dtype), reduce="amin")
        rows = torch.nonzero(torch.isfinite(first)).flatten()
        self.t_due[rows] = first[rows] + SERVE_OPEN_S

    def _shot_outcomes(self, contacts: dict[str, Tensor]) -> None:
        """
        The shot in flight's outcome from this step's static contacts: its first table contact on the receiver's half is
        legal, on its own half (or any other static surface but the net) it is not. Per env, the step's earliest static
        contact other than the net closes the shot.
        """

        from lib.physics.vectorized._torch import torch
        from lib.physics.vectorized.geometry import BOARD, NET, STATIC_COUNT, TABLE_TOP

        surface = contacts["surface"]
        closing = (surface < STATIC_COUNT) & (surface != NET)
        if not bool(closing.any()):
            return
        env = contacts["env"][closing]
        time = contacts["time"][closing].to(torch.float64)
        kind = surface[closing]
        x = contacts["point"][closing][:, 0].to(torch.float64)
        first = torch.full((self.n,), float("inf"), dtype=torch.float64, device=env.device)
        first.scatter_reduce_(0, env, time, reduce="amin")
        pick = time == first[env]
        # one closing contact per env (rows at the same earliest time are the same contact)
        chosen_kind = torch.full((self.n,), -1, dtype=kind.dtype, device=env.device)
        chosen_x = torch.zeros(self.n, dtype=torch.float64, device=env.device)
        has = torch.zeros(self.n, dtype=torch.bool, device=env.device)
        chosen_kind[env[pick]] = kind[pick]
        chosen_x[env[pick]] = x[pick]
        has[env[pick]] = True
        rows = torch.nonzero(has & (self.shot_player >= 0)).flatten()
        if rows.numel() == 0:
            return
        k = self.shot_player[rows]
        table = (chosen_kind[rows] == TABLE_TOP) | (chosen_kind[rows] == BOARD)
        legal = table & (torch.sign(chosen_x[rows]) == self.end[rows, 1 - k].to(torch.float64))
        self.decided[rows, k] += 1
        self.legal[rows, k] += legal.long()
        self.shot_player[rows] = -1

    def _record(self, rows: Tensor) -> None:
        self.completed.append(
            {
                "env": rows.clone(),
                "start_s": (self.rally_start[rows] - self.t_origin).clone(),
                "duration_s": (self.t - self.rally_start[rows]).clone(),
                "hits": self.hits[rows].clone(),
                "legal": self.legal[rows].clone(),
                "decided": self.decided[rows].clone(),
                "robot_receives": self.robot_receives[rows].clone(),
                "over_reason": self.over_reason[rows].clone(),
                "serve": self.serve[rows].clone(),
            }
        )

    def _decide(self, rows: Tensor) -> None:
        """The opponent's receives go to RallySim's decision; the robot's wait for :meth:`robot_due`."""

        robot = (1 - self.hitter[rows]) == ROBOT
        mine = rows[robot]
        if mine.numel():
            self.phase[mine] = ROBOT_DUE
        theirs = rows[~robot]
        if theirs.numel():
            super()._decide(theirs)
            self.counts["opponent_decisions"] += int(theirs.numel())

    def robot_due(self) -> Tensor:
        """The rows whose incoming ball waits for the robot's receive to be classified."""

        from lib.physics.vectorized._torch import torch

        finished: Any = self.finished
        return torch.nonzero((self.phase == ROBOT_DUE) & ~finished).flatten()

    def classify_receives(self, rows: Tensor, legal: Tensor) -> None:
        """
        The robot's receives: a ball bound for its half opens a receive, which only the ball's next event decides (the
        robot's hit, or the ball dead); any other ball is let go (``robot_let_go``, the opponent's error).
        """

        from lib.physics.vectorized._torch import torch

        mine, out = rows[legal], rows[~legal]
        self.phase[mine] = EXECUTED
        self.deadline[mine] = float("inf")
        self.receive_open[mine] = True
        self.robot_receives[mine] += 1
        self.phase[out] = OVER
        self.over_reason[out] = torch.full_like(out, 8)
        self.receive_open[out] = False
        self.counts["robot_receives"] += int(mine.numel())
        self.counts["robot_let_go"] += int(out.numel())

    def opponent_blade(self, t_start: float, h: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        The opponent's blade at the end of the step ``[t_start, t_start + h]`` (table frame; game axes ``(w, x, y, z)``)
        and whether it teleports there: wherever it is parked at either end of the step, so it never sweeps from below
        the floor to its contact or back.
        """

        from lib.opponent.blade import WINDOW_S, step_states

        command = self.commands[OPPONENT]
        park_p, park_q = self.parks[OPPONENT]
        _p0, _v0, _q0, p1, v1, q1 = step_states(command, t_start, h, park_p, park_q)
        live = command.active & (t_start + h >= command.t - WINDOW_S) & (t_start <= command.t + WINDOW_S)
        teleport = ~(live & self.opponent_live)
        self.opponent_live = live
        return p1, v1, q1, teleport


# ---- the scene's hooks ----------------------------------------------------------------------------------------------


class CoupledRally:
    """
    The robot against HIM's opponent in every env of a :func:`lib.robots.table_scene.build_table_scene` scene: the
    scene's hooks (:class:`lib.robots.table_scene.TableHooks`), and :meth:`play`, one graded pass of a policy.

    :param opponent_model: The opponent model's directory (``None``: fetched, :func:`lib.opponent.fetch_opponent_model`).
    :param seed: The feeds' and the opponent's draws.
    :param forecast_through_serve: Predict a serve through its own-half bounce (:func:`predict_flight`).
    :param cap_time: A rally's longest duration, s.
    """

    observation_size: int = len(FORECAST_PRIOR_MEAN)

    def __init__(self, opponent_model: Any = None, *, seed: int = 0, forecast_through_serve: bool = True, cap_time: float = 30.0) -> None:
        self.opponent_model = opponent_model
        self.seed = int(seed)
        self.forecast_through_serve = bool(forecast_through_serve)
        self.cap_time = cap_time
        self.rallies: Rallies | None = None
        self.rng = np.random.default_rng(self.seed)
        self.ingested_t = -float("inf")
        self.flight: dict[str, Tensor] = {}

    def restart(self) -> None:
        """Forget every rally: the next :meth:`observe` builds a fresh opponent and rally state (a new graded pass)."""

        self.rallies = None
        self.rng = np.random.default_rng(self.seed)
        self.ingested_t = -float("inf")

    # ---- every physics step ---------------------------------------------------------------------------------------

    def opponent_blade(self, scene: Any, t_end: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Ingest the step the coupling last flew (its contacts), then return the opponent's blade at the end of the step it
        flies next (``coupling.t + h`` on the ball's clock; ``t_end`` is the scene's matching time).
        """

        rallies = self._bind(scene)
        coupling = scene.coupling
        t = float(coupling.t)
        if t > self.ingested_t + 1e-9:
            contacts = coupling.contacts.flat() if getattr(coupling, "contacts", None) is not None else None
            rallies.ingest(contacts, t)
            self.ingested_t = t
        return rallies.opponent_blade(t, float(coupling.h))

    # ---- every policy step ----------------------------------------------------------------------------------------

    def observe(self, scene: Any) -> Tensor:
        """Serve the idle envs, classify the robot's receives, and return the ball forecast ``(N, 30)``."""

        self._bind(scene)
        self._serve(scene)
        self._receive(scene)
        return self.forecast(scene)

    def reset(self, scene: Any, env_ids: Tensor) -> None:
        """The rows were reset (a fall, or a new pass): their rallies end, cut by the reset."""

        from lib.physics.vectorized._torch import torch

        if self.rallies is None:
            return
        rows = torch.as_tensor(env_ids, device=self.rallies.in_rally.device).long()
        self.rallies.abort(rows)
        self.flight["t_hit"][rows] = float("nan")  # a fresh episode's flight is predicted afresh

    def _bind(self, scene: Any) -> Rallies:
        """The rally state and the opponent, built on first use against the scene's coupled ball."""

        if self.rallies is not None:
            return self.rallies
        from lib.opponent import load_him_opponent
        from lib.physics.vectorized._torch import torch

        coupling = scene.coupling
        params = coupling.physics.params
        opponent = load_him_opponent(params, seed=self.seed, model_dir=self.opponent_model)
        stance = torch.as_tensor(opponent.rules.draw_stance(np.ones(scene.num_envs, dtype=np.int64), self.rng), dtype=params.dtype, device=params.device)
        stance = -stance if -ROBOT_END > 0 else stance
        self.rallies = Rallies(opponent, stance, coupling.active, t0=float(coupling.t), h=float(coupling.h), cap_time=self.cap_time)
        n, device, dtype = scene.num_envs, params.device, params.dtype
        self.flight = {
            # each flight's predicted bounce on its receiver's half, keyed by the flight's last hit
            "t_hit": torch.full((n,), float("nan"), dtype=dtype, device=device),
            "t_bounce": torch.full((n,), float("inf"), dtype=dtype, device=device),
            "after": torch.zeros(n, 9, dtype=dtype, device=device),
            "ok": torch.zeros(n, dtype=torch.bool, device=device),
            "toward_robot": torch.zeros(n, dtype=torch.bool, device=device),
            # a serve's own-half bounce before the predicted one (``inf``: none) and the ball just after it
            "t_pre": torch.full((n,), float("inf"), dtype=dtype, device=device),
            "pre_after": torch.zeros(n, 9, dtype=dtype, device=device),
        }
        return self.rallies

    def _serve(self, scene: Any) -> None:
        """New rallies where none is on and the robot has settled: the opponent's recorded feeds, under nominal physics."""

        from lib.physics.vectorized._torch import torch

        rallies, coupling = self.rallies, scene.coupling
        assert rallies is not None
        t = float(coupling.t)
        idle = rallies.idle(t)
        if idle.numel():
            idle = idle[scene.episode_length_buf[idle] >= round(SETTLE_S / scene.step_dt)]
        if idle.numel() == 0:
            return
        params = coupling.physics.params
        states = torch.as_tensor(feed_states(rallies.opponent.rules, idle.numel(), self.rng), dtype=params.dtype, device=params.device)
        coupling.launch(idle, states)
        rallies.start(idle, states, t)

    def _receive(self, scene: Any) -> None:
        """The robot's receives: each incoming ball classified as bound for its half, or let go."""

        rallies, coupling = self.rallies, scene.coupling
        assert rallies is not None
        due = rallies.robot_due()
        if due.numel() == 0:
            return
        t_ball = float(coupling.t)
        bounced = rallies.seen[due] & (rallies.t_b[due] <= t_ball)
        rallies.classify_receives(due, bound_for_robot(coupling.physics.params, coupling.ball[due], due, bounced))

    def forecast(self, scene: Any) -> Tensor:
        """
        The :data:`FORECAST_PRIOR_MEAN`-shaped forecast now (:func:`forecast_columns_from`), in the robot's base frame.
        Each flight's bounce on its receiver's half is predicted once, when the flight starts (a new hit or serve).
        """

        from lib.physics.vectorized._torch import torch

        position, rotation = scene.base_frame()
        rallies, coupling, flight = self.rallies, scene.coupling, self.flight
        assert rallies is not None
        params = coupling.physics.params
        ball = coupling.ball.to(params.dtype)
        t_now = float(coupling.t)
        live = coupling.active.bool()
        fresh = live & ~(flight["t_hit"] == rallies.t_hit.to(flight["t_hit"].dtype))
        if bool(fresh.any()):
            rows = torch.nonzero(fresh).flatten()
            # a serve: the rally's opening ball, before any paddle touch
            serve = rallies.serve[rows] & (rallies.shots[rows] == 0) if self.forecast_through_serve else None
            predicted = predict_flight(params, ball[rows], rows, t_now, serve)
            flight["t_bounce"][rows] = predicted.t_bounce.to(flight["t_bounce"].dtype)
            flight["after"][rows] = predicted.after.to(flight["after"].dtype)
            flight["ok"][rows] = predicted.ok
            flight["toward_robot"][rows] = predicted.toward_robot
            flight["t_pre"][rows] = predicted.t_pre.to(flight["t_pre"].dtype)
            flight["pre_after"][rows] = predicted.pre_after.to(flight["pre_after"].dtype)
            flight["t_hit"][rows] = rallies.t_hit[rows].to(flight["t_hit"].dtype)
        return forecast_columns_from(
            params,
            ball,
            live,
            t_now,
            flight["t_bounce"].to(params.dtype),
            flight["after"],
            flight["ok"],
            flight["toward_robot"],
            position,
            rotation,
            scene.table_offset,
            t_pre=flight["t_pre"].to(params.dtype) if self.forecast_through_serve else None,
            pre_after=flight["pre_after"] if self.forecast_through_serve else None,
        )

    # ---- a graded pass --------------------------------------------------------------------------------------------

    def play(self, scene: Any, actor: Any, seconds: float, *, on_step: Any = None) -> dict[str, Any]:
        """
        One pass: every env reset to its stance, fresh rallies, then ``seconds`` of the actor playing.

        :param actor: Observations ``(N, 146)`` in, joint actions ``(N, 29)`` out.
        :param on_step: Called after every policy step with the step index (the film).
        :return: The pass's record: the completed rallies, the falls, the simulated span.
        """

        import torch

        self.restart()
        falls_before = scene.falls
        steps = round(seconds / scene.step_dt)
        with torch.inference_mode():
            observation = scene.reset_all()
            for step in range(steps):
                observation = scene.step(actor(observation))
                if on_step is not None:
                    on_step(step)
        rallies = self.rallies
        assert rallies is not None
        return {
            "completed": list(rallies.completed),
            "counts": dict(rallies.counts),
            "falls": scene.falls - falls_before,
            "env_seconds": scene.num_envs * steps * scene.step_dt,
            "seconds": steps * scene.step_dt,
        }


__all__ = [
    "FORECAST_PRIOR_MEAN",
    "FORECAST_TIMES",
    "FORECAST_VALID_FLAGS",
    "OVER_REASONS",
    "CoupledRally",
    "FlightForecast",
    "Rallies",
    "bound_for_robot",
    "feed_states",
    "forecast_columns_from",
    "predict_flight",
]
