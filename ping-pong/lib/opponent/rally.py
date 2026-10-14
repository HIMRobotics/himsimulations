"""
Rallies and single receives of distilled players in :class:`lib.physics.vectorized.BallBatch`: the generator's rally
loop (``Rally.play``) over many environments, for validation (O1-O4) and as the reference for wiring the opponent
into a robot environment.

Per environment the loop is the generator's: the receiver decides at the other player's hit plus 0.15 s from the
ball it perceives then (``ball_at``: re-propagated from the last event's after-state); if the ball has not yet
bounced it waits for the bounce on its half and, if the bounce plus 0.11 s comes before its contact's render tick,
refines; it then executes with noise and flies its blade (:mod:`.blade`) through the commanded contact; a receiver
that does not touch the ball by its contact plus 0.12 s has missed. The rally is over when a receiver does not plan
(stopped, unreachable, let go) or misses; the environment finishes at the decisive event (the floor, or the second
static contact after the last paddle touch).

Decisions are processed in batches every ``cadence`` steps from the exact decision-time ball (re-propagation), and at
once when a contact is near, so the batched timing changes nothing the generator would see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.geometry import STATIC_COUNT
from lib.physics.vectorized.params import VecParams
from lib.physics.vectorized.sim import BallBatch

from .ballistics import propagate
from .blade import blade_track, idle_command, parked_pose
from .opponent import ATTEMPT, Opponent, Plan
from .ops import yaw_xy
from .strike import ContactCommand
from .strokes import BladePlan, Piece

Tensor = Any

AWAIT, PRE_BOUNCE, REFINE_DUE, FINALIZE_DUE, EXECUTED, OVER = range(6)
TICK_S = 1.0 / 24.0


def pelvis_eval(start: Tensor, end: Tensor, coeffs: Tensor, rest: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """
    ``PelvisTrack.at`` for rows of move slots: the last move begun by ``t`` (evaluated at ``min(t, its end)``), else the
    rest pose. ``t`` is ``(R,)`` or ``(R, S)``; slots are ``(R, K)``, ``(R, K)``, ``(R, K, 6, 2)``, ``(R, 2)``.
    """

    single = t.dim() == 1
    tt = t[:, None] if single else t
    began = start[:, None, :] <= tt[..., None]
    key = torch.where(began, start[:, None, :].expand_as(began), torch.full_like(began, -float("inf"), dtype=start.dtype))
    slot = torch.argmax(key, -1)
    any_move = began.any(-1)
    s0 = start.gather(1, slot)
    e0 = end.gather(1, slot)
    r, n = slot.shape
    cc = coeffs[torch.arange(r, device=slot.device)[:, None].expand(r, n), slot]
    s = (torch.minimum(tt, e0) - s0).clamp_min(0.0)[..., None]
    x = cc[..., 0, :] + s * (cc[..., 1, :] + s * (cc[..., 2, :] + s * (cc[..., 3, :] + s * (cc[..., 4, :] + s * cc[..., 5, :]))))
    v = cc[..., 1, :] + s * (2.0 * cc[..., 2, :] + s * (3.0 * cc[..., 3, :] + s * (4.0 * cc[..., 4, :] + s * 5.0 * cc[..., 5, :])))
    moving = any_move & (tt <= e0)
    position = torch.where(any_move[..., None], x, rest[:, None, :].expand_as(x))
    velocity = torch.where(moving[..., None], v, torch.zeros_like(v))
    if single:
        return position[:, 0], velocity[:, 0]
    return position, velocity


class PelvisBatch:
    """Each player's planar hips: quintic moves between holds (``track.PelvisTrack``), ``K`` move slots per row."""

    def __init__(self, rest: Tensor, slots: int = 4) -> None:
        n = rest.shape[0]
        self.rest = rest.clone()
        self.start = torch.full((n, slots), float("inf"), dtype=rest.dtype, device=rest.device)
        self.end = torch.full((n, slots), float("inf"), dtype=rest.dtype, device=rest.device)
        self.coeffs = torch.zeros((n, slots, 6, 2), dtype=rest.dtype, device=rest.device)

    def at(self, rows: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Position and velocity of ``rows`` at times ``t`` (``(R,)`` or ``(R, S)``)."""

        return pelvis_eval(self.start[rows], self.end[rows], self.coeffs[rows], self.rest[rows], t)

    def slots(self, rows: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(start, end, coefficients, rest)`` of ``rows``: the input :meth:`Opponent.decide_dense` takes."""

        return self.start[rows], self.end[rows], self.coeffs[rows], self.rest[rows]

    def move(self, rows: Tensor, start: Tensor, end: Tensor, target: Tensor) -> None:
        """From the plan at ``start`` (keeping its velocity) to rest at ``target`` by ``end`` (rows where ``end > start``)."""

        go = end > start
        rows, start, end, target = rows[go], start[go], end[go], target[go]
        if rows.numel() == 0:
            return
        x0, v0 = self.at(rows, start)
        t = (end - start)[:, None]
        x1 = target
        c3 = (20.0 * (x1 - x0) - (12.0 * v0) * t) / (2.0 * t**3)
        c4 = (30.0 * (x0 - x1) + (16.0 * v0) * t) / (2.0 * t**4)
        c5 = (12.0 * (x1 - x0) - (6.0 * v0) * t) / (2.0 * t**5)
        coeffs = torch.stack([x0, v0, torch.zeros_like(x0), c3, c4, c5], 1)
        s_all, e_all = self.start[rows], self.end[rows]
        later = s_all >= start[:, None]
        s_all = torch.where(later, torch.full_like(s_all, float("inf")), s_all)
        e_all = torch.where(later, torch.full_like(e_all, float("inf")), torch.minimum(e_all, start[:, None]))
        # The new move goes in the free slot, or replaces the oldest (moves before the latest started one are spent).
        free = torch.isinf(s_all)
        slot = torch.where(free.any(1), torch.argmax(free.long(), 1), torch.argmin(s_all, 1))
        s_all[torch.arange(rows.numel(), device=rows.device), slot] = start
        e_all[torch.arange(rows.numel(), device=rows.device), slot] = end
        self.start[rows] = s_all
        self.end[rows] = e_all
        c_all = self.coeffs[rows]
        c_all[torch.arange(rows.numel(), device=rows.device), slot] = coeffs
        self.coeffs[rows] = c_all


@dataclass
class ReceiveLog:
    """Every decision made, concatenated after the run (one row per receive)."""

    parts: list[dict[str, Tensor]] = field(default_factory=list)

    def add(self, **columns: Tensor) -> None:
        self.parts.append({k: v.detach() for k, v in columns.items()})

    def tensors(self) -> dict[str, np.ndarray]:
        if not self.parts:
            return {}
        return {k: torch.cat([p[k] for p in self.parts]).cpu().numpy() for k in self.parts[0]}


class RallySim:
    """
    ``N`` environments of rallies (or single receives) between distilled players.

    :param opponent: The players' decision process (both players use it, each with its own profile).
    :param state: Ball states ``(N, 9)`` at ``t0``.
    :param t0: Start time (per row the hitter's hit ``t_hit`` may be earlier).
    :param end: ``(N, 2)``: each player's end x-sign (player 0 is A).
    :param profile: ``(N, 2)``: each player's profile (0 suited, 1 opponent).
    :param stance: ``(N, 2, 2)``: each player's ready stance, table frame.
    :param hitter: ``(N,)``: who hit the ball now in flight.
    :param t_hit: ``(N,)``: when.
    :param pelvis: ``(N, 2, 2)``: the hips now (default: the stances).
    :param max_receives: Decisions per environment (``None``: play the rally out).
    """

    def __init__(
        self,
        opponent: Opponent,
        state: Tensor,
        *,
        t0: float,
        end: Tensor,
        profile: Tensor,
        stance: Tensor,
        hitter: Tensor,
        t_hit: Tensor,
        pelvis: Tensor | None = None,
        max_receives: int | None = None,
        dt: float = 5e-3,
        cadence: int = 8,
        compile: bool | None = None,
        cap_time: float = 60.0,
        cap_shots: int = 60,
        first_blade: BladePlan | None = None,
        seed: int = 0,
    ) -> None:
        self.opponent = opponent
        params: VecParams = opponent.params
        self.params = params
        n = params.n
        self.n = n
        dev, dtype = params.device, params.dtype
        self.dt, self.cadence = dt, cadence
        self.t = float(t0)
        self.ball = BallBatch(params, state, t0=t0, terminal_kinds=("floor",), graze=True, compile=compile)
        self.end, self.profile, self.stance = end.to(dtype), profile.long(), stance.to(dtype)
        self.pelvis = PelvisBatch(stance.reshape(-1, 2) if pelvis is None else pelvis.reshape(-1, 2))
        self.hitter = hitter.long().clone()
        self.t_hit = t_hit.to(dtype).clone()
        self.phase = torch.full((n,), AWAIT, dtype=torch.long, device=dev)
        self.t_due = torch.maximum(self.t_hit + opponent.rules.limits["arm_latency_s"], torch.full_like(self.t_hit, t0))
        self.max_receives = max_receives
        self.receives = torch.zeros(n, dtype=torch.long, device=dev)
        self.shots = torch.zeros(n, dtype=torch.long, device=dev)
        self.static_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.finished = torch.zeros(n, dtype=torch.bool, device=dev)
        self.over_reason = torch.full((n,), -1, dtype=torch.long, device=dev)
        """Why the rally ended: a status code (1 stopped, 2 unreachable, 3 let go), 4 missed, 5 cap, -1 still on."""
        self.capped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cap_time, self.cap_shots = cap_time, cap_shots
        self.seen = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t_b = torch.full((n,), float("nan"), dtype=dtype, device=dev)
        self.after_b = torch.zeros((n, 9), dtype=dtype, device=dev)
        self.ev_t = torch.full((n, 2), -float("inf"), dtype=dtype, device=dev)
        self.ev_after = state.to(dtype).reshape(n, 1, 9).repeat(1, 2, 1).clone()
        self.ev_t[:, 1] = t0
        self.deadline = torch.full((n,), float("inf"), dtype=dtype, device=dev)
        self.plan: Plan = opponent._empty_plan(
            torch.arange(n, device=dev),
            torch.zeros(n, dtype=torch.long, device=dev),
            torch.ones(n, dtype=dtype, device=dev),
            self.t_hit.clone(),
            self.t_due.clone(),
            torch.zeros(n, dtype=dtype, device=dev),
            torch.zeros((n, 9), dtype=dtype, device=dev),
            torch.zeros(n, dtype=torch.bool, device=dev),
            torch.zeros(n, dtype=dtype, device=dev),
            torch.full((n,), -1, dtype=torch.long, device=dev),
        )
        self.commands = [idle_command(n, dtype, dev), idle_command(n, dtype, dev)]
        if opponent.body is not None:  # body-first: the contact tick's states (NaN: the constant twist)
            for c in self.commands:
                c.tick0 = (
                    torch.full((n,), float("nan"), dtype=dtype, device=dev),
                    torch.zeros((n, 3), dtype=dtype, device=dev),
                    torch.zeros((n, 3), dtype=dtype, device=dev),
                )
                c.tick1 = (
                    torch.full((n,), float("nan"), dtype=dtype, device=dev),
                    torch.zeros((n, 3), dtype=dtype, device=dev),
                    torch.zeros((n, 3), dtype=dtype, device=dev),
                )
        self.parks = [parked_pose(self.end[:, k], params.table_half_length, params.table_height) for k in range(2)]
        self.log = ReceiveLog()
        # Each player's blade plan (paddle-first): at rest at a ready pose, then each stroke's follow-through and
        # recovery (the generator's ``PaddleTrack`` after ``schedule``), for the next stroke's check.
        flat_profile = self.profile.reshape(-1)
        flat_stance = self.stance.reshape(-1, 2)
        flat_flip = self.end.reshape(-1) > 0.0
        env2 = torch.arange(n, device=dev).repeat_interleave(2)
        half = (params.table_half_length[env2], params.table_half_width[env2])
        self.blade_rest = opponent.strokes.draw_ready(flat_profile, flat_stance, flat_flip, -params.table_height[env2], half, opponent.generator)
        zero = torch.zeros(2 * n, dtype=dtype, device=dev)
        q = torch.zeros((2 * n, 4), dtype=dtype, device=dev)
        q[:, 0] = 1.0
        self.follow = Piece(
            zero.clone(), zero.clone(), torch.zeros((2 * n, 6, 3), dtype=dtype, device=dev), torch.zeros((2 * n, 6, 3), dtype=dtype, device=dev), q.clone()
        )
        self.recover = Piece(
            zero.clone(), zero.clone(), torch.zeros((2 * n, 6, 3), dtype=dtype, device=dev), torch.zeros((2 * n, 6, 3), dtype=dtype, device=dev), q.clone()
        )
        self.has_stroke = torch.zeros(2 * n, dtype=torch.bool, device=dev)
        self.first_blade = first_blade
        # Body-first: each player's hand before its first stroke is its body's start pose; after a stroke, see
        # ``_hand_at``. The time of each player's last executed contact.
        self.body_mode = opponent.body is not None
        self.last_contact = torch.full((2 * n,), -float("inf"), dtype=dtype, device=dev)
        self.hand_start = opponent.body.start_hand(flat_profile, flat_stance, flat_flip, opponent.generator) if opponent.body is not None else None
        self.steps = 0
        self.decide_seconds = 0.0
        self._log_seen = 0

    # ---- ball state between events -----------------------------------------------------------------

    def ball_at(self, rows: Tensor, t: Tensor) -> Tensor:
        """The ball at ``t`` (no later than now): re-propagated from the last event at or before ``t`` (its after-state)."""

        use_last = self.ev_t[rows, 1] <= t
        t_ev = torch.where(use_last, self.ev_t[rows, 1], self.ev_t[rows, 0])
        after = torch.where(use_last[:, None], self.ev_after[rows, 1], self.ev_after[rows, 0])
        return propagate(self.params, after, rows, t - t_ev)

    # ---- the loop --------------------------------------------------------------------------------

    def run(self, max_time: float) -> None:
        """Step until every environment finished or ``max_time`` (absolute)."""

        while self.t < max_time - 1e-12:
            self.step()
            if bool(self.finished.all()):
                break

    def step(self) -> None:
        track = blade_track(self.commands, self.t, self.dt, self.parks)
        self.ball.step(self.dt, track)
        self.t += self.dt
        self.steps += 1
        self._events()
        inactive = ~self.ball.active
        self.finished |= inactive
        over = self.phase == OVER
        self.finished |= over & (self.static_count >= 2)
        cap = ~self.finished & ((self.t >= self.cap_time) | (self.shots >= self.cap_shots))
        if bool(cap.any()):
            self.capped |= cap
            self.finished |= cap
        missed = (self.phase == EXECUTED) & (self.t > self.deadline)
        if bool(missed.any()):
            self.phase = torch.where(missed, torch.full_like(self.phase, OVER), self.phase)
            self.over_reason = torch.where(missed, torch.full_like(self.over_reason, 4), self.over_reason)
        self.ball.active &= ~self.finished
        if self.steps % self.cadence == 0:
            self._process()
        else:
            due = (
                ((self.phase == REFINE_DUE) & (self.t_due <= self.t))
                | (self.phase == FINALIZE_DUE)
                | ((self.phase == PRE_BOUNCE) & (self.t >= self.plan.first.time - 0.03))
            )
            if bool((due & (self.plan.first.time - self.t < 0.06 + self.dt)).any()):
                self._process(decisions=False)

    def _events(self) -> None:
        log = self.ball.log
        assert log is not None
        chunks = range(self._log_seen, len(log.env))
        self._log_seen = len(log.env)
        for i in chunks:
            env, time, surface = log.env[i], log.time[i], log.surface[i]
            point, after = log.point[i], log.after[i]
            self.ev_t[env, 0] = self.ev_t[env, 1]
            self.ev_after[env, 0] = self.ev_after[env, 1]
            self.ev_t[env, 1] = time
            self.ev_after[env, 1] = after
            paddle = surface >= STATIC_COUNT
            player = (surface - STATIC_COUNT).clamp(min=0)
            receiver = 1 - self.hitter[env]
            fresh = paddle & ((player == receiver) | (time - self.t_hit[env] > 0.0625))
            if bool(fresh.any()):
                e = env[fresh]
                self.hitter[e] = player[fresh]
                self.t_hit[e] = time[fresh]
                self.shots[e] += 1
                self.static_count[e] = 0
                self.seen[e] = False
                for k in range(2):
                    self.commands[k].active[e] = False
                more = torch.ones_like(e, dtype=torch.bool) if self.max_receives is None else self.receives[e] < self.max_receives
                live = more & (self.phase[e] != OVER)
                self.phase[e] = torch.where(live, torch.full_like(e, AWAIT), torch.full_like(e, OVER))
                self.t_due[e] = time[fresh] + self.opponent.rules.limits["arm_latency_s"]
                self.deadline[e] = float("inf")
            static = ~paddle
            if bool(static.any()):
                e = env[static]
                self.static_count[e] += 1
                table = (surface[static] == 0) | (surface[static] == 3)
                receiver_end = self.end[e, (1 - self.hitter[e])]
                own = table & (torch.sign(point[static, 0]) == receiver_end) & ~self.seen[e] & (time[static] > self.t_hit[e])
                if bool(own.any()):
                    eb = e[own]
                    self.seen[eb] = True
                    self.t_b[eb] = time[static][own]
                    self.after_b[eb] = after[static][own]
                    waiting = self.phase[eb] == PRE_BOUNCE
                    if bool(waiting.any()):
                        ew = eb[waiting]
                        first_t = self.plan.first.time[ew]
                        t_late = self.t_b[ew] + self.opponent.rules.limits["late_latency_s"]
                        tick_start = torch.floor(first_t / TICK_S + 1e-9) * TICK_S
                        refine = (self.t_b[ew] <= first_t - 0.03) & (t_late < tick_start - 1e-6)
                        self.phase[ew] = torch.where(refine, torch.full_like(ew, REFINE_DUE), torch.full_like(ew, FINALIZE_DUE))
                        self.t_due[ew] = torch.where(refine, t_late, self.t_b[ew])

    def _process(self, decisions: bool = True) -> None:
        import time as _time

        start = _time.time()
        now = torch.full((self.n,), self.t, dtype=self.params.dtype, device=self.params.device)
        opp = self.opponent
        # Decisions.
        due = torch.nonzero((self.phase == AWAIT) & (self.t_due <= now) & ~self.finished).flatten() if decisions else now[:0].long()
        if due.numel():
            self._decide(due)
        # A planned receiver that saw no bounce by its contact - 0.03 s plays without the late correction.
        late = torch.nonzero((self.phase == PRE_BOUNCE) & (now >= self.plan.first.time - 0.03) & ~self.finished).flatten()
        if late.numel():
            self.phase[late] = FINALIZE_DUE
            self.t_due[late] = now[late]
        refine = torch.nonzero((self.phase == REFINE_DUE) & (self.t_due <= now) & ~self.finished).flatten()
        if refine.numel():
            t_late = self.t_due[refine]
            ball = self.ball_at(refine, t_late)
            t_contact = self.t_b[refine] + self.plan.draw_after_bounce_s[refine]
            k = 1 - self.hitter[refine]
            pelvis, _ = self.pelvis.at(refine * 2 + k, t_contact)
            opp.refine(self.plan, refine, t_late, ball, self.t_b[refine], pelvis)
            self.phase[refine] = FINALIZE_DUE
        final = torch.nonzero((self.phase == FINALIZE_DUE) & ~self.finished).flatten()
        if final.numel():
            self._finalize(final)
        self.decide_seconds += _time.time() - start

    def _decide(self, rows: Tensor) -> None:
        opp = self.opponent
        k = 1 - self.hitter[rows]
        t_decide = self.t_due[rows]
        ball = self.ball_at(rows, t_decide)
        bounced = self.seen[rows] & (self.t_b[rows] <= t_decide)
        pelvis_now, _ = self.pelvis.at(rows * 2 + k, t_decide)
        slots = rows * 2 + k
        blade = self._blade_at(slots, (torch.floor(t_decide / TICK_S + 1e-9) + 1.0) * TICK_S)
        if self.first_blade is not None:
            first = self.receives[rows] == 0
            if bool(first.any()):
                fb = self.first_blade.take(rows)
                for name in ("p", "v", "a", "q", "w"):
                    getattr(blade, name)[first] = getattr(fb, name)[first]
        pelvis = self.pelvis

        def track(sub: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
            r_rows = slots[sub].repeat_interleave(times.shape[1])
            p, v = pelvis.at(r_rows, times.reshape(-1))
            return p.reshape((*times.shape, 2)), v.reshape((*times.shape, 2))

        plan = opp.decide(
            rows,
            self.profile[rows, k],
            self.end[rows, k],
            t_decide,
            self.t_hit[rows],
            ball,
            self.stance[rows, k],
            pelvis_now,
            bounced,
            self.t_b[rows],
            self.after_b[rows],
            blade,
            track,
        )
        self.receives[rows] += 1
        self._store(rows, plan)
        attempt = plan.status == ATTEMPT
        a_rows = rows[attempt]
        if a_rows.numel():
            ka = k[attempt]
            approach_start = torch.maximum(t_decide[attempt], self.t_hit[a_rows] + opp.rules.limits["footwork_latency_s"])
            self.pelvis.move(a_rows * 2 + ka, approach_start, plan.first.time[attempt] - 0.05, plan.pelvis_needed[attempt])
            idx = torch.nonzero(attempt).flatten()
            self._set_command(a_rows, ka, (plan.held if self.body_mode else plan.first).take(idx))
            if plan.follow is not None and plan.recover is not None and plan.ready is not None:
                slot = a_rows * 2 + ka
                for name in ("start", "end", "position", "rotation", "reference"):
                    getattr(self.follow, name)[slot] = getattr(plan.follow, name)[idx]
                    getattr(self.recover, name)[slot] = getattr(plan.recover, name)[idx]
                self.blade_rest.p[slot] = plan.ready.p[idx]
                self.blade_rest.q[slot] = plan.ready.q[idx]
                self.has_stroke[slot] = True
            pre = plan.pre_bounce[attempt]
            self.phase[a_rows] = torch.where(pre, torch.full_like(a_rows, PRE_BOUNCE), torch.full_like(a_rows, FINALIZE_DUE))
        no = rows[~attempt]
        if no.numel():
            self.phase[no] = OVER
            self.over_reason[no] = plan.status[~attempt]
        self.log.add(
            env=rows,
            receiver=k,
            profile=self.profile[rows, k],
            status=plan.status,
            t_hit=self.t_hit[rows],
            t_decide=t_decide,
            t_bounce=plan.t_bounce,
            pre_bounce=plan.pre_bounce,
            p_stop=plan.p_stop,
            plan_t=plan.first.time,
            plan_tau=plan.first.time - plan.t_bounce,
            draw_tau=plan.draw_after_bounce_s,
            paddle_speed=torch.sqrt((plan.first.face_velocity**2).sum(-1)),
            face=(plan.held if self.body_mode else plan.first).face,
            plausibility=plan.plausibility,
            draws_feasible=plan.draws_feasible,
            ood=plan.draw_ood,
            contact=plan.first.ball_in[:, 0:3],
            pelvis=plan.pelvis_needed,
            pelvis_now=pelvis_now,
            bounce=plan.bounce,
        )

    def _blade_at(self, slots: Tensor, t: Tensor) -> BladePlan:
        """
        Each player's blade plan state at ``t``: in its last follow-through or recovery, else held at its ready pose
        (paddle-first); body-first, its hand (:meth:`_hand_at`).
        """

        if self.body_mode:
            return self._hand_at(slots, t)
        rest = self.blade_rest.take(slots)
        out = BladePlan(rest.p.clone(), torch.zeros_like(rest.p), torch.zeros_like(rest.p), rest.q.clone(), torch.zeros_like(rest.p))
        has = self.has_stroke[slots]
        follow, recover = self.follow.take(slots), self.recover.take(slots)
        in_follow = has & (t >= follow.start) & (t <= follow.end)
        in_recover = has & ~in_follow & (t >= recover.start) & (t <= recover.end)
        for mask, piece in ((in_follow, follow), (in_recover, recover)):
            if bool(mask.any()):
                st = piece.state(t)
                for name in ("p", "v", "a", "q", "w"):
                    getattr(out, name)[mask] = getattr(st, name)[mask]
        return out

    def _hand_at(self, slots: Tensor, t: Tensor) -> BladePlan:
        """
        Body-first: the hand at ``t``. Before the player's first stroke, its body's start pose (at rest); after a stroke,
        the hand as the runtime's bodies carry it at that time since their contact (``BodyFeasibility.rally_hand``).
        """

        opp = self.opponent
        assert opp.body is not None and self.hand_start is not None
        n2 = slots.shape[0]
        env = torch.div(slots, 2, rounding_mode="floor")
        k = slots - 2 * env
        p = self.hand_start[slots].clone()
        v = torch.zeros_like(p)
        played = torch.isfinite(self.last_contact[slots])
        if bool(played.any()):
            hips, _ = self.pelvis.at(slots, t)
            rp, rv = opp.body.rally_hand(self.profile[env, k], hips, self.end[env, k] > 0.0, t - self.last_contact[slots], opp.generator)
            p = torch.where(played[:, None], rp, p)
            v = torch.where(played[:, None], rv, v)
        q = torch.zeros((n2, 4), dtype=p.dtype, device=p.device)
        q[:, 0] = 1.0
        return BladePlan(p, v, torch.zeros_like(p), q, torch.zeros_like(p))

    def _store(self, rows: Tensor, plan: Plan) -> None:
        from dataclasses import fields as dc_fields

        from .strike import Strike

        for f in dc_fields(self.plan):
            source = getattr(plan, f.name)
            target = getattr(self.plan, f.name)
            if source is None or f.name in ("follow", "recover", "ready"):
                continue
            if isinstance(source, Strike):
                for g in dc_fields(source):
                    getattr(target, g.name)[rows] = getattr(source, g.name)
            elif f.name == "env":
                continue
            else:
                target[rows] = source

    def _set_command(self, rows: Tensor, player: Tensor, strike: Any) -> None:
        for k in range(2):
            sel = player == k
            if not bool(sel.any()):
                continue
            r = rows[sel]
            c = self.commands[k]
            c.t[r] = strike.time[sel]
            c.position[r] = strike.position[sel]
            c.orientation[r] = strike.orientation[sel]
            c.velocity[r] = strike.velocity[sel]
            c.angular_velocity[r] = strike.angular_velocity[sel]
            c.face[r] = strike.face[sel]
            c.active[r] = True
            if c.tick0 is not None and c.tick1 is not None:
                c.tick0[0][r] = float("nan")
                c.tick1[0][r] = float("nan")

    def _finalize(self, rows: Tensor) -> None:
        opp = self.opponent
        k = 1 - self.hitter[rows]
        command: ContactCommand = opp.execute(self.plan, rows, torch.full_like(self.t_hit[rows], self.t))
        self.last_contact[rows * 2 + k] = torch.where(command.active, command.t, self.last_contact[rows * 2 + k])
        for p in range(2):
            sel = k == p
            if not bool(sel.any()):
                continue
            r = rows[sel]
            c = self.commands[p]
            c.t[r] = command.t[sel]
            c.position[r] = command.position[sel]
            c.orientation[r] = command.orientation[sel]
            c.velocity[r] = command.velocity[sel]
            c.angular_velocity[r] = command.angular_velocity[sel]
            c.face[r] = command.face[sel]
            c.active[r] = command.active[sel]
            if c.tick0 is not None and c.tick1 is not None and command.tick0 is not None and command.tick1 is not None:
                for mine, theirs in ((c.tick0, command.tick0), (c.tick1, command.tick1)):
                    for target, source in zip(mine, theirs, strict=True):
                        target[r] = source[sel]
        ret = opp.rules.constants["pelvis_return_s"]
        stance = self.stance[rows, k]
        self.pelvis.move(rows * 2 + k, command.t + ret[0], command.t + ret[1], stance)
        self.phase[rows] = EXECUTED
        self.deadline[rows] = command.t + opp.rules.constants["contact_wait_s"]
        final = self.plan.final.take(rows)
        point = final.ball_in[:, 0:3] - self.params.radius[rows][:, None] * final.normal
        face_velocity = command.velocity + torch.cross(command.angular_velocity, point - command.position, dim=-1)
        self.log.add(
            env=rows,
            executed_t=command.t,
            noise=self.plan.noise[rows],
            refined=self.plan.refined[rows],
            exec_speed=torch.sqrt((face_velocity**2).sum(-1)),
            noise_applied=self.plan.noise_applied[rows],
            exec_face=command.face,
        )

    def receive_table(self) -> dict[str, np.ndarray]:
        """The decisions, one row per receive (``ReceiveLog``; executions are separate rows with ``executed_t``)."""

        decided = [p for p in self.log.parts if "status" in p]
        executed = [p for p in self.log.parts if "executed_t" in p]
        out = {k: torch.cat([p[k] for p in decided]).cpu().numpy() for k in decided[0]} if decided else {}
        if executed:
            out["exec_env"] = torch.cat([p["env"] for p in executed]).cpu().numpy()
            out["exec_t"] = torch.cat([p["executed_t"] for p in executed]).cpu().numpy()
            out["exec_noise"] = torch.cat([p["noise"] for p in executed]).cpu().numpy()
            out["exec_refined"] = torch.cat([p["refined"] for p in executed]).cpu().numpy()
            out["exec_speed"] = torch.cat([p["exec_speed"] for p in executed]).cpu().numpy()
            out["exec_noise_applied"] = torch.cat([p["noise_applied"] for p in executed]).cpu().numpy()
            out["exec_face"] = torch.cat([p["exec_face"] for p in executed]).cpu().numpy()
        return out


def table_frame_stance(local: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Stances drawn in the hitter frame to the table frame for players at ``end``."""

    return np.where((end > 0)[:, None], -local, local)


def yaw_stance(stance_local: Tensor, end: Tensor) -> Tensor:
    return yaw_xy(stance_local, end > 0.0)


@dataclass
class Setup:
    """The start of ``N`` episodes: the ball, who hit it and when, ends, profiles, stances."""

    state: np.ndarray
    t0: float
    end: np.ndarray
    profile: np.ndarray
    stance: np.ndarray
    hitter: np.ndarray
    t_hit: np.ndarray
    feed: bool = True
    pelvis: np.ndarray | None = None

    def tensors(self, device: Any, dtype: Any) -> dict[str, Tensor]:
        def t(x: np.ndarray, kind: Any = None) -> Tensor:
            return torch.as_tensor(x, dtype=kind or dtype, device=device)

        out = {
            "state": t(self.state),
            "end": t(self.end),
            "profile": t(self.profile, torch.long),
            "stance": t(self.stance),
            "hitter": t(self.hitter, torch.long),
            "t_hit": t(self.t_hit),
        }
        if self.pelvis is not None:
            out["pelvis"] = t(self.pelvis)
        return out


def feed_setup(rules: Any, n: int, rng: np.random.Generator, world: Any) -> Setup:
    """
    ``n`` feed-started episodes as ``generate.build_rally`` draws them: the robot's end, both stances, the feeder (the
    suited player with the recorded share), a recorded first shot with its jitter, and when the feeder hit it (traced
    back under the world's flight to the feeder's end line, at most 0.6 s).
    """

    from lib.physics.reference.flight import FlightModel, rk4_step

    robot_end = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    end = np.column_stack([robot_end, -robot_end])
    profile = np.tile(np.array([0, 1]), (n, 1))
    stance = np.zeros((n, 2, 2))
    for k in range(2):
        local = rules.draw_stance(np.full(n, k), rng)
        stance[:, k] = table_frame_stance(local, end[:, k])
    feeder = np.where(rng.random(n) < rules.header["feed_profile_share"], 0, 1)
    _index, local_feed = rules.draw_feed(feeder, rng)
    flip = end[np.arange(n), feeder] > 0
    state = local_feed.copy()
    for start in (0, 3, 6):
        state[flip, start : start + 2] *= -1.0
    model = FlightModel.from_params(world)
    t_hit = np.zeros(n)
    for i in range(n):
        y = [float(v) for v in state[i]]
        t = 0.0
        sign = -1.0 if flip[i] else 1.0
        while t > -0.6:
            y = rk4_step(model, y, -1e-3 * 10)
            t -= 0.01
            if sign * y[0] <= -1.40 or y[2] < world.ball.radius:
                break
        t_hit[i] = t
    return Setup(state, 0.0, end, profile, stance, feeder, t_hit, True)


__all__ = ["AWAIT", "EXECUTED", "OVER", "PelvisBatch", "RallySim", "ReceiveLog", "Setup", "feed_setup", "table_frame_stance"]
