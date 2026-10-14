"""
The distilled human opponent: the generator's receiver (``Rally.plan_return``, ``refine``, ``execute`` of the
offline data generator) over many environments at once, in torch.

A decision, per receiving environment, at the other player's hit plus the cited orientation latency (0.15 s):

1. perceive the ball (perception noise at the noise scale) and predict its own-half bounce under the row's physics
   (or take the bounce already seen);
2. the fitted stop hazard (``hazard_scale`` times the 4-feature logistic; 0 in competitive points) may stop the rally;
3. draw ``D`` shot intents (h2, :mod:`.intent`, the only learned part);
4. per intent, the **contact phase** by rule: along the predicted path every 10 ms from 0.06 to 0.70 s after the
   bounce, before the ball comes down again, at least 0.12 s after the decision, with the hips moved from where they
   are within the footwork envelope (0.30 s latency, 4 m/s^2, 1.8 m/s) and no further forward than the table's edge,
   the arm inside the recorded blade-offset envelope and the capture's body-reach model; the time nearest the
   intent's, plus the footwork shortfall at 1 s per metre;
5. the launch to the intended landing and flight time, the face normal for the intended spin (two passes), the
   rigid-blade strike with the donor's grip roll;
6. the stroke's feasibility: body-first (the v1 release, the default), the body runtime's verdict and the contact it
   holds (:mod:`.body`: the donor reach rule, the fitted acceptance and nudge, the same-outcome rule); paddle-first,
   the motion check against the capture's limits (:mod:`.strokes`);
7. the first intent that is feasible and plausible (the 891-state density), else the most plausible feasible one,
   else ``unreachable``.

After the real bounce plus 0.11 s, :meth:`Opponent.refine` replans within the cited late-correction caps; then
:meth:`Opponent.execute` applies the generator's execution noise (body-first, as the body re-holds it, with the
generator's fallbacks to part of the noise or the first plan) and returns the kinematic contact command
(:class:`lib.opponent.strike.ContactCommand`) that :mod:`.blade` flies through
:class:`lib.physics.vectorized.BallBatch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.params import VecParams

from .ballistics import fly, path_grid, predict_bounce, propagate, solve_launch
from .body import BodyFeasibility, Contact, classify_landing
from .intent import IntentDraws, KnnIntent, MdnIntent
from .ops import (
    cross,
    dot,
    frame_exp,
    norm,
    quat_conjugate,
    quat_from_rotvec,
    quat_multiply,
    quat_to_matrix,
    quat_to_rotvec,
    reference_from_game,
    rotation_from_rotvec,
    spin_frame,
    strike_frame,
    yaw_states,
    yaw_vectors,
    yaw_xy,
)
from .rules import RuleBook
from .strike import ContactCommand, Strike, execute, forward, normal_for_spin, plan_strike
from .strokes import BladePlan, Piece, StrokeCheck

TICK_S = 1.0 / 24.0

Tensor = Any

ATTEMPT, STOP, UNREACHABLE, LET_GO = 0, 1, 2, 3
STATUS_NAMES = ("attempt", "stopped", "unreachable", "let_go")


@dataclass
class OpponentConfig:
    """
    The opponent's knobs. ``None`` takes the production fit the rules carry (``noise_v1.json``; v2: noise 0.7, hazard 0.8).

    :param hazard_scale: Multiplies the stop hazard (0 in competitive points: the opponent never gives up a ball).
    :param noise_scale: Multiplies the execution and perception noise anchors (the skill knob).
    :param perception: Perception noise on.
    :param plausibility: Prefer plausible paddle states (the generator's rule).
    :param draws: Intents drawn per decision (the generator's redraw budget, 5).
    """

    hazard_scale: float | None = None
    noise_scale: float | None = None
    perception: bool = True
    plausibility: bool = True
    draws: int | None = None
    paddle_speed_scale: float = 1.0
    stroke_check: bool = True
    """Hold each stroke to the target mode's feasibility (``False``: none, a diagnostic)."""
    feasibility: str | None = None
    """``body`` (the v1 release's body-first runtime, distilled: :mod:`.body`), ``paddle`` (the paddle-first motion check,
    :mod:`.strokes`) or ``none``; ``None`` takes ``body`` when the rules carry it."""
    profiles: tuple[int, ...] = (0, 1)
    """The profiles that will decide (a static hint for the body-first reach rule: an environment whose opponent is
    always profile 1 passes ``(1,)`` and halves its cost)."""


@dataclass
class Plan:
    """One decision per row (all tensors ``(M, ...)``)."""

    status: Tensor
    env: Tensor
    profile: Tensor
    end: Tensor
    t_hit: Tensor
    t_decide: Tensor
    t_bounce: Tensor
    bounce: Tensor
    pre_bounce: Tensor
    p_stop: Tensor
    draw_after_bounce_s: Tensor
    draw_spin: Tensor
    draw_face: Tensor
    draw_tip: Tensor
    draw_ood: Tensor
    draw_donor: Tensor
    draws_feasible: Tensor
    first: Strike
    final: Strike
    held: Strike
    """Body-first: the contact the body holds for ``first`` (the runtime's nudge); paddle-first: ``first``."""
    pelvis_needed: Tensor
    intended_landing: Tensor
    intended_flight_s: Tensor
    plausibility: Tensor
    refined: Tensor
    noise: Tensor = field(default=None)
    noise_applied: Tensor = field(default=None)
    """The share of the execution noise played (1, 0.5, 0.25, 0; -1 the first plan as held, ``execute_kept_first``)."""
    follow: Piece | None = None
    recover: Piece | None = None
    ready: BladePlan | None = None

    def take(self, index: Tensor) -> Plan:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = None if v is None else (v.take(index) if isinstance(v, (Strike, Piece, BladePlan)) else v[index])
        return Plan(**out)

    @property
    def swing_start(self) -> Tensor:
        """When the forward swing starts: 0.2 s before the first plan's contact (the captured backswing)."""

        return self.first.time - 0.2


def _empty_strike(m: int, dtype: Any, device: Any) -> Strike:
    def z(*shape: int) -> Tensor:
        return torch.full((m, *shape), float("nan"), dtype=dtype, device=device)

    return Strike(
        time=z(),
        ball_in=z(9),
        position=z(3),
        orientation=z(4),
        rotation=z(3, 3),
        velocity=z(3),
        angular_velocity=z(3),
        normal=z(3),
        face=torch.zeros(m, dtype=torch.long, device=device),
        v_out=z(3),
        spin_out=z(3),
        face_velocity=z(3),
        ok=torch.zeros(m, dtype=torch.bool, device=device),
    )


def _scatter_strike(target: Strike, index: Tensor, source: Strike) -> None:
    for f in fields(target):
        getattr(target, f.name)[index] = getattr(source, f.name)


def _capped(v: Tensor, limit: float) -> Tensor:
    n = norm(v)
    return torch.where((n > limit)[:, None], v * (limit / n.clamp_min(1e-300))[:, None], v)


class Opponent:
    """
    The distilled human receiver over a batch of environments.

    :param params: The environments' physics (the world; the players' belief equals it, as in v1 data).
    :param intent: The h2 model (:class:`MdnIntent`, or :class:`KnnIntent` for the generator's exact draw).
    :param rules: The generator's rules.
    :param config: The knobs.
    :param generator: A ``torch.Generator`` for reproducible draws.
    """

    def __init__(self, params: VecParams, intent: KnnIntent | MdnIntent, rules: RuleBook, config: OpponentConfig | None = None, generator: Any = None) -> None:
        self.params = params
        self.intent = intent
        self.rules = rules
        self.config = config or OpponentConfig()
        self.generator = generator
        fitted = rules.fitted
        self.hazard_scale = fitted["hazard_scale"] if self.config.hazard_scale is None else self.config.hazard_scale
        self.noise_scale = fitted["noise_scale"] if self.config.noise_scale is None else self.config.noise_scale
        self.draws = int(rules.constants["draw_attempts"] if self.config.draws is None else self.config.draws)
        c = rules.constants
        self.noise_sd = torch.tensor(
            [c["noise_anchors"][k] for k in ("timing_s", "direction_rad", "speed_frac", "position_m")], dtype=params.dtype, device=params.device
        )
        self.perception_sd = torch.tensor(
            [c["perception_anchors"]["position_m"]] * 3 + [c["perception_anchors"]["velocity_mps"]] * 3 + [c["perception_anchors"]["spin_radps"]] * 3,
            dtype=params.dtype,
            device=params.device,
        )
        self.counters: dict[str, int] = {}
        self.strokes = StrokeCheck(rules)
        mode = self.config.feasibility or ("body" if "body" in rules.header else "paddle")
        if not self.config.stroke_check and self.config.feasibility is None:
            mode = "none"
        if mode not in ("body", "paddle", "none"):
            raise ValueError(f"feasibility {mode!r}: body, paddle or none")
        self.feasibility = mode
        self.body = BodyFeasibility(rules.arrays, rules.header["body"], params.device, params.dtype) if mode == "body" else None
        self._dense = False
        """Inside :meth:`decide_dense`: masks instead of row compaction, fixed step counts, no host synchronisation."""
        self._ood = intent if isinstance(intent, KnnIntent) else KnnIntent(rules)

    @staticmethod
    def compile_status() -> dict[str, Any]:
        """Whether the hot bodies run compiled or fell back to eager (:func:`.ballistics.compile_status`): log it with results."""

        from .ballistics import compile_status

        return compile_status()

    # ---- perception -------------------------------------------------------------------------------

    def perceive(self, ball: Tensor) -> Tensor:
        if not self.config.perception or self.noise_scale <= 0.0:
            return ball.clone()
        z = torch.randn(ball.shape, dtype=ball.dtype, device=ball.device, generator=self.generator)
        return ball + z * self.perception_sd * self.noise_scale

    def _count(self, name: str, value: Tensor | int) -> None:
        if self._dense:
            return
        self.counters[name] = self.counters.get(name, 0) + int(value)

    # ---- the decision ----------------------------------------------------------------------------

    def decide(
        self,
        env: Tensor,
        profile: Tensor,
        end: Tensor,
        t_decide: Tensor,
        t_hit: Tensor,
        ball: Tensor,
        stance: Tensor,
        pelvis_now: Tensor,
        bounced: Tensor | None = None,
        bounce_t: Tensor | None = None,
        bounce_state: Tensor | None = None,
        blade: BladePlan | None = None,
        pelvis_track: Any = None,
        hazard_scale: Tensor | None = None,
    ) -> Plan:
        """
        Decide for ``M`` receivers.

        :param env: Environment of each row (into the physics).
        :param profile: 0 suited, 1 opponent (whose fitted rules and intent to use).
        :param end: The receiver end's x-sign.
        :param t_decide: The decision time (the other player's hit + 0.15 s, or later), s.
        :param t_hit: The other player's hit, s.
        :param ball: The true ball at ``t_decide`` ``(M, 9)``.
        :param stance: The receiver's ready stance (table frame).
        :param pelvis_now: The receiver's hips at ``t_decide`` (table frame).
        :param bounced: The ball has already bounced on the receiver's half (with ``bounce_t``, ``bounce_state``).
        :param blade: The receiver's blade plan state at the tick after the decision (default: at rest at its ready pose).
        :param pelvis_track: ``(rows, times (R, S)) -> (positions, velocities)`` of the receiver's hips plan before
            this stroke (default: at rest at ``pelvis_now``).
        :param hazard_scale: Per-row stop-hazard scale (default the config's): 0 for competitive points, the fit (0.6)
            for cooperative play.
        """

        params, rules = self.params, self.rules
        m = env.shape[0]
        dtype, device = params.dtype, params.device
        flip = end > 0.0
        perceived = self.perceive(ball)
        seen = torch.zeros(m, dtype=torch.bool, device=device) if bounced is None else bounced
        seen_t = torch.zeros(m, dtype=dtype, device=device) if bounce_t is None or bounced is None else bounce_t
        seen_state = torch.zeros((m, 9), dtype=dtype, device=device) if bounce_state is None or bounced is None else bounce_state
        predicted = predict_bounce(params, perceived, env, end)
        t_bounce = torch.where(seen, seen_t, t_decide + predicted.t)
        bounce = torch.where(seen[:, None], seen_state + (perceived - ball), predicted.after)
        known = seen | predicted.ok
        status = torch.full((m,), LET_GO, dtype=torch.long, device=device)
        local = yaw_states(bounce, flip)
        stance_local = yaw_xy(stance, flip)
        speed_after = torch.sqrt(local[:, 3] ** 2 + local[:, 4] ** 2)
        scale = torch.full((m,), float(self.hazard_scale), dtype=dtype, device=device) if hazard_scale is None else hazard_scale.to(dtype)
        p_stop = scale * rules.stop_probability(profile, local[:, 0:2], t_bounce - t_hit, speed_after, stance_local)
        u = torch.rand(m, dtype=dtype, device=device, generator=self.generator)
        stop = known & (u < p_stop)
        status = torch.where(stop, torch.full_like(status, STOP), status)
        plan = self._empty_plan(env, profile, end, t_hit, t_decide, t_bounce, bounce, torch.logical_not(seen), p_stop, status)
        rows = torch.nonzero(known & ~stop).flatten()
        self._count("decisions", m)
        self._count("let_go", (~known).sum())
        self._count("stopped", stop.sum())
        if rows.numel() == 0:
            return plan
        query = torch.stack([local[rows, 0], local[rows, 1], local[rows, 3], local[rows, 4], local[rows, 5], stance_local[rows, 0], stance_local[rows, 1]], -1)
        # ``draw_ready``: the recorded ready pose this stroke recovers to (redrawn at each decision, as the generator does).
        floor_z = -params.table_height[env[rows]]
        half = (params.table_half_length[env[rows]], params.table_half_width[env[rows]])
        ready = self.strokes.draw_ready(profile[rows], stance[rows], flip[rows], floor_z, half, self.generator)
        begin = ready if blade is None else blade.take(rows)
        if pelvis_track is None:
            now_rows = pelvis_now[rows]

            def track(sub: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
                p = now_rows[sub][:, None, :].expand((*times.shape, 2))
                return p, torch.zeros_like(p)

        else:

            def track(sub: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
                return pelvis_track(rows[sub], times)

        draws = self.intent.draw(profile[rows], query, self.draws, generator=self.generator)
        if not isinstance(self.intent, KnnIntent):
            draws.ood = self._ood.out_of_distribution(profile[rows], query)[:, None].expand(rows.numel(), self.draws)
        self._count("ood", draws.ood[:, 0].sum())
        chosen = self._plan_draws(plan, rows, draws, pelvis_now[rows], begin, ready, track, stance[rows])
        status[rows] = torch.where(chosen, torch.full_like(rows, ATTEMPT), torch.full_like(rows, UNREACHABLE))
        plan.status = status
        self._count("unreachable", (~chosen).sum())
        return plan

    def _empty_plan(
        self,
        env: Tensor,
        profile: Tensor,
        end: Tensor,
        t_hit: Tensor,
        t_decide: Tensor,
        t_bounce: Tensor,
        bounce: Tensor,
        pre: Tensor,
        p_stop: Tensor,
        status: Tensor,
    ) -> Plan:
        m = env.shape[0]
        dtype, device = self.params.dtype, self.params.device

        def z(*shape: int) -> Tensor:
            return torch.full((m, *shape), float("nan"), dtype=dtype, device=device)

        return Plan(
            status=status,
            env=env,
            profile=profile,
            end=end,
            t_hit=t_hit,
            t_decide=t_decide,
            t_bounce=t_bounce,
            bounce=bounce,
            pre_bounce=pre,
            p_stop=p_stop,
            draw_after_bounce_s=z(),
            draw_spin=z(2),
            draw_face=torch.zeros(m, dtype=torch.long, device=device),
            draw_tip=z(3),
            draw_ood=torch.zeros(m, dtype=torch.bool, device=device),
            draw_donor=torch.full((m,), -1, dtype=torch.long, device=device),
            draws_feasible=torch.zeros(m, dtype=torch.long, device=device),
            first=_empty_strike(m, dtype, device),
            final=_empty_strike(m, dtype, device),
            held=_empty_strike(m, dtype, device),
            pelvis_needed=z(2),
            intended_landing=z(2),
            intended_flight_s=z(),
            plausibility=z(),
            refined=torch.zeros(m, dtype=torch.bool, device=device),
            noise=z(4),
            noise_applied=z(),
        )

    def _contact_phase(self, plan: Plan, rows: Tensor, draws: IntentDraws, pelvis_now: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """``Rally.contact_phase`` for every draw: ``(found (R, D), t (R, D), ball (R, D, 9), pelvis (R, D, 2), step index)``."""

        params, rules = self.params, self.rules
        c = rules.constants
        first, step, count = c["contact_grid_s"]
        env = plan.env[rows]
        profile = plan.profile[rows]
        flip = plan.end[rows] > 0.0
        r, d = rows.numel(), self.draws
        grid = path_grid(params, plan.bounce[rows], env, first, step, int(count))
        steps = first + step * torch.arange(int(count), dtype=grid.dtype, device=grid.device)
        times = plan.t_bounce[rows][:, None] + steps[None, :]
        z = grid[:, :, 2]
        half_length = params.table_half_length[env][:, None]
        radius = params.radius[env][:, None]
        low = (z < c["min_contact_height_m"]) | ((grid[:, :, 0].abs() < half_length) & (z < radius + 0.01))
        high = ~low
        any_high_before = torch.cumsum(high.long(), 1) - high.long() > 0
        breaks = low & any_high_before & (grid[:, :, 5] < 0.0)
        alive = torch.cumsum(breaks.long(), 1) == 0
        valid = high & alive & (times >= plan.t_decide[rows][:, None] + c["reaction_min_s"])
        # Per draw: where the hips must be for the donor's blade offset, as far as footwork allows.
        offset = yaw_xy(draws.offset, flip)
        ball_xy = grid[:, None, :, 0:2]
        limit = -(half_length[:, 0] + rules.hips_front[profile])
        flip_k = flip[:, None, None].expand(r, d, int(count))
        wanted = rules.behind_line(ball_xy - offset[:, :, None, :], flip_k, limit[:, None, None].expand(r, d, int(count)))
        now = pelvis_now[:, None, None, :]
        move = wanted - now
        reach = rules.footwork_reach(times - plan.t_hit[rows][:, None])[:, None, :]
        distance = norm(move)
        scaled = now + move * (reach / distance.clamp_min(1e-300))[..., None]
        pelvis = torch.where((distance <= reach)[..., None], wanted, rules.behind_line(scaled, flip_k, limit[:, None, None].expand(r, d, int(count))))
        within = norm(pelvis - now) <= reach + 1e-9
        arm = yaw_xy(ball_xy - pelvis, flip_k)
        profile_k = profile[:, None, None].expand(r, d, int(count))
        envelope = rules.envelope_ok(profile_k, arm)
        body = rules.arm_excess(profile_k, grid[:, None, :, 0:3].expand(r, d, int(count), 3), pelvis, flip_k, rules.blade_reach[profile][:, None, None]) <= 0.0
        ok = valid[:, None, :] & within & envelope & body
        cost = (steps[None, None, :] - draws.after_bounce_s[:, :, None]).abs() + norm(wanted - pelvis)
        cost = torch.where(ok, cost, torch.full_like(cost, float("inf")))
        best = torch.argmin(cost, -1)
        found = ok.any(-1)
        t = times[:, None, :].expand(r, d, int(count)).gather(2, best[..., None])[..., 0]
        ball = grid[:, None, :, :].expand(r, d, int(count), 9).gather(2, best[..., None, None].expand(r, d, 1, 9))[:, :, 0]
        pel = pelvis.gather(2, best[..., None, None].expand(r, d, 1, 2))[:, :, 0]
        return found, t, ball, pel, best

    def _plan_draws(
        self,
        plan: Plan,
        rows: Tensor,
        draws: IntentDraws,
        pelvis_now: Tensor,
        begin: BladePlan | None = None,
        ready: BladePlan | None = None,
        track: Any = None,
        stance: Tensor | None = None,
        allowed: Tensor | None = None,
    ) -> Tensor:
        """
        Make each draw concrete (``Rally.candidate``) and choose one per row; fills ``plan`` and returns ``chosen``.

        :param allowed: Dense mode: the rows that may attempt (the others are computed and masked out).
        """

        params, rules = self.params, self.rules
        c = rules.constants
        r, d = rows.numel(), self.draws
        found, t_contact, ball, pelvis, _k = self._contact_phase(plan, rows, draws, pelvis_now)
        flat = draws.flat()
        env = plan.env[rows].repeat_interleave(d)
        end = plan.end[rows].repeat_interleave(d)
        flip = end > 0.0
        ball = ball.reshape(-1, 9)
        t_contact = t_contact.reshape(-1)
        pelvis = pelvis.reshape(-1, 2)
        feasible = found.reshape(-1).clone()
        if allowed is not None:
            feasible = feasible & allowed.repeat_interleave(d)
        self._count("reject_reach", (~feasible).sum())
        dense = self._dense
        steps = 40 if dense else None
        live = torch.arange(r * d, device=feasible.device) if dense else torch.nonzero(feasible).flatten()
        dtype, device = params.dtype, params.device
        n_all = r * d
        strikes = _empty_strike(n_all, dtype, device)
        density = torch.full((n_all,), -float("inf"), dtype=dtype, device=device)
        landing = torch.full((n_all, 2), float("nan"), dtype=dtype, device=device)
        flight_t = torch.full((n_all,), float("nan"), dtype=dtype, device=device)
        follows = recovers = None
        helds: Strike | None = None
        if dense or live.numel():
            e, fl, b = env[live], flip[live], ball[live]
            margin = c["target_margin_m"]
            hl, hw = params.table_half_length[e], params.table_half_width[e]
            target_local = torch.stack(
                [
                    torch.minimum(torch.maximum(flat.landing[live, 0], margin + 0.05 + 0 * hl), hl - margin),
                    torch.minimum(torch.maximum(flat.landing[live, 1], -hw + margin), hw - margin),
                ],
                -1,
            )
            target = yaw_xy(target_local, fl)
            flight = flat.flight_s[live]
            guess_local = flat.guess[live]
            have = torch.isfinite(guess_local).all(-1)
            g = params.gravity[e]
            radius = params.radius[e]
            ballistic = torch.stack([(target[:, 0] - b[:, 0]) / flight, (target[:, 1] - b[:, 1]) / flight, (radius - b[:, 2]) / flight + 0.5 * g * flight], -1)
            guess = torch.where(have[:, None], yaw_vectors(torch.nan_to_num(guess_local), fl), ballistic)
            e1, e2 = spin_frame(guess)
            spin = flat.spin[live, 0:1] * e1 + flat.spin[live, 1:2] * e2
            launch = solve_launch(params, e, b[:, 0:3], spin, target, flight, ballistic, steps=steps)
            v_out = launch.velocity
            ok = launch.ok & feasible[live]
            self._count("reject_launch", (~ok).sum())
            shoulder = rules.shoulder(pelvis[live], fl)
            tip = yaw_vectors(flat.tip[live], fl)
            face = flat.face[live]
            strike = None
            for npass in range(2):
                start = frame_exp(strike_frame(v_out - b[:, 3:6]), flat.normal_rel[live])
                normal, residual, n_ok = normal_for_spin(params, e, b, v_out, flat.spin[live], start)
                n_ok = n_ok & (residual <= c["normal_residual_max_radps"])
                self._count("reject_normal", (ok & ~n_ok).sum())
                ok = ok & n_ok
                strike = plan_strike(params, e, t_contact[live], b, v_out, normal, face, shoulder, tip)
                self._count("reject_strike", (ok & ~strike.ok).sum())
                ok = ok & strike.ok
                if npass == 0:
                    second = solve_launch(params, e, b[:, 0:3], strike.spin_out, target, flight, v_out, steps=steps)
                    self._count("reject_launch_spin", (ok & ~second.ok).sum())
                    ok = ok & second.ok
                    v_out = torch.where(second.ok[:, None], second.velocity, v_out)
            assert strike is not None
            strike.ok = ok
            _scatter_strike(strikes, live, strike)
            # Plausibility of the paddle state (hitter frame).
            pel3 = torch.cat([pelvis[live], torch.full_like(pelvis[live, :1], c["pelvis_height_m"])], -1)
            u_l = yaw_vectors(strike.face_velocity, fl)
            n_l = yaw_vectors(strike.normal, fl)
            along = dot(u_l, n_l)
            brush = norm(u_l - along[:, None] * n_l)
            offset = yaw_vectors(strike.position, fl)[:, 0:2] - yaw_vectors(pel3, fl)[:, 0:2]
            coords = torch.stack(
                [norm(u_l), brush, torch.asin(n_l[:, 2].clamp(-1.0, 1.0)), torch.atan2(n_l[:, 1], n_l[:, 0]), offset[:, 0], offset[:, 1], b[:, 2]], -1
            )
            dens = rules.plausibility(coords)
            intended = fly(params, torch.cat([b[:, 0:3], strike.v_out, strike.spin_out], -1), e, steps=steps)
            if self.feasibility == "body" and begin is not None:
                wanted = classify_landing(params, e, b, intended)
                ok, held = self._body(plan, rows, live, ok, strike, t_contact[live], pelvis[live], pelvis_now, begin, b, e, fl, wanted, steps)
                if helds is None:
                    helds = _empty_strike(n_all, dtype, device)
                _scatter_strike(helds, live, held)
                strike.ok = ok
                _scatter_strike(strikes, live, strike)
            elif self.feasibility == "paddle" and begin is not None and ready is not None and track is not None and stance is not None:
                stroke_ok, follow, recover = self._stroke(plan, rows, live, ok, strike, t_contact[live], pelvis[live], begin, ready, track, stance)
                self._count("reject_stroke", (ok & ~stroke_ok).sum())
                ok = ok & stroke_ok
                strike.ok = ok
                _scatter_strike(strikes, live, strike)
                if follows is None or recovers is None:
                    follows = _empty_piece(n_all, dtype, device)
                    recovers = _empty_piece(n_all, dtype, device)
                _scatter_piece(follows, live, follow)
                _scatter_piece(recovers, live, recover)
            density[live] = torch.where(ok, dens, torch.full_like(dens, -float("inf")))
            landing[live] = intended.landing
            flight_t[live] = intended.t
            feasible[live] = ok
        feasible = feasible.reshape(r, d)
        density = density.reshape(r, d)
        plausible = density >= rules.plausibility_threshold if self.config.plausibility else feasible
        good = feasible & plausible
        index = torch.arange(d, device=device)[None, :].expand(r, d)
        first_good = torch.where(good, index, torch.full_like(index, d)).min(1).values
        best_density = torch.argmax(torch.where(feasible, density, torch.full_like(density, -float("inf"))), 1)
        pick = torch.where(first_good < d, first_good, best_density)
        chosen = feasible.any(1)
        flat_pick = torch.arange(r, device=device) * d + pick
        if dense:
            self._fill_dense(plan, chosen, flat_pick, strikes, pelvis, landing, flight_t, density, flat, feasible, follows, recovers, ready, helds)
            return chosen
        sel = torch.nonzero(chosen).flatten()
        target_rows = rows[sel]
        source = flat_pick[sel]
        _scatter_strike(plan.first, target_rows, strikes.take(source))
        _scatter_strike(plan.final, target_rows, strikes.take(source))
        _scatter_strike(plan.held, target_rows, (strikes if helds is None else helds).take(source))
        plan.pelvis_needed[target_rows] = pelvis[source]
        plan.intended_landing[target_rows] = landing[source]
        plan.intended_flight_s[target_rows] = flight_t[source]
        plan.plausibility[target_rows] = density.reshape(-1)[source]
        plan.draw_after_bounce_s[target_rows] = flat.after_bounce_s[source]
        plan.draw_spin[target_rows] = flat.spin[source]
        plan.draw_face[target_rows] = flat.face[source]
        plan.draw_tip[target_rows] = flat.tip[source]
        plan.draw_ood[target_rows] = flat.ood[source]
        plan.draw_donor[target_rows] = flat.donor[source]
        plan.draws_feasible[rows] = feasible.sum(1)
        if follows is not None and recovers is not None and ready is not None:
            if plan.follow is None or plan.recover is None or plan.ready is None:
                m_all = plan.status.shape[0]
                plan.follow = _empty_piece(m_all, dtype, device)
                plan.recover = _empty_piece(m_all, dtype, device)
                plan.ready = BladePlan.at_rest(torch.zeros((m_all, 3), dtype=dtype, device=device), torch.zeros((m_all, 4), dtype=dtype, device=device))
            plan_follow, plan_recover, plan_ready = plan.follow, plan.recover, plan.ready
            _scatter_piece(plan_follow, target_rows, follows.take(source))
            _scatter_piece(plan_recover, target_rows, recovers.take(source))
            sel_ready = ready.take(sel)
            plan_ready.p[target_rows] = sel_ready.p
            plan_ready.q[target_rows] = sel_ready.q
        return chosen

    def _stroke_slots(
        self, track: Any, owner: Tensor, approach: Tensor, a_start: Tensor, a_end: Tensor, coeff_a: Tensor, r_start: Tensor, coeff_r: Tensor, return_s: float
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        The hips over this stroke as move slots (``PelvisTrack.move`` twice: the approach, then the return to the
        stance), from the receiver's current slots: moves starting at or after the first new move are dropped and the
        rest end by then (a move without an approach leaves the old plan until the return).
        """

        start, end, coeffs, rest = track.slots(owner)
        cut = torch.where(approach, a_start, r_start)[:, None]
        later = start >= cut
        inf = torch.full_like(start, float("inf"))
        start = torch.where(later, inf, start)
        end = torch.where(later, inf, torch.minimum(end, cut))
        a_s = torch.where(approach, a_start, torch.full_like(a_start, float("inf")))
        start = torch.cat([start, a_s[:, None], r_start[:, None]], 1)
        end = torch.cat([end, torch.where(approach, a_end, a_s)[:, None], (r_start + return_s)[:, None]], 1)
        coeffs = torch.cat([coeffs, coeff_a[:, None], coeff_r[:, None]], 1)
        return start, end, coeffs, rest

    def _fill_dense(
        self,
        plan: Plan,
        chosen: Tensor,
        flat_pick: Tensor,
        strikes: Strike,
        pelvis: Tensor,
        landing: Tensor,
        flight_t: Tensor,
        density: Tensor,
        flat: IntentDraws,
        feasible: Tensor,
        follows: Piece | None,
        recovers: Piece | None,
        ready: BladePlan | None,
        helds: Strike | None = None,
    ) -> None:
        """The chosen draw of every row into ``plan`` by masks (dense mode: the plan's rows are all the rows)."""

        def put(target: Tensor, value: Tensor) -> None:
            target.copy_(torch.where(chosen.reshape(chosen.shape + (1,) * (target.dim() - 1)), value.to(target.dtype), target))

        held_source = strikes if helds is None else helds
        for f in fields(plan.first):
            put(getattr(plan.first, f.name), getattr(strikes, f.name)[flat_pick])
            put(getattr(plan.final, f.name), getattr(strikes, f.name)[flat_pick])
            put(getattr(plan.held, f.name), getattr(held_source, f.name)[flat_pick])
        put(plan.pelvis_needed, pelvis[flat_pick])
        put(plan.intended_landing, landing[flat_pick])
        put(plan.intended_flight_s, flight_t[flat_pick])
        put(plan.plausibility, density.reshape(-1)[flat_pick])
        put(plan.draw_after_bounce_s, flat.after_bounce_s[flat_pick])
        put(plan.draw_spin, flat.spin[flat_pick])
        put(plan.draw_face, flat.face[flat_pick])
        put(plan.draw_tip, flat.tip[flat_pick])
        put(plan.draw_ood, flat.ood[flat_pick])
        put(plan.draw_donor, flat.donor[flat_pick])
        plan.draws_feasible.copy_(feasible.sum(1))
        if follows is not None and recovers is not None and ready is not None:
            m_all = plan.status.shape[0]
            dtype, device = self.params.dtype, self.params.device
            if plan.follow is None or plan.recover is None or plan.ready is None:
                plan.follow = _empty_piece(m_all, dtype, device)
                plan.recover = _empty_piece(m_all, dtype, device)
                plan.ready = BladePlan.at_rest(torch.zeros((m_all, 3), dtype=dtype, device=device), torch.zeros((m_all, 4), dtype=dtype, device=device))
            for name in ("start", "end", "position", "rotation", "reference"):
                put(getattr(plan.follow, name), getattr(follows, name)[flat_pick])
                put(getattr(plan.recover, name), getattr(recovers, name)[flat_pick])
            put(plan.ready.p, ready.p)
            put(plan.ready.q, ready.q)

    def decide_dense(
        self,
        env: Tensor,
        profile: Tensor,
        end: Tensor,
        t_decide: Tensor,
        t_hit: Tensor,
        ball: Tensor,
        stance: Tensor,
        pelvis_now: Tensor,
        seen: Tensor,
        seen_t: Tensor,
        seen_state: Tensor,
        blade: BladePlan,
        slots: tuple[Tensor, Tensor, Tensor, Tensor],
        hazard_scale: Tensor,
        valid: Tensor,
    ) -> Plan:
        """
        :meth:`decide` for every row at once, without row compaction or host synchronisation (CUDA-graph capturable).

        :param slots: The receiver's hips plan as :class:`lib.opponent.rally.PelvisBatch` slots
            ``(start (M, K), end (M, K), coefficients (M, K, 6, 2), rest (M, 2))``.
        :param valid: Rows that are real (padding rows decide nothing).
        """

        from .rally import pelvis_eval

        self._dense = True
        try:
            params, rules = self.params, self.rules
            m = env.shape[0]
            dtype, device = params.dtype, params.device
            flip = end > 0.0
            perceived = self.perceive(ball)
            predicted = predict_bounce(params, perceived, env, end, steps=38)
            t_bounce = torch.where(seen, seen_t, t_decide + predicted.t)
            bounce = torch.where(seen[:, None], seen_state + (perceived - ball), predicted.after)
            known = (seen | predicted.ok) & valid
            status = torch.full((m,), LET_GO, dtype=torch.long, device=device)
            local = yaw_states(bounce, flip)
            stance_local = yaw_xy(stance, flip)
            speed_after = torch.sqrt(local[:, 3] ** 2 + local[:, 4] ** 2)
            p_stop = hazard_scale * rules.stop_probability(profile, local[:, 0:2], t_bounce - t_hit, speed_after, stance_local)
            u = torch.rand(m, dtype=dtype, device=device, generator=self.generator)
            stop = known & (u < p_stop)
            status = torch.where(stop, torch.full_like(status, STOP), status)
            plan = self._empty_plan(env, profile, end, t_hit, t_decide, t_bounce, bounce, torch.logical_not(seen), p_stop, status)
            allowed = known & ~stop
            query = torch.stack([local[:, 0], local[:, 1], local[:, 3], local[:, 4], local[:, 5], stance_local[:, 0], stance_local[:, 1]], -1)
            floor_z = -params.table_height[env]
            half = (params.table_half_length[env], params.table_half_width[env])
            ready = self.strokes.draw_ready_dense(profile, stance, flip, floor_z, half, self.generator)
            track = _SlotTrack(*slots, pelvis_eval)

            draws = self.intent.draw(profile, query, self.draws, generator=self.generator)
            if not isinstance(self.intent, KnnIntent):
                draws.ood = self._ood.out_of_distribution(profile, query)[:, None].expand(m, self.draws)
            rows = torch.arange(m, device=device)
            chosen = self._plan_draws(plan, rows, draws, pelvis_now, blade, ready, track, stance, allowed)
            plan.status = torch.where(allowed, torch.where(chosen, torch.full_like(status, ATTEMPT), torch.full_like(status, UNREACHABLE)), status)
            return plan
        finally:
            self._dense = False

    def _body(
        self,
        plan: Plan,
        rows: Tensor,
        live: Tensor,
        ok: Tensor,
        strike: Strike,
        t_c: Tensor,
        pelvis_needed: Tensor,
        pelvis_now: Tensor,
        begin: BladePlan,
        ball: Tensor,
        env: Tensor,
        flip: Tensor,
        wanted: Tensor,
        steps: int | None,
    ) -> tuple[Tensor, Strike]:
        """
        ``Rally._stroke_body`` at a decision for each live draw (body-first): the body runtime's verdict (:mod:`.body`:
        the reach rule, the acceptance, the nudge, the same-outcome rule) and the contact the body holds.
        """

        assert self.body is not None
        owner = live // self.draws
        profile = plan.profile[rows][owner]
        t_from = (torch.floor(plan.t_decide[rows][owner] / TICK_S + 1e-9) + 1.0) * TICK_S
        hand = begin.take(owner)
        request = Contact(t_c, strike.position, strike.orientation, strike.velocity, strike.angular_velocity)
        held_ok, held, cause, held_face = self.body.hold_decision(
            self.params,
            env,
            profile,
            flip,
            t_from,
            request,
            strike.face,
            ball,
            wanted,
            pelvis_needed,
            pelvis_now[owner],
            hand.p,
            hand.v,
            self.generator,
            steps,
            tuple(self.config.profiles),
        )
        self._count("reject_body_reach", (ok & (cause == 1)).sum())
        self._count("reject_body_runtime", (ok & (cause == 2)).sum())
        self._count("reject_body_outcome", (ok & (cause == 3)).sum())
        ok = ok & held_ok
        return ok, self._held_strike(strike, held, held_face, env, ok)

    def _held_strike(self, strike: Strike, held: Contact, face: Tensor, env: Tensor, ok: Tensor) -> Strike:
        """A held contact as a :class:`Strike` (the request's ball, the rubber it presents; the impact law's outgoing ball)."""

        rotation = reference_from_game(quat_to_matrix(held.orientation))
        z = rotation[:, :, 2]
        toward = torch.where((dot(z, strike.ball_in[:, 0:3] - held.position) >= 0.0)[:, None], z, -z)
        v_out, spin_out = forward(self.params, env, strike.ball_in, held.position, rotation, held.velocity, held.angular_velocity, toward)
        point = strike.ball_in[:, 0:3] - self.params.radius[env][:, None] * toward
        sign = torch.where(face == 0, torch.ones_like(held.t), -torch.ones_like(held.t))
        return Strike(
            time=held.t,
            ball_in=strike.ball_in,
            position=held.position,
            orientation=held.orientation,
            rotation=rotation,
            velocity=held.velocity,
            angular_velocity=held.angular_velocity,
            normal=z * sign[:, None],
            face=face,
            v_out=v_out,
            spin_out=spin_out,
            face_velocity=held.velocity + cross(held.angular_velocity, point - held.position),
            ok=ok,
        )

    def _stroke(
        self,
        plan: Plan,
        rows: Tensor,
        live: Tensor,
        ok: Tensor,
        strike: Strike,
        t_c: Tensor,
        pelvis_needed: Tensor,
        begin: BladePlan,
        ready: BladePlan,
        track: Any,
        stance: Tensor,
    ) -> tuple[Tensor, Piece, Piece]:
        """``Rally.stroke`` (paddle-first) for each live draw: the stroke check against the capture's motion limits."""

        d = self.draws
        owner = live // d
        env = plan.env[rows][owner]
        profile = plan.profile[rows][owner]
        flip = plan.end[rows][owner] > 0.0
        t_decide = plan.t_decide[rows][owner]
        t_hit = plan.t_hit[rows][owner]
        t_from = (torch.floor(t_decide / TICK_S + 1e-9) + 1.0) * TICK_S
        lim = self.rules.limits
        ret = self.rules.constants["pelvis_return_s"]
        a_start = torch.maximum(t_decide, t_hit + lim["footwork_latency_s"])
        a_end = t_c - 0.05
        approach = a_end > a_start
        p0, v0 = track(owner, a_start[:, None])
        p0, v0 = p0[:, 0], v0[:, 0]
        coeff_a = _quintic2(p0, v0, pelvis_needed, (a_end - a_start).clamp_min(1e-6))
        r_start = t_c + ret[0]
        # The return move starts from the plan at r_start: the approach's end (at rest) or, without an approach, the
        # old plan there.
        q0, w0 = track(owner, r_start[:, None])
        r_from = torch.where(approach[:, None], pelvis_needed, q0[:, 0])
        r_vel = torch.where(approach[:, None], torch.zeros_like(pelvis_needed), w0[:, 0])
        coeff_r = _quintic2(r_from, r_vel, stance[owner], torch.full_like(t_c, ret[1] - ret[0]))

        def pelvis(sub: Tensor, times: Tensor) -> Tensor:
            current, _ = track(owner[sub], times)
            a = _eval2(coeff_a[sub], (times - a_start[sub][:, None]).clamp(min=0.0).minimum((a_end - a_start)[sub][:, None].clamp_min(0.0)))
            r = _eval2(coeff_r[sub], (times - r_start[sub][:, None]).clamp(min=0.0, max=ret[1] - ret[0]))
            use_a = approach[sub][:, None] & (times >= a_start[sub][:, None])
            out = torch.where(use_a[..., None], a, current)
            return torch.where((times >= r_start[sub][:, None])[..., None], r, out)

        contact = BladePlan(strike.position, strike.velocity, torch.zeros_like(strike.position), strike.orientation, strike.angular_velocity)
        floor_z = -self.params.table_height[env]
        half = (self.params.table_half_length[env], self.params.table_half_width[env])
        if self._dense:
            slots = self._stroke_slots(track, owner, approach, a_start, a_end, coeff_a, r_start, coeff_r, ret[1] - ret[0])
            result, follow, recover = self.strokes.check_dense(profile, flip, begin.take(owner), t_from, t_c, contact, slots, ready.take(owner), floor_z, half)
        else:
            result, follow, recover = self.strokes.check(profile, flip, begin.take(owner), t_from, t_c, contact, pelvis, ready.take(owner), floor_z, half)
        return result & ok, follow, recover

    # ---- the late correction and the execution ----------------------------------------------------

    def refine(
        self, plan: Plan, rows: Tensor, t_now: Tensor, ball_now: Tensor, t_bounce: Tensor, pelvis_at_contact: Tensor, mask: Tensor | None = None
    ) -> None:
        """
        ``Rally.refine`` for ``rows`` of ``plan``: perceive at ``t_now`` (the real bounce plus 0.11 s), replan to the
        intended landing at the drawn contact time and move the first plan towards it within the cited caps.

        :param pelvis_at_contact: The receiver's hips at the drawn contact time (its track).
        :param mask: Rows that may refine (the others keep their plan); with it nothing synchronises with the host
            (fixed step counts), so a call can be graph-captured.
        """

        if rows.numel() == 0:
            return
        steps = 70 if mask is not None else None
        params, rules = self.params, self.rules
        lim = rules.limits
        env = plan.env[rows]
        flip = plan.end[rows] > 0.0
        first = plan.first.take(rows)
        perceived = self.perceive(ball_now)
        t_contact = t_bounce + plan.draw_after_bounce_s[rows]
        go = t_contact >= t_now + 0.03
        if mask is not None:
            go = go & mask
        ball = propagate(params, perceived, env, torch.where(go, t_contact - t_now, torch.zeros_like(t_now)), steps=steps)
        launch = solve_launch(params, env, ball[:, 0:3], first.spin_out, plan.intended_landing[rows], plan.intended_flight_s[rows], first.v_out, steps=steps)
        go = go & torch.isfinite(launch.residual) & (launch.residual <= rules.constants["launch_residual_max"])
        normal, _res, n_ok = normal_for_spin(params, env, ball, launch.velocity, plan.draw_spin[rows], first.normal)
        go = go & n_ok
        second = plan_strike(
            params,
            env,
            t_contact,
            ball,
            launch.velocity,
            normal,
            plan.draw_face[rows],
            rules.shoulder(pelvis_at_contact, flip),
            yaw_vectors(plan.draw_tip[rows], flip),
        )
        go = go & second.ok
        preswing = t_now <= plan.swing_start[rows]
        time_cap = torch.where(preswing, torch.full_like(t_now, lim["preswing_time_s"]), torch.full_like(t_now, lim["late_time_s"]))
        offset_cap = torch.where(preswing, torch.full_like(t_now, lim["preswing_offset_m"]), torch.full_like(t_now, lim["late_offset_m"]))
        dt = torch.minimum(torch.maximum(second.time - first.time, -time_cap), time_cap)
        dp_raw = second.position - first.position
        dpn = norm(dp_raw)
        dp = torch.where((dpn > offset_cap)[:, None], dp_raw * (offset_cap / dpn.clamp_min(1e-300))[:, None], dp_raw)
        turn = _capped(quat_to_rotvec(quat_multiply(second.orientation, quat_conjugate(first.orientation))), lim["late_rotation_rad"])
        dv = _capped(second.velocity - first.velocity, lim["late_speed_mps"])
        rotation = rotation_from_rotvec(turn) @ first.rotation
        sign = torch.where(second.face == 0, torch.ones_like(t_now), -torch.ones_like(t_now))
        final = Strike(
            time=first.time + dt,
            ball_in=second.ball_in,
            position=first.position + dp,
            orientation=quat_multiply(quat_from_rotvec(turn), first.orientation),
            rotation=rotation,
            velocity=first.velocity + dv,
            angular_velocity=second.angular_velocity,
            normal=rotation[:, :, 2] * sign[:, None],
            face=second.face,
            v_out=second.v_out,
            spin_out=second.spin_out,
            face_velocity=second.face_velocity,
            ok=torch.ones_like(go),
        )
        merged = final.where(go, first)
        _scatter_strike(plan.final, rows, merged)
        plan.refined[rows] = go
        if mask is None:
            self._count("refined", go.sum())

    def execute(self, plan: Plan, rows: Tensor, t_now: Tensor | None = None) -> ContactCommand:
        """
        ``Rally.execute`` for ``rows``: the final plan with the execution noise at the noise scale. Body-first (with
        ``t_now``, the execution time): the noisy contact as the body re-holds it (:meth:`_execute_body`).
        """

        final = plan.final.take(rows)
        sd = (self.noise_sd * self.noise_scale).expand(rows.numel(), 4)
        command, noise = execute(final, self.params.radius[plan.env[rows]], sd, self.generator, self.config.paddle_speed_scale)
        command.active = plan.status[rows] == ATTEMPT
        plan.noise[rows] = noise
        plan.noise_applied[rows] = torch.ones_like(final.time)
        if self.feasibility != "body" or t_now is None:
            return command
        return self._execute_body(plan, rows, final, command, t_now)

    def _execute_body(self, plan: Plan, rows: Tensor, final: Strike, command: ContactCommand, t_now: Tensor) -> ContactCommand:
        """
        The generator's execution attempts (``Rally.play``): the noisy contact; with a late correction, the first held
        contact moved by the same noise; then the noise at half, a quarter and none; each re-held on the first plan's
        body (``renudge``, :meth:`lib.opponent.body.BodyFeasibility.hold_execution`) and kept if it
        lands as requested. The first that holds is played; if none does, the first plan's held contact as laid
        (``execute_kept_first``, no noise).
        """

        assert self.body is not None
        m = rows.numel()
        env = plan.env[rows]
        flip = plan.end[rows] > 0.0
        t_from = (torch.floor(t_now / TICK_S + 1e-9) + 1.0) * TICK_S
        clean = Contact(final.time, final.position, final.orientation, final.velocity, final.angular_velocity)
        noisy = Contact(command.t, command.position, command.orientation, command.velocity, command.angular_velocity)
        h = plan.held.take(rows)
        first = Contact(h.time, h.position, h.orientation, h.velocity, h.angular_velocity)
        attempts = [noisy, _shifted(first, clean, noisy), _blend(clean, noisy, 0.5), _blend(clean, noisy, 0.25), clean]
        fractions = (1.0, 1.0, 0.5, 0.25, 0.0)
        allowed = torch.stack([torch.ones_like(plan.refined[rows]), plan.refined[rows], *([torch.ones_like(plan.refined[rows])] * 3)])
        k = len(attempts)

        def cat(name: str) -> Tensor:
            return torch.cat([getattr(a, name) for a in attempts])

        request = Contact(cat("t"), cat("position"), cat("orientation"), cat("velocity"), cat("angular_velocity"))
        firsts = Contact(
            *(
                getattr(first, name).repeat((k,) + (1,) * (getattr(first, name).dim() - 1))
                for name in ("t", "position", "orientation", "velocity", "angular_velocity")
            )
        )
        ok, held, wanted, faces = self.body.hold_execution(
            self.params,
            env.repeat(k),
            plan.profile[rows].repeat(k),
            flip.repeat(k),
            t_from.repeat(k),
            request,
            firsts,
            final.face.repeat(k),
            final.ball_in.repeat(k, 1),
            self.generator,
            first_face=h.face.repeat(k),
        )
        ok = ok.reshape(k, m) & allowed
        index = torch.arange(k, device=ok.device)[:, None].expand(k, m)
        pick = torch.where(ok, index, torch.full_like(index, k)).min(0).values
        kept = pick == k
        source = pick.clamp(max=k - 1) * m + torch.arange(m, device=ok.device)
        chosen = held.take(source).where(~kept, first)
        command.t, command.position, command.orientation = chosen.t, chosen.position, chosen.orientation
        command.velocity, command.angular_velocity = chosen.velocity, chosen.angular_velocity
        command.face = torch.where(kept, h.face, faces[source])
        t0, p0, v0, t1, p1, v1 = self.body.realise(plan.profile[rows], flip, chosen, final.ball_in, self.generator)
        command.tick0, command.tick1 = (t0, p0, v0), (t1, p1, v1)
        share = torch.as_tensor(fractions, dtype=final.time.dtype, device=final.time.device)[pick.clamp(max=k - 1)]
        plan.noise_applied[rows] = torch.where(kept, -torch.ones_like(share), share)
        self._count("execute_kept_first", (kept & command.active).sum())
        self._count("execute_partial_noise", (~kept & (share < 1.0) & command.active).sum())
        self._count("execute_refine_dropped", (~kept & (pick == 1) & command.active).sum())
        for c, name in enumerate(("in", "net", "own_half", "out")):
            self._count(f"execute_request_{name}", ((wanted[:m] == c) & command.active).sum())
        return command


class _SlotTrack:
    """The receivers' hips plans as move slots: callable as a track, and its slots for the dense stroke check."""

    def __init__(self, start: Tensor, end: Tensor, coeffs: Tensor, rest: Tensor, evaluate: Any) -> None:
        self.start, self.end, self.coeffs, self.rest, self.evaluate = start, end, coeffs, rest, evaluate

    def __call__(self, sub: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
        return self.evaluate(self.start[sub], self.end[sub], self.coeffs[sub], self.rest[sub], times)

    def slots(self, sub: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.start[sub], self.end[sub], self.coeffs[sub], self.rest[sub]


def _quintic2(x0: Tensor, v0: Tensor, x1: Tensor, duration: Tensor) -> Tensor:
    """A planar quintic from ``(x0, v0, a0 = 0)`` to rest at ``x1`` (``PelvisTrack.move``): ``(R, 6, 2)``."""

    t = duration[:, None]
    c3 = (20.0 * (x1 - x0) - 12.0 * v0 * t) / (2.0 * t**3)
    c4 = (30.0 * (x0 - x1) + 16.0 * v0 * t) / (2.0 * t**4)
    c5 = (12.0 * (x1 - x0) - 6.0 * v0 * t) / (2.0 * t**5)
    return torch.stack([x0, v0, torch.zeros_like(x0), c3, c4, c5], 1)


def _eval2(c: Tensor, s: Tensor) -> Tensor:
    s = s[..., None]
    c = c[:, None]
    return c[:, :, 0] + s * (c[:, :, 1] + s * (c[:, :, 2] + s * (c[:, :, 3] + s * (c[:, :, 4] + s * c[:, :, 5]))))


def _empty_piece(m: int, dtype: Any, device: Any) -> Piece:
    z = torch.zeros(m, dtype=dtype, device=device)
    q = torch.zeros((m, 4), dtype=dtype, device=device)
    q[:, 0] = 1.0
    return Piece(z.clone(), z.clone(), torch.zeros((m, 6, 3), dtype=dtype, device=device), torch.zeros((m, 6, 3), dtype=dtype, device=device), q)


def _scatter_piece(target: Piece, index: Tensor, source: Piece) -> None:
    for name in ("start", "end", "position", "rotation", "reference"):
        getattr(target, name)[index] = getattr(source, name)


def _shifted(base: Contact, clean: Contact, noisy: Contact) -> Contact:
    """``episode._shifted_knot``: ``base`` moved by the noise that took ``clean`` to ``noisy``."""

    turn = quat_multiply(noisy.orientation, quat_conjugate(clean.orientation))
    return Contact(
        base.t + (noisy.t - clean.t),
        base.position + (noisy.position - clean.position),
        quat_multiply(turn, base.orientation),
        base.velocity + (noisy.velocity - clean.velocity),
        base.angular_velocity + (noisy.angular_velocity - clean.angular_velocity),
    )


def _blend(a: Contact, b: Contact, fraction: float) -> Contact:
    """``episode._blend_knots``: ``a`` moved ``fraction`` of the way to ``b`` (linear; the rotation by its vector)."""

    turn = quat_to_rotvec(quat_multiply(b.orientation, quat_conjugate(a.orientation)))
    return Contact(
        a.t + fraction * (b.t - a.t),
        a.position + fraction * (b.position - a.position),
        quat_multiply(quat_from_rotvec(fraction * turn), a.orientation),
        a.velocity + fraction * (b.velocity - a.velocity),
        a.angular_velocity + fraction * (b.angular_velocity - a.angular_velocity),
    )


def unit_cross_check(a: Tensor, b: Tensor) -> Tensor:
    """``|a x b|`` (kept for tests of the frame helpers)."""

    return norm(cross(a, b))


__all__ = ["ATTEMPT", "LET_GO", "STATUS_NAMES", "STOP", "UNREACHABLE", "Opponent", "OpponentConfig", "Plan"]
