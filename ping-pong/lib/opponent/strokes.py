"""
Whether a stroke can be laid like a person: the generator's paddle-first stroke check (``Rally.stroke`` with
``motion.violation``), batched.

A drawn shot that survives the contact phase, the launch and the strike is still discarded by the generator when its
blade cannot get there like a recorded player: from the blade's plan state at the tick after the decision, a
reposition to a backswing and the forward swing to the contact (the nearest captured stroke's backswing 0.2 s out,
moved with this contact, then as captured, then at 0.25, 0.15 and 0.12 s, then straight backswings along the contact
velocity from 0.30 s down to 0.08 s, finally one swing straight from where the blade is), a follow-through
(captured, then straight, shortening) and a recovery to a recorded ready pose (0.5 s, lengthened up to 1.5 s) must
each keep, sampled at 120 Hz: blade speed and turn rate under the capture's p99.9 (with the planner's 0.95 margin)
outside the stroke window and under the in-window p99.9 inside it; reach from the hips within the capture's p99.9
(0.97) and its p99 outside the window; blade height within the capture's span; the hand (through the grip) within
the body-reach model; the hand at least 2 cm over the board. Pieces are quintics in position and in rotation vector
(``track.Primitive``). On robot-speed balls this rule, not the contact phase, is what makes most balls unreachable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from lib.physics.vectorized._torch import torch

from .ops import quat_conjugate, quat_from_rotvec, quat_multiply, quat_to_matrix, quat_to_rotvec, yaw_quaternions, yaw_vectors

Tensor = Any
TICK_S = 1.0 / 24.0


@dataclass
class BladePlan:
    """A blade plan state per row: position, velocity, acceleration, orientation (wxyz), angular velocity."""

    p: Tensor
    v: Tensor
    a: Tensor
    q: Tensor
    w: Tensor

    def take(self, index: Tensor) -> BladePlan:
        return BladePlan(self.p[index], self.v[index], self.a[index], self.q[index], self.w[index])

    @classmethod
    def at_rest(cls, p: Tensor, q: Tensor) -> BladePlan:
        z = torch.zeros_like(p)
        return cls(p, z, z.clone(), q, z.clone())


def quintic(x0: Tensor, v0: Tensor, a0: Tensor, x1: Tensor, v1: Tensor, a1: Tensor, duration: Tensor) -> Tensor:
    """``track.quintic`` per row: coefficients ``(R, 6, 3)``."""

    t = duration[:, None]
    c3 = (20.0 * (x1 - x0) - (8.0 * v1 + 12.0 * v0) * t - (3.0 * a0 - a1) * t * t) / (2.0 * t**3)
    c4 = (30.0 * (x0 - x1) + (14.0 * v1 + 16.0 * v0) * t + (3.0 * a0 - 2.0 * a1) * t * t) / (2.0 * t**4)
    c5 = (12.0 * (x1 - x0) - 6.0 * (v1 + v0) * t - (a0 - a1) * t * t) / (2.0 * t**5)
    return torch.stack([x0, v0, 0.5 * a0, c3, c4, c5], 1)


@dataclass
class Piece:
    """``track.Primitive`` per row: quintic position and rotation vector over ``[start, end]``, ``exp(r) ref``."""

    start: Tensor
    end: Tensor
    position: Tensor
    rotation: Tensor
    reference: Tensor

    @classmethod
    def between(cls, start: Tensor, end: Tensor, begin: BladePlan, finish: BladePlan) -> Piece:
        duration = (end - start).clamp_min(1e-9)
        position = quintic(begin.p, begin.v, begin.a, finish.p, finish.v, finish.a, duration)
        r0 = quat_to_rotvec(quat_multiply(begin.q, quat_conjugate(finish.q)))
        zero = torch.zeros_like(r0)
        rotation = quintic(r0, begin.w, zero, zero, finish.w, zero, duration)
        return cls(start, end, position, rotation, finish.q)

    def sample(self, times: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Positions, velocities ``(R, S, 3)`` and orientations ``(R, S, 4)`` at ``times`` ``(R, S)``."""

        s = torch.minimum(torch.maximum(times - self.start[:, None], torch.zeros_like(times)), (self.end - self.start)[:, None])[..., None]
        c = self.position[:, None]
        x = c[:, :, 0] + s * (c[:, :, 1] + s * (c[:, :, 2] + s * (c[:, :, 3] + s * (c[:, :, 4] + s * c[:, :, 5]))))
        v = c[:, :, 1] + s * (2.0 * c[:, :, 2] + s * (3.0 * c[:, :, 3] + s * (4.0 * c[:, :, 4] + s * 5.0 * c[:, :, 5])))
        r = self.rotation[:, None]
        rv = r[:, :, 0] + s * (r[:, :, 1] + s * (r[:, :, 2] + s * (r[:, :, 3] + s * (r[:, :, 4] + s * r[:, :, 5]))))
        q = quat_from_rotvec(rv)
        return x, v, quat_multiply(q, self.reference[:, None, :].expand(q.shape))

    def state(self, t: Tensor) -> BladePlan:
        s = torch.minimum(torch.maximum(t - self.start, torch.zeros_like(t)), self.end - self.start)[:, None]
        c = self.position
        p = c[:, 0] + s * (c[:, 1] + s * (c[:, 2] + s * (c[:, 3] + s * (c[:, 4] + s * c[:, 5]))))
        v = c[:, 1] + s * (2.0 * c[:, 2] + s * (3.0 * c[:, 3] + s * (4.0 * c[:, 4] + s * 5.0 * c[:, 5])))
        a = 2.0 * c[:, 2] + s * (6.0 * c[:, 3] + s * (12.0 * c[:, 4] + s * 20.0 * c[:, 5]))
        r = self.rotation
        rv = r[:, 0] + s * (r[:, 1] + s * (r[:, 2] + s * (r[:, 3] + s * (r[:, 4] + s * r[:, 5]))))
        w = r[:, 1] + s * (2.0 * r[:, 2] + s * (3.0 * r[:, 3] + s * (4.0 * r[:, 4] + s * 5.0 * r[:, 5])))
        q = quat_multiply(quat_from_rotvec(rv), self.reference)
        return BladePlan(p, v, a, q / torch.sqrt((q * q).sum(-1, keepdim=True)), w)

    def take(self, index: Tensor) -> Piece:
        return Piece(self.start[index], self.end[index], self.position[index], self.rotation[index], self.reference[index])


def durations(longest: float, shortest: float) -> list[float]:
    out = [longest]
    while out[-1] * 0.85 > shortest:
        out.append(out[-1] * 0.85)
    if out[-1] > shortest:
        out.append(shortest)
    return out


def quat_angles(a: Tensor, b: Tensor) -> Tensor:
    w = (a * b).sum(-1)
    vector = a[..., :1] * b[..., 1:] - b[..., :1] * a[..., 1:] - torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    return 2.0 * torch.atan2(torch.sqrt((vector * vector).sum(-1)), w.abs())


def _rep_plan(plan: BladePlan, count: int) -> BladePlan:
    """Each row's plan state repeated ``count`` times (candidate-minor)."""

    return BladePlan(*(getattr(plan, f).repeat_interleave(count, 0) for f in ("p", "v", "a", "q", "w")))


def _rebase(coeffs: Tensor, start: Tensor, origin: Tensor) -> Tensor:
    """Pelvis move coefficients are in time since each move's start: unchanged by moving the time origin."""

    del start, origin
    return coeffs


def broken_core(
    k: Tensor,
    starts: Tensor,
    ends: Tensor,
    pos: Tensor,
    rot: Tensor,
    ref: Tensor,
    t_c: Tensor,
    slot_start: Tensor,
    slot_end: Tensor,
    slot_coeffs: Tensor,
    slot_rest: Tensor,
    lim: Tensor,
    grip: Tensor,
    box_lo: Tensor,
    box_hi: Tensor,
    arm: Tensor,
    sign: Tensor,
    floor_z: Tensor,
    half: Tensor,
    rate: float,
    window: float,
    board: float,
    hips_h: float,
) -> Tensor:
    """
    ``motion.violation`` of consecutive pieces sampled at ``rate`` (one pure function, compiled): ``(R,)`` true where a
    limit breaks. Times are relative to each row's origin. Pieces: ``starts``, ``ends`` ``(R, P)``, quintic
    coefficients ``(R, P, 6, 3)`` in position and rotation vector, end orientations ``(R, P, 4)``. ``lim`` columns:
    speed and in-window speed, turn and in-window turn, reach p99.9 (margined) and p99, height low and high.
    """

    span = (ends[:, -1] - starts[:, 0]).clamp_min(0.0)
    n_reg = torch.ceil(span * rate - 1e-5).clamp_min(0.0)
    count = k.shape[0]
    times = torch.where(k[None, :] == n_reg[:, None], ends[:, -1:], starts[:, :1] + k[None, :] / rate)
    valid = k[None, :] <= n_reg[:, None]
    pieces = starts.shape[1]
    p_all = torch.zeros((*times.shape, 3), dtype=times.dtype, device=times.device)
    v_all = torch.zeros_like(p_all)
    q_all = torch.zeros((*times.shape, 4), dtype=times.dtype, device=times.device)
    for i in range(pieces):
        s = torch.minimum(torch.maximum(times - starts[:, i : i + 1], torch.zeros_like(times)), (ends[:, i : i + 1] - starts[:, i : i + 1]))[..., None]
        c = pos[:, i][:, None]
        x = c[:, :, 0] + s * (c[:, :, 1] + s * (c[:, :, 2] + s * (c[:, :, 3] + s * (c[:, :, 4] + s * c[:, :, 5]))))
        v = c[:, :, 1] + s * (2.0 * c[:, :, 2] + s * (3.0 * c[:, :, 3] + s * (4.0 * c[:, :, 4] + s * 5.0 * c[:, :, 5])))
        rr = rot[:, i][:, None]
        rv = rr[:, :, 0] + s * (rr[:, :, 1] + s * (rr[:, :, 2] + s * (rr[:, :, 3] + s * (rr[:, :, 4] + s * rr[:, :, 5]))))
        angle = torch.sqrt((rv * rv).sum(-1, keepdim=True))
        small = angle < 1e-12
        half_angle = 0.5 * angle
        scale = torch.where(small, 0.5 * torch.ones_like(angle), torch.sin(half_angle) / torch.where(small, torch.ones_like(angle), angle))
        dq = torch.cat([torch.cos(half_angle), rv * scale], -1)
        q = quat_multiply(dq, ref[:, i][:, None, :].expand_as(dq))
        inside = torch.ones_like(valid) if i == 0 else times >= starts[:, i : i + 1]
        if i + 1 < pieces:
            inside = inside & (times < starts[:, i + 1 : i + 2])
        p_all = torch.where(inside[..., None], x, p_all)
        v_all = torch.where(inside[..., None], v, v_all)
        q_all = torch.where(inside[..., None], q, q_all)
    # The hips (``pelvis_eval`` on the move slots).
    began = slot_start[:, None, :] <= times[..., None]
    key = torch.where(began, slot_start[:, None, :].expand_as(began), torch.full_like(began, -1e30, dtype=times.dtype))
    slot = torch.argmax(key, -1)
    any_move = began.any(-1)
    s0 = slot_start.gather(1, slot)
    e0 = slot_end.gather(1, slot)
    cc = torch.gather(slot_coeffs[:, None].expand(-1, count, -1, -1, -1), 2, slot[:, :, None, None, None].expand(-1, -1, 1, 6, 2))[:, :, 0]
    sp = (torch.minimum(times, e0) - s0).clamp_min(0.0)[..., None]
    xp = cc[..., 0, :] + sp * (cc[..., 1, :] + sp * (cc[..., 2, :] + sp * (cc[..., 3, :] + sp * (cc[..., 4, :] + sp * cc[..., 5, :]))))
    pelvis = torch.where(any_move[..., None], xp, slot_rest[:, None, :].expand_as(xp))
    # The limits.
    inside_w = (times - t_c[:, None]).abs() <= window
    speed = torch.sqrt((v_all * v_all).sum(-1))
    bad = valid & ((~inside_w & (speed > lim[:, 0:1])) | (inside_w & (speed > lim[:, 1:2])))
    w = (q_all[:, 1:] * q_all[:, :-1]).sum(-1)
    vec = q_all[:, 1:, :1] * q_all[:, :-1, 1:] - q_all[:, :-1, :1] * q_all[:, 1:, 1:] - torch.cross(q_all[:, 1:, 1:], q_all[:, :-1, 1:], dim=-1)
    dt = (times[:, 1:] - times[:, :-1]).clamp_min(1e-9)
    turn = 2.0 * torch.atan2(torch.sqrt((vec * vec).sum(-1)), w.abs()) / dt
    pair = valid[:, 1:] & valid[:, :-1]
    outside = ~(inside_w[:, 1:] | inside_w[:, :-1])
    bad_turn = pair & ((outside & (turn > lim[:, 2:3])) | (~outside & (turn > lim[:, 3:4])))
    reach = torch.sqrt(((p_all[..., 0:2] - pelvis) ** 2).sum(-1))
    bad = bad | (valid & ((reach > lim[:, 4:5]) | (~inside_w & (reach > lim[:, 5:6]))))
    height = p_all[..., 2] - floor_z[:, None]
    bad = bad | (valid & ((height < lim[:, 6:7]) | (height > lim[:, 7:8])))
    # The hand through the grip, against the body-reach model and the board.
    qw, qx, qy, qz = q_all[..., 0], q_all[..., 1], q_all[..., 2], q_all[..., 3]
    norm_q = torch.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm_q, qx / norm_q, qy / norm_q, qz / norm_q
    gx, gy, gz = grip[:, None, 0], grip[:, None, 1], grip[:, None, 2]
    hand_x = p_all[..., 0] - ((1 - 2 * (qy * qy + qz * qz)) * gx + 2 * (qx * qy - qz * qw) * gy + 2 * (qx * qz + qy * qw) * gz)
    hand_y = p_all[..., 1] - (2 * (qx * qy + qz * qw) * gx + (1 - 2 * (qx * qx + qz * qz)) * gy + 2 * (qy * qz - qx * qw) * gz)
    hand_z = p_all[..., 2] - (2 * (qx * qz - qy * qw) * gx + 2 * (qy * qz + qx * qw) * gy + (1 - 2 * (qx * qx + qy * qy)) * gz)
    lx = (hand_x - pelvis[..., 0]) * sign[:, None]
    ly = (hand_y - pelvis[..., 1]) * sign[:, None]
    lz = hand_z - hips_h
    gap_x = torch.maximum(torch.maximum(box_lo[:, None, 0] - lx, torch.zeros_like(lx)), lx - box_hi[:, None, 0])
    gap_y = torch.maximum(torch.maximum(box_lo[:, None, 1] - ly, torch.zeros_like(ly)), ly - box_hi[:, None, 1])
    gap_z = torch.maximum(torch.maximum(box_lo[:, None, 2] - lz, torch.zeros_like(lz)), lz - box_hi[:, None, 2])
    excess = torch.sqrt(gap_x * gap_x + gap_y * gap_y + gap_z * gap_z) - arm[:, None]
    bad = bad | (valid & (excess > 0.0))
    over = (hand_x.abs() <= half[:, 0:1]) & (hand_y.abs() <= half[:, 1:2])
    bad = bad | (valid & over & (hand_z < board))
    return bad.any(1) | bad_turn.any(1) | (n_reg > count - 1)


class StrokeCheck:
    """The paddle-first stroke check for one batch of candidate contacts."""

    def __init__(self, rules: Any) -> None:
        self.rules = rules
        h = rules.header
        dev, dt = rules.device, rules.dtype
        self.c = h["constants"]["stroke"]
        self.motion = [h["profiles"][role]["motion"] for role in ("suited", "opponent")]

        def t(x: Any) -> Tensor:
            return torch.as_tensor(x, dtype=dt, device=dev)

        a = rules.arrays
        self.strokes = []
        self.ready = []
        for role in ("suited", "opponent"):
            self.strokes.append(
                {f"{pose}_{kind}": t(a[f"stroke_{role}_{pose}_{kind}"]) for pose in ("contact", "back", "follow") for kind in ("offset", "quat")}
            )
            self.ready.append((t(a[f"ready_{role}_offset"]), t(a[f"ready_{role}_quat"])))
        self.grip_offset = torch.stack([t(m["grip_axes"]).T @ t(m["grip_centre"]) for m in self.motion])
        self.speed = t([m["speed_mps"] for m in self.motion])
        self.turn = t([m["turn_radps"] for m in self.motion])
        self.speed_in = t([m["speed_inside_mps"] for m in self.motion])
        self.turn_in = t([min(m["turn_inside_radps"], self.c["chain_turn_radps"]) for m in self.motion])
        self.reach = t([m["reach_m"] for m in self.motion])
        self.reach_p99 = t([m["reach_p99_m"] for m in self.motion])
        self.height = t([m["height_above_floor_m"] for m in self.motion])
        self.window = self.motion[0]["contact_window_s"] - TICK_S

    # ---- pieces of the body model ------------------------------------------------------------------------------

    def hands(self, profile: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """``motion.hands``: the hand from blade poses (``p`` ``(..., 3)``, ``q`` ``(..., 4)``) through the grip."""

        offset = self.grip_offset[profile].to(p.dtype)
        offset = offset.reshape(offset.shape[:1] + (1,) * (p.dim() - 2) + (3,))
        return p - (quat_to_matrix(q) @ offset.expand(p.shape)[..., None])[..., 0]

    def violation(
        self,
        profile: Tensor,
        times: Tensor,
        valid: Tensor,
        p: Tensor,
        v: Tensor,
        q: Tensor,
        pelvis: Tensor,
        t_c: Tensor,
        flip: Tensor,
        floor_z: Tensor,
        half: tuple[Tensor, Tensor],
    ) -> Tensor:
        """``motion.violation`` for samples ``(R, S)`` (masked by ``valid``): true where any limit is broken."""

        c = self.c
        inside = (times - t_c[:, None]).abs() <= self.window
        speed = torch.sqrt((v * v).sum(-1))
        bad = valid & ~inside & (speed > c["plan_speed_margin"] * self.speed[profile][:, None])
        bad |= valid & inside & (speed > self.speed_in[profile][:, None])
        dt = (times[:, 1:] - times[:, :-1]).clamp_min(1e-9)
        turn = quat_angles(q[:, 1:], q[:, :-1]) / dt
        pair = valid[:, 1:] & valid[:, :-1]
        outside = ~(inside[:, 1:] | inside[:, :-1])
        bad_turn = pair & ((outside & (turn > c["plan_speed_margin"] * self.turn[profile][:, None])) | (~outside & (turn > self.turn_in[profile][:, None])))
        reach = torch.sqrt(((p[..., 0:2] - pelvis) ** 2).sum(-1))
        bad |= valid & ((reach > c["plan_reach_margin"] * self.reach[profile][:, None]) | (~inside & (reach > self.reach_p99[profile][:, None])))
        height = p[..., 2] - floor_z[:, None]
        bad |= valid & ((height < self.height[profile, 0][:, None]) | (height > self.height[profile, 1][:, None]))
        hand = self.hands(profile, p, q)
        rows, s = times.shape
        excess = self.rules.arm_excess(profile[:, None].expand(rows, s), hand, pelvis, flip[:, None].expand(rows, s), self.rules.arm_reach[profile][:, None])
        bad |= valid & (excess > 0.0)
        over = (hand[..., 0].abs() <= half[0][:, None]) & (hand[..., 1].abs() <= half[1][:, None])
        bad |= valid & over & (hand[..., 2] < c["plan_board_clearance_m"])
        return bad.any(1) | bad_turn.any(1)

    # ---- ready poses -------------------------------------------------------------------------------------------

    def draw_ready(self, profile: Tensor, stance: Tensor, flip: Tensor, floor_z: Tensor, half: tuple[Tensor, Tensor], generator: Any = None) -> BladePlan:
        """``episode.draw_ready``: a recorded ready pose at the stance within the limits (up to 50 draws, else the last)."""

        r = profile.shape[0]
        k = int(self.c["ready_draws"])
        dev = stance.device
        offsets = torch.zeros((r, k, 3), dtype=stance.dtype, device=dev)
        quats = torch.zeros((r, k, 4), dtype=stance.dtype, device=dev)
        for p in (0, 1):
            rows = torch.nonzero(profile == p).flatten()
            if rows.numel() == 0:
                continue
            off, qu = self.ready[p]
            pick = torch.randint(0, off.shape[0], (rows.numel(), k), device=dev, generator=generator)
            offsets[rows] = off[pick]
            quats[rows] = qu[pick]
        flip_k = flip[:, None].expand(r, k)
        local = yaw_vectors(offsets, flip_k)
        height_m = self.rules.constants["pelvis_height_m"]
        position = torch.stack([stance[:, None, 0] + local[..., 0], stance[:, None, 1] + local[..., 1], height_m + local[..., 2]], -1)
        orientation = yaw_quaternions(quats, flip_k)
        profile_k = profile[:, None].expand(r, k)
        height = position[..., 2] - floor_z[:, None]
        reach = torch.sqrt(((position[..., 0:2] - stance[:, None, :]) ** 2).sum(-1))
        hand = self.hands(profile, position, orientation)
        within = self.rules.arm_excess(profile_k, hand, stance[:, None, :].expand(r, k, 2), flip_k, self.rules.arm_reach[profile][:, None]) <= 0.0
        over = (hand[..., 0].abs() <= half[0][:, None]) & (hand[..., 1].abs() <= half[1][:, None])
        clear = ~over | (hand[..., 2] >= self.c["plan_board_clearance_m"])
        ok = (
            (height >= self.height[profile, 0][:, None])
            & (height <= self.height[profile, 1][:, None])
            & (reach <= self.reach_p99[profile][:, None])
            & within
            & clear
        )
        index = torch.arange(k, device=dev)[None, :].expand(r, k)
        first = torch.where(ok, index, torch.full_like(index, k - 1)).min(1).values
        sel = first[:, None, None]
        return BladePlan.at_rest(position.gather(1, sel.expand(r, 1, 3))[:, 0], orientation.gather(1, sel.expand(r, 1, 4))[:, 0])

    # ---- the check ---------------------------------------------------------------------------------------------

    def check(
        self,
        profile: Tensor,
        flip: Tensor,
        begin: BladePlan,
        t_from: Tensor,
        t_c: Tensor,
        contact: BladePlan,
        pelvis: Any,
        ready: BladePlan,
        floor_z: Tensor,
        half: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Piece, Piece]:
        """
        Whether each row's stroke can be laid, and its follow-through and recovery pieces (for the blade plan after).

        :param pelvis: ``pelvis(rows (R',), times (R', S)) -> (R', S, 2)``: the hips of those rows over the stroke (with
            this stroke's approach and return moves).
        """

        r = profile.shape[0]
        dev, dtype = t_c.device, t_c.dtype
        c = self.c
        rate = c["rate_hz"]
        hips_h = self.rules.constants["pelvis_height_m"]

        everyone = torch.arange(r, device=dev)

        def hips(t: Tensor) -> Tensor:
            xy = pelvis(everyone, t[:, None])[:, 0]
            return torch.cat([xy, torch.full_like(xy[:, :1], hips_h)], -1)

        # The captured stroke nearest this contact (``capture_stroke``): offset from the hips plus 0.1 m per radian.
        local_contact = yaw_vectors(contact.p - hips(t_c), flip)
        local_quat = yaw_quaternions(contact.q, flip)
        pick = torch.zeros(r, dtype=torch.long, device=dev)
        chosen: dict[str, Tensor] = {}
        for p in (0, 1):
            rows = torch.nonzero(profile == p).flatten()
            if rows.numel() == 0:
                continue
            table = self.strokes[p]
            quats = table["contact_quat"]
            cosine = (local_quat[rows] @ quats.T).abs() / torch.sqrt((quats * quats).sum(-1))[None, :]
            angle = 2.0 * torch.acos(cosine.clamp(0.0, 1.0))
            cost = torch.cdist(local_contact[rows], table["contact_offset"]) + 0.1 * angle
            pick[rows] = torch.argmin(cost, 1)
            for key, value in table.items():
                if key not in chosen:
                    chosen[key] = torch.zeros((r, value.shape[1]), dtype=dtype, device=dev)
                chosen[key][rows] = value[pick[rows]]
        turn = quat_multiply(local_quat, quat_conjugate(chosen["contact_quat"]))

        def captured(pose: str, moved: bool, t: Tensor) -> BladePlan:
            offset = chosen[f"{pose}_offset"]
            quat = chosen[f"{pose}_quat"]
            if moved:
                offset = offset + (local_contact - chosen["contact_offset"])
                quat = quat_multiply(turn, quat)
            return BladePlan.at_rest(hips(t) + yaw_vectors(offset, flip), yaw_quaternions(quat, flip))

        speed = torch.sqrt((contact.v * contact.v).sum(-1))
        direction = torch.where((speed > 1e-9)[:, None], contact.v / speed.clamp_min(1e-9)[:, None], torch.zeros_like(contact.v))

        def straight(sign: float, length: float) -> BladePlan:
            return BladePlan.at_rest(
                contact.p + sign * direction * (0.5 * speed * length)[:, None], quat_multiply(quat_from_rotvec(sign * 0.5 * length * contact.w), contact.q)
            )

        def broken(pieces: list[Piece], sub: Tensor) -> Tensor:
            need = torch.ones(sub.numel(), dtype=torch.bool, device=dev)
            start, end = pieces[0].start, pieces[-1].end
            span = (end - start).clamp_min(0.0)
            count = int(torch.ceil(span.max() * rate).item()) + 2 if span.numel() else 2
            k = torch.arange(count, device=dev, dtype=dtype)[None, :]
            n_reg = torch.ceil(span * rate - 1e-9).clamp_min(0)
            times = start[:, None] + k / rate
            last = k == n_reg[:, None]
            times = torch.where(last, end[:, None], times)
            valid = (k <= n_reg[:, None]) & need[:, None]
            p_all = torch.zeros((*times.shape, 3), dtype=dtype, device=dev)
            v_all = torch.zeros_like(p_all)
            q_all = torch.zeros((*times.shape, 4), dtype=dtype, device=dev)
            q_all[..., 0] = 1.0
            for i, piece in enumerate(pieces):
                lo = piece.start[:, None]
                inside = times >= lo if i > 0 else torch.ones_like(valid)
                if i + 1 < len(pieces):
                    inside = inside & (times < pieces[i + 1].start[:, None])
                pp, vv, qq = piece.sample(times)
                p_all = torch.where(inside[..., None], pp, p_all)
                v_all = torch.where(inside[..., None], vv, v_all)
                q_all = torch.where(inside[..., None], qq, q_all)
            pel = pelvis(sub, times)
            return self.violation(profile[sub], times, valid, p_all, v_all, q_all, pel, t_c[sub], flip[sub], floor_z[sub], (half[0][sub], half[1][sub]))

        contact_plan = BladePlan(contact.p, contact.v, torch.zeros_like(contact.p), contact.q, contact.w)
        # The lead: reposition to a backswing, then the swing.
        lead_ok = torch.zeros(r, dtype=torch.bool, device=dev)
        candidates: list[tuple[float, Any]] = [
            (c["swing_captured_s"], lambda s: captured("back", True, t_c - s)),
            (c["swing_captured_s"], lambda s: captured("back", False, t_c - s)),
            *[(s, (lambda s2: captured("back", True, t_c - s2))) for s in (0.25, 0.15, 0.12)],
            *[(s, (lambda s2: straight(-1.0, s2))) for s in durations(c["swing_s"], c["swing_min_s"])],
        ]
        for swing_s, make in candidates:
            todo = ~lead_ok & (t_c - swing_s > t_from + TICK_S)
            if not bool(todo.any()):
                continue
            rows = torch.nonzero(todo).flatten()
            t_swing = (t_c - swing_s)[rows]
            back = make(swing_s).take(rows)
            pieces = [
                Piece.between(t_from[rows], t_swing, begin.take(rows), back),
                Piece.between(t_swing, t_c[rows], back, contact_plan.take(rows)),
            ]
            fails = self._sub(broken, pieces, rows)
            lead_ok[rows] = ~fails
        rest = torch.nonzero(~lead_ok).flatten()
        if rest.numel():
            fails = self._sub(broken, [Piece.between(t_from[rest], t_c[rest], begin.take(rest), contact_plan.take(rest))], rest)
            lead_ok[rest] = ~fails
        # The follow-through.
        follow_ok = torch.zeros(r, dtype=torch.bool, device=dev)
        follow_end = torch.full_like(t_c, float("nan"))
        follow_pose = BladePlan.at_rest(contact.p.clone(), contact.q.clone())
        options: list[tuple[float, Any]] = [
            (c["follow_captured_s"], lambda s: captured("follow", True, t_c + s)),
            (c["follow_captured_s"], lambda s: captured("follow", False, t_c + s)),
            *[(s, (lambda s2: captured("follow", True, t_c + s2))) for s in (0.3, 0.18, 0.15)],
            *[(s, (lambda s2: straight(1.0, s2))) for s in durations(c["follow_s"], c["follow_min_s"])],
        ]
        for follow_s, make in options:
            todo = lead_ok & ~follow_ok
            if not bool(todo.any()):
                break
            rows = torch.nonzero(todo).flatten()
            pose = make(follow_s).take(rows)
            piece = Piece.between(t_c[rows], t_c[rows] + follow_s, contact_plan.take(rows), pose)
            ok = ~self._sub(broken, [piece], rows)
            good = rows[ok]
            follow_ok[good] = True
            follow_end[good] = t_c[good] + follow_s
            follow_pose.p[good] = pose.p[ok]
            follow_pose.q[good] = pose.q[ok]
        # The recovery to the ready pose.
        recover_ok = torch.zeros(r, dtype=torch.bool, device=dev)
        recover_end = torch.full_like(t_c, float("nan"))
        recover_s = c["recover_s"]
        while recover_s <= c["recover_max_s"] + 1e-9:
            todo = follow_ok & ~recover_ok
            if not bool(todo.any()):
                break
            rows = torch.nonzero(todo).flatten()
            piece = Piece.between(follow_end[rows], follow_end[rows] + recover_s, follow_pose.take(rows), ready.take(rows))
            ok = ~self._sub(broken, [piece], rows)
            recover_ok[rows[ok]] = True
            recover_end[rows[ok]] = follow_end[rows[ok]] + recover_s
            recover_s *= 1.25
        ok = lead_ok & follow_ok & recover_ok
        follow = Piece.between(t_c, torch.where(ok, follow_end, t_c + 0.2), contact_plan, follow_pose)
        recover = Piece.between(follow.end, torch.where(ok, recover_end, follow.end + 0.5), follow_pose, ready)
        return ok, follow, recover

    # ---- the dense (graph-capturable) check ----------------------------------------------------------------

    LEAD_S = 1.6
    """Longest lead (decision tick to contact) the dense check samples, s; longer leads are rejected. Measured on
    30,129 attempts (O1 receives, O2 rallies, robot balls): maximum 1.30 s, none above 1.5 s."""
    FOLLOW_S = 0.31
    RECOVER_S = 1.53

    def draw_ready_dense(self, profile: Tensor, stance: Tensor, flip: Tensor, floor_z: Tensor, half: tuple[Tensor, Tensor], generator: Any = None) -> BladePlan:
        """:meth:`draw_ready` without row compaction (both profiles' pools drawn, the row's own selected)."""

        r = profile.shape[0]
        k = int(self.c["ready_draws"])
        dev = stance.device
        offsets = torch.zeros((r, k, 3), dtype=stance.dtype, device=dev)
        quats = torch.zeros((r, k, 4), dtype=stance.dtype, device=dev)
        for p in (0, 1):
            off, qu = self.ready[p]
            u = torch.rand((r, k), dtype=stance.dtype, device=dev, generator=generator)
            pick = (u * off.shape[0]).long().clamp(max=off.shape[0] - 1)
            mine = (profile == p)[:, None, None]
            offsets = torch.where(mine, off[pick], offsets)
            quats = torch.where(mine, qu[pick], quats)
        return self._accept_ready(profile, stance, flip, floor_z, half, offsets, quats)

    def _accept_ready(
        self, profile: Tensor, stance: Tensor, flip: Tensor, floor_z: Tensor, half: tuple[Tensor, Tensor], offsets: Tensor, quats: Tensor
    ) -> BladePlan:
        r, k = offsets.shape[0], offsets.shape[1]
        dev = stance.device
        flip_k = flip[:, None].expand(r, k)
        local = yaw_vectors(offsets, flip_k)
        height_m = self.rules.constants["pelvis_height_m"]
        position = torch.stack([stance[:, None, 0] + local[..., 0], stance[:, None, 1] + local[..., 1], height_m + local[..., 2]], -1)
        orientation = yaw_quaternions(quats, flip_k)
        profile_k = profile[:, None].expand(r, k)
        height = position[..., 2] - floor_z[:, None]
        reach = torch.sqrt(((position[..., 0:2] - stance[:, None, :]) ** 2).sum(-1))
        hand = self.hands(profile, position, orientation)
        within = self.rules.arm_excess(profile_k, hand, stance[:, None, :].expand(r, k, 2), flip_k, self.rules.arm_reach[profile][:, None]) <= 0.0
        over = (hand[..., 0].abs() <= half[0][:, None]) & (hand[..., 1].abs() <= half[1][:, None])
        clear = ~over | (hand[..., 2] >= self.c["plan_board_clearance_m"])
        ok = (
            (height >= self.height[profile, 0][:, None])
            & (height <= self.height[profile, 1][:, None])
            & (reach <= self.reach_p99[profile][:, None])
            & within
            & clear
        )
        index = torch.arange(k, device=dev)[None, :].expand(r, k)
        first = torch.where(ok, index, torch.full_like(index, k - 1)).min(1).values
        sel = first[:, None, None]
        return BladePlan.at_rest(position.gather(1, sel.expand(r, 1, 3))[:, 0], orientation.gather(1, sel.expand(r, 1, 4))[:, 0])

    def limits_of(self, profile: Tensor, dtype: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Per-row limits for :func:`broken_core`: ``(limits (R, 8), grip offset (R, 3), box low, box high, arm reach)``."""

        c = self.c
        lim = torch.stack(
            [
                c["plan_speed_margin"] * self.speed[profile],
                self.speed_in[profile],
                c["plan_speed_margin"] * self.turn[profile],
                self.turn_in[profile],
                c["plan_reach_margin"] * self.reach[profile],
                self.reach_p99[profile],
                self.height[profile, 0],
                self.height[profile, 1],
            ],
            -1,
        )
        rules = self.rules
        return (
            lim.to(dtype),
            self.grip_offset[profile].to(dtype),
            rules.shoulder_low[profile].to(dtype),
            rules.shoulder_high[profile].to(dtype),
            rules.arm_reach[profile].to(dtype),
        )

    def check_dense(
        self,
        profile: Tensor,
        flip: Tensor,
        begin: BladePlan,
        t_from: Tensor,
        t_c: Tensor,
        contact: BladePlan,
        slots: tuple[Tensor, Tensor, Tensor, Tensor],
        ready: BladePlan,
        floor_z: Tensor,
        half: tuple[Tensor, Tensor],
        *,
        chunk: int = 8192,
        sample_dtype: Any = None,
    ) -> tuple[Tensor, Piece, Piece]:
        """
        :meth:`check` with every candidate evaluated for every row (masks, a fixed sample grid, rows in fixed chunks):
        no host synchronisation, so it can be captured in a CUDA graph. The hips over the stroke are given as move
        ``slots`` (:func:`lib.opponent.rally.pelvis_eval`), this stroke's approach and return
        included. The 120 Hz limit checks (:func:`broken_core`, compiled) run in ``sample_dtype`` (float32 by default)
        on times relative to each row's decision tick. Leads longer than :attr:`LEAD_S` are rejected.
        """

        from .ballistics import compiled
        from .rally import pelvis_eval

        r = profile.shape[0]
        dev, _dtype = t_c.device, t_c.dtype
        sdt = sample_dtype or torch.float32
        c = self.c
        rate = c["rate_hz"]
        hips_h = self.rules.constants["pelvis_height_m"]
        s_start, s_end, s_coeffs, s_rest = slots

        def hips(t: Tensor) -> Tensor:
            xy, _v = pelvis_eval(s_start, s_end, s_coeffs, s_rest, t)
            return torch.cat([xy, torch.full_like(xy[:, :1], hips_h)], -1)

        local_contact = yaw_vectors(contact.p - hips(t_c), flip)
        local_quat = yaw_quaternions(contact.q, flip)
        chosen: dict[str, Tensor] = {}
        for p in (0, 1):
            table = self.strokes[p]
            quats = table["contact_quat"]
            cosine = (local_quat @ quats.T).abs() / torch.sqrt((quats * quats).sum(-1))[None, :]
            angle = 2.0 * torch.acos(cosine.clamp(0.0, 1.0))
            pick = torch.argmin(torch.cdist(local_contact, table["contact_offset"]) + 0.1 * angle, 1)
            mine = (profile == p)[:, None]
            for key, value in table.items():
                chosen[key] = torch.where(mine, value[pick], chosen[key]) if key in chosen else value[pick]
        turn = quat_multiply(local_quat, quat_conjugate(chosen["contact_quat"]))

        def captured(pose: str, moved: bool, t: Tensor) -> BladePlan:
            offset = chosen[f"{pose}_offset"]
            quat = chosen[f"{pose}_quat"]
            if moved:
                offset = offset + (local_contact - chosen["contact_offset"])
                quat = quat_multiply(turn, quat)
            return BladePlan.at_rest(hips(t) + yaw_vectors(offset, flip), yaw_quaternions(quat, flip))

        speed = torch.sqrt((contact.v * contact.v).sum(-1))
        direction = torch.where((speed > 1e-9)[:, None], contact.v / speed.clamp_min(1e-9)[:, None], torch.zeros_like(contact.v))

        def straight(sign: float, length: float) -> BladePlan:
            return BladePlan.at_rest(
                contact.p + sign * direction * (0.5 * speed * length)[:, None], quat_multiply(quat_from_rotvec(sign * 0.5 * length * contact.w), contact.q)
            )

        # Everything the limit check reads, in sample precision and relative to each row's decision tick.
        origin = t_from
        lim, grip, box_lo, box_hi, arm = self.limits_of(profile, sdt)
        rel_slots = ((s_start - origin[:, None]).to(sdt), (s_end - origin[:, None]).to(sdt), _rebase(s_coeffs, s_start, origin).to(sdt), s_rest.to(sdt))
        sign = torch.where(flip, -torch.ones_like(t_c), torch.ones_like(t_c)).to(sdt)
        rel_tc = (t_c - origin).to(sdt)
        floor_s = floor_z.to(sdt)
        half_s = torch.stack([half[0], half[1]], -1).to(sdt)
        core = compiled(broken_core)

        def broken(pieces: list[Piece], longest: float, times_each: int = 1) -> Tensor:
            count = math.ceil(longest * rate) + 2
            k = torch.arange(count, device=dev, dtype=sdt)
            base = origin.repeat_interleave(times_each)
            starts = torch.stack([(p.start - base) for p in pieces], -1).to(sdt)
            ends = torch.stack([(p.end - base) for p in pieces], -1).to(sdt)
            pos = torch.stack([p.position for p in pieces], 1).to(sdt)
            rot = torch.stack([p.rotation for p in pieces], 1).to(sdt)
            ref = torch.stack([p.reference for p in pieces], 1).to(sdt)
            per_row = [rel_tc, *rel_slots, lim, grip, box_lo, box_hi, arm, sign, floor_s, half_s]
            if times_each > 1:
                per_row = [x.repeat_interleave(times_each, 0) for x in per_row]
            (tc_r, s0_r, s1_r, s2_r, s3_r, lim_r, grip_r, lo_r, hi_r, arm_r, sign_r, floor_r, half_r) = per_row
            rows_all = starts.shape[0]
            out = []
            for lo in range(0, rows_all, chunk):
                sl = slice(lo, lo + chunk)
                out.append(
                    core(
                        k,
                        starts[sl],
                        ends[sl],
                        pos[sl],
                        rot[sl],
                        ref[sl],
                        tc_r[sl],
                        s0_r[sl],
                        s1_r[sl],
                        s2_r[sl],
                        s3_r[sl],
                        lim_r[sl],
                        grip_r[sl],
                        lo_r[sl],
                        hi_r[sl],
                        arm_r[sl],
                        sign_r[sl],
                        floor_r[sl],
                        half_r[sl],
                        float(rate),
                        float(self.window),
                        float(c["plan_board_clearance_m"]),
                        float(hips_h),
                    )
                )
            return torch.cat(out)

        contact_plan = BladePlan(contact.p, contact.v, torch.zeros_like(contact.p), contact.q, contact.w)

        def batch(poses: list[BladePlan]) -> BladePlan:
            """Candidates side by side: rows ``(R * C,)``, candidate-minor."""

            def st(name: str) -> Tensor:
                return torch.stack([getattr(pose, name) for pose in poses], 1).reshape(-1, getattr(poses[0], name).shape[-1])

            return BladePlan(st("p"), st("v"), st("a"), st("q"), st("w"))

        def rep(x: Tensor, count: int) -> Tensor:
            return x.repeat_interleave(count, 0)

        # The lead: every backswing candidate at once (a reposition then the swing), then one straight swing.
        lead = [
            (c["swing_captured_s"], captured("back", True, t_c - c["swing_captured_s"])),
            (c["swing_captured_s"], captured("back", False, t_c - c["swing_captured_s"])),
            *[(sw, captured("back", True, t_c - sw)) for sw in (0.25, 0.15, 0.12)],
            *[(sw, straight(-1.0, sw)) for sw in durations(c["swing_s"], c["swing_min_s"])],
        ]
        n_lead = len(lead)
        swing_s = torch.stack([torch.full_like(t_c, sw) for sw, _pose in lead], 1)
        valid = (t_c[:, None] - swing_s) > (t_from[:, None] + TICK_S)
        t_swing = torch.where(valid, t_c[:, None] - swing_s, (t_from + 0.5 * (t_c - t_from))[:, None]).reshape(-1)
        backs = batch([pose for _sw, pose in lead])
        lead_pieces = [
            Piece.between(rep(t_from, n_lead), t_swing, _rep_plan(begin, n_lead), backs),
            Piece.between(t_swing, rep(t_c, n_lead), backs, _rep_plan(contact_plan, n_lead)),
        ]
        lead_ok = (valid & ~broken(lead_pieces, self.LEAD_S, n_lead).reshape(r, n_lead)).any(1)
        lead_ok = lead_ok | ~broken([Piece.between(t_from, t_c, begin, contact_plan)], self.LEAD_S, 1)
        # The follow-through: the first candidate within the limits.
        options = [
            (c["follow_captured_s"], captured("follow", True, t_c + c["follow_captured_s"])),
            (c["follow_captured_s"], captured("follow", False, t_c + c["follow_captured_s"])),
            *[(fs, captured("follow", True, t_c + fs)) for fs in (0.3, 0.18, 0.15)],
            *[(fs, straight(1.0, fs)) for fs in durations(c["follow_s"], c["follow_min_s"])],
        ]
        n_follow = len(options)
        follow_s = torch.stack([torch.full_like(t_c, fs) for fs, _pose in options], 1)
        poses = batch([pose for _fs, pose in options])
        ends = (t_c[:, None] + follow_s).reshape(-1)
        follow_fine = ~broken([Piece.between(rep(t_c, n_follow), ends, _rep_plan(contact_plan, n_follow), poses)], self.FOLLOW_S, n_follow).reshape(r, n_follow)
        follow_ok = follow_fine.any(1)
        index = torch.arange(n_follow, device=dev)[None, :].expand(r, n_follow)
        pick = torch.where(follow_fine, index, torch.full_like(index, n_follow)).min(1).values.clamp(max=n_follow - 1)
        flat = torch.arange(r, device=dev) * n_follow + pick
        follow_end = torch.where(follow_ok, ends[flat], t_c + 0.2)
        follow_pose = BladePlan.at_rest(torch.where(follow_ok[:, None], poses.p[flat], contact.p), torch.where(follow_ok[:, None], poses.q[flat], contact.q))
        # The recovery to the ready pose: the first duration within the limits.
        lengths = []
        recover_s = c["recover_s"]
        while recover_s <= c["recover_max_s"] + 1e-9:
            lengths.append(recover_s)
            recover_s *= 1.25
        n_rec = len(lengths)
        rec_s = torch.stack([torch.full_like(t_c, x) for x in lengths], 1)
        rec_end = (follow_end[:, None] + rec_s).reshape(-1)
        rec_fine = ~broken(
            [Piece.between(rep(follow_end, n_rec), rec_end, _rep_plan(follow_pose, n_rec), _rep_plan(ready, n_rec))], self.RECOVER_S, n_rec
        ).reshape(r, n_rec)
        recover_ok = follow_ok & rec_fine.any(1)
        index = torch.arange(n_rec, device=dev)[None, :].expand(r, n_rec)
        pick = torch.where(rec_fine, index, torch.full_like(index, n_rec)).min(1).values.clamp(max=n_rec - 1)
        recover_end = torch.where(recover_ok, rec_end[torch.arange(r, device=dev) * n_rec + pick], follow_end + 0.5)
        ok = lead_ok & follow_ok & recover_ok
        follow = Piece.between(t_c, follow_end, contact_plan, follow_pose)
        recover = Piece.between(follow_end, recover_end, follow_pose, ready)
        return ok, follow, recover

    @staticmethod
    def _sub(broken: Any, pieces: list[Piece], rows: Tensor, chunk: int = 8192) -> Tensor:
        """``broken`` over ``rows`` (pieces already restricted to them), in chunks (bounded memory)."""

        if rows.numel() <= chunk:
            return broken(pieces, rows)
        out = []
        for start in range(0, rows.numel(), chunk):
            idx = torch.arange(start, min(start + chunk, rows.numel()), device=rows.device)
            out.append(broken([p.take(idx) for p in pieces], rows[idx]))
        return torch.cat(out)


__all__ = ["BladePlan", "Piece", "StrokeCheck", "durations", "quintic"]
