"""
What the robot should do, frame by frame (50 Hz): the human's motion mapped onto the robot's proportions, and the
paddle path it must hold through every contact.

**Segments.** A segment is one rally of the player's contacts (at most :data:`RALLY_GAP_S` apart, inside one
continuous capture stretch), padded by :data:`PAD_S` either side. Each segment is solved on its own.

**The joint map** (human key joint -> robot body):

============================  ==========================================================================
``Hip``                       pelvis: its horizontal path unscaled (contacts keep their place at the
                              table), its height scaled by the leg ratio, its orientation rest-relative
``Spine02``                   torso orientation, relative to the rest pose
``L_Foot``, ``L_ToeBase``     left foot: ankle placed about the pelvis at the leg ratio, yaw heel to toe
``R_Foot``, ``R_ToeBase``     right foot, the same
``L_Hand`` - ``L_Upperarm``   the free hand about its shoulder, in the torso frame, at the arm ratio
the capture's paddle          the paddle body (corrected at contacts, below)
============================  ==========================================================================

**The paddle path.** Away from contacts it is the capture's paddle. At each contact it is the paddle state solved from
the ball: the position correction's slow part (a 15 s Gaussian regression, horizontal) moves the whole body as
placement error and the rest moves the blade; a minimal rotation takes the capture's face normal onto the solved one;
both are held over +-:data:`HOLD_S` and blended between contacts; a velocity bump makes the path pass each contact at
the solved face velocity.

**Reach.** Where the robot's shorter arm cannot hold the blade through a contact's window from the human's stance, the
body moves by the static reach solve's offset (held around the contact, blended between contacts), and a planted foot
whose stance moves more than 5 cm takes a step rather than sliding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from lib.retarget.model import RobotModel, body_poses
from lib.retarget.source import FLOOR_Z_M, HumanSession, HumanStrokes, Stroke, rz

FPS = 50.0
RALLY_GAP_S = 3.0
PAD_S = 1.5
HOLD_S = 0.10  # the paddle is held on its path +-0.1 s around every contact (and graded there)
WINDOW_MARGIN_S = 0.02  # the solver's tight paddle window is the hold plus one frame
VEL_BUMP_S = 0.25
SHIFT_SIGMA_S = 15.0
STANCE_HOLD_S = 0.30  # a contact's body offset is held this long either side of it
REACH_OFFSETS_S = (-0.10, -0.05, 0.0, 0.05, 0.10)  # the instants the static reach covers through the hold
STEP_MIN_M = 0.05  # a planted foot whose stance moves more than this steps
STEP_HALF_FRAMES = 6  # an inserted step lasts 2 x 6 frames (0.24 s)
STEP_LIFT_M = 0.05


@dataclass
class Segment:
    session: int
    t: np.ndarray  # (T,) camera clock, 50 Hz
    strokes: list[Stroke]


@dataclass
class Targets:
    """Per-frame targets of one segment (table frame)."""

    t: np.ndarray
    pelvis_p: np.ndarray  # (T, 3)
    pelvis_r: np.ndarray  # (T, 3, 3)
    torso_r: np.ndarray  # (T, 3, 3)
    foot_p: np.ndarray  # (T, 2, 3) foot-link origins
    foot_yaw: np.ndarray  # (T, 2)
    planted: np.ndarray  # (T, 2) bool
    lhand_rel: np.ndarray  # (T, 3) free hand about its shoulder, torso frame
    paddle_p: np.ndarray  # (T, 3) the paddle path: blade centre
    paddle_r: np.ndarray  # (T, 3, 3) paddle frame (x tip, y forehand normal, z across)
    near: np.ndarray  # (T,) 0..1, 1 at contacts, 0 beyond +-0.35 s
    contacts: list[dict]  # per contact: t, p, n (struck), v, face, stroke, and the capture's own paddle there


@dataclass(frozen=True)
class Scales:
    leg: float  # robot / human leg (hip joint to ankle)
    arm: float  # robot / human arm (shoulder to wrist)
    hip_drop: float  # pelvis origin above its hip joints, m
    sole_z: float  # the sole below the foot-link origin, m


def segments(src: HumanSession) -> list[Segment]:
    """The session's rallies (see the module docstring)."""
    out: list[Segment] = []
    st = src.strokes
    clip_of = lambda t: int(src.clip[int(np.clip(np.searchsorted(src.t, t), 0, len(src.t) - 1))])  # noqa: E731
    groups: list[list[Stroke]] = []
    for i, s in enumerate(st):
        if i and s.t - st[i - 1].t <= RALLY_GAP_S and clip_of(s.t) == clip_of(st[i - 1].t):
            groups[-1].append(s)
        else:
            groups.append([s])
    for g in groups:
        in_clip = src.t[src.clip == clip_of(g[0].t)]
        t0, t1 = max(g[0].t - PAD_S, in_clip[0] + 0.05), min(g[-1].t + PAD_S, in_clip[-1] - 0.05)
        out.append(Segment(session=src.session, t=t0 + np.arange(int(np.floor((t1 - t0) * FPS)) + 1) / FPS, strokes=g))
    return out


def scales(model: RobotModel, human: HumanStrokes) -> Scales:
    """The robot's proportions against the human's, from both rest poses."""
    _r, p = body_poses(model, np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, model.nq)))
    hip, ankle = p[0, model.hips[0]], p[0, model.feet[0]]
    sh, el, wr = (p[0, i] for i in model.racket_arm)
    return Scales(
        leg=float(hip[2] - ankle[2]) / human.rest_lengths["leg_m"],
        arm=float(np.linalg.norm(el - sh) + np.linalg.norm(wr - el)) / human.rest_lengths["arm_m"],
        hip_drop=float(-hip[2]),
        sole_z=float(model.sole[:, 2].min()),
    )


# ------------------------------------------------------------------------------------------------- helpers


def _interp(t: np.ndarray, tk: np.ndarray, v: np.ndarray) -> np.ndarray:
    flat = v.reshape(len(tk), -1)
    return np.stack([np.interp(t, tk, flat[:, k]) for k in range(flat.shape[1])], -1).reshape(len(t), *v.shape[1:])


def _interp_rot(t: np.ndarray, tk: np.ndarray, r: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(_interp(t, tk, r))
    u[..., :, 2] *= np.sign(np.linalg.det(u @ vt))[..., None]
    return u @ vt


def min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation vector of the minimal rotation taking unit ``a`` onto unit ``b``."""
    c = np.cross(a, b)
    s = np.linalg.norm(c)
    return np.zeros(3) if s < 1e-12 else c / s * np.arctan2(s, float(a @ b))


def blended(t: np.ndarray, tc: np.ndarray, values: np.ndarray, hold: float) -> np.ndarray:
    """Values held over +-hold around each tc and smoothstep-blended between consecutive ones."""
    out = np.zeros((len(t), values.shape[1]))
    if len(tc) == 0:
        return out
    order = np.argsort(tc)
    tc, values = tc[order], values[order]
    out[:] = values[0]
    for i in range(len(tc) - 1):
        a, b = tc[i] + hold, tc[i + 1] - hold
        if b <= a:
            a = b = 0.5 * (tc[i] + tc[i + 1])
        u = np.clip(np.where(t <= a, 0.0, np.where(t >= b, 1.0, (t - a) / max(b - a, 1e-9))), 0.0, 1.0)
        m = t > tc[i]
        s = u[m] * u[m] * (3 - 2 * u[m])
        out[m] = values[i] * (1 - s)[:, None] + values[i + 1] * s[:, None]
    return out


def bump(x: np.ndarray, half: float) -> np.ndarray:
    """1 at 0 with zero slope, 0 beyond +-half (a raised cosine)."""
    return np.where(np.abs(x) < half, 0.5 * (1 + np.cos(np.pi * x / half)), 0.0)


def placement_shift(src: HumanSession, t: np.ndarray) -> np.ndarray:
    """(n, 3) the slow, horizontal part of (solved - captured) blade centre at the session's solved contacts."""
    solved = [s for s in src.strokes if s.source == "paddle_inverse"]
    if not solved:
        return np.zeros((len(t), 3))
    tc = np.array([s.t for s in solved])
    d = np.stack([s.blade_p - s.donor_blade_p for s in solved])
    w = np.exp(-0.5 * ((tc[:, None] - tc[None]) / SHIFT_SIGMA_S) ** 2)
    smooth = (w @ d) / w.sum(1, keepdims=True)
    smooth[:, 2] = 0.0
    return _interp(np.asarray(t, dtype=np.float64), tc, smooth)


# ------------------------------------------------------------------------------------------------- the paddle path


def contacts(src: HumanSession, strokes: list[Stroke]) -> list[dict]:
    """Each contact's paddle state (``p``, struck normal ``n``, ``v``) and the capture's own paddle there (``raw_*``,
    moved by the placement shift). A ``capture`` stroke has no solved state: its target is the raw one."""
    out = []
    for s in strokes:
        dp, dr, dv = src.paddle_at([s.t])
        raw = {"raw_p": dp[0] + placement_shift(src, np.array([s.t]))[0], "raw_n": dr[0][:, 2] * (1.0 if s.face == 0 else -1.0), "raw_v": dv[0]}
        state = {"p": s.blade_p, "n": s.normal, "v": s.vel} if s.source == "paddle_inverse" else {"p": raw["raw_p"], "n": raw["raw_n"], "v": raw["raw_v"]}
        out.append({"t": s.t, "face": s.face, "stroke": s, "donor_p": dp[0], "donor_v": dv[0]} | state | raw)
    return out


def paddle_path(src: HumanSession, cts: list[dict], t: np.ndarray, raw: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The paddle path at times ``t``: blade centre (n, 3) and paddle frame (n, 3, 3). ``raw``: the capture's own paddle,
    uncorrected (the negative control)."""
    t = np.asarray(t, dtype=np.float64)
    pp, pr, _pv = src.paddle_at(t)
    pp = pp + placement_shift(src, t)
    paddle_r = np.stack([pr[..., 1], pr[..., 2], pr[..., 0]], -1)  # capture axes (X across, Y tip, Z forehand) -> paddle frame
    if raw or not cts:
        return pp, paddle_r
    tc = np.array([c["t"] for c in cts])
    corr_p = blended(t, tc, np.array([c["p"] - c["raw_p"] for c in cts]), HOLD_S)
    corr_r = blended(t, tc, np.array([min_rotation(c["raw_n"], c["n"]) for c in cts]), HOLD_S)
    order = np.argsort(tc)
    gaps = np.diff(np.r_[-np.inf, tc[order], np.inf])
    for rank, j in enumerate(order):  # the blend is flat at each contact, so the path passes it at the solved velocity
        half = min(VEL_BUMP_S, 0.45 * min(gaps[rank], gaps[rank + 1]))
        corr_p += ((cts[j]["v"] - cts[j]["donor_v"])[None] * (t - tc[j])[:, None]) * bump(t - tc[j], half)[:, None]
    return pp + corr_p, Rotation.from_rotvec(corr_r).as_matrix() @ paddle_r


# ------------------------------------------------------------------------------------------------- the body


def _rest_relative(human: HumanStrokes, joint: str, r: np.ndarray) -> np.ndarray:
    """A joint's orientation relative to its rest pose, with the rest (facing -Y) mapped to the robot facing -Y."""
    return r @ human.rest_rot[joint].T @ rz(-np.pi / 2)


def _pelvis(src: HumanSession, human: HumanStrokes, sc: Scales, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pelvis target before any reach offset: (n, 3) position, (n, 3, 3) orientation."""
    hips = _interp(t, src.t, src.pos["Hip"]) + placement_shift(src, t)
    hip_joint_z = 0.5 * (_interp(t, src.t, src.pos["L_Thigh"])[:, 2] + _interp(t, src.t, src.pos["R_Thigh"])[:, 2]) - FLOOR_Z_M
    z = FLOOR_Z_M + (hip_joint_z - human.rest_lengths["ankle_height_m"]) * sc.leg - sc.sole_z + sc.hip_drop
    return np.c_[hips[:, :2], z], _rest_relative(human, "Hip", _interp_rot(t, src.t, src.rot["Hip"]))


def reach_problem(src: HumanSession, human: HumanStrokes, sc: Scales) -> dict[str, np.ndarray]:
    """Per stroke of the session, the paddle path and the pelvis target at :data:`REACH_OFFSETS_S` around its contact:
    the static reach solve's inputs (``optimize.solve_reach``)."""
    st = src.strokes
    tt = np.array([s.t for s in st])[:, None] + np.array(REACH_OFFSETS_S)[None]
    p, r = paddle_path(src, contacts(src, st), tt.ravel())
    face = np.array([s.face for s in st])
    n = r[:, :, 1] * np.repeat(np.where(face == 0, 1.0, -1.0), tt.shape[1])[:, None]
    pel_p, pel_r = _pelvis(src, human, sc, tt.ravel())
    k, s = tt.shape
    return {"p": p.reshape(k, s, 3), "n": n.reshape(k, s, 3), "face": face, "pelvis_p": pel_p.reshape(k, s, 3), "pelvis_r": pel_r.reshape(k, s, 3, 3)}


def insert_steps(planted: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Split a foot's plant where its stance offset moves more than :data:`STEP_MIN_M`: the foot lifts over 2 x
    :data:`STEP_HALF_FRAMES` frames around the move (the right a step after the left, never both feet up). Returns the
    new plants and the inserted steps (foot, first frame, end frame)."""
    out, steps = planted.copy(), []
    for s in (0, 1):
        for _guard in range(200):
            for a, b in _runs(out[:, s]):
                move = np.linalg.norm(delta[a:b] - delta[a], axis=1)
                if move.max() <= STEP_MIN_M:
                    continue
                m = a + int(np.argmax(move >= 0.5 * move.max()))
                if s == 1:
                    m = min(m + 2 * STEP_HALF_FRAMES, b - 1)
                lo, hi = max(a, m - STEP_HALF_FRAMES), min(b, m + STEP_HALF_FRAMES)
                if out[lo:hi, 1 - s].all():
                    out[lo:hi, s] = False
                    steps.append((s, lo, hi))
                    break
            else:
                break
    return out, steps


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.nonzero(edges == 1)[0].tolist(), np.nonzero(edges == -1)[0].tolist(), strict=True))


def build_targets(src: HumanSession, seg: Segment, human: HumanStrokes, sc: Scales, reach_xy: dict[str, np.ndarray]) -> Targets:
    """One segment's targets (see the module docstring); ``reach_xy`` is the static reach's body offset per stroke id."""
    t = seg.t
    shift = placement_shift(src, t)
    pts = {k: _interp(t, src.t, src.pos[k]) + shift for k in ("Hip", "L_Foot", "R_Foot", "L_ToeBase", "R_ToeBase")}
    cts = contacts(src, seg.strokes)
    paddle_p, paddle_r = paddle_path(src, cts, t)
    pelvis_p, pelvis_r = _pelvis(src, human, sc, t)
    delta = blended(t, np.array([s.t for s in seg.strokes]), np.array([reach_xy[s.id] for s in seg.strokes]), STANCE_HOLD_S)
    body = pelvis_p[:, :2] + delta
    pelvis_p = np.c_[body, pelvis_p[:, 2]]
    foot_p, foot_yaw = np.zeros((len(t), 2, 3)), np.zeros((len(t), 2))
    ankle_h = human.rest_lengths["ankle_height_m"]
    for f, side in enumerate("LR"):
        ankle, toe = pts[f"{side}_Foot"], pts[f"{side}_ToeBase"]
        foot_p[:, f, :2] = body + sc.leg * (ankle[:, :2] - pts["Hip"][:, :2])
        foot_p[:, f, 2] = FLOOR_Z_M - sc.sole_z + sc.leg * np.maximum(ankle[:, 2] - FLOOR_Z_M - ankle_h, 0.0)
        foot_yaw[:, f] = np.arctan2(toe[:, 1] - ankle[:, 1], toe[:, 0] - ankle[:, 0])
    k = np.clip(np.searchsorted(src.t, t), 0, len(src.t) - 1)
    planted, steps = insert_steps(src.foot_contact[k].copy(), delta)
    for f, a, b in steps:  # an inserted step lifts the foot
        lift = FLOOR_Z_M - sc.sole_z + STEP_LIFT_M * np.sin(np.pi * np.linspace(0.0, 1.0, b - a))
        foot_p[a:b, f, 2] = np.maximum(foot_p[a:b, f, 2], lift)
    spine = _rest_relative(human, "Spine02", _interp_rot(t, src.t, src.rot["Spine02"]))
    hand = _interp(t, src.t, src.pos["L_Hand"]) - _interp(t, src.t, src.pos["L_Upperarm"])
    near = np.zeros(len(t))
    for c in cts:
        near = np.maximum(near, bump(t - c["t"], 0.35))
    return Targets(
        t=t,
        pelvis_p=pelvis_p,
        pelvis_r=pelvis_r,
        torso_r=spine,
        foot_p=foot_p,
        foot_yaw=foot_yaw,
        planted=planted,
        lhand_rel=np.einsum("tji,tj->ti", spine, hand) * sc.arm,
        paddle_p=paddle_p,
        paddle_r=paddle_r,
        near=near,
        contacts=cts,
    )


__all__ = ["FPS", "HOLD_S", "Scales", "Segment", "Targets", "build_targets", "contacts", "insert_steps", "paddle_path", "reach_problem", "scales", "segments"]
