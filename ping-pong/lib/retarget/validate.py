"""
The quality checks that decide whether a retargeted stroke is good: graded per stroke on the solved motion, and per
frame for everything the package keeps. No threshold here is loosened to make a stroke pass.

=============  ==========================================================================================
reach          the paddle path: the blade centre within 1 cm and the forehand normal within 5 degrees of the
               path through +-0.1 s of the contact (graded every 1 ms), and the same at the contact against
               the paddle state solved from the ball; every joint within 0.95 of its range
speed          every joint below 0.8 of its speed limit
torque         inverse dynamics within each motor's torque at its speed (``dynamics.py``)
balance        the unsupported wrench moves the capture point at most 3 cm over any 0.2 s (half the G1's sole
               width: a push the robot absorbs without stepping)
feet           no planted sole slides more than 1 cm or sinks more than 5 mm into the floor
=============  ==========================================================================================

Speed, torque and balance are graded from 0.5 s before to 0.3 s after the contact. A failing stroke is rejected with
its first failing check, in the order above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from lib.retarget.dynamics import inverse_dynamics
from lib.retarget.model import RobotModel, body_poses
from lib.retarget.optimize import Solution
from lib.retarget.source import FLOOR_Z_M, HumanSession
from lib.retarget.targets import HOLD_S, Segment, Targets, contacts, paddle_path


@dataclass(frozen=True)
class Gates:
    blade_pos_m: float = 0.01
    blade_face_deg: float = 5.0
    limit_fraction: float = 0.95
    speed_fraction: float = 0.80
    torque_ratio: float = 1.0
    capture_shift_m: float = 0.03
    foot_slide_m: float = 0.01
    foot_penetration_m: float = 0.005
    stroke_window_s: tuple[float, float] = (-0.5, 0.3)


GATES = Gates()
STEP_M = 0.05  # a stroke is "stepping" when the static reach moves the robot's stance more than this (a label, not a check)
CHECKS = ("reach", "speed", "torque", "balance", "feet")


def interpolate(sol: Solution, te: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The solution at times ``te``: pelvis position and joints linearly, pelvis orientation by slerp."""
    x = (te - sol.t[0]) * (len(sol.t) - 1) / (sol.t[-1] - sol.t[0])
    j = np.clip(np.floor(x).astype(int), 0, len(sol.t) - 2)
    u = np.clip(x - j, 0.0, 1.0)[:, None]
    rel = Rotation.from_matrix(np.einsum("nji,njk->nik", sol.root_r[j], sol.root_r[j + 1])).as_rotvec()
    return sol.root_p[j] * (1 - u) + sol.root_p[j + 1] * u, sol.root_r[j] @ Rotation.from_rotvec(rel * u).as_matrix(), sol.q[j] * (1 - u) + sol.q[j + 1] * u


def blade_at(m: RobotModel, sol: Solution, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The robot's blade centre (n, 3) and paddle frame (n, 3, 3) at times ``te``."""
    r, p = body_poses(m, *interpolate(sol, te))
    return p[:, -1], r[:, -1]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.einsum("...i,...i->...", a, b), -1.0, 1.0)))


def grade_segment(m: RobotModel, src: HumanSession, seg: Segment, tg: Targets, sol: Solution, reach_xy: dict) -> tuple[list[dict], np.ndarray]:
    """Every stroke's check values and verdict, and the frames (50 Hz) failing any check (see the module docstring)."""
    g = GATES
    dyn = inverse_dynamics(m, sol.t, sol.root_p, sol.root_r, sol.q)
    rot, pos = body_poses(m, sol.root_p, sol.root_r, sol.q)
    mid, half = (m.lower + m.upper) / 2, (m.upper - m.lower) / 2
    limit_use = np.abs(sol.q - mid) / half
    speed_ratio, torque_ratio = np.abs(dyn.qd) / m.speed_limit, dyn.torque_ratio
    feet = list(m.feet)
    corners = np.einsum("nfij,kj->nfki", rot[:, feet], m.sole) + pos[:, feet][:, :, None]
    penetration = np.maximum(FLOOR_Z_M - corners[..., 2].min(-1), 0.0)
    plants = []  # every plant's slide: the sole centre's horizontal travel from where it landed
    for f in (0, 1):
        edges = np.diff(np.r_[0, tg.planted[:, f].astype(np.int8), 0])
        for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True):
            sole = corners[a:b, f].mean(1)
            plants.append((sol.t[a], sol.t[b - 1], float(np.linalg.norm(sole[:, :2] - sole[0, :2], axis=1).max())))
    cts = contacts(src, seg.strokes)
    rows = []
    for c in cts:
        tc, s = c["t"], c["stroke"]
        te = tc + np.arange(-HOLD_S, HOLD_S + 1e-9, 0.001)
        bp, br = blade_at(m, sol, te)
        b0, r0 = blade_at(m, sol, np.array([tc - 0.001, tc, tc + 0.001]))
        sign = 1.0 if c["face"] == 0 else -1.0
        solved = s.source == "paddle_inverse"
        w0, w1 = tc + g.stroke_window_s[0], tc + g.stroke_window_s[1]
        win, fwin = (dyn.t >= w0) & (dyn.t <= w1), (sol.t >= w0) & (sol.t <= w1)
        sr, tr = speed_ratio[win].max(0), torque_ratio[win].max(0)
        row = {
            "stroke_id": s.id,
            "session": seg.session,
            "source": s.source,
            "face": s.face,
            "kind": ("forehand" if s.face == 0 else "backhand") + ("-stepping" if np.linalg.norm(reach_xy[s.id]) > STEP_M else "-stance"),
            "t_contact_s": tc,
            "blade_p": s.blade_p.tolist(),
            "blade_n": s.normal.tolist(),
            "blade_v": s.vel.tolist(),
            **_path_errors(src, cts, te, bp, br, raw=False),
            "contact_pos_m": float(np.linalg.norm(b0[1] - c["p"])) if solved else np.nan,
            "contact_face_deg": float(_angle_deg(sign * r0[1][:, 1], c["n"])) if solved else np.nan,
            "contact_vel_err_mps": float(np.linalg.norm((b0[2] - b0[0]) / 0.002 - c["v"])) if solved else np.nan,
            "limit_use_max": float(limit_use[fwin].max()),
            "speed_ratio_max": float(sr.max()),
            "speed_joint": m.joint_names[int(np.argmax(sr))],
            "torque_ratio_max": float(tr.max()),
            "torque_joint": m.joint_names[int(np.argmax(tr))],
            "capture_shift_m": dyn.capture_shift(w0, w1),
            "foot_slide_max_m": max([slide for t0, t1, slide in plants if t1 >= w0 and t0 <= w1] or [0.0]),
            "foot_penetration_max_m": float(penetration[fwin].max()),
            # the negative control: the same motion against the capture's own, uncorrected paddle
            **{f"raw_{k}": v for k, v in _path_errors(src, cts, te, bp, br, raw=True).items()},
        }
        rows.append(row | verdict(row))
    # every frame the package keeps must pass too
    bad_dyn = (dyn.capture_shift_series() > g.capture_shift_m) | (np.nanmax(np.where(np.isfinite(torque_ratio), torque_ratio, 9.9), 1) > g.torque_ratio)
    bad_dyn |= speed_ratio.max(1) > g.speed_fraction
    bad = np.interp(sol.t, dyn.t, bad_dyn.astype(float)) > 0
    bad |= (limit_use.max(1) > g.limit_fraction + 1e-9) | (penetration.max(1) > g.foot_penetration_m)
    for t0, t1, slide in plants:
        if slide > g.foot_slide_m:
            bad |= (sol.t >= t0) & (sol.t <= t1)
    return rows, bad


def _path_errors(src: HumanSession, cts: list[dict], te: np.ndarray, bp: np.ndarray, br: np.ndarray, raw: bool) -> dict[str, float]:
    """The robot's blade against the paddle path (``raw``: the capture's own, uncorrected paddle) through the hold."""
    rp, rr = paddle_path(src, cts, te, raw=raw)
    return {"blade_pos_max_m": float(np.linalg.norm(bp - rp, axis=1).max()), "blade_face_max_deg": float(_angle_deg(br[:, :, 1], rr[:, :, 1]).max())}


def verdict(row: dict) -> dict:
    """``kept`` and the first failing check (``cause``) of one stroke's values."""
    g = GATES
    within = lambda v, limit: np.isnan(v) or v <= limit  # noqa: E731
    ok = {
        "reach": row["blade_pos_max_m"] <= g.blade_pos_m
        and within(row["contact_pos_m"], g.blade_pos_m)
        and row["blade_face_max_deg"] <= g.blade_face_deg
        and within(row["contact_face_deg"], g.blade_face_deg)
        and row["limit_use_max"] <= g.limit_fraction + 1e-9,
        "speed": row["speed_ratio_max"] <= g.speed_fraction + 1e-9,
        "torque": row["torque_ratio_max"] <= g.torque_ratio,
        "balance": row["capture_shift_m"] <= g.capture_shift_m,
        "feet": row["foot_slide_max_m"] <= g.foot_slide_m and row["foot_penetration_max_m"] <= g.foot_penetration_m,
    }
    cause = next((k for k in CHECKS if not ok[k]), "")
    return {"kept": cause == "", "cause": cause}


__all__ = ["CHECKS", "GATES", "STEP_M", "Gates", "blade_at", "grade_segment", "interpolate", "verdict"]
