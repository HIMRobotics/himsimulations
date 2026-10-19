"""
The human side: HIM's suited player, per recorded session, from the bundle ``ping-pong/him-mocap-strokes``
(``tools/build_mocap_strokes.py`` builds it; ``docs/retargeting.md`` describes it).

Every array is on the robot-canonical table frame (origin at the table-top centre, z up, the player at -X, floor at
-0.76 m) and the camera clock (real time). A stroke's paddle state is the one solved from the ball (``paddle_inverse``)
or, where the ball gave none, the capture's own paddle (``capture``).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

FLOOR_Z_M = -0.76


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) w, x, y, z -> (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) w, x, y, z with w >= 0."""
    q = Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
    return (q * np.where(q[:, :1] < 0, -1.0, 1.0)).reshape(*m.shape[:-2], 4)


def rz(a) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    o, one = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, -s, o], -1), np.stack([s, c, o], -1), np.stack([o, o, one], -1)], -2)


@dataclass
class Stroke:
    """One contact: when, which rubber, and the paddle state the robot must reproduce."""

    id: str
    source: str  # "paddle_inverse" (solved from the ball) or "capture" (the capture's own paddle)
    t: float  # camera clock, s
    face: int  # 0 forehand rubber (+y of the paddle frame), 1 backhand
    blade_p: np.ndarray  # (3,) blade centre, m
    normal: np.ndarray  # (3,) struck-face normal, into the ball
    vel: np.ndarray  # (3,) face velocity, m/s
    donor_blade_p: np.ndarray  # (3,) the capture's own blade centre at the contact


@dataclass
class HumanSession:
    """One session's human source: per capture frame (30 Hz of mocap time) on the camera clock."""

    session: int
    robot_end: int  # +1: recorded from the far end and yawed 180 degrees
    t: np.ndarray  # (N,) s
    clip: np.ndarray  # (N,) continuous capture stretch
    pos: dict[str, np.ndarray]  # key joint -> (N, 3) position
    rot: dict[str, np.ndarray]  # key joint -> (N, 3, 3) world orientation
    paddle_p: np.ndarray  # (N, 3) the capture's blade centre
    paddle_r: np.ndarray  # (N, 3, 3) its axes: X across, Y tip, Z forehand normal
    foot_contact: np.ndarray  # (N, 2) left, right planted
    strokes: list[Stroke]
    _spline: CubicSpline | None = field(default=None, repr=False)

    def paddle_at(self, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The capture's paddle at camera times ``t``: blade centre (n, 3) and velocity (n, 3) from a cubic spline, axes
        (n, 3, 3) interpolated linearly and re-orthonormalised."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if self._spline is None:
            self._spline = CubicSpline(self.t, self.paddle_p, axis=0)
        f = np.interp(t, self.t, np.arange(len(self.t), dtype=np.float64))
        f0 = np.clip(np.floor(f).astype(int), 0, len(self.t) - 2)
        u = (f - f0)[:, None, None]
        uu, _s, vv = np.linalg.svd(self.paddle_r[f0] * (1 - u) + self.paddle_r[f0 + 1] * u)
        return self._spline(t), uu @ vv, self._spline(t, 1)


@dataclass
class HumanStrokes:
    """Every session, and the body's rest: segment lengths (m) and the key joints' rest orientations."""

    sessions: dict[int, HumanSession]
    rest_lengths: dict[str, float]
    rest_rot: dict[str, np.ndarray]
    manifest: dict


def load_mocap_strokes(path: str | Path, sessions: tuple[int, ...] | None = None) -> HumanStrokes:
    """Read the bundle (a zip of ``manifest.json`` and ``mocap.npz``) for the listed sessions (default: all)."""
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        d = np.load(io.BytesIO(z.read("mocap.npz")), allow_pickle=False)
        names = [str(n) for n in d["joint_names"]]
        out = {}
        for k in sessions or manifest["sessions"]:
            p = f"s{k}/"
            pos, quat = d[p + "joint_pos"].astype(np.float64), quat_to_matrix(d[p + "joint_quat"])
            strokes = [
                Stroke(id=str(i), source=str(src), t=float(t), face=int(f), blade_p=bp, normal=n, vel=v, donor_blade_p=dbp)
                for i, src, t, f, bp, n, v, dbp in zip(
                    *(d[p + "stroke_" + c] for c in ("id", "source", "t", "face", "blade_p", "normal", "vel", "donor_blade_p")), strict=True
                )
            ]
            out[int(k)] = HumanSession(
                session=int(k),
                robot_end=int(d[p + "robot_end"]),
                t=d[p + "time_s"],
                clip=d[p + "clip"],
                pos={n: pos[:, j] for j, n in enumerate(names)},
                rot={n: quat[:, j] for j, n in enumerate(names)},
                paddle_p=d[p + "paddle_pos"].astype(np.float64),
                paddle_r=quat_to_matrix(d[p + "paddle_quat"]),
                foot_contact=d[p + "foot_contact"],
                strokes=strokes,
            )
        rest = quat_to_matrix(d["rest_quat"])
    return HumanStrokes(sessions=out, rest_lengths=manifest["rest_lengths_m"], rest_rot={n: rest[j] for j, n in enumerate(names)}, manifest=manifest)


__all__ = ["FLOOR_Z_M", "HumanSession", "HumanStrokes", "Stroke", "load_mocap_strokes", "matrix_to_quat", "quat_to_matrix", "rz"]
