"""
The retargeted strokes as a package: ``motion.npz`` (the kept clips), ``strokes.csv`` (every stroke's check values and
verdict) and ``manifest.json`` (conventions, robot, checks, counts, lineage), zipped.

**Kept clips.** A rally's motion is cut 0.6 s before to 0.4 s after every rejected stroke, and 0.3 s either side of
every frame that fails a check; the pieces of at least 1 s are kept.

**motion.npz** has the layout of ``ping-pong/g1-retargeted-strokes`` (the worked example's reference output): flat
arrays over every kept frame at 50 Hz with per-clip offsets ``clip_start``. Per frame: ``joint_pos``, ``joint_vel``,
``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w`` (every link, then ``paddle``), ``root_*`` and
``paddle_*`` (the same, for the pelvis and the paddle), ``foot_contact``, ``contact`` (the frame nearest a stroke),
``t_to_contact_s`` and ``time_s``. Per clip: ``clip_session``, ``clip_robot_end``, ``clip_t0_s``. Per kept stroke:
``stroke_id``, ``stroke_clip``, ``stroke_frame`` and ``stroke_*`` values. Names, not order: ``joint_names`` and
``body_names`` say which column is which.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lib.retarget.model import RobotModel, body_poses
from lib.retarget.source import matrix_to_quat
from lib.retarget.targets import FPS

SCHEMA = "him-retargeted-strokes/1"
CUT_BEFORE_S, CUT_AFTER_S, CUT_GROW_S, MIN_CLIP_S = 0.6, 0.4, 0.3, 1.0
CSV_FIELDS = (
    "stroke_id",
    "session",
    "source",
    "face",
    "kind",
    "t_contact_s",
    "kept",
    "cause",
    "in_motion",
    "blade_pos_max_m",
    "blade_face_max_deg",
    "contact_pos_m",
    "contact_face_deg",
    "contact_vel_err_mps",
    "limit_use_max",
    "speed_ratio_max",
    "speed_joint",
    "torque_ratio_max",
    "torque_joint",
    "capture_shift_m",
    "foot_slide_max_m",
    "foot_penetration_max_m",
    "raw_blade_pos_max_m",
    "raw_blade_face_max_deg",
)
NPZ_STROKE_FIELDS = (
    "session",
    "source",
    "face",
    "kind",
    "t_contact_s",
    "blade_p",
    "blade_v",
    "blade_n",
    "blade_pos_max_m",
    "blade_face_max_deg",
    "contact_pos_m",
    "contact_face_deg",
    "contact_vel_err_mps",
    "speed_ratio_max",
    "torque_ratio_max",
    "capture_shift_m",
    "foot_slide_max_m",
)


@dataclass
class Clip:
    session: int
    robot_end: int
    t: np.ndarray  # (T,) camera clock
    root_p: np.ndarray
    root_r: np.ndarray
    q: np.ndarray
    planted: np.ndarray  # (T, 2)
    contacts: list[tuple[str, float]]  # (stroke id, contact time) inside the clip


def kept_pieces(t: np.ndarray, bad: np.ndarray, rejected_tc: list[float]) -> list[tuple[int, int]]:
    """``[start, end)`` frame ranges of a segment that survive the cuts (see the module docstring)."""
    cut = bad.copy()
    for tc in rejected_tc:
        cut |= (t >= tc - CUT_BEFORE_S) & (t <= tc + CUT_AFTER_S)
    grow = round(CUT_GROW_S * FPS)
    for i in np.nonzero(cut)[0]:
        cut[max(0, i - grow) : i + grow + 1] = True
    edges = np.diff(np.r_[1, cut.astype(np.int8), 1])
    starts, ends = np.nonzero(edges == -1)[0], np.nonzero(edges == 1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True) if (b - a) / FPS >= MIN_CLIP_S]


def clip_arrays(m: RobotModel, c: Clip) -> dict[str, np.ndarray]:
    """One clip's per-frame arrays (velocities by central differences at 50 Hz)."""
    dt = 1.0 / FPS
    rot, pos = body_poses(m, c.root_p, c.root_r, c.q)
    lin, ang = np.gradient(pos, dt, axis=0), np.zeros(pos.shape)
    if len(c.t) >= 3:
        rel = np.einsum("tbij,tbkj->tbik", rot[2:], rot[:-2])
        ang[1:-1] = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(len(c.t) - 2, -1, 3) / (2 * dt)
        ang[0], ang[-1] = ang[1], ang[-2]
    quat = matrix_to_quat(rot)
    for k in range(1, len(quat)):  # sign-continuous per body
        quat[k, (quat[k] * quat[k - 1]).sum(-1) < 0] *= -1
    contact, t_to = np.zeros(len(c.t), bool), np.full(len(c.t), np.inf)
    for _sid, tc in c.contacts:
        contact[int(np.argmin(np.abs(c.t - tc)))] = True
        d = tc - c.t
        t_to = np.where(np.abs(d) < np.abs(t_to), d, t_to)
    f32 = lambda a: a.astype(np.float32)  # noqa: E731
    return {
        "joint_pos": f32(c.q),
        "joint_vel": f32(np.gradient(c.q, dt, axis=0)),
        "body_pos_w": f32(pos),
        "body_quat_w": f32(quat),
        "body_lin_vel_w": f32(lin),
        "body_ang_vel_w": f32(ang),
        "root_pos_w": f32(pos[:, 0]),
        "root_quat_w": f32(quat[:, 0]),
        "root_lin_vel_w": f32(lin[:, 0]),
        "root_ang_vel_w": f32(ang[:, 0]),
        "paddle_pos_w": f32(pos[:, -1]),
        "paddle_quat_w": f32(quat[:, -1]),
        "paddle_lin_vel_w": f32(lin[:, -1]),
        "paddle_ang_vel_w": f32(ang[:, -1]),
        "foot_contact": c.planted.astype(bool),
        "contact": contact,
        "t_to_contact_s": t_to,
        "time_s": c.t.astype(np.float64),
    }


def write_package(out: Path, m: RobotModel, clips: list[Clip], strokes: list[dict], manifest: dict) -> Path:
    """Write the package's files into ``out`` and zip them as ``<out>.zip``; returns the zip."""
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, list] = {}
    starts, where = [0], {}
    for k, c in enumerate(clips):
        for name, v in clip_arrays(m, c).items():
            arrays.setdefault(name, []).append(v)
        for sid, tc in c.contacts:
            where[sid] = (k, starts[-1] + int(np.argmin(np.abs(c.t - tc))))
        starts.append(starts[-1] + len(c.t))
    if not clips:  # nothing kept: the same arrays, with no frames
        still = Clip(0, 0, np.arange(3) / FPS, np.zeros((3, 3)), np.tile(np.eye(3), (3, 1, 1)), np.tile(m.default_q, (3, 1)), np.ones((3, 2), bool), [])
        arrays = {name: [v[:0]] for name, v in clip_arrays(m, still).items()}
    npz: dict[str, np.ndarray] = {name: np.concatenate(v) for name, v in arrays.items()}
    kept = [s for s in strokes if s["kept"] and s["stroke_id"] in where]
    npz |= {
        "fps": np.array(FPS),
        "joint_names": np.array(m.joint_names),
        "body_names": np.array(m.body_names),
        "clip_start": np.array(starts, dtype=np.int64),
        "clip_session": np.array([c.session for c in clips], dtype=np.int16),
        "clip_robot_end": np.array([c.robot_end for c in clips], dtype=np.int8),
        "clip_t0_s": np.array([c.t[0] for c in clips], dtype=np.float64),
        "stroke_id": np.array([s["stroke_id"] for s in kept], dtype=str),
        "stroke_clip": np.array([where[s["stroke_id"]][0] for s in kept], dtype=np.int32),
        "stroke_frame": np.array([where[s["stroke_id"]][1] for s in kept], dtype=np.int64),
    }
    for f in NPZ_STROKE_FIELDS:
        npz[f"stroke_{f}"] = np.array([s[f] for s in kept]) if kept else np.zeros(0)
    np.savez_compressed(out / "motion.npz", **npz)  # pyright: ignore[reportArgumentType]
    for s in strokes:
        s["in_motion"] = s["stroke_id"] in where and s["kept"]
    with open(out / "strokes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(strokes)
    files = {
        name: {"sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(), "bytes": (out / name).stat().st_size} for name in ("motion.npz", "strokes.csv")
    }
    manifest = manifest | {
        "schema": SCHEMA,
        "motion": {"frames": starts[-1], "clips": len(clips), "strokes": len(kept), "seconds": starts[-1] / FPS},
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=_plain))
    path = out.with_suffix(".zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name in ("manifest.json", "motion.npz", "strokes.csv"):
            z.write(out / name, name)
    return path


def _plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


__all__ = ["SCHEMA", "Clip", "clip_arrays", "kept_pieces", "write_package"]
