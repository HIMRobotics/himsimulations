"""
Build the human-stroke bundle ``ping-pong/him-mocap-strokes`` (a one-off export; its output is the published asset).

The bundle is HIM's suited player, every session, in the table frame on the camera clock, with the paddle state at
each contact. Inputs:

* ``--donors``: the donor library zip (``suited.npz``: the capture on the suited player's game body, ActorCore
  Party_M_0001 at scale 0.9255, conditioned onto the shakehand grip; 30 Hz of mocap time, in a per-frame character
  frame);
* ``--skeleton``: the ActorCore rest skeletons (joints, parents, rest translations and rotations); by default the
  replay's ``lib/replay/data/actorcore_skeletons.json``;
* ``--sync``: per session ``session_<k>_mocap_world.npz``, the mocap-to-table registration ``X = Rz(yaw) M + [d(t), z0]``
  and each frame's camera time ``t_cam_rel`` (the mocap clock runs 2-4 % fast against the camera);
* ``--paddle``: per session ``session_<k>_strokes.parquet``, the paddle state solved from the ball at each contact
  (blade centre, struck-face normal, face velocity), and ``inputs.json`` (its build id).

The output frame is robot-canonical: the suited player at -X (a 180 degree yaw for sessions recorded from the other
end, never a mirror), z up, floor at z = -0.76 m, the table-top centre at the origin.

    uv run python tools/build_mocap_strokes.py --donors D.zip --sync DIR --paddle DIR --out him-mocap-strokes-v1.zip
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project root, for lib/

from lib.retarget.source import FLOOR_Z_M, matrix_to_quat, quat_to_matrix, rz

SCHEMA = "him-mocap-strokes/1"
SUITED_END = {1: "8E", 2: "8E", 3: "8F", 4: "8F", 5: "8F", 6: "8E", 7: "8E"}  # the camera end the suited player stood at
KEY_JOINTS = tuple("Hip Pelvis L_Thigh R_Thigh L_Calf R_Calf L_Foot R_Foot L_ToeBase R_ToeBase Waist Spine01 Spine02 NeckTwist01 Head".split()) + tuple(
    "L_Upperarm R_Upperarm L_Forearm R_Forearm L_Hand R_Hand".split()
)
STROKE_COLUMNS = ("id", "source", "t", "face", "blade_p", "normal", "vel", "donor_blade_p", "speed_mps", "contact_sd_m", "normal_sd_deg", "ball_v_out")
FIXED_TIME = (2026, 8, 15, 0, 0, 0)  # every zip entry's time: the capture's date, so rebuilds are byte-identical
SAME_STROKE_S = 0.6  # a library contact this close to a solved one is that stroke (one player's hits are >= 0.8 s apart)


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def human_fk(skel: dict, local: np.ndarray, root_m: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """World quaternions (n, J, 4) and table-frame positions (n, J, 3) of the skeleton's joints (parents first)."""
    rot, pos = np.zeros(local.shape), np.zeros((*local.shape[:-1], 3))
    for i, parent in enumerate(skel["parents"]):
        offset = root_m if skel["joints"][i] == "Hip" else skel["offsets"][i]
        if parent < 0:
            rot[:, i], pos[:, i] = local[:, i], offset
        else:
            rot[:, i] = qmul(rot[:, parent], local[:, i])
            pos[:, i] = pos[:, parent] + qrot(rot[:, parent], np.broadcast_to(offset, pos[:, i].shape))
    return rot, np.array([0.0, 0.0, FLOOR_Z_M]) + scale * pos


def load_inputs(args) -> tuple[dict, dict, dict, dict]:
    with zipfile.ZipFile(args.donors) as z:
        header = json.loads(z.read("manifest.json"))["roles"]["suited"]
        donors = dict(np.load(io.BytesIO(z.read("suited.npz"))))
        quality = dict(np.load(io.BytesIO(z.read("suited-quality.npz"))))
    entry = json.loads(Path(args.skeleton).read_text())["bodies"][header["body"]]
    rest = np.asarray(entry["rest_rotation_wxyz"], dtype=np.float64)
    skel = {
        "joints": list(entry["joints"]),
        "parents": list(entry["parents"]),
        "rest": rest / np.linalg.norm(rest, axis=1, keepdims=True),
        "offsets": np.asarray(entry["rest_translation_m"], dtype=np.float64),
    }
    for name, off in zip(header["translated_joints"], header["translated_offsets_m"], strict=True):  # the track's lengthened bones
        skel["offsets"][skel["joints"].index(name)] = off
    return header, donors, quality, skel


def build_session(k: int, header: dict, donors: dict, quality: dict, skel: dict, args) -> dict[str, np.ndarray]:
    """One session's arrays (see the module docstring)."""
    frames = np.nonzero(donors["session"] == k)[0]
    reg = np.load(Path(args.sync) / f"session_{k}_mocap_world.npz")
    if np.ptp(reg["yaw"]) > 1e-9 or len(reg["t_cam_rel"]) != len(frames):
        raise ValueError(f"session {k}: the registration does not match the donor frames")
    scale = float(header["body_scale"])
    local = np.broadcast_to(skel["rest"], (len(frames), *skel["rest"].shape)).copy()
    local[:, [skel["joints"].index(n) for n in header["joints"]]] = donors["local_rotation_wxyz"][frames].astype(np.float64)
    local[:, skel["joints"].index("Hip")] = donors["root_rotation_wxyz"][frames].astype(np.float64)
    root = (donors["root_position_m"][frames].astype(np.float64) - [0.0, 0.0, FLOOR_Z_M]) / scale
    world_q, world_p = human_fk(skel, local, root, scale)
    # character frame (facing -Y) -> registered capture -> robot-canonical table frame
    end = 1 if SUITED_END[k] == "8E" else -1
    canon, r_reg = rz(np.pi) if end > 0 else np.eye(3), rz(float(reg["yaw"][0]))
    rc = canon @ r_reg @ rz(donors["cyaw_rad"][frames].astype(np.float64) + np.pi / 2)
    origin = (np.c_[donors["cpos_m"][frames], np.zeros(len(frames))] @ r_reg.T + np.c_[reg["drift_xy"], np.zeros(len(frames))]) @ canon.T
    ji = [skel["joints"].index(n) for n in KEY_JOINTS]
    pos = np.einsum("nij,nkj->nki", rc, world_p[:, ji]) + origin[:, None]
    rot = rc[:, None] @ quat_to_matrix(world_q[:, ji])
    paddle_p = np.einsum("nij,nj->ni", rc, donors["paddle_position_m"][frames].astype(np.float64)) + origin
    paddle_r = rc @ quat_to_matrix(donors["paddle_rotation_wxyz"][frames].astype(np.float64))  # game axes: X across, Y tip, Z forehand
    t = np.asarray(reg["t_cam_rel"], dtype=np.float64)
    strokes = session_strokes(k, t, frames, paddle_p, paddle_r, canon, quality, args.paddle)
    out = {
        "time_s": t,
        "clip": (np.searchsorted(donors["clip_start"], frames, side="right") - 1).astype(np.int32),
        "donor_frame": frames,
        "joint_pos": pos.astype(np.float32),
        "joint_quat": matrix_to_quat(rot).astype(np.float32),
        "paddle_pos": paddle_p.astype(np.float32),
        "paddle_quat": matrix_to_quat(paddle_r).astype(np.float32),
        "foot_contact": donors["foot_contact"][frames].astype(bool),
        "robot_end": np.int8(end),
    }
    return out | {f"stroke_{c}": np.array([row[j] for row in strokes]) for j, c in enumerate(STROKE_COLUMNS)}


def session_strokes(k, t, frames, paddle_p, paddle_r, canon, quality, paddle_dir) -> list[tuple]:
    """Every contact (:data:`STROKE_COLUMNS`): the paddle state solved from the ball (``paddle_inverse``), else the
    capture's own paddle (``capture``)."""
    import pyarrow.parquet as pq

    spline = CubicSpline(t, paddle_p, axis=0)

    def donor(tc: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # blade centre, axes, velocity at camera time tc
        f = float(np.interp(tc, t, np.arange(len(t), dtype=np.float64)))
        f0 = int(np.clip(np.floor(f), 0, len(t) - 2))
        u, _s, vt = np.linalg.svd(paddle_r[f0] * (1 - (f - f0)) + paddle_r[f0 + 1] * (f - f0))
        return spline(tc), u @ vt, spline(tc, 1)

    tab = pq.read_table(Path(paddle_dir) / f"session_{k}_strokes.parquet").to_pydict()
    vec = lambda r, name: canon @ np.array([tab[f"{name}_{a}"][r] for a in "xyz"], dtype=np.float64)  # noqa: E731
    rows = []
    for r in range(len(tab["stroke_id"])):
        if tab["hitter"][r] != "suited" or tab["session"][r] != k or not tab["ok"][r]:
            continue
        tc = float(tab["t_contact_s"][r])
        if not t[0] + 0.05 < tc < t[-1] - 0.05:
            continue
        n = vec(r, "normal") / np.linalg.norm(vec(r, "normal"))
        dp, dr, _dv = donor(tc)
        face = 0 if dr[:, 2] @ n > 0 else 1
        extra = (float(tab["speed_mps"][r]), float(tab["contact_sd_m"][r]), float(tab["normal_sd_deg"][r]), vec(r, "ball_v_out_pred"))
        rows.append((str(tab["stroke_id"][r]), "paddle_inverse", tc, face, vec(r, "blade_centre"), n, vec(r, "vel"), dp, *extra))
    solved_t = np.array([row[2] for row in rows])
    for fr in quality["frame"][np.isin(quality["frame"], frames)]:
        tc = float(t[int(np.searchsorted(frames, fr))])
        if len(solved_t) and np.min(np.abs(solved_t - tc)) < SAME_STROKE_S:
            continue  # the same stroke as a solved one
        dp, dr, dv = donor(tc)
        face = 0 if dr[0, 2] > 0 else 1  # no ball: the rubber facing the net (+X)
        rows.append((f"s{k}_lib{int(fr)}", "capture", tc, face, dp, dr[:, 2] * (1.0 - 2.0 * face), dv, dp, np.nan, np.nan, np.nan, np.full(3, np.nan)))
    return sorted(rows, key=lambda row: row[2])


def rest_values(skel: dict, scale: float) -> tuple[dict, np.ndarray]:
    """The body's rest segment lengths (m, at body scale) and the key joints' rest world rotations (J, 4)."""
    root = skel["offsets"][skel["joints"].index("Hip")]
    q, p = human_fk(skel, skel["rest"][None], root, scale)
    j = lambda n: p[0, skel["joints"].index(n)]  # noqa: E731
    d = lambda a, b: float(np.linalg.norm(j(a) - j(b)))  # noqa: E731
    lengths = {
        "leg_m": 0.5 * (d("L_Thigh", "L_Calf") + d("L_Calf", "L_Foot")) + 0.5 * (d("R_Thigh", "R_Calf") + d("R_Calf", "R_Foot")),
        "arm_m": d("R_Upperarm", "R_Forearm") + d("R_Forearm", "R_Hand"),
        "hip_height_m": float(0.5 * (j("L_Thigh")[2] + j("R_Thigh")[2]) - FLOOR_Z_M),
        "ankle_height_m": float(0.5 * (j("L_Foot")[2] + j("R_Foot")[2]) - FLOOR_Z_M),
        "height_m": float(j("Head")[2] - FLOOR_Z_M),
    }
    return lengths, matrix_to_quat(quat_to_matrix(q[0, [skel["joints"].index(n) for n in KEY_JOINTS]]))


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """``np.savez_compressed``'s format with fixed entry times, so the same inputs give the same bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, a in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            with z.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(a), allow_pickle=False)
    return buf.getvalue()


def stamp(path: Path) -> dict:
    data = Path(path).read_bytes()
    return {"file": Path(path).name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the human-stroke bundle ping-pong/him-mocap-strokes (see the module docstring).")
    for name in ("--donors", "--sync", "--paddle", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--skeleton", default=str(Path(__file__).resolve().parents[1] / "lib" / "replay" / "data" / "actorcore_skeletons.json"))
    ap.add_argument("--sessions", default="1,2,3,4,5,6,7")
    args = ap.parse_args()
    header, donors, quality, skel = load_inputs(args)
    sessions = [int(k) for k in args.sessions.split(",")]
    lengths, rest_quat = rest_values(skel, float(header["body_scale"]))
    arrays = {"joint_names": np.array(KEY_JOINTS), "rest_quat": rest_quat}
    for k in sessions:
        arrays |= {f"s{k}/{name}": v for name, v in build_session(k, header, donors, quality, skel, args).items()}
        print(f"session {k}: {len(arrays[f's{k}/time_s'])} frames, {len(arrays[f's{k}/stroke_id'])} strokes", flush=True)
    paddle = Path(args.paddle)
    manifest = {
        "schema": SCHEMA,
        "body": header["body"],
        "body_scale": header["body_scale"],
        "sessions": sessions,
        "frames": {k: len(arrays[f"s{k}/time_s"]) for k in sessions},
        "strokes": {k: len(arrays[f"s{k}/stroke_id"]) for k in sessions},
        "rest_lengths_m": lengths,
        "conventions": {
            "frame": "robot-canonical table frame: origin at the table-top centre, z up, the suited player at -X (sessions recorded from the other end are yawed 180 degrees, never mirrored), floor at z = -0.76 m; metres, radians",
            "time": "camera clock (real time), session seconds; frames at the capture's 30 Hz (mocap clock), so time_s steps are 32-35 ms",
            "quaternions": "w, x, y, z; joint_quat is each key joint's world orientation, rest_quat the same joints at the rest pose (character frame, facing -Y)",
            "paddle": "the capture's paddle through the shakehand grip: paddle_pos the blade centre, paddle_quat axes X across, Y towards the tip, Z the forehand normal",
            "strokes": "stroke_source paddle_inverse: blade centre, struck-face normal (into the ball) and face velocity solved from the ball; capture: no ball solution, the capture's own paddle. stroke_face 0 forehand rubber, 1 backhand",
        },
        "lineage": {
            "donors": stamp(Path(args.donors)) | {"suited_sha256_16": header.get("sha256_16")},
            "skeleton": stamp(Path(args.skeleton)),
            "paddle_build": json.loads((paddle / "inputs.json").read_text()).get("build_id"),
            "paddle": [stamp(paddle / f"session_{k}_strokes.parquet") for k in sessions],
            "sync": [stamp(Path(args.sync) / f"session_{k}_mocap_world.npz") for k in sessions],
            "tool": stamp(Path(__file__)),
        },
    }
    out = Path(args.out)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(zipfile.ZipInfo("manifest.json", date_time=FIXED_TIME), json.dumps(manifest, indent=1))
        z.writestr(zipfile.ZipInfo("mocap.npz", date_time=FIXED_TIME), npz_bytes(arrays))
    print(json.dumps(stamp(out)))


if __name__ == "__main__":
    main()
