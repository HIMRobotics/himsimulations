"""
The retargeter without the human data: the robot model and its kinematics, the dynamics, the solver on a motion the
robot can make, and the quality checks, which must keep a good stroke and reject named corruptions of it.

The probe rally itself (``unitree-g1-smoke``) runs on Antioch; ``docs/retargeting.md`` records its local result.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from lib.retarget.dynamics import inverse_dynamics
from lib.retarget.model import body_poses, robot_model, rotvec_to_matrix
from lib.retarget.optimize import Problem, Solution, support_distance
from lib.retarget.package import Clip, kept_pieces, write_package
from lib.retarget.run import recheck_package
from lib.retarget.source import FLOOR_Z_M, HumanSession, Stroke
from lib.retarget.targets import Segment, Targets, blended, bump, contacts, insert_steps, min_rotation
from lib.retarget.validate import GATES, grade_segment, verdict
from lib.robots import UNITREE_G1

G1 = robot_model(UNITREE_G1)


def test_the_g1_model_is_the_urdf_with_the_paddle_and_no_hands():
    assert G1.nq == 29 and len(G1.body_names) == 31 and G1.body_names[-1] == "paddle"
    assert G1.total_mass == pytest.approx(33.1786, abs=1e-3)  # URDF without the rubber hands, plus the 177.5 g paddle
    assert G1.paddle_offset[0] == pytest.approx(0.2105)
    assert [G1.joint_names[j] for j in G1.joints_moving(G1.torso)] == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    reach = [G1.joint_names[j] for j in G1.joints_moving(G1.paddle_link)]
    assert reach[:3] == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] and len(reach) == 10 and all(n.startswith("right_") for n in reach[3:])
    wrist = G1.joint_names.index("right_wrist_yaw_joint")  # W4010: 5 N m flat to 15.3 rad/s, zero at 24.76, clamped at 22
    for speed, torque in ((0.0, 5.0), (15.3, 5.0), (20.0, 5.0 * (24.76 - 20.0) / (24.76 - 15.3)), (22.5, 0.0)):
        qd = np.zeros(29)
        qd[wrist] = speed
        assert G1.torque_available(qd)[wrist] == pytest.approx(torque)


def test_the_urdf_is_pinned(tmp_path):
    altered = tmp_path / "g1.urdf"
    altered.write_bytes(UNITREE_G1.urdf.read_bytes().replace(b'<mass value="', b'<mass value="1', 1))
    with pytest.raises(ValueError, match="pinned URDF"):
        robot_model(UNITREE_G1.model_copy(update={"urdf": altered}))


def test_kinematics_match_an_independent_chain():
    rng = np.random.default_rng(0)
    q = rng.uniform(G1.lower, G1.upper, (20, G1.nq))
    root_p, root_r = rng.normal(0, 1, (20, 3)), Rotation.random(20, rng=np.random.default_rng(1)).as_matrix()
    rot, pos = body_poses(G1, root_p, root_r, q)
    for n in range(20):  # plain numpy, one pose at a time: parent frame, joint origin, rotation about the axis
        r, p = [root_r[n]], [root_p[n]]
        for i in range(1, len(G1.links)):
            k = G1.parents[i]
            r.append(r[k] @ G1.origin_rot[i] @ Rotation.from_rotvec(G1.axis[i] * q[n, i - 1]).as_matrix())
            p.append(p[k] + r[k] @ G1.origin_xyz[i])
        np.testing.assert_allclose(rot[n, :-1], np.array(r), atol=1e-12)
        np.testing.assert_allclose(pos[n, :-1], np.array(p), atol=1e-12)
        np.testing.assert_allclose(pos[n, -1], p[G1.paddle_link] + r[G1.paddle_link] @ [0.2105, 0, 0], atol=1e-12)
    _r, stance = body_poses(G1, np.array([[0.0, 0.0, 0.78]]), np.eye(3)[None], G1.default_q[None])
    assert stance[0, G1.racket_arm[0], 2] == pytest.approx(1.07, abs=0.01)  # the shoulder 1.07 m up with the pelvis at 0.78 m
    v = torch.tensor([[0.0, 0.0, 0.0], [0.3, -1.2, 2.0]], dtype=torch.float64)
    np.testing.assert_allclose(rotvec_to_matrix(v).numpy(), Rotation.from_rotvec(v.numpy()).as_matrix(), atol=1e-12)


def _stance_height() -> float:
    """Pelvis z that puts the lowest sole corner on the floor in the ready stance."""
    rot, pos = body_poses(G1, np.zeros((1, 3)), np.eye(3)[None], G1.default_q[None])
    corners = np.einsum("fij,kj->fki", rot[0, list(G1.feet)], G1.sole) + pos[0, list(G1.feet)][:, None]
    return FLOOR_Z_M - float(corners[..., 2].min())


def test_a_still_stance_is_supported_by_the_feet():
    t = np.arange(0, 0.6, 0.02)
    n = len(t)
    dyn = inverse_dynamics(G1, t, np.tile([0.0, 0.0, _stance_height()], (n, 1)), np.tile(np.eye(3), (n, 1, 1)), np.tile(G1.default_q, (n, 1)))
    assert np.abs(dyn.force_res).max() < 1e-3  # N: the feet carry the weight, only the regulariser's residue is left
    assert dyn.capture_shift() < 1e-5
    assert np.nanmax(dyn.torque_ratio) < 0.5


def test_target_helpers():
    t = np.linspace(0, 3, 301)
    v = blended(t, np.array([1.0, 2.0]), np.array([[0.0], [1.0]]), 0.1)
    assert v[np.abs(t - 1.0) < 0.1 - 1e-9].max() == 0.0 and v[np.abs(t - 2.0) < 0.1 - 1e-9].min() == 1.0 and np.all(np.diff(v[:, 0]) >= -1e-12)
    assert bump(np.array([0.0, 0.3]), 0.25).tolist() == [1.0, 0.0]
    assert np.allclose(Rotation.from_rotvec(min_rotation(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))).apply([1.0, 0, 0]), [0, 1, 0])
    delta = np.zeros((200, 2))
    delta[100:, 0] = 0.2  # the stance moves 20 cm half way: each foot steps once, never both in the air
    planted, steps = insert_steps(np.ones((200, 2), bool), delta)
    assert sorted(f for f, _a, _b in steps) == [0, 1] and planted.any(1).all()
    heel, toe = torch.tensor([[[0.0, 0.1], [0.0, -0.1]]], dtype=torch.float64), torch.tensor([[[0.17, 0.1], [0.17, -0.1]]], dtype=torch.float64)
    both, left = torch.tensor([[True, True]]), torch.tensor([[True, False]])
    assert float(support_distance(torch.tensor([[0.08, 0.0]], dtype=torch.float64), heel, toe, both)) == pytest.approx(0.0)
    assert float(support_distance(torch.tensor([[0.3, 0.0]], dtype=torch.float64), heel, toe, both)) == pytest.approx(0.13)
    assert float(support_distance(torch.tensor([[0.08, 0.0]], dtype=torch.float64), heel, toe, left)) == pytest.approx(0.1)


# ------------------------------------------------------------------------------------------- a stroke the G1 can make


def _swing(n: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2 s of the ready stance with a slow right-arm swing: time, pelvis position and orientation, joints."""
    t = np.arange(n) / 50.0
    q = np.tile(G1.default_q, (n, 1))
    for name, amp in (("right_shoulder_pitch_joint", -0.5), ("right_shoulder_yaw_joint", 0.3), ("waist_yaw_joint", 0.2)):
        q[:, G1.joint_names.index(name)] += amp * np.sin(np.pi * t / t[-1])
    return t, np.tile([-1.9, 0.0, _stance_height()], (n, 1)), np.tile(np.eye(3), (n, 1, 1)), q


def _human_from(t: np.ndarray, root_p: np.ndarray, q: np.ndarray) -> tuple[HumanSession, Segment, Targets]:
    """A human source whose paddle is the swing's own blade, one stroke at the middle, and the swing's targets."""
    rot, pos = body_poses(G1, root_p, np.tile(np.eye(3), (len(t), 1, 1)), q)
    paddle_r = rot[:, -1][..., [2, 0, 1]]  # paddle frame (x tip, y forehand, z across) -> capture axes (X across, Y tip, Z forehand)
    k = len(t) // 2
    vel = (pos[k + 1, -1] - pos[k - 1, -1]) * 25.0
    stroke = Stroke(
        id="s1_r000_h01", source="paddle_inverse", t=float(t[k]), face=0, blade_p=pos[k, -1], normal=rot[k, -1][:, 1], vel=vel, donor_blade_p=pos[k, -1]
    )
    src = HumanSession(1, -1, t, np.zeros(len(t), int), {}, {}, pos[:, -1], paddle_r, np.ones((len(t), 2), bool), [stroke])
    tg = Targets(
        t=t,
        pelvis_p=root_p,
        pelvis_r=np.tile(np.eye(3), (len(t), 1, 1)),
        torso_r=rot[:, G1.torso],
        foot_p=pos[:, list(G1.feet)],
        foot_yaw=np.zeros((len(t), 2)),
        planted=np.ones((len(t), 2), bool),
        lhand_rel=np.einsum("tji,tj->ti", rot[:, G1.torso], pos[:, G1.free_arm[2]] - pos[:, G1.free_arm[0]]),
        paddle_p=pos[:, -1],
        paddle_r=rot[:, -1],
        near=bump(t - t[k], 0.35),
        contacts=contacts(src, [stroke]),
    )
    return src, Segment(1, t, [stroke]), tg


def test_the_solver_follows_a_motion_the_robot_can_make():
    t, root_p, _r, q = _swing()
    _src, _seg, tg = _human_from(t, root_p, q)
    sol = Problem(tg, G1).solve(adam_iters=200, lbfgs_iters=1000)  # the shipped Adam schedule; L-BFGS shortened (6000 in a run)
    _rot, pos = body_poses(G1, sol.root_p, sol.root_r, sol.q)
    assert np.linalg.norm(pos[:, -1] - tg.paddle_p, axis=1).max() < 0.002  # the blade on its path (0.3 mm measured)
    assert np.linalg.norm(pos[:, list(G1.feet)] - pos[:1, list(G1.feet)], axis=-1).max() < 0.001  # planted feet stay put


def test_the_checks_keep_a_good_stroke_and_reject_each_corruption():
    t, root_p, root_r, q = _swing()
    src, seg, tg = _human_from(t, root_p, q)
    reach = {"s1_r000_h01": np.zeros(2)}

    def grade(rp=root_p, qq=q) -> dict:
        rows, _bad = grade_segment(G1, src, seg, tg, Solution(t, rp, root_r, qq, 0.0, 0.0), reach)
        return rows[0]

    good = grade()
    assert good["kept"], good
    assert good["raw_blade_pos_max_m"] < GATES.blade_pos_m  # here the raw paddle is the path itself
    shifted = grade(rp=root_p + np.array([0.02, 0.0, 0.0]))  # the whole robot 2 cm off: the blade misses its path
    assert not shifted["kept"] and shifted["cause"] == "reach" and shifted["blade_pos_max_m"] > GATES.blade_pos_m
    sliding = grade(rp=root_p + np.c_[np.linspace(0, 0.05, len(t)), np.zeros((len(t), 2))])  # the planted feet skate 5 cm
    assert not sliding["kept"] and sliding["foot_slide_max_m"] > GATES.foot_slide_m
    bent = q.copy()
    elbow = G1.joint_names.index("left_elbow_joint")
    bent[:, elbow] = G1.upper[elbow] - 0.01 * (G1.upper[elbow] - G1.lower[elbow])  # the free elbow at 0.98 of its range
    assert grade(qq=bent)["cause"] == "reach" and grade(qq=bent)["limit_use_max"] > GATES.limit_fraction
    shaking = q.copy()
    wrist = G1.joint_names.index("left_wrist_roll_joint")
    shaking[:, wrist] += 0.3 * np.sin(2 * np.pi * 6.0 * t)  # the free wrist shaking at 11 rad/s peak: 0.31 of its limit, fine
    assert grade(qq=shaking)["kept"]
    shaking[:, wrist] = G1.default_q[wrist] + 1.2 * np.sin(2 * np.pi * 6.0 * t)  # at 45 rad/s: past the 37 rad/s limit
    too_fast = grade(qq=shaking)
    assert too_fast["cause"] == "speed" and too_fast["speed_ratio_max"] > 1.0


def test_verdict_names_the_first_failing_check():
    base = {
        "blade_pos_max_m": 0.005,
        "contact_pos_m": 0.002,
        "blade_face_max_deg": 2.0,
        "contact_face_deg": 1.0,
        "limit_use_max": 0.9,
        "speed_ratio_max": 0.5,
        "torque_ratio_max": 0.8,
        "capture_shift_m": 0.01,
        "foot_slide_max_m": 0.002,
        "foot_penetration_max_m": 0.0,
    }
    assert verdict(base) == {"kept": True, "cause": ""}
    for change, cause in (
        ({"blade_pos_max_m": 0.02}, "reach"),
        ({"contact_face_deg": 6.0}, "reach"),
        ({"limit_use_max": 0.96}, "reach"),
        ({"speed_ratio_max": 0.85}, "speed"),
        ({"torque_ratio_max": 1.2}, "torque"),
        ({"capture_shift_m": 0.05}, "balance"),
        ({"foot_slide_max_m": 0.02}, "feet"),
        ({"foot_penetration_max_m": 0.01}, "feet"),
        ({"blade_pos_max_m": 0.02, "capture_shift_m": 0.05}, "reach"),
    ):
        assert verdict(base | change)["cause"] == cause
    assert verdict(base | {"contact_pos_m": np.nan, "contact_face_deg": np.nan})["kept"]  # capture strokes have no solved contact


def test_the_package_round_trips_and_is_rechecked(tmp_path: Path):
    t, root_p, root_r, q = _swing()
    clip = Clip(1, -1, t, root_p, root_r, q, np.ones((len(t), 2), bool), [("s1_r000_h01", float(t[50]))])
    row = {
        "stroke_id": "s1_r000_h01",
        "session": 1,
        "source": "paddle_inverse",
        "face": 0,
        "kind": "forehand-stance",
        "t_contact_s": float(t[50]),
        "kept": True,
        "cause": "",
        "blade_p": [0.0, 0.0, 0.0],
        "blade_v": [0.0, 0.0, 0.0],
        "blade_n": [0.0, 1.0, 0.0],
        "blade_pos_max_m": 0.004,
        "blade_face_max_deg": 1.0,
        "contact_pos_m": 0.001,
        "contact_face_deg": 0.5,
        "contact_vel_err_mps": 0.1,
        "speed_ratio_max": 0.3,
        "torque_ratio_max": 0.6,
        "capture_shift_m": 0.01,
        "foot_slide_max_m": 0.0,
    }
    path = write_package(tmp_path / "pkg", G1, [clip], [row], {"run": "test"})
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist()) == {"manifest.json", "motion.npz", "strokes.csv"}
    d = np.load(tmp_path / "pkg" / "motion.npz")
    assert list(d["joint_names"]) == G1.joint_names and d["body_pos_w"].shape == (len(t), 31, 3)
    assert np.allclose(np.linalg.norm(d["body_quat_w"], axis=-1), 1.0, atol=1e-6) and bool(d["contact"][50]) and int(d["stroke_frame"][0]) == 50
    check = recheck_package(path, G1)
    assert check["ok"] and check["foot_slide_max_m"] < 1e-6, check
    assert kept_pieces(t, np.zeros(len(t), bool), [float(t[50])]) == []  # a 2 s rally cut 0.6 s + 0.4 s around its one rejected stroke, grown 0.3 s
    assert kept_pieces(t, np.zeros(len(t), bool), []) == [(0, len(t))]
