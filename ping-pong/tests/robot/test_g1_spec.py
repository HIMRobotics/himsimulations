"""
The Unitree G1's spec (:data:`lib.robots.UNITREE_G1`) against its vendored URDF, and against the robot the trained
policy played on: joints, links, the action scale, the default pose and the observation's width.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ElementTree

import pytest

from lib.policy.actor import ACTOR_INPUTS, CHECKPOINT_INPUTS, POLICY_JOINTS, ROBOT_INPUTS, width
from lib.robots import ROBOTS, UNITREE_G1
from lib.robots.table_scene import observation_terms

TRAINED_ACTION_SCALE_RAD: dict[str, float] = {
    ".*_hip_pitch_joint": 0.22,
    ".*_hip_yaw_joint": 0.22,
    ".*_hip_roll_joint": 0.3475,
    ".*_knee_joint": 0.23166666666666666,
    "waist_yaw_joint": 0.11,
    "waist_roll_joint": 0.3125,
    "waist_pitch_joint": 0.3125,
    ".*_ankle_pitch_joint": 0.3125,
    ".*_ankle_roll_joint": 0.3125,
    ".*_shoulder_pitch_joint": 0.15625,
    ".*_shoulder_roll_joint": 0.15625,
    ".*_shoulder_yaw_joint": 0.15625,
    ".*_elbow_joint": 0.15625,
    ".*_wrist_roll_joint": 0.15625,
    ".*_wrist_pitch_joint": 0.03125,
    ".*_wrist_yaw_joint": 0.03125,
}
"""The per-joint action scale the policy was trained and graded with, rad per unit action."""

TRAINED_DEFAULT_POSE: dict[str, float] = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": -0.1,
}
"""The default pose the checkpoint records (its action origin)."""


@pytest.fixture(scope="module")
def urdf() -> ElementTree.Element:
    return ElementTree.parse(UNITREE_G1.urdf).getroot()


def test_the_registry_holds_the_g1() -> None:
    assert ROBOTS == {"unitree-g1": UNITREE_G1}


def test_the_urdf_is_the_pinned_one() -> None:
    assert hashlib.sha256(UNITREE_G1.urdf.read_bytes()).hexdigest() == UNITREE_G1.urdf_sha256


def test_the_joints_are_the_urdf_revolute_joints_in_order(urdf: ElementTree.Element) -> None:
    revolute = [joint.get("name") for joint in urdf.iter("joint") if joint.get("type") == "revolute"]
    assert list(UNITREE_G1.joint_names) == revolute
    assert sorted(POLICY_JOINTS) == sorted(UNITREE_G1.joint_names), "the policy's joint order is a permutation of the spec's joints"


def test_every_named_link_is_a_urdf_link(urdf: ElementTree.Element) -> None:
    links = {link.get("name") for link in urdf.iter("link")}
    spec = UNITREE_G1
    named = {spec.links.pelvis, spec.links.torso, *spec.links.feet, *spec.links.racket_arm, spec.paddle.link}
    named |= {*spec.colliders.mesh_hull_links, *spec.colliders.cylinders, *spec.colliders.fitted_capsule_links}
    named |= {link for pair in spec.colliders.filtered_pairs for link in pair}
    assert spec.retarget is not None
    named |= {*spec.retarget.racket_arm, *spec.retarget.free_arm, *spec.retarget.urdf_drop_links}
    named |= set(spec.retarget.collision_spheres)
    assert named - links == set()


def test_the_urdf_cylinders_are_the_specs(urdf: ElementTree.Element) -> None:
    for link in urdf.iter("link"):
        name = link.get("name")
        if name not in UNITREE_G1.colliders.cylinders:
            continue
        cylinder = link.find("collision/geometry/cylinder")
        origin = link.find("collision/origin")
        assert cylinder is not None and origin is not None
        spec = UNITREE_G1.colliders.cylinders[name]
        assert float(cylinder.get("radius", "nan")) == pytest.approx(spec.radius_m)
        assert float(cylinder.get("length", "nan")) == pytest.approx(spec.length_m)
        assert [float(v) for v in origin.get("xyz", "").split()] == pytest.approx(spec.xyz_m)
        assert [float(v) for v in origin.get("rpy", "").split()] == pytest.approx(spec.rpy_rad, abs=1e-4)


LINKAGE_GROUP = "2xN5020-16"
"""The ankles and the waist roll and pitch, each driven by two N5020 through a linkage: the policy trained with the
actuator table's 50 N m and 37 rad/s, where this URDF writes 35 N m and 30 rad/s."""


def test_the_motor_limits_are_the_urdfs_but_the_linkage_groups(urdf: ElementTree.Element) -> None:
    limits = {joint.get("name"): joint.find("limit") for joint in urdf.iter("joint") if joint.get("type") == "revolute"}
    for name in UNITREE_G1.joint_names:
        motor = UNITREE_G1.motor(name)
        limit = limits[name]
        assert limit is not None
        effort, velocity = float(limit.get("effort", "nan")), float(limit.get("velocity", "nan"))
        if motor.name == LINKAGE_GROUP:
            assert (effort, velocity, motor.peak_torque, motor.speed_limit) == (35.0, 30.0, 50.0, 37.0), name
            continue
        assert motor.peak_torque == pytest.approx(effort), name
        assert motor.speed_limit == pytest.approx(velocity), name


def test_the_action_scale_and_default_pose_are_the_trained_ones() -> None:
    assert UNITREE_G1.action_scale == pytest.approx(TRAINED_ACTION_SCALE_RAD)
    assert UNITREE_G1.default_pose == TRAINED_DEFAULT_POSE
    assert UNITREE_G1.paddle.offset_m == pytest.approx(0.2105)


def test_the_observation_is_3j_plus_29_plus_the_forecast() -> None:
    j = UNITREE_G1.num_joints
    assert width(observation_terms(j)) == 3 * j + 29 == width(ROBOT_INPUTS) == 116
    assert width(ACTOR_INPUTS) == 146
    assert width(CHECKPOINT_INPUTS) - width(ACTOR_INPUTS) == 102


def test_every_default_pose_and_scale_expression_names_a_joint() -> None:
    for expression in (*UNITREE_G1.default_pose, *UNITREE_G1.action_scale):
        assert any(re.fullmatch(expression, joint) for joint in UNITREE_G1.joint_names), expression
