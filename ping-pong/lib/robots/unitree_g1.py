"""
The Unitree G1 29-DoF (rev 1.0, standard 4010 wrist) with a paddle in place of its right hand, as a
:class:`~lib.robots.spec.RobotSpec`.

- **Body:** Isaac's G1 USD with the two Dex3 hands deactivated; every link collides through the URDF's collider
  (``urdf/``, Unitree's BSD-3 URDF from ``unitree_ros`` at 5994d4faef).
- **Motors:** five motor types. Peak torque and joint speed limit come from Unitree's URDF; the torque-speed knee,
  no-load speed and reflected rotor inertia come from Unitree's measured actuator table (``unitree_rl_lab``). The
  exception is the ankles and the waist roll and pitch, each driven by two N5020 through a linkage: they take the
  actuator table's 50 N m and 37 rad/s, where the URDF writes 35 N m and 30 rad/s. The PD gains are Unitree's
  deployed G1 gains (arm kp 40, kd 1).
- **Trained on exactly this:** the policy in ``lib/policy`` was trained and graded with these motors, this default
  pose (``v2``: the right elbow raised 0.1 rad so the paddle clears the table at every reset) and these action scales.
  Change any of them and the policy no longer matches its robot.
- **Actions:** a unit action asks for a quarter of the joint's peak torque through its PD (BeyondMimic's scale),
  added to the ready stance.
- **Colliders:** the USD gives only the pelvis, torso, knees, feet and wrist-yaw links a collider; every other link gets
  the URDF's (:data:`G1_COLLIDERS`), so the arms and legs meet the table and the robot itself.
- **Paddle:** the ``ping-pong/paddle`` asset fixed 0.2105 m along ``right_wrist_yaw_link``'s +x: the handle butt
  sits in the wrist flange 0.0415 m out, and the blade centre is 0.169 m beyond it. Mass properties are the asset's
  mesh volumes at plywood and rubber densities: 177.5 g, centre of mass 17.4 mm behind the blade centre.
"""

from __future__ import annotations

import math
from pathlib import Path

from lib.robots.spec import CylinderCollider, FallRule, LinkColliders, MotorGroup, PaddleMount, RetargetSpec, RobotLinks, RobotSpec

URDF: Path = Path(__file__).with_name("urdf") / "g1_29dof_rev_1_0.urdf"
"""Unitree's G1 29-DoF URDF, vendored unmodified with its BSD 3-Clause license (``urdf/LICENSE``, ``urdf/NOTICE``)."""

_SIDES = ("left", "right")

G1_COLLIDERS = LinkColliders(
    # The URDF collides these links with their own meshes, which the USD's visuals are; the USD gives them none.
    mesh_hull_links=tuple(
        f"{side}_{link}_link"
        for side in _SIDES
        for link in ("hip_pitch", "hip_roll", "hip_yaw", "ankle_pitch", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch")
    ),
    cylinders={
        "left_shoulder_pitch_link": CylinderCollider(radius_m=0.03, length_m=0.05, xyz_m=(0.0, 0.04, -0.01), rpy_rad=(0.0, math.pi / 2, 0.0)),
        "right_shoulder_pitch_link": CylinderCollider(radius_m=0.03, length_m=0.05, xyz_m=(0.0, -0.04, -0.01), rpy_rad=(0.0, math.pi / 2, 0.0)),
        "left_shoulder_roll_link": CylinderCollider(radius_m=0.03, length_m=0.03, xyz_m=(-0.004, 0.006, -0.053), rpy_rad=(0.0, 0.0, 0.0)),
        "right_shoulder_roll_link": CylinderCollider(radius_m=0.03, length_m=0.03, xyz_m=(-0.004, -0.006, -0.053), rpy_rad=(0.0, 0.0, 0.0)),
    },
    # The URDF gives the two waist motor housings, enclosed by the pelvis and the torso, no collision.
    fitted_capsule_links=("waist_yaw_link", "waist_roll_link"),
    # The waist-yaw housing's capsule lies inside the torso's hull at every waist angle: unfiltered, PhysX pushes the
    # two apart with about 2 MN in the ready stance.
    filtered_pairs=(("waist_yaw_link", "torso_link"),),
)
"""The URDF's colliders for every G1 link the USD leaves without one."""

_ARM = (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint")
_WRIST = (".*_wrist_pitch_joint", ".*_wrist_yaw_joint")

UNITREE_G1 = RobotSpec(
    name="unitree-g1",
    usd="{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
    prune=("_hand_",),
    urdf=URDF,
    urdf_sha256="c0ae739c640c3e2c00d1bdd8810b5d6e59601487bd1a3995859f9543269ee5c8",
    joint_names=(
        *(f"{side}_{joint}_joint" for side in _SIDES for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        *(
            f"{side}_{joint}_joint"
            for side in _SIDES
            for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
        ),
    ),
    actuators=(
        MotorGroup(
            name="N7520-14.3",
            joints=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
            peak_torque=88.0,
            speed_limit=32.0,
            knee_speed=22.63,
            no_load_speed=35.52,
            armature=0.01017752,
            kp={".*_hip_pitch_joint": 100.0, ".*_hip_yaw_joint": 100.0, "waist_yaw_joint": 200.0},
            kd={".*_hip_pitch_joint": 2.0, ".*_hip_yaw_joint": 2.0, "waist_yaw_joint": 5.0},
        ),
        MotorGroup(
            name="N7520-22.5",
            joints=(".*_hip_roll_joint", ".*_knee_joint"),
            peak_torque=139.0,
            speed_limit=20.0,
            knee_speed=14.5,
            no_load_speed=22.7,
            armature=0.025101925,
            kp={".*_hip_roll_joint": 100.0, ".*_knee_joint": 150.0},
            kd={".*_hip_roll_joint": 2.0, ".*_knee_joint": 4.0},
        ),
        # The ankles and the waist roll and pitch are each driven by two N5020 through a linkage.
        MotorGroup(
            name="2xN5020-16",
            joints=(".*_ankle_pitch_joint", ".*_ankle_roll_joint", "waist_roll_joint", "waist_pitch_joint"),
            peak_torque=50.0,
            speed_limit=37.0,
            knee_speed=30.86,
            no_load_speed=40.13,
            armature=2 * 0.003609725,
            kp={".*_ankle_pitch_joint": 40.0, ".*_ankle_roll_joint": 40.0, "waist_roll_joint": 40.0, "waist_pitch_joint": 40.0},
            kd={".*_ankle_pitch_joint": 2.0, ".*_ankle_roll_joint": 2.0, "waist_roll_joint": 5.0, "waist_pitch_joint": 5.0},
        ),
        MotorGroup(
            name="N5020-16",
            joints=_ARM,
            peak_torque=25.0,
            speed_limit=37.0,
            knee_speed=30.86,
            no_load_speed=40.13,
            armature=0.003609725,
            kp=dict.fromkeys(_ARM, 40.0),
            kd=dict.fromkeys(_ARM, 1.0),
        ),
        MotorGroup(
            name="W4010-25",
            joints=_WRIST,
            peak_torque=5.0,
            speed_limit=22.0,
            knee_speed=15.3,
            no_load_speed=24.76,
            armature=0.00425,
            kp=dict.fromkeys(_WRIST, 40.0),
            kd=dict.fromkeys(_WRIST, 1.0),
        ),
    ),
    action_fraction=0.25,
    # The ready stance: slight knee bend, arms relaxed, the right forearm forward with the elbow raised 0.1 rad so
    # the paddle clears the table at every start position.
    default_pose={
        ".*_hip_pitch_joint": -0.10,
        ".*_knee_joint": 0.30,
        ".*_ankle_pitch_joint": -0.20,
        "left_shoulder_roll_joint": 0.20,
        "right_shoulder_roll_joint": -0.20,
        "left_elbow_joint": 0.6,
        "right_elbow_joint": -0.1,
    },
    base_height_m=0.78,
    paddle=PaddleMount(
        link="right_wrist_yaw_link",
        offset_m=0.0415 + 0.169,
        mass_kg=0.1775,
        com_m=(-0.0174, 0.0, 0.0),
        inertia=(2.22e-4, 8.11e-4, 5.97e-4),
        blade_half_length=0.0785,
        blade_half_width=0.075,
        blade_thickness=0.0136,
        handle_size=(0.10, 0.023, 0.034),
        handle_centre_x=-0.119,
    ),
    links=RobotLinks(
        pelvis="pelvis",
        torso="torso_link",
        feet=("left_ankle_roll_link", "right_ankle_roll_link"),
        racket_arm=(
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
            "right_wrist_pitch_link",
            "right_wrist_yaw_link",
        ),
    ),
    fall=FallRule(pelvis_below_m=0.45, tilt_above_rad=1.0),
    stance_m=(0.35, 0.75),
    colliders=G1_COLLIDERS,
    retarget=RetargetSpec(
        racket_arm=("right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link"),
        free_arm=("left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link"),
        urdf_drop_links=("left_rubber_hand", "right_rubber_hand"),
        # The URDF's four contact spheres on each ankle-roll link, at the sole.
        sole_corners_m=((-0.05, 0.025, -0.035), (-0.05, -0.025, -0.035), (0.12, 0.03, -0.035), (0.12, -0.03, -0.035)),
        # Coarse spheres per link for the retargeter's self-collision, table and floor penalties, sized from the joint
        # offsets and the URDF meshes' extents: conservative stand-ins, not the PhysX colliders. The paddle's own
        # spheres are added by lib/retarget.
        collision_spheres={
            "pelvis": (((0.0, 0.0, -0.03), 0.1), ((0.0, 0.07, -0.08), 0.07), ((0.0, -0.07, -0.08), 0.07)),
            "torso_link": (((0.0, 0.0, 0.1), 0.11), ((0.0, 0.0, 0.22), 0.11), ((0.0, 0.0, 0.45), 0.09)),
            "left_hip_yaw_link": (((-0.02, 0.0, -0.05), 0.065), ((-0.02, 0.0, -0.11), 0.065), ((-0.02, 0.0, -0.17), 0.065)),
            "left_knee_link": (((0.0, 0.0, -0.04), 0.05), ((0.0, 0.0, -0.125), 0.05), ((0.0, 0.0, -0.21), 0.05), ((0.0, 0.0, -0.295), 0.05)),
            "left_ankle_roll_link": (((0.0, 0.0, -0.015), 0.03), ((0.08, 0.0, -0.015), 0.03)),
            "left_shoulder_roll_link": (((0.0, 0.0, -0.05), 0.05),),
            "left_shoulder_yaw_link": (((0.0, 0.0, -0.03), 0.045), ((0.01, 0.0, -0.08), 0.045)),
            "left_elbow_link": (((0.03, 0.0, -0.005), 0.04), ((0.08, 0.0, -0.008), 0.035)),
            "left_wrist_roll_link": (((0.02, 0.0, 0.0), 0.035),),
            "left_wrist_yaw_link": (((0.02, 0.0, 0.0), 0.03),),
            "right_hip_yaw_link": (((-0.02, 0.0, -0.05), 0.065), ((-0.02, 0.0, -0.11), 0.065), ((-0.02, 0.0, -0.17), 0.065)),
            "right_knee_link": (((0.0, 0.0, -0.04), 0.05), ((0.0, 0.0, -0.125), 0.05), ((0.0, 0.0, -0.21), 0.05), ((0.0, 0.0, -0.295), 0.05)),
            "right_ankle_roll_link": (((0.0, 0.0, -0.015), 0.03), ((0.08, 0.0, -0.015), 0.03)),
            "right_shoulder_roll_link": (((0.0, 0.0, -0.05), 0.05),),
            "right_shoulder_yaw_link": (((0.0, 0.0, -0.03), 0.045), ((0.01, 0.0, -0.08), 0.045)),
            "right_elbow_link": (((0.03, 0.0, -0.005), 0.04), ((0.08, 0.0, -0.008), 0.035)),
            "right_wrist_roll_link": (((0.02, 0.0, 0.0), 0.035),),
            "right_wrist_yaw_link": (((0.02, 0.0, 0.0), 0.03),),
        },
    ),
)
"""The Unitree G1 29-DoF with the paddle."""

__all__ = ["G1_COLLIDERS", "UNITREE_G1", "URDF"]
