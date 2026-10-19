"""
Anatomical joint angles of a posed skeleton, the one measurement the joint limits and the arm IK share.

**Frames.** Every skeleton this project poses rests in the same T-pose: the body faces -Y with +Z up, the arms point
along the body's sides (the right arm along -X) with the palms down and the thumbs forward (the forearm in its
neutral, mid-pronation position), the legs straight and the feet forward. Both the ActorCore CC3 bodies and the
Rokoko capture skeleton rest that way (checked by forward kinematics of their rest poses), so a joint's rotation
*relative to its rest*, expressed in the rest pose's world axes, has the same anatomical meaning on either skeleton.

For a joint whose anatomical parent segment has world rotation ``P`` (rest ``P0``) and whose own segment has ``C``
(rest ``C0``), the rest-relative delta in rest-world axes is ``D = (P P0^T)^T (C C0^T)``; it is the identity at rest
whatever the bones' own local axes are. ``D`` is then written in the anatomical basis of the right side, ``x``
lateral (the body's right, world -X), ``y`` forward (world -Y), ``z`` up (world +Z), and a left-side joint is first
mirrored through the sagittal plane, so the two sides share one sign convention.

**Decompositions** (proximal to distal, intrinsic; signs as named; all angles in degrees):

- *Elbow*: the forearm's direction in the humerus frame. Flexion is its angle about the hinge axis (``z`` at rest:
  palm down, the forearm neutral) from straight, positive bending forward; a negative flexion is hyperextension.
  ``elbow_offplane`` is its angle out of the hinge plane (a hinge joint has none; the skeleton's 3-DOF elbow can).
- *Forearm and wrist*: the hand relative to the humerus with the elbow's flexion and off-plane angle removed
  (``F0``), as ``R_x(-pronation) R_y(wrist flexion) R_z(radial deviation)``. Pronation therefore counts every twist
  between the humerus and the hand about the forearm, whether the skeleton carries it on the forearm or the hand.
  Positive: pronation, wrist flexion (palmar), radial deviation (toward the thumb).
- *Shoulder* (humerothoracic, the humerus relative to the chest): from the hanging arm, ``elevation`` (0 hanging,
  90 horizontal, 180 overhead), ``plane`` of elevation (0 abduction in the coronal plane, +90 forward flexion,
  negative behind the body, above 90 across it) and axial ``rotation`` (positive internal) as the twist about the
  humerus left after the minimal-arc swing from hanging (defined everywhere but straight overhead).
- *Clavicle*: the clavicle's direction in the chest frame: ``elevation`` (lateral end up) and ``protraction``
  (lateral end forward).
- *Spine* (chest relative to pelvis) and *neck* (head relative to chest): ``R_x R_y R_z`` as ``flexion`` (forward),
  ``lateral`` bend (toward the body's right, positive) and ``twist`` (toward the body's left, positive).
- *Hip* (thigh relative to pelvis): ``flexion`` (forward), ``abduction`` (away from the midline), ``rotation``
  (internal positive). *Knee*: ``flexion`` (positive bending), ``varus``, ``rotation``. *Ankle*: ``dorsiflexion``,
  ``inversion``, ``adduction`` (toe in).

The angles of a frame are pure functions of its joints' world rotations, so they apply to the capture, the session
tracks and the replay's played poses alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.replay.players import rotations as rot

Array = NDArray[Any]

BASIS: Array = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
"""Columns: the right side's lateral, forward and up axes in rest-world coordinates (the rest body faces -Y, +Z up)."""
MIRROR: Array = np.diag([-1.0, 1.0, 1.0])
"""Reflection through the sagittal plane (world X): a left-side delta ``M D M`` reads as a right-side one."""


@dataclass(frozen=True)
class AnatomyMap:
    """
    Which joint of one skeleton carries each anatomical segment (``{side}`` is ``R`` or ``L`` where it appears).

    :ivar pelvis: The pelvis segment (the thighs' parent).
    :ivar thorax: The chest (the clavicles' parent).
    :ivar head: The head.
    :ivar clavicle: Clavicle joint, with ``{side}``.
    :ivar humerus: Upper-arm joint.
    :ivar forearm: Forearm joint (the elbow).
    :ivar hand: Hand joint (the wrist).
    :ivar thigh: Thigh joint (the hip).
    :ivar shank: Shank joint (the knee).
    :ivar foot: Foot joint (the ankle).
    :ivar forearm_twists: Forearm twist joints, proximal first (skin carriers; may be empty).
    """

    pelvis: str
    thorax: str
    head: str
    clavicle: str
    humerus: str
    forearm: str
    hand: str
    thigh: str
    shank: str
    foot: str
    forearm_twists: tuple[str, ...] = ()

    def joint(self, segment: str, side: str = "R") -> str:
        """The joint name of a segment on a side."""

        return str(getattr(self, segment)).format(side=side)


ACTORCORE = AnatomyMap(
    pelvis="Pelvis",
    thorax="Spine02",
    head="Head",
    clavicle="{side}_Clavicle",
    humerus="{side}_Upperarm",
    forearm="{side}_Forearm",
    hand="{side}_Hand",
    thigh="{side}_Thigh",
    shank="{side}_Calf",
    foot="{side}_Foot",
    forearm_twists=("{side}_ForearmTwist01", "{side}_ForearmTwist02"),
)
"""The ActorCore CC3 skeleton (both replay bodies)."""

_ROKOKO_SIDE = {"R": "Right", "L": "Left"}

ROKOKO = AnatomyMap(
    pelvis="Hips",
    thorax="Spine4",
    head="Head",
    clavicle="{side}Shoulder",
    humerus="{side}Arm",
    forearm="{side}ForeArm",
    hand="{side}Hand",
    thigh="{side}Thigh",
    shank="{side}Shin",
    foot="{side}Foot",
)
"""The Rokoko capture skeleton; its sides are spelled ``Right``/``Left`` (:func:`side_name`)."""


def side_name(anatomy: AnatomyMap, side: str) -> str:
    """The ``{side}`` spelling a skeleton uses (``R``/``L`` for ActorCore, ``Right``/``Left`` for Rokoko)."""

    return _ROKOKO_SIDE[side] if anatomy is ROKOKO else side


SIDES: tuple[str, ...] = ("R", "L")


def _matrices(q: ArrayLike) -> Array:
    return rot.to_matrix(np.asarray(q, dtype=np.float64))


def _delta(parent: Array, parent_rest: Array, child: Array, child_rest: Array, *, mirror: bool) -> Array:
    """The child's rest-relative rotation in its parent's rest frame, in the right side's anatomical basis, ``(F, 3, 3)``."""

    dp = parent @ np.swapaxes(parent_rest, -1, -2)
    dc = child @ np.swapaxes(child_rest, -1, -2)
    d = np.swapaxes(dp, -1, -2) @ dc
    if mirror:
        d = MIRROR @ d @ MIRROR
    return BASIS.T @ d @ BASIS


def _xyz(r: Array) -> tuple[Array, Array, Array]:
    """Intrinsic ``R = R_x(a) R_y(b) R_z(c)``: ``(a, b, c)`` in radians (``b`` in [-90, 90] degrees)."""

    b = np.arcsin(np.clip(r[..., 0, 2], -1.0, 1.0))
    a = np.arctan2(-r[..., 1, 2], r[..., 2, 2])
    c = np.arctan2(-r[..., 0, 1], r[..., 0, 0])
    return a, b, c


def _rx(angle: Array) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    out = np.zeros((*np.shape(angle), 3, 3))
    out[..., 1, 1], out[..., 1, 2], out[..., 2, 1], out[..., 2, 2], out[..., 0, 0] = c, -s, s, c, 1.0
    return out


def _rz(angle: Array) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    out = np.zeros((*np.shape(angle), 3, 3))
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1], out[..., 2, 2] = c, -s, s, c, 1.0
    return out


def _ry(angle: Array) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    out = np.zeros((*np.shape(angle), 3, 3))
    out[..., 0, 0], out[..., 0, 2], out[..., 2, 0], out[..., 2, 2], out[..., 1, 1] = c, s, -s, c, 1.0
    return out


def _forearm_and_wrist(forearm: Array, hand: Array) -> tuple[dict[str, Array], Array]:
    """
    Elbow, forearm and wrist angles (radians) from the forearm's and hand's deltas relative to the humerus, and the
    untwisted forearm frame ``F0`` (the humerus turned by the elbow's flexion and off-plane angle only).
    """

    v = forearm[..., :, 0]
    flexion = np.arctan2(v[..., 1], v[..., 0])
    offplane = np.arcsin(np.clip(v[..., 2], -1.0, 1.0))
    untwisted = _rz(flexion) @ _ry(-offplane)
    g = np.swapaxes(untwisted, -1, -2) @ hand
    a, b, c = _xyz(g)
    return {"elbow_flexion": flexion, "elbow_offplane": offplane, "forearm_pronation": -a, "wrist_flexion": b, "wrist_deviation": c}, untwisted


def _shoulder(d: Array) -> dict[str, Array]:
    """Humerothoracic angles (radians) from the humerus's delta relative to the chest (rest: the T-pose's 90 degree abduction)."""

    hanging_to_t = _ry(np.full(d.shape[:-2], -np.pi / 2.0))
    q = d @ hanging_to_t
    h = q[..., :, 2] * -1.0
    elevation = np.arccos(np.clip(-h[..., 2], -1.0, 1.0))
    plane = np.arctan2(h[..., 1], h[..., 0])
    down = np.broadcast_to(np.array([0.0, 0.0, -1.0]), h.shape)
    swing = rot.to_matrix(rot.min_arc(down, h))
    twist = np.swapaxes(swing, -1, -2) @ q
    rotation = np.arctan2(twist[..., 1, 0], twist[..., 0, 0])
    return {"shoulder_elevation": elevation, "shoulder_plane": plane, "shoulder_rotation": rotation}


def anatomical_angles(
    world_rotation: ArrayLike,
    rest_world_rotation: ArrayLike,
    joints: tuple[str, ...] | list[str],
    anatomy: AnatomyMap = ACTORCORE,
    *,
    sides: tuple[str, ...] = SIDES,
    twists: bool = True,
    hinge_azimuth_deg: Mapping[str, float] | None = None,
) -> dict[str, Array]:
    """
    Every anatomical angle of every frame, degrees.

    :param world_rotation: World joint rotations ``(F, J, 4)`` (wxyz) or matrices ``(F, J, 3, 3)``.
    :param rest_world_rotation: The rest pose's world joint rotations ``(J, 4)`` or ``(J, 3, 3)``.
    :param joints: Joint names in that order.
    :param anatomy: The skeleton's segment joints.
    :param sides: Sides to measure.
    :param twists: Also report each forearm twist joint's share of the pronation (``{side}.forearm_twist{k}``, its
        twist about the forearm in the same decomposition; the skin between them wrings by their differences).
    :param hinge_azimuth_deg: Per side, where the skeleton's elbow hinge sits about the humerus in the rest pose: the
        direction the forearm bends, from forward (0, the anatomical T-pose: palm down, the crease forward) toward up.
        The humerus's anatomical frame is the bone's turned by it about its long axis (a functional calibration: the
        median bend direction of the skeleton's own bent-elbow frames), so a hinge elbow reads no off-plane angle and
        the pronation and humeral rotation are measured about the real hinge. ``None``: 0 on both sides.
    :return: ``{side}.{angle}`` for :data:`ANGLE_NAMES` and the names of :data:`AXIAL_NAMES`, each ``(F,)``.
    """

    w = np.asarray(world_rotation, dtype=np.float64)
    r = np.asarray(rest_world_rotation, dtype=np.float64)
    w = w if w.shape[-2:] == (3, 3) else _matrices(w)
    r = r if r.shape[-2:] == (3, 3) else _matrices(r)
    index = {name: k for k, name in enumerate(joints)}

    def seg(name: str) -> tuple[Array, Array]:
        k = index[name]
        return w[:, k], r[k]

    out: dict[str, Array] = {}
    pelvis, pelvis0 = seg(anatomy.pelvis)
    thorax, thorax0 = seg(anatomy.thorax)
    head, head0 = seg(anatomy.head)
    for prefix, (child, child0, parent, parent0) in (("spine", (thorax, thorax0, pelvis, pelvis0)), ("neck", (head, head0, thorax, thorax0))):
        a, b, c = _xyz(_delta(parent, parent0, child, child0, mirror=False))
        out[f"{prefix}_flexion"], out[f"{prefix}_lateral"], out[f"{prefix}_twist"] = -a, b, c
    for side in sides:
        spelled = side_name(anatomy, side)
        mirror = side == "L"

        def joint(segment: str, spelled: str = spelled) -> tuple[Array, Array]:
            return seg(anatomy.joint(segment, spelled))

        clavicle, clavicle0 = joint("clavicle")
        humerus, humerus0 = joint("humerus")
        forearm, forearm0 = joint("forearm")
        hand, hand0 = joint("hand")
        thigh, thigh0 = joint("thigh")
        shank, shank0 = joint("shank")
        foot, foot0 = joint("foot")
        angles: dict[str, Array] = {}
        c_dir = _delta(thorax, thorax0, clavicle, clavicle0, mirror=mirror)[..., :, 0]
        angles["clavicle_elevation"] = np.arcsin(np.clip(c_dir[..., 2], -1.0, 1.0))
        angles["clavicle_protraction"] = np.arctan2(c_dir[..., 1], c_dir[..., 0])
        hinge = _rx(np.radians((hinge_azimuth_deg or {}).get(side, 0.0)))
        hinge_t = np.swapaxes(hinge, -1, -2)
        angles.update(_shoulder(_delta(thorax, thorax0, humerus, humerus0, mirror=mirror) @ hinge))
        arm, untwisted = _forearm_and_wrist(
            hinge_t @ _delta(humerus, humerus0, forearm, forearm0, mirror=mirror),
            hinge_t @ _delta(humerus, humerus0, hand, hand0, mirror=mirror),
        )
        angles.update(arm)
        if twists and anatomy.forearm_twists:
            for k, name in enumerate(anatomy.forearm_twists, start=1):
                twist_joint = name.format(side=spelled)
                if twist_joint not in index:
                    continue
                tw, tw0 = seg(twist_joint)
                g = np.swapaxes(untwisted, -1, -2) @ hinge_t @ _delta(humerus, humerus0, tw, tw0, mirror=mirror)
                angles[f"forearm_twist{k}"] = -_xyz(g)[0]
        a, b, c = _xyz(_delta(pelvis, pelvis0, thigh, thigh0, mirror=mirror))
        angles["hip_flexion"], angles["hip_abduction"], angles["hip_rotation"] = a, -b, c
        a, b, c = _xyz(_delta(thigh, thigh0, shank, shank0, mirror=mirror))
        angles["knee_flexion"], angles["knee_varus"], angles["knee_rotation"] = -a, b, c
        a, b, c = _xyz(_delta(shank, shank0, foot, foot0, mirror=mirror))
        angles["ankle_dorsiflexion"], angles["ankle_inversion"], angles["ankle_adduction"] = a, b, c
        out.update({f"{side}.{name}": value for name, value in angles.items()})
    return {name: np.degrees(value) for name, value in out.items()}


__all__ = ["ACTORCORE", "BASIS", "MIRROR", "ROKOKO", "SIDES", "AnatomyMap", "anatomical_angles", "side_name"]
