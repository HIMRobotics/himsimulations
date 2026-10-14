"""
The shakehand grip: one consistent paddle-in-hand transform and finger pose for both players.

**Why it exists.** The session tracks' recorded grip puts the handle 1.5 cm on the *back* of the hand, so the blade
lies across the back of the hand and the face that meets forehand balls is on the dorsal side (the grip is "flipped").
The paddle itself is right (paddle IMU and ball fits), but the hand is turned about 180 degrees about the handle from
any real grip: forearm readings past 160 degrees of supination, wrists hooked at 90 degrees, fingers that close on
nothing.

**This grip** puts the paddle where a shakehand player holds it, on the ActorCore rig's own hand geometry:

- the handle (the ``ping-pong/paddle`` asset: 100 mm long, 26-34 x 23 mm, its centre 129 mm below the
  blade centre along the asset's -Y) lies diagonally across the palm, from the hypothenar heel toward the gap between
  thumb and index (:data:`HANDLE_ANGLE_DEG` off the finger axis), its axis :data:`HANDLE_PALMAR_M` palmward of the hand
  joint (the rig's palm surface is 2.9-3.5 cm palmward, the handle 11.5 mm half-thick);
- the forehand rubber faces the palm, the backhand rubber the back of the hand;
- the middle, ring and little fingers wrap the handle; the index finger lies along the backhand face near the blade's
  edge; the thumb rests on the forehand face at the blade's shoulder (:func:`finger_pose`, solved once per rig).

The paddle state (position, orientation, velocity) is the physics truth; the hand is derived from it through this
grip, and the arm (:mod:`lib.replay.players.arm_ik`) holds it within the joint limits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lib.replay.players import rotations as rot
from lib.replay.players.kinematics import BodyPlacement, forward_kinematics, rest_pose
from lib.replay.players.skeleton import SkeletonRest, embedded_skeleton

Array = NDArray[Any]

HANDLE_ANGLE_DEG: float = 42.0
"""The handle's angle off the finger axis toward the thumb: the line from the hypothenar heel to the index knuckle on
the rig (Party_M: 42 degrees)."""
HANDLE_PALMAR_M: float = 0.030
"""The handle axis's distance palmward of the hand joint (palm surface 2.9-3.5 cm, handle half thickness 1.15 cm, less
the palm's give)."""
HANDLE_ALONG_M: float = 0.050
"""The handle centre's distance along the finger axis from the wrist (the middle of the palm)."""
HANDLE_ACROSS_M: float = -0.005
"""The handle centre's offset toward the thumb side (negative: toward the little finger)."""
BLADE_FROM_HANDLE_M: float = 0.129
"""Blade centre to handle centre along the asset's handle axis (blade half length 79 mm plus half the 100 mm handle)."""
BLADE_ROLL_DEG: float = 30.0
"""The blade's turn about the handle axis from the palm-parallel plane (positive: the forehand face toward the thumb's
side), fitted to the recorded contacts: holding each of the 1,271 contacts of sessions 1-7 (both players, the capture's
own body at the contact, the paddles' rubbers labelled palm-side, :func:`palm_faces`) with the limit-aware arm, the
share held within 2 degrees and 2 cm (Party / Uniform) is 0.81 / 0.70 at -15, 0.91 / 0.86 at 0, 0.98 / 0.94 at 15,
0.99 / 0.96 at 20, 0.99 / 0.97 at 25, 1.00 / 0.98 at 30 and 1.00 / 0.96 at 40; at 30 the forearm's pronation over the
contacts is centred in its range (p5-p95 -54..35 and -30..50 degrees). The handle angle (35-50 degrees) barely
matters."""
HANDLE_HALF_WIDTH_M: float = 0.015
BLADE_HALF_THICKNESS_M: float = 0.0075
"""The asset's blade with rubbers: 15 mm."""
BLADE_HALF_LENGTH_M: float = 0.079
FINGER_RADIUS_M: float = 0.009
FINGERS: tuple[str, ...] = ("Thumb", "Index", "Mid", "Ring", "Pinky")


@dataclass(frozen=True)
class HandAxes:
    """The rest hand's anatomical axes in the hand joint frame: finger direction, thumb side, palmar normal."""

    fingers: Array
    thumb_side: Array
    palm: Array


@dataclass(frozen=True)
class ShakehandGrip:
    """
    The paddle in the hand and the fingers around it, for one body and side.

    :ivar body: ``asset@version``.
    :ivar hand_joint: The hand joint (e.g. ``R_Hand``).
    :ivar blade_centre_m: Blade centre (mid-thickness) in the hand joint frame, metres at body scale 1.
    :ivar handle_axis: Unit handle-to-tip direction (the asset's +Y) in the hand joint frame.
    :ivar forehand_normal: Unit forehand-rubber normal (the asset's +Z) in the hand joint frame: palmar.
    :ivar finger_local: Local rotation (wxyz) of every finger joint of the hand.
    :ivar residual_m: The finger solve's contact misses, per finger (how far each lies from its target surface).
    """

    body: str
    hand_joint: str
    blade_centre_m: Array
    handle_axis: Array
    forehand_normal: Array
    finger_local: dict[str, Array] = field(repr=False)
    residual_m: dict[str, float] = field(default_factory=dict)

    @property
    def matrix(self) -> Array:
        """The paddle's axes in the hand joint frame: columns width (X), tip (Y), forehand normal (Z)."""

        normal = self.forehand_normal
        tip = self.handle_axis - float(self.handle_axis @ normal) * normal
        tip = tip / np.linalg.norm(tip)
        return np.column_stack([np.cross(tip, normal), tip, normal])

    def grip_spec(self) -> dict[str, Any]:
        """The fields of a :class:`~lib.replay.session.GripSpec`."""

        return {
            "hand_joint": self.hand_joint,
            "blade_centre_m": tuple(float(v) for v in self.blade_centre_m),
            "handle_axis": tuple(float(v) for v in self.handle_axis),
            "forehand_normal": tuple(float(v) for v in self.forehand_normal),
            "source": "recommended",
        }

    def hand_from_paddle(self, paddle_position: Array, paddle_rotation: Array, scale: float) -> tuple[Array, Array]:
        """
        The hand joint's world pose that holds a paddle: ``(position (..., 3), rotation matrix (..., 3, 3))``.

        :param paddle_position: Blade centre, world, ``(..., 3)``.
        :param paddle_rotation: Paddle axes as columns (width, tip, forehand normal), ``(..., 3, 3)``.
        :param scale: The body's uniform scale.
        """

        hand_rotation = np.asarray(paddle_rotation, dtype=np.float64) @ self.matrix.T
        position = np.asarray(paddle_position, dtype=np.float64) - np.einsum("...ij,j->...i", hand_rotation, scale * self.blade_centre_m)
        return position, hand_rotation

    def paddle_from_hand(self, hand_position: Array, hand_rotation: Array, scale: float) -> tuple[Array, Array]:
        """The paddle a hand pose holds: ``(blade centre (..., 3), paddle axes (..., 3, 3))``."""

        rotation = np.asarray(hand_rotation, dtype=np.float64)
        return np.asarray(hand_position, dtype=np.float64) + np.einsum("...ij,j->...i", rotation, scale * self.blade_centre_m), rotation @ self.matrix

    def apply_fingers(self, skeleton: SkeletonRest, local_rotation: Array) -> Array:
        """``local_rotation`` (``(..., J, 4)``) with the hand's finger joints set to the grip, a copy."""

        out = np.array(local_rotation, dtype=np.float64, copy=True)
        for name, q in self.finger_local.items():
            out[..., skeleton.index(name), :] = q
        return out


RELABEL: Array = np.diag([-1.0, 1.0, -1.0])
"""Right-multiplied onto paddle axes (width, tip, normal): the same blade with its rubbers' labels swapped (turned 180
degrees about its tip axis). Physics treats both rubbers alike; only which face is called the forehand changes."""


def forehand_on_back(forehand_normal: Any, skeleton: SkeletonRest, side: str = "R") -> bool:
    """
    Whether a grip's forehand normal (in the hand joint's frame) points out of the back of the hand: the session
    packages' recorded grip does; the shakehand grip's forehand rubber faces the palm. Such paddles are held with their labels swapped (:data:`RELABEL`,
    :func:`palm_faces`).
    """

    return bool(float(np.asarray(forehand_normal, dtype=np.float64) @ hand_axes(skeleton, side).palm) < 0.0)


def hand_axes(skeleton: SkeletonRest, side: str = "R") -> HandAxes:
    """The rest hand's finger, thumb-side and palmar axes, in the hand joint's own frame."""

    local, root = rest_pose(skeleton)
    world, position = forward_kinematics(skeleton, local, root, BodyPlacement(1.0, (0.0, 0.0, 0.0)))
    hand = skeleton.index(f"{side}_Hand")
    p = position
    fingers = p[skeleton.index(f"{side}_Mid1")] - p[hand]
    fingers /= np.linalg.norm(fingers)
    across = p[skeleton.index(f"{side}_Pinky1")] - p[skeleton.index(f"{side}_Index1")]
    across -= float(across @ fingers) * fingers
    across /= np.linalg.norm(across)
    sign = 1.0 if side == "R" else -1.0
    palm = sign * np.cross(fingers, across)
    palm /= np.linalg.norm(palm)
    to_local = rot.to_matrix(world[hand]).T
    return HandAxes(fingers=to_local @ fingers, thumb_side=to_local @ (-across), palm=to_local @ palm)


def paddle_frame(axes: HandAxes) -> tuple[Array, Array, Array]:
    """The paddle in the hand frame from the hand's axes: ``(blade centre, handle axis, forehand normal)``."""

    angle = math.radians(HANDLE_ANGLE_DEG)
    handle_axis = math.cos(angle) * axes.fingers + math.sin(angle) * axes.thumb_side
    handle_axis /= np.linalg.norm(handle_axis)
    # The blade plane contains the handle axis and lies parallel to the palm: its normal is the palm normal made
    # orthogonal to the handle axis (forehand rubber palmar).
    normal = axes.palm - float(axes.palm @ handle_axis) * handle_axis
    normal /= np.linalg.norm(normal)
    normal = rot.to_matrix(rot.from_axis_angle(handle_axis, math.radians(BLADE_ROLL_DEG))) @ normal
    centre = HANDLE_ALONG_M * axes.fingers + HANDLE_ACROSS_M * axes.thumb_side + HANDLE_PALMAR_M * axes.palm
    return centre + BLADE_FROM_HANDLE_M * handle_axis, handle_axis, normal


def _finger_chain(skeleton: SkeletonRest, side: str, finger: str) -> tuple[int, int, int, int | None]:
    names = [f"{side}_{finger}{k}" for k in (1, 2, 3)]
    idx = [skeleton.index(n) for n in names]
    tip = next((k for k, parent in enumerate(skeleton.parents) if parent == idx[2]), None)
    return idx[0], idx[1], idx[2], tip


def _finger_points(skeleton: SkeletonRest, local: Array, side: str, finger: str) -> Array:
    """The finger's joint positions and its tip (the last joint plus its bone length), in the hand frame, rest pose."""

    root = rest_pose(skeleton)[1]
    world, position = forward_kinematics(skeleton, local, root, BodyPlacement(1.0, (0.0, 0.0, 0.0)))
    hand = skeleton.index(f"{side}_Hand")
    to_local = rot.to_matrix(world[hand]).T
    a, b, c, tip = _finger_chain(skeleton, side, finger)
    pts = [position[a], position[b], position[c]]
    end = position[tip] if tip is not None else position[c] + (position[c] - position[b]) * 0.8
    pts.append(end)
    out = np.array([to_local @ (p - position[hand]) for p in pts])
    # Points along each phalanx (the flesh wraps along them, not only at the joints).
    return np.concatenate([out, 0.5 * (out[:-1] + out[1:])])


def _pose_finger(skeleton: SkeletonRest, local: Array, side: str, finger: str, params: Array, axes_world: HandAxes) -> Array:
    """Local rotations of one finger from ``params``: ``(spread, flex1, flex2, flex3)`` degrees (thumb: plus opposition)."""

    out = local.copy()
    rest_local, root = rest_pose(skeleton)
    world, position = forward_kinematics(skeleton, rest_local, root, BodyPlacement(1.0, (0.0, 0.0, 0.0)))
    idx = _finger_chain(skeleton, side, finger)[:3]
    for k, joint in enumerate(idx):
        child = idx[k + 1] if k < 2 else None
        bone = (position[child] - position[joint]) if child is not None else rot.rotate(world[joint], np.array([0.0, 1.0, 0.0]))
        bone = bone / np.linalg.norm(bone)
        axis_w = np.cross(bone, axes_world.palm)
        if finger == "Thumb":
            axis_w = 0.5 * np.cross(bone, axes_world.palm) - 0.5 * np.cross(bone, axes_world.fingers) * (1.0 if side == "R" else -1.0)
        axis_w /= np.linalg.norm(axis_w)
        to_joint = rot.conjugate(world[joint])
        q = rot.from_axis_angle(rot.rotate(to_joint, axis_w), math.radians(float(params[1 + k])))
        if k == 0:
            q = rot.multiply(rot.from_axis_angle(rot.rotate(to_joint, axes_world.palm), math.radians(float(params[0])) * (1.0 if side == "R" else -1.0)), q)
            if finger == "Thumb":
                q = rot.multiply(
                    rot.from_axis_angle(rot.rotate(to_joint, axes_world.fingers), math.radians(float(params[4])) * (1.0 if side == "R" else -1.0)), q
                )
        out[joint] = rot.multiply(skeleton.rest_rotation[joint], q)
    return out


def _world_axes(skeleton: SkeletonRest, side: str) -> HandAxes:
    local, root = rest_pose(skeleton)
    world, _ = forward_kinematics(skeleton, local, root, BodyPlacement(1.0, (0.0, 0.0, 0.0)))
    m = rot.to_matrix(world[skeleton.index(f"{side}_Hand")])
    a = hand_axes(skeleton, side)
    return HandAxes(fingers=m @ a.fingers, thumb_side=m @ a.thumb_side, palm=m @ a.palm)


def _finger_cost(points: Array, finger: str, centre: Array, axis: Array, normal: Array, params: Array) -> float:
    """How far a finger lies from where the grip puts it (metres squared), plus a small preference for less flexion."""

    rel = points - centre
    along = rel @ axis
    radial = rel - np.outer(along, axis)
    dist = np.linalg.norm(radial, axis=1)
    side = rel @ normal
    if finger in ("Mid", "Ring", "Pinky"):
        # wrap the handle: every point past the knuckle on the circle of the handle plus the finger
        wrap = HANDLE_HALF_WIDTH_M + FINGER_RADIUS_M
        sel = np.r_[1:4, 5:7]
        cost = float(np.sum((dist[sel] - wrap) ** 2))
    elif finger == "Index":
        # along the backhand face (dorsal of the blade), its tip near the blade's lower edge
        face = -(BLADE_HALF_THICKNESS_M + FINGER_RADIUS_M)
        sel = np.r_[2:4, 6]
        cost = float(np.sum((side[sel] - face) ** 2)) + 0.5 * float((along[3] - (BLADE_FROM_HANDLE_M - BLADE_HALF_LENGTH_M + 0.025)) ** 2)
    else:
        # the thumb on the forehand face at the blade's shoulder
        face = BLADE_HALF_THICKNESS_M + FINGER_RADIUS_M
        cost = float((side[3] - face) ** 2 + (side[2] - face) ** 2) + float((along[3] - (BLADE_FROM_HANDLE_M - BLADE_HALF_LENGTH_M + 0.005)) ** 2)
    return cost + 1e-7 * float(np.sum(np.asarray(params) ** 2))


FINGERS_PATH = Path(__file__).resolve().parents[1] / "data" / "grip_fingers_v1.json"
"""The finger poses solved for the ActorCore bodies, keyed by body, side and the grip geometry's constants: with them
the grip is numpy-only; a body or geometry not in the file is solved (with scipy) instead."""


def _geometry_key() -> str:
    return json.dumps([HANDLE_ANGLE_DEG, HANDLE_PALMAR_M, HANDLE_ALONG_M, HANDLE_ACROSS_M, BLADE_FROM_HANDLE_M, BLADE_ROLL_DEG, FINGER_RADIUS_M])


def _stored_fingers(body: str, side: str) -> tuple[dict[str, Array], dict[str, float]] | None:
    if not FINGERS_PATH.exists():
        return None
    document = json.loads(FINGERS_PATH.read_text(encoding="utf-8"))
    entry = document.get("grips", {}).get(f"{body}|{side}")
    if entry is None or entry.get("geometry") != _geometry_key():
        return None
    return {name: np.asarray(q, dtype=np.float64) for name, q in entry["finger_local"].items()}, dict(entry["residual_m"])


@cache
def shakehand_grip(body: str, side: str = "R") -> ShakehandGrip:
    """The shakehand grip of a catalog body's hand (the stored finger pose, else solved once; cached)."""

    stored = _stored_fingers(body, side)
    if stored is None:
        return _solve_grip(body, side)
    skeleton = embedded_skeleton(body)
    blade_centre, handle_axis, normal = paddle_frame(hand_axes(skeleton, side))
    finger_local, residual = stored
    return ShakehandGrip(
        body=body,
        hand_joint=f"{side}_Hand",
        blade_centre_m=blade_centre,
        handle_axis=handle_axis,
        forehand_normal=normal,
        finger_local=finger_local,
        residual_m=residual,
    )


RELAXED_FINGERS_DEG: dict[str, tuple[float, ...]] = {
    "Thumb": (10.0, 10.0, 25.0, 30.0, 20.0),
    "Index": (0.0, 35.0, 42.0, 22.0),
    "Mid": (0.0, 38.0, 47.0, 24.0),
    "Ring": (0.0, 40.0, 52.0, 28.0),
    "Pinky": (-3.0, 42.0, 57.0, 30.0),
}
"""The free hand's relaxed curl, ``(spread, MCP, PIP, DIP)`` degrees (thumb: ``(spread, CMC, MCP, IP, opposition)``):
the position of function's metacarpophalangeal flexion (35-45 degrees), and the interphalangeal flexion near the top
of the relaxed hand's ranges, cascading from index to little finger (PIP 5-45 / 5-50 / 10-55 / 15-60, DIP 5-25 / 5-25 /
10-30 / 15-30; thumb MCP 10-45, IP 20-50: US patent 9,622,523, "Ergonomic work gloves", relaxed-hand measurements;
resting-posture PIP 40-60 and DIP 20-40). At the ranges' middles the curl read as nearly straight in renders."""


@cache
def relaxed_fingers(body: str, side: str = "L") -> dict[str, Array]:
    """Local rotations (wxyz) of one hand's finger joints in :data:`RELAXED_FINGERS_DEG`'s relaxed curl (cached)."""

    skeleton = embedded_skeleton(body)
    axes_world = _world_axes(skeleton, side)
    local = np.array(skeleton.rest_rotation, dtype=np.float64)
    for finger, params in RELAXED_FINGERS_DEG.items():
        local = _pose_finger(skeleton, local, side, finger, np.asarray(params, dtype=np.float64), axes_world)
    return {skeleton.joints[i]: local[i].copy() for f in FINGERS for i in _finger_chain(skeleton, side, f)[:3]}


def apply_relaxed_fingers(skeleton: SkeletonRest, local_rotation: Array, body: str, side: str = "L") -> Array:
    """``local_rotation`` (``(..., J, 4)``) with one hand's fingers in the relaxed curl, a copy."""

    out = np.array(local_rotation, dtype=np.float64, copy=True)
    for name, q in relaxed_fingers(body, side).items():
        out[..., skeleton.index(name), :] = q
    return out


def _solve_grip(body: str, side: str) -> ShakehandGrip:
    """The shakehand grip with its finger pose solved (scipy's L-BFGS-B, several starts per finger)."""

    from scipy.optimize import minimize

    skeleton = embedded_skeleton(body)
    axes = hand_axes(skeleton, side)
    blade_centre, handle_axis, normal = paddle_frame(axes)
    handle_centre = blade_centre - BLADE_FROM_HANDLE_M * handle_axis
    axes_world = _world_axes(skeleton, side)
    local = np.array(skeleton.rest_rotation, dtype=np.float64)
    starts = {
        "Thumb": [0.0, 20.0, 15.0, 10.0, 30.0],
        "Index": [0.0, 15.0, 10.0, 5.0],
        "Mid": [0.0, 75.0, 90.0, 45.0],
        "Ring": [0.0, 78.0, 90.0, 45.0],
        "Pinky": [0.0, 80.0, 88.0, 40.0],
    }
    bounds = {
        "Thumb": [(-40.0, 40.0), (-30.0, 70.0), (-20.0, 70.0), (-10.0, 70.0), (-40.0, 90.0)],
        "Index": [(-25.0, 25.0), (-10.0, 60.0), (-5.0, 60.0), (-5.0, 50.0)],
        "Mid": [(-15.0, 15.0), (0.0, 95.0), (0.0, 105.0), (0.0, 75.0)],
        "Ring": [(-15.0, 15.0), (0.0, 95.0), (0.0, 105.0), (0.0, 75.0)],
        "Pinky": [(-20.0, 20.0), (0.0, 95.0), (0.0, 105.0), (0.0, 75.0)],
    }
    residual: dict[str, float] = {}
    for finger in FINGERS:

        def cost(params: Array, finger: str = finger, local: Array = local) -> float:
            posed = _pose_finger(skeleton, local, side, finger, params, axes_world)
            return _finger_cost(_finger_points(skeleton, posed, side, finger), finger, handle_centre, handle_axis, normal, params)

        # several starts (the thumb's rest orientation differs between rigs, and one start can stall on a bound)
        rng = np.random.default_rng(7)
        candidates = [np.array(starts[finger], dtype=np.float64)]
        lows, highs = np.array(bounds[finger]).T
        candidates += [lows + (highs - lows) * rng.random(len(lows)) for _ in range(11)]
        best = min((minimize(cost, x0, method="L-BFGS-B", bounds=bounds[finger]) for x0 in candidates), key=lambda r: float(r.fun))
        local = _pose_finger(skeleton, local, side, finger, best.x, axes_world)
        residual[finger] = round(math.sqrt(max(float(best.fun), 0.0)), 4)
    finger_local = {skeleton.joints[i]: local[i].copy() for f in FINGERS for i in _finger_chain(skeleton, side, f)[:3]}
    return ShakehandGrip(
        body=body,
        hand_joint=f"{side}_Hand",
        blade_centre_m=blade_centre,
        handle_axis=handle_axis,
        forehand_normal=normal,
        finger_local=finger_local,
        residual_m=residual,
    )


__all__ = [
    "BLADE_FROM_HANDLE_M",
    "BLADE_ROLL_DEG",
    "HANDLE_ANGLE_DEG",
    "HANDLE_PALMAR_M",
    "RELABEL",
    "RELAXED_FINGERS_DEG",
    "HandAxes",
    "ShakehandGrip",
    "apply_relaxed_fingers",
    "forehand_on_back",
    "hand_axes",
    "paddle_frame",
    "relaxed_fingers",
    "shakehand_grip",
]
