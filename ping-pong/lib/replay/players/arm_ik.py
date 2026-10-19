"""
Limit-aware arm IK: the paddle arm holds a hand pose with every joint inside its human range, or the hand pose moves to
the nearest one it can hold.

The arm is the anatomical chain: a 3-DOF shoulder, an elbow that is a hinge in the rig's own hinge plane, the forearm's
pronation, the wrist's flexion and deviation. For a wrist position the elbow lies on a circle about the shoulder-wrist
line (the swivel); for each swivel the humerus's frame follows from the hinge, and the hand's orientation then splits
into pronation, wrist flexion and wrist deviation exactly (:mod:`~lib.replay.players.anatomy`'s
decomposition). The solver:

1. **Reach.** Where the wrist is beyond the arm, the chest leans and turns toward it about the waist (spine flexion,
   lateral bend and twist inside their ranges, spread over ``Waist``, ``Spine01`` and ``Spine02``), never more than
   :data:`BOARD_TOLERANCE_M` into the table's board; what is still out of reach moves the hand (``residual_m``).
2. **Swivel.** 72 swivels per frame are scored by their joints' excess over the ranges (pronation, wrist, shoulder
   rotation, elevation and plane) and their distance from the pose's own swivel; over a sequence the path is chosen
   by dynamic programming with a cost on swivel speed, so the elbow never flips; where every swivel leaves the shoulder
   out of range, the chest also turns (spine twist).
3. **Orientation.** At the chosen swivel the pronation, wrist flexion and deviation are clamped into their ranges: the
   hand's rotation is the target's where it is feasible and the nearest feasible one where not (``projected``).
4. **Pose.** The humerus, forearm (with the pronation, as the replay carries it), forearm twist joints (their shares of
   the pronation) and hand are written as local rotations; the wrist joint lands on the (reachable) target.

The left arm is solved in the body mirrored through its sagittal plane (as
:mod:`~lib.replay.players.anatomy` measures a left joint), without the chest's help; the hinge calibration
is the rig's (:meth:`~lib.replay.players.limits.JointLimits.hinge`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.replay.players import rotations as rot
from lib.replay.players.anatomy import BASIS, MIRROR, _rx, _ry, _rz, _shoulder, _xyz
from lib.replay.players.kinematics import FOREARM_TWIST_SHARES, BodyPlacement, forward_kinematics, rest_pose
from lib.replay.players.limits import JointLimits, load_joint_limits
from lib.replay.players.skeleton import SkeletonRest

Array = NDArray[Any]

SWIVELS: int = 72
"""Swivel candidates per frame (5 degrees apart)."""
REACH_FRACTION: float = 0.995
"""The arm's usable reach as a fraction of its length (a straight arm is a singular pose)."""
ELBOW_REF_WEIGHT: float = 2000.0
"""Cost per squared metre of elbow away from its reference (the posed elbow, or the lowest elbow on its circle): 10 cm
costs as much as 4.5 degrees of range excess. Elbow positions, not swivel angles: a swivel angle's zero (the circle's
"down") flips where the shoulder-wrist line passes vertical, and angle costs then flip the elbow with it."""
HUMERUS_SPEED_WEIGHT: float = 2000.0
"""Cost per unit of squared Frobenius change of the humerus's rotation relative to the posed humerus between frames at
:data:`REFERENCE_FPS` (sequences): a 10 degree change in 1/30 s costs 61, a flip 16000. At another rate the weight
scales with the rate squared, so it prices angular speed, not steps."""
REFERENCE_FPS: float = 30.0
ELBOW_HINT_WEIGHT: float = 8000.0
"""Cost per squared body metre the elbow lies from :func:`solve_arm`'s ``elbow_hint`` (four times the posed elbow's
:data:`ELBOW_REF_WEIGHT`: the previous rendered arm dominates, the track pulls back as the adaptation fades)."""
HUMERUS_HINT_WEIGHT: float = 1000.0
"""Cost per unit of squared Frobenius difference between a candidate's humerus frame and :func:`solve_arm`'s
``humerus_hint`` (the previous rendered upper arm), at a straight elbow, tapering to nothing at
:data:`HUMERUS_HINT_FLEXION_DEG`. Near a straight elbow every swivel puts the elbow in nearly the same place, so the
elbow hint cannot hold the upper arm's roll: without this hint the per-tick solve re-split the roll between the upper
arm and the forearm, about 45 degrees of each in one frame while the hand and the elbow moved smoothly. With it such
one-frame re-splits fell to a third and the arm's IK miss p95 stayed at most 1.4 cm; applied at every elbow flexion,
a weight of 400 also held bent arms back."""
HUMERUS_HINT_FLEXION_DEG: float = 45.0
"""Elbow flexion at which the humerus hint has tapered to nothing (linear from a straight elbow)."""
CLEARANCE_RELATIVE_SLACK_M: float = 0.01
"""A relative clearance (:func:`solve_arm`'s ``clearance_relative``) prices only going this much deeper than the posed
arm: millimetres of depth between the posed arm and its own swivel candidate never move a held arm."""
CLEARANCE_RANGE_SLACK_DEG2: float = 1.0
"""Squared degrees of shoulder range excess beyond the frame's least that a swivel candidate may carry and still be
chosen for its clearance (:data:`CLEARANCE_WEIGHT`): the clearance never buys a shoulder out of its range (at 1e7
without this slack, a recorded session's shoulders were out of range in 6.2% of frames)."""
CLEARANCE_WEIGHT: float = 1.0e7
"""Cost per squared metre that a swivel candidate's forearm reaches into the torso
(:mod:`~lib.replay.players.clearance`: the shared arm-into-body definition's forearm samples and torso), divided
by the frame's weight (:func:`solve_arm`'s ``frame_weight``). 1 cm costs 1000, as much as one degree of shoulder range
excess. Share of conditioned frames with an arm more than 1 cm into the torso (the skinned-mesh measure, suited /
opponent): without it 16% / 22% (the posed elbow reference alone), 7.2% / 6.7% at 1e5, 2.4% / 2.5% at 1e6, 0.7% / 1.1%
at 1e7 everywhere, and 1.8% / 1.4% at 1e7 kept 0.15 s from the contacts (the setting used: a contact's blade is never
turned for it; the capture itself reads 0.1% / 0.0%)."""
ELBOW_WINDOW_M: float = 0.12
"""Runtime (one frame, the posed arm the reference): elbows farther than this from the posed one are not candidates."""
HUMERUS_WINDOW_DEG: float = 45.0
"""Runtime: humerus frames turned farther than this from the posed one are not candidates."""
ElbowRef = Literal["posed", "low"]
Continuity = Literal["posed", "absolute"]
"""How the swivel path is kept continuous: ``posed`` (the humerus moves as the posed humerus moves: the capture's own
arm is the evidence) or ``absolute`` (the humerus's own turn between frames, for a posed arm that is not evidence)."""
TWISTS_DEG: tuple[float, ...] = (0.0, -12.0, 12.0, -24.0, 24.0)
"""Chest turns tried where every swivel leaves an arm angle out of range."""
TWIST_WEIGHT: float = 0.05
"""Cost per squared degree of chest turn."""
REPOSITION_MIN_DEG: float = 0.25
"""A shoulder out of range by more than this (root of its summed squared excess) at every swivel lets the wrist move."""
REPOSITION_STEPS_M: tuple[float, ...] = (0.03, 0.06, 0.10, 0.15)
"""The wrist moves tried, along each world axis."""
REPOSITION_WEIGHT: float = 1.0e4
"""Cost per squared metre the wrist moves between strokes (the paddle IMU's position there is loose evidence): 10 cm
costs as much as 1 degree of shoulder excess at :data:`SHOULDER_WEIGHT`."""
CONTACT_FRAME_WEIGHT: float = 10.0
"""Frames weighted at least this (:func:`solve_arm`'s ``frame_weight``) are contact frames: pinned in the clamp's rate
limit."""
TWIST_SPEED_WEIGHT: float = 1.0
"""Cost per squared degree the chest's turn changes between frames at :data:`REFERENCE_FPS` (sequences)."""
TWIST_SMOOTH_S: float = 0.3
"""The chest's turn path is smoothed over this long (a 24 degree turn then takes about 0.15 s: 160 deg/s, inside the
spine's 277)."""
BOARD_TOLERANCE_M: float = 0.01
"""How far a body part may go into the table's board: it may touch the board, not go into it."""
BOARD_HALF: tuple[float, float] = (1.37, 0.7625)
BOARD_THICKNESS_M: float = 0.03
TORSO_POINTS: tuple[tuple[str, float], ...] = (("Spine02", 0.13), ("NeckTwist01", 0.09), ("Head", 0.10), ("Spine01", 0.13))
"""Torso points and radii kept out of the board when the chest leans (the replay's body capsules, ANSUR II means)."""
ARM_ANGLES: tuple[str, ...] = ("forearm_pronation", "wrist_flexion", "wrist_deviation", "shoulder_elevation", "shoulder_rotation", "shoulder_plane")
"""The angles the swivel changes (the elbow's flexion is fixed by the wrist's distance)."""
COMFORT_WEIGHT: float = 20.0
"""Cost of an angle at the edge of its range relative to its middle (``(z^4)`` with ``z`` the distance from the middle
in half-widths): a joint at its limit costs as much as 4.5 degrees of excess, so the demand spreads over the joints
instead of saturating one."""
COMFORT_ANGLES: tuple[str, ...] = ("forearm_pronation", "wrist_flexion", "wrist_deviation", "shoulder_rotation")
SHOULDER_WEIGHT: float = 1000.0
"""The shoulder's range excess weighs this much more than the hand's in the swivel choice: the hand's is projected away
(the paddle turns into range) where the shoulder's would stay as a contortion. At 10 the conditioned tracks kept the
shoulder 1-45 degrees out of range on 1.2% / 0.5% of session 5's frames; at 1000, with the wrist free to give way
between strokes (:data:`REPOSITION_WEIGHT`), 0.45% / 0.23%, for one more of 207 contacts held beyond 2 degrees
(0.98 / 1.00 held)."""


@dataclass(frozen=True)
class ArmModel:
    """One body's paddle arm, ready to solve."""

    skeleton: SkeletonRest
    body: str
    side: str
    clavicle: int
    upperarm: int
    forearm: int
    hand: int
    twists: tuple[tuple[int, float], ...]
    spine: tuple[int, int, int]
    thorax: int
    pelvis: int
    upper_m: float
    lower_m: float
    rest_world: Array = field(repr=False)
    hinge_deg: float = 0.0
    limits: JointLimits = field(default_factory=load_joint_limits, repr=False)

    @classmethod
    def for_skeleton(cls, skeleton: SkeletonRest, body: str, side: str = "R", *, limits: JointLimits | None = None) -> ArmModel:
        if side not in ("R", "L"):
            raise ValueError(f"no arm side {side!r}")
        table = limits or load_joint_limits()
        local, root = rest_pose(skeleton)
        world, _ = forward_kinematics(skeleton, local, root, BodyPlacement(1.0, (0.0, 0.0, 0.0)))
        twists = tuple((skeleton.index(f"{side}_{name}"), share) for name, share in FOREARM_TWIST_SHARES if f"{side}_{name}" in skeleton.joints)
        return cls(
            skeleton=skeleton,
            body=body,
            side=side,
            clavicle=skeleton.index(f"{side}_Clavicle"),
            upperarm=skeleton.index(f"{side}_Upperarm"),
            forearm=skeleton.index(f"{side}_Forearm"),
            hand=skeleton.index(f"{side}_Hand"),
            twists=twists,
            spine=(skeleton.index("Waist"), skeleton.index("Spine01"), skeleton.index("Spine02")),
            thorax=skeleton.index("Spine02"),
            pelvis=skeleton.index("Pelvis"),
            upper_m=float(np.linalg.norm(skeleton.rest_translation[skeleton.index(f"{side}_Forearm")])),
            lower_m=float(np.linalg.norm(skeleton.rest_translation[skeleton.index(f"{side}_Hand")])),
            rest_world=rot.to_matrix(world),
            hinge_deg=table.hinge(body)[side],
            limits=table,
        )


@dataclass(frozen=True)
class ArmSolution:
    """
    The solved arm over ``F`` frames.

    :ivar local_rotation: Every joint's local rotation ``(F, J, 4)``.
    :ivar hand_position: The wrist joint's world position the arm holds ``(F, 3)``.
    :ivar hand_rotation: The hand's world rotation it holds ``(F, 3, 3)``.
    :ivar projected: Frames whose hand rotation had to move into the joints' ranges.
    :ivar projection_deg: How far the hand's rotation moved, degrees.
    :ivar residual_m: How far the wrist stayed from the target (beyond the arm and chest's reach), m.
    :ivar swivel_deg: The chosen swivel, degrees (0: elbow straight down from the shoulder-wrist line).
    :ivar chest_deg: The chest's added rotation about the waist, degrees.
    :ivar angles: The arm's anatomical angles at the solution, degrees.
    :ivar excess_deg: Their excess over the ranges, degrees.
    :ivar cost: The chosen pose's cost per frame (range excess, comfort, swivel distance).
    """

    local_rotation: Array
    hand_position: Array
    hand_rotation: Array
    projected: Array
    projection_deg: Array
    residual_m: Array
    swivel_deg: Array
    chest_deg: Array
    angles: dict[str, Array]
    excess_deg: dict[str, Array]
    cost: Array = field(default_factory=lambda: np.zeros(0))
    body_depth_m: Array = field(default_factory=lambda: np.zeros(0))
    """Per frame, how far the chosen arm goes into the torso at the clearance points (m; 0 when clear, empty when the
    clearance was not evaluated)."""


def _excess(values: Array, bounds: tuple[float, float]) -> Array:
    return np.maximum(np.maximum(bounds[0] - values, values - bounds[1]), 0.0)


def _unit(v: Array) -> Array:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _world(model: ArmModel, local: Array, root: Array, placement: BodyPlacement) -> tuple[Array, Array]:
    q, p = forward_kinematics(model.skeleton, local, root, placement)
    return rot.to_matrix(q), p


def _clearance_weights(frame_weight: Array) -> Array:
    """
    Per frame, the spin choice's clearance weight: :data:`CLEARANCE_WEIGHT` over the frame's weight, and none on the
    contact frames (from :data:`CONTACT_FRAME_WEIGHT`), whose blade is the ball's evidence: the clearance never spins
    a contact's blade (spinning it there costs strokes).
    """

    fw = np.maximum(np.asarray(frame_weight, dtype=np.float64), 1.0)
    return np.where(fw >= CONTACT_FRAME_WEIGHT, 0.0, CLEARANCE_WEIGHT / fw)


def _clearance_torso(model: ArmModel, pos: Array, scale: float) -> Any:
    """The body's torso cylinders for joint positions ``(F, J, 3)`` (:func:`~lib.replay.players.clearance.torso_frames`),
    or ``None`` for a rig without a fitted torso."""

    from lib.replay.players.clearance import has_torso, torso_frames

    return torso_frames(model.skeleton.joints, pos, scale, model.body) if has_torso(model.body) else None


def _forearm_depth(model: ArmModel, torso: Any, rows: Array, elbows: Array, wrists: Array, scale: float) -> Array:
    """
    How deep each candidate's forearm reaches into the torso (m, clipped at 0): the shared definition's forearm samples
    from ``elbows`` (``(n, K, 3)``) to ``wrists`` (``(n, 3)``), world metres, against the torso of frames ``rows``.
    """

    from lib.replay.players.clearance import ARM_SEGMENTS, TorsoFrames, points_depth

    _j0, _j1, radius, samples = ARM_SEGMENTS[model.side][0]
    w = np.linspace(0.0, 1.0, samples)
    points = elbows[:, :, None, :] * (1.0 - w)[None, None, :, None] + wrists[:, None, None, :] * w[None, None, :, None]
    sub = TorsoFrames(
        torso.origin[:, rows],
        torso.axis[:, rows],
        torso.length[:, rows],
        torso.forward[:, rows],
        torso.lateral[:, rows],
        torso.forward_centre,
        torso.forward_axis,
        torso.lateral_centre,
        torso.lateral_axis,
    )
    return np.maximum(np.max(points_depth(sub, points, radius * scale), axis=-1), 0.0)


def _chest_rotation(model: ArmModel, local: Array, world: Array, r: Array) -> Array:
    """Spread a world rotation ``r`` (``(F, 3, 3)``) over the waist and spine joints (``r^(1/3)`` about each)."""

    rv = rot.to_rotvec(rot.from_matrix(r))
    third = rot.to_matrix(rot.from_rotvec(rv / 3.0))
    out = local.copy()
    for joint in model.spine:
        parent = int(model.skeleton.parents[joint])
        wp = world[:, parent]
        conj = np.swapaxes(wp, -1, -2) @ third @ wp
        out[:, joint] = rot.multiply(rot.from_matrix(conj), local[:, joint])
    return out


def _board_depth(points: Array, radius: float) -> Array:
    """How far a sphere goes into the table's board (positive: inside), m."""

    hx, hy = BOARD_HALF
    lo = np.array([-hx, -hy, -BOARD_THICKNESS_M])
    hi = np.array([hx, hy, 0.0])
    outside = np.maximum(np.maximum(lo - points, points - hi), 0.0)
    gap = np.linalg.norm(outside, axis=-1)
    inside = np.all((points >= lo) & (points <= hi), axis=-1)
    depth_inside = np.min(np.minimum(points - lo, hi - points), axis=-1)
    return np.where(inside, radius + depth_inside, radius - gap)


def _spine_angles(model: ArmModel, world: Array) -> dict[str, Array]:
    pel, tho = model.pelvis, model.thorax
    rest = model.rest_world
    d = BASIS.T @ np.swapaxes(world[:, pel] @ rest[pel].T, -1, -2) @ (world[:, tho] @ rest[tho].T) @ BASIS
    a, b, c = _xyz(d)
    return {"spine_flexion": np.degrees(-a), "spine_lateral": np.degrees(b), "spine_twist": np.degrees(c)}


def _lean(model: ArmModel, local: Array, root: Array, placement: BodyPlacement, target: Array, reach: float) -> tuple[Array, Array]:
    """The chest's lean and turn toward wrist targets beyond reach, inside the spine's ranges and out of the board."""

    world, pos = _world(model, local, root, placement)
    shoulder = pos[:, model.upperarm]
    pivot = pos[:, model.spine[0]]
    need = np.linalg.norm(target - shoulder, axis=-1) > reach
    chest = np.zeros((len(local), 3))
    if not np.any(need):
        return local, chest
    a = shoulder - pivot
    b = target - pivot
    axis = _unit(np.cross(a, b))
    full = np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))
    ranges = model.limits.ranges
    lo, hi = np.zeros(len(local)), np.where(need, full, 0.0)
    best = local.copy()
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        r = rot.to_matrix(rot.from_axis_angle(axis, mid))
        cand = _chest_rotation(model, local, world, r)
        w2, p2 = _world(model, cand, root, placement)
        spine = _spine_angles(model, w2)
        ok = np.ones(len(local), dtype=bool)
        for name, value in spine.items():
            ok &= _excess(value, ranges[name]) <= 0.0
        for joint, radius in TORSO_POINTS:
            if joint in model.skeleton.joints:
                ok &= _board_depth(p2[:, model.skeleton.index(joint)], radius * placement.scale) <= BOARD_TOLERANCE_M
        reached = np.linalg.norm(target - p2[:, model.upperarm], axis=-1) <= reach
        grow = ok & ~reached
        # inside the ranges and still short: lean further; beyond a range (or reached): lean less
        lo = np.where(grow, mid, lo)
        hi = np.where(grow, hi, mid)
        keep = ok
        best[keep] = cand[keep]
        chest[keep] = np.degrees(rot.to_rotvec(rot.from_axis_angle(axis[keep], mid[keep])))
    return best, chest


def _smooth_envelope(vectors: Array, width: int) -> Array:
    """Per frame the largest of the vectors within ``width`` frames (by length), smoothed with a triangle kernel."""

    lengths = np.linalg.norm(vectors, axis=-1)
    padded = np.pad(lengths, width, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * width + 1)
    pick = np.clip(np.arange(len(vectors)) + np.argmax(windows, axis=1) - width, 0, len(vectors) - 1)
    envelope = vectors[pick]
    kernel = np.r_[np.arange(1, width + 2), np.arange(width, 0, -1)].astype(np.float64)
    kernel /= kernel.sum()
    padded_v = np.pad(envelope, ((width, width), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded_v[:, c], kernel, mode="valid") for c in range(vectors.shape[1])], axis=1)


def _elbow_reference_cost(elbows: Array, posed: Array, ref: ElbowRef, weight: float) -> Array:
    """Per candidate, ``weight`` times the squared distance of its elbow (``(F, K, 3)``) from the reference elbow."""

    if ref == "low":
        lowest = elbows[np.arange(len(elbows)), np.argmin(elbows[..., 2], axis=1)]
        return weight * np.sum((elbows - lowest[:, None]) ** 2, axis=-1)
    return weight * np.sum((elbows - posed[:, None]) ** 2, axis=-1)


HUMERUS_LIMIT_PENALTY: float = 1.0e6
"""Cost per unit of squared Frobenius change of the humerus's world frame beyond what its speed limit allows in a
frame: in effect a hard limit (the arm never turns through its swivel faster than the upper arm can turn; where the
paddle asks for more, the hand is projected instead). A softer 1e4 held one more of session 5's 207 contacts, but let
conditioned tracks swing the humerus faster than the replay's runtime clamp lets the arm follow: a recorded session's
IK miss p95 rose from 2.4 to 5.8 cm, even with the softer limit capped at 0.9 of the glitch ceiling."""


def _humerus_path(
    cost: Array,
    relative: Array,
    weight: float,
    absolute: Array | None = None,
    max_step_rad: float | None = None,
) -> Array:
    """
    The candidate per frame minimising the summed cost plus ``weight`` times the squared change, between frames, of the
    candidate humerus's rotation relative to the posed humerus (Frobenius; about ``2 theta^2`` for a turn ``theta``):
    the arm then moves as the posed arm moves and never turns through its swivel or its roll in a frame. (The elbow's
    position alone cannot see a flip of a nearly straight arm's roll.)

    :param relative: Candidate humerus frames times the posed frame's transpose, ``(F, K, 3, 3)``.
    """

    frames, k = cost.shape
    flat = relative.reshape(frames, k, 9)
    hard = absolute is not None and max_step_rad is not None
    flat_abs = absolute.reshape(frames, k, 9) if absolute is not None else flat
    threshold = 4.0 * (1.0 - math.cos(max_step_rad)) if max_step_rad is not None else math.inf
    acc = cost[0].copy()
    back = np.zeros((frames, k), dtype=np.int64)
    for f in range(1, frames):
        d = flat[f][None, :, :] - flat[f - 1][:, None, :]
        total = acc[:, None] + weight * np.sum(d * d, axis=-1)
        if hard:
            da = flat_abs[f][None, :, :] - flat_abs[f - 1][:, None, :]
            total = total + HUMERUS_LIMIT_PENALTY * np.maximum(np.sum(da * da, axis=-1) - threshold, 0.0)
        back[f] = np.argmin(total, axis=0)
        acc = total[back[f], np.arange(k)] + cost[f]
    out = np.zeros(frames, dtype=np.int64)
    out[-1] = int(np.argmin(acc))
    for f in range(frames - 1, 0, -1):
        out[f - 1] = back[f, out[f]]
    return out


def _swivel_axes(model: ArmModel, shoulder: Array, wrist: Array) -> tuple[Array, Array, Array, Array, Array]:
    """The elbow circle of each frame: ``(n, u, v, a, r)``, its axis (shoulder to wrist), the swivel's zero (straight
    down) and quarter turn, the centre's distance from the shoulder and the radius (body metres)."""

    up, lo = model.upper_m, model.lower_m
    rel = wrist - shoulder
    d = np.linalg.norm(rel, axis=-1)
    n = rel / np.maximum(d[:, None], 1e-9)
    a = (up * up - lo * lo + d * d) / np.maximum(2.0 * d, 1e-9)
    r = np.sqrt(np.maximum(up * up - a * a, 0.0))
    down = np.array([0.0, 0.0, -1.0])
    u = down - np.sum(n * down, axis=-1, keepdims=True) * n
    fallback = np.cross(n, np.array([1.0, 0.0, 0.0]))
    u = np.where(np.linalg.norm(u, axis=-1, keepdims=True) < 1e-6, fallback, u)
    u = _unit(u)
    v = np.cross(n, u)
    return n, u, v, a, r


def _swivel_grid(model: ArmModel, shoulder: Array, wrist: Array, elbow: Array) -> Array:
    """
    Per frame, :data:`SWIVELS` swivels evenly around the circle starting at the swivel of ``elbow`` (the posed elbow's
    direction projected onto the circle's plane), radians ``(F, K)``: the posed arm is itself a candidate, so an arm
    asked to hold the hand it already holds stays exactly where it is.
    """

    n, u, v, _a, _r = _swivel_axes(model, shoulder, wrist)
    e = elbow - shoulder
    e = e - np.sum(e * n, axis=-1, keepdims=True) * n
    own = np.arctan2(np.sum(e * v, axis=-1), np.sum(e * u, axis=-1))
    own = np.where(np.linalg.norm(e, axis=-1) > 1e-9, own, 0.0)
    steps = np.radians(np.arange(SWIVELS) * (360.0 / SWIVELS))
    return _wrapped_rad(own[:, None] + steps[None, :])


def _wrapped_rad(x: Array) -> Array:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _candidates(model: ArmModel, shoulder: Array, wrist: Array, hand_anat: Array, thorax_anat: Array, phis: Array) -> dict[str, Array]:
    """Every swivel candidate's humerus frame and arm angles: arrays ``(F, K, ...)``, angles in degrees. ``phis``
    (radians) is ``(K,)`` for every frame or ``(F, K)`` per frame."""

    up, lo = model.upper_m, model.lower_m
    n, u, v, a, r = _swivel_axes(model, shoulder, wrist)
    d = np.linalg.norm(wrist - shoulder, axis=-1)
    ph = np.broadcast_to(phis, (len(n), np.shape(phis)[-1]))
    e = np.cos(ph)[..., None] * u[:, None] + np.sin(ph)[..., None] * v[:, None]  # (F, K, 3)
    elbow = shoulder[:, None] + a[:, None, None] * n[:, None] + r[:, None, None] * e
    x = _unit(elbow - shoulder[:, None])
    y = -e - np.sum(-e * x, axis=-1, keepdims=True) * x
    y = _unit(y)
    z = np.cross(x, y)
    humerus = np.stack([x, y, z], axis=-1)  # (F, K, 3, 3) columns
    cos_inner = np.clip((up * up + lo * lo - d * d) / (2.0 * up * lo), -1.0, 1.0)
    flexion = math.pi - np.arccos(cos_inner)  # (F,)
    untwisted = humerus @ _rz(flexion)[:, None]
    g = np.swapaxes(untwisted, -1, -2) @ hand_anat[:, None]
    pa, pb, pc = _xyz(g)
    shoulder_angles = _shoulder(np.swapaxes(thorax_anat, -1, -2)[:, None] @ humerus)
    return {
        "humerus": humerus,
        "elbow": elbow,
        "flexion": flexion,
        "untwisted": untwisted,
        "n": n,
        "u": u,
        "v": v,
        "centre": shoulder + a[:, None] * n,
        "forearm_pronation": np.degrees(-pa),
        "wrist_flexion": np.degrees(pb),
        "wrist_deviation": np.degrees(pc),
        "shoulder_elevation": np.degrees(shoulder_angles["shoulder_elevation"]),
        "shoulder_rotation": np.degrees(shoulder_angles["shoulder_rotation"]),
        "shoulder_plane": np.degrees(shoulder_angles["shoulder_plane"]),
    }


def _excess_cost(model: ArmModel, cand: dict[str, Array], *, comfort: bool = True, shoulder_weight: Any = None) -> tuple[Array, Array, Array]:
    """
    Per candidate: the squared range excess summed over the swivel-dependent angles plus (``comfort``) the comfort cost
    (rising near each range's edge), the shoulder's squared excess alone, and every angle's squared excess unweighted.
    """

    ranges = model.limits.ranges
    total = np.zeros(cand["forearm_pronation"].shape)
    shoulder = np.zeros_like(total)
    every = np.zeros_like(total)
    weight_of_shoulder = np.asarray(SHOULDER_WEIGHT if shoulder_weight is None else shoulder_weight, dtype=np.float64)
    if weight_of_shoulder.ndim == 1:
        weight_of_shoulder = weight_of_shoulder[:, None]
    for name in ARM_ANGLES:
        ex = _excess(cand[name], ranges[name])
        if name == "shoulder_plane":
            ex = np.where(cand["shoulder_elevation"] > 45.0, ex, 0.0)
        every += ex * ex
        if name.startswith("shoulder"):
            total += weight_of_shoulder * ex * ex
            shoulder += ex * ex
        else:
            total += ex * ex
        if comfort and name in COMFORT_ANGLES:
            lo, hi = ranges[name]
            z = (cand[name] - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
            total += COMFORT_WEIGHT * np.minimum(z * z, 4.0) ** 2
    return total, shoulder, every


def _wrapped(deg: Array) -> Array:
    return (deg + 180.0) % 360.0 - 180.0


def _path(cost: Array, phis_deg: Array, speed_weight: float) -> Array:
    """The swivel index per frame minimising the summed cost plus ``speed_weight`` times the squared change (Viterbi)."""

    frames, k = cost.shape
    step = _wrapped(phis_deg[:, None] - phis_deg[None, :])
    trans = speed_weight * step * step
    acc = cost[0].copy()
    back = np.zeros((frames, k), dtype=np.int64)
    for f in range(1, frames):
        total = acc[:, None] + trans
        back[f] = np.argmin(total, axis=0)
        acc = total[back[f], np.arange(k)] + cost[f]
    out = np.zeros(frames, dtype=np.int64)
    out[-1] = int(np.argmin(acc))
    for f in range(frames - 1, 0, -1):
        out[f - 1] = back[f, out[f]]
    return out


PROJECTION_CONTINUITY_WEIGHT: float = 1.0
"""Cost per squared degree the clamped hand (pronation, wrist flexion, deviation) jumps from the previous frame's when
choosing between the two Euler branches of a projection (sequences)."""


def _rate_limited(angles: Array, max_step: Array, pinned: Array | None = None) -> Array:
    """Angle sequences ``(F, K)`` (degrees, inside their ranges) whose frame-to-frame change is held to ``max_step``
    (``(K,)``): forward and backward passes, averaged; ``pinned`` frames keep their angles."""

    fixed = np.zeros(len(angles), dtype=bool) if pinned is None else np.asarray(pinned, dtype=bool)

    def sweep(seq: Array, keep: Array) -> Array:
        out = seq.copy()
        for f in range(1, len(out)):
            if not keep[f]:
                out[f] = out[f - 1] + np.clip(out[f] - out[f - 1], -max_step, max_step)
        return out

    out = 0.5 * (sweep(angles, fixed) + sweep(angles[::-1], fixed[::-1])[::-1])
    out[fixed] = angles[fixed]
    return out


def _clamp_hand(
    model: ArmModel, untwisted: Array, hand_anat: Array, *, sequence: bool = False, fps: float | None = None, pinned: Array | None = None
) -> tuple[Array, dict[str, Array], Array]:
    """
    The hand frame with its pronation and wrist angles clamped into range. Both Euler branches of the decomposition
    are tried; per frame the one leaving the smaller excess wins, or, over a sequence, the one that also keeps the
    clamped angles nearest the previous frame's (:data:`PROJECTION_CONTINUITY_WEIGHT`).
    """

    ranges = model.limits.ranges
    g = np.swapaxes(untwisted, -1, -2) @ hand_anat
    a, b, c = _xyz(g)
    excess, clamped = [], []
    for pron, flex, dev in ((-a, b, c), (-(a + math.pi), math.pi - b, c + math.pi)):
        p_deg, f_deg, d_deg = (_wrapped(np.degrees(x)) for x in (pron, flex, dev))
        excess.append(
            _excess(p_deg, ranges["forearm_pronation"]) ** 2 + _excess(f_deg, ranges["wrist_flexion"]) ** 2 + _excess(d_deg, ranges["wrist_deviation"]) ** 2
        )
        clamped.append(
            np.stack(
                [np.clip(p_deg, *ranges["forearm_pronation"]), np.clip(f_deg, *ranges["wrist_flexion"]), np.clip(d_deg, *ranges["wrist_deviation"])],
                axis=-1,
            )
        )
    ex = np.stack(excess, axis=-1)  # (F, 2)
    cl = np.stack(clamped, axis=-2)  # (F, 2, 3)
    pick = np.argmin(ex, axis=-1)
    if sequence and len(ex) > 1:
        for f in range(1, len(ex)):
            jump = np.sum(_wrapped(cl[f] - cl[f - 1, pick[f - 1]][None]) ** 2, axis=-1)
            pick[f] = int(np.argmin(ex[f] + PROJECTION_CONTINUITY_WEIGHT * jump))
    chosen = cl[np.arange(len(ex)), pick]
    if sequence and fps is not None and len(ex) > 1:
        # Where the clamp moves to the other end of a range (the paddle asked the hand past it), the hand goes there no
        # faster than its joints turn: through the range, at the pronation's and the wrist's speed limits.
        limits = model.limits
        rates = np.array(
            [
                limits.speed_limit_deg_s(model.skeleton.joints[model.twists[0][0]])
                if model.twists
                else limits.speed_limit_deg_s(model.skeleton.joints[model.forearm]),
                limits.speed_limit_deg_s(model.skeleton.joints[model.hand]),
                limits.speed_limit_deg_s(model.skeleton.joints[model.hand]),
            ]
        )
        chosen = _rate_limited(chosen, rates / fps, pinned)
    pron, flex, dev = (np.radians(chosen[:, k]) for k in range(3))
    hand = untwisted @ _rx(-pron) @ _ry(flex) @ _rz(dev)
    return hand, {"forearm_pronation": chosen[:, 0], "wrist_flexion": chosen[:, 1], "wrist_deviation": chosen[:, 2]}, pron


def solve_arm(
    model: ArmModel,
    local_rotation: ArrayLike,
    root_translation: ArrayLike,
    placement: BodyPlacement,
    hand_position: ArrayLike,
    hand_rotation: ArrayLike,
    *,
    elbow_ref: ElbowRef = "posed",
    elbow_window_m: float | None = None,
    continuity: Continuity = "posed",
    fps: float = REFERENCE_FPS,
    smooth: bool = True,
    trunk: bool = True,
    comfort: bool = True,
    twist: bool | None = None,
    frame_weight: ArrayLike | None = None,
    reposition: bool = False,
    elbow_hint: ArrayLike | None = None,
    humerus_hint: ArrayLike | None = None,
    clearance: bool = True,
    clearance_weight: ArrayLike | None = None,
    clearance_relative: bool = True,
) -> ArmSolution:
    """
    Hold a hand pose (or the nearest one the joints allow) with the paddle arm, frame by frame.

    :param model: The body's arm (:meth:`ArmModel.for_skeleton`).
    :param local_rotation: The posed body, ``(F, J, 4)`` (or ``(J, 4)`` for one frame).
    :param root_translation: Its root translation in skeleton space, ``(F, 3)``.
    :param placement: Skeleton space in the table frame.
    :param hand_position: The wrist joint's wanted world position, ``(F, 3)``.
    :param hand_rotation: The hand's wanted world rotation, matrices ``(F, 3, 3)``.
    :param elbow_ref: The elbow to stay near: ``posed`` (the posed body's own) or ``low`` (the lowest on its circle,
        the natural hanging elbow).
    :param elbow_window_m: Exclude elbows farther than this from the posed one (runtime: no swivel jumps).
    :param continuity: See :data:`Continuity`.
    :param fps: The frames' rate (sequences): the continuity cost prices angular speed.
    :param smooth: Choose the swivel path over the whole sequence (dynamic programming); ``False`` per frame.
    :param trunk: Let the chest lean and turn for reach and for the shoulder's range.
    :param twist: Let the chest turn where every swivel leaves an angle out of range (default: ``trunk``).
    :param frame_weight: Per frame, how much its range excess (and comfort) counts against the path's continuity
        (sequences; default 1): the contacts, where the paddle is the ball's evidence, weigh more, and their clamped
        hand is never moved by the clamp's rate limit (from :data:`CONTACT_FRAME_WEIGHT` up).
    :param elbow_hint: World elbow positions (``(F, 3)`` or ``(3,)``) to stay near as well (:data:`ELBOW_HINT_WEIGHT`):
        the replay's runtime passes the elbow its previous rendered pose implies, so the per-tick solve never jumps
        between two equally good swivels.
    :param clearance: Keep the forearm out of the torso (:data:`CLEARANCE_WEIGHT`; rigs without a fitted torso are not
        checked).
    :param clearance_weight: Per frame, the clearance's weight (default :data:`CLEARANCE_WEIGHT` over the frame weight).
    :param clearance_relative: Price only going deeper than the posed arm itself (the default: a runtime correction of
        a conditioned body, whose held pose stays exact); ``False`` prices any depth (conditioning a captured track).
    :param humerus_hint: World rotations of the upper arm (``(F, 3, 3)`` or ``(3, 3)``) to stay near as well, near a
        straight elbow (:data:`HUMERUS_HINT_WEIGHT`): the runtime passes its previous rendered upper arm, so the roll the
        elbow hint cannot see never re-splits between the upper arm and the forearm in one frame.
    :param reposition: Sequences: where the shoulder stays out of range at every swivel and chest turn, let the wrist
        move (:data:`REPOSITION_STEPS_M`, smoothed over time) on frames below :data:`CONTACT_FRAME_WEIGHT`.
    :param comfort: Prefer swivels whose angles sit away from their ranges' edges. Off for a runtime correction of an
        arm already chosen with it (a conditioned track): the arm then keeps its own swivel unless a range forces it
        off, so a hand asked to stay where it is stays exactly there.
    :return: The solution.
    """

    single = np.ndim(local_rotation) == 2
    mirrored = model.side == "L"
    turn = trunk if twist is None else twist
    if mirrored:
        trunk = turn = False  # the chest leans and turns for the paddle arm only
    local = np.array(local_rotation, dtype=np.float64, ndmin=3)
    root = np.array(root_translation, dtype=np.float64, ndmin=2)
    target = np.array(hand_position, dtype=np.float64, ndmin=2)
    target_rot = np.asarray(hand_rotation, dtype=np.float64).reshape(-1, 3, 3)
    frames = len(local)
    scale = placement.scale
    reach = REACH_FRACTION * scale * (model.upper_m + model.lower_m)
    chest = np.zeros((frames, 3))
    if trunk:
        leaned, chest = _lean(model, local, root, placement, target, reach)
        if smooth and frames > 1 and np.any(chest != 0.0):
            # over a sequence the lean is a smooth envelope of the frames' leans (it starts before it is needed and
            # never snaps), applied again to the unleaned body
            chest = _smooth_envelope(chest, max(1, round(TWIST_SMOOTH_S * fps / 2)))
            world0, _p0 = _world(model, local, root, placement)
            leaned = _chest_rotation(model, local, world0, rot.to_matrix(rot.from_rotvec(np.radians(chest))))
        local = leaned
    world, pos = _world(model, local, root, placement)
    real_world, real_pos = world, pos
    flip = np.array([-1.0, 1.0, 1.0])
    if mirrored:
        # the left arm is solved as a right arm in the body mirrored through its sagittal plane (world X)
        world = MIRROR @ world @ MIRROR
        pos = pos * flip
        target = target * flip
        target_rot = MIRROR @ target_rot @ MIRROR
    shoulder = pos[:, model.upperarm]
    rel = target - shoulder
    dist = np.linalg.norm(rel, axis=-1)
    flex_max = math.radians(model.limits.ranges["elbow_flexion"][1])
    shortest = scale * math.sqrt(model.upper_m**2 + model.lower_m**2 + 2.0 * model.upper_m * model.lower_m * math.cos(flex_max))
    held = shoulder + rel / np.maximum(dist[:, None], 1e-9) * np.clip(dist, shortest, reach)[:, None]
    residual = np.linalg.norm(held - target, axis=-1)
    rest = MIRROR @ model.rest_world @ MIRROR if mirrored else model.rest_world
    hinge = _rx(np.full(frames, math.radians(model.hinge_deg)))
    hand_anat = target_rot @ rest[model.hand].T @ BASIS
    thorax_anat = world[:, model.thorax] @ rest[model.thorax].T @ BASIS
    # Work in body metres: the arm's lengths are at body scale 1.
    s_body = shoulder / scale
    w_body = held / scale
    posed_elbow = pos[:, model.forearm] / scale
    phis = _swivel_grid(model, s_body, w_body, posed_elbow)
    cand = _candidates(model, s_body, w_body, hand_anat, thorax_anat, phis)
    fw = np.ones(frames) if frame_weight is None else np.broadcast_to(np.asarray(frame_weight, dtype=np.float64), (frames,))
    at_contact = fw >= CONTACT_FRAME_WEIGHT
    sw = np.full(frames, SHOULDER_WEIGHT)
    excess_cost, shoulder_sq, every_cost = _excess_cost(model, cand, comfort=comfort, shoulder_weight=sw)
    posed_humerus = world[:, model.upperarm] @ rest[model.upperarm].T @ BASIS @ hinge
    window_cos = math.cos(math.radians(HUMERUS_WINDOW_DEG))

    def elbow_cost(elbows: Array, posed: Array, humeri: Array, posed_frames: Array) -> Array:
        c = _elbow_reference_cost(elbows, posed, elbow_ref, ELBOW_REF_WEIGHT)
        if elbow_window_m is not None:
            turned = (np.trace(np.swapaxes(humeri, -1, -2) @ posed_frames[:, None], axis1=-2, axis2=-1) - 1.0) / 2.0 < window_cos
            far = (np.linalg.norm(elbows - posed[:, None], axis=-1) > elbow_window_m / scale) | turned
            nearest = np.argmin(np.linalg.norm(elbows - posed[:, None], axis=-1), axis=1)
            far[np.arange(len(far)), nearest] = False
            c = np.where(far, 1e12, c)
        return c

    cost = fw[:, None] * excess_cost + elbow_cost(cand["elbow"], posed_elbow, cand["humerus"], posed_humerus)
    if elbow_hint is not None:
        hint = np.broadcast_to(np.asarray(elbow_hint, dtype=np.float64), (frames, 3)) * (flip if mirrored else 1.0) / scale
        cost = cost + ELBOW_HINT_WEIGHT * np.sum((cand["elbow"] - hint[:, None]) ** 2, axis=-1)
    if humerus_hint is not None:
        hinted = np.broadcast_to(np.asarray(humerus_hint, dtype=np.float64), (frames, 3, 3))
        if mirrored:
            hinted = MIRROR @ hinted @ MIRROR
        hinted = hinted @ rest[model.upperarm].T @ BASIS @ hinge
        taper = np.clip(1.0 - np.degrees(cand["flexion"]) / HUMERUS_HINT_FLEXION_DEG, 0.0, 1.0)
        cost = cost + (HUMERUS_HINT_WEIGHT * taper)[:, None] * np.sum((cand["humerus"] - hinted[:, None]) ** 2, axis=(-2, -1))
    torso = _clearance_torso(model, real_pos, scale) if clearance else None

    def body_depth(rows: Array, elbows: Array, wrists: Array, posed: Any = None) -> Array:
        """How deep each candidate's forearm reaches into the torso (m): elbows ``(n, K, 3)`` in the solve's (mirrored)
        body metres, wrists ``(n, 3)`` in its world metres; against the posed body's torso at frames ``rows``, or
        ``posed`` (a torso of these frames alone)."""

        t = torso if posed is None else posed
        if t is None:
            return np.zeros(elbows.shape[:2])
        e, w = elbows * scale, wrists
        if mirrored:
            e, w = e * flip, w * flip
        return _forearm_depth(model, t, rows if posed is None else np.arange(len(rows)), e, w, scale)

    clear_w = CLEARANCE_WEIGHT / np.maximum(fw, 1.0) if clearance_weight is None else np.broadcast_to(np.asarray(clearance_weight, dtype=np.float64), (frames,))
    # At runtime only going deeper than the posed (conditioned) arm costs, so a pose the track holds exactly stays
    # exact; conditioning prices any depth.
    allowance = (
        body_depth(np.arange(frames), posed_elbow[:, None, :], pos[:, model.hand])[:, 0] + CLEARANCE_RELATIVE_SLACK_M
        if clearance_relative
        else np.zeros(frames)
    )

    def clearance_cost(rows: Array, elbows: Array, wrists: Array, posed: Any = None) -> Array:
        return clear_w[rows, None] * np.maximum(body_depth(rows, elbows, wrists, posed) - allowance[rows, None], 0.0) ** 2

    # The ranges come first: a candidate whose shoulder is further out of range than the frame's best by more than
    # CLEARANCE_RANGE_SLACK_DEG2 gains nothing from its clearance (it pays at least the clearance of the best candidate
    # within the ranges).
    clear_cost = clearance_cost(np.arange(frames), cand["elbow"], held)
    range_ok = shoulder_sq <= np.min(shoulder_sq, axis=1, keepdims=True) + CLEARANCE_RANGE_SLACK_DEG2
    floor = np.min(np.where(range_ok, clear_cost, np.inf), axis=1, keepdims=True)
    cost = cost + np.where(range_ok, clear_cost, np.maximum(clear_cost, floor))
    if turn:
        # Where every swivel leaves an angle out of range, try turning the chest as well. Over a sequence the turn is
        # chosen as a path (a cost on its change) and smoothed, so the chest never snaps between turns.
        stuck = np.min(every_cost, axis=1) > 0.0
        if np.any(stuck):
            twists = np.asarray(TWISTS_DEG, dtype=np.float64)

            def turned_at(idx: Array, angle: Array) -> tuple[Any, ...]:
                vertical = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (len(idx), 3))
                r = rot.to_matrix(rot.from_axis_angle(vertical, np.radians(angle)))
                turned = _chest_rotation(model, local[idx], world[idx], r)
                w2, p2 = _world(model, turned, root[idx], placement)
                spine = _spine_angles(model, w2)
                in_range = np.ones(len(idx), dtype=bool)
                for name, value in spine.items():
                    in_range &= _excess(value, model.limits.ranges[name]) <= 0.0
                s2 = p2[:, model.upperarm]
                h2 = (
                    s2
                    + (target[idx] - s2)
                    / np.maximum(np.linalg.norm(target[idx] - s2, axis=-1, keepdims=True), 1e-9)
                    * np.clip(np.linalg.norm(target[idx] - s2, axis=-1), shortest, reach)[:, None]
                )
                phis2 = _swivel_grid(model, s2 / scale, h2 / scale, p2[:, model.forearm] / scale)
                c2 = _candidates(model, s2 / scale, h2 / scale, hand_anat[idx], w2[:, model.thorax] @ rest[model.thorax].T @ BASIS, phis2)
                e2, _sh2, _ev2 = _excess_cost(model, c2, comfort=comfort, shoulder_weight=sw[idx])
                ph2 = w2[:, model.upperarm] @ rest[model.upperarm].T @ BASIS @ hinge[idx]
                cost2 = fw[idx, None] * e2 + elbow_cost(c2["elbow"], p2[:, model.forearm] / scale, c2["humerus"], ph2) + TWIST_WEIGHT * angle[:, None] ** 2
                if torso is not None:
                    cost2 = cost2 + clearance_cost(idx, c2["elbow"], h2, _clearance_torso(model, p2, scale))
                # never at the cost of reach: the turned shoulder holds the wrist at least as close
                ok = in_range & (np.linalg.norm(h2 - target[idx], axis=-1) <= np.linalg.norm(held[idx] - target[idx], axis=-1) + 1e-3)
                return turned, w2, p2, s2, h2, phis2, c2, cost2, ok

            table = np.full((frames, len(twists)), np.inf)
            table[:, 0] = cost.min(axis=1)
            idx = np.flatnonzero(stuck)
            for column, twist in enumerate(twists[1:], start=1):
                *_rest, cost2, ok = turned_at(idx, np.full(len(idx), twist))
                table[idx, column] = np.where(ok, cost2.min(axis=1), np.inf)
            if smooth and frames > 1:
                free = ~stuck[:, None] & (np.arange(len(twists))[None, :] > 0)
                table = np.where(free, table[:, :1] + TWIST_WEIGHT * twists[None, :] ** 2, table)
                path = _path(np.where(np.isfinite(table), table, 1e12), twists, TWIST_SPEED_WEIGHT * (fps / REFERENCE_FPS) ** 2)
                width = max(1, round(TWIST_SMOOTH_S * fps / 2))
                kernel = np.r_[np.arange(1, width + 2), np.arange(width, 0, -1)].astype(np.float64)
                kernel /= kernel.sum()
                angle = np.convolve(np.pad(twists[path], width, mode="edge"), kernel, mode="valid")
            else:
                angle = twists[np.argmin(table, axis=1)]
            sel = np.flatnonzero(np.abs(angle) > 1e-6)
            if len(sel):
                turned, w2, p2, s2, h2, phis2, c2, cost2, ok = turned_at(sel, angle[sel])
                sel, keep = sel[ok], ok
                local[sel] = turned[keep]
                world[sel], pos[sel] = w2[keep], p2[keep]
                shoulder[sel], held[sel] = s2[keep], h2[keep]
                for key in cand:
                    cand[key][sel] = c2[key][keep]
                cost[sel] = cost2[keep]
                phis[sel] = phis2[keep]
                chest[sel] += angle[sel][:, None] * np.array([0.0, 0.0, 1.0])
            residual = np.linalg.norm(held - target, axis=-1)
    if reposition and smooth and frames > 1:
        # Where the shoulder stays out of its range at every swivel and chest turn, away from the contacts the hand
        # gives way instead: the wrist target moves (a few centimetres, smoothly over time) to where the shoulder holds.
        thorax_now = world[:, model.thorax] @ rest[model.thorax].T @ BASIS
        elbow_now = pos[:, model.forearm] / scale
        _total, shoulder_now, _every = _excess_cost(model, cand, comfort=comfort, shoulder_weight=sw)
        out_of_range = (np.sqrt(np.min(shoulder_now, axis=1)) > REPOSITION_MIN_DEG) & (fw < CONTACT_FRAME_WEIGHT)
        if np.any(out_of_range):

            def moved_to(idx: Array, shift: Array) -> tuple[Any, ...]:
                wanted = target[idx] + shift
                s2 = shoulder[idx]
                h2 = (
                    s2
                    + (wanted - s2)
                    / np.maximum(np.linalg.norm(wanted - s2, axis=-1, keepdims=True), 1e-9)
                    * np.clip(np.linalg.norm(wanted - s2, axis=-1), shortest, reach)[:, None]
                )
                phis2 = _swivel_grid(model, s2 / scale, h2 / scale, elbow_now[idx])
                c2 = _candidates(model, s2 / scale, h2 / scale, hand_anat[idx], thorax_now[idx], phis2)
                e2, _sh2, _ev2 = _excess_cost(model, c2, comfort=comfort, shoulder_weight=sw[idx])
                cost2 = fw[idx, None] * e2 + elbow_cost(c2["elbow"], elbow_now[idx], c2["humerus"], posed_humerus[idx])
                cost2 = cost2 + clearance_cost(idx, c2["elbow"], h2)
                return h2, phis2, c2, cost2

            idx = np.flatnonzero(out_of_range)
            best_cost = cost[idx].min(axis=1)
            best_shift = np.zeros((len(idx), 3))
            for step in REPOSITION_STEPS_M:
                for direction in np.vstack([np.eye(3), -np.eye(3)]):
                    shift = np.broadcast_to(scale * step * direction, (len(idx), 3))
                    _h2, _p2, _c2, cost2 = moved_to(idx, shift)
                    total = cost2.min(axis=1) + REPOSITION_WEIGHT * step * step
                    better = total < best_cost
                    best_cost = np.where(better, total, best_cost)
                    best_shift[better] = shift[better]
            shifts = np.zeros((frames, 3))
            shifts[idx] = best_shift
            width = max(1, round(TWIST_SMOOTH_S * fps / 2))
            shifts = _smooth_envelope(shifts, width)
            # never at a contact: the smoothed move tapers to nothing over the window around each contact frame
            near_contact = _smooth_envelope(at_contact.astype(np.float64)[:, None], width)[:, 0]
            shifts = shifts * np.clip(1.0 - near_contact, 0.0, 1.0)[:, None]
            sel = np.flatnonzero(np.linalg.norm(shifts, axis=-1) > 1e-6)
            if len(sel):
                h2, phis2, c2, cost2 = moved_to(sel, shifts[sel])
                held[sel] = h2
                for key in cand:
                    cand[key][sel] = c2[key]
                cost[sel] = cost2
                phis[sel] = phis2
            residual = np.linalg.norm(held - target, axis=-1)

    if smooth and frames > 1:
        relative = cand["humerus"] @ np.swapaxes(posed_humerus, -1, -2)[:, None] if continuity == "posed" else cand["humerus"]
        limit = model.limits.speed_limit_deg_s(model.skeleton.joints[model.upperarm])
        choice = _humerus_path(cost, relative, HUMERUS_SPEED_WEIGHT * (fps / REFERENCE_FPS) ** 2, cand["humerus"], math.radians(limit) / fps)
    else:
        choice = np.argmin(cost, axis=1)
    f_idx = np.arange(frames)
    humerus = cand["humerus"][f_idx, choice]
    untwisted = cand["untwisted"][f_idx, choice]
    hand_frame, wrist_angles, pron_rad = _clamp_hand(model, untwisted, hand_anat, sequence=smooth and frames > 1, fps=fps, pinned=fw >= CONTACT_FRAME_WEIGHT)
    forearm_frame = untwisted @ _rx(-pron_rad)
    # Back to the bones' world rotations: segment frame = delta @ rest-anatomical frame.
    w_hum = humerus @ np.swapaxes(BASIS @ hinge, -1, -2) @ rest[model.upperarm]
    w_fore = forearm_frame @ BASIS.T @ rest[model.forearm]
    w_hand = hand_frame @ BASIS.T @ rest[model.hand]
    twist_world = [(joint, (untwisted @ _rx(-share * pron_rad)) @ BASIS.T @ rest[joint]) for joint, share in model.twists]
    if mirrored:
        w_hum, w_fore, w_hand = (MIRROR @ w @ MIRROR for w in (w_hum, w_fore, w_hand))
        twist_world = [(joint, MIRROR @ w @ MIRROR) for joint, w in twist_world]
        target_rot = MIRROR @ target_rot @ MIRROR
        target = target * flip
        world = real_world
    out = local.copy()
    parents = model.skeleton.parents
    w_clav = world[:, model.clavicle]
    out[:, model.upperarm] = rot.from_matrix(np.swapaxes(w_clav, -1, -2) @ w_hum)
    out[:, model.forearm] = rot.from_matrix(np.swapaxes(w_hum, -1, -2) @ w_fore)
    out[:, model.hand] = rot.from_matrix(np.swapaxes(w_fore, -1, -2) @ w_hand)
    w_prev = {model.forearm: w_fore}
    for joint, w_tw in twist_world:
        parent = int(parents[joint])
        w_parent = w_prev.get(parent, w_fore)
        out[:, joint] = rot.from_matrix(np.swapaxes(w_parent, -1, -2) @ w_tw)
        w_prev[joint] = w_tw
    _w3, p3 = _world(model, out, root, placement)
    hand_position_out = p3[:, model.hand]
    body_depth_m = np.zeros(0)
    if torso is not None:
        from lib.replay.players.clearance import arm_body_depth

        body_depth_m = np.maximum(arm_body_depth(model.skeleton.joints, p3, scale, model.body, sides=(model.side,)), 0.0)
    projection = np.degrees(rot.angle(rot.from_matrix(np.swapaxes(target_rot, -1, -2) @ w_hand)))
    angles = {
        "elbow_flexion": np.degrees(cand["flexion"]),
        "forearm_pronation": wrist_angles["forearm_pronation"],
        "wrist_flexion": wrist_angles["wrist_flexion"],
        "wrist_deviation": wrist_angles["wrist_deviation"],
        "shoulder_elevation": cand["shoulder_elevation"][f_idx, choice],
        "shoulder_rotation": cand["shoulder_rotation"][f_idx, choice],
        "shoulder_plane": cand["shoulder_plane"][f_idx, choice],
    }
    excess = {name: _excess(value, model.limits.ranges[name]) for name, value in angles.items()}
    excess["shoulder_plane"] = np.where(angles["shoulder_elevation"] > 45.0, excess["shoulder_plane"], 0.0)
    solution = ArmSolution(
        local_rotation=out,
        hand_position=hand_position_out,
        hand_rotation=w_hand,
        projected=projection > 0.05,
        projection_deg=projection,
        residual_m=np.linalg.norm(hand_position_out - target, axis=-1),
        swivel_deg=_wrapped(np.degrees(phis[f_idx, choice])),
        chest_deg=chest,
        angles=angles,
        excess_deg=excess,
        cost=cost[f_idx, choice],
        body_depth_m=body_depth_m,
    )
    if single:
        return ArmSolution(
            local_rotation=solution.local_rotation[0],
            hand_position=solution.hand_position[0],
            hand_rotation=solution.hand_rotation[0],
            projected=solution.projected[0],
            projection_deg=solution.projection_deg[0],
            residual_m=solution.residual_m[0],
            swivel_deg=solution.swivel_deg[0],
            chest_deg=solution.chest_deg[0],
            angles={k: v[0] for k, v in solution.angles.items()},
            excess_deg={k: v[0] for k, v in solution.excess_deg.items()},
            cost=solution.cost[0],
            body_depth_m=solution.body_depth_m[0] if len(solution.body_depth_m) else solution.body_depth_m,
        )
    del residual
    return solution


SPINS_DEG: tuple[float, ...] = tuple(float(x) for x in np.arange(-180.0, 180.0, 15.0))
"""Spins of the paddle about its face normal tried per frame (the ball fixes the normal, not the blade's turn about it)."""
SPIN_SPEED_WEIGHT: float = 0.1
"""Cost per squared degree of spin change between consecutive frames at :data:`REFERENCE_FPS` (a 15 degree change in
1/30 s costs as much as 4.7 degrees of range excess); the spin path is chosen on frames at most that rate apart and
interpolated between them."""
SPIN_PRIOR_WEIGHT: float = 0.002
"""Cost per squared degree of spin away from the paddle's own (a weak preference: the spin is not evidence)."""
PROJECTION_WEIGHT: float = 1.0
"""Cost per squared degree the hand had to turn into range."""
RESIDUAL_WEIGHT: float = 1.0e4
"""Cost per squared metre the wrist stayed short."""
PADDLE_TURN_LIMIT_DEG_S: float = 1500.0
"""Fastest a held paddle turns: the recorded players' largest blade turn rate (1336-1486 deg/s over the capture). A
paddle path turning faster is one no hand follows: it is slowed to this before the arm holds it (the arm would otherwise turn through a joint's
whole range in a frame)."""
CHUNK: int = 4096
"""Frames (times spins) evaluated at once."""


@dataclass(frozen=True)
class PaddleHold:
    """
    A paddle held through the grip by the limit-aware arm.

    :ivar arm: The arm's solution.
    :ivar paddle_position: The blade centre the hand holds, ``(F, 3)``.
    :ivar paddle_rotation: The paddle axes the hand holds (width, tip, forehand normal), ``(F, 3, 3)``.
    :ivar spin_deg: The spin applied about the face normal, degrees.
    :ivar tilt_deg: How far the face normal moved beyond the spin (the projection the joints forced), degrees.
    :ivar moved_m: How far the blade centre moved, m.
    """

    arm: ArmSolution
    paddle_position: Array
    paddle_rotation: Array
    spin_deg: Array
    tilt_deg: Array
    moved_m: Array


def _spun_each(paddle_rotation: Array, spins_rad: Array) -> Array:
    """Every frame's paddle axes spun about their own face normal by every spin: ``(F, S, 3, 3)``."""

    normal = paddle_rotation[:, :, 2]
    axis = np.broadcast_to(normal[:, None], (len(normal), len(spins_rad), 3))
    angle = np.broadcast_to(spins_rad[None], (len(normal), len(spins_rad)))
    return rot.to_matrix(rot.from_axis_angle(axis, angle)) @ paddle_rotation[:, None]


def _spun(paddle_rotation: Array, spins_rad: Array) -> Array:
    """Each frame's paddle axes spun about their own face normal by that frame's spin: ``(F, 3, 3)``."""

    return rot.to_matrix(rot.from_axis_angle(paddle_rotation[:, :, 2], spins_rad)) @ paddle_rotation


def _limit_turn_rate(rotation: Array, limit_deg_s: float, fps: float) -> Array:
    """Rotation matrices ``(F, 3, 3)`` whose frame-to-frame turn is held to a rate (forward and backward passes, averaged)."""

    q = rot.continuous(rot.from_matrix(rotation))
    step = math.radians(limit_deg_s) / fps

    def sweep(seq: Array) -> Array:
        res = seq.copy()
        for k in range(1, len(res)):
            angle = float(rot.angle(rot.multiply(rot.conjugate(res[k - 1]), res[k])))
            if angle > step:
                res[k] = rot.slerp(res[k - 1], res[k], step / angle)
        return res

    forward = sweep(q)
    backward = sweep(forward[::-1])[::-1]
    return rot.to_matrix(rot.normalize(rot.slerp(forward, backward, np.full(len(q), 0.5))))


def spin_scores(
    model: ArmModel,
    grip: Any,
    local: Array,
    root: Array,
    placement: BodyPlacement,
    position: Array,
    rotation: Array,
    spins_rad: Array,
    weight: Array,
    sampled: Array,
    *,
    elbow_ref: ElbowRef = "posed",
    elbow_window_m: float | None = None,
    trunk: bool = True,
    comfort: bool = True,
    clearance: bool = True,
    clearance_relative: bool = True,
    clearance_weight: ArrayLike | None = None,
) -> Array:
    """
    Per sampled frame and spin, the cost of holding the paddle spun about its face normal: the arm's best swivel's
    range excess and comfort, plus ``weight`` per squared degree the hand had to turn and :data:`RESIDUAL_WEIGHT` per
    squared metre the wrist stayed short, ``(len(sampled), len(spins_rad))``.

    The chest's lean for reach barely depends on the spin: it leans once (toward the unspun wrists), and the spins are
    scored on the leaned body.
    """

    spun_all = _spun_each(rotation[sampled], spins_rad)  # (N, S, 3, 3)
    score = np.zeros((len(sampled), len(spins_rad)))
    flat_f = np.repeat(np.arange(len(sampled)), len(spins_rad))
    flat_s = np.tile(np.arange(len(spins_rad)), len(sampled))
    scored_local = local
    if trunk:
        base_p, _base_r = grip.hand_from_paddle(position, rotation, placement.scale)
        scored_local, _chest = _lean(model, local, root, placement, base_p, REACH_FRACTION * placement.scale * (model.upper_m + model.lower_m))
    for start in range(0, len(flat_f), CHUNK):
        f = flat_f[start : start + CHUNK]
        k = flat_s[start : start + CHUNK]
        g = sampled[f]
        hand_p, hand_r = grip.hand_from_paddle(position[g], spun_all[f, k], placement.scale)
        for turn in (False, True):
            part = (weight[g] >= TWIST_SCORE_WEIGHT) if turn else np.ones(len(g), dtype=bool)
            if (turn and not trunk) or not np.any(part):
                continue
            sol = solve_arm(
                model,
                scored_local[g[part]],
                root[g[part]],
                placement,
                hand_p[part],
                hand_r[part],
                elbow_ref=elbow_ref,
                elbow_window_m=elbow_window_m,
                smooth=False,
                trunk=False,
                comfort=comfort,
                twist=turn,
                clearance=clearance,
                clearance_weight=_clearance_weights(np.clip(weight[g[part]] / PROJECTION_WEIGHT, 1.0, 100.0))
                if clearance_weight is None
                else np.asarray(clearance_weight, dtype=np.float64)[f[part]],
                clearance_relative=clearance_relative,
            )
            w = weight[g[part]]
            score[f[part], k[part]] = sol.cost + w * sol.projection_deg**2 + np.maximum(RESIDUAL_WEIGHT, w * 1.0e4) * sol.residual_m**2
    return score


TWIST_SCORE_WEIGHT: float = 10.0
"""Frames whose face weight is at least this (the contacts) are scored with the chest free to turn as well."""


def sampled_frames(frames: int, fps: float | None, smooth: bool) -> tuple[Array, int]:
    """The frames a spin path is scored on (at most :data:`REFERENCE_FPS` apart, the last always) and their stride."""

    rate = REFERENCE_FPS if fps is None else fps
    stride = max(1, round(rate / REFERENCE_FPS)) if smooth else 1
    return np.unique(np.r_[np.arange(0, frames, stride), frames - 1]), stride


def smooth_path(values: Array, stride: int) -> Array:
    """A per-frame path interpolated from sampled frames, smoothed with a triangle ``4 * stride + 1`` frames wide."""

    if len(values) <= 4:
        return values
    half = 2 * stride
    kernel = np.r_[np.arange(1, half + 2), np.arange(half, 0, -1)].astype(np.float64)
    kernel /= kernel.sum()
    padded = np.pad(values, half, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def hold_paddle(
    model: ArmModel,
    grip: Any,
    local_rotation: ArrayLike,
    root_translation: ArrayLike,
    placement: BodyPlacement,
    paddle_position: ArrayLike,
    paddle_rotation: ArrayLike,
    *,
    spins_deg: tuple[float, ...] = SPINS_DEG,
    spin_prior_deg: ArrayLike | None = None,
    spin_path_deg: ArrayLike | None = None,
    projection_weight: ArrayLike | None = None,
    elbow_ref: ElbowRef = "posed",
    elbow_window_m: float | None = None,
    continuity: Continuity = "posed",
    fps: float | None = None,
    smooth: bool = True,
    trunk: bool = True,
    comfort: bool = True,
    reposition: bool = False,
    elbow_hint: ArrayLike | None = None,
    humerus_hint: ArrayLike | None = None,
    clearance: bool = True,
    clearance_relative: bool = True,
    clearance_weight: ArrayLike | None = None,
) -> PaddleHold:
    """
    Hold paddle poses through a grip (:class:`~lib.replay.players.grip.ShakehandGrip`), the paddle free to
    spin about its face normal: per frame every spin is scored by the arm's best swivel (range excess, comfort,
    projection, reach), the spin path is chosen with a cost on its speed (and a weak pull toward ``spin_prior_deg``),
    smoothed, and the arm solved along it. The paddle the hand then holds keeps the face normal and blade centre
    wherever the joints allow (``tilt_deg`` and ``moved_m`` say how far they forced it).

    :param paddle_position: Blade centres, ``(F, 3)``.
    :param paddle_rotation: Paddle axes as columns (width, tip, forehand normal), ``(F, 3, 3)``.
    :param spins_deg: The spins tried, degrees (relative to ``paddle_rotation``).
    :param spin_prior_deg: The spin to stay near (default 0: the paddle as given).
    :param spin_path_deg: The spin per frame, degrees, already chosen (no search).
    :param projection_weight: Per frame, the cost per squared degree the face normal may be turned into the joints'
        ranges (default :data:`PROJECTION_WEIGHT`): high where the normal is the ball's evidence (at contacts), low where
        it is the paddle IMU's (about 20-30 degrees of error between strokes).
    :param elbow_ref: See :func:`solve_arm`.
    :param elbow_window_m: See :func:`solve_arm`.
    :param continuity: See :data:`Continuity`.
    :param fps: The frames' rate (sequences): the paddle's turn rate is first held to :data:`PADDLE_TURN_LIMIT_DEG_S`.
    :param comfort: See :func:`solve_arm`.
    :param reposition: See :func:`solve_arm` (the paddle moves with the hand).
    :param elbow_hint: See :func:`solve_arm`.
    :param humerus_hint: See :func:`solve_arm`.
    :param clearance: See :func:`solve_arm` (the spins are scored with it too).
    :param clearance_relative: See :func:`solve_arm`.
    :param clearance_weight: Per frame, the clearance's weight in the swivel and the spin choices (default: from the
        projection weights, none at the contacts; :func:`solve_arm`).
    """

    local = np.array(local_rotation, dtype=np.float64, ndmin=3)
    root = np.array(root_translation, dtype=np.float64, ndmin=2)
    position = np.array(paddle_position, dtype=np.float64, ndmin=2)
    rotation = np.asarray(paddle_rotation, dtype=np.float64).reshape(-1, 3, 3)
    frames = len(local)
    if fps is not None and frames > 1:
        rotation = _limit_turn_rate(rotation, PADDLE_TURN_LIMIT_DEG_S, fps)
    rate = REFERENCE_FPS if fps is None else fps
    if spin_path_deg is not None:
        chosen = np.radians(np.broadcast_to(np.asarray(spin_path_deg, dtype=np.float64), (frames,)))
    else:
        spins = np.radians(np.asarray(spins_deg, dtype=np.float64))
        prior = np.zeros(frames) if spin_prior_deg is None else np.broadcast_to(np.asarray(spin_prior_deg, dtype=np.float64), (frames,))
        weight = (
            np.full(frames, PROJECTION_WEIGHT) if projection_weight is None else np.broadcast_to(np.asarray(projection_weight, dtype=np.float64), (frames,))
        )
        sampled, stride = sampled_frames(frames, fps, smooth)
        score = spin_scores(
            model,
            grip,
            local,
            root,
            placement,
            position,
            rotation,
            spins,
            weight,
            sampled,
            elbow_ref=elbow_ref,
            elbow_window_m=elbow_window_m,
            trunk=trunk,
            comfort=comfort,
            clearance=clearance,
            clearance_relative=clearance_relative,
            clearance_weight=None if clearance_weight is None else np.broadcast_to(np.asarray(clearance_weight, dtype=np.float64), (frames,))[sampled],
        )
        spins_d = np.degrees(spins)
        score = score + SPIN_PRIOR_WEIGHT * _wrapped(spins_d[None, :] - prior[sampled, None]) ** 2
        effective = rate / stride
        choice = _path(score, spins_d, SPIN_SPEED_WEIGHT * (effective / REFERENCE_FPS) ** 2) if (smooth and len(sampled) > 1) else np.argmin(score, axis=1)
        chosen = np.interp(np.arange(frames), sampled, np.unwrap(spins[choice]))
        if smooth and frames > 4:
            chosen = smooth_path(chosen, stride)
    target_rotation = _spun(rotation, chosen)
    hand_p, hand_r = grip.hand_from_paddle(position, target_rotation, placement.scale)
    arm = solve_arm(
        model,
        local,
        root,
        placement,
        hand_p,
        hand_r,
        elbow_ref=elbow_ref,
        elbow_window_m=elbow_window_m,
        continuity=continuity,
        fps=rate,
        smooth=smooth,
        trunk=trunk,
        comfort=comfort,
        reposition=reposition,
        elbow_hint=elbow_hint,
        humerus_hint=humerus_hint,
        clearance=clearance,
        clearance_relative=clearance_relative,
        clearance_weight=clearance_weight,
        frame_weight=None
        if projection_weight is None
        else np.clip(np.broadcast_to(np.asarray(projection_weight, dtype=np.float64), (frames,)) / PROJECTION_WEIGHT, 1.0, 100.0),
    )
    held_p, held_r = grip.paddle_from_hand(arm.hand_position, arm.hand_rotation, placement.scale)
    tilt = np.degrees(np.arccos(np.clip(np.sum(held_r[:, :, 2] * rotation[:, :, 2], axis=-1), -1.0, 1.0)))
    return PaddleHold(
        arm=arm,
        paddle_position=held_p,
        paddle_rotation=held_r,
        spin_deg=_wrapped(np.degrees(chosen)),
        tilt_deg=tilt,
        moved_m=np.linalg.norm(held_p - position, axis=-1),
    )


def clamp_clavicles(
    skeleton: SkeletonRest, local_rotation: Array, root_translation: Array, placement: BodyPlacement, limits: JointLimits, sides: tuple[str, ...] = ("R", "L")
) -> tuple[Array, int]:
    """
    Each clavicle's direction held inside its range (:mod:`~lib.replay.players.anatomy`'s elevation and
    protraction, the chest's frame): an out-of-range clavicle is turned by the shortest arc onto the nearest in-range
    direction, its children riding along (the arm then holds the hand from the clamped shoulder). The retarget copies
    the suit's clavicle, whose bind differs from the rig's; its protraction leaves the rig's range in 3.6% of session
    2's suited frames.

    :return: The clamped local rotations (a copy) and how many clavicle-frames changed.
    """

    out = np.array(local_rotation, dtype=np.float64, copy=True)
    rest_local, rest_root = rest_pose(skeleton)
    rest_world = rot.to_matrix(forward_kinematics(skeleton, rest_local, rest_root, BodyPlacement(1.0, (0.0, 0.0, 0.0)))[0])
    thorax = skeleton.index("Spine02")
    elevation_range = limits.ranges["clavicle_elevation"]
    protraction_range = limits.ranges["clavicle_protraction"]
    changed = 0
    for side in sides:
        name = f"{side}_Clavicle"
        if name not in skeleton.joints:
            continue
        clavicle = skeleton.index(name)
        world_q, _positions = forward_kinematics(skeleton, out, root_translation, placement)
        world = rot.to_matrix(world_q)
        dp = world[:, thorax] @ rest_world[thorax].T
        d = np.swapaxes(dp, -1, -2) @ (world[:, clavicle] @ rest_world[clavicle].T)
        mirror = MIRROR if side == "L" else np.eye(3)
        direction = (BASIS.T @ mirror @ d @ mirror @ BASIS)[:, :, 0]
        elevation = np.degrees(np.arcsin(np.clip(direction[:, 2], -1.0, 1.0)))
        protraction = np.degrees(np.arctan2(direction[:, 1], direction[:, 0]))
        e, p = np.clip(elevation, *elevation_range), np.clip(protraction, *protraction_range)
        bad = np.flatnonzero((np.abs(e - elevation) > 1e-6) | (np.abs(p - protraction) > 1e-6))
        if not len(bad):
            continue
        er, pr = np.radians(e[bad]), np.radians(p[bad])
        target = np.stack([np.cos(er) * np.cos(pr), np.cos(er) * np.sin(pr), np.sin(er)], axis=-1)
        turn = rot.to_matrix(rot.min_arc(direction[bad], target))  # in the anatomical basis
        world_turn = dp[bad] @ mirror @ BASIS @ turn @ BASIS.T @ mirror @ np.swapaxes(dp[bad], -1, -2)
        parent = int(skeleton.parents[clavicle])
        new_world = rot.from_matrix(world_turn @ world[bad, clavicle])
        out[bad, clavicle] = rot.multiply(rot.conjugate(world_q[bad, parent]), new_world)
        changed += len(bad)
    return out, changed


def limit_joint_speed(local_rotation: Array, joint: int, limit_deg_s: float, fps: float) -> Array:
    """
    A joint's local rotation track with its angular speed held to a limit (a forward then a backward pass, each
    turning a too-fast step back to the limit; the two passes averaged), a copy.
    """

    out = np.array(local_rotation, dtype=np.float64, copy=True)
    q = out[:, joint].copy()
    step = math.radians(limit_deg_s) / fps

    def sweep(seq: Array) -> Array:
        res = seq.copy()
        for k in range(1, len(res)):
            delta = rot.multiply(rot.conjugate(res[k - 1]), res[k])
            angle = float(rot.angle(delta))
            if angle > step:
                res[k] = rot.slerp(res[k - 1], res[k], step / angle)
        return res

    forward = sweep(q)
    backward = sweep(forward[::-1])[::-1]
    out[:, joint] = rot.normalize(rot.slerp(forward, backward, np.full(len(q), 0.5)))
    return out


__all__ = [
    "ARM_ANGLES",
    "BOARD_TOLERANCE_M",
    "SPINS_DEG",
    "SWIVELS",
    "ArmModel",
    "ArmSolution",
    "Continuity",
    "PaddleHold",
    "clamp_clavicles",
    "hold_paddle",
    "limit_joint_speed",
    "sampled_frames",
    "smooth_path",
    "solve_arm",
    "spin_scores",
]
