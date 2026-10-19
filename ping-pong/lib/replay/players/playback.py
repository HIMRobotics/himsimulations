"""
Play a character's motion track, with the stroke controller's adaptations applied through arm IK.

Each render tick the animator samples the track at the (warped) motion time,
slerping between frames; joints the track does not drive keep the body's
rest pose. A stroke's adaptation (a time warp, a blade-centre offset and a
blade tilt, :class:`lib.replay.players.strokes.StrokePlan`) fades in
over :attr:`AdaptationBounds.blend_s`, holds through contact and the follow-through and fades out over :attr:`AdaptationBounds.release_s`:

- the motion time is ``tau(t) = t - w(t) * dt``, so the recorded contact pose
  arrives ``dt`` later (``w`` ramps with zero slope, so ``tau`` stays monotonic
  and the swing speed at contact is the recorded one);
- the paddle target is the recorded paddle moved by ``w * dp`` and turned by
  ``w`` of the tilt; two-bone IK on the paddle arm (keeping the arm's own bend
  plane) and the hand's world rotation realise it; the twist the IK adds to the
  hand is moved into the forearm (the recorded twist stays where it was).

A stroke whose package carries a reconstructed paddle contact state
(:class:`lib.replay.session.PaddleContactTarget`)
first has its recorded paddle corrected onto that state, with the same blend
profile around its contact: the blade's velocity at contact set to the
reconstructed one (a windowed term that leaves the contact pose in place), the
face turned onto the reconstructed normal and
the blade moved onto the reconstructed blade centre (or, without one, so the
face passes through the contact point, in-plane only when the point would
fall off the face). That *reference correction* is the
capture's own error, logged per stroke; the stroke controller plans and is
bounded relative to the corrected reference (:meth:`CharacterAnimator.reference_paddle`).

Where a correction or an adaptation applies, the paddle the physics drives is
the plan's (the corrected reference with the adaptation), and the arm IK makes
the hand follow it; how far the hand stays from holding it is reported as a
motion-QA quantity. Elsewhere the paddle is the recorded one, in the hand.

**The shakehand grip and the joint limits.** When the replay gives the animator its strokes
(:meth:`CharacterAnimator.set_references`), the track is first *conditioned*
(:meth:`CharacterAnimator.condition`): the paddle every frame (the track's own, which the paddle IMU and the
reconstruction calibrated) is held through :mod:`~lib.replay.players.grip`'s shakehand grip by
:mod:`~lib.replay.players.arm_ik`'s limit-aware arm (the elbow a hinge in the rig's plane, the paddle's
orientation shared over the forearm, wrist, swivel, shoulder and chest, the paddle free to spin about its face
normal), with the fingers around the handle. Where the joints cannot hold a paddle pose it moves to the nearest one
they can: at contacts the face normal is held tightly (it is the ball's evidence), between strokes loosely (the paddle
IMU's, about 20-30 degrees of error). The suit's hand roll (130-150 degrees off) and the recorded grip, which puts the
blade on the back of the hand, are not used. At runtime the same arm holds the corrected and adapted paddle, so a
plan's blade and the physics paddle are always the hand's.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lib.replay.players import rotations as rot
from lib.replay.players.arm_ik import (
    CLEARANCE_WEIGHT,
    ELBOW_WINDOW_M,
    ArmModel,
    _smooth_envelope,
    clamp_clavicles,
    hold_paddle,
    limit_joint_speed,
    solve_arm,
)
from lib.replay.players.clearance import ARM_SEGMENTS, has_torso, push_out, torso_frames
from lib.replay.players.grip import RELABEL, ShakehandGrip, apply_relaxed_fingers, forehand_on_back, relaxed_fingers, shakehand_grip
from lib.replay.players.kinematics import BodyPlacement, apply_two_bone, chain_kinematics, forward_kinematics, transfer_added_twist
from lib.replay.players.skeleton import ROOT_JOINT, SkeletonRest
from lib.replay.players.strokes import AdaptationBounds, PaddleState, StrokePlan
from lib.replay.session import CharacterSpec, MotionTrack, StrokeTarget, TrackArrays

Array = NDArray[Any]

FLOOR_Z_M: float = -0.76
ARM_SOFTNESS: float = 0.04
"""Soft-IK band of the paddle arm, as a fraction of its length (about 2 cm): the elbow eases toward straight instead
of locking, so a reference on a nearly straight arm no longer pops it (:func:`~lib.replay.players.kinematics.soft_reach`).
At the arm's exact full length the reach falls short by 0.37 of the band (about 8 mm), which the reach bisection
reports as shortfall."""
VELOCITY_PROBE_S: float = 0.004
"""Half-width of the central difference that gives the recorded paddle's velocity."""


def grip_matrix(spec: CharacterSpec) -> Array:
    """The paddle's axes in the hand joint frame: columns X (width), Y (tip), Z (forehand)."""

    normal = np.asarray(spec.grip.forehand_normal, dtype=np.float64)
    tip = np.asarray(spec.grip.handle_axis, dtype=np.float64)
    tip = tip - float(tip @ normal) * normal
    tip /= np.linalg.norm(tip)
    return np.column_stack([np.cross(tip, normal), tip, normal])


def adaptation_weight(t: float, contact_s: float, bounds: AdaptationBounds) -> float:
    """Blend weight of an adaptation at time ``t``: 0, a smooth ramp in, 1 through contact and the follow-through, a longer smooth ramp out, 0."""

    if t < contact_s:
        offset, plateau, ramp = contact_s - t, bounds.hold_s, bounds.blend_s
    else:
        offset, plateau, ramp = t - contact_s, bounds.hold_s + bounds.follow_through_s, bounds.release_s
    if offset <= plateau:
        return 1.0
    return float(1.0 - rot.smoothstep((offset - plateau) / ramp))


FACE_HALF_WIDTH_M: float = 0.075
FACE_HALF_LENGTH_M: float = 0.0785
FACE_HALF_THICKNESS_M: float = 0.0068
"""Blade outline semi-axes and half thickness with rubbers (``lib.physics`` paddle priors)."""
CONTACT_INSIDE_SHARE: float = 0.8
"""A reconstructed contact point within this share of the outline needs no in-plane correction."""
TURN_COST_M_PER_RAD: float = 0.10
"""Weight of the blade's turn about the reconstructed normal against the hand's displacement: 10 cm of reach per radian."""
REACH_TOLERANCE_M: float = 0.005
"""The hand holds the paddle to within this; beyond it the arm cannot reach the planned blade (a motion-QA finding)."""
HOLD_BISECTIONS: int = 6
"""Bisections of the held share of a nearest blade that the arm holds at contact but not throughout its window."""
IK_DELTA_RATE_RAD_S: float = math.radians(1000.0)
"""Fastest any joint may move away from (or back toward) the recorded track under the arm IK: about 42 degrees a frame at
24 Hz. Glitches are kept out by the runtime clamp on the arm's speed and velocity change
(:meth:`CharacterAnimator._clamp_speed`), so this only stops the IK from snapping. At 300 deg/s plans at their bounds
lagged by 2.8-3.0 cm at p95 and at 600 by up to 2.4 cm; at 1000 the worst recorded session misses by 1.4 cm, with at
most one glitch episode per session (about five allowed)."""
ARM_TWIST_LIMIT_RAD: float = math.radians(90.0)
"""Soft bound on the twist the arm IK adds to the forearm (about the forearm's pronation range either way)."""
ARM_TWIST_RATE_RAD_S: float = math.radians(600.0)
"""Soft bound on how fast that added twist may change: below the 900 deg/s motion-QA pop rate."""
REGISTRATION_SD_M: dict[str, float] = {"suited": 0.10, "opponent": 0.05}
"""One-sigma position error of the captured blade itself, by role. Suited: the registered mocap's arms land 15-25 cm
off the visible arms at full reach, about 1.5-2.5 sigma of 10 cm. Opponent: the inferred motion's blade centre meets
the reconstructed ball within p50 2.7-3.9 cm and p90 8-8.6 cm, about 5 cm."""
REGISTRATION_SD_DEG: float = 5.0
"""One-sigma orientation error of the captured blade (the registration's yaw groups sit within 4 degrees)."""
REFERENCE_REJECT_D2: float = 25.745
"""A reconstructed contact state whose disagreement with the capture has a squared Mahalanobis distance above this is not
used as a reference: the chi-square(5) quantile with a 1e-4 upper tail (3 position axes against the contact point's sd
and the registration's, 2 normal axes against the normal's sd and the registration's; about 5 sigma combined). Over
the 909 corrections of seven sessions a model that held would reject about 0.1 by chance, so a rejection says the
capture and the reconstruction disagree beyond their stated uncertainty."""
REFERENCE_DISAGREE_D2: float = 15.086
"""The chi-square(5) 1 % tail: corrections beyond it are listed as disagreeing (``disagreeing_references``)."""
PLAN_REACH_SAMPLES_S: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10)
"""Times around a planned contact at which the plan's blade must be reachable (the adaptation's weight there), at
least; :func:`reach_lags` adds the rest of the window the adaptation holds at strength."""
REACH_LAG_STEP_S: float = 0.10
"""Spacing of the reach samples beyond :data:`PLAN_REACH_SAMPLES_S`."""
REACH_MIN_WEIGHT: float = 0.25
"""Reach samples where the adaptation weighs less than this are not checked (the arm eases back onto the track)."""


def reach_lags(bounds: AdaptationBounds) -> tuple[float, ...]:
    """
    The times (relative to contact) at which an adaptation's blade must be held: :data:`PLAN_REACH_SAMPLES_S` and every
    :data:`REACH_LAG_STEP_S` over the ramp in, the plateau through the follow-through and the ramp out while it still
    weighs :data:`REACH_MIN_WEIGHT` (checking only +-0.1 s leaves the follow-through unchecked, where planned blades
    were missed by 2-10 cm).
    """

    start = -(bounds.hold_s + bounds.blend_s)
    end = bounds.hold_s + bounds.follow_through_s + bounds.release_s
    grid = np.arange(start, end + 1e-9, REACH_LAG_STEP_S)
    lags = sorted({round(float(lag), 4) for lag in (*PLAN_REACH_SAMPLES_S, *grid)})
    return tuple(lag for lag in lags if adaptation_weight(lag, 0.0, bounds) >= REACH_MIN_WEIGHT or lag in PLAN_REACH_SAMPLES_S)


REFERENCE_PROFILE = AdaptationBounds(blend_s=0.15, hold_s=0.03, follow_through_s=0.0, release_s=0.25)
"""The time profile of a reference correction: the capture's error matters at contact, so the blade is corrected over
0.15 s into contact and released over 0.25 s after it, not across the whole follow-through."""
VELOCITY_WINDOW_S: float = 0.05
"""Width of the window that carries a reference velocity correction: the offset ``dv (t - t_c) g((t - t_c) / tau)``
with ``g(x) = exp(-x^2 / 2)`` adds ``dv`` to the blade's velocity at contact without moving its contact pose, for an
excursion of at most ``0.61 |dv| tau`` (3 cm per m/s), a wrist snap the 30 fps capture under-reads."""


CONTACT_WINDOW_S: float = 0.10
"""Within this of a contact the paddle's face normal is the ball's evidence and is held tightly."""
CONTACT_PROJECTION_WEIGHT: float = 100.0
"""Cost per squared degree the face normal turns into the joints' ranges near a contact."""
FREE_PROJECTION_WEIGHT: float = 0.2
"""The same away from contacts (the paddle IMU's normal, about 20-30 degrees of error)."""
CONTACT_PIN_S: float = 0.034
"""Frames this close to a contact (and at least the one either side) are pinned: the paddle and its velocity there are
the ball's evidence (one frame either side at 30 fps; at 120 fps four)."""
RUNTIME_SPEED_SHARE: float = 0.95
"""The share of its glitch ceiling a paddle-arm joint may turn at, at runtime (:meth:`CharacterAnimator.pose_at`)."""
FREE_HAND_BLADE_CLEARANCE_M: float = 0.17
"""The free hand's least distance from the blade centre: the recorded tracks' p5 (sessions 2, 5 and 7: 17-29 cm; p1
11-18 cm)."""
FREE_HAND_ARM_CLEARANCE_M: float = 0.25
"""The free hand's least distance from the playing forearm: the recorded tracks' p5 (25-39 cm; p1 17-25 cm)."""
FREE_HAND_MAX_ELEVATION_DEG: float = 90.0
"""The free wrist no higher than its shoulder (its direction from the shoulder at most this from straight down): the
recorded tracks put it higher in under 1% of frames (p99 59-97 degrees), never beside the head."""
FREE_HAND_SMOOTH_S: float = 0.3
"""The free hand's moves are smoothed over this long."""
RUNTIME_CLEARANCE_CONTACT_S: float = 0.15
"""The runtime arm IK keeps the forearm out of the torso only this far from a contact (relative to the conditioned
track: it never goes deeper than the track's own arm); nearer, the planned blade is held as it is (acting at the
contacts costs strokes)."""
FREE_HAND_TORSO_CLEARANCE_M: float = 0.03
"""The free wrist keeps this clear of the torso (:func:`~lib.replay.players.clearance.push_out`)."""
REFERENCE_FPS_GLITCH: float = 24.0
"""The glitch rule's reference rate (:data:`~lib.replay.players.limits.REFERENCE_FPS`): a joint's velocity
may change by at most its ceiling times this per second."""
RUNTIME_SPEED_GAP_S: float = 0.2
"""Rendered poses further apart than this (a cut, a seek) are not speed-clamped against each other."""
NEAR_PROJECTION_WEIGHT: float = 5.0
"""The same inside a contact window but off the frames bracketing the contact (those are pinned at
:data:`CONTACT_PROJECTION_WEIGHT`): weighted, but free to be smoothed into the motion around it."""
TILT_TOLERANCE_DEG: float = 1.0
"""A plan whose face the arm holds within this is held; beyond it the plan is projected (its tilt too)."""

REFERENCE_BLEND_SPEED_MPS: float = 1.5
"""Fastest the blend of a reference correction may move the blade (about a third of a stroke's hand speed)."""
REFERENCE_BLEND_TURN_DEG_S: float = 300.0
"""Fastest it may turn the blade (a third of the 900 deg/s motion-QA pop rate)."""


def reference_profile(offset_m: float, turn_deg: float) -> AdaptationBounds:
    """
    The blend of one reference correction: :data:`REFERENCE_PROFILE`, its ramps lengthened so the correction never
    moves the blade faster than :data:`REFERENCE_BLEND_SPEED_MPS` or turns it faster than
    :data:`REFERENCE_BLEND_TURN_DEG_S` (the soft-IK philosophy for time: a large but legitimate correction eases in
    and out instead of whipping the arm through the IK in a few frames).
    """

    ramp = max(offset_m / REFERENCE_BLEND_SPEED_MPS, turn_deg / REFERENCE_BLEND_TURN_DEG_S)
    return dataclasses.replace(REFERENCE_PROFILE, blend_s=max(REFERENCE_PROFILE.blend_s, ramp), release_s=max(REFERENCE_PROFILE.release_s, ramp))


@dataclass(frozen=True)
class PlanReach:
    """
    What the arm can hold of a plan.

    :ivar worst_m: The hand's largest residual over the samples around contact, m.
    :ivar held_offset_m: At contact, the blade the arm holds, as an offset from the reference blade, m.
    :ivar shortfall_m: At contact, the planned blade less the held one, m.
    :ivar held_rotation_rad: At contact, the blade turn the arm holds (rotation vector from the reference blade), or
        ``None`` when it holds the plan's.
    :ivar tilt_shortfall_deg: How far the held blade's turn falls short of the plan's at contact, degrees.
    """

    worst_m: float
    held_offset_m: Array
    shortfall_m: Array
    held_rotation_rad: Array | None = None
    tilt_shortfall_deg: float = 0.0

    @property
    def reachable(self) -> bool:
        return self.worst_m <= REACH_TOLERANCE_M and self.tilt_shortfall_deg <= TILT_TOLERANCE_DEG


@dataclass(frozen=True)
class ReferenceCorrection:
    """
    One stroke's correction of the recorded paddle onto its reconstructed contact state.

    :ivar stroke: Stroke index.
    :ivar contact_s: Contact time (motion time), s.
    :ivar offset_m: Blade-centre shift, m.
    :ivar rotation: Face turn as a rotation vector (about the blade centre), rad.
    :ivar recorded_speed_mps: The recorded paddle's speed at contact.
    :ivar reconstructed_speed_mps: The reconstructed paddle speed at contact.
    :ivar velocity_delta: Reconstructed minus recorded blade velocity at contact, imposed through the window
        :data:`VELOCITY_WINDOW_S`, m/s.
    :ivar unreachable_m: How far the reconstructed blade lay beyond the arm's reach at contact (the correction keeps
        only the reachable part), m.
    """

    stroke: int
    contact_s: float
    offset_m: Array
    rotation: Array
    recorded_speed_mps: float
    reconstructed_speed_mps: float
    velocity_delta: Array
    unreachable_m: float = 0.0
    profile: AdaptationBounds = field(default_factory=lambda: REFERENCE_PROFILE)

    def offset_at(self, motion_time: float) -> Array:
        """The unblended blade-centre correction at a motion time: the pose shift plus the windowed velocity term."""

        lag = motion_time - self.contact_s
        x = lag / VELOCITY_WINDOW_S
        return self.offset_m + self.velocity_delta * lag * math.exp(-0.5 * x * x)

    @property
    def velocity_change_mps(self) -> float:
        return float(np.linalg.norm(self.velocity_delta))

    @property
    def offset_cm(self) -> float:
        return 100.0 * float(np.linalg.norm(self.offset_m))

    @property
    def tilt_deg(self) -> float:
        return math.degrees(float(np.linalg.norm(self.rotation)))


@dataclass
class ActiveAdaptation:
    """
    A plan the animator is applying, with a cross-fade from the plan it replaced.

    :ivar plan: The current plan.
    :ivar previous: The plan it replaced, faded out over ``fade_s`` from ``replaced_at``.
    :ivar replaced_at: When the replacement happened, s.
    :ivar fade_s: Cross-fade length, s.
    """

    plan: StrokePlan
    previous: StrokePlan | None = None
    replaced_at: float = -math.inf
    fade_s: float = 0.1

    def parameters(self, t: float) -> tuple[float, Array, Array, float]:
        """``(contact_s, offset, rotation vector, time shift)`` at ``t``, cross-faded."""

        plan = self.plan
        if self.previous is None or t >= self.replaced_at + self.fade_s:
            return plan.contact_s, plan.offset_m, plan.rotation, plan.time_shift_s
        u = float(rot.smoothstep((t - self.replaced_at) / self.fade_s))
        old = self.previous
        return (
            (1 - u) * old.contact_s + u * plan.contact_s,
            (1 - u) * old.offset_m + u * plan.offset_m,
            (1 - u) * old.rotation + u * plan.rotation,
            (1 - u) * old.time_shift_s + u * plan.time_shift_s,
        )


@dataclass(frozen=True)
class AnimatedPose:
    """
    One evaluated pose.

    :ivar time_s: Session time, s.
    :ivar motion_time_s: The warped track time sampled, s.
    :ivar local_rotation: Every joint's local rotation ``(J, 4)``.
    :ivar root_translation: ``Hip`` local translation, skeleton metres (body scale 1).
    :ivar paddle_position: Blade centre, table frame, m.
    :ivar paddle_rotation: Blade axes (columns X, Y, Z).
    :ivar weight: Adaptation weight applied.
    :ivar ik_residual_m: Distance the arm IK left between hand and target, m.
    :ivar local_translation: The track's per-frame translations of its translated joints at this pose,
        ``(T, 3)`` metres at body scale 1 in :attr:`CharacterAnimator.translated` order, or ``None``.
    :ivar ik_active: Whether the arm IK chased a corrected or adapted paddle this pose; the paddle is then the plan's,
        and :attr:`ik_residual_m` is how far the hand stayed from holding it (a motion-QA quantity).
    :ivar held_paddle_position: Where the hand actually holds the blade centre (forward kinematics of the solved arm).
    :ivar hand_gap_m: How far the carried paddle (:attr:`paddle_position`) is from the blade the posed hand holds
        through the grip, m: the motion gate's "the paddle never leaves the hand" measurement (0 by construction here).
    """

    time_s: float
    motion_time_s: float
    local_rotation: Array
    root_translation: Array
    paddle_position: Array
    paddle_rotation: Array
    weight: float
    ik_residual_m: float
    local_translation: Array | None = None
    ik_active: bool = False
    held_paddle_position: Array | None = None
    hand_gap_m: float = 0.0


def free_hand_target(
    skeleton: SkeletonRest,
    placement: BodyPlacement,
    blade_offset: Array,
    local: Array,
    root: Array,
    wanted: Array,
    free_side: str,
    paddle_side: str,
    fps: float,
) -> tuple[Array, Array]:
    """
    Where a free hand goes (the track's conditioning): its track's own position
    (``wanted``, ``(F, 3)``), moved only as far as it must to stay :data:`FREE_HAND_BLADE_CLEARANCE_M` from the blade
    centre and :data:`FREE_HAND_ARM_CLEARANCE_M` from the playing forearm, :data:`FREE_HAND_TORSO_CLEARANCE_M` out of
    the torso, and no higher than its shoulder (:data:`FREE_HAND_MAX_ELEVATION_DEG`); the move is smoothed over
    :data:`FREE_HAND_SMOOTH_S`.

    :param blade_offset: The blade centre in the paddle hand's frame, m (body scale applied).
    :param local: The body (``(F, J, 4)``) with its paddle arm already holding the paddle.
    :return: The targets and the per-frame move, m.
    """

    world, positions = forward_kinematics(skeleton, local, root, placement)
    hand = positions[:, skeleton.index(f"{paddle_side}_Hand")]
    elbow = positions[:, skeleton.index(f"{paddle_side}_Forearm")]
    blade = hand + np.einsum("fij,j->fi", rot.to_matrix(world[:, skeleton.index(f"{paddle_side}_Hand")]), blade_offset)
    shoulder = positions[:, skeleton.index(f"{free_side}_Upperarm")]
    target = np.array(wanted, dtype=np.float64, copy=True)
    # no higher than the shoulder: the wrist's direction from the shoulder at most FREE_HAND_MAX_ELEVATION_DEG from down
    v = target - shoulder
    length = np.linalg.norm(v, axis=-1)
    down = np.array([0.0, 0.0, -1.0])
    elevation = np.degrees(np.arccos(np.clip((v @ down) / np.maximum(length, 1e-9), -1.0, 1.0)))
    high = elevation > FREE_HAND_MAX_ELEVATION_DEG
    if np.any(high):
        axis = np.cross(v[high], down)
        axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-9)
        turn = rot.to_matrix(rot.from_axis_angle(axis, np.radians(elevation[high] - FREE_HAND_MAX_ELEVATION_DEG)))
        target[high] = shoulder[high] + np.einsum("fij,fj->fi", turn, v[high])
    # clear of the blade and the playing forearm
    for _ in range(2):
        away = target - blade
        gap = np.linalg.norm(away, axis=-1)
        short = np.maximum(FREE_HAND_BLADE_CLEARANCE_M - gap, 0.0)
        target = target + away / np.maximum(gap, 1e-9)[:, None] * short[:, None]
        segment = hand - elbow
        u = np.clip(np.sum((target - elbow) * segment, axis=-1) / np.maximum(np.sum(segment * segment, axis=-1), 1e-9), 0.0, 1.0)
        nearest = elbow + u[:, None] * segment
        away = target - nearest
        gap = np.linalg.norm(away, axis=-1)
        short = np.maximum(FREE_HAND_ARM_CLEARANCE_M - gap, 0.0)
        target = target + away / np.maximum(gap, 1e-9)[:, None] * short[:, None]
        # and out of the torso (the wrist, as the shared arm-into-body definition's hand sphere)
        if has_torso(skeleton.body):
            torso = torso_frames(skeleton.joints, positions, placement.scale, skeleton.body)
            target = push_out(torso, target, ARM_SEGMENTS[free_side][0][2] * placement.scale, FREE_HAND_TORSO_CLEARANCE_M)
    move = target - np.asarray(wanted, dtype=np.float64)
    width = max(1, round(FREE_HAND_SMOOTH_S * fps / 2))
    move = _smooth_envelope(move, width)
    return np.asarray(wanted, dtype=np.float64) + move, np.linalg.norm(move, axis=-1)


@dataclass
class CharacterAnimator:
    """
    Plays one role's track on one body.

    :param skeleton: The spawned body's rest skeleton. A track with translated joints replaces their offsets with
        the track's median ones (:attr:`skeleton` is then that skeleton), so the IK and the forward kinematics
        work on the bones the track plays; the per-frame translations go to the ``Pose`` prim, and their largest
        departure from the median is :attr:`translation_spread_m`.
    :param track: The track's metadata.
    :param arrays: The track's arrays.
    :param bounds: Adaptation bounds.
    """

    skeleton: SkeletonRest
    track: MotionTrack
    arrays: TrackArrays
    bounds: AdaptationBounds = field(default_factory=AdaptationBounds)
    active: ActiveAdaptation | None = None
    fading: tuple[ActiveAdaptation, float, float] | None = None
    """An adaptation for another stroke that a new plan replaced while it still had weight, with when and at what
    weight it was replaced: it fades out under the new plan (:meth:`_fading_weight`) instead of vanishing in one frame."""
    held_frames: int = 0
    references: list[ReferenceCorrection] = field(default_factory=list)
    rejected_references: list[tuple[int, float, float, float]] = field(default_factory=list)
    disagreeing_references: list[dict[str, Any]] = field(default_factory=list)
    ik_state: tuple[float, Array] | None = None
    """The last rendered pose's ``(time, per-joint rotation of the IK away from the track)``: the rate limit's history."""
    twist_state: tuple[float, float, float] | None = None
    """The last rendered pose's ``(time, IK twist demand, twist added to the forearm)``: the history that keeps the
    added twist continuous."""
    last_twist: tuple[float, float] = (0.0, 0.0)
    conditioning: dict[str, Any] = field(default_factory=dict)
    """What :meth:`condition` did (empty until it ran): how far it turned and moved the paddle, the joint-limit
    violations before and after."""
    speed_state: tuple[float, Array, Array] | None = None
    """The last rendered pose's ``(time, both arms' local rotations, their angular velocities)``: the runtime clamp's
    history."""
    speed_clamped: int = 0
    runtime_clearance: bool = False
    """Whether the runtime arm IK of the pose being evaluated keeps the forearm out of the torso: only farther than
    :data:`RUNTIME_CLEARANCE_CONTACT_S` from every contact the animator adapts to, so a contact's blade is never turned
    for it."""

    def __post_init__(self) -> None:
        spec = self.track.character
        self.placement = BodyPlacement(scale=spec.body_scale, origin=(0.0, 0.0, FLOOR_Z_M))
        self.driven = np.array([self.skeleton.index(name) for name in self.track.joints], dtype=np.int64)
        self.root_index = self.skeleton.index(ROOT_JOINT)
        hand = spec.grip.hand_joint
        prefix = hand.split("_", 1)[0]
        self.arm = (f"{prefix}_Upperarm", f"{prefix}_Forearm", hand)
        self.hand_chain = self.skeleton.chain(hand)
        self.grip = grip_matrix(spec)
        self.blade_offset = spec.body_scale * np.asarray(spec.grip.blade_centre_m, dtype=np.float64)
        self.local_rotation = np.asarray(self.arrays.local_rotation, dtype=np.float64)
        self.root_position = np.asarray(self.arrays.root_position, dtype=np.float64)
        self.root_rotation = rot.continuous(np.asarray(self.arrays.root_rotation, dtype=np.float64))
        self.translated = np.array([self.skeleton.index(name) for name in self.track.translated_joints], dtype=np.int64)
        self.local_translation = None if self.arrays.local_translation is None else np.asarray(self.arrays.local_translation, dtype=np.float64)
        self.arm_model: ArmModel | None = None
        self.shakehand: ShakehandGrip | None = None
        self.translation_spread_m = 0.0
        if self.local_translation is not None and len(self.translated):
            median = np.median(self.local_translation, axis=0)
            self.translation_spread_m = float(np.max(np.linalg.norm(self.local_translation - median[None], axis=-1)))
            self.skeleton = self.skeleton.with_offsets(self.track.translated_joints, median)
            self.hand_chain = self.skeleton.chain(hand)

    # ---- track sampling ----------------------------------------------------

    def sample(self, motion_time: float, *, count: bool = False) -> tuple[Array, Array]:
        """
        The recorded pose at a motion time: full local rotations and the root translation.

        Times outside the track hold its first or last frame (counted in :attr:`held_frames` when ``count``).
        """

        track = self.track
        position = (motion_time - track.start_s) * track.fps
        if count and (position < 0.0 or position > track.frame_count - 1):
            self.held_frames += 1
        position = min(max(position, 0.0), track.frame_count - 1.0)
        index = min(math.floor(position), track.frame_count - 2)
        u = position - index
        local = np.array(self.skeleton.rest_rotation, dtype=np.float64)
        local[self.driven] = rot.slerp(self.local_rotation[index], self.local_rotation[index + 1], np.full(len(self.driven), u))
        local[self.root_index] = rot.slerp(self.root_rotation[index], self.root_rotation[index + 1], u)
        root_world = (1.0 - u) * self.root_position[index] + u * self.root_position[index + 1]
        return local, self.placement.to_skeleton(root_world)

    def _paddle_from(self, local: Array, root: Array) -> tuple[Array, Array, Array, Array]:
        rotations, positions = chain_kinematics(self.skeleton, self.hand_chain, local, root, self.placement)
        hand_rotation = rot.to_matrix(rotations[-1])
        hand_position = positions[-1]
        return hand_position + hand_rotation @ self.blade_offset, hand_rotation @ self.grip, hand_position, hand_rotation

    def set_references(self, strokes: list[StrokeTarget]) -> list[ReferenceCorrection]:
        """
        Correct the recorded paddle onto each stroke's reconstructed contact state (strokes without one keep the capture).

        :param strokes: This role's strokes.
        :return: The corrections, also kept for :meth:`reference_paddle` and :meth:`pose_at`.
        """

        if self.shakehand is None:
            contacts = [stroke.paddle.time_s if stroke.paddle is not None else stroke.time_s for stroke in strokes if stroke.hitter == self.track.role]
            self.condition(contacts)
        corrections: list[ReferenceCorrection] = []
        self.rejected_references = []
        self.disagreeing_references = []
        for stroke in strokes:
            target = stroke.paddle
            if target is None or stroke.hitter != self.track.role:
                continue
            recorded = self.nominal_paddle(target.time_s)
            _blade, _blade_rotation, hand, _hand_rotation = self._paddle_from(*self.sample(target.time_s))
            n_target = np.asarray(target.normal, dtype=np.float64)
            n_recorded = recorded.rotation[:, 2]
            face = n_recorded if float(n_recorded @ n_target) >= 0.0 else -n_recorded
            axis = np.cross(face, n_target)
            angle = math.atan2(float(np.linalg.norm(axis)), float(face @ n_target))
            minimal = rot.to_matrix(rot.from_rotvec(axis / np.linalg.norm(axis) * angle if np.linalg.norm(axis) > 1e-12 else np.zeros(3)))
            turned = minimal @ recorded.rotation
            shift = self._reference_shift(target, recorded.position, turned, n_target)
            # The ball fixes the face normal, not the blade's turn about it: take the turn that keeps the hand near the
            # recorded hand without a large roll, so the arm IK can reach the reference.
            turned = self._turn_toward_hand(turned, n_target, recorded.position + shift, hand, recorded.rotation) @ turned
            shift = self._reference_shift(target, recorded.position, turned, n_target)
            # Only as much of the correction as the arm holds through its window: the nearest offset held at samples
            # around contact; what it cannot reach is reported, never shown as a paddle leaving the hand.
            rotvec = rot.to_rotvec(rot.from_matrix(turned @ recorded.rotation.T))
            profile = reference_profile(float(np.linalg.norm(shift)), math.degrees(float(np.linalg.norm(rotvec))))
            samples = []
            for lag in reach_lags(profile):
                weight = adaptation_weight(target.time_s + lag, target.time_s, profile)
                if weight <= 0.0:
                    continue
                local, root = self.sample(target.time_s + lag)
                position, rotation, _hand, _hand_rotation = self._paddle_from(local, root)
                samples.append((local, root, position, rotation, weight))
            # Whether the capture and the reconstruction agree is a question about the evidence, asked of the whole
            # correction the reconstruction asks for, before the arm's reach is taken into account.
            required_shift, required_turn_deg = shift, math.degrees(float(np.linalg.norm(rotvec)))
            _worst, held_shift, held_rotvec = self._hold_offset(samples, shift, rotvec)
            unreachable = float(np.linalg.norm(shift - held_shift))
            shift = held_shift
            if held_rotvec is not None:
                # the arm cannot hold the reconstructed face within the joints' ranges: the reference is the face it holds
                rotvec = held_rotvec
            turn_deg = math.degrees(float(np.linalg.norm(rotvec)))
            normal_deg = math.degrees(angle)
            position_sd = math.hypot(target.contact_point_sd_m or 0.0, REGISTRATION_SD_M.get(self.track.role, 0.10))
            normal_sd = math.hypot(target.normal_sd_deg or 0.0, REGISTRATION_SD_DEG)
            d2 = (float(np.linalg.norm(required_shift)) / position_sd) ** 2 + (normal_deg / normal_sd) ** 2
            if d2 > REFERENCE_DISAGREE_D2:
                self.disagreeing_references.append(
                    {
                        "stroke": stroke.index,
                        "time_s": round(target.time_s, 3),
                        "offset_cm": round(100 * float(np.linalg.norm(required_shift)), 1),
                        "held_offset_cm": round(100 * float(np.linalg.norm(shift)), 1),
                        "normal_change_deg": round(normal_deg, 1),
                        "turn_deg": round(required_turn_deg, 1),
                        "position_sd_cm": round(100 * position_sd, 1),
                        "normal_sd_deg": round(normal_sd, 1),
                        "d": round(math.sqrt(d2), 2),
                        "rejected": d2 > REFERENCE_REJECT_D2,
                    }
                )
            if d2 > REFERENCE_REJECT_D2:
                # Beyond the evidence: the capture and the reconstruction disagree; the stroke keeps the capture.
                self.rejected_references.append((stroke.index, float(np.linalg.norm(shift)), turn_deg, math.sqrt(d2)))
                continue
            corrections.append(
                ReferenceCorrection(
                    stroke=stroke.index,
                    contact_s=target.time_s,
                    offset_m=shift,
                    rotation=rotvec,
                    unreachable_m=unreachable,
                    profile=reference_profile(float(np.linalg.norm(shift)), turn_deg),
                    recorded_speed_mps=float(np.linalg.norm(recorded.velocity)),
                    reconstructed_speed_mps=float(np.linalg.norm(target.velocity_mps)),
                    velocity_delta=np.asarray(target.velocity_mps, dtype=np.float64) - recorded.velocity,
                )
            )
        self.references = corrections
        return corrections

    def condition(self, contact_times: list[float] | tuple[float, ...] = ()) -> dict[str, Any]:
        """
        Hold the track's paddle, every frame, through the shakehand grip with the limit-aware arm (module docstring).

        The paddle arm, the chest (where it leans for reach) and the fingers of the track change; the paddle the hand
        then holds is the track's, spun about its face normal and, where the joints force it, turned and moved
        (reported in :attr:`conditioning`). The grip becomes the shakehand grip.

        :param contact_times: This role's contact times (motion time), where the face normal is held tightly.
        """

        from lib.replay.players.anatomy import ACTORCORE, anatomical_angles
        from lib.replay.players.kinematics import rest_pose
        from lib.replay.players.limits import limit_violations, load_joint_limits, repair_glitches

        spec = self.track.character
        side = spec.grip.hand_joint.split("_", 1)[0]
        model = ArmModel.for_skeleton(self.skeleton, spec.body, side)
        grip = shakehand_grip(spec.body, side)
        track = self.track
        count = track.frame_count
        local = np.broadcast_to(self.skeleton.rest_rotation, (count, *self.skeleton.rest_rotation.shape)).copy()
        local[:, self.driven] = self.local_rotation
        local[:, self.root_index] = self.root_rotation
        root = self.placement.to_skeleton(self.root_position)
        world, positions = forward_kinematics(self.skeleton, local, root, self.placement)
        hand = self.skeleton.index(spec.grip.hand_joint)
        hand_rotation = rot.to_matrix(world[:, hand])
        paddle_position = positions[:, hand] + np.einsum("fij,j->fi", hand_rotation, self.blade_offset)
        paddle_rotation = hand_rotation @ self.grip
        # The rubbers' labels: the recorded grip ("flipped") names the face on the back of the hand the forehand; the
        # shakehand grip's forehand rubber faces the palm. The same blade with its labels swapped (turned 180 degrees
        # about its tip axis: physics treats both rubbers alike) is held as the capture's hand held it, palm toward the
        # net on forehands (81% of right-side contacts), instead of the whole hand turned over about the handle.
        relabelled = forehand_on_back(self.grip[:, 2], self.skeleton, side)
        if relabelled:
            paddle_rotation = paddle_rotation @ RELABEL
        times = track.start_s + np.arange(count) / track.fps
        contacts = np.asarray(sorted(contact_times), dtype=np.float64)
        gap = np.full(count, np.inf) if not len(contacts) else np.min(np.abs(times[:, None] - contacts[None, :]), axis=1)
        near = gap <= CONTACT_WINDOW_S
        # the frames bracketing a contact are pinned (held hardest, never smoothed); the rest of the window weighs more
        # than the free motion but may be smoothed into it
        at = gap <= max(1.0 / track.fps, CONTACT_PIN_S)
        weight = np.where(at, CONTACT_PROJECTION_WEIGHT, np.where(near, NEAR_PROJECTION_WEIGHT, FREE_PROJECTION_WEIGHT))
        limits = load_joint_limits()
        # The clavicles first: inside their range (the retarget copies the suit's clavicle, whose bind differs from
        # the rig's) and no faster than their speed limit (its clavicle assist moved them faster than any human source:
        # capture 217 deg/s, the rig's clips 137); the arms then hold the hands from where the shoulders are.
        prepared, clavicle_frames = clamp_clavicles(self.skeleton, local, root, self.placement, limits)
        for clavicle in ("R_Clavicle", "L_Clavicle"):
            prepared = limit_joint_speed(prepared, self.skeleton.index(clavicle), limits.speed_limit_deg_s(clavicle), track.fps)
        # The elbow stays near the capture's own (``posed``), and the forearm out of the torso (the arm's clearance
        # cost). The lowest elbow on its circle (``low``) folds the forearm into the belly between strokes: an arm more
        # than 1 cm into the torso in 60% / 38% of sampled frames (suited / opponent, the skinned-mesh measure; the
        # capture 0.1% / 0.0%), 16% / 22% with the posed elbow alone and 1.8% / 1.4% with the clearance kept from the
        # contacts.
        held = hold_paddle(
            model,
            grip,
            prepared,
            root,
            self.placement,
            paddle_position,
            paddle_rotation,
            projection_weight=weight,
            elbow_ref="posed",
            fps=track.fps,
            reposition=True,
            # the clearance acts where the runtime's does too (away from the contacts), so the runtime arm IK holds a
            # conditioned pose exactly when it engages near a contact
            clearance_weight=np.where(gap > RUNTIME_CLEARANCE_CONTACT_S, CLEARANCE_WEIGHT, 0.0),
            clearance_relative=False,
        )
        conditioned = grip.apply_fingers(self.skeleton, held.arm.local_rotation)
        # The free arm holds its own hand where the track has it, inside the joints' ranges (its elbow on the rig's
        # hinge), clear of the paddle and the playing arm and no higher than its shoulder (free_hand_target); its
        # fingers relax.
        free_side = "L" if side == "R" else "R"
        free_hand = self.skeleton.index(f"{free_side}_Hand")
        free_target, free_moved = self._free_hand_target(conditioned, root, positions[:, free_hand], free_side, side, track.fps)
        free = solve_arm(
            ArmModel.for_skeleton(self.skeleton, spec.body, free_side),
            conditioned,
            root,
            self.placement,
            free_target,
            rot.to_matrix(world[:, free_hand]),
            fps=track.fps,
            reposition=True,
            clearance_relative=False,
        )
        conditioned = apply_relaxed_fingers(self.skeleton, free.local_rotation, spec.body, free_side)
        # Whatever the solve left too abrupt (a clamp moving to a range's other end, a reach limit), smoothed away; the
        # frames bracketing each contact are pinned (their paddle is the ball's evidence).
        repaired_joints = [
            self.skeleton.index(name)
            for name in (
                "Waist",
                "Spine01",
                "Spine02",
                *(f"{s_}_{part}" for s_ in ("R", "L") for part in ("Clavicle", "Upperarm", "Forearm", "ForearmTwist01", "ForearmTwist02", "Hand")),
            )
            if name in self.skeleton.joints
        ]
        repaired, repaired_frames = repair_glitches(
            conditioned[:, repaired_joints], track.fps, [self.skeleton.joints[i] for i in repaired_joints], pinned=at, limits=limits
        )
        conditioned = conditioned.copy()
        conditioned[:, repaired_joints] = repaired
        rest_local, rest_root = rest_pose(self.skeleton)
        rest_world, _ = forward_kinematics(self.skeleton, rest_local, rest_root, self.placement)
        hinge = limits.hinge(spec.body)

        def violations(poses: Array) -> dict[str, Any]:
            w, _ = forward_kinematics(self.skeleton, poses, root, self.placement)
            v = limit_violations(anatomical_angles(w, rest_world, self.skeleton.joints, ACTORCORE, hinge_azimuth_deg=hinge), limits=limits)
            return {"frames_any_gated": v.frames_any_gated, "share_any_gated": round(v.frames_any_gated / max(v.frames, 1), 4), "worst": v.worst(4)}

        self.conditioning = {
            "grip": "shakehand",
            "rubbers_relabelled": relabelled,
            "frames": count,
            "contact_frames": int(near.sum()),
            "tilt_at_contacts_deg_p50_p95_max": [round(float(np.percentile(held.tilt_deg[near], q)), 2) for q in (50, 95, 100)] if near.any() else None,
            "tilt_elsewhere_deg_p50_p95_max": [round(float(np.percentile(held.tilt_deg[~near], q)), 2) for q in (50, 95, 100)],
            "blade_moved_at_contacts_cm_p95_max": [round(100 * float(np.percentile(held.moved_m[near], q)), 2) for q in (95, 100)] if near.any() else None,
            "spin_deg_p50": round(float(np.median(held.spin_deg)), 1),
            "glitch_repaired_frames": repaired_frames,
            "clavicle_clamped_frames": clavicle_frames,
            "free_hand_moved_cm_p95_max": [round(100 * float(np.percentile(free_moved, q)), 2) for q in (95, 100)],
            "violations_before": violations(local),
            "violations_after": violations(conditioned),
        }
        # The joints conditioning writes (both arms, their twist joints, the clavicles, the chest, the paddle hand's
        # fingers) are driven from now on, whether or not the track drove them.
        written = [
            self.skeleton.index(name)
            for name in (
                *(f"{s_}_{part}" for s_ in ("R", "L") for part in ("Clavicle", "Upperarm", "Forearm", "Hand", "ForearmTwist01", "ForearmTwist02")),
                "Waist",
                "Spine01",
                "Spine02",
                *grip.finger_local,
                *relaxed_fingers(spec.body, free_side),
            )
            if name in self.skeleton.joints
        ]
        driven = list(self.driven) + [index for index in written if index not in set(int(k) for k in self.driven)]
        self.driven = np.asarray(driven, dtype=np.int64)
        self.local_rotation = conditioned[:, self.driven]
        self.arm_model = model
        self.shakehand = grip
        self.grip = grip.matrix
        self.blade_offset = spec.body_scale * np.asarray(grip.blade_centre_m, dtype=np.float64)
        return self.conditioning

    def _free_hand_target(self, local: Array, root: Array, wanted: Array, free_side: str, paddle_side: str, fps: float) -> tuple[Array, Array]:
        """:func:`free_hand_target` for this animator's body and grip."""

        return free_hand_target(self.skeleton, self.placement, self.blade_offset, local, root, wanted, free_side, paddle_side, fps)

    @staticmethod
    def _reference_shift(target: Any, recorded_centre: Array, turned: Array, n_target: Array) -> Array:
        """The blade-centre shift onto a reconstructed contact state, for the blade's axes ``turned``."""

        # The face is moved only as far as the contact needs: through the reconstructed contact point along the normal,
        # in its plane only when the point would fall off the face. The reconstruction's blade centre assumes a centre
        # hit, so moving the blade there would add up to a blade radius of correction the ball does not ask for.
        relative = np.asarray(target.contact_point_m, dtype=np.float64) - recorded_centre
        normal_shift = float(relative @ n_target) - FACE_HALF_THICKNESS_M
        in_plane = relative - float(relative @ n_target) * n_target
        u, v = float(in_plane @ turned[:, 0]) / FACE_HALF_WIDTH_M, float(in_plane @ turned[:, 1]) / FACE_HALF_LENGTH_M
        reach = math.hypot(u, v)
        excess = in_plane * (1.0 - CONTACT_INSIDE_SHARE / reach) if reach > CONTACT_INSIDE_SHARE else np.zeros(3)
        return normal_shift * n_target + excess

    def _turn_toward_hand(self, blade_rotation: Array, normal: Array, centre: Array, hand: Array, recorded_rotation: Array | None = None) -> Array:
        """
        The rotation about the face normal that keeps the hand (for a blade at ``centre``) near ``hand`` without spinning the blade.

        The ball fixes the normal only. Turning the blade about it swings the hand around the blade's lever; the turn
        minimises the hand's displacement plus :data:`TURN_COST_M_PER_RAD` per radian of the blade's total turn from
        the recording, so the arm can reach the reference without a large, unnatural roll.
        """

        lever = blade_rotation @ self.grip.T @ self.blade_offset
        best: tuple[float, Array] | None = None
        for degrees in range(-180, 180, 5):
            turn = rot.to_matrix(rot.from_axis_angle(normal, math.radians(degrees)))
            displacement = float(np.linalg.norm(centre - turn @ lever - hand))
            total = turn @ blade_rotation @ (recorded_rotation.T if recorded_rotation is not None else blade_rotation.T)
            angle = math.acos(min(1.0, max(-1.0, (float(np.trace(total)) - 1.0) / 2.0)))
            cost = displacement + TURN_COST_M_PER_RAD * angle
            if best is None or cost < best[0]:
                best = (cost, turn)
        assert best is not None
        return best[1]

    def _correction(self, motion_time: float) -> tuple[Array, Array, float]:
        """The blended reference correction at a motion time: ``(offset, rotation vector, weight)``."""

        offset, rotation, weight = np.zeros(3), np.zeros(3), 0.0
        for correction in self.references:
            w = adaptation_weight(motion_time, correction.contact_s, correction.profile)
            if w > weight:
                offset, rotation, weight = w * correction.offset_at(motion_time), w * correction.rotation, w
        return offset, rotation, weight

    def _corrected(self, motion_time: float, local: Array, root: Array) -> tuple[Array, Array]:
        position, rotation, _hand, _hand_rotation = self._paddle_from(local, root)
        offset, turn, weight = self._correction(motion_time)
        if weight > 0.0:
            rotation = rot.to_matrix(rot.from_rotvec(turn)) @ rotation
            position = position + offset
        return position, rotation

    def reference_paddle(self, motion_time: float) -> PaddleState:
        """The recorded paddle corrected onto the reconstructed contact states: what the stroke controller adapts from."""

        if not self.references:
            return self.nominal_paddle(motion_time)
        position, rotation = self._corrected(motion_time, *self.sample(motion_time))
        before = self._corrected(motion_time - VELOCITY_PROBE_S, *self.sample(motion_time - VELOCITY_PROBE_S))
        after = self._corrected(motion_time + VELOCITY_PROBE_S, *self.sample(motion_time + VELOCITY_PROBE_S))
        velocity = (after[0] - before[0]) / (2.0 * VELOCITY_PROBE_S)
        delta = rot.multiply(rot.from_matrix(after[1]), rot.conjugate(rot.from_matrix(before[1])))
        angular = rot.to_rotvec(delta) / (2.0 * VELOCITY_PROBE_S)
        return PaddleState(position=position, rotation=rotation, velocity=velocity, angular_velocity=angular)

    def nominal_paddle(self, motion_time: float) -> PaddleState:
        """The recorded (unadapted) paddle at a motion time, with its twist from a central difference."""

        local, root = self.sample(motion_time)
        position, rotation, _hand, _hand_rotation = self._paddle_from(local, root)
        before = self._paddle_from(*self.sample(motion_time - VELOCITY_PROBE_S))
        after = self._paddle_from(*self.sample(motion_time + VELOCITY_PROBE_S))
        velocity = (after[0] - before[0]) / (2.0 * VELOCITY_PROBE_S)
        delta = rot.multiply(rot.from_matrix(after[1]), rot.conjugate(rot.from_matrix(before[1])))
        angular = rot.to_rotvec(delta) / (2.0 * VELOCITY_PROBE_S)
        return PaddleState(position=position, rotation=rotation, velocity=velocity, angular_velocity=angular)

    # ---- adaptation ---------------------------------------------------------

    def adopt(self, plan: StrokePlan, now: float) -> None:
        """
        Start applying a plan: cross-fading from the current one if it is for the same stroke; a current one for
        another stroke that still has weight fades out under the new plan (:attr:`fading`). Dropping it would make the
        plan's weight fall from as much as 0.87 to 0 in one frame, which the arm then follows at the IK rate limit.
        """

        if self.active is not None and self.active.plan.stroke == plan.stroke:
            self.active = ActiveAdaptation(plan=plan, previous=self.active.plan, replaced_at=now)
        else:
            if self.active is not None:
                left = adaptation_weight(now, self.active.parameters(now)[0], self.active.plan.bounds)
                if left > 0.0 and left >= self._fading_weight(now):
                    self.fading = (self.active, now, left)
            self.active = ActiveAdaptation(plan=plan)

    def _fading_weight(self, t: float) -> float:
        """
        The replaced adaptation's weight at ``t``: its own release once past its contact (never above the weight it
        had when replaced), else a smooth fade from that weight over its release time; 0 without one.
        """

        if self.fading is None:
            return 0.0
        adaptation, replaced_at, weight = self.fading
        contact = adaptation.parameters(t)[0]
        if t >= contact:
            return min(weight, adaptation_weight(t, contact, adaptation.plan.bounds))
        return float(weight * (1.0 - rot.smoothstep((t - replaced_at) / max(adaptation.plan.bounds.release_s, 1e-6))))

    def release_finished(self, now: float) -> None:
        """Drop an adaptation whose fade-out has ended."""

        if self.active is not None:
            bounds = self.active.plan.bounds
            if now > self.active.plan.contact_s + bounds.hold_s + bounds.follow_through_s + bounds.release_s:
                self.active = None

    def pose_at(self, t: float, *, count: bool = True) -> AnimatedPose:
        """
        The pose to show at session time ``t`` (the motion time equals the session time, warped near strokes).

        Both arms then turn no joint faster than :data:`RUNTIME_SPEED_SHARE` of its glitch
        ceiling since the last rendered pose (:meth:`_clamp_speed`): an adaptation on top of a fast recorded stroke
        never makes a forearm spin faster than any human's; what the slowed arm misses of the planned paddle is an IK
        miss, and the paddle stays in the hand.

        :return: The pose and the paddle it carries.
        """

        pose = self._pose_unclamped(t, count=count)
        clamped = self._clamp_speed(t, pose.local_rotation, count=count) if self.shakehand is not None else None
        if clamped is not None:
            held, held_rotation, _hand, _hand_rotation = self._paddle_from(clamped, pose.root_translation)
            moved = float(np.linalg.norm(held - pose.paddle_position))
            pose = dataclasses.replace(
                pose,
                local_rotation=clamped,
                paddle_position=held,
                paddle_rotation=held_rotation,
                ik_residual_m=pose.ik_residual_m + moved,
                ik_active=pose.ik_active or moved > 1e-9,
                held_paddle_position=held,
            )
        in_hand, _rotation, _hand, _hand_rotation = self._paddle_from(pose.local_rotation, pose.root_translation)
        return dataclasses.replace(pose, hand_gap_m=float(np.linalg.norm(np.asarray(pose.paddle_position) - in_hand)))

    def _clamp_speed(self, t: float, local: Array, *, count: bool) -> Array | None:
        """
        The paddle arm's joints held, since the last rendered pose, to :data:`RUNTIME_SPEED_SHARE` of their glitch
        ceilings in speed and in change of velocity (the glitch rule's two tests: the ceiling reached from rest in no
        less than 1/24 s), or ``None`` when no joint needed it (the history is kept when ``count``).
        """

        assert self.arm_model is not None
        joints = self._clamp_joints()
        current = local[joints]
        state = self.speed_state
        dt = 0.0 if state is None else t - state[0]
        if state is None or not 1e-6 < dt <= RUNTIME_SPEED_GAP_S:
            if count:
                self.speed_state = (t, current.copy(), np.zeros((len(joints), 3)))
            return None
        previous, velocity = state[1], state[2]
        names = [self.skeleton.joints[int(j)] for j in joints]
        ceiling = np.radians(RUNTIME_SPEED_SHARE * self.arm_model.limits.glitch_speeds(names))
        wanted = rot.to_rotvec(rot.multiply(rot.conjugate(previous), current))  # this step, body frame
        expected = velocity * dt
        change = wanted - expected
        most_change = ceiling * REFERENCE_FPS_GLITCH * dt * dt
        size = np.linalg.norm(change, axis=-1)
        step = np.where((size > most_change)[:, None], expected + change * (most_change / np.maximum(size, 1e-12))[:, None], wanted)
        length = np.linalg.norm(step, axis=-1)
        step = np.where((length > ceiling * dt)[:, None], step * (ceiling * dt / np.maximum(length, 1e-12))[:, None], step)
        limited_any = bool(np.any(np.linalg.norm(step - wanted, axis=-1) > 1e-9))
        limited = rot.normalize(rot.multiply(previous, rot.from_rotvec(step)))
        if count:
            self.speed_state = (t, limited.copy() if limited_any else current.copy(), step / dt)
            self.speed_clamped += int(limited_any)
        if not limited_any:
            return None
        out = np.array(local, copy=True)
        out[joints] = limited
        return out

    def _clamp_joints(self) -> Array:
        """Both arms' joints: the paddle arm's, and the free arm's, which the time warp near strokes can otherwise
        speed past a ceiling it was conditioned to (a clavicle at 228 against its 227 deg/s)."""

        names = [f"{side}_{part}" for side in ("R", "L") for part in ("Clavicle", "Upperarm", "Forearm", "ForearmTwist01", "ForearmTwist02", "Hand")]
        return np.array([self.skeleton.index(name) for name in names if name in self.skeleton.joints], dtype=np.int64)

    def _pose_unclamped(self, t: float, *, count: bool = True) -> AnimatedPose:
        """:meth:`pose_at` before the runtime speed clamp."""

        weight = 0.0
        shift = 0.0
        offset = np.zeros(3)
        tilt = np.zeros(3)
        if self.active is not None:
            contact, offset, tilt, shift = self.active.parameters(t)
            weight = adaptation_weight(t, contact, self.active.plan.bounds)
        fading = self._fading_weight(t)
        if fading > 0.0 and self.fading is not None:
            # A replaced plan's share fades out under the new one's: the offset, the turn and the time shift stay
            # continuous.
            _contact, old_offset, old_tilt, old_shift = self.fading[0].parameters(t)
            total = weight + fading
            offset = (weight * np.asarray(offset, dtype=np.float64) + fading * np.asarray(old_offset, dtype=np.float64)) / total
            tilt = (weight * np.asarray(tilt, dtype=np.float64) + fading * np.asarray(old_tilt, dtype=np.float64)) / total
            shift = (weight * shift + fading * old_shift) / total
            weight = min(total, 1.0)
        elif self.fading is not None and count:
            self.fading = None
        motion_time = t - weight * shift
        local, root = self.sample(motion_time, count=count)
        paddle_position, paddle_rotation, _hand, _hand_rotation = self._paddle_from(local, root)
        correction_offset, correction_turn, correction_weight = self._correction(motion_time) if self.references else (np.zeros(3), np.zeros(3), 0.0)
        adapting = weight > 0.0 and (np.any(offset != 0.0) or np.any(tilt != 0.0))
        residual = 0.0
        if adapting or correction_weight > 0.0:
            reference_rotation = rot.to_matrix(rot.from_rotvec(correction_turn)) @ paddle_rotation
            reference_position = paddle_position + correction_offset
            turn = rot.to_matrix(rot.from_rotvec(weight * tilt))
            target_rotation = turn @ reference_rotation
            target_position = reference_position + weight * offset
            history = None if self.twist_state is None else (t - self.twist_state[0], self.twist_state[1], self.twist_state[2])
            track_local = local
            contacts = [self.active.plan.contact_s] if self.active is not None else []
            contacts += [reference.contact_s for reference in self.references]
            self.runtime_clearance = not contacts or min(abs(motion_time - c) for c in contacts) > RUNTIME_CLEARANCE_CONTACT_S
            local, held, held_rotation, shortfall = self._reach(
                local, root, paddle_position, paddle_rotation, target_position, target_rotation, history=history
            )
            if count:
                self.twist_state = (t, *self.last_twist)
            limited = self._limit_ik(t, track_local, local, count=count)
            if limited is not None:
                local = limited
                held, held_rotation, _hand, _hand_rotation = self._paddle_from(local, root)
                shortfall = float(np.linalg.norm(target_position - held))
            # The paddle never leaves the hand: the arm reaches as far toward the planned paddle as it can, the paddle
            # is where the hand holds it, and what the arm could not reach is a motion-QA finding (``ik_residual_m``).
            return AnimatedPose(
                time_s=t,
                motion_time_s=motion_time,
                local_rotation=local,
                root_translation=root,
                paddle_position=held,
                paddle_rotation=held_rotation,
                weight=weight,
                ik_residual_m=shortfall,
                local_translation=self.sample_translation(motion_time),
                ik_active=True,
                held_paddle_position=held,
            )
        if count:
            self.twist_state = (t, 0.0, 0.0)
        limited = self._limit_ik(t, local, local, count=count)
        if limited is not None:
            # The arm is still easing back from an adaptation onto the track.
            held, held_rotation, _hand, _hand_rotation = self._paddle_from(limited, root)
            return AnimatedPose(
                time_s=t,
                motion_time_s=motion_time,
                local_rotation=limited,
                root_translation=root,
                paddle_position=held,
                paddle_rotation=held_rotation,
                weight=weight,
                ik_residual_m=float(np.linalg.norm(paddle_position - held)),
                local_translation=self.sample_translation(motion_time),
                ik_active=True,
                held_paddle_position=held,
            )
        return AnimatedPose(
            time_s=t,
            motion_time_s=motion_time,
            local_rotation=local,
            root_translation=root,
            paddle_position=paddle_position,
            paddle_rotation=paddle_rotation,
            weight=weight,
            ik_residual_m=residual,
            local_translation=self.sample_translation(motion_time),
        )

    def _limit_ik(self, t: float, track: Array, solved: Array, *, count: bool) -> Array | None:
        """
        The arm IK's change to the track, rate-limited: each joint's rotation away from the track (``solved`` against
        ``track``) moves at most :data:`IK_DELTA_RATE_RAD_S` from the previous rendered frame's (a large but
        legitimate correction chased in one frame reads as 50-80 degree pops of the forearm and hand). An acceleration
        limit on this change as well was rejected: at 14,400-24,000 deg/s^2 the IK miss rose to 2.2-7.7 cm at p95 while
        the pops barely fell. ``None`` when there is nothing to limit (no IK now and none left from before).
        """

        delta = rot.multiply(solved, rot.conjugate(track))
        if self.ik_state is None:
            if not np.any(rot.angle(delta) > 1e-9):
                return None
            # The IK starts from the track: the previous rendered frame had no change to it.
            previous_time, previous = t - 1.0 / 24.0, rot.identity(delta.shape[:-1])
        else:
            previous_time, previous = self.ik_state
        step = rot.angle(rot.multiply(delta, rot.conjugate(previous)))
        allowed = IK_DELTA_RATE_RAD_S * max(abs(t - previous_time), 1.0 / 240.0)
        fraction = np.clip(allowed / np.maximum(step, 1e-12), 0.0, 1.0)
        limited_delta = rot.slerp(previous, delta, fraction)
        if count:
            settled = not np.any(rot.angle(limited_delta) > 1e-6)
            self.ik_state = None if settled else (t, limited_delta)
        return rot.multiply(limited_delta, track)

    def _solve(
        self, local: Array, root: Array, target_position: Array, target_rotation: Array, *, history: tuple[float, float, float] | None = None
    ) -> tuple[Array, float, tuple[float, float]]:
        """
        Arm IK toward a blade pose, on a copy of ``local``: the solved pose, the hand's residual (m) and the twist added
        to the forearm (rad).

        :param history: ``(seconds since the previous rendered pose, its twist demand, its added twist)``: keeps the
            added twist on its continuous branch and rate-limited (:func:`~lib.replay.players.kinematics.soft_twist`).
        """

        if self.shakehand is not None and self.arm_model is not None:
            # The limit-aware arm holds the blade through the shakehand grip (or the nearest blade the joints allow);
            # it writes the pronation on the forearm and its shares on the twist joints itself. At runtime the elbow
            # also stays near where the previous rendered arm puts it (its IK change carried onto this pose): the
            # per-tick solve never jumps between two equally good swivels, which the IK rate limit would then have to
            # chase (otherwise most of a session's misses over 2 cm come from it). Near a straight elbow, where
            # every swivel puts the elbow in the same place, the previous upper arm's roll is held too.
            hint = None
            humerus_hint = None
            if history is not None and self.ik_state is not None:
                carried = rot.multiply(self.ik_state[1], local)
                carried_world, carried_positions = chain_kinematics(self.skeleton, self.hand_chain, carried, root, self.placement)
                chain = list(self.hand_chain)
                hint = carried_positions[chain.index(self.arm_model.forearm)]
                humerus_hint = rot.to_matrix(carried_world[chain.index(self.arm_model.upperarm)])
            held = hold_paddle(
                self.arm_model,
                self.shakehand,
                local,
                root,
                self.placement,
                target_position,
                target_rotation,
                spins_deg=(0.0,),
                elbow_window_m=ELBOW_WINDOW_M,
                smooth=False,
                trunk=False,
                comfort=False,
                elbow_hint=hint,
                humerus_hint=humerus_hint,
                clearance=self.runtime_clearance,
                clearance_relative=True,
            )
            residual = float(np.linalg.norm(held.paddle_position[0] - np.asarray(target_position, dtype=np.float64)))
            return held.arm.local_rotation[0], residual, (0.0, 0.0)
        solved = local.copy()
        hand_rotation = target_rotation @ self.grip.T
        hand_position = target_position - hand_rotation @ self.blade_offset
        hand_before = solved[self.skeleton.index(self.arm[2])].copy()
        solution = apply_two_bone(self.skeleton, solved, root, self.placement, self.arm, hand_position, rot.from_matrix(hand_rotation), softness=ARM_SOFTNESS)
        previous, rate = (None, 0.0) if history is None else ((history[1], history[2]), ARM_TWIST_RATE_RAD_S * max(abs(history[0]), 1.0 / 240.0))
        twist = transfer_added_twist(
            self.skeleton, solved, hand_before, self.arm[1], self.arm[2], previous=previous, limit_rad=ARM_TWIST_LIMIT_RAD, rate_rad=rate
        )
        return solved, float(solution.residual_m), twist

    def _reach(
        self,
        local: Array,
        root: Array,
        recorded_position: Array,
        recorded_rotation: Array,
        target_position: Array,
        target_rotation: Array,
        *,
        history: tuple[float, float, float] | None = None,
    ) -> tuple[Array, Array, Array, float]:
        """
        The arm's reach toward a blade pose at runtime: the farthest fraction of the move from the recorded blade that
        the arm holds within :data:`REACH_TOLERANCE_M`, the paddle where the hand then holds it, and the shortfall, m.
        Plans and references are already projected onto what the arm holds (:meth:`plan_reach`), so this only acts on
        what is left between their samples. The added twist of the pose chosen is kept in :attr:`last_twist`.
        """

        solved, residual, twist = self._solve(local, root, target_position, target_rotation, history=history)
        if residual <= REACH_TOLERANCE_M:
            self.last_twist = twist
            held, held_rotation, _hand, _hand_rotation = self._paddle_from(solved, root)
            return solved, held, held_rotation, 0.0
        lo, best = 0.0, local.copy()
        self.last_twist = (0.0, 0.0)
        recorded_q = rot.from_matrix(recorded_rotation)
        target_q = rot.from_matrix(target_rotation)
        hi = 1.0
        for _ in range(HOLD_BISECTIONS + 2):
            mid = 0.5 * (lo + hi)
            position = recorded_position + mid * (target_position - recorded_position)
            rotation = rot.to_matrix(rot.slerp(recorded_q, target_q, mid))
            candidate, residual, twist = self._solve(local, root, position, rotation, history=history)
            if residual <= REACH_TOLERANCE_M:
                lo, best = mid, candidate
                self.last_twist = twist
            else:
                hi = mid
        held, held_rotation, _hand, _hand_rotation = self._paddle_from(best, root)
        return best, held, held_rotation, float(np.linalg.norm(target_position - held))

    def plan_reach(self, plan: StrokePlan) -> PlanReach:
        """
        Whether the arm holds a plan's blade (offset and tilt, as :meth:`pose_at` applies them) at
        :data:`PLAN_REACH_SAMPLES_S` around the contact, each at the adaptation's weight there, and if not, the nearest
        offset it does hold throughout: each sample's held blade (the soft-IK arm extended toward the planned one) is a
        candidate, and the one leaving the smallest worst residual over all the samples wins.

        :return: The reach: the plan's largest residual over the samples; the chosen offset (the plan's own when it is
            held); the planned offset less it, m.
        """

        planned = np.asarray(plan.offset_m, dtype=np.float64)
        samples = []
        for lag in reach_lags(plan.bounds):
            t = plan.contact_s + lag
            weight = adaptation_weight(t, plan.contact_s, plan.bounds)
            if weight <= 0.0:
                continue
            motion_time = t - weight * plan.time_shift_s
            local, root = self.sample(motion_time)
            position, rotation, _hand, _hand_rotation = self._paddle_from(local, root)
            offset, turn, correction = self._correction(motion_time) if self.references else (np.zeros(3), np.zeros(3), 0.0)
            if correction > 0.0:
                rotation = rot.to_matrix(rot.from_rotvec(turn)) @ rotation
                position = position + offset
            samples.append((local, root, position, rotation, weight))
        planned_turn = np.asarray(plan.rotation, dtype=np.float64)
        worst, chosen, held_turn = self._hold_offset(samples, planned, planned_turn)
        shortfall = (
            0.0 if held_turn is None else math.degrees(float(rot.angle(rot.multiply(rot.from_rotvec(held_turn), rot.conjugate(rot.from_rotvec(planned_turn))))))
        )
        return PlanReach(worst_m=worst, held_offset_m=chosen, shortfall_m=planned - chosen, held_rotation_rad=held_turn, tilt_shortfall_deg=shortfall)

    def _hold_offset(self, samples: list[tuple[Array, Array, Array, Array, float]], offset: Array, rotvec: Array) -> tuple[float, Array, Array | None]:
        """
        The nearest blade offset the arm holds at every sample ``(local, root, base position, base rotation, weight)``,
        the blade at each being ``base + weight * offset`` turned by ``weight * rotvec``: each sample's held blade (the
        soft-IK arm extended toward the wanted one) is a candidate, and the one leaving the smallest worst residual wins.

        :return: The wanted offset's largest residual over the samples, the chosen offset (``offset`` itself when it is
            held within :data:`REACH_TOLERANCE_M`), and, when the arm cannot hold the wanted turn within
            :data:`TILT_TOLERANCE_DEG` at every sample, the largest share of it that it does (a rotation vector;
            ``None`` when it holds the whole turn).
        """

        def residuals(candidate: Array) -> tuple[float, list[Array]]:
            worst, held_offsets = 0.0, []
            for local, root, position, rotation, weight in samples:
                target_rotation = rot.to_matrix(rot.from_rotvec(weight * rotvec)) @ rotation
                solved, residual, _twist = self._solve(local, root, position + weight * candidate, target_rotation)
                held, _held_rotation, _hand, _hand_rotation = self._paddle_from(solved, root)
                worst = max(worst, residual)
                held_offsets.append((held - position) / weight)
            return worst, held_offsets

        def turn_held(candidate: Array, scale: float) -> bool:
            for local, root, position, rotation, weight in samples:
                wanted = rot.to_matrix(rot.from_rotvec(weight * scale * rotvec)) @ rotation
                solved, _residual, _twist = self._solve(local, root, position + weight * candidate, wanted)
                _held, held_rotation, _hand, _hand_rotation = self._paddle_from(solved, root)
                if math.degrees(float(rot.angle(rot.from_matrix(held_rotation @ wanted.T)))) > TILT_TOLERANCE_DEG:
                    return False
            return True

        def held_turn(candidate: Array) -> Array | None:
            # The turn the arm holds at every sample (not only the fullest): the largest share of the wanted turn
            # held within TILT_TOLERANCE_DEG throughout, so the runtime arm is never asked for a face it cannot hold.
            if not samples or not np.any(rotvec != 0.0) or turn_held(candidate, 1.0):
                return None
            lo, hi = 0.0, 1.0
            for _ in range(HOLD_BISECTIONS):
                mid = 0.5 * (lo + hi)
                if turn_held(candidate, mid):
                    lo = mid
                else:
                    hi = mid
            return lo * rotvec

        worst, held_offsets = residuals(offset)
        if worst <= REACH_TOLERANCE_M or not samples:
            return worst, offset, held_turn(offset)
        best_worst, best_index = min((residuals(candidate)[0], index) for index, candidate in enumerate(held_offsets))
        chosen = held_offsets[best_index]
        if best_worst > REACH_TOLERANCE_M:
            # Held at contact but not throughout: the largest share of it the arm holds at every sample.
            lo, hi = 0.0, 1.0
            for _ in range(HOLD_BISECTIONS):
                mid = 0.5 * (lo + hi)
                if residuals(mid * chosen)[0] <= REACH_TOLERANCE_M:
                    lo = mid
                else:
                    hi = mid
            chosen = lo * chosen
        return worst, chosen, held_turn(chosen)

    def sample_translation(self, motion_time: float) -> Array | None:
        """The translated joints' local translations at a motion time (linear between frames), or ``None``."""

        if self.local_translation is None:
            return None
        track = self.track
        position = min(max((motion_time - track.start_s) * track.fps, 0.0), track.frame_count - 1.0)
        index = min(math.floor(position), track.frame_count - 2)
        u = position - index
        return (1.0 - u) * self.local_translation[index] + u * self.local_translation[index + 1]


__all__ = [
    "FLOOR_Z_M",
    "ActiveAdaptation",
    "AnimatedPose",
    "CharacterAnimator",
    "PlanReach",
    "ReferenceCorrection",
    "adaptation_weight",
    "free_hand_target",
    "grip_matrix",
]
