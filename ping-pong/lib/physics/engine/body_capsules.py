"""
The players' bodies as moving capsules for the ball: a swept sphere-capsule contact, pure numpy.

The characters render as skinned meshes with no colliders, so without this the
ball would pass through them. Each render tick the replay loop has every joint's world position from the animated skeleton
(``forward_kinematics``); :func:`capsules_from_joints` turns them into one
capsule per body segment, and :class:`BodyMotion` moves the capsules linearly
from the previous tick's pose to this one, as the paddle segments do. Each
physics step :func:`sweep_body` finds the first touch of the ball on any
capsule over the step (the ball at constant velocity, the capsules moving),
and the stepper resolves it with the reference impact law against a moving,
infinitely heavy surface (:func:`lib.physics.engine.contacts.impose_contact`
with :data:`BODY_CONTACT` and the capsule point's velocity).

**Segments** (:data:`BODY_CAPSULES`): lower and upper torso along the spine, a
hip bar and a shoulder bar for the torso's breadth, the head, and each upper
arm, forearm, hand, thigh and shin. The radii are the ANSUR II male means
(US Army 2012 anthropometric survey, 4,082 men, public CSV from
``tools.openlab.psu.edu/publicData/ANSUR_II_MALE_Public.csv``), converted as ``r = circumference / (2 pi)`` or half a
breadth or depth and averaged along tapering segments:

=============  =====  ===========================================================
capsule        r, m   basis (ANSUR II male mean)
=============  =====  ===========================================================
torso (2)      0.13   chest depth 254 mm, waist depth 238, buttock depth 246 (half)
hips           0.09   hip breadth 346 mm, less the hip joints' half spacing (~85)
shoulders      0.075  bideltoid breadth 510 mm, less the shoulder joints' half spacing (~178)
head           0.09   head circumference 574 mm (r 0.091); breadth 154, length 200
upper arm      0.05   biceps circumference flexed 358 mm (r 0.057), relaxed less
forearm        0.04   forearm circumference flexed 310 (r 0.049) to wrist 176 (0.028)
hand           0.04   hand breadth about 9 cm; 10 cm beyond the wrist joint
thigh          0.08   thigh circumference 625 (r 0.0995) to lower thigh 409 (0.065)
shin           0.05   calf circumference 392 (r 0.062) to ankle 229 (0.036)
=============  =====  ===========================================================

The skeleton's ``Head`` joint sits at the skull base, about 0.21 m below the top
of the head (measured: joint 1.50 m above the floor on a 1.71 m body), so the head capsule runs from 0.07 to 0.12 m beyond it along the
neck. The radii are fixed (an average man; the characters are 1.71-1.78 m tall,
against the survey's mean stature of 1.756 m); ``scale`` adjusts them.

**Material** (:data:`BODY_CONTACT`). No measurement of a table-tennis ball on a
clothed body was found. Clothing over soft tissue is compliant and lossy, so the
normal restitution is judged at 0.3 (plausible range 0.2-0.4) and friction at
0.5 (fabric against ABS; judgement). The body is far heavier than the 2.7 g
ball and is driven by the animation, so it is an infinite effective mass.

**The paddle arm.** Near a player's own stroke the hand and forearm that hold
the paddle are left out (:func:`excluded_capsules`): the blade, not the
forearm, must meet the ball, and the animated hand can sit a few centimetres
off the reconstructed blade.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.physics.engine.contacts import SweptContact
from lib.physics.params import ContactMaterial

Array = NDArray[np.float64]

BODY_CONTACT: ContactMaterial = ContactMaterial(restitution=0.3, friction=0.5)
"""Ball against a clothed body segment: judgement (no measurement found); restitution plausible 0.2-0.4."""

PADDLE_ARM_BEFORE_S: float = 0.30
"""The paddle hand and forearm are left out from this long before the player's own contact..."""
PADDLE_ARM_AFTER_S: float = 0.15
"""...to this long after it."""

EMBEDDED_DEPTH_M: float = 0.005
"""A ball that starts a step deeper than this inside a capsule (a placement or a capsule that was just re-enabled) is
not pushed out: it was never outside, so there is no physical contact to resolve; it leaves on its own."""

SWEEP_BISECTIONS: int = 12
"""Bisections of a bracketed touch: the contact time is located to ``dt / 4096``."""


@dataclass(frozen=True)
class CapsuleSpec:
    """
    One body segment as a capsule between two skeleton joints.

    :ivar name: Segment name (``torso_lower``, ``R_forearm``...).
    :ivar start: Joint at one end.
    :ivar end: Joint at the other end.
    :ivar radius_m: Capsule radius, m.
    :ivar extend: ``None``: the capsule runs from ``start`` to ``end``. ``(a, b)``: it lies along the ``start`` to ``end``
        direction from ``a`` to ``b`` metres beyond ``end`` (the head beyond the skull-base joint, the hand beyond the wrist).
    """

    name: str
    start: str
    end: str
    radius_m: float
    extend: tuple[float, float] | None = None


def _limb(side: str) -> tuple[CapsuleSpec, ...]:
    s = side
    return (
        CapsuleSpec(f"{s}_upperarm", f"{s}_Upperarm", f"{s}_Forearm", 0.05),
        CapsuleSpec(f"{s}_forearm", f"{s}_Forearm", f"{s}_Hand", 0.04),
        CapsuleSpec(f"{s}_hand", f"{s}_Forearm", f"{s}_Hand", 0.04, extend=(0.0, 0.10)),
        CapsuleSpec(f"{s}_thigh", f"{s}_Thigh", f"{s}_Calf", 0.08),
        CapsuleSpec(f"{s}_shin", f"{s}_Calf", f"{s}_Foot", 0.05),
    )


BODY_CAPSULES: tuple[CapsuleSpec, ...] = (
    CapsuleSpec("torso_lower", "Hip", "Spine02", 0.13),
    CapsuleSpec("torso_upper", "Spine02", "NeckTwist01", 0.13),
    CapsuleSpec("hips", "L_Thigh", "R_Thigh", 0.09),
    CapsuleSpec("shoulders", "L_Upperarm", "R_Upperarm", 0.075),
    CapsuleSpec("head", "NeckTwist01", "Head", 0.09, extend=(0.07, 0.12)),
    *_limb("L"),
    *_limb("R"),
)
"""The capsules of one body (see the module docstring for the radii)."""


@dataclass(frozen=True, eq=False)
class Capsules:
    """
    One body's capsules at one instant, world (table) frame.

    :ivar names: Segment names, ``(K,)``.
    :ivar a: One end of each axis, m, ``(K, 3)``.
    :ivar b: The other end, m, ``(K, 3)``.
    :ivar radius: Radii, m, ``(K,)``.
    """

    names: tuple[str, ...]
    a: Array
    b: Array
    radius: Array


def capsules_from_joints(joint_names: Sequence[str], positions: ArrayLike, *, specs: Sequence[CapsuleSpec] = BODY_CAPSULES, scale: float = 1.0) -> Capsules:
    """
    Build a body's capsules from its joints' world positions.

    :param joint_names: The skeleton's joint names, in the order of ``positions``.
    :param positions: Joint world positions, m, ``(J, 3)`` (the game loop's ``forward_kinematics`` output).
    :param specs: The segments.
    :param scale: Multiplies every radius (and the head and hand extensions).
    :return: The capsules.
    :raises KeyError: If a segment's joint is not in the skeleton
    """

    joints = np.asarray(positions, dtype=np.float64)
    index = {name: k for k, name in enumerate(joint_names)}
    a = np.empty((len(specs), 3))
    b = np.empty((len(specs), 3))
    for k, spec in enumerate(specs):
        p, q = joints[index[spec.start]], joints[index[spec.end]]
        if spec.extend is None:
            a[k], b[k] = p, q
        else:
            axis = q - p
            length = float(np.linalg.norm(axis))
            unit = axis / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
            a[k], b[k] = q + scale * spec.extend[0] * unit, q + scale * spec.extend[1] * unit
    return Capsules(names=tuple(spec.name for spec in specs), a=a, b=b, radius=scale * np.array([spec.radius_m for spec in specs]))


@dataclass(frozen=True, eq=False)
class BodyMotion:
    """
    One body's capsules over one render tick: each end moves linearly from ``start`` (at ``t0``) to ``end`` (at ``t1``).

    :ivar t0: Tick start, game time s.
    :ivar t1: Tick end, s.
    :ivar start: Capsules at ``t0``.
    :ivar end: Capsules at ``t1`` (same segments and radii).
    """

    t0: float
    t1: float
    start: Capsules
    end: Capsules
    low: Array = field(init=False, repr=False)
    high: Array = field(init=False, repr=False)
    bounds: tuple[float, float, float, float, float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        r = self.start.radius[:, None]
        corners = np.stack([self.start.a, self.start.b, self.end.a, self.end.b])
        low = corners.min(axis=0) - r
        high = corners.max(axis=0) + r
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        lo, hi = low.min(axis=0), high.max(axis=0)
        object.__setattr__(self, "bounds", (float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2])))

    @property
    def names(self) -> tuple[str, ...]:
        """Segment names."""

        return self.start.names

    def at(self, t: float | Array) -> tuple[Array, Array]:
        """Capsule ends at time ``t`` (clamped to the tick), ``(K, 3)`` each; with an array of times, ``(T, K, 3)``."""

        span = self.t1 - self.t0
        u = np.clip((np.asarray(t, dtype=np.float64) - self.t0) / span, 0.0, 1.0) if span > 0.0 else np.ones_like(np.asarray(t, dtype=np.float64))
        u = u[..., None, None]
        return self.start.a + u * (self.end.a - self.start.a), self.start.b + u * (self.end.b - self.start.b)

    def velocities(self) -> tuple[Array, Array]:
        """Velocity of each capsule end over the tick, m/s, ``(K, 3)`` each."""

        span = self.t1 - self.t0
        if span <= 0.0:
            return np.zeros_like(self.start.a), np.zeros_like(self.start.b)
        return (self.end.a - self.start.a) / span, (self.end.b - self.start.b) / span


def _closest(centre: Array, a: Array, b: Array) -> tuple[Array, Array]:
    """Parameter along each segment of the point closest to ``centre``, and that point; broadcasting over leading axes."""

    axis = b - a
    length2 = np.sum(axis * axis, axis=-1)
    s = np.where(length2 > 1e-18, np.sum((centre - a) * axis, axis=-1) / np.where(length2 > 1e-18, length2, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    return s, a + s[..., None] * axis


def excluded_capsules(t: float, contact_times: Iterable[float], hand_joint: str) -> frozenset[str]:
    """
    The capsules to leave out of a player's body at time ``t``: the paddle hand and forearm near the player's own contact.

    :param t: Game time, s.
    :param contact_times: The player's planned or scheduled contact times, s.
    :param hand_joint: The joint that holds the paddle (``R_Hand`` or ``L_Hand``; the character's grip hand joint).
    :return: Segment names to skip; empty away from the player's contacts.
    """

    if any(-PADDLE_ARM_BEFORE_S <= t - contact <= PADDLE_ARM_AFTER_S for contact in contact_times):
        side = hand_joint.split("_", 1)[0]
        return frozenset({f"{side}_hand", f"{side}_forearm"})
    return frozenset()


@dataclass(frozen=True)
class BodyTouch:
    """
    A ball's first touch on a body within a step.

    :ivar contact: The touch, ready for :func:`lib.physics.engine.contacts.impose_contact`.
    :ivar capsule: The segment touched.
    """

    contact: SweptContact
    capsule: str


def sweep_body(
    ball: ArrayLike,
    motion: BodyMotion,
    t: float,
    dt: float,
    *,
    radius: float,
    excluded: frozenset[str] = frozenset(),
    embedded_depth: float = EMBEDDED_DEPTH_M,
) -> tuple[BodyTouch | None, int]:
    """
    Sweep the ball against one body's moving capsules over the step ``[t, t + dt]``.

    The ball moves at constant velocity (as in the other hybrid sweeps); each capsule's ends move linearly within the
    tick. The gap (ball centre to capsule axis, less both radii) is evaluated at the step's start, middle and end; the
    first bracketed root is bisected. A ball that starts the step inside a capsule by up to ``embedded_depth`` touches
    at once (and is moved out by its depth); a deeper one is left alone. A touch counts only if the ball approaches the
    surface point, whose velocity is the capsule's motion at the touch.

    :param ball: Ball state at the step start, 9 floats.
    :param motion: The body's capsules over the current tick.
    :param t: Step start, game time s.
    :param dt: Step, s.
    :param radius: Ball radius, m.
    :param excluded: Segment names to skip (:func:`excluded_capsules`).
    :param embedded_depth: See :data:`EMBEDDED_DEPTH_M`.
    :return: ``(touch or None, embedded)``: the first approaching touch, and how many capsules the ball started deeper
        inside than ``embedded_depth`` (left alone).
    """

    state = np.asarray(ball, dtype=np.float64)
    x, y, z, vx, vy, vz = (float(v) for v in state[:6])
    ex, ey, ez = x + vx * dt, y + vy * dt, z + vz * dt
    lo_x, lo_y, lo_z, hi_x, hi_y, hi_z = motion.bounds
    if (
        min(x, ex) - radius > hi_x
        or max(x, ex) + radius < lo_x
        or min(y, ey) - radius > hi_y
        or max(y, ey) + radius < lo_y
        or min(z, ez) - radius > hi_z
        or max(z, ez) + radius < lo_z
    ):
        return None, 0
    c0 = state[:3]
    v = state[3:6]
    c1 = c0 + v * dt
    step_low = np.minimum(c0, c1) - radius
    step_high = np.maximum(c0, c1) + radius
    candidate = np.all((motion.low <= step_high) & (motion.high >= step_low), axis=1)
    if excluded:
        candidate &= np.array([name not in excluded for name in motion.names])
    rows = np.flatnonzero(candidate)
    if rows.size == 0:
        return None, 0
    reach = motion.start.radius[rows] + radius
    times = np.array([t, t + 0.5 * dt, t + dt])
    a_all, b_all = motion.at(times)
    a, b = a_all[:, rows], b_all[:, rows]
    centres = np.stack([c0, c0 + 0.5 * v * dt, c1])[:, None, :]
    _s, point = _closest(centres, a, b)
    gaps = np.linalg.norm(centres - point, axis=-1) - reach[None, :]
    embedded = gaps[0] < -embedded_depth
    inside = (gaps[0] <= 0.0) & ~embedded
    lower = np.zeros(rows.size)
    upper = np.full(rows.size, -1.0)
    upper[inside] = 0.0
    mid = ~inside & ~embedded & (gaps[0] > 0.0) & (gaps[1] <= 0.0)
    upper[mid] = 0.5
    late = ~inside & ~embedded & ~mid & (gaps[0] > 0.0) & (gaps[1] > 0.0) & (gaps[2] <= 0.0)
    lower[late], upper[late] = 0.5, 1.0
    bracket = mid | late
    if bracket.any():
        k = np.flatnonzero(bracket)
        lo, hi = lower[k].copy(), upper[k].copy()
        for _ in range(SWEEP_BISECTIONS):
            f = 0.5 * (lo + hi)
            a_f, b_f = motion.at(t + f * dt)
            a_k = a_f[np.arange(k.size), rows[k]]
            b_k = b_f[np.arange(k.size), rows[k]]
            centre_f = c0 + (f * dt)[:, None] * v
            _sk, point_k = _closest(centre_f, a_k, b_k)
            gap_f = np.linalg.norm(centre_f - point_k, axis=-1) - reach[k]
            touching = gap_f <= 0.0
            hi = np.where(touching, f, hi)
            lo = np.where(touching, lo, f)
        upper[k] = hi
    hits = np.flatnonzero(upper >= 0.0)
    if hits.size == 0:
        return None, int(embedded.sum())
    va_all, vb_all = motion.velocities()
    for j in hits[np.argsort(upper[hits])]:
        fraction = float(upper[j])
        row = int(rows[j])
        when = t + fraction * dt
        a_t, b_t = motion.at(when)
        centre = c0 + v * (fraction * dt)
        s, point = _closest(centre, a_t[row], b_t[row])
        offset = centre - point
        distance = float(np.linalg.norm(offset))
        if distance < 1e-9:
            continue
        normal = offset / distance
        surface_velocity = (1.0 - float(s)) * va_all[row] + float(s) * vb_all[row]
        if float((v - surface_velocity) @ normal) >= 0.0:
            continue
        depth = max(float(reach[j]) - distance, 0.0) if fraction == 0.0 else 0.0
        contact = SweptContact(
            fraction=fraction,
            ball=np.concatenate([centre, state[3:9]]),
            normal=normal,
            surface_velocity=surface_velocity,
            blade_offset_m=None,
            depth=depth,
        )
        return BodyTouch(contact=contact, capsule=motion.names[row]), int(embedded.sum())
    return None, int(embedded.sum())


__all__ = [
    "BODY_CAPSULES",
    "BODY_CONTACT",
    "EMBEDDED_DEPTH_M",
    "PADDLE_ARM_AFTER_S",
    "PADDLE_ARM_BEFORE_S",
    "BodyMotion",
    "BodyTouch",
    "CapsuleSpec",
    "Capsules",
    "capsules_from_joints",
    "excluded_capsules",
    "sweep_body",
]
