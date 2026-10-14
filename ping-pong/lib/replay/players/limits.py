"""
Human joint limits: per-joint angular speed limits and anatomical angle ranges (:mod:`lib.replay.players.anatomy`
defines the angles), used by the arm IK and by the replay's motion QA.

The numbers live in :data:`JOINT_LIMITS_PATH` (``data/joint_limits_v2.json``) with their method, evidence, citations and
input lineage. In short:

- **Speed limit** of a joint: the fastest the raw capture moved it (59,154 frames at 30 Hz, before the retarget's
  850 deg/s cap and low-pass), times the 90th percentile of its 30 Hz under-sampling of stroke peaks (cubic-spline
  resampling at 240 Hz around 692 strokes), cross-checked against elite table-tennis stroke kinematics (Bankosz and
  Winiarski 2018). Left and right share the larger side's value: it is what this player's body did.
- **Glitch ceiling** of a joint: the fastest any human source we hold moved it: this player's capture, the ActorCore
  rigs' 26 authored clips of other actors, and the cited elite and healthy-adult means + 2 SD (Bankosz and Winiarski
  2018; Roijezon et al. 2010).

A **glitch** (:func:`detect_glitches`) is a frame where a joint turns faster than its glitch ceiling, or changes its
velocity within one frame by more than that ceiling (reaching it from rest in under one 24 fps frame, 41.7 ms; the
capture never changes a joint's velocity by more than 0.43-0.93 of its own top speed per frame). Physiological speed is
never a glitch: the capture played back at 24 fps has none, however fast its strokes are.

Joints are named as on the ActorCore CC3 skeleton (``R_Forearm``, ``Spine02``...). Twist and share joints take their
pair's limit; fingers, toes and the face (which the capture does not animate) take the nearest limb's as a stand-in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.replay.players import rotations as rot

JOINT_LIMITS_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "joint_limits_v2.json"
"""The limits data (``joint_limits_v2``: v1's speeds and ranges plus the gate ranges)."""
REFERENCE_FPS: float = 24.0
"""The frame the glitch rule's velocity change is judged over (the replay's render rate): no joint reaches its ceiling
speed from rest in under 1/24 s."""

_STAND_IN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(Thumb|Index|Mid|Ring|Pinky)\d", "finger"),
    (r"Toe", "toe"),
    (r"(Eye|Jaw|Teeth|Tongue|Facial)", "face"),
    (r"(Breast|Ribs)", "breast_ribs"),
    (r"RL_BoneRoot", "pelvis"),
)


@dataclass(frozen=True)
class JointLimits:
    """
    The joint limits table.

    :ivar version: The data file's version tag.
    :ivar speed_deg_s: This player's speed limit per group, deg/s (body feasibility).
    :ivar glitch_deg_s: The human glitch ceiling per group, deg/s.
    :ivar ranges: Per anatomical angle (:mod:`~lib.replay.players.anatomy`'s names without the side), its
        ``(low, high)`` range in degrees: the cited clinical range intersected with the natural envelope.
    :ivar gate_ranges: Per angle, the range the motion gates hold it to, degrees: the clinical bound where that binds,
        else the furthest any natural source went (``gate_method`` in the data file); wider than :attr:`ranges`, which
        the arm IK targets.
    :ivar gated: The angles the motion gates hold (arm, clavicle, spine); the rest are reported.
    :ivar hinge_azimuth_deg: Each rig's elbow hinge in its rest pose (``anatomy.anatomical_angles``'s calibration).
    :ivar joint_groups: ActorCore joint pattern (``{s}`` for the side) to group.
    :ivar stand_ins: Stand-in group per evidence-free class (finger, toe, face, breast_ribs).
    :ivar document: The whole data file.
    """

    version: str
    speed_deg_s: dict[str, float]
    glitch_deg_s: dict[str, float]
    ranges: dict[str, tuple[float, float]]
    gate_ranges: dict[str, tuple[float, float]]
    gated: tuple[str, ...]
    hinge_azimuth_deg: dict[str, float]
    joint_groups: dict[str, str]
    stand_ins: dict[str, str]
    document: dict[str, Any] = field(repr=False, compare=False)

    def group(self, joint: str) -> str:
        """
        The limit group of an ActorCore joint.

        :raises KeyError: For a joint the table cannot place
        """

        if joint in self.joint_groups:
            return self.joint_groups[joint]
        side = re.match(r"^([LR])_(.+)$", joint)
        if side:
            key = "{s}_" + side.group(2)
            if key in self.joint_groups:
                return self.joint_groups[key]
        for pattern, kind in _STAND_IN_PATTERNS:
            if re.search(pattern, joint):
                return self.stand_ins.get(kind, kind)
        raise KeyError(f"no joint limit for {joint!r}")

    def speed_limit_deg_s(self, joint: str) -> float:
        """The joint's speed limit (this player's, for body feasibility), deg/s."""

        return self.speed_deg_s[self.group(joint)]

    def glitch_speed_deg_s(self, joint: str) -> float:
        """The joint's glitch ceiling (the fastest any human source moved it), deg/s."""

        return self.glitch_deg_s[self.group(joint)]

    def glitch_step_change_deg(self, joint: str, fps: float = REFERENCE_FPS) -> float:
        """
        The largest velocity change in one sample that is not a glitch, degrees per sample (the angle between
        consecutive steps): the ceiling reached from rest over :data:`REFERENCE_FPS`'s frame, ``ceiling * 24 / fps^2``.
        """

        return self.glitch_speed_deg_s(joint) * REFERENCE_FPS / fps**2

    def speed_limits(self, joints: Sequence[str]) -> NDArray[np.float64]:
        """Speed limits of several joints, deg/s."""

        return np.array([self.speed_limit_deg_s(j) for j in joints], dtype=np.float64)

    def glitch_speeds(self, joints: Sequence[str]) -> NDArray[np.float64]:
        """Glitch ceilings of several joints, deg/s."""

        return np.array([self.glitch_speed_deg_s(j) for j in joints], dtype=np.float64)

    def hinge(self, body: str) -> dict[str, float]:
        """A rig's hinge calibration for :func:`~lib.replay.players.anatomy.anatomical_angles` (both sides)."""

        value = self.hinge_azimuth_deg.get(body, 0.0)
        return {"R": value, "L": value}

    @property
    def largest_glitch_speed_deg_s(self) -> float:
        """The largest glitch ceiling of any joint (the fallback when joints are unnamed)."""

        return max(self.glitch_deg_s.values())


@cache
def load_joint_limits(path: Path = JOINT_LIMITS_PATH) -> JointLimits:
    """The joint limits table (cached)."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = document["groups"]
    stand_ins = dict(document.get("stand_ins", {}))
    speed = {name: float(entry["speed_limit_deg_s"]) for name, entry in groups.items()}
    glitch = {name: float(entry["glitch_speed_deg_s"]) for name, entry in groups.items()}
    for kind, group in stand_ins.items():
        speed.setdefault(kind, speed[group])
        glitch.setdefault(kind, glitch[group])
    ranges = {name: (float(entry["range_deg"][0]), float(entry["range_deg"][1])) for name, entry in document.get("ranges", {}).items()}
    gate_ranges = {
        name: (float(entry["gate_range_deg"][0]), float(entry["gate_range_deg"][1])) if "gate_range_deg" in entry else ranges[name]
        for name, entry in document.get("ranges", {}).items()
    }
    gated = tuple(name for name, entry in document.get("ranges", {}).items() if entry.get("gated"))
    hinge = {body: float(value) for body, value in document.get("hinge_azimuth_deg", {}).items() if isinstance(value, (int, float))}
    return JointLimits(
        version=str(document["version"]),
        speed_deg_s=speed,
        glitch_deg_s=glitch,
        ranges=ranges,
        gate_ranges=gate_ranges,
        gated=gated,
        hinge_azimuth_deg=hinge,
        joint_groups=dict(document["actorcore_joint_groups"]),
        stand_ins=stand_ins,
        document=document,
    )


@dataclass(frozen=True)
class Glitches:
    """
    The glitches in one sampled motion.

    :ivar fps: Sample rate, Hz.
    :ivar frames: Per sample, whether any joint glitched there (a speed beyond its ceiling on the step into or out of
        the sample, or a velocity change beyond its ceiling at the sample), ``(T,)``.
    :ivar episodes: Runs of glitching samples (gaps of at most one sample merged): the count the gate grades.
    :ivar speed_events: Joint-steps faster than their joint's glitch ceiling.
    :ivar step_change_events: Joint-samples whose velocity changed by more than their joint's ceiling in one sample.
    :ivar worst: Up to ten worst joint events ``(sample, joint, kind, value, limit)``, largest ratio first.
    :ivar max_speed_ratio: Largest speed over its ceiling.
    :ivar max_step_change_ratio: Largest velocity change over its limit.
    :ivar named: Whether per-joint ceilings were used (``False``: unnamed joints got the largest ceiling).
    """

    fps: float
    frames: NDArray[np.bool_]
    episodes: int
    speed_events: int
    step_change_events: int
    worst: tuple[tuple[int, str, str, float, float], ...]
    max_speed_ratio: float
    max_step_change_ratio: float
    named: bool


def detect_glitches(rotations: ArrayLike, fps: float, joints: Sequence[str] | None = None, *, limits: JointLimits | None = None) -> Glitches:
    """
    Glitches in a sampled motion: a joint turning faster than its glitch ceiling, or changing its velocity in one
    sample by more than the ceiling allows (the second difference on SO(3), the angle between consecutive steps,
    against :meth:`JointLimits.glitch_step_change_deg`).

    :param rotations: Local joint rotations ``(T, K, 4)`` (wxyz), in time order at ``fps``.
    :param fps: Sample rate, Hz.
    :param joints: The ``K`` joints' ActorCore names; ``None`` gives every joint the largest ceiling (a weaker check,
        reported as ``named=False``).
    :param limits: The table (default :func:`load_joint_limits`).
    :return: The glitches.
    """

    table = limits or load_joint_limits()
    q = np.asarray(rotations, dtype=np.float64)
    count = q.shape[0]
    if count < 2:
        return Glitches(fps, np.zeros(count, dtype=bool), 0, 0, 0, (), 0.0, 0.0, joints is not None)
    k = q.shape[1]
    names = list(joints) if joints is not None else [f"joint{i}" for i in range(k)]
    speed_limit = table.glitch_speeds(names) if joints is not None else np.full(k, table.largest_glitch_speed_deg_s)
    step_limit = speed_limit * REFERENCE_FPS / fps**2
    steps = rot.multiply(rot.conjugate(q[:-1]), q[1:])  # (T-1, K, 4) body-frame steps
    speed = np.degrees(rot.angle(steps)) * fps
    speed_ratio = speed / speed_limit[None, :]
    frames = np.zeros(count, dtype=bool)
    fast = speed_ratio > 1.0
    frames[:-1] |= fast.any(axis=1)
    frames[1:] |= fast.any(axis=1)
    if count >= 3:
        change = np.degrees(rot.angle(rot.multiply(rot.conjugate(steps[:-1]), steps[1:])))  # (T-2, K) at samples 1..T-2
        change_ratio = change / step_limit[None, :]
    else:
        change_ratio = np.zeros((0, k))
    abrupt = change_ratio > 1.0
    frames[1:-1] |= abrupt.any(axis=1)
    events: list[tuple[float, int, str, str, float, float]] = []
    for i, j in zip(*np.nonzero(fast), strict=True):
        events.append((float(speed_ratio[i, j]), int(i), names[j], "speed_deg_s", float(speed[i, j]), float(speed_limit[j])))
    for i, j in zip(*np.nonzero(abrupt), strict=True):
        events.append((float(change_ratio[i, j]), int(i) + 1, names[j], "step_change_deg", float(change_ratio[i, j] * step_limit[j]), float(step_limit[j])))
    events.sort(key=lambda e: -e[0])
    flagged = np.flatnonzero(frames)
    episodes = 0
    last = -10
    for i in flagged:
        if i - last > 2:
            episodes += 1
        last = int(i)
    return Glitches(
        fps=fps,
        frames=frames,
        episodes=episodes,
        speed_events=int(fast.sum()),
        step_change_events=int(abrupt.sum()),
        worst=tuple((e[1], e[2], e[3], round(e[4], 2), round(e[5], 2)) for e in events[:10]),
        max_speed_ratio=float(speed_ratio.max()) if speed_ratio.size else 0.0,
        max_step_change_ratio=float(change_ratio.max()) if change_ratio.size else 0.0,
        named=joints is not None,
    )


def _per_joint_flags(q: NDArray[np.float64], fps: float, names: list[str], table: JointLimits) -> NDArray[np.bool_]:
    """Per sample and joint, whether the joint glitches there (:func:`detect_glitches`'s two tests), ``(T, K)``."""

    speed_limit = table.glitch_speeds(names)
    step_limit = speed_limit * REFERENCE_FPS / fps**2
    steps = rot.multiply(rot.conjugate(q[:-1]), q[1:])
    fast = np.degrees(rot.angle(steps)) * fps > speed_limit[None, :]
    flags = np.zeros(q.shape[:2], dtype=bool)
    flags[:-1] |= fast
    flags[1:] |= fast
    if len(q) >= 3:
        change = np.degrees(rot.angle(rot.multiply(rot.conjugate(steps[:-1]), steps[1:])))
        flags[1:-1] |= change > step_limit[None, :]
    return flags


def repair_glitches(
    rotations: ArrayLike,
    fps: float,
    joints: Sequence[str],
    *,
    pinned: ArrayLike | None = None,
    limits: JointLimits | None = None,
    pad: int = 2,
    iterations: int = 8,
    detect_fps: float = 30.0,
) -> tuple[NDArray[np.float64], int]:
    """
    Glitches (:func:`detect_glitches`) removed from a sampled motion: each glitching joint's samples around a glitch
    (``pad`` either side, widening while a glitch remains) are replaced by a cubic Hermite curve in the rotation's
    tangent space that meets the motion on either side with its own velocity. Unpinned samples only: a window that
    would reach a pinned sample stops short of it.

    :param rotations: Local joint rotations ``(T, K, 4)`` (wxyz).
    :param joints: The ``K`` joints' ActorCore names.
    :param pinned: Samples never changed, ``(T,)``.
    :param detect_fps: Glitches are detected at about this rate (every ``round(fps / detect_fps)``-th sample of a
        faster motion, as the replay's render and the capture see it): finite differences at 120 fps see spikes in a
        fast but smooth stroke that no rendered frame shows.
    :return: The repaired rotations (a copy) and how many samples changed.
    """

    table = limits or load_joint_limits()
    q = rot.continuous(np.array(rotations, dtype=np.float64, copy=True), axis=0)
    count, k = q.shape[:2]
    names = list(joints)
    fixed = np.zeros(count, dtype=bool) if pinned is None else np.asarray(pinned, dtype=bool)
    changed = np.zeros((count, k), dtype=bool)
    width = pad
    stride = max(1, round(fps / detect_fps))
    for _ in range(iterations):
        if stride > 1:
            coarse = _per_joint_flags(q[::stride], fps / stride, names, table)
            flags = np.zeros((count, k), dtype=bool)
            for i in np.flatnonzero(coarse.any(axis=1)):
                flags[max(0, (i - 1) * stride) : min(count, (i + 1) * stride + 1)] |= coarse[i][None, :]
        else:
            flags = _per_joint_flags(q, fps, names, table)
        if not flags.any():
            break
        for j in np.flatnonzero(flags.any(axis=0)):
            bad = np.flatnonzero(flags[:, j])
            runs: list[list[int]] = [[int(bad[0]), int(bad[0])]]
            for i in bad[1:]:
                if i - runs[-1][1] <= 2 * width + 1:
                    runs[-1][1] = int(i)
                else:
                    runs.append([int(i), int(i)])
            for a, b in runs:
                lo, hi = max(1, a - width), min(count - 2, b + width)
                before = np.flatnonzero(fixed[: a + 1])
                after = np.flatnonzero(fixed[b:]) + b
                if len(before):
                    lo = max(lo, int(before[-1]) + 2)
                if len(after):
                    hi = min(hi, int(after[0]) - 2)
                if lo > hi:
                    continue
                left, right = lo - 1, hi + 1
                n = right - left
                ql, qr = q[left, j], q[right, j]
                v_left = rot.to_rotvec(rot.multiply(rot.conjugate(q[left - 1, j]), ql)) if left >= 1 else np.zeros(3)
                v_right = rot.to_rotvec(rot.multiply(rot.conjugate(qr), q[right + 1, j])) if right + 1 < count else np.zeros(3)
                end = rot.to_rotvec(rot.multiply(rot.conjugate(ql), qr))
                u = (np.arange(1, n) / n)[:, None]
                h10, h01, h11 = u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
                tangent = h10 * (v_left * n) + h01 * end + h11 * (v_right * n)
                q[lo : hi + 1, j] = rot.multiply(np.broadcast_to(ql, (n - 1, 4)), rot.from_rotvec(tangent))
                changed[lo : hi + 1, j] = True
        width += 2
    return rot.normalize(q), int(changed.any(axis=1).sum())


def angle_excess(angles: dict[str, Any], *, limits: JointLimits | None = None, gate: bool = False) -> dict[str, NDArray[np.float64]]:
    """
    How far each angle lies outside its range, degrees (0 inside), for every angle the table bounds.

    The shoulder's plane of elevation is bounded only above 45 degrees of elevation (below, it is ill-defined).

    :param angles: :func:`~lib.replay.players.anatomy.anatomical_angles`'s output (or any subset).
    :param gate: Against the gate ranges (:attr:`JointLimits.gate_ranges`) instead of the IK's ranges.
    """

    table = limits or load_joint_limits()
    bounds = table.gate_ranges if gate else table.ranges
    out: dict[str, NDArray[np.float64]] = {}
    for name, value in angles.items():
        base = name.split(".", 1)[-1]
        if base not in bounds:
            continue
        lo, hi = bounds[base]
        v = np.asarray(value, dtype=np.float64)
        excess = np.maximum(np.maximum(lo - v, v - hi), 0.0)
        if base == "shoulder_plane":
            side = name.split(".", 1)[0] + "." if "." in name else ""
            elevation = angles.get(f"{side}shoulder_elevation")
            if elevation is not None:
                excess = np.where(np.asarray(elevation, dtype=np.float64) > 45.0, excess, 0.0)
        out[name] = excess
    return out


@dataclass(frozen=True)
class LimitViolations:
    """
    Per-frame joint-limit violations of one motion.

    :ivar frames: Samples graded.
    :ivar count: Per angle, the samples outside its gate range by more than ``tolerance_deg``.
    :ivar max_excess_deg: Per angle, the largest excess, degrees.
    :ivar frames_any_gated: Samples with any gated angle out of range by more than ``tolerance_deg``.
    :ivar tolerance_deg: The excess below which a sample is not counted.
    """

    frames: int
    count: dict[str, int]
    max_excess_deg: dict[str, float]
    frames_any_gated: int
    tolerance_deg: float

    def worst(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.max_excess_deg.items(), key=lambda item: -item[1])[:n]


def limit_violations(angles: dict[str, Any], *, tolerance_deg: float = 0.5, limits: JointLimits | None = None) -> LimitViolations:
    """
    Count, per angle, the samples outside its gate range (the clinical bound, or the furthest a natural source went)
    and the largest excess: the gate's measurement.
    """

    table = limits or load_joint_limits()
    excess = angle_excess(angles, limits=table, gate=True)
    frames = max((len(np.atleast_1d(v)) for v in excess.values()), default=0)
    any_gated = np.zeros(frames, dtype=bool)
    for name, value in excess.items():
        if name.split(".", 1)[-1] in table.gated:
            any_gated |= np.atleast_1d(value) > tolerance_deg
    return LimitViolations(
        frames=frames,
        count={name: int(np.sum(np.atleast_1d(v) > tolerance_deg)) for name, v in excess.items()},
        max_excess_deg={name: round(float(np.max(v)) if np.size(v) else 0.0, 2) for name, v in excess.items()},
        frames_any_gated=int(any_gated.sum()),
        tolerance_deg=tolerance_deg,
    )


__all__ = [
    "JOINT_LIMITS_PATH",
    "REFERENCE_FPS",
    "Glitches",
    "JointLimits",
    "LimitViolations",
    "angle_excess",
    "detect_glitches",
    "limit_violations",
    "load_joint_limits",
    "repair_glitches",
]
