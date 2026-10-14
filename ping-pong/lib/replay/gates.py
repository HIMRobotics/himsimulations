"""
The replay's tests. A replay passes when all three pass and it ended on its terminal condition (the serve schedule
exhausted and the last point ended physically, not the ``max_sim_s`` cap):

- **Motion QA** of the motion the players actually played (adaptations included, sampled at the render rate): no
  non-finite pose, no frame outside the recorded track, at most one glitch per minute (a joint faster than any human
  source moved it, or reaching that speed from rest within one frame: :func:`lib.replay.players.limits.detect_glitches`),
  and the paddle never leaving the hand (the hand-to-paddle gap at most 1 cm on every frame).
- **Physical invariants**: no tunnelling through the table, net, floor or a paddle; no ball leaving the room; no engine
  or fixed-surface bounce that gains energy; at most four net touches per point; and the opening drop tests (a ball
  released 30 cm above the table) rebounding inside the ITTF band, 23-26 cm.
- **Telemetry completeness**: 3D and scalar channels at 10 Hz or more; every rendered frame of every camera in the
  recording at 24 fps, none dropped; every recorded Orbbec frame of the replay's span at its own time; the blueprint,
  the event log, and an MP4 whose frame count equals the rendered frames.

The thresholds are frozen with :data:`GATE_VERSION`. How many recorded strokes the simulation realised (a physical
paddle contact where the recording had one) is reported beside the verdict, never graded: a physical return diverges
from the recorded one, so a realised share well below 100 % is expected.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from lib.physics.engine.contacts import REST_SPEED_MPS, ContactEvent
from lib.replay.players.limits import detect_glitches, load_joint_limits

GATE_VERSION: str = "replay-tests/14"
"""Version of the frozen thresholds: a changed criterion changes the version."""


class GateThresholds(BaseModel):
    """
    The tests' frozen thresholds.

    :ivar pop_deg_per_s: A driven joint turning faster than this between samples is a pop: logged for frame review,
        never graded (real strokes exceed it).
    :ivar max_glitches_per_minute: Glitch episodes allowed per minute of play.
    :ivar energy_gain_tolerance: Relative kinetic-energy gain an engine bounce may show before it counts as a gain.
    :ivar drop_rebound_band_m: ITTF table test: a ball dropped from 30 cm rebounds 23-26 cm.
    :ivar max_net_touches_per_point: Net touches allowed between two placements (:func:`net_touches`).
    :ivar net_slow_episode_gap_s: Slow cord touches closer than this are one touch.
    :ivar max_hand_gap_m: The paddle never leaves the hand: the gap on every frame.
    :ivar min_tick_share: 3D and scalar telemetry ticks logged against the expected count.
    :ivar min_telemetry_hz: Slowest 3D and scalar telemetry rate.
    :ivar min_video_fps: Slowest camera video rate in the recording.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pop_deg_per_s: float = 900.0
    max_glitches_per_minute: float = 1.0
    energy_gain_tolerance: float = 0.02
    drop_rebound_band_m: tuple[float, float] = (0.23, 0.26)
    max_net_touches_per_point: int = 4
    net_slow_episode_gap_s: float = 0.020
    max_hand_gap_m: float = 0.01
    min_tick_share: float = 0.95
    min_telemetry_hz: float = 10.0
    min_video_fps: float = 24.0


THRESHOLDS = GateThresholds()


class GateResult(BaseModel):
    """
    One test's verdict.

    :ivar name: Test name.
    :ivar passed: Verdict.
    :ivar detail: One line with the measurements against the thresholds.
    :ivar measurements: Raw values with units in their names.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    passed: bool
    detail: str
    measurements: dict[str, float | int | str | bool | None]


# ---- motion QA -------------------------------------------------------------------------------------------------------


@dataclass
class MotionSamples:
    """
    One player's played motion, sampled every render tick.

    :ivar fps: Sample rate, Hz.
    :ivar joint_names: Joints in ``positions``.
    :ivar positions: Table-frame joint positions, one ``(J, 3)`` per sample.
    :ivar driven_rotations: Local rotations of the driven joints, one ``(K, 4)`` per sample.
    :ivar driven_joints: The driven joints' names (each is held to its own glitch ceiling).
    :ivar held_frames: Samples taken outside the track's time span.
    :ivar ik_miss_m: Per sample with the arm IK active, the hand's miss of the planned paddle, m.
    :ivar hand_gap_m: Per sample, the carried paddle's distance from the one the posed hand holds, m.
    """

    fps: float
    joint_names: tuple[str, ...]
    driven_joints: tuple[str, ...]
    positions: list[NDArray[np.float64]] = field(default_factory=list)
    driven_rotations: list[NDArray[np.float64]] = field(default_factory=list)
    held_frames: int = 0
    ik_miss_m: list[float] = field(default_factory=list)
    hand_gap_m: list[float] = field(default_factory=list)

    def add(
        self, positions: NDArray[np.float64], driven_rotations: NDArray[np.float64], *, ik_miss_m: float | None = None, hand_gap_m: float | None = None
    ) -> None:
        """Append one tick (``ik_miss_m`` when the arm IK chased the planned paddle that tick)."""

        self.positions.append(np.asarray(positions, dtype=np.float64))
        self.driven_rotations.append(np.asarray(driven_rotations, dtype=np.float64))
        if ik_miss_m is not None:
            self.ik_miss_m.append(float(ik_miss_m))
        if hand_gap_m is not None:
            self.hand_gap_m.append(float(hand_gap_m))


def evaluate_motion(samples: Mapping[str, MotionSamples], *, thresholds: GateThresholds = THRESHOLDS) -> GateResult:
    """
    Motion QA over every player: finite poses, no frame outside the track, glitches within the allowance, and the
    paddle in the hand on every frame.
    """

    measurements: dict[str, float | int | str | bool | None] = {}
    failures: list[str] = []
    limits = load_joint_limits()
    for role, sample in samples.items():
        count = len(sample.positions)
        if count < 3:
            failures.append(f"{role}: only {count} samples")
            continue
        positions = np.stack(sample.positions)
        rotations = np.stack(sample.driven_rotations)
        finite = bool(np.all(np.isfinite(positions)) and np.all(np.isfinite(rotations)))
        dots = np.abs(np.sum(rotations[1:] * rotations[:-1], axis=-1))
        steps = np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))
        glitches = detect_glitches(rotations, sample.fps, sample.driven_joints, limits=limits)
        minutes = count / sample.fps / 60.0
        allowed = round(thresholds.max_glitches_per_minute * max(minutes, 1.0 / 60.0), 2)
        hand_gap = max(sample.hand_gap_m) if sample.hand_gap_m else None
        measurements |= {
            f"{role}_samples": count,
            f"{role}_finite": finite,
            f"{role}_held_frames": sample.held_frames,
            f"{role}_glitches": glitches.episodes,
            f"{role}_glitches_allowed": allowed,
            f"{role}_glitch_worst": "; ".join(
                f"{joint} {kind} {value:g}/{limit:g} at sample {index}" for index, joint, kind, value, limit in glitches.worst[:3]
            )
            or None,
            f"{role}_pops": int(np.sum(steps > thresholds.pop_deg_per_s / sample.fps)),
            f"{role}_max_joint_step_deg": round(float(np.max(steps)), 2),
            f"{role}_ik_miss_max_m": round(max(sample.ik_miss_m), 4) if sample.ik_miss_m else 0.0,
            f"{role}_hand_gap_max_m": None if hand_gap is None else round(hand_gap, 4),
        }
        checks = (
            (finite, "non-finite pose"),
            (sample.held_frames == 0, f"{sample.held_frames} frames outside the track"),
            (glitches.episodes <= allowed, f"{glitches.episodes} glitches > {allowed:.1f}"),
            (
                hand_gap is not None and hand_gap <= thresholds.max_hand_gap_m,
                "hand-to-paddle gap not measured" if hand_gap is None else f"the paddle leaves the hand by {hand_gap * 100:.1f} cm",
            ),
        )
        failures.extend(f"{role}: {reason}" for ok, reason in checks if not ok)
    detail = "; ".join(
        f"{role}: glitches {measurements.get(f'{role}_glitches')} (allowed {measurements.get(f'{role}_glitches_allowed')}), "
        f"hand gap max {measurements.get(f'{role}_hand_gap_max_m')} m, frames outside the track {measurements.get(f'{role}_held_frames')}, "
        f"finite {measurements.get(f'{role}_finite')}"
        for role in samples
        if f"{role}_samples" in measurements
    )
    if failures:
        detail = f"FAILED ({'; '.join(failures)}) | {detail}"
    return GateResult(name="motion QA", passed=not failures, detail=detail, measurements=measurements)


# ---- physical invariants ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DropTest:
    """
    One opening drop test.

    :ivar drop_height_m: Ball bottom above the table at release, m.
    :ivar rebound_m: Ball bottom above the table at the first rebound's apex, m (nan when it never rebounded).
    :ivar reference_m: The reference model's rebound, m.
    :ivar restitution: The measured normal restitution of the bounce.
    """

    drop_height_m: float
    rebound_m: float
    reference_m: float
    restitution: float


def net_touches(events: Sequence[ContactEvent], regions: Mapping[int, str], thresholds: GateThresholds = THRESHOLDS) -> dict[str, int]:
    """
    One point's net touches, so legitimate cord physics is not read as chatter:

    - ``bounces``: cord contacts the ball leaves at or above the rest speed along the contact normal;
    - ``slow_episodes``: runs of slower cord touches (a ball rolling over the cord), a new run starting only after a gap
      of :attr:`GateThresholds.net_slow_episode_gap_s` from the last one's end;
    - ``mesh``: touches on the mesh below the top tape (each a catch).
    """

    counts = {"bounces": 0, "slow_episodes": 0, "mesh": 0}
    last_slow_end: float | None = None
    for event in sorted(events, key=lambda item: item.time_s):
        if regions.get(event.index) == "mesh":
            counts["mesh"] += 1
            continue
        leaving = float(np.asarray(event.after[3:6]) @ np.asarray(event.normal))
        if event.regime != "rolling" and leaving >= REST_SPEED_MPS:
            counts["bounces"] += 1
            continue
        if last_slow_end is None or event.time_s - last_slow_end >= thresholds.net_slow_episode_gap_s:
            counts["slow_episodes"] += 1
        last_slow_end = max(last_slow_end or -math.inf, event.time_s + float(event.duration_s))
    return counts


def evaluate_invariants(
    events: Sequence[ContactEvent],
    tunnels: dict[str, int],
    drops: Sequence[DropTest],
    *,
    point_starts: Sequence[float] = (),
    net_regions: Mapping[int, str] | None = None,
    thresholds: GateThresholds = THRESHOLDS,
) -> GateResult:
    """
    No tunnelling (nor a ball leaving the room), no engine bounce gaining energy, drop tests in the ITTF band, and no
    net-contact chatter.

    :param events: The replay's contact events.
    :param tunnels: Tunnelling counts by surface (``room`` counts balls that left the room).
    :param drops: The opening drop tests.
    :param point_starts: Times balls were placed (serves, continuity balls); net touches are counted between them.
    :param net_regions: The part of the net each net event touched (``mesh``, ``cord``, ``post``, ``clamp``).
    """

    # The engine's own bounces, and the fixed surfaces' however they are resolved: none may gain energy.
    regions = net_regions or {}
    engine = [
        event
        for event in events
        if (event.mode == "engine" and event.surface in ("table", "net", "floor", "wall"))
        or event.surface in ("frame", "floor", "wall")
        or regions.get(event.index) in ("post", "clamp")
    ]
    gains = [event for event in engine if event.energy_after_j > event.energy_before_j * (1.0 + thresholds.energy_gain_tolerance) + 1e-9]
    low, high = thresholds.drop_rebound_band_m
    drops_ok = bool(drops) and all(low <= drop.rebound_m <= high for drop in drops)
    starts = np.sort(np.asarray(point_starts, dtype=np.float64))
    nets = [event for event in events if event.surface == "net"]
    spans = np.searchsorted(starts, [event.time_s for event in nets], side="right")
    per_point = [net_touches([event for event, span in zip(nets, spans, strict=True) if span == point], regions, thresholds) for point in set(spans.tolist())]
    net_most = max((sum(touches.values()) for touches in per_point), default=0)
    bodies = [event for event in events if event.surface == "body"]
    measurements: dict[str, float | int | str | bool | None] = {
        **{f"tunnel_{name}": count for name, count in tunnels.items()},
        "engine_contacts": len(engine),
        "energy_gain_events": len(gains),
        "drop_tests": len(drops),
        "drop_rebound_mm": ", ".join(f"{1e3 * drop.rebound_m:.1f}" for drop in drops),
        "drop_reference_mm": round(1e3 * drops[0].reference_m, 1) if drops else None,
        "drop_restitution": ", ".join(f"{drop.restitution:.3f}" for drop in drops),
        "net_touches_per_point_max": net_most,
        "net_mesh_touches": sum(touches["mesh"] for touches in per_point),
        "body_touches": len(bodies),
    }
    passed = sum(tunnels.values()) == 0 and not gains and drops_ok and net_most <= thresholds.max_net_touches_per_point
    detail = (
        f"tunnelling {tunnels}, {len(gains)} of {len(engine)} engine contacts gained energy (> {thresholds.energy_gain_tolerance:.0%}), "
        f"at most {net_most} net touches per point (<= {thresholds.max_net_touches_per_point}), "
        f"drop rebounds {measurements['drop_rebound_mm']} mm against [{1e3 * low:.0f}, {1e3 * high:.0f}] (reference {measurements['drop_reference_mm']} mm)"
    )
    return GateResult(name="physical invariants", passed=passed, detail=detail, measurements=measurements)


# ---- telemetry completeness ------------------------------------------------------------------------------------------


def evaluate_telemetry(
    *,
    ticks_3d: int,
    expected_ticks_3d: int,
    rate_3d_hz: float,
    images: dict[str, int],
    expected_images: int,
    images_dropped: int,
    video_fps: float,
    real_frames: dict[str, int],
    real_frames_expected: dict[str, int],
    events: int,
    blueprint: bool,
    clip_frames: int | None,
    rendered_frames: int,
    clip_duration_s: float | None,
    fps: float,
    thresholds: GateThresholds = THRESHOLDS,
) -> GateResult:
    """
    Channels at their rates, every camera frame, every recorded frame, the blueprint, the event log, and an MP4 holding
    every rendered frame.

    :param images: Camera frames logged per channel.
    :param expected_images: Frames each camera rendered for the video.
    :param images_dropped: Frames the streams lost.
    :param video_fps: Camera video rate in the recording.
    :param real_frames: Recorded frames logged per camera.
    :param real_frames_expected: Recorded frames per camera captured in the replay's span.
    :param fps: The MP4's frame rate.
    """

    failures: list[str] = []
    if ticks_3d < thresholds.min_tick_share * expected_ticks_3d:
        failures.append(f"3D ticks {ticks_3d} < {thresholds.min_tick_share:.0%} of {expected_ticks_3d}")
    if rate_3d_hz < thresholds.min_telemetry_hz:
        failures.append(f"3D telemetry at {rate_3d_hz:.1f} Hz < {thresholds.min_telemetry_hz:g} Hz")
    if events == 0:
        failures.append("no events")
    if not blueprint:
        failures.append("no blueprint")
    if not images:
        failures.append("no camera video")
    for channel, count in images.items():
        if count < expected_images:
            failures.append(f"{channel} logged {count} of {expected_images} frames")
    if images_dropped:
        failures.append(f"{images_dropped} camera frames dropped")
    if video_fps < thresholds.min_video_fps:
        failures.append(f"camera video at {video_fps:g} fps < {thresholds.min_video_fps:g} fps")
    for camera, expected in real_frames_expected.items():
        if real_frames.get(camera, 0) != expected:
            failures.append(f"{camera} recorded frames logged {real_frames.get(camera, 0)} of {expected} in the replay's span")
    if clip_frames is None:
        failures.append("no MP4")
    elif clip_frames != rendered_frames:
        failures.append(f"MP4 holds {clip_frames} of {rendered_frames} rendered frames")
    elif clip_duration_s is None or abs(clip_duration_s - clip_frames / fps) > 2.0 / fps:
        failures.append(f"MP4 duration {clip_duration_s} s for {clip_frames} frames at {fps} fps")
    measurements: dict[str, float | int | str | bool | None] = {
        "ticks_3d": ticks_3d,
        "expected_ticks_3d": expected_ticks_3d,
        "rate_3d_hz": round(rate_3d_hz, 3),
        **{f"frames_{channel.replace('/', '_')}": count for channel, count in images.items()},
        "expected_frames_per_camera": expected_images,
        "frames_dropped": images_dropped,
        "video_fps": video_fps,
        **{f"real_frames_{camera}": real_frames.get(camera, 0) for camera in real_frames_expected},
        **{f"real_frames_expected_{camera}": expected for camera, expected in real_frames_expected.items()},
        "events_logged": events,
        "blueprint": blueprint,
        "clip_frames": clip_frames,
        "rendered_frames": rendered_frames,
        "clip_duration_s": clip_duration_s,
    }
    detail = (
        f"3D {ticks_3d}/{expected_ticks_3d} at {rate_3d_hz:.1f} Hz, video {images} of {expected_images} each at {video_fps:g} fps "
        f"({images_dropped} dropped), recorded {real_frames} of {real_frames_expected}, events {events}, MP4 {clip_frames}/{rendered_frames} frames"
    )
    if failures:
        detail = f"FAILED ({'; '.join(failures)}) | {detail}"
    return GateResult(name="telemetry completeness", passed=not failures, detail=detail, measurements=measurements)


__all__ = [
    "GATE_VERSION",
    "THRESHOLDS",
    "DropTest",
    "GateResult",
    "GateThresholds",
    "MotionSamples",
    "evaluate_invariants",
    "evaluate_motion",
    "evaluate_telemetry",
    "net_touches",
]
