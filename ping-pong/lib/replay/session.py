"""
A recorded HIM session as the replay reads it: one zip holding a JSON manifest and an npz of arrays.

``package.json`` is the :class:`SessionPackage` manifest (frozen, validated, unknown fields forbidden): the session,
its provenance and time base, the camera calibration epochs, the two players' motion tracks, the serve schedule, every
stroke with its reconstructed paddle contact, the empirical shot distributions, and the recorded Orbbec frames.
``arrays.npz`` holds every array the manifest names, under fixed keys (:func:`track_keys`, :func:`distribution_key`,
:data:`FRAME_BYTES_KEY`); no pickled objects.

Conventions:

- **Frame.** The table frame: origin at the playing-surface centre, +Z up, +X towards the 8E camera's half, +Y away
  from the cameras; metres, radians; the floor at ``z = -0.76``.
- **Time.** Session seconds on the overhead cameras' clock. Every time in the package uses it.
- **Rotations.** Quaternions ``(w, x, y, z)``, Hamilton, active.
- **Roles.** ``suited`` is the motion-captured player, ``opponent`` the other player, whose motion is inferred from
  the overhead video. An end is ``8E`` (``x > 0``) or ``8F`` (``x < 0``).
- **Tracks.** Joint names are the body's CC3 names. The root joint (``Hip``) travels in the ``root_position`` and
  ``root_rotation`` arrays; every other listed joint carries its local rotation; unlisted joints keep the rest pose.
- **Strokes.** A stroke's time is the paddle-ball contact. Within a point the hitters alternate, starting with the
  receiver.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MANIFEST_NAME: str = "package.json"
ARRAYS_NAME: str = "arrays.npz"
FRAME_BYTES_KEY: str = "frames/jpeg_bytes"
FRAME_OFFSETS_KEY: str = "frames/jpeg_offsets"
QUATERNION_NORM_TOLERANCE: float = 1e-3

Role = Literal["suited", "opponent"]
End = Literal["8E", "8F"]
ROLES: tuple[Role, Role] = ("suited", "opponent")
END_SIGN: dict[str, float] = {"8E": 1.0, "8F": -1.0}
"""The half each end plays from: +1 for ``x > 0``."""

_FROZEN = ConfigDict(frozen=True, extra="forbid")
Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]


def other_role(role: Role) -> Role:
    """The other player."""

    return "opponent" if role == "suited" else "suited"


class SessionRef(BaseModel):
    """
    Which recorded session the package reconstructs.

    :ivar index: Session number, 0 (the test take) to 7.
    :ivar camera_key: The Orbbec recording key, e.g. ``215350`` or ``220448_60fps``.
    :ivar mocap_clip: The Rokoko clip, e.g. ``Clip_3``.
    :ivar suited_end: The suited player's end.
    :ivar duration_s: Session length on the time base, s.
    """

    model_config = _FROZEN

    index: int = Field(ge=0, le=7)
    camera_key: str = Field(min_length=1)
    mocap_clip: str | None = None
    suited_end: End
    duration_s: float = Field(gt=0.0)


class Provenance(BaseModel):
    """
    Where the package came from.

    :ivar kind: ``recorded`` (built from the capture) or ``synthetic`` (generated; never evidence about the recording).
    :ivar producers: Tools that wrote it, with versions.
    :ivar created_utc: ISO 8601 creation time.
    :ivar inputs: Input name to content hash or version, so a run record traces back to its data.
    :ivar notes: Free text, e.g. known gaps.
    """

    model_config = _FROZEN

    kind: Literal["recorded", "synthetic"]
    producers: tuple[str, ...] = Field(min_length=1)
    created_utc: str = Field(min_length=1)
    inputs: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class TimeBase(BaseModel):
    """
    The package's single clock.

    :ivar clock: What session time zero is, in words.
    :ivar t0_unix_s: Unix time of session time zero, s, when known.
    :ivar alignment_uncertainty_s: One-sigma uncertainty of the motion tracks against the camera clock, s.
    """

    model_config = _FROZEN

    clock: str = Field(min_length=1)
    t0_unix_s: float | None = None
    alignment_uncertainty_s: float = Field(ge=0.0)


class CameraEpochRef(BaseModel):
    """
    One overhead camera's extrinsic epoch over a stretch of the session.

    :ivar label: ``8E`` or ``8F``.
    :ivar serial: Device serial.
    :ivar epoch: Epoch label in the calibration set, e.g. ``8F-B``.
    :ivar start_s: First session time the epoch covers, s.
    :ivar end_s: Last session time it covers, s.
    """

    model_config = _FROZEN

    label: End
    serial: str = Field(min_length=1)
    epoch: str = Field(min_length=1)
    start_s: float
    end_s: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s <= self.start_s:
            raise ValueError(f"camera epoch {self.epoch} ends at {self.end_s} s, not after its start {self.start_s} s")
        return self


class CalibrationRefs(BaseModel):
    """
    The camera calibration the package was built against.

    :ivar source: Calibration file or asset, with its version.
    :ivar sha256: Content hash of that file, when it is a file.
    :ivar epochs: Every camera epoch that covers part of the session.
    """

    model_config = _FROZEN

    source: str = Field(min_length=1)
    sha256: str | None = None
    epochs: tuple[CameraEpochRef, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _both_cameras(self) -> Self:
        labels = {epoch.label for epoch in self.epochs}
        if labels != {"8E", "8F"}:
            raise ValueError(f"calibration epochs must cover both cameras, got {sorted(labels)}")
        return self

    def epoch_at(self, label: End, time_s: float) -> CameraEpochRef:
        """
        The epoch of one camera at a session time (the nearest one when none covers it).

        :param label: ``8E`` or ``8F``.
        :param time_s: Session time, s.
        :return: The epoch.
        """

        candidates = [epoch for epoch in self.epochs if epoch.label == label]
        covering = [epoch for epoch in candidates if epoch.start_s <= time_s <= epoch.end_s]
        if covering:
            return covering[0]
        return min(candidates, key=lambda epoch: min(abs(time_s - epoch.start_s), abs(time_s - epoch.end_s)))


def _unit(vector: Vec3, name: str) -> Vec3:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm < 1e-9:
        raise ValueError(f"{name} has zero length")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


class GripSpec(BaseModel):
    """
    How the paddle sits in the hand, in the hand joint's frame (body metres, before the body scale).

    :ivar hand_joint: The paddle hand's joint, e.g. ``R_Hand``.
    :ivar blade_centre_m: Blade centre (mid-thickness) in the hand joint frame.
    :ivar handle_axis: Unit direction from the handle towards the blade tip (the paddle asset's +Y).
    :ivar forehand_normal: Unit forehand-rubber normal (the paddle asset's +Z).
    :ivar source: ``default`` (rest-pose shakehand rule), ``recommended`` (the retarget's scripted grip)
        or ``imu-fit`` (orientation fitted to the paddle IMU).
    """

    model_config = _FROZEN

    hand_joint: str = Field(min_length=1)
    blade_centre_m: Vec3
    handle_axis: Vec3
    forehand_normal: Vec3
    source: Literal["default", "recommended", "imu-fit"]

    @field_validator("handle_axis", "forehand_normal")
    @classmethod
    def _normalised(cls, value: Vec3) -> Vec3:
        return _unit(value, "grip axis")

    @model_validator(mode="after")
    def _near_orthogonal(self) -> Self:
        dot = sum(a * b for a, b in zip(self.handle_axis, self.forehand_normal, strict=True))
        if abs(dot) > 0.2:
            raise ValueError(f"grip handle axis and forehand normal are {math.degrees(math.acos(max(-1.0, min(1.0, dot)))):.1f} deg apart, not near 90")
        return self


class CharacterSpec(BaseModel):
    """
    The body that plays one role.

    :ivar asset: Catalog asset name.
    :ivar version: Pinned catalog version.
    :ivar look: Looks variant, or ``None``.
    :ivar body_scale: Uniform scale on top of the asset's centimetre units (0.9255 matches the suited player's hip height).
    :ivar handedness: Paddle hand.
    :ivar grip: Paddle in the hand.
    """

    model_config = _FROZEN

    asset: str = Field(min_length=1)
    version: str = Field(min_length=1)
    look: str | None = None
    body_scale: float = Field(gt=0.5, lt=1.5)
    handedness: Literal["right", "left"]
    grip: GripSpec

    @property
    def body(self) -> str:
        """``asset@version``."""

        return f"{self.asset}@{self.version}"


class MotionTrack(BaseModel):
    """
    One character's motion on the session time base.

    :ivar role: Which player.
    :ivar end: Which end the player plays from.
    :ivar source: ``mocap`` (retargeted capture), ``procedural`` (assembled opponent) or ``synthetic``.
    :ivar character: The body and grip.
    :ivar joints: Driven joints (CC3 names), excluding the root ``Hip``.
    :ivar fps: Uniform frame rate, Hz.
    :ivar start_s: Session time of frame 0, s.
    :ivar frame_count: Frames in every array of the track.
    :ivar qa: Producer's motion-QA numbers (name to value), recorded with every run.
    :ivar translated_joints: Joints (a subset of :attr:`joints`) whose local translation is animated too, e.g.
        lengthened arm bones; their ``local_translation_m`` array is ``(F, len(translated_joints), 3)``, metres at
        body scale 1 in the parent joint's frame, like the rest skeleton's offsets. Empty: rest offsets.
    """

    model_config = _FROZEN

    role: Role
    end: End
    source: Literal["mocap", "procedural", "synthetic"]
    character: CharacterSpec
    joints: tuple[str, ...] = Field(min_length=1)
    fps: float = Field(gt=0.0, le=480.0)
    start_s: float
    frame_count: int = Field(ge=2)
    qa: dict[str, float] = Field(default_factory=dict)
    translated_joints: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _translated_joints_are_track_joints(self) -> Self:
        unknown = sorted(set(self.translated_joints) - set(self.joints))
        if unknown or len(set(self.translated_joints)) != len(self.translated_joints):
            raise ValueError(f"translated joints must be distinct track joints; not track joints: {unknown}")
        return self

    @field_validator("joints")
    @classmethod
    def _no_root_no_duplicates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if "Hip" in value:
            raise ValueError("the root joint Hip travels in root_position/root_rotation, not in joints")
        if len(set(value)) != len(value):
            raise ValueError("track joints repeat a name")
        return value

    @property
    def end_s(self) -> float:
        """Session time of the last frame, s."""

        return self.start_s + (self.frame_count - 1) / self.fps


class ServeState(BaseModel):
    """
    A fitted ball state just after a paddle contact (a point's first contact, or any recorded stroke).

    :ivar position_m: Ball centre, table frame.
    :ivar velocity_mps: Ball velocity.
    :ivar spin_radps: Ball angular velocity (usually a prior, the data barely constrain it).
    :ivar fit_rms_px: Reprojection RMS of the fit, px.
    :ivar detections: Detections the fit used.
    :ivar extrapolated_s: How far the fitted flight was propagated (backwards) from its first detection to the contact, s.
    """

    model_config = _FROZEN

    position_m: Vec3
    velocity_mps: Vec3
    spin_radps: Vec3
    fit_rms_px: float | None = None
    detections: int | None = None
    extrapolated_s: float = Field(default=0.0, ge=0.0)


class AdaptationEnvelope(BaseModel):
    """
    How uncertain a stroke's reference is, per adaptation axis: the one-sigma the evidence-derived bounds scale.

    :ivar time_sd_s: Contact timing: the reconstructed contact time's sd, the mocap time map's local sd and the
        IMU-to-camera offset's sd, in quadrature, s.
    :ivar position_sd_m: The reconstructed contact point's sd, m.
    :ivar normal_sd_deg: The reconstructed face normal's sd, degrees.
    :ivar source: ``stroke`` (this stroke's reconstruction) or ``session`` (the session's medians, for a stroke
        without one).
    """

    model_config = _FROZEN

    time_sd_s: float = Field(gt=0.0)
    position_sd_m: float = Field(gt=0.0)
    normal_sd_deg: float = Field(gt=0.0)
    source: Literal["stroke", "session"]


class ScheduledServe(BaseModel):
    """
    How one point's ball comes into play.

    :ivar index: Point number, from 0.
    :ivar time_s: The first paddle contact, session time, s.
    :ivar server: Who strikes it.
    :ivar kind: ``serve`` (it must bounce on the server's half first) or ``rally_start``
        (the recording picks the point up mid-rally: the first shot is an ordinary stroke).
    :ivar fitted: The fitted post-contact ball state, when the data support one.
    :ivar source: ``observed``, ``inferred`` or ``synthetic``.
    :ivar observed_end: How the recorded point ended, in the tracker's words, when known.
    :ivar landing: Where the first shot's ball was observed to land on the far half, when it was.
    """

    model_config = _FROZEN

    index: int = Field(ge=0)
    time_s: float
    server: Role
    kind: Literal["serve", "rally_start"] = "serve"
    fitted: ServeState | None = None
    source: Literal["observed", "inferred", "synthetic"]
    observed_end: str | None = None
    landing: LandingZone | None = None


class LandingZone(BaseModel):
    """
    Where a stroke's ball landed (or should land) on the far half.

    :ivar centre_m: Table-frame ``(x, y)``.
    :ivar radius_m: Radius of the zone, m.
    """

    model_config = _FROZEN

    centre_m: Vec2
    radius_m: float = Field(gt=0.0, le=1.0)


class PaddleContactTarget(BaseModel):
    """
    The paddle state at a stroke's contact, reconstructed from the ball.

    The reference impact model inverted on the fitted incoming and outgoing flights: where the face was, which way it
    faced and how it moved when it struck. The stroke controller measures its adaptation from this state, not from
    the raw motion-capture hand, so the human-plausible bounds apply to what the simulation changes, not to the
    capture's own error (which is logged as the reference correction).

    :ivar time_s: Contact time, session time, s.
    :ivar time_sd_s: Its uncertainty, s, when reconstructed.
    :ivar contact_point_m: Ball-face contact point (on the face), table frame, m.
    :ivar blade_centre_m: The blade centre (mid-thickness) at contact, when the reconstruction places it.
    :ivar normal: Unit face normal at contact, pointing out of the striking face toward the outgoing ball.
    :ivar velocity_mps: Paddle velocity at the contact point, m/s.
    :ivar angular_velocity_radps: Paddle angular velocity, when resolved.
    :ivar contact_point_sd_m: One-sigma uncertainty of the contact point, m.
    :ivar normal_sd_deg: One-sigma uncertainty of the normal's direction, degrees.
    :ivar velocity_sd_mps: One-sigma uncertainty of the velocity, m/s.
    :ivar source: Producer and version.
    """

    model_config = _FROZEN

    time_s: float
    time_sd_s: float | None = Field(default=None, ge=0.0)
    contact_point_m: Vec3
    blade_centre_m: Vec3 | None = None
    normal: Vec3
    velocity_mps: Vec3
    angular_velocity_radps: Vec3 | None = None
    contact_point_sd_m: float | None = Field(default=None, ge=0.0)
    normal_sd_deg: float | None = Field(default=None, ge=0.0)
    velocity_sd_mps: float | None = Field(default=None, ge=0.0)
    source: str = ""

    @field_validator("normal")
    @classmethod
    def _unit(cls, value: Vec3) -> Vec3:
        length = float(np.linalg.norm(value))
        if abs(length - 1.0) > 1e-3:
            raise ValueError(f"the face normal must be a unit vector, got length {length:.4f}")
        return value


class StrokeTarget(BaseModel):
    """
    One stroke of a rally.

    :ivar index: Stroke number across the session, from 0.
    :ivar point: The point (serve index) it belongs to.
    :ivar time_s: Paddle-ball contact, session time, s.
    :ivar hitter: Who hits.
    :ivar landing: Observed landing zone on the far half, or ``None`` (sampled from the distributions).
    :ivar kind: Stroke type, when known.
    :ivar source: ``observed``, ``inferred`` or ``synthetic``.
    :ivar fitted: The tracker's fitted ball state just after this contact, when a flight fit starts close
        enough to it; a continuity ball replays it (labelled, never physics evidence).
    :ivar paddle: The reconstructed paddle state at contact, when available: the reference the stroke
        controller adapts from.
    :ivar observed_ball_m: The tracker's incoming ball centre at the contact, when observed but not reconstructed
        into a paddle state (a point start is placed to meet it).
    :ivar envelope: The reference's uncertainty per adaptation axis (evidence-derived bounds).
    """

    model_config = _FROZEN

    index: int = Field(ge=0)
    point: int = Field(ge=0)
    time_s: float
    hitter: Role
    landing: LandingZone | None = None
    kind: Literal["forehand", "backhand", "unknown"] = "unknown"
    source: Literal["observed", "inferred", "synthetic"]
    fitted: ServeState | None = None
    paddle: PaddleContactTarget | None = None
    observed_ball_m: Vec3 | None = None
    envelope: AdaptationEnvelope | None = None


class DistributionRef(BaseModel):
    """
    One empirical shot distribution, stored as samples.

    :ivar name: Key name, e.g. ``landing_xy_8F`` or ``speed_at_net_mps``.
    :ivar unit: SI unit.
    :ivar dims: Columns per sample.
    :ivar count: Samples.
    :ivar description: What was measured and how.
    """

    model_config = _FROZEN

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    unit: str
    dims: int = Field(ge=1, le=6)
    count: int = Field(ge=1)
    description: str = ""


class RealFrameRef(BaseModel):
    """
    One recorded Orbbec colour frame carried for telemetry comparison.

    :ivar index: Position in the JPEG buffer.
    :ivar label: Camera, ``8E`` or ``8F``.
    :ivar serial: Device serial.
    :ivar time_s: Capture time, session time, s.
    :ivar source: Where it came from (recording key and frame index).
    """

    model_config = _FROZEN

    index: int = Field(ge=0)
    label: End
    serial: str
    time_s: float
    source: str = ""


class SessionPackage(BaseModel):
    """
    The manifest of one session package.

    :ivar schema_version: :data:`SCHEMA_VERSION`.
    :ivar package_id: Stable identity, e.g. ``session-2-215350-v1``.
    :ivar session: The recorded session.
    :ivar provenance: Origin and inputs.
    :ivar time_base: The clock every time refers to.
    :ivar calibration: Camera calibration references.
    :ivar tracks: Exactly one motion track per role.
    :ivar serves: The serve schedule, one per point, in time order.
    :ivar strokes: Every stroke after the serves, in time order.
    :ivar distributions: Empirical shot distributions.
    :ivar frames: Real frames for telemetry comparison, in time order.
    :ivar envelope: The session-level reference uncertainty, for strokes without their own.
    """

    model_config = _FROZEN

    schema_version: Literal[1]
    package_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    session: SessionRef
    provenance: Provenance
    time_base: TimeBase
    calibration: CalibrationRefs
    tracks: tuple[MotionTrack, ...]
    serves: tuple[ScheduledServe, ...] = Field(min_length=1)
    strokes: tuple[StrokeTarget, ...] = ()
    distributions: tuple[DistributionRef, ...] = ()
    frames: tuple[RealFrameRef, ...] = ()
    envelope: AdaptationEnvelope | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        roles = sorted(track.role for track in self.tracks)
        if roles != sorted(ROLES):
            raise ValueError(f"a package needs exactly one track per role {ROLES}, got {roles}")
        ends = {track.role: track.end for track in self.tracks}
        if ends["suited"] != self.session.suited_end or ends["opponent"] == ends["suited"]:
            raise ValueError(f"track ends {ends} disagree with the suited end {self.session.suited_end}")
        if [serve.index for serve in self.serves] != list(range(len(self.serves))):
            raise ValueError("serve indices must run 0..n-1")
        if any(later.time_s <= earlier.time_s for earlier, later in zip(self.serves, self.serves[1:], strict=False)):
            raise ValueError("serves must be in strictly increasing time order")
        if [stroke.index for stroke in self.strokes] != list(range(len(self.strokes))):
            raise ValueError("stroke indices must run 0..n-1")
        if any(later.time_s <= earlier.time_s for earlier, later in zip(self.strokes, self.strokes[1:], strict=False)):
            raise ValueError("strokes must be in strictly increasing time order")
        for stroke in self.strokes:
            if stroke.point >= len(self.serves):
                raise ValueError(f"stroke {stroke.index} names point {stroke.point}, but there are {len(self.serves)} serves")
            serve = self.serves[stroke.point]
            if stroke.time_s <= serve.time_s:
                raise ValueError(f"stroke {stroke.index} at {stroke.time_s} s precedes its serve at {serve.time_s} s")
            if stroke.point + 1 < len(self.serves) and stroke.time_s >= self.serves[stroke.point + 1].time_s:
                raise ValueError(f"stroke {stroke.index} at {stroke.time_s} s is after the next point's serve")
        for point, serve in enumerate(self.serves):
            hitters = [stroke.hitter for stroke in self.strokes if stroke.point == point]
            expected = [other_role(serve.server) if position % 2 == 0 else serve.server for position in range(len(hitters))]
            if hitters != expected:
                raise ValueError(f"point {point}: hitters {hitters} do not alternate from the receiver of {serve.server}'s serve")
        if [frame.index for frame in self.frames] != list(range(len(self.frames))):
            raise ValueError("frame indices must run 0..n-1")
        if len({distribution.name for distribution in self.distributions}) != len(self.distributions):
            raise ValueError("distribution names repeat")
        return self

    def track(self, role: Role) -> MotionTrack:
        """The motion track of one role."""

        return next(track for track in self.tracks if track.role == role)

    def strokes_of_point(self, point: int) -> tuple[StrokeTarget, ...]:
        """The strokes after one serve, in order."""

        return tuple(stroke for stroke in self.strokes if stroke.point == point)

    def role_at(self, end: End) -> Role:
        """The role playing from one end."""

        return next(track.role for track in self.tracks if track.end == end)

    def first_points(self, count: int) -> SessionPackage:
        """
        The same package restricted to its first ``count`` points (the smoke case plays one).

        :param count: Points to keep, at least one.
        :return: A validated manifest; the arrays are unchanged.
        """

        if count < 1:
            raise ValueError(f"keep at least one point, got {count}")
        serves = self.serves[:count]
        strokes = tuple(stroke for stroke in self.strokes if stroke.point < count)
        return SessionPackage.model_validate(
            self.model_dump() | {"serves": [serve.model_dump() for serve in serves], "strokes": [stroke.model_dump() for stroke in strokes]}
        )


def track_keys(role: Role) -> dict[str, str]:
    """The array keys of one role's track."""

    return {
        "local_rotation": f"tracks/{role}/local_rotation_wxyz",
        "root_position": f"tracks/{role}/root_position_m",
        "root_rotation": f"tracks/{role}/root_rotation_wxyz",
        "local_translation": f"tracks/{role}/local_translation_m",
    }


def distribution_key(name: str) -> str:
    """The array key of one distribution."""

    return f"distributions/{name}"


@dataclass(frozen=True)
class TrackArrays:
    """
    One track's arrays, validated.

    :ivar local_rotation: ``(F, J, 4)`` local rotations of the track's joints.
    :ivar root_position: ``(F, 3)`` root (``Hip``) table-frame positions, m.
    :ivar root_rotation: ``(F, 4)`` root table-frame orientations.
    :ivar local_translation: ``(F, T, 3)`` local translations of the track's translated joints, m, or ``None``.
    """

    local_rotation: NDArray[np.float64]
    root_position: NDArray[np.float64]
    root_rotation: NDArray[np.float64]
    local_translation: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class LoadedSessionPackage:
    """
    A manifest with its arrays, checked against each other.

    :ivar manifest: The validated manifest.
    :ivar arrays: Every array by key (read-only).
    :ivar sha256: Content hash of the package file (or of its in-memory bytes).
    :ivar origin: Where it was loaded from.
    """

    manifest: SessionPackage
    arrays: Mapping[str, NDArray]
    sha256: str
    origin: str

    def __post_init__(self) -> None:
        for array in self.arrays.values():
            array.setflags(write=False)
        for track in self.manifest.tracks:
            self.track_arrays(track.role)
        for distribution in self.manifest.distributions:
            self.distribution(distribution.name)
        if self.manifest.frames:
            offsets = self._frame_offsets()
            if offsets.shape != (len(self.manifest.frames) + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
                raise ValueError("frame offsets must start at 0 and increase, one more than the frames")
            if int(offsets[-1]) != self._array(FRAME_BYTES_KEY).size:
                raise ValueError("frame offsets do not end at the JPEG buffer's length")

    def _array(self, key: str) -> NDArray:
        if key not in self.arrays:
            raise ValueError(f"package {self.manifest.package_id} lacks array {key!r}")
        return self.arrays[key]

    def track_arrays(self, role: Role) -> TrackArrays:
        """
        One role's arrays, shape- and unit-checked.

        :raises ValueError: On a missing array, a wrong shape, a non-finite value or a non-unit quaternion
        """

        return validated_track_arrays(self.manifest.track(role), self._array)

    def distribution(self, name: str) -> NDArray[np.float64]:
        """One distribution's samples, ``(count, dims)``."""

        reference = next((item for item in self.manifest.distributions if item.name == name), None)
        if reference is None:
            raise KeyError(f"package {self.manifest.package_id} has no distribution {name!r}")
        samples = np.asarray(self._array(distribution_key(name)), dtype=np.float64).reshape(reference.count, reference.dims)
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"distribution {name} holds non-finite samples")
        return samples

    def _frame_offsets(self) -> NDArray[np.int64]:
        return np.asarray(self._array(FRAME_OFFSETS_KEY), dtype=np.int64)

    def frame_jpeg(self, index: int) -> bytes:
        """The encoded JPEG of one real frame."""

        offsets = self._frame_offsets()
        buffer = self._array(FRAME_BYTES_KEY)
        return bytes(np.asarray(buffer[offsets[index] : offsets[index + 1]], dtype=np.uint8).tobytes())

    def frames_of(self, label: End) -> tuple[RealFrameRef, ...]:
        """One camera's real frames, in time order."""

        return tuple(frame for frame in self.manifest.frames if frame.label == label)

    def restricted(self, points: int) -> LoadedSessionPackage:
        """This package with only its first ``points`` points (same arrays and hash)."""

        return LoadedSessionPackage(manifest=self.manifest.first_points(points), arrays=self.arrays, sha256=self.sha256, origin=self.origin)


def validated_track_arrays(track: MotionTrack, fetch: Callable[[str], NDArray]) -> TrackArrays:
    """
    One track's arrays, shape- and unit-checked.

    :param track: The track's metadata.
    :param fetch: The array of a key (raising for a missing one).
    :raises ValueError: On a wrong shape, a non-finite value or a non-unit quaternion
    """

    role = track.role
    keys = track_keys(role)
    frames, joints = track.frame_count, len(track.joints)
    local = np.asarray(fetch(keys["local_rotation"]), dtype=np.float64)
    position = np.asarray(fetch(keys["root_position"]), dtype=np.float64)
    rotation = np.asarray(fetch(keys["root_rotation"]), dtype=np.float64)
    for name, array, shape in (
        ("local_rotation", local, (frames, joints, 4)),
        ("root_position", position, (frames, 3)),
        ("root_rotation", rotation, (frames, 4)),
    ):
        if array.shape != shape:
            raise ValueError(f"track {role} {name} has shape {array.shape}, expected {shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"track {role} {name} holds non-finite values")
    for name, array in (("local_rotation", local), ("root_rotation", rotation)):
        norm_error = float(np.max(np.abs(np.linalg.norm(array, axis=-1) - 1.0)))
        if norm_error > QUATERNION_NORM_TOLERANCE:
            raise ValueError(f"track {role} {name} has quaternions {norm_error:.2e} away from unit length")
    translation = None
    if track.translated_joints:
        translation = np.asarray(fetch(keys["local_translation"]), dtype=np.float64)
        if translation.shape != (frames, len(track.translated_joints), 3):
            raise ValueError(f"track {role} local_translation has shape {translation.shape}, expected {(frames, len(track.translated_joints), 3)}")
        if not np.all(np.isfinite(translation)):
            raise ValueError(f"track {role} local_translation holds non-finite values")
    return TrackArrays(local_rotation=local, root_position=position, root_rotation=rotation, local_translation=translation)


def decode_session_package(data: bytes, *, origin: str) -> LoadedSessionPackage:
    """
    Read and validate a package from its zip bytes.

    :param data: Zip bytes.
    :param origin: Where they came from, recorded with the run.
    :return: The loaded package.
    :raises ValueError: If the archive lacks a member, the manifest is invalid, or an array disagrees with it
    """

    with zipfile.ZipFile(io.BytesIO(data)) as handle:
        names = set(handle.namelist())
        if names != {MANIFEST_NAME, ARRAYS_NAME}:
            raise ValueError(f"a session package holds exactly {MANIFEST_NAME} and {ARRAYS_NAME}, found {sorted(names)}")
        manifest = SessionPackage.model_validate(json.loads(handle.read(MANIFEST_NAME)))
        with np.load(io.BytesIO(handle.read(ARRAYS_NAME)), allow_pickle=False) as npz:
            arrays = {key: np.asarray(npz[key]) for key in npz.files}
    return LoadedSessionPackage(manifest=manifest, arrays=arrays, sha256=hashlib.sha256(data).hexdigest(), origin=origin)


def read_session_package(path: Path | str) -> LoadedSessionPackage:
    """Read and validate a package file."""

    return decode_session_package(Path(path).read_bytes(), origin=str(path))


_ASSET_SOURCE = re.compile(r"^asset:(?P<name>[^@]+)@(?P<version>[^@]+)$")


def load_session_package(source: str) -> LoadedSessionPackage:
    """
    Load a session package.

    :param source: ``asset:<name>@<version>`` (fetched with ``antioch.fetch_asset``) or a local file path.
    :return: The validated package.
    :raises ValueError: For an asset source without a pinned version
    """

    match = _ASSET_SOURCE.match(source)
    if match is not None:
        import antioch

        path = antioch.fetch_asset(match["name"], version=match["version"])
        return decode_session_package(Path(path).read_bytes(), origin=source)
    if source.startswith("asset:"):
        raise ValueError(f"package source {source!r} must pin a version: asset:<name>@<version>")
    return read_session_package(source)


__all__ = [
    "END_SIGN",
    "FRAME_BYTES_KEY",
    "FRAME_OFFSETS_KEY",
    "ROLES",
    "AdaptationEnvelope",
    "CalibrationRefs",
    "CameraEpochRef",
    "CharacterSpec",
    "DistributionRef",
    "End",
    "GripSpec",
    "LandingZone",
    "LoadedSessionPackage",
    "MotionTrack",
    "PaddleContactTarget",
    "Provenance",
    "RealFrameRef",
    "Role",
    "ScheduledServe",
    "ServeState",
    "SessionPackage",
    "SessionRef",
    "StrokeTarget",
    "TimeBase",
    "TrackArrays",
    "decode_session_package",
    "distribution_key",
    "load_session_package",
    "other_role",
    "read_session_package",
    "track_keys",
    "validated_track_arrays",
]
