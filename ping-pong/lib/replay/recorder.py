"""
The replay's recording: the simulated cameras beside the real ones at 24 fps, the 3D state, the event log and the MP4.

Everything goes under one ``antioch.Logger`` prefix (``tt``):

- ``world/...``: the table, net and floor; the ball with its one-second trail and the stroke controller's predicted
  path; each paddle's blade and face normal; contact normals; both skeletons; the cameras as pinholes;
- ``scalars/...``: ball speed, height and spin; paddle speeds; each stroke's adaptation; the loop's cost; motion QA;
- ``events``: serves, contacts, point endings, continuity balls and plans;
- ``cams/orbbec_8E/sim``, ``cams/orbbec_8F/sim`` and ``cams/broadcast``: every rendered frame (24 fps, H.264), each at
  the simulation time it shows, with the simulated ball reprojected through the camera's calibration on ``.../ball``;
- ``cams/orbbec_8E/real`` and ``cams/orbbec_8F/real``: every recorded Orbbec frame of the replay's span at its own
  capture time (about 19-25 fps, never resampled).

The compositor's MP4 carries the broadcast camera at full quality. The recorder observes and never commands, and it
counts what it logged so the replay can grade its own telemetry.
"""

from __future__ import annotations

import contextlib
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import antioch
import numpy as np
from numpy.typing import NDArray

from lib.him_lab import PinholeSpec
from lib.physics.engine.contacts import ContactEvent
from lib.telemetry.cameras import CameraRig
from lib.telemetry.compositor import CompositorClient, FrameSpool
from lib.telemetry.video import CameraStreamSink, StreamStats, log_at

REPLAY_LOGGER = antioch.Logger("tt")
PREFIX: str = "/tt"
VIDEO_FPS: float = 24.0
SINK_QUEUE_FRAMES: int = 48
"""Two seconds of video per camera between the step loop and its encoder."""
SINK_BLOCK_S: float = 30.0
"""How long a full queue may hold the step loop before a frame would drop: never, in practice (a drop fails the test)."""

SKELETON_JOINTS: tuple[str, ...] = (
    "Hip",
    "Waist",
    "Spine01",
    "Spine02",
    "NeckTwist01",
    "Head",
    "L_Clavicle",
    "L_Upperarm",
    "L_Forearm",
    "L_Hand",
    "R_Clavicle",
    "R_Upperarm",
    "R_Forearm",
    "R_Hand",
    "L_Thigh",
    "L_Calf",
    "L_Foot",
    "L_ToeBase",
    "R_Thigh",
    "R_Calf",
    "R_Foot",
    "R_ToeBase",
)
SKELETON_BONES: tuple[tuple[str, str], ...] = (
    ("Hip", "Waist"),
    ("Waist", "Spine01"),
    ("Spine01", "Spine02"),
    ("Spine02", "NeckTwist01"),
    ("NeckTwist01", "Head"),
    ("Spine02", "L_Clavicle"),
    ("L_Clavicle", "L_Upperarm"),
    ("L_Upperarm", "L_Forearm"),
    ("L_Forearm", "L_Hand"),
    ("Spine02", "R_Clavicle"),
    ("R_Clavicle", "R_Upperarm"),
    ("R_Upperarm", "R_Forearm"),
    ("R_Forearm", "R_Hand"),
    ("Hip", "L_Thigh"),
    ("L_Thigh", "L_Calf"),
    ("L_Calf", "L_Foot"),
    ("L_Foot", "L_ToeBase"),
    ("Hip", "R_Thigh"),
    ("R_Thigh", "R_Calf"),
    ("R_Calf", "R_Foot"),
    ("R_Foot", "R_ToeBase"),
)
ROLE_COLOURS: dict[str, tuple[int, int, int]] = {"suited": (30, 30, 30), "opponent": (40, 70, 160)}
SURFACE_COLOURS: dict[str, tuple[int, int, int]] = {
    "paddle": (220, 30, 30),
    "table": (60, 140, 255),
    "net": (240, 240, 240),
    "floor": (130, 130, 130),
    "other": (200, 200, 0),
}
BALL_COLOUR: tuple[int, int, int] = (255, 120, 20)
CONTINUITY_COLOUR: tuple[int, int, int] = (190, 60, 220)
TRAIL_S: float = 1.0


@dataclass
class TelemetryCounts:
    """
    What the recorder logged, for the telemetry test.

    :ivar ticks_3d: 3D and scalar ticks logged.
    :ivar images: Camera frames logged per channel (filled when the sinks close).
    :ivar video: Each camera stream's accounting.
    :ivar real_frames: Recorded frames logged per Orbbec camera.
    :ivar events: Text events logged.
    :ivar contacts: Contact arrows logged.
    :ivar pops: Motion-QA pops logged.
    :ivar blueprint: Whether the blueprint was sent.
    """

    ticks_3d: int = 0
    images: dict[str, int] = field(default_factory=dict)
    video: dict[str, StreamStats] = field(default_factory=dict)
    real_frames: dict[str, int] = field(default_factory=dict)
    events: int = 0
    contacts: int = 0
    pops: int = 0
    blueprint: bool = False


def _stream(channel: str, crf: int) -> CameraStreamSink:
    return CameraStreamSink(REPLAY_LOGGER, channel, VIDEO_FPS, crf=crf, queue_frames=SINK_QUEUE_FRAMES, block_s=SINK_BLOCK_S)


class Recorder:
    """
    One replay's recording: the cameras' render products, their video streams, the 3D channels and the MP4 spool.

    :param cameras: Every camera's pinhole, by name (the Orbbecs and ``broadcast``).
    :param roles: The players.
    :param orbbecs: The virtual Orbbecs' render products.
    :param broadcast: The broadcast camera's render product.
    :param directory: Where the run's reports and the clip go.
    :param compositor: The MP4 encoder, or ``None`` without a clip.
    """

    def __init__(
        self,
        cameras: dict[str, PinholeSpec],
        roles: tuple[str, ...],
        orbbecs: CameraRig,
        broadcast: CameraRig,
        *,
        directory: Path,
        compositor: CompositorClient | None = None,
    ) -> None:
        self.cameras = cameras
        self.roles = roles
        self.orbbecs = orbbecs
        self.broadcast = broadcast
        self.directory = directory
        self.compositor = compositor
        self.spool = FrameSpool(directory / "clip") if compositor is not None else None
        self.tunnels: list[NDArray[np.float64]] = []
        self.counts = TelemetryCounts()
        self.trail: deque[tuple[float, NDArray[np.float64]]] = deque()
        self.recent_contacts: deque[ContactEvent] = deque(maxlen=12)
        self.sinks: dict[str, Any] = {}
        self.real_sinks: dict[str, Any] = {}

    # ---- once per run ---------------------------------------------------------------------------------------------

    def send_blueprint(self, run: antioch.ScenarioRun) -> None:
        """Set the viewer layout (before stepping)."""

        run.set_blueprint(replay_blueprint(tuple(self.cameras)))
        self.counts.blueprint = True

    def log_static(self, table: tuple[float, float, float], net_height: float, net_half_span: float) -> None:
        """The table, net, floor and camera pinholes; ``table`` is (length, width, height), m."""

        import rerun as rr

        length, width, height = table
        REPLAY_LOGGER.value("world/table", rr.Boxes3D(centers=[[0.0, 0.0, -0.0125]], half_sizes=[[length / 2, width / 2, 0.0125]], colors=[[40, 70, 120]]))
        hl, hw = length / 2, width / 2
        lines = [[[-hl, -hw, 0.001], [hl, -hw, 0.001], [hl, hw, 0.001], [-hl, hw, 0.001], [-hl, -hw, 0.001]], [[-hl, 0.0, 0.001], [hl, 0.0, 0.001]]]
        REPLAY_LOGGER.value("world/table/lines", rr.LineStrips3D(lines, colors=[[240, 240, 240]], radii=[0.004]))
        REPLAY_LOGGER.value(
            "world/net",
            rr.LineStrips3D(
                [[[0.0, -net_half_span, 0.0], [0.0, -net_half_span, net_height], [0.0, net_half_span, net_height], [0.0, net_half_span, 0.0]]],
                colors=[[230, 230, 230]],
                radii=[0.004],
            ),
        )
        REPLAY_LOGGER.value("world/floor", rr.Boxes3D(centers=[[0.0, 0.0, -height - 0.005]], half_sizes=[[4.5, 3.5, 0.005]], colors=[[160, 160, 155, 90]]))
        for name, camera in self.cameras.items():
            rotation = np.asarray(camera.rotation)
            centre = -rotation.T @ np.asarray(camera.translation)
            REPLAY_LOGGER.value(f"world/cameras/{name}", rr.Transform3D(translation=centre.tolist(), mat3x3=rotation.T.tolist()))
            REPLAY_LOGGER.value(
                f"world/cameras/{name}",
                rr.Pinhole(
                    focal_length=[camera.fx, camera.fy],
                    principal_point=[camera.cx + 0.5, camera.cy + 0.5],
                    width=camera.width,
                    height=camera.height,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.35,
                ),
            )
        REPLAY_LOGGER.value("scalars/ball", rr.SeriesLines(names=["speed m/s", "height m", "spin rev/s"]))
        REPLAY_LOGGER.value("scalars/paddle_speed", rr.SeriesLines(names=list(self.roles)))
        REPLAY_LOGGER.value("scalars/adaptation", rr.SeriesPoints(names=["time warp ms", "offset cm", "tilt deg"]))
        REPLAY_LOGGER.value("scalars/cost", rr.SeriesLines(names=["wall s per sim s", "tick ms", "physics ms per step"]))
        for role in self.roles:
            REPLAY_LOGGER.value(f"scalars/motion_qa/{role}", rr.SeriesLines(names=["largest joint turn deg/frame", "arm shortfall cm"]))

    # ---- every telemetry tick -------------------------------------------------------------------------------------

    def log_tick(
        self,
        *,
        time_s: float,
        ball: NDArray[np.float64],
        ball_in_play: bool,
        continuity: bool,
        prediction: NDArray[np.float64] | None,
        paddles: dict[str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
        skeletons: dict[str, NDArray[np.float64]],
        cost: tuple[float, float, float],
    ) -> None:
        """
        The 3D state and scalars at one tick.

        :param time_s: Session time, s.
        :param ball: Ball state (position, velocity, spin).
        :param ball_in_play: Whether the ball is live (a parked ball is not drawn).
        :param continuity: Whether the live ball is a continuity ball.
        :param prediction: The current plan's predicted ball path, ``(N, 3)``.
        :param paddles: Per role ``(position, rotation, velocity)``.
        :param skeletons: Per role the joint positions in :data:`SKELETON_JOINTS` order.
        :param cost: ``(wall s per sim s, tick ms, physics ms per step)``.
        """

        import rerun as rr

        position = ball[:3]
        if ball_in_play:
            colour = CONTINUITY_COLOUR if continuity else BALL_COLOUR
            REPLAY_LOGGER.value("world/ball", rr.Points3D([position], radii=[0.02], colors=[colour]))
            self.trail.append((time_s, position.copy()))
        else:
            REPLAY_LOGGER.value("world/ball", rr.Clear(recursive=False))
            self.trail.clear()
        while self.trail and time_s - self.trail[0][0] > TRAIL_S:
            self.trail.popleft()
        if len(self.trail) >= 2:
            REPLAY_LOGGER.value("world/ball/trail", rr.LineStrips3D([np.array([p for _t, p in self.trail])], colors=[[255, 170, 60]], radii=[0.004]))
        if prediction is not None and len(prediction) >= 2:
            REPLAY_LOGGER.value("world/ball/predicted", rr.LineStrips3D([prediction], colors=[[120, 220, 255]], radii=[0.003]))
        angles = np.linspace(0.0, 2.0 * math.pi, 25)
        for role, (paddle_position, rotation, _velocity) in paddles.items():
            outline = paddle_position[None, :] + np.outer(np.cos(angles) * 0.075, rotation[:, 0]) + np.outer(np.sin(angles) * 0.0785, rotation[:, 1])
            REPLAY_LOGGER.value(f"world/paddles/{role}", rr.LineStrips3D([outline], colors=[[200, 20, 20]], radii=[0.004]))
            REPLAY_LOGGER.value(f"world/paddles/{role}/normal", rr.Arrows3D(origins=[paddle_position], vectors=[0.12 * rotation[:, 2]], colors=[[255, 80, 80]]))
        index = {name: position for position, name in enumerate(SKELETON_JOINTS)}
        for role, joints in skeletons.items():
            colour = ROLE_COLOURS.get(role, (100, 100, 100))
            REPLAY_LOGGER.value(f"world/players/{role}/joints", rr.Points3D(joints, radii=[0.018], colors=[colour]))
            REPLAY_LOGGER.value(
                f"world/players/{role}/bones",
                rr.LineStrips3D([[joints[index[a]], joints[index[b]]] for a, b in SKELETON_BONES], colors=[colour], radii=[0.008]),
            )
        speed = float(np.linalg.norm(ball[3:6])) if ball_in_play else math.nan
        REPLAY_LOGGER.value(
            "scalars/ball",
            rr.Scalars(
                [speed, float(position[2]) if ball_in_play else math.nan, float(np.linalg.norm(ball[6:9])) / (2 * math.pi) if ball_in_play else math.nan]
            ),
        )
        REPLAY_LOGGER.value(
            "scalars/paddle_speed", rr.Scalars([float(np.linalg.norm(paddles[role][2])) if role in paddles else math.nan for role in self.roles])
        )
        REPLAY_LOGGER.value("scalars/cost", rr.Scalars(list(cost)))
        self.counts.ticks_3d += 1

    def log_retired(self, positions: dict[int, NDArray[np.float64]]) -> None:
        """Retired balls still playing out (grey), by index."""

        import rerun as rr

        if positions:
            REPLAY_LOGGER.value("world/retired_balls", rr.Points3D(np.array(list(positions.values())), radii=[0.02], colors=[[150, 150, 150]]))
        else:
            REPLAY_LOGGER.value("world/retired_balls", rr.Clear(recursive=False))

    def log_tunnel(self, position: NDArray[np.float64], surface: str) -> None:
        """A tunnelling (or a ball leaving the room): a red marker where it was seen, kept for the whole replay."""

        import rerun as rr

        self.tunnels.append(np.asarray(position, dtype=np.float64))
        REPLAY_LOGGER.value(
            "world/tunnels", rr.Points3D(np.array(self.tunnels), radii=[0.03], colors=[[255, 30, 30]], labels=[f"tunnel ({surface})"] * len(self.tunnels))
        )

    def log_contacts(self, events: list[ContactEvent]) -> None:
        """Contact normals at hits and bounces (the last dozen stay drawn)."""

        import rerun as rr

        if not events:
            return
        self.recent_contacts.extend(events)
        contacts = list(self.recent_contacts)
        REPLAY_LOGGER.value(
            "world/contacts",
            rr.Arrows3D(
                origins=[event.position for event in contacts],
                vectors=[0.15 * event.normal for event in contacts],
                colors=[SURFACE_COLOURS.get(event.surface, (255, 255, 0)) for event in contacts],
                radii=[0.006],
            ),
        )
        self.counts.contacts += len(events)

    def log_motion_qa(self, role: str, step_deg: float, popped: list[tuple[str, float]], ik_miss_m: float) -> None:
        """One frame's motion QA: the largest joint turn since the last frame, the arm's shortfall, and every pop."""

        import rerun as rr

        REPLAY_LOGGER.value(f"scalars/motion_qa/{role}", rr.Scalars([step_deg, 100.0 * ik_miss_m]))
        for joint, degrees in popped:
            REPLAY_LOGGER.value("motion_qa", rr.TextLog(f"pop: {role} {joint} turned {degrees:.0f} deg in one frame", level="WARN"))
            self.counts.pops += 1

    def log_adaptation(self, time_warp_s: float, offset_m: float, tilt_rad: float) -> None:
        """One stroke plan's adaptation."""

        import rerun as rr

        REPLAY_LOGGER.value("scalars/adaptation", rr.Scalars([1e3 * time_warp_s, 1e2 * offset_m, math.degrees(tilt_rad)]))

    def event(self, text: str, level: str = "INFO") -> None:
        """One line in the event log (also printed)."""

        import rerun as rr

        REPLAY_LOGGER.value("events", rr.TextLog(text, level=level))
        print(f"[replay] {text}", flush=True)
        self.counts.events += 1

    # ---- camera video ---------------------------------------------------------------------------------------------

    @staticmethod
    def channel(name: str) -> str:
        """The simulated view's channel of a camera."""

        return "cams/broadcast" if name == "broadcast" else f"cams/{name}/sim"

    def open_video(self, factory: Callable[[str, int], Any] = _stream) -> None:
        """
        One stream per camera, and one per Orbbec for its recorded frames (before the first frame).

        :param factory: ``(channel, crf) -> sink`` (a test substitutes one).
        """

        for name, camera in self.cameras.items():
            self.sinks[name] = factory(self.channel(name), 23)
            if camera.calibrated:
                self.real_sinks[name] = factory(f"cams/{name}/real", 28)

    def log_frames(self, frames: dict[str, NDArray[np.uint8]], ball: NDArray[np.float64] | None, sim_time: float) -> None:
        """
        One rendered frame per camera at the recording time it shows, with the simulated ball reprojected.

        :param frames: Rendered frames by camera name.
        :param ball: Ball centre at that time, or ``None`` out of play.
        :param sim_time: Recording time the frames show, s.
        """

        import rerun as rr

        for name, frame in frames.items():
            sink = self.sinks.get(name)
            if sink is None:
                continue
            sink.push(frame, sim_time)
            camera = self.cameras.get(name)
            if camera is None or not camera.calibrated:
                continue
            pixel = camera.project(ball) if ball is not None else None
            overlay = rr.Points2D([pixel], radii=[5.0], colors=[[0, 255, 120]]) if pixel is not None else rr.Clear(recursive=False)
            log_at(f"{PREFIX[1:]}/{self.channel(name)}/ball", overlay, sim_time)
            log_at(f"{PREFIX[1:]}/cams/{name}/real/ball", overlay, sim_time)

    def log_real_frame(self, name: str, jpeg: bytes, sim_time: float) -> None:
        """One recorded Orbbec frame (``orbbec_8E`` or ``orbbec_8F``) at its own capture time on the recording's timeline."""

        sink = self.real_sinks.get(name)
        if sink is not None:  # a camera without a stream is caught by the telemetry test's recorded-frame count
            sink.push_jpeg(jpeg, sim_time)

    def close_video(self) -> None:
        """
        Drain every stream and count what reached the recording (idempotent).

        :raises RuntimeError: If a stream lost frames or failed
        """

        errors: list[str] = []
        for name, sink in self.sinks.items():
            stats = sink.close()
            self.counts.video[self.channel(name)] = stats
            self.counts.images[self.channel(name)] = stats.logged
            errors += [f"{stats.entity_path}: {message}" for message in stats.errors]
        for name, sink in self.real_sinks.items():
            stats = sink.close()
            self.counts.video[f"cams/{name}/real"] = stats
            self.counts.real_frames[name] = stats.logged
            errors += [f"{stats.entity_path}: {message}" for message in stats.errors]
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self) -> None:
        """Release everything: the streams (if still open), the render products and the compositor connection."""

        with contextlib.suppress(RuntimeError):
            self.close_video()
        for rig in (self.orbbecs, self.broadcast):
            rig.close()
        if self.compositor is not None:
            self.compositor.close()


def replay_blueprint(cameras: tuple[str, ...]) -> Any:
    """The layout: 3D lab and broadcast on top, recorded and simulated Orbbec views below, then series and events."""

    import rerun.blueprint as rrb

    def channel(path: str) -> str:
        return f"{PREFIX}/{path}"

    top: list[Any] = [rrb.Spatial3DView(name="Lab (3D)", origin=channel("world"), contents=[f"{channel('world')}/**"])]
    if "broadcast" in cameras:
        top.append(rrb.Spatial2DView(name="Broadcast", origin=channel("cams/broadcast")))
    orbbec: list[Any] = []
    for label in ("8F", "8E"):
        name = f"orbbec_{label}"
        if name not in cameras:
            continue
        real = rrb.Spatial2DView(name=f"{label} recorded", origin=channel(f"cams/{name}/real"))
        sim = rrb.Spatial2DView(name=f"{label} simulated", origin=channel(f"cams/{name}/sim"))
        orbbec.extend([real, sim] if label == "8F" else [sim, real])

    def series(name: str, path: str) -> Any:
        return rrb.TimeSeriesView(name=name, origin=channel(path))

    bottom = [
        series("Ball", "scalars/ball"),
        series("Paddle speed", "scalars/paddle_speed"),
        series("Stroke adaptation", "scalars/adaptation"),
        series("Cost", "scalars/cost"),
        series("Motion QA", "scalars/motion_qa"),
        rrb.TextLogView(name="Events", origin=channel("events")),
        rrb.TextLogView(name="Motion QA", origin=channel("motion_qa")),
    ]
    rows = (
        rrb.Vertical(rrb.Horizontal(*top), rrb.Horizontal(*orbbec), rrb.Horizontal(*bottom), row_shares=[3, 2, 2])
        if orbbec
        else rrb.Vertical(rrb.Horizontal(*top), rrb.Horizontal(*bottom), row_shares=[3, 2])
    )
    # Every frame carries sim_time; at 1x the camera video plays at 24 fps.
    return rrb.Blueprint(rows, rrb.TimePanel(timeline="sim_time", playback_speed=1.0, loop_mode="all"), collapse_panels=True)


__all__ = ["REPLAY_LOGGER", "SKELETON_JOINTS", "Recorder", "TelemetryCounts", "replay_blueprint"]
