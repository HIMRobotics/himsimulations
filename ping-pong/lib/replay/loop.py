"""
One recorded session replayed render tick by render tick: warm-up, drop tests, then every point.

:func:`lib.replay.run.replay_him_session` composes the lab, attaches the live ball, spawns the players and hands them
to :class:`ReplayLoop`, which:

1. **Warms up** the renderer: 900 frames with physics paused, so RTX has converged after the lighting change before
   any frame is logged.
2. **Pre-rolls** six rendered seconds before the session clock starts and runs the opening drop tests: a ball released
   with its bottom 30 cm above the table, at staggered heights so the contacts fall at different phases of the
   physics step, whose rebound must land in the ITTF band.
3. **Plays**: every render tick it applies the physics events to the rally referee, carries out what the referee
   schedules (serves, stroke plans, continuity balls), poses both players at the tick's end time, hands the physics
   the paddles' path for the tick, steps, and records the frames, telemetry and motion samples. Every rendered camera
   frame goes to the recording at the time it shows (the render products deliver the previous step's frame), and
   every recorded Orbbec frame of the replay's span at its own capture time.
4. **Stops** on the terminal condition (every point played and the last one ended physically) or the cap.

The players bend their recorded strokes to meet the live ball: each stroke is planned from the ball's predicted flight
(:func:`lib.replay.players.strokes.plan_stroke`) within human adaptation bounds, and the arm IK carries the paddle to
the plan. A point that ends before the recording did continues with a labelled **continuity ball**, never evidence.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import antioch
import numpy as np
from numpy.typing import NDArray

from lib.him_lab import isp_luts
from lib.physics.engine.body_capsules import BodyMotion, Capsules, capsules_from_joints, excluded_capsules
from lib.physics.engine.stepper import GamePhysics, PaddleSegment
from lib.physics.params import BallTableParams
from lib.physics.reference.simulate import simulate
from lib.replay.ball import PARK_POSITION_M, RENDER_DT, SPARE_PATHS, set_ball_scale, set_ball_visible
from lib.replay.gates import THRESHOLDS, DropTest, MotionSamples
from lib.replay.players import rotations as rot
from lib.replay.players.bodies import CharacterHandle
from lib.replay.players.kinematics import forward_kinematics
from lib.replay.players.playback import AnimatedPose, CharacterAnimator
from lib.replay.players.strokes import (
    EVIDENCE_K,
    FIXED_V1_BOUNDS,
    AdaptationBounds,
    ReplanAllowance,
    StrokePlan,
    evidence_v3_bounds,
    plan_stroke,
    predict,
    project_plan,
    replan_allowance,
    solve_launch,
    solve_through,
    state_at,
)
from lib.replay.recorder import SKELETON_JOINTS, Recorder
from lib.replay.referee import ContinuityAction, PlanAction, Rally, ServeAction
from lib.replay.session import END_SIGN, LoadedSessionPackage, Role, other_role

Array = NDArray[np.float64]

WARMUP_FRAMES: int = 900
"""Frames RTX needs to converge after the lighting is applied (440-900 measured), rendered before any logged frame."""
PRE_ROLL_S: float = 6.0
"""Rendered seconds before the session clock starts: the drop tests run here."""
DROP_TESTS: int = 5
LEAD_S: float = 1.5
"""Session seconds played before the first serve (the players settle into the recording)."""
TELEMETRY_EVERY: int = 2
"""Rendered frames per 3D and scalar tick (12 Hz); every rendered frame is logged as video (24 fps)."""

POP_DEG_PER_S: float = THRESHOLDS.pop_deg_per_s
"""The motion-QA pop rate, flagged per frame in the telemetry."""
DROP_HEIGHT_M: float = 0.30
DROP_STAGGER_M: float = 0.0005
"""Release-height step between drops: about a fifth of a physics step of fall time each."""
DROP_TIMEOUT_S: float = 1.2
VELOCITY_PROBE_S: float = 0.005
SERVE_COOLDOWN_S: float = RENDER_DT + 0.09375
"""After a placement the striker's blade cannot touch the ball it just released for this long (a render tick and 90 steps at 960 Hz)."""
CONTINUITY_FLIGHT_S: float = 0.45
CONTINUITY_MAX_NUDGE_MPS: float = 2.0
"""Largest change a continuity ball's recorded launch velocity may take to meet the next reconstructed contact."""
SWING_SEARCH_S: float = 0.25
"""How far from the recorded hit time a point start looks for the server's actual swing contact."""
POINT_START_MAX_NUDGE_MPS: float = 6.0
"""Largest change a point-start ball (solved from the server's paddle, or fitted) may take to meet the receiver's recorded contact."""
CONTINUITY_MAX_MISS_M: float = 0.03
"""A nudge that meets the next contact within this succeeds; a worse best effort is kept only inside the stroke's bounds."""
AIM_LAUNCH_STEP_S: float = 0.05
"""Launch-time grid when aiming a placed or replayed ball (within :data:`SWING_SEARCH_S` of the swing contact)."""
AIM_ARRIVALS: int = 5
"""Arrival times tried across the receiving stroke's time bound."""
RETIRED_OFF_CAMERA_S: float = 0.25
"""A retired ball is parked once it has been outside every camera's image for this long."""
RETIRED_FADE_S: float = 0.3
"""A retired ball that must leave the play while a camera sees it shrinks out over this (a continuity device)."""
PLAY_VOLUME_MARGIN_M: float = 0.2
"""The play volume: the table's footprint grown by this, from just below the top up to 2 m."""
TABLE_REST_SPEED_MPS: float = 0.3
"""A ball on the table top slower than this is lying there (collected between points)."""
LAUNCH_MIN_BLADE_SPEED_MPS: float = 1.0
"""A ball leaves a blade only while it swings: at least this fast (a ball launched from a slower blade looks placed)."""
LAUNCH_MIN_TOWARD_MPS: float = 0.3
"""... and moving toward the far end at least this fast."""
DEAD_BALL_PARK_S: float = 0.6
"""A dead ball is parked this long after its point ended (it would otherwise roll and bounce on the floor until the next placement)."""
CONTINUITY_TOPSPIN_RADPS: float = 60.0
RETURN_MISS_WINDOW_S: float = 0.3
"""A realised return's next-goal miss is its closest approach within this of the goal's recorded time."""


def _segment_distances(p0: Array, p1: Array, a: Array, b: Array) -> Array:
    """Shortest distances between the segment ``p0 p1`` and each segment ``a_i b_i`` ``(N, 3)``."""

    d1 = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    d2 = b - a
    r = np.asarray(p0, dtype=np.float64)[None] - a
    aa = float(d1 @ d1)
    ee = np.sum(d2 * d2, axis=1)
    ff = np.sum(r * d2, axis=1)
    cc = r @ d1
    bb = d2 @ d1
    denom = aa * ee - bb * bb
    s = np.where(denom > 1e-12, np.clip((bb * ff - cc * ee) / np.where(denom > 1e-12, denom, 1.0), 0.0, 1.0), 0.0)
    t = np.where(ee > 1e-12, (bb * s + ff) / np.where(ee > 1e-12, ee, 1.0), 0.0)
    t_clipped = np.clip(t, 0.0, 1.0)
    s = np.where(t != t_clipped, np.clip((t_clipped * bb - cc) / max(aa, 1e-12), 0.0, 1.0), s)
    closest_1 = np.asarray(p0, dtype=np.float64)[None] + s[:, None] * d1[None]
    closest_2 = a + t_clipped[:, None] * d2
    return np.linalg.norm(closest_1 - closest_2, axis=1)


def human_limits() -> Any:
    """The ``priors.json`` ``adaptation_*`` nominals the human re-plan allowance uses (its defaults where a key is missing)."""

    import json

    from lib.physics.params import PRIORS_PATH
    from lib.replay.players.strokes import HumanLimits

    defaults = HumanLimits()
    if not PRIORS_PATH.is_file():
        return defaults
    priors = json.loads(PRIORS_PATH.read_text(encoding="utf-8"))

    def value(key: str, default: float) -> float:
        entry = priors.get(key)
        return float(entry["value"]) if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)) else default

    return HumanLimits(
        latency_s=value("adaptation_latency_s", defaults.latency_s),
        timing_latency_s=value("adaptation_latency_timing_s", defaults.timing_latency_s),
        orientation_latency_s=value("adaptation_orientation_latency_s", defaults.orientation_latency_s),
        time_shift_late_s=value("adaptation_time_shift_max_s", defaults.time_shift_late_s),
        time_shift_preswing_s=value("adaptation_time_shift_preswing_max_s", defaults.time_shift_preswing_s),
        offset_late_m=value("adaptation_offset_max_m", defaults.offset_late_m),
        offset_preswing_m=value("adaptation_offset_preswing_max_m", defaults.offset_preswing_m),
        rotation_late_deg=value("adaptation_rotation_max_deg", defaults.rotation_late_deg),
        accel_max_mps2=value("adaptation_correction_accel_max_mps2", defaults.accel_max_mps2),
        speed_max_mps=value("adaptation_correction_speed_max_mps", defaults.speed_max_mps),
        source="priors.json adaptation_*" if any(key.startswith("adaptation_") for key in priors) else "defaults (priors.json has no adaptation_* keys)",
    )


def reference_drop_rebound(params: BallTableParams, drop_height: float = DROP_HEIGHT_M, *, speed_curve: bool = True) -> float:
    """
    The reference model's rebound (ball bottom above the table) for a drop.

    :param speed_curve: Whether the table restitution follows its speed curve: true for the hybrid table (the
        reference impact law), false for PhysX's native table, which applies one restitution at every speed.
    """

    model = params if speed_curve else params.model_copy(update={"table_contact": params.table_contact.model_copy(update={"restitution_curve": None})})
    radius = params.ball.radius
    trajectory = simulate([0.6, 0.25, radius + drop_height, 0, 0, 0, 0, 0, 0], model, dt=1e-4, max_time=1.5, terminal_events=("floor",))
    bounce = next(event for event in trajectory.events if event.kind == "table")
    after = trajectory.times > bounce.time
    return float(np.max(trajectory.positions[after, 2]) - radius)


@dataclass(frozen=True)
class PlanSummary:
    """
    One stroke plan's magnitudes (required, before clamping).

    :ivar stroke: Stroke index.
    :ivar time_shift_s: Time warp, s.
    :ivar offset_m: Blade-centre offset, m.
    :ivar rotation_rad: Blade tilt, rad.
    :ivar within_bounds: Whether nothing was clamped.
    :ivar reachable: Whether the predicted ball reached the stroke window.
    :ivar clamp_time: The warp sat at its bound.
    :ivar clamp_offset: The offset needed more than its bound.
    :ivar clamp_tilt: The tilt needed more than its bound.
    :ivar clamp_reach: The arm could not hold the planned blade; the plan was cut to what it can.
    """

    stroke: int
    time_shift_s: float
    offset_m: float
    rotation_rad: float
    within_bounds: bool
    reachable: bool
    clamp_time: bool = False
    clamp_offset: bool = False
    clamp_tilt: bool = False
    clamp_reach: bool = False


@dataclass
class ReplayRecord:
    """
    Everything the loop measured.

    :ivar drops: The opening drop tests.
    :ivar plans: Every stroke plan's magnitudes.
    :ivar plan_details: Every plan, for the artifact.
    :ivar motion: Played motion per role.
    :ivar placements: Serves, continuity balls and drops placed, ``(time, label)``.
    :ivar terminal_reached: Whether the replay ended on its terminal condition.
    :ivar game_start_s: Session time the replay clock started.
    :ivar game_end_s: Session time it stopped.
    :ivar ticks: Rendered ticks after the pre-roll.
    :ivar pre_roll_ticks: Rendered pre-roll ticks.
    :ivar loop_wall_s: Wall time of those ticks, s.
    :ivar pre_roll_wall_s: Wall time of the pre-roll, s.
    :ivar warmup_frames: Frames rendered with physics paused before the pre-roll.
    :ivar warmup_wall_s: Their wall time, s.
    :ivar warmup_physics_steps: Physics steps that ran during the warm-up (0 when the renderer advanced alone).
    :ivar tick_ms: Per-tick wall times, ms.
    :ivar plan_ms: Per-plan wall times, ms.
    :ivar ik_residual_m: Largest arm IK residual over the replay, m.
    :ivar fk_check_mm: Largest paddle-hand difference between this module's FK and UsdSkel's, mm.
    :ivar nudges: Every placed or replayed ball's aim (:meth:`ReplayLoop._aimed_launch`): goal, residual, status.
    :ivar nudge_failures: The aims whose best effort missed outside the receiving stroke's bounds.
    :ivar incoming: Each planned stroke's incoming ball: ``placed`` (a point start), ``replayed`` (a continuity
        ball) or ``live`` (the other player's physical hit).
    """

    drops: list[DropTest] = field(default_factory=list)
    plans: list[PlanSummary] = field(default_factory=list)
    plan_details: list[dict[str, Any]] = field(default_factory=list)
    motion: dict[str, MotionSamples] = field(default_factory=dict)
    placements: list[tuple[float, str]] = field(default_factory=list)
    terminal_reached: bool = False
    game_start_s: float = 0.0
    game_end_s: float = 0.0
    ticks: int = 0
    pre_roll_ticks: int = 0
    loop_wall_s: float = 0.0
    pre_roll_wall_s: float = 0.0
    warmup_frames: int = 0
    warmup_wall_s: float = 0.0
    warmup_physics_steps: int = 0
    tick_ms: list[float] = field(default_factory=list)
    plan_ms: list[float] = field(default_factory=list)
    ik_residual_m: float = 0.0
    fk_check_mm: list[float] = field(default_factory=list)
    nudges: list[dict[str, Any]] = field(default_factory=list)
    nudge_failures: list[dict[str, Any]] = field(default_factory=list)
    incoming: dict[int, str] = field(default_factory=dict)
    hidden_launches: int = 0
    launch_blade_distance_m: list[float] = field(default_factory=list)


class ReplayLoop:
    """
    The per-tick orchestration of one replay.

    :param run: The scenario run.
    :param package: The session package, restricted to the points to play.
    :param belief: The players' physics (always nominal, whatever the engine runs).
    :param physics: The live ball's per-step physics.
    :param characters: Spawned bodies by role.
    :param animators: Motion players by role.
    :param recorder: The cameras, telemetry and MP4 spool.
    :param max_sim_s: Safety cap on simulated session time from the start; reaching it fails the replay.
    :param world: The simulator world (default ``antioch.world()``).
    :param stage: The stage (default ``antioch.stage()``).
    :param show_ball: Shows or hides the ball's render mesh (default: on the stage).
    :param check_usdskel: Compare the paddle hand with UsdSkel's own forward kinematics every 150 ticks.
    :param scale_spare: Scales a retired ball's render mesh (default: on the stage).
    """

    def __init__(
        self,
        run: antioch.ScenarioRun,
        package: LoadedSessionPackage,
        belief: BallTableParams,
        physics: GamePhysics,
        characters: dict[str, CharacterHandle],
        animators: dict[str, CharacterAnimator],
        recorder: Recorder,
        *,
        max_sim_s: float,
        world: Any = None,
        stage: Any = None,
        show_ball: Callable[[bool], None] | None = None,
        check_usdskel: bool = True,
        scale_spare: Callable[[int, float], None] | None = None,
    ) -> None:
        self.run = run
        self.max_sim_s = max_sim_s
        self.package = package
        self.manifest = package.manifest
        self.belief = belief
        self.physics = physics
        self.characters = characters
        self.animators = animators
        self.recorder = recorder
        self.world = world if world is not None else antioch.world()
        self.stage = stage if stage is not None else antioch.stage()
        self.show_ball = show_ball if show_ball is not None else (lambda visible: set_ball_visible(self.stage, visible))
        self.scale_spare = scale_spare if scale_spare is not None else (lambda index, scale: set_ball_scale(self.stage, SPARE_PATHS[index], scale))
        self._collecting: dict[int, float] = {}
        self._placement_cursor = 0
        self.retired_fades = 0
        self._tunnel_cursor = 0
        self.check_usdskel = check_usdskel
        self.rally = Rally(self.manifest)
        self.record = ReplayRecord()
        self.bounds = AdaptationBounds()
        self._last_driven: dict[str, Array] = {}
        self._capsules: dict[str, Capsules] = {}
        self._retired_cursor = 0
        self._off_camera_since: dict[int, float] = {}
        self.human_limits = human_limits()
        self._replans: dict[int, ReplanAllowance] = {}
        self.rng = np.random.default_rng(int(self.manifest.session.index) * 1009 + 7)
        self.event_cursor = 0
        self.log_cursor = 0
        self.continuity_live = False
        self.ball_visible = False
        self.prediction: Array | None = None
        self.paddle_state: dict[str, tuple[Array, Array, Array]] = {}
        self.start_s = self.manifest.serves[0].time_s - LEAD_S
        # The replay ends with the recording: the session's end, or the end of the shorter motion track if that comes
        # first (a frame sampled past a track's last one is held and fails motion QA).
        self.session_end_s = min([self.manifest.session.duration_s, *(track.end_s for track in self.manifest.tracks)])
        self.telemetry_every = TELEMETRY_EVERY
        self.real_frames = {f"orbbec_{label}": package.frames_of(label) for label in ("8E", "8F")}  # type: ignore[arg-type]
        self.real_cursor = {name: 0 for name in self.real_frames}
        self.video_frames = 0
        self.isp_luts = isp_luts()
        self.references = {
            role: {
                correction.stroke: correction for correction in animator.set_references([stroke for stroke in self.manifest.strokes if stroke.hitter == role])
            }
            for role, animator in animators.items()
        }
        self.skeleton_index = {role: [handle.skeleton.index(name) for name in SKELETON_JOINTS] for role, handle in characters.items()}
        self.stroke_times = {role: tuple(stroke.time_s for stroke in self.manifest.strokes if stroke.hitter == role) for role in characters}
        for role, animator in animators.items():
            self.record.motion[role] = MotionSamples(
                fps=1.0 / RENDER_DT,
                joint_names=SKELETON_JOINTS,
                driven_joints=tuple(animator.skeleton.joints[int(index)] for index in animator.driven),
            )

    # ---- helpers ------------------------------------------------------------------

    def _seen(self, position: Array) -> bool:
        """Whether any camera sees a ball at ``position``: inside its image and not hidden behind a player's body."""

        radius = self.belief.ball.radius
        for camera in self.recorder.cameras.values():
            if camera.project(position) is None:
                continue
            centre = -np.asarray(camera.rotation).T @ np.asarray(camera.translation)
            hidden = False
            for motion in self.physics.body_motion.values():
                capsules = motion.end
                if np.any(_segment_distances(centre, position, capsules.a, capsules.b) < capsules.radius - radius):
                    hidden = True
                    break
            if not hidden:
                return True
        return False

    def _in_play_volume(self, position: Array) -> bool:
        params = self.belief
        return (
            abs(float(position[0])) <= 0.5 * params.table.length + PLAY_VOLUME_MARGIN_M
            and abs(float(position[1])) <= 0.5 * params.table.width + PLAY_VOLUME_MARGIN_M
            and -0.05 <= float(position[2]) <= 2.0
        )

    def _on_table_top(self, state: Array) -> bool:
        params = self.belief
        return (
            abs(float(state[0])) <= 0.5 * params.table.length
            and abs(float(state[1])) <= 0.5 * params.table.width
            and float(state[2]) <= params.ball.radius + 0.02
            and float(state[2]) >= params.ball.radius - 0.01
            and float(np.linalg.norm(state[3:6])) < TABLE_REST_SPEED_MPS
        )

    def _tend_retired(self, now: float) -> None:
        """
        Keep the play to one ball: retired balls (the old ball after each placement) play out, but leave the play in time.

        - A retired ball no camera sees (or that a player's body hides) for :data:`RETIRED_OFF_CAMERA_S` is collected.
        - At a new placement every retired ball inside the play volume (above the table area or on its top), and at any
          time one lying on the table top, must go: at once if no camera sees it, else it shrinks out over
          :data:`RETIRED_FADE_S` (a continuity device, logged as one). The dead ball in play lying on the table top
          between points is handed to a retired ball and collected the same way.
        - Each retired ball's visibility goes to the physics, which never reuses one a camera sees.
        """

        placed = False
        for time_s, index, label in self.physics.retired[self._retired_cursor :]:
            self._event(f"t={time_s:.2f} s the ball in play retires as retired ball {index} and plays out ({label})")
            placed = placed or not label.startswith("collect")
        self._retired_cursor = len(self.physics.retired)
        for time_s, surface, position in self.physics.tunnel_log[self._tunnel_cursor :]:
            self._event(
                f"t={time_s:.4f} s TUNNEL: the ball passed through the {surface} at ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m",
                "ERROR",
            )
            self.recorder.log_tunnel(position, surface)
        self._tunnel_cursor = len(self.physics.tunnel_log)
        if not self.physics.spares:
            return
        # The dead ball lying on the table top between points: hand it to a retired ball, which is then collected.
        if self.rally.phase == "over" and self.physics.parked_at is None and self.physics.pending is None and self.physics.collect_request is None:
            if self._on_table_top(self.physics.ball_state()):
                self.physics.collect_ball("collected from the table top between points", PARK_POSITION_M)
                self._set_ball_visible(False)
        positions, _orientations, linear, _angular = self.physics.read()
        # Each executed placement: how far it left from the nearest blade (a launch must leave a swinging blade).
        for time_s, label, distance in self.physics.placement_blade_distance[self._placement_cursor :]:
            self.record.launch_blade_distance_m.append(distance)
            if distance > 0.2:
                self._event(f"t={time_s:.2f} s the ball was placed {100 * distance:.0f} cm from the nearest blade ({label})", "WARN")
        self._placement_cursor = len(self.physics.placement_blade_distance)
        for index, row in enumerate(self.physics.spare_rows()):
            if self.physics.spare_parked[index]:
                self._off_camera_since.pop(index, None)
                if index in self._collecting:
                    del self._collecting[index]
                    self.scale_spare(index, 1.0)
                self.physics.spare_visible[index] = False
                continue
            position = positions[row]
            state = np.concatenate([position, linear[row]])
            visible = self._seen(position)
            self.physics.spare_visible[index] = visible
            if index in self._collecting:
                progress = (now - self._collecting[index]) / RETIRED_FADE_S
                if progress >= 1.0 or not visible:
                    self.physics.park_spare(index)
                else:
                    self.scale_spare(index, 1.0 - float(rot.smoothstep(progress)))
                continue
            if (placed and self._in_play_volume(position)) or self._on_table_top(state):
                if not visible:
                    self.physics.park_spare(index)
                    self._event(f"t={now:.2f} s retired ball {index} collected out of view")
                else:
                    self._collecting[index] = now
                    self.retired_fades += 1
                    self._event(
                        f"t={now:.2f} s retired ball {index} is in play on camera and shrinks out over {RETIRED_FADE_S:.1f} s "
                        "(a continuity device, not physics)",
                        "WARN",
                    )
                continue
            if visible:
                self._off_camera_since.pop(index, None)
                continue
            since = self._off_camera_since.setdefault(index, now)
            if now - since >= RETIRED_OFF_CAMERA_S:
                self.physics.park_spare(index)
                self._off_camera_since.pop(index, None)

    def _park_dead_ball(self, now: float) -> None:
        """Park the ball once its point has been over for :data:`DEAD_BALL_PARK_S` and nothing is scheduled to replace it yet."""

        if self.rally.phase != "over" or self.physics.parked_at is not None or self.physics.pending is not None or not self.rally.points:
            return
        ended = self.rally.points[-1].ended_at_s
        if ended is not None and now - ended >= DEAD_BALL_PARK_S:
            self.physics.park_ball(PARK_POSITION_M)
            self._set_ball_visible(False)
            self.continuity_live = False

    def _isp(self, name: str, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """The virtual Orbbec's fixed white balance (:func:`lib.him_lab.isp_luts`), as the recorded frames carry theirs."""

        luts = self.isp_luts.get(name)
        if luts is None:
            return frame
        out = np.empty((*frame.shape[:2], 3), dtype=np.uint8)
        for channel in range(3):
            out[..., channel] = luts[channel][frame[..., channel]]
        return out

    def recording_time(self, session_s: float) -> float:
        """
        The recording's ``sim_time`` of a session time.

        The recording's clock counts physics steps since it attached and the
        replay clock counts them from the session's origin, so the two differ
        by a constant the run reports.
        """

        recorded = getattr(self.run, "sim_s", None)
        offset = (recorded - self.physics.time_s) if recorded is not None else -self.physics.time_offset_s
        return session_s + offset

    def _log_real_frames(self, until_s: float) -> None:
        """Every recorded frame captured in the replay's span up to ``until_s``, at its own time."""

        for name, frames in self.real_frames.items():
            cursor = self.real_cursor[name]
            while cursor < len(frames) and frames[cursor].time_s <= until_s:
                frame = frames[cursor]
                if frame.time_s > self.start_s:
                    self.recorder.log_real_frame(name, self.package.frame_jpeg(frame.index), self.recording_time(frame.time_s))
                cursor += 1
            self.real_cursor[name] = cursor

    def real_frames_in_span(self) -> dict[str, int]:
        """Recorded frames per camera captured in the replay's span (what telemetry must have logged)."""

        start, end = self.record.game_start_s, self.record.game_end_s
        return {name: sum(1 for frame in frames if start < frame.time_s <= end) for name, frames in self.real_frames.items()}

    def _check_physics(self) -> None:
        if self.physics.error is not None:
            raise RuntimeError(f"the physics callback failed: {self.physics.error}")

    def _set_ball_visible(self, visible: bool) -> None:
        if visible != self.ball_visible:
            self.show_ball(visible)
            self.ball_visible = visible

    def _paddle_end(self, role: str, t: float, *, count: bool = True) -> tuple[AnimatedPose, Array, Array, Array]:
        """The pose at ``t`` and the paddle's position, orientation and velocity there."""

        animator = self.animators[role]
        pose = animator.pose_at(t, count=count)
        before = animator.pose_at(t - VELOCITY_PROBE_S, count=False).paddle_position
        after = animator.pose_at(t + VELOCITY_PROBE_S, count=False).paddle_position
        velocity = (after - before) / (2.0 * VELOCITY_PROBE_S)
        return pose, pose.paddle_position, rot.from_matrix(pose.paddle_rotation), velocity

    def _pose_players(self, t0: float, t1: float) -> dict[str, AnimatedPose]:
        """Write every body's pose for ``t1`` and give the physics each paddle's segment over ``[t0, t1]``."""

        poses: dict[str, AnimatedPose] = {}
        for role, handle in self.characters.items():
            self.animators[role].release_finished(t0)
            pose, position, orientation, velocity = self._paddle_end(role, t1)
            handle.write_pose(pose.local_rotation, pose.root_translation, self.animators[role].translated, pose.local_translation)
            previous = self.paddle_state.get(role, (position, orientation, velocity))
            knot = None
            contact = self._contact_within(role, t0, t1)
            if contact is not None:
                _mid, knot_position, knot_orientation, knot_velocity = self._paddle_end(role, contact, count=False)
                knot = (contact, knot_position, knot_velocity, knot_orientation)
            self.physics.segments[role] = PaddleSegment(
                t0=t0, t1=t1, p0=previous[0], v0=previous[2], p1=position, v1=velocity, q0=previous[1], q1=orientation, knot=knot
            )
            self.paddle_state[role] = (position, orientation, velocity)
            self.record.ik_residual_m = max(self.record.ik_residual_m, pose.ik_residual_m)
            poses[role] = pose
        return poses

    def _contact_within(self, role: str, t0: float, t1: float) -> float | None:
        """A contact the role's blade is meant to make inside ``(t0, t1)``: the active plan's, else a reconstructed one."""

        animator = self.animators[role]
        margin = 0.002
        if animator.active is not None and t0 + margin < animator.active.plan.contact_s < t1 - margin:
            return animator.active.plan.contact_s
        for correction in self.references.get(role, {}).values():
            if t0 + margin < correction.contact_s < t1 - margin:
                return correction.contact_s
        return None

    def _pose_bodies(self, t0: float, t1: float, poses: dict[str, AnimatedPose]) -> dict[str, Array]:
        """
        Give the physics each posed body's capsules over ``[t0, t1]`` (a ball that meets a body bounces off it).

        Near a player's own contact (scheduled or planned) the paddle hand and forearm are left out, so the blade, not
        the arm, meets the ball. A character that is not posed this tick has no body in the physics.

        :return: Each posed body's joint world positions at ``t1`` (the motion QA reuses them).
        """

        positions: dict[str, Array] = {}
        for role in tuple(self.physics.body_motion):
            if role not in poses:
                del self.physics.body_motion[role]
                self.physics.body_excluded.pop(role, None)
                self._capsules.pop(role, None)
        for role, pose in poses.items():
            animator = self.animators[role]
            _rotations, joints = forward_kinematics(animator.skeleton, pose.local_rotation, pose.root_translation, animator.placement)
            current = capsules_from_joints(animator.skeleton.joints, joints)
            self.physics.body_motion[role] = BodyMotion(t0, t1, self._capsules.get(role, current), current)
            self._capsules[role] = current
            contacts = self.stroke_times[role] if animator.active is None else (*self.stroke_times[role], animator.active.plan.contact_s)
            self.physics.body_excluded[role] = excluded_capsules(t0, contacts, animator.track.character.grip.hand_joint)
            positions[role] = joints
        return positions

    def _sample_motion(self, poses: dict[str, AnimatedPose], world_joints: dict[str, Array]) -> dict[str, Array]:
        skeletons: dict[str, Array] = {}
        for role, pose in poses.items():
            animator = self.animators[role]
            positions = world_joints[role]
            joints = positions[self.skeleton_index[role]]
            driven = pose.local_rotation[animator.driven]
            self.record.motion[role].add(joints, driven, ik_miss_m=pose.ik_residual_m if pose.ik_active else None, hand_gap_m=pose.hand_gap_m)
            previous = self._last_driven.get(role)
            if previous is not None:
                # Per-frame motion QA in the telemetry, so a pop can be found on the timeline: the largest joint turn
                # this frame (and which joint), and how far the arm fell short of the planned paddle.
                dots = np.clip(np.abs(np.sum(driven * previous, axis=-1)), -1.0, 1.0)
                steps = np.degrees(2.0 * np.arccos(dots))
                limit = POP_DEG_PER_S * RENDER_DT
                popped = [(animator.skeleton.joints[int(animator.driven[index])], float(steps[index])) for index in np.flatnonzero(steps > limit)]
                self.recorder.log_motion_qa(role, float(steps.max()), popped, pose.ik_residual_m if pose.ik_active else 0.0)
            self._last_driven[role] = driven.copy()
            skeletons[role] = joints
        return skeletons

    def _far_target(self, hitter: Role) -> tuple[float, float]:
        """A landing target on the far half from the session's empirical distribution (or the half's centre)."""

        far = self.rally.end_of(other_role(hitter))
        name = f"landing_xy_{far.lower()}"
        names = {distribution.name for distribution in self.manifest.distributions}
        if name in names:
            samples = self.package.distribution(name)
            choice = samples[int(self.rng.integers(len(samples)))]
            return float(choice[0]), float(choice[1])
        return END_SIGN[far] * 0.9, 0.0

    def _event(self, text: str, level: str = "INFO") -> None:
        self.recorder.event(text, level)

    # ---- actions ----------------------------------------------------------------------

    def _serve(self, action: ServeAction, now: float) -> None:
        serve = action.serve
        server = serve.server
        self.continuity_live = False
        # The ball leaves the server's blade at the server's actual swing contact (peak blade speed near the recorded
        # time), so it never appears in mid-air or leaves a player who is not swinging.
        hit_s = self._swing_contact(server, serve.time_s, now)
        position = self._blade_launch(server, hit_s)
        if serve.fitted is not None:
            velocity = np.asarray(serve.fitted.velocity_mps, dtype=np.float64)
            spin = np.asarray(serve.fitted.spin_radps, dtype=np.float64)
            source = f"the fitted velocity from the blade ({100 * np.linalg.norm(np.asarray(serve.fitted.position_m) - position):.1f} cm from the fitted start)"
        elif serve.kind == "rally_start":
            sign_far = -END_SIGN[self.rally.end_of(server)]
            target = self._far_target(server)
            spin = CONTINUITY_TOPSPIN_RADPS * np.cross((0.0, 0.0, 1.0), (sign_far, 0.0, 0.0))
            guess = np.array([(target[0] - position[0]) / CONTINUITY_FLIGHT_S, (target[1] - position[1]) / CONTINUITY_FLIGHT_S, 1.5])
            velocity, residual = solve_launch(
                self.belief, position, spin, targets=[target], flight_time_s=CONTINUITY_FLIGHT_S, initial_velocity=guess, t0=hit_s
            )
            source = f"rally start solved from the striker's blade toward ({target[0]:.2f}, {target[1]:.2f}) (residual {residual:.3f})"
        else:
            sign = END_SIGN[self.rally.end_of(server)]
            spin = CONTINUITY_TOPSPIN_RADPS * np.cross((0.0, 0.0, 1.0), (-sign, 0.0, 0.0))
            receiver = self._far_target(server)
            velocity, residual = solve_launch(
                self.belief, position, spin, targets=[(sign * 0.9, 0.0), receiver], flight_time_s=None, initial_velocity=(-sign * 5.5, 0.0, -1.0), t0=hit_s
            )
            source = f"solved from the server's blade (residual {residual:.3f})"
        first = next((item for item in self.manifest.strokes if item.point == action.point), None)
        # Aimed at the receiver's first contact (reconstructed, observed, or the reference blade's face), the launch
        # time searched within the swing window around the recorded hit.
        hit_s, position, velocity, placed = self._aimed_launch(
            "point start",
            server,
            first,
            hit_s,
            serve.time_s,
            now,
            np.asarray(velocity, dtype=np.float64),
            np.asarray(spin, dtype=np.float64),
            cap_mps=POINT_START_MAX_NUDGE_MPS,
        )
        aimed = not placed.startswith(", not aimed")
        label = f"serve {action.point}" + (" (placed)" if aimed else "")
        self.physics.place_ball(position, velocity, spin, label=label, at_time=hit_s, retire=True)
        self._cool_down(server, hit_s)
        self.record.placements.append((hit_s, label))
        self._set_ball_visible(True)
        verb = "serves" if serve.kind == "serve" else "starts the rally"
        if aimed:
            # A point-start ball placed to meet the receiver's recorded contact: never physics evidence (like a
            # continuity ball); the receiver's stroke counts only through a physical contact.
            self.continuity_live = True
            self._event(
                f"t={serve.time_s:.2f} s point {action.point}: PLACED ball for {server}'s {'serve' if serve.kind == 'serve' else 'rally start'} "
                f"at {np.linalg.norm(velocity):.1f} m/s, {source}{placed}; not physics evidence",
                "WARN",
            )
        else:
            self._event(f"t={serve.time_s:.2f} s point {action.point}: {server} {verb} at {np.linalg.norm(velocity):.1f} m/s, {source}{placed}")

    def _plan(self, action: PlanAction, now: float) -> None:
        stroke = action.stroke
        animator = self.animators[stroke.hitter]
        target = stroke.landing.centre_m if stroke.landing is not None else self._far_target(stroke.hitter)
        following = next(
            (item for item in self.manifest.strokes if item.point == stroke.point and item.index == stroke.index + 1 and item.hitter != stroke.hitter), None
        )
        next_contact = None
        if following is not None and following.paddle is not None:
            reconstructed = following.paddle
            next_contact = (reconstructed.time_s, np.asarray(reconstructed.contact_point_m) + np.asarray(reconstructed.normal) * self.belief.ball.radius)
        elif following is not None:
            recorded = self.animators[following.hitter].reference_paddle(following.time_s)
            toward_net = -np.sign(recorded.position[0])
            face = recorded.normal if recorded.normal[0] * toward_net >= 0.0 else -recorded.normal
            gap = self.belief.ball.radius + 0.5 * self.belief.paddle.thickness
            next_contact = (following.time_s, recorded.position + face * gap)
        next_window = self.bounds_for(following).max_time_shift_s if following is not None and next_contact is not None else 0.0
        started = time.perf_counter()
        plan: StrokePlan = plan_stroke(
            self.belief,
            stroke=stroke.index,
            scheduled_s=stroke.time_s,
            ball_state=self.physics.ball_state(),
            now_s=now,
            nominal_paddle=animator.reference_paddle,
            target_xy=target,
            bounds=self.bounds_for(stroke),
            next_contact=next_contact,
            next_window_s=next_window,
        )
        if plan.reachable:
            # Reachability is a planner constraint: a blade the arm cannot hold becomes the nearest one it holds; it stays
            # within bounds while the ball still meets that face, else it is a clamp with cause ``reach``.
            reach = animator.plan_reach(plan)
            if not reach.reachable:
                plan = project_plan(
                    self.belief,
                    plan,
                    animator.reference_paddle,
                    reach.held_offset_m,
                    reach.shortfall_m,
                    next_contact=next_contact,
                    next_window_s=next_window,
                    held_rotation_rad=reach.held_rotation_rad,
                    tilt_shortfall_deg=reach.tilt_shortfall_deg,
                )
        self.record.plan_ms.append(1e3 * (time.perf_counter() - started))
        if plan.reachable:
            animator.adopt(plan, now)
        incoming = self._incoming_class(stroke.hitter, now)
        self.record.incoming[stroke.index] = incoming
        miss = self._ball_miss(stroke, plan, now)
        self.prediction = plan.prediction
        self.record.plans.append(
            PlanSummary(
                stroke=stroke.index,
                time_shift_s=plan.required_time_shift_s,
                offset_m=plan.required_offset_m,
                rotation_rad=plan.required_rotation_rad,
                within_bounds=plan.within_bounds,
                reachable=plan.reachable,
                clamp_time=plan.clamp_time,
                clamp_offset=plan.clamp_offset,
                clamp_tilt=plan.clamp_tilt,
                clamp_reach=plan.clamp_reach,
            )
        )
        self.record.plan_details.append(
            {
                "stroke": stroke.index,
                "hitter": stroke.hitter,
                "replan": action.replan,
                "planned_at_s": round(now, 4),
                "scheduled_s": stroke.time_s,
                "contact_s": round(plan.contact_s, 4),
                "time_shift_ms": round(1e3 * plan.time_shift_s, 2),
                "offset_cm": [round(100 * float(v), 2) for v in plan.offset_m],
                "required_offset_cm": round(100 * plan.required_offset_m, 2) if math.isfinite(plan.required_offset_m) else None,
                "tilt_deg": round(math.degrees(float(np.linalg.norm(plan.rotation))), 2),
                **self._bound_parts(stroke, plan),
                "bound_time_ms": round(1e3 * plan.bounds.max_time_shift_s, 1),
                "bound_offset_cm": round(100 * plan.bounds.max_offset_m, 2),
                "bound_tilt_deg": round(math.degrees(plan.bounds.max_rotation_rad), 2),
                "clamp_time": plan.clamp_time,
                "clamp_offset": plan.clamp_offset,
                "clamp_tilt": plan.clamp_tilt,
                "clamp_reach": plan.clamp_reach,
                "reach_fraction": round(plan.reach_fraction, 3),
                "reach_projected": bool(plan.reach_shortfall_m > 0.0 and not plan.clamp_reach),
                "reach_shortfall_cm": round(100 * plan.reach_shortfall_m, 2),
                "predicted_next_miss_cm": round(100 * plan.predicted_next_miss_m, 2) if math.isfinite(plan.predicted_next_miss_m) else None,
                "predicted_next_dt_ms": round(1e3 * plan.predicted_next_dt_s, 1) if math.isfinite(plan.predicted_next_dt_s) else None,
                "required_tilt_deg": round(math.degrees(plan.required_rotation_rad), 2),
                "within_bounds": plan.within_bounds,
                "reachable": plan.reachable,
                "target_xy": list(plan.target_xy),
                "predicted_landing_xy": list(plan.landing.point) if plan.landing is not None else None,
                "predicted_on_table": plan.landing.on_table if plan.landing is not None else None,
                "plan_ms": round(self.record.plan_ms[-1], 1),
                "incoming": incoming,
                **miss,
                **self._reference_details(stroke.hitter, stroke.index),
            }
        )
        if plan.reachable:
            self.recorder.log_adaptation(plan.time_shift_s, float(np.linalg.norm(plan.offset_m)), float(np.linalg.norm(plan.rotation)))
        self._event(
            f"t={now:.2f} s {'re-plan' if action.replan else 'plan'} stroke {stroke.index} ({stroke.hitter}): "
            + (
                f"warp {1e3 * plan.time_shift_s:+.0f} ms, offset {100 * plan.required_offset_m:.1f} cm, tilt {math.degrees(plan.required_rotation_rad):.1f} deg"
                + ("" if plan.within_bounds else " (clamped)")
                if plan.reachable
                else "ball never reaches the stroke window"
            )
        )

    def _contact_goal(self, stroke: Any) -> tuple[Array, float, str]:
        """
        Where and when a placed or replayed ball should meet ``stroke``'s player: the reconstructed contact (ball centre),
        else the tracker's observed incoming ball, else the reference blade's struck face at the scheduled contact.

        :return: Ball-centre goal (m), its time (s) and what it is.
        """

        radius = self.belief.ball.radius
        if stroke.paddle is not None:
            target = stroke.paddle
            return (
                np.asarray(target.contact_point_m, dtype=np.float64) + np.asarray(target.normal, dtype=np.float64) * radius,
                float(target.time_s),
                "reconstructed contact",
            )
        if stroke.observed_ball_m is not None:
            return np.asarray(stroke.observed_ball_m, dtype=np.float64), float(stroke.time_s), "observed incoming ball"
        blade = self.animators[stroke.hitter].reference_paddle(stroke.time_s)
        face = blade.rotation[:, 2]
        # The struck face looks toward the net (the ball comes from the other end).
        if face[0] * END_SIGN[self.rally.end_of(stroke.hitter)] > 0.0:
            face = -face
        return blade.position + face * (radius + 0.5 * self.belief.paddle.thickness), float(stroke.time_s), "reference blade face"

    def _flight_miss(self, position: Array, velocity: Array, spin: Array, goal: Array, flight_s: float) -> float:
        trajectory = predict(self.belief, np.concatenate([position, velocity, spin]), 0.0, horizon_s=flight_s + 0.02)
        state = state_at(trajectory, flight_s)
        return float(np.linalg.norm((state[:3] if state is not None else trajectory.states[-1][:3]) - goal))

    def _aimed_launch(
        self,
        kind: str,
        launcher: Role,
        following: Any,
        launch_s: float,
        recorded_s: float,
        earliest_s: float,
        velocity: Array,
        spin: Array,
        *,
        cap_mps: float,
    ) -> tuple[float, Array, Array, str]:
        """
        Aim a placed or replayed ball from ``launcher``'s blade at the next stroke's contact (:meth:`_contact_goal`).

        The launch velocity is nudged (by at most ``cap_mps``, so the flight stays near the recorded one) jointly with
        the launch time (from ``launch_s`` outwards, within :data:`SWING_SEARCH_S` of the recorded hit ``recorded_s``
        and never before ``earliest_s``) and the
        arrival time (within the receiving stroke's time bound). The first solution within
        :data:`CONTINUITY_MAX_MISS_M` wins, searching outwards from the nominal times; otherwise the best effort is
        kept, accepted when its miss lies inside the receiving stroke's offset bound and counted in
        :attr:`ReplayRecord.nudge_failures` (with its residual) when it does not. Never silently un-nudged.

        :return: Launch time, launch position, velocity, and a phrase for the event log.
        """

        started = time.perf_counter()
        entry: dict[str, Any] = {"kind": kind, "launcher": launcher, "launch_s": round(launch_s, 4)}
        # A ball leaves a blade only while it swings: the launch times searched are the swinging ones in the window.
        window = sorted(
            {launch_s} | {float(t) for t in np.arange(recorded_s - SWING_SEARCH_S, recorded_s + SWING_SEARCH_S + 1e-9, AIM_LAUNCH_STEP_S)},
            key=lambda t: abs(t - launch_s),
        )
        swinging = [t for t in window if t >= earliest_s - 1e-9 and self._swinging(launcher, t)]
        if not swinging:
            try:
                return self._hidden_launch(kind, launcher, following, max(launch_s, earliest_s), velocity, spin, entry)
            finally:
                entry["solve_ms"] = round(1e3 * (time.perf_counter() - started), 1)
                self.record.nudges.append(entry)
        if launch_s not in swinging:
            launch_s = swinging[0]
            entry["launch_s"] = round(launch_s, 4)
        position = self._blade_launch(launcher, launch_s)
        try:
            if following is None or following.hitter == launcher:
                entry |= {"stroke": None, "source": "none", "status": "no next stroke"}
                return launch_s, position, velocity, ", not aimed: no next stroke in this point"
            goal, goal_s, source = self._contact_goal(following)
            bound = self.bounds_for(following)
            entry |= {"stroke": following.index, "source": source}
            launches = sorted(swinging, key=lambda t: abs(t - launch_s))
            arrivals = sorted((float(a) for a in np.linspace(-bound.max_time_shift_s, bound.max_time_shift_s, AIM_ARRIVALS)), key=abs)
            best: tuple[float, float, float, Array, Array, float] | None = None
            for t_launch in launches:
                start = self._blade_launch(launcher, t_launch)
                for shift in arrivals:
                    flight = goal_s + shift - t_launch
                    if flight < 0.1:
                        continue
                    solved, miss = solve_through(self.belief, start, spin, goal, flight, velocity)
                    change = float(np.linalg.norm(solved - velocity))
                    if change > cap_mps:
                        solved = velocity + (solved - velocity) * (cap_mps / change)
                        miss, change = self._flight_miss(start, solved, spin, goal, flight), cap_mps
                    if best is None or miss < best[0]:
                        best = (miss, t_launch, shift, start, solved, change)
                    if miss <= CONTINUITY_MAX_MISS_M:
                        break
                if best is not None and best[0] <= CONTINUITY_MAX_MISS_M:
                    break
            if best is None:
                entry |= {"status": "no flight", "residual_m": None}
                self.record.nudge_failures.append(entry)
                return launch_s, position, velocity, f", not aimed: stroke {following.index}'s contact is under 0.1 s away"
            miss, t_launch, shift, start, solved, change = best
            entry |= {
                "residual_m": round(miss, 4),
                "change_mps": round(change, 3),
                "launch_shift_s": round(t_launch - launch_s, 3),
                "arrival_shift_s": round(shift, 3),
                "bound_offset_m": round(bound.max_offset_m, 4),
            }
            where = f"stroke {following.index}'s {source}"
            timing = (
                f" (launch {1e3 * (t_launch - launch_s):+.0f} ms, arrival {1e3 * shift:+.0f} ms)"
                if abs(t_launch - launch_s) > 1e-9 or abs(shift) > 1e-9
                else ""
            )
            if miss <= CONTINUITY_MAX_MISS_M:
                entry["status"] = "nudged"
                return t_launch, start, solved, f", nudged {change:.2f} m/s to meet {where} within {100 * miss:.1f} cm{timing}"
            if miss <= bound.max_offset_m:
                entry["status"] = "best effort"
                return t_launch, start, solved, f", nudged {change:.2f} m/s toward {where}: best effort {100 * miss:.1f} cm, inside its bounds{timing}"
            entry["status"] = "failed"
            self.record.nudge_failures.append(entry)
            return t_launch, start, solved, f", NUDGE FAILED: best effort misses {where} by {100 * miss:.1f} cm after {change:.1f} m/s{timing}"
        finally:
            entry["solve_ms"] = round(1e3 * (time.perf_counter() - started), 1)
            self.record.nudges.append(entry)

    def _blade_launch(self, role: Role, t: float) -> Array:
        """Where a ball leaves a player's blade at ``t``: the blade centre plus the face (toward the far end) and a ball radius."""

        pose = self.animators[role].pose_at(t, count=False)
        sign_far = END_SIGN[self.rally.end_of(other_role(role))]
        normal = pose.paddle_rotation[:, 2]
        if normal[0] * sign_far < 0.0:
            normal = -normal
        return pose.paddle_position + normal * (self.belief.ball.radius + 0.5 * self.belief.paddle.thickness + 0.003)

    def _swing_contact(self, role: Role, around_s: float, now: float) -> float:
        """The player's swing contact near a recorded time: the blade's peak speed within :data:`SWING_SEARCH_S`, not before ``now``."""

        animator = self.animators[role]
        times = np.arange(max(around_s - SWING_SEARCH_S, now), around_s + SWING_SEARCH_S + 1e-9, 0.01)
        if len(times) == 0:
            return max(around_s, now)
        speeds = [float(np.linalg.norm(animator.reference_paddle(float(t)).velocity)) for t in times]
        return float(times[int(np.argmax(speeds))])

    def _incoming_class(self, hitter: Role, now: float) -> str:
        """The ball a stroke meets: ``placed`` (a point start), ``replayed`` (a continuity ball) or ``live`` (a physical hit)."""

        placed = [(t, label) for t, label in self.record.placements if not label.startswith("drop") and t <= now + 1e-9]
        hit = max(
            (event.time_s for event in self.physics.events if event.surface == "paddle" and event.actor != hitter and event.time_s <= now), default=-math.inf
        )
        if not placed and not math.isfinite(hit):
            return "unknown"
        if placed and placed[-1][0] >= hit:
            return "replayed" if placed[-1][1].startswith("continuity") else "placed"
        return "live"

    def _ball_miss(self, stroke: Any, plan: StrokePlan, now: float) -> dict[str, Any]:
        """
        How far the live ball passes from the stroke's goal (:meth:`_contact_goal`): at the planned contact, and at its
        closest approach within the stroke's time bound. For a live ball this is the physical return's divergence from
        the recording.
        """

        goal, goal_s, source = self._contact_goal(stroke)
        result: dict[str, Any] = {"goal_source": source, "ball_miss_cm": None, "ball_closest_cm": None}
        if plan.ball_at_contact is not None:
            result["ball_miss_cm"] = round(100 * float(np.linalg.norm(plan.ball_at_contact[:3] - goal)), 2)
        window = plan.bounds.max_time_shift_s
        if goal_s + window > now:
            trajectory = predict(self.belief, self.physics.ball_state(), now, horizon_s=goal_s + window + 0.05 - now)
            states = [state_at(trajectory, float(t)) for t in np.linspace(max(now, goal_s - window), goal_s + window, 41)]
            distances = [float(np.linalg.norm(state[:3] - goal)) for state in states if state is not None]
            if distances:
                result["ball_closest_cm"] = round(100 * min(distances), 2)
        return result

    def return_misses(self, window_s: float = RETURN_MISS_WINDOW_S) -> list[dict[str, Any]]:
        """
        Every realised stroke's return against the other player's next goal (:meth:`_contact_goal`): the
        reference-model flight from the contact's post-impact state (spin included), its closest approach within
        ``window_s`` of the goal's recorded time, and its miss at that time. The live-return divergence, per stroke.
        """

        last: dict[int, dict[str, Any]] = {}
        for plan in self.record.plan_details:
            last[int(plan["stroke"])] = plan
        out: list[dict[str, Any]] = []
        for item in self.rally.strokes:
            if item.status != "realised" or item.contact_event is None:
                continue
            following = self._following(item.stroke)
            if following is None or following.hitter == item.stroke.hitter:
                continue
            goal, goal_s, source = self._contact_goal(following)
            event = self.physics.events[item.contact_event]
            if goal_s + window_s <= event.time_s:
                continue
            trajectory = predict(self.belief, np.asarray(event.after, dtype=np.float64), event.time_s, horizon_s=goal_s + window_s - event.time_s)
            states = [state_at(trajectory, float(t)) for t in np.linspace(max(event.time_s, goal_s - window_s), goal_s + window_s, 61)]
            distances = [float(np.linalg.norm(state[:3] - goal)) for state in states if state is not None]
            at = state_at(trajectory, goal_s)
            plan = last.get(item.stroke.index)
            out.append(
                {
                    "stroke": item.stroke.index,
                    "goal_source": source,
                    "plan": "none" if plan is None else ("within" if plan["within_bounds"] else "clamped"),
                    "incoming": None if plan is None else plan.get("incoming"),
                    "closest_cm": round(100 * min(distances), 2) if distances else round(100 * float(np.linalg.norm(trajectory.positions[-1] - goal)), 2),
                    "at_time_cm": round(100 * float(np.linalg.norm(at[:3] - goal)), 2) if at is not None else None,
                    "predicted_cm": None if plan is None else plan.get("predicted_next_miss_cm"),
                }
            )
        return out

    def _cool_down(self, role: Role, placed_at: float) -> None:
        """Keep ``role``'s blade off the ball it releases for :data:`SERVE_COOLDOWN_S` after the placement (which may be later)."""

        delay = max(0, round((placed_at - self.physics.time_s) / self.physics.dt))
        self.physics.cooldown_until[role] = self.physics.step_index + delay + round(SERVE_COOLDOWN_S / self.physics.dt)

    def _swinging(self, role: Role, t: float) -> bool:
        """Whether ``role``'s blade is in a stroke at ``t``: fast enough, and moving toward the far end."""

        _pose, _position, _orientation, velocity = self._paddle_end(role, t, count=False)
        toward = float(velocity[0]) * END_SIGN[self.rally.end_of(other_role(role))]
        return float(np.linalg.norm(velocity)) >= LAUNCH_MIN_BLADE_SPEED_MPS and toward >= LAUNCH_MIN_TOWARD_MPS

    def _hidden_launch(
        self, kind: str, launcher: Role, following: Any, launch_s: float, velocity: Array, spin: Array, entry: dict[str, Any]
    ) -> tuple[float, Array, Array, str]:
        """
        No swing near the recorded hit: start the ball in flight where no camera sees it (off every image, or behind a
        player's body) rather than from a still blade, aimed at the next stroke's goal (or the far half).
        """

        _pose, blade, _orientation, _velocity = self._paddle_end(launcher, launch_s, count=False)
        behind = END_SIGN[self.rally.end_of(launcher)]
        candidates = [
            np.array([blade[0] + behind * depth, blade[1] + side, height])
            for depth in (0.3, 0.6, 1.0, 1.5)
            for height in (0.4, 0.9, 1.4)
            for side in (0.0, -0.5, 0.5, -1.0, 1.0)
        ]
        start = next((point for point in candidates if not self._seen(point)), None)
        self.record.hidden_launches += 1
        if start is None:
            entry |= {"status": "still blade", "stroke": None if following is None else following.index, "source": "none"}
            self._event(f"t={launch_s:.2f} s no swing near the recorded hit and no hidden start found: the ball leaves a still blade", "WARN")
            return launch_s, self._blade_launch(launcher, launch_s), velocity, ", from a still blade (no swing, nowhere hidden)"
        if following is not None and following.hitter != launcher:
            goal, goal_s, source = self._contact_goal(following)
            flight = max(goal_s - launch_s, 0.3)
            guess = (goal - start) / flight + np.array([0.0, 0.0, 0.5 * self.belief.aero.gravity * flight])
            solved, miss = solve_through(self.belief, start, spin, goal, flight, guess)
            entry |= {"status": "hidden start", "stroke": following.index, "source": source, "residual_m": round(miss, 4)}
            phrase = f", started in flight out of view (no swing within {SWING_SEARCH_S:.2f} s), aimed at stroke {following.index}'s {source} within {100 * miss:.1f} cm"
        else:
            target = self._far_target(launcher)
            guess = np.array([(target[0] - start[0]) / CONTINUITY_FLIGHT_S, (target[1] - start[1]) / CONTINUITY_FLIGHT_S, 1.5])
            solved, residual = solve_launch(self.belief, start, spin, targets=[target], flight_time_s=CONTINUITY_FLIGHT_S, initial_velocity=guess, t0=launch_s)
            entry |= {"status": "hidden start", "stroke": None, "source": "far half", "residual_m": round(residual, 4)}
            phrase = f", started in flight out of view (no swing within {SWING_SEARCH_S:.2f} s) toward the far half"
        entry["launch_s"] = round(launch_s, 4)
        return launch_s, start, solved, phrase

    def _following(self, stroke: Any) -> Any:
        """The next stroke of the same point, if any."""

        return next((item for item in self.manifest.strokes if item.point == stroke.point and item.index == stroke.index + 1), None)

    def bounds_for(self, stroke: Any) -> AdaptationBounds:
        """
        A stroke's adaptation bounds: two sigma of its reference's uncertainty per axis (the session's where it has none)
        plus the human re-plan allowance for how early the live ball's deviation from the recorded flight became
        visible (:meth:`replan_for`); 60 ms, 8 cm and 10 degrees when the package carries no uncertainty at all.
        """

        envelope = stroke.envelope or self.manifest.envelope
        if envelope is None:
            return FIXED_V1_BOUNDS
        return evidence_v3_bounds(envelope.time_sd_s, envelope.position_sd_m, envelope.normal_sd_deg, self.replan_for(stroke))

    def _bound_parts(self, stroke: Any, plan: StrokePlan) -> dict[str, Any]:
        """The bounds' two parts per axis (reference, human re-plan), and which part each required adaptation reached."""

        envelope = stroke.envelope or self.manifest.envelope
        if envelope is None:
            return {}
        allowance = self.replan_for(stroke)
        reference = (EVIDENCE_K * envelope.time_sd_s, EVIDENCE_K * envelope.position_sd_m, math.radians(EVIDENCE_K * envelope.normal_sd_deg))
        human = (allowance.time_shift_s, allowance.offset_m, allowance.rotation_rad)
        required = (abs(plan.required_time_shift_s), plan.required_offset_m, plan.required_rotation_rad)

        def part(need: float, ref: float, extra: float) -> str:
            if not math.isfinite(need):
                return "unreachable"
            return "reference" if need <= ref + 1e-9 else ("human" if need <= ref + extra + 1e-9 else "beyond")

        return {
            "replan_regime": allowance.regime,
            "seen_before_contact_ms": None if allowance.seen_s is None else round(1e3 * allowance.seen_s, 1),
            "ref_time_ms": round(1e3 * reference[0], 1),
            "human_time_ms": round(1e3 * human[0], 1),
            "ref_offset_cm": round(100 * reference[1], 2),
            "human_offset_cm": round(100 * human[1], 2),
            "ref_tilt_deg": round(math.degrees(reference[2]), 2),
            "human_tilt_deg": round(math.degrees(human[2]), 2),
            "part_time": part(required[0], reference[0], human[0]),
            "part_offset": part(required[1], reference[1], human[1]),
            "part_tilt": part(required[2], reference[2], human[2]),
        }

    def replan_for(self, stroke: Any) -> ReplanAllowance:
        """
        The human re-plan allowance of a stroke: from when its live incoming ball (the previous stroke's physical return)
        first left the recorded flight by more than the reference envelope's position allowance. A placed or replayed
        ball is the recording itself, so it leaves nothing to re-plan.
        """

        cached = self._replans.get(stroke.index)
        if cached is not None:
            return cached
        allowance = replan_allowance(self._seen_before_contact(stroke), self.human_limits)
        self._replans[stroke.index] = allowance
        return allowance

    def _seen_before_contact(self, stroke: Any) -> float | None:
        previous = next((item for item in self.manifest.strokes if item.point == stroke.point and item.index == stroke.index - 1), None)
        if previous is None or previous.fitted is None:
            return None
        record = next((item for item in self.rally.strokes if item.stroke.index == previous.index), None)
        if record is None or record.status != "realised" or record.contact_event is None:
            return None
        event = self.physics.events[record.contact_event]
        envelope = stroke.envelope or self.manifest.envelope
        threshold = EVIDENCE_K * (envelope.position_sd_m if envelope is not None else 0.023)
        fitted = previous.fitted
        recorded_velocity = np.asarray(fitted.velocity_mps, dtype=np.float64)
        if stroke.paddle is not None and stroke.time_s - previous.time_s >= 0.1:
            goal = np.asarray(stroke.paddle.contact_point_m) + np.asarray(stroke.paddle.normal) * self.belief.ball.radius
            solved, miss = solve_through(
                self.belief, np.asarray(fitted.position_m), np.asarray(fitted.spin_radps), goal, stroke.time_s - previous.time_s, recorded_velocity
            )
            if miss <= CONTINUITY_MAX_MISS_M and float(np.linalg.norm(solved - recorded_velocity)) <= CONTINUITY_MAX_NUDGE_MPS:
                recorded_velocity = solved
        horizon = stroke.time_s - previous.time_s + 0.05
        recorded = predict(
            self.belief, np.concatenate([np.asarray(fitted.position_m), recorded_velocity, np.asarray(fitted.spin_radps)]), previous.time_s, horizon_s=horizon
        )
        live = predict(self.belief, np.asarray(event.after, dtype=np.float64), event.time_s, horizon_s=max(0.05, stroke.time_s - event.time_s + 0.05))
        for t in np.arange(max(previous.time_s, event.time_s), stroke.time_s, 0.01):
            a, b = state_at(recorded, float(t)), state_at(live, float(t))
            if a is None or b is None:
                continue
            if float(np.linalg.norm(a[:3] - b[:3])) > threshold:
                return float(stroke.time_s - t)
        return None

    def _reference_details(self, hitter: Role, stroke: int) -> dict[str, Any]:
        """The stroke's reference correction (the capture's error against the reconstructed contact state), if any."""

        correction = self.references.get(hitter, {}).get(stroke)
        if correction is None:
            return {"reference": "recorded paddle"}
        return {
            "reference": "reconstructed contact state",
            "reference_offset_cm": round(correction.offset_cm, 2),
            "reference_tilt_deg": round(correction.tilt_deg, 2),
            "recorded_speed_mps": round(correction.recorded_speed_mps, 2),
            "reconstructed_speed_mps": round(correction.reconstructed_speed_mps, 2),
            "reference_velocity_change_mps": round(correction.velocity_change_mps, 2),
        }

    def _continuity(self, action: ContinuityAction, now: float) -> None:
        stroke = action.stroke
        hitter = stroke.hitter
        at = max(stroke.time_s, now)
        if stroke.fitted is not None:
            fitted = stroke.fitted
            # The replayed flight leaves the hitter's blade, at the blade's actual pose, with the recorded velocity
            # nudged to meet the next recorded contact.
            at, start, velocity, through = self._aimed_launch(
                "continuity",
                hitter,
                self._following(stroke),
                at,
                stroke.time_s,
                now,
                np.asarray(fitted.velocity_mps, dtype=np.float64),
                np.asarray(fitted.spin_radps, dtype=np.float64),
                cap_mps=CONTINUITY_MAX_NUDGE_MPS,
            )
            self.physics.place_ball(start, velocity, np.asarray(fitted.spin_radps), label=f"continuity stroke {stroke.index}", at_time=at, retire=True)
            self._cool_down(hitter, at)
            self.record.placements.append((at, f"continuity stroke {stroke.index}"))
            self.continuity_live = True
            self._set_ball_visible(True)
            self._event(
                f"t={at:.2f} s CONTINUITY ball for stroke {stroke.index} ({hitter}): the recorded flight replayed from the tracker's fit "
                f"({np.linalg.norm(fitted.velocity_mps):.1f} m/s{through}); not physics evidence",
                "WARN",
            )
            return
        pose = self.animators[hitter].pose_at(at, count=False)
        sign_far = END_SIGN[self.rally.end_of(other_role(hitter))]
        normal = pose.paddle_rotation[:, 2]
        if normal[0] * sign_far < 0.0:
            normal = -normal
        position = pose.paddle_position + normal * (self.belief.ball.radius + 0.5 * self.belief.paddle.thickness + 0.003)
        target = stroke.landing.centre_m if stroke.landing is not None else self._far_target(hitter)
        direction = np.array([target[0] - position[0], target[1] - position[1], 0.0])
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        spin = np.asarray(CONTINUITY_TOPSPIN_RADPS * np.cross((0.0, 0.0, 1.0), direction), dtype=np.float64)
        guess = np.array([(target[0] - position[0]) / CONTINUITY_FLIGHT_S, (target[1] - position[1]) / CONTINUITY_FLIGHT_S, 1.5])
        velocity, residual = solve_launch(self.belief, position, spin, targets=[target], flight_time_s=CONTINUITY_FLIGHT_S, initial_velocity=guess, t0=at)
        at, position, velocity, through = self._aimed_launch(
            "continuity", hitter, self._following(stroke), at, stroke.time_s, now, velocity, spin, cap_mps=CONTINUITY_MAX_NUDGE_MPS
        )
        self.physics.place_ball(position, velocity, spin, label=f"continuity stroke {stroke.index}", at_time=at, retire=True)
        self._cool_down(hitter, at)
        self.record.placements.append((at, f"continuity stroke {stroke.index}"))
        self.continuity_live = True
        self._set_ball_visible(True)
        self._event(
            f"t={at:.2f} s CONTINUITY ball for stroke {stroke.index} ({hitter}) toward ({target[0]:.2f}, {target[1]:.2f}), residual {residual:.3f}{through}; "
            "not physics evidence",
            "WARN",
        )

    # ---- phases --------------------------------------------------------------------------

    def drop_tests(self) -> None:
        """
        Pose everyone at the start, warm the renderer up with physics paused, then pre-roll the opening drop tests: a
        ball released 30 cm above the table, whose rebound the physical-invariants test holds to the ITTF band.
        """

        ticks = math.ceil(PRE_ROLL_S / RENDER_DT)
        self.start(ticks)
        self.warm_up(WARMUP_FRAMES)
        self.pre_roll(ticks)

    def start(self, pre_roll_ticks: int) -> None:
        """Pose everyone at the replay's start, place the paddles, park the ball and start the physics callback."""

        self.physics.time_offset_s = self.start_s - pre_roll_ticks * RENDER_DT
        poses: dict[str, tuple[Array, Array]] = {}
        for role, handle in self.characters.items():
            pose, position, orientation, velocity = self._paddle_end(role, self.start_s)
            handle.write_pose(pose.local_rotation, pose.root_translation, self.animators[role].translated, pose.local_translation)
            self.paddle_state[role] = (position, orientation, np.zeros(3))
            poses[role] = (position, orientation)
            self.physics.segments[role] = PaddleSegment(
                t0=self.physics.time_offset_s, t1=self.start_s, p0=position, v0=np.zeros(3), p1=position, v1=np.zeros(3), q0=orientation, q1=orientation
            )
            del velocity
        self.physics.place_paddles(poses)
        self.physics.park_ball(PARK_POSITION_M)
        self.show_ball(False)
        self.ball_visible = False
        self.physics.register(self.world)

    def warm_up(self, frames: int) -> None:
        """
        Render ``frames`` frames with physics paused (the session clock does not move), so RTX converges before any logged frame.

        ``World.render`` advances Kit's renderer without stepping physics; if
        a build steps physics there after all, the steps are counted and the
        parked ball and held paddles keep the scene unchanged.
        """

        started = time.perf_counter()
        callbacks = self.physics.stats.callbacks
        render = getattr(self.world, "render", None)
        for _ in range(frames):
            self.run.raise_if_cancelled()
            if render is not None:
                render()
            else:
                self.world.step(render=True)
        self._check_physics()
        self.record.warmup_frames = frames
        self.record.warmup_wall_s = time.perf_counter() - started
        self.record.warmup_physics_steps = self.physics.stats.callbacks - callbacks

    def pre_roll(self, ticks: int) -> None:
        """Render the pre-roll and run the drop tests (the session clock has not started)."""

        started = time.perf_counter()
        radius = self.belief.ball.radius
        reference = reference_drop_rebound(self.physics.params, speed_curve=self.physics.settings.table.mode == "hybrid")
        drop = -1
        drop_started = -math.inf
        drop_event_cursor = 0
        for tick in range(ticks):
            self.run.raise_if_cancelled()
            self._check_physics()
            now = self.physics.time_s
            waiting = drop >= 0 and drop < DROP_TESTS
            if waiting:
                table = next((event for event in self.physics.events[drop_event_cursor:] if event.surface == "table"), None)
                done = table is not None and table.index in self.physics.apexes
                if done or now - drop_started > DROP_TIMEOUT_S:
                    rebound = self.physics.apexes[table.index] - radius if done and table is not None else math.nan
                    restitution = table.normal_restitution if table is not None else math.nan
                    self.record.drops.append(
                        DropTest(drop_height_m=DROP_HEIGHT_M + drop * DROP_STAGGER_M, rebound_m=rebound, reference_m=reference, restitution=restitution)
                    )
                    self._event(f"drop test {drop}: rebound {1e3 * rebound:.1f} mm (reference {1e3 * reference:.1f} mm), e {restitution:.3f}")
                    drop += 1
                    waiting = False
            if not waiting and drop < DROP_TESTS and tick >= 3:
                drop = max(drop, 0)
                if drop < DROP_TESTS:
                    height = DROP_HEIGHT_M + drop * DROP_STAGGER_M
                    self.physics.place_ball((-0.55 - 0.08 * drop, 0.25, radius + height), np.zeros(3), np.zeros(3), label=f"drop {drop}")
                    drop_started = now
                    drop_event_cursor = len(self.physics.events)
                    self._set_ball_visible(True)
                    self.record.placements.append((now, f"drop {drop}"))
            if drop >= DROP_TESTS and self.physics.parked_at is None and self.physics.pending is None:
                self.physics.park_ball(PARK_POSITION_M)
                self._set_ball_visible(False)
            self.world.step(render=True)
        if drop < DROP_TESTS:
            raise RuntimeError(f"the pre-roll ended after {len(self.record.drops)} of {DROP_TESTS} drop tests; lengthen PRE_ROLL_S")
        self.physics.park_ball(PARK_POSITION_M)
        self._set_ball_visible(False)
        self.event_cursor = len(self.physics.events)
        self.record.pre_roll_ticks = ticks
        self.record.pre_roll_wall_s = time.perf_counter() - started

    def play(self) -> None:
        """Run the replay ticks until the rally is finished or the cap is reached."""

        record = self.record
        record.game_start_s = self.start_s
        loop_started = time.perf_counter()
        tick = 0
        cap = self.start_s + self.max_sim_s
        last_cost = (0.0, 0.0, 0.0)
        shown_ball: Array | None = None
        while True:
            self.run.raise_if_cancelled()
            self._check_physics()
            tick_started = time.perf_counter()
            t0 = self.physics.time_s
            t1 = t0 + RENDER_DT
            events = self.physics.take_events(self.event_cursor)
            self.event_cursor += len(events)
            live_events = [event for event in events if event.time_s >= self.start_s]
            self.rally.on_events(live_events, t0)
            if live_events:
                self.recorder.log_contacts(live_events)
                for event in live_events:
                    if event.surface == "paddle":
                        self.continuity_live = False
                        self._event(
                            f"t={event.time_s:.3f} s {event.actor} paddle contact: in {np.linalg.norm(event.before[3:6]):.1f} m/s, out {np.linalg.norm(event.after[3:6]):.1f} m/s, "
                            f"blade offset {100 * np.hypot(*(event.blade_offset_m or (0.0, 0.0))):.1f} cm, {event.regime}"
                        )
                    else:
                        self._event(f"t={event.time_s:.3f} s {event.surface} contact on the {event.side} side ({event.mode}), e {event.normal_restitution:.3f}")
            for time_s, level, text in self.rally.log[self.log_cursor :]:
                self._event(f"t={time_s:.2f} s {text}", level)
            self.log_cursor = len(self.rally.log)
            if self.rally.finished:
                record.terminal_reached = True
                break
            if t1 > self.session_end_s:
                # The recording ends here: the replay ends with it rather than playing past the last recorded frame.
                record.terminal_reached = True
                self._event(f"t={t0:.2f} s the recorded session ends; the game ends with it")
                break
            # A dead ball is not parked: it plays out (bounce, roll) until the next placement retires it, and a retired
            # ball is parked only once it is off every camera.
            self._tend_retired(t0)
            if t0 >= cap:
                self._event(f"t={t0:.2f} s the {self.max_sim_s:.0f} s cap ended the game before its terminal condition", "ERROR")
                break
            for action in self.rally.due(t0, t1):
                if isinstance(action, ServeAction):
                    self._serve(action, t0)
                elif isinstance(action, PlanAction):
                    self._plan(action, t0)
                else:
                    self._continuity(action, t0)
            for time_s, level, text in self.rally.log[self.log_cursor :]:
                self._event(f"t={time_s:.2f} s {text}", level)
            self.log_cursor = len(self.rally.log)
            poses = self._pose_players(t0, t1)
            world_joints = self._pose_bodies(t0, t1, poses)
            step_started = time.perf_counter()
            callbacks = self.physics.stats.callbacks
            physics_seconds = self.physics.stats.seconds
            self.world.step(render=True)
            step_wall = time.perf_counter() - step_started
            frame = self.recorder.broadcast.read().get("broadcast")
            if self.recorder.spool is not None:
                if frame is None:
                    raise RuntimeError(f"the broadcast camera delivered no frame at tick {tick}")
                self.recorder.spool.submit(frame)
            skeletons = self._sample_motion(poses, world_joints)
            ball = self.physics.ball_state()
            in_play = self.physics.parked_at is None
            images = {name: self._isp(name, image) for name, image in self.recorder.orbbecs.read().items() if image is not None}
            if frame is not None:
                images["broadcast"] = frame
            # A render product's annotator returns the frame rendered one step earlier (the reprojected ball otherwise
            # runs one tick of travel ahead), so these frames show t0.
            self.recorder.log_frames(images, shown_ball, self.recording_time(t0))
            self.video_frames += 1
            self._log_real_frames(t1)
            shown_ball = ball[:3].copy() if in_play else None
            if tick % self.telemetry_every == 0:
                if self.physics.spares:
                    rows = self.physics.read()[0]
                    self.recorder.log_retired({index: rows[row] for index, row in enumerate(self.physics.spare_rows()) if not self.physics.spare_parked[index]})
                paddles = {role: (state[0], rot.to_matrix(state[1]), state[2]) for role, state in self.paddle_state.items()}
                self.recorder.log_tick(
                    time_s=t1,
                    ball=ball,
                    ball_in_play=in_play,
                    continuity=self.continuity_live,
                    prediction=self.prediction,
                    paddles=paddles,
                    skeletons=skeletons,
                    cost=last_cost,
                )
            if self.check_usdskel and tick % 150 == 0:
                self._check_fk(poses)
            steps = max(1, self.physics.stats.callbacks - callbacks)
            tick_wall = time.perf_counter() - tick_started
            record.tick_ms.append(1e3 * tick_wall)
            elapsed = time.perf_counter() - loop_started
            last_cost = (elapsed / max(t1 - self.start_s, 1e-6), 1e3 * tick_wall, 1e3 * (self.physics.stats.seconds - physics_seconds) / steps)
            del step_wall
            tick += 1
            if tick % 300 == 0:
                print(f"[game] tick {tick} t={t1:.2f} s wall/sim {last_cost[0]:.2f}", flush=True)
        self.physics.body_motion.clear()
        self.physics.body_excluded.clear()
        record.ticks = tick
        record.game_end_s = self.physics.time_s
        record.loop_wall_s = time.perf_counter() - loop_started

    def _check_fk(self, poses: dict[str, AnimatedPose]) -> None:
        """Compare this module's hand position with UsdSkel's own computation (the paddle sits in the rendered hand)."""

        from lib.replay.players.bodies import usdskel_joint_world

        for role, pose in poses.items():
            handle = self.characters[role]
            animator = self.animators[role]
            _rotations, positions = forward_kinematics(animator.skeleton, pose.local_rotation, pose.root_translation, animator.placement)
            joint = animator.track.character.grip.hand_joint
            ours = positions[handle.skeleton.index(joint)]
            theirs = usdskel_joint_world(self.stage, handle, joint)
            self.record.fk_check_mm.append(float(1e3 * np.linalg.norm(ours - theirs)))


__all__ = ["DROP_HEIGHT_M", "PlanSummary", "ReplayLoop", "ReplayRecord", "reference_drop_rebound"]
