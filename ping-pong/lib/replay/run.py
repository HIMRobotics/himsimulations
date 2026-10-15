"""
``replay_him_session``: one recorded HIM session replayed in Isaac Sim with live ball physics, and graded.

The ball is never replayed: it is live, calibrated physics from the first serve. Both players follow their recorded
motion (the suited player's motion capture, the opponent's motion inferred from the overhead video) and bend each
recorded stroke, within human bounds, to meet the ball where the physics actually brings it. The virtual overhead
Orbbecs render beside the recorded frames at 24 fps, so every moment of the replay can be compared with the real one.

:func:`replay_him_session` reads top to bottom in the order things happen: load the session, build the lab, attach the
live ball, spawn the players, open the recorder, run the ITTF drop tests, play every point, then grade. The run passes
when it ends on its terminal condition (the last point ends physically) and its three tests pass: motion QA, physical
invariants and telemetry completeness (``lib/replay/gates.py``). The negative control's preset expects them to fail.

What to change lives in ``lib/config.py``: a :class:`~lib.config.ReplayParams` preset picks the session (1-7), how
many points to play, and the engine's ball physics (``TABLE_PHYSICS``). Add a preset to make a new case.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import antioch
from pydantic import BaseModel, ConfigDict

from lib.config import TABLE_PHYSICS, ReplayParams
from lib.him_lab import (
    BROADCAST,
    AuthoredCamera,
    OrbbecCalibration,
    author_him_cameras,
    build_him_lab,
    clear_stage,
    load_calibration_set,
    orbbec_pinhole,
    presentation_pinhole,
)
from lib.physics.params import BallTableParams, nominal_params
from lib.replay.ball import GAME_PHYSICS_DT, RENDER_DT, LiveBall, attach_live_ball, bind_live_ball, set_physics_dt
from lib.replay.gates import GATE_VERSION, GateResult, evaluate_invariants, evaluate_motion, evaluate_telemetry
from lib.replay.loop import LEAD_S, ReplayLoop
from lib.replay.players import spawn_players
from lib.replay.recorder import VIDEO_FPS, Recorder
from lib.replay.session import LoadedSessionPackage, load_session_package


class ReplayReport(BaseModel):
    """
    What one replay measured.

    :ivar session: The recorded session.
    :ivar table_physics: The ball physics the engine ran.
    :ivar gate_version: Version of the frozen thresholds.
    :ivar tests: Each test's verdict: motion QA, physical invariants, telemetry completeness.
    :ivar terminal_reached: Whether the replay ended on its terminal condition, not the cap.
    :ivar tests_passed: Every test passed and the terminal condition was reached.
    :ivar expect_pass: Whether the preset expects that (``False`` for a negative control).
    :ivar passed: The verdict: the tests came out as expected.
    :ivar realised_share: Recorded strokes the simulation realised with a physical paddle contact (reported, not graded).
    :ivar counts: Stroke outcomes, point endings and continuity balls.
    :ivar artifact_directory: Where the reports were written.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: int
    table_physics: str
    gate_version: str
    tests: tuple[GateResult, ...]
    terminal_reached: bool
    tests_passed: bool
    expect_pass: bool
    passed: bool
    realised_share: float | None
    counts: dict[str, int]
    artifact_directory: str


def replay_him_session(run: antioch.ScenarioRun, params: ReplayParams) -> ReplayReport:
    """
    Replay one recorded HIM session with live ball physics, and grade it.

    :param run: The active scenario run.
    :param params: The session, the points to play, the engine's ball physics and the cap.
    :return: The report; its checks are recorded on ``run``.
    """

    session = load_him_session(params)  # mocap, serves, strokes, camera epochs, real frames
    physics = engine_physics(params)  # the calibrated priors, or the negative control's dead table
    world, stage = antioch.world(), antioch.stage()
    world.stop()
    ball: LiveBall | None = None
    recorder: Recorder | None = None
    try:
        build_him_lab(stage, physics)  # room, table, net, colliders, calibrated lighting
        cameras = author_him_cameras(stage, camera_calibrations(session))  # the Orbbecs at the session's epochs, the broadcast view
        ball = attach_live_ball(world, stage, physics, roles(session))  # hybrid contacts, calibrated impact law, RK4 at 3840 Hz
        world.reset()
        bind_live_ball(world, ball)
        players = spawn_players(stage, session)  # the recorded player and the inferred opponent (ActorCore bodies)
        recorder = open_recorder(run, params, cameras, session, physics)  # simulated beside real at 24 fps, and the MP4
        replay = ReplayLoop(run, session, nominal_params(), ball.physics, players.bodies, players.animators, recorder, max_sim_s=params.max_sim_s)
        replay.drop_tests()  # ITTF 30 cm drops: the table's physical invariant
        replay.play()  # every point, each player bending its recorded strokes to meet the live ball, until the last ends physically
        return grade(run, params, session, replay)  # motion QA, physical invariants, telemetry completeness, the terminal condition
    finally:
        if recorder is not None:
            recorder.close()
        if ball is not None:
            ball.physics.unregister(world)
        world.stop()
        clear_stage(stage)
        set_physics_dt(world, GAME_PHYSICS_DT)


# ---- the steps --------------------------------------------------------------------------------------------------------


def load_him_session(params: ReplayParams) -> LoadedSessionPackage:
    """The session's package from the asset store, restricted to the points to play."""

    package = load_session_package(params.package)
    return package.restricted(params.points) if params.points is not None else package


def engine_physics(params: ReplayParams) -> BallTableParams:
    """The ball physics the engine runs: the calibrated priors with the preset's overrides (the players always believe the priors)."""

    nominal = nominal_params()
    overrides = TABLE_PHYSICS[params.table_physics]
    return nominal.with_values(overrides) if overrides else nominal


def roles(session: LoadedSessionPackage) -> tuple[str, ...]:
    """The players' roles, in the package's order."""

    return tuple(track.role for track in session.manifest.tracks)


def camera_calibrations(session: LoadedSessionPackage) -> dict[str, OrbbecCalibration]:
    """Each overhead camera's calibration at the replay's start (its extrinsic epoch from the package)."""

    manifest = session.manifest
    start_s = manifest.serves[0].time_s - LEAD_S
    calibrations = load_calibration_set()
    return {label: calibrations.select(label, epoch=manifest.calibration.epoch_at(label, start_s).epoch) for label in ("8E", "8F")}  # type: ignore[arg-type]


def open_recorder(
    run: antioch.ScenarioRun, params: ReplayParams, cameras: dict[str, AuthoredCamera], session: LoadedSessionPackage, physics: BallTableParams
) -> Recorder:
    """
    The recorder: render products for every camera, one H.264 stream per camera and per recorded Orbbec view, the
    blueprint, the static scene and, with a clip, the compositor's MP4 spool (the compositor must be healthy first).
    """

    from lib.telemetry.cameras import CameraRig
    from lib.telemetry.compositor import CompositorClient

    directory = Path(params.artifacts_dir) / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    pinholes = {f"orbbec_{label}": orbbec_pinhole(calibration) for label, calibration in camera_calibrations(session).items()}
    pinholes["broadcast"] = presentation_pinhole(BROADCAST)
    recorder = Recorder(
        pinholes,
        roles(session),
        CameraRig({name: camera for name, camera in cameras.items() if camera.calibrated}),
        CameraRig({"broadcast": cameras["broadcast"]}),
        directory=directory,
        compositor=CompositorClient() if params.clip else None,
    )
    recorder.send_blueprint(run)
    recorder.log_static((physics.table.length, physics.table.width, physics.table.height), physics.net.height, physics.net_half_span)
    recorder.open_video()
    return recorder


def grade(run: antioch.ScenarioRun, params: ReplayParams, session: LoadedSessionPackage, replay: ReplayLoop) -> ReplayReport:
    """
    Close the video, encode the MP4, run the three tests, and record the verdict, the results and the reports on ``run``.
    """

    record, physics, recorder = replay.record, replay.physics, replay.recorder
    physics.unregister(replay.world)
    recorder.close_video()
    clip_frames, clip_duration = encode_clip(run, params, recorder)
    stats = physics.stats
    tunnels = {
        "table": stats.tunnel_table,
        "net": stats.tunnel_net,
        "floor": stats.tunnel_floor,
        "paddle": stats.tunnel_paddle,
        "room": stats.left_room,
        "frame": stats.tunnel_frame,
    }
    for role, samples in record.motion.items():
        samples.held_frames = replay.animators[role].held_frames
    tests = (
        evaluate_motion(record.motion),
        evaluate_invariants(
            [event for event in physics.events if event.time_s >= replay.start_s],
            tunnels,
            record.drops,
            point_starts=[time_s for time_s, label in record.placements if not label.startswith("drop")],
            net_regions=physics.event_regions,
        ),
        evaluate_telemetry(
            ticks_3d=recorder.counts.ticks_3d,
            expected_ticks_3d=math.ceil(record.ticks / replay.telemetry_every),
            rate_3d_hz=1.0 / (replay.telemetry_every * RENDER_DT),
            images=recorder.counts.images,
            expected_images=record.ticks,
            images_dropped=sum(video.dropped for video in recorder.counts.video.values()),
            video_fps=VIDEO_FPS,
            real_frames=recorder.counts.real_frames,
            real_frames_expected=replay.real_frames_in_span(),
            events=recorder.counts.events,
            blueprint=recorder.counts.blueprint,
            clip_frames=clip_frames,
            rendered_frames=record.ticks,
            clip_duration_s=clip_duration,
            fps=1.0 / RENDER_DT,
        ),
    )
    tests_passed = all(test.passed for test in tests) and record.terminal_reached
    rally = replay.rally
    report = ReplayReport(
        session=params.session,
        table_physics=params.table_physics,
        gate_version=GATE_VERSION,
        tests=tests,
        terminal_reached=record.terminal_reached,
        tests_passed=tests_passed,
        expect_pass=params.expect_pass,
        passed=tests_passed == params.expect_pass,
        realised_share=round(rally.realised_share(), 4) if rally.strokes else None,
        counts=rally.counts(),
        artifact_directory=str(recorder.directory),
    )
    record_verdict(run, params, report)
    record_results(run, session, replay, report)
    return report


def encode_clip(run: antioch.ScenarioRun, params: ReplayParams, recorder: Recorder) -> tuple[int | None, float | None]:
    """Encode the spooled broadcast frames into the MP4 through the compositor and attach it; ``(None, None)`` without a clip."""

    if recorder.spool is None or recorder.compositor is None:
        return None, None
    recorder.spool.close()
    result = recorder.compositor.encode(
        recorder.spool,
        run_id=run.scenario_run_id or "local",
        scenario=run.name,
        clip=f"session-{params.session}-{params.table_physics}",
        fps=1.0 / RENDER_DT,
        destination=recorder.directory / "broadcast.mp4",
        cancel_check=run.raise_if_cancelled,
    )
    run.add_artifact(result.path, name="broadcast.mp4", content_type="video/mp4", description="the broadcast camera, every rendered frame, H.264")
    return result.receipt.frames_written, result.receipt.probe.duration_s


def record_verdict(run: antioch.ScenarioRun, params: ReplayParams, report: ReplayReport) -> None:
    """The run's checks: each test and the terminal condition, or for a negative control, that its tests fail."""

    if not params.expect_pass:
        detail = "; ".join(f"{test.name}: {'pass' if test.passed else 'FAIL'} ({test.detail})" for test in report.tests)
        run.check("the negative control fails its tests", not report.tests_passed, detail=f"{params.table_physics}: {detail}")
        return
    run.check("the replay ended on its terminal condition, not the cap", report.terminal_reached, detail="every point played and the last ended physically")
    for test in report.tests:
        run.check(test.name, test.passed, detail=test.detail)


def record_results(run: antioch.ScenarioRun, session: LoadedSessionPackage, replay: ReplayLoop, report: ReplayReport) -> None:
    """Scalar results for the console; the full report and the contact log as artifacts."""

    record, physics, manifest = replay.record, replay.physics, session.manifest
    sim_s = record.game_end_s - record.game_start_s
    results: dict[str, Any] = {
        "session": report.session,
        "table_physics": report.table_physics,
        "expect_pass": report.expect_pass,
        "tests_passed": report.tests_passed,
        "gate_version": report.gate_version,
        "package": f"{manifest.package_id} (sha256 {session.sha256[:16]})",
        "terminal_reached": report.terminal_reached,
        "points_played": len(replay.rally.points),
        "strokes_scheduled": len(replay.rally.strokes),
        "realised_share": report.realised_share,
        **report.counts,
        "sim_s": round(sim_s, 3),
        "ticks": record.ticks,
        "wall_per_sim_s": round(record.loop_wall_s / max(sim_s, 1e-6), 3),
        "contacts_hybrid": physics.stats.contacts_hybrid,
        "arm_ik_residual_max_mm": round(1e3 * record.ik_residual_m, 2),
        "hidden_launches": record.hidden_launches,
        "nudges_failed": len(record.nudge_failures),
    }
    for test in report.tests:
        key = test.name.replace(" ", "_")
        results[f"test_{key}"] = test.passed
        results.update({f"{key}_{name}": value for name, value in test.measurements.items()})
    run.add_results({key: value for key, value in results.items() if value is not None})
    directory = Path(report.artifact_directory)
    payload = {
        "report": report.model_dump(mode="json"),
        "results": results,
        "drops": [drop.__dict__ for drop in record.drops],
        "plans": record.plan_details,
        "strokes": [
            {"index": item.stroke.index, "point": item.stroke.point, "hitter": item.stroke.hitter, "status": item.status, "contact_time_s": item.contact_time_s}
            for item in replay.rally.strokes
        ],
        "points": [{"index": point.serve.index, "server": point.serve.server, "endings": point.endings} for point in replay.rally.points],
        "placements": record.placements,
        "nudges": record.nudges,
        "returns": replay.return_misses(),
        "rally_log": replay.rally.log,
    }
    path = directory / "replay_report.json"
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    run.add_artifact(path, description="replay report: tests, results, drop tests, plans, strokes, points, placements")
    contacts = directory / "contacts.json"
    contacts.write_text(
        json.dumps([contact_record(event, physics.event_regions.get(index)) for index, event in enumerate(physics.events)], indent=1), encoding="utf-8"
    )
    run.add_artifact(contacts, description="every ball contact (the drop tests included) with its pre- and post-impact state")


def contact_record(event: Any, region: str | None = None) -> dict[str, Any]:
    """One contact event as JSON."""

    def rounded(values: Any) -> list[float]:
        return [round(float(v), 5) for v in values]

    return {
        "index": event.index,
        "time_s": round(event.time_s, 5),
        "surface": event.surface,
        "side": event.side,
        "mode": event.mode,
        "actor": event.actor,
        "position": rounded(event.position),
        "normal": rounded(event.normal),
        "before": rounded(event.before),
        "after": rounded(event.after),
        "approach_speed_mps": round(event.approach_speed_mps, 4),
        "normal_restitution": None if not math.isfinite(event.normal_restitution) else round(event.normal_restitution, 4),
        "regime": event.regime,
        "region": region,
    }


__all__ = ["ReplayReport", "replay_him_session"]
