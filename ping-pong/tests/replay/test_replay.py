"""
The replay end to end on CPU, on the first point of HIM session 2 (:mod:`tests.replay.offline`), and its tests.

The engine boundary is a stand-in; the stepper, the stroke planner, the referee, the players' motion and the tests are
the production code. The nominal point passes motion QA and the physical invariants and ends physically; the dead
table (the negative control, restitution 0.30) fails the physical invariants on its drop tests. This is not a
simulation run: the evidence comes from the recorded Antioch runs.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from lib.config import REPLAY_PRESETS
from lib.physics.params import nominal_params
from lib.replay.gates import DropTest, evaluate_invariants, evaluate_motion, evaluate_telemetry
from lib.replay.loop import ReplayLoop
from lib.replay.run import engine_physics
from lib.replay.session import read_session_package
from tests.replay.offline import FIXTURE, offline_replay


def _replay(preset: str) -> ReplayLoop:
    return offline_replay(read_session_package(FIXTURE), engine_physics(REPLAY_PRESETS[preset]), nominal_params())


def _invariants(replay: ReplayLoop) -> bool:
    physics, record = replay.physics, replay.record
    stats = physics.stats
    tunnels = {"table": stats.tunnel_table, "net": stats.tunnel_net, "floor": stats.tunnel_floor, "paddle": stats.tunnel_paddle, "room": stats.left_room}
    events = [event for event in physics.events if event.time_s >= replay.start_s]
    starts = [time_s for time_s, label in record.placements if not label.startswith("drop")]
    return evaluate_invariants(events, tunnels, record.drops, point_starts=starts, net_regions=physics.event_regions).passed


@pytest.fixture(scope="module")
def nominal() -> ReplayLoop:
    return _replay("session-2-first-point")


def test_the_first_point_ends_physically_and_passes_motion_qa_and_the_physical_invariants(nominal: ReplayLoop) -> None:
    record = nominal.record
    assert record.terminal_reached and nominal.rally.finished
    assert len(record.drops) == 5 and all(0.23 <= drop.rebound_m <= 0.26 for drop in record.drops)
    for role, samples in record.motion.items():
        samples.held_frames = nominal.animators[role].held_frames
    motion = evaluate_motion(record.motion)
    assert motion.passed, motion.detail
    assert _invariants(nominal)
    counts = nominal.rally.counts()
    assert counts["strokes_pending"] == counts["strokes_planned"] == 0, "every recorded stroke was played"
    assert counts["strokes_realised"] >= 4, counts
    assert sum(1 for event in nominal.physics.events if event.surface == "paddle") >= counts["strokes_realised"]


def test_every_frame_is_logged_and_every_placed_ball_is_aimed(nominal: ReplayLoop) -> None:
    record, recorder = nominal.record, nominal.recorder
    recorder.close_video()
    assert recorder.counts.images == {"cams/orbbec_8E/sim": record.ticks}, "every rendered frame, 24 fps"
    assert nominal.telemetry_every == 2 and recorder.counts.ticks_3d == math.ceil(record.ticks / 2), "3D telemetry at 12 Hz"
    assert record.warmup_frames == 900 and record.warmup_physics_steps == 0, "RTX converges with physics paused"
    placed = [label for _time, label in record.placements if not label.startswith("drop")]
    assert len(record.nudges) == len(placed) > 0
    assert all(nudge["status"] in ("nudged", "best effort", "failed", "no next stroke", "no flight", "hidden start") for nudge in record.nudges)


def test_the_dead_table_negative_control_fails_its_drop_tests() -> None:
    replay = _replay("nc-dead-table")
    assert replay.record.terminal_reached, "the negative control still ends physically"
    assert all(drop.rebound_m < 0.1 for drop in replay.record.drops)
    assert not _invariants(replay)


def test_the_invariants_hold_the_drop_rebound_to_the_ittf_band() -> None:
    def drops(rebound: float) -> list[DropTest]:
        return [DropTest(drop_height_m=0.3, rebound_m=rebound, reference_m=0.242, restitution=0.9)] * 5

    assert evaluate_invariants([], {"table": 0}, drops(0.242)).passed
    assert not evaluate_invariants([], {"table": 0}, drops(0.27)).passed
    assert not evaluate_invariants([], {"table": 1}, drops(0.242)).passed, "a tunnelling fails"
    assert not evaluate_invariants([], {"table": 0}, []).passed, "no drop test fails"


def test_telemetry_needs_every_frame_the_recorded_frames_and_the_mp4() -> None:
    complete: dict[str, Any] = {
        "ticks_3d": 50,
        "expected_ticks_3d": 50,
        "rate_3d_hz": 12.0,
        "images": {"cams/broadcast": 100},
        "expected_images": 100,
        "images_dropped": 0,
        "video_fps": 24.0,
        "real_frames": {"orbbec_8E": 80},
        "real_frames_expected": {"orbbec_8E": 80},
        "events": 3,
        "blueprint": True,
        "clip_frames": 100,
        "rendered_frames": 100,
        "clip_duration_s": 100 / 24,
        "fps": 24.0,
    }
    assert evaluate_telemetry(**complete).passed
    broken: tuple[dict[str, Any], ...] = (
        {"images_dropped": 1},
        {"images": {"cams/broadcast": 99}},
        {"real_frames": {"orbbec_8E": 79}},
        {"clip_frames": None},
        {"rate_3d_hz": 6.0},
    )
    for change in broken:
        assert not evaluate_telemetry(**(complete | change)).passed, change
