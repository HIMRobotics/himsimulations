"""
Retarget HIM's mocap strokes onto a humanoid, check every stroke, and publish the kept ones (CPU only).

*Retargeting* moves motion recorded on one body (HIM's player, in a motion-capture suit) onto another (the robot),
whose limbs, joints and motors differ. This module finds robot motions that put the paddle where the human's was at
each contact, solving inverse kinematics over a whole rally, then checks every stroke for reach, speed, torque,
balance and feet. A stroke the robot can't make is rejected with its reason, never faked. In four steps:

    human   = load_mocap_strokes(asset)            # HIM's strokes on the table frame, the paddle state at each contact
    robot   = robot_model(ROBOTS[name])            # URDF kinematics, joint limits, motors, the paddle on the wrist
    rallies = retarget(human, robot)               # reach -> targets -> whole-rally IK -> quality checks, per rally
    package = write_package(kept_clips(rallies))   # the kept motion and every stroke's check values

:func:`retarget` is the algorithm; :func:`retarget_him_strokes` is the scenario around it (fetch, publish, telemetry,
the run's checks). ``docs/retargeting.md`` describes both and what a new robot must provide. Which sessions and
rallies to solve is a :class:`~lib.config.RetargetParams` preset in ``lib/config.py``.
"""

from __future__ import annotations

import hashlib
import itertools
import multiprocessing
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from lib.config import RetargetParams
from lib.retarget.model import RobotModel, body_poses, robot_model
from lib.retarget.optimize import Problem, Solution, solve_reach
from lib.retarget.package import Clip, kept_pieces, write_package
from lib.retarget.source import FLOOR_Z_M, HumanStrokes, load_mocap_strokes, quat_to_matrix
from lib.retarget.targets import Segment, Targets, build_targets, reach_problem, scales, segments
from lib.retarget.validate import GATES, grade_segment

if TYPE_CHECKING:
    import antioch

PLAYBACK_RALLIES = 12  # the telemetry plays back at most this many solved rallies (evenly spaced), every frame at 50 Hz
_SHARED: dict = {}  # the inputs the forked workers inherit


@dataclass
class RallyResult:
    """One solved rally: its targets, the robot's motion, every stroke's checks, and the frames failing any check."""

    session: int
    robot_end: int
    segment: Segment
    targets: Targets
    solution: Solution
    strokes: list[dict]
    bad_frames: np.ndarray


def retarget(
    human: HumanStrokes,
    robot: RobotModel,
    rallies: tuple[str, ...] = (),
    workers: int = 4,
    adam_iters: int = 200,
    lbfgs_iters: int = 6000,
    on_rally: Callable[[RallyResult, int, int], None] | None = None,
) -> list[RallyResult]:
    """
    Retarget every rally of ``human`` onto ``robot`` and check each stroke.

    1. **Reach**, per session: where must the robot stand to hold each contact's paddle path? (static IK over every
       stroke of the session, so a rally's result never depends on which rallies are selected)
    2. Per rally, in its own process (one CPU thread, longest first):

       * **targets**: the human's motion at the robot's proportions, the stance moved where the reach needs it, and the
         paddle path through the contacts;
       * **solve**: per-frame LM IK, then Adam and L-BFGS over the whole rally (smooth, balanced, feet planted);
       * **check**: reach (paddle path and joint limits), speed, torque, balance, feet, per stroke; failing frames.

    :param rallies: Stroke-id prefixes of the rallies to solve (empty: all), e.g. ``("s1_r001",)``.
    :param on_rally: Called with each result as it arrives, and the count done and total.
    """
    sc = scales(robot, human)
    picked = [
        (k, i)
        for k, src in human.sessions.items()
        for i, seg in enumerate(segments(src))
        if not rallies or any(s.id.startswith(r) for s in seg.strokes for r in rallies)
    ]
    sessions = sorted({k for k, _ in picked})
    _SHARED.update(human=human, robot=robot, scales=sc, settings=(adam_iters, lbfgs_iters))
    ctx = multiprocessing.get_context("fork")  # the workers inherit the inputs; autograd never runs in this process
    with ctx.Pool(max(1, min(workers, len(sessions)))) as pool:
        _SHARED["reach"] = dict(zip(sessions, pool.map(_reach_job, sessions), strict=True))
    order = sorted(picked, key=lambda ki: -len(segments(human.sessions[ki[0]])[ki[1]].t))
    out = []
    with ctx.Pool(max(1, min(workers, len(order)))) as pool:
        for result in pool.imap_unordered(_rally_job, order):
            out.append(result)
            if on_rally:
                on_rally(result, len(out), len(order))
    return sorted(out, key=lambda r: (r.session, r.segment.t[0]))


def _reach_job(session: int) -> dict:
    """The static reach of every stroke of one session (two threads)."""
    import torch

    torch.set_num_threads(2)
    human, robot = _SHARED["human"], _SHARED["robot"]
    src = human.sessions[session]
    r = solve_reach(robot, reach_problem(src, human, _SHARED["scales"]))
    return {s.id: r["xy"][j] for j, s in enumerate(src.strokes)}


def _rally_job(key: tuple[int, int]) -> RallyResult:
    """One rally: targets, solve and checks (one thread)."""
    import torch

    torch.set_num_threads(1)
    human, robot, sc = _SHARED["human"], _SHARED["robot"], _SHARED["scales"]
    src, reach = human.sessions[key[0]], _SHARED["reach"][key[0]]
    seg = segments(src)[key[1]]
    tg = build_targets(src, seg, human, sc, reach)
    sol = Problem(tg, robot).solve(*_SHARED["settings"])
    rows, bad = grade_segment(robot, src, seg, tg, sol, reach)
    return RallyResult(session=key[0], robot_end=src.robot_end, segment=seg, targets=tg, solution=sol, strokes=rows, bad_frames=bad)


def kept_clips(results: list[RallyResult]) -> list[Clip]:
    """The motion that survives the checks: each rally cut around its rejected strokes and failing frames."""
    clips = []
    for r in results:
        s, t = r.solution, r.solution.t
        for a, b in kept_pieces(t, r.bad_frames, [row["t_contact_s"] for row in r.strokes if not row["kept"]]):
            inside = [(row["stroke_id"], row["t_contact_s"]) for row in r.strokes if t[a] <= row["t_contact_s"] <= t[b - 1]]
            clips.append(Clip(r.session, r.robot_end, t[a:b], s.root_p[a:b], s.root_r[a:b], s.q[a:b], r.targets.planted[a:b], inside))
    return clips


def summarise(rows: list[dict]) -> dict:
    """Counts and blade errors, by kind and by first failing check."""
    solved = [r for r in rows if r["source"] == "paddle_inverse"]
    p50 = lambda rs: round(float(np.median([r["blade_pos_max_m"] for r in rs])) * 100, 3) if rs else None  # noqa: E731
    return {
        "strokes": len(rows),
        "kept": sum(r["kept"] for r in rows),
        "solved_strokes": len(solved),
        "solved_kept": sum(r["kept"] for r in solved),
        "in_motion": sum(bool(r.get("in_motion")) for r in rows),
        "causes": dict(Counter(r["cause"] for r in rows if not r["kept"])),
        "by_kind": {
            k: {"strokes": sum(r["kind"] == k for r in rows), "kept": sum(r["kept"] and r["kind"] == k for r in rows)}
            for k in sorted({r["kind"] for r in rows})
        },
        "blade_p50_cm_all": p50(rows),
        "blade_p50_cm_kept": p50([r for r in rows if r["kept"]]),
    }


def recheck_package(path: Path, robot: RobotModel) -> dict:
    """Re-check the written motion itself, independent of the solver's bookkeeping: every frame inside 0.95 of the joint
    ranges and 0.8 of the speed limits (50 Hz central differences), no planted sole sliding more than 1 cm or sinking
    more than 5 mm, and every packaged stroke within the paddle-path tolerances."""
    import io
    import zipfile

    with zipfile.ZipFile(path) as z:
        d = np.load(io.BytesIO(z.read("motion.npz")))
        q, qd, names = d["joint_pos"].astype(np.float64), d["joint_vel"].astype(np.float64), [str(n) for n in d["joint_names"]]
        cols = [names.index(n) for n in robot.joint_names]
        mid, half = (robot.lower + robot.upper) / 2, (robot.upper - robot.lower) / 2
        limit = float((np.abs(q[:, cols] - mid) / half).max()) if len(q) else 0.0
        speed = float((np.abs(qd[:, cols]) / robot.speed_limit).max()) if len(q) else 0.0
        slide = sink = 0.0
        starts, bodies = d["clip_start"], [str(n) for n in d["body_names"]]
        for a, b in itertools.pairwise(starts):
            for f, foot in enumerate(robot.feet):
                pos, quat = d["body_pos_w"][a:b, bodies.index(robot.links[foot])], d["body_quat_w"][a:b, bodies.index(robot.links[foot])]
                corners = np.einsum("nij,kj->nki", quat_to_matrix(quat), robot.sole) + pos[:, None].astype(np.float64)
                sink = max(sink, float(np.maximum(FLOOR_Z_M - corners[..., 2], 0.0).max()))
                planted = d["foot_contact"][a:b, f]
                edges = np.diff(np.r_[0, planted.astype(np.int8), 0])
                for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True):
                    sole = corners[s:e].mean(1)[:, :2]
                    slide = max(slide, float(np.linalg.norm(sole - sole[0], axis=1).max()))
        blade = float(d["stroke_blade_pos_max_m"].max()) if len(d["stroke_id"]) else 0.0
        face = float(d["stroke_blade_face_max_deg"].max()) if len(d["stroke_id"]) else 0.0
    g = GATES
    ok = limit <= g.limit_fraction + 1e-6 and speed <= g.speed_fraction + 1e-3 and slide <= g.foot_slide_m and sink <= g.foot_penetration_m + 1e-6
    ok = ok and blade <= g.blade_pos_m and face <= g.blade_face_deg
    return {
        "ok": ok,
        "limit_use_max": limit,
        "speed_ratio_max": speed,
        "foot_slide_max_m": slide,
        "sole_sink_max_m": sink,
        "blade_pos_max_m": blade,
        "blade_face_max_deg": face,
    }


# ---------------------------------------------------------------------------------------------------- the scenario


class RetargetReport(BaseModel):
    """
    What one run measured; ``passed`` is its verdict.

    :ivar strokes: Strokes solved and checked; ``kept`` of them pass every check.
    :ivar solved_strokes: Of those, the strokes whose paddle state was solved from the ball; ``solved_kept`` kept.
    :ivar in_motion: Kept strokes inside the packaged motion.
    :ivar causes: Rejected strokes by their first failing check.
    :ivar by_kind: Strokes and kept strokes per kind (forehand/backhand, stance/stepping).
    :ivar blade_p50_cm_all: Median over every stroke of the blade's largest distance from the paddle path, cm.
    :ivar blade_p50_cm_kept: The same over the kept strokes.
    :ivar raw_control_fail: Solved strokes failing the paddle-path check against the capture's raw paddle.
    :ivar recheck: The packaged motion's maxima, re-read from the file.
    :ivar asset: The published package.
    :ivar seconds: Wall time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    strokes: int
    kept: int
    solved_strokes: int
    solved_kept: int
    in_motion: int
    causes: dict[str, int]
    by_kind: dict[str, dict[str, int]]
    blade_p50_cm_all: float | None
    blade_p50_cm_kept: float | None
    raw_control_fail: int
    recheck: dict[str, float | bool]
    asset: dict[str, str | bool]
    seconds: float


def retarget_him_strokes(run: antioch.ScenarioRun, params: RetargetParams) -> RetargetReport:
    """
    Retarget HIM's mocap strokes onto a humanoid, check them, and publish the kept strokes (see the module docstring).

    The kept strokes are published to the asset store as ``ping-pong/<robot>-retargeted-strokes``, versioned by the
    first 8 characters of the run ID, and attached to the run as a zip of ``motion.npz``, ``strokes.csv`` and
    ``manifest.json``.

    :param run: The active scenario run; the checks, results, package and telemetry are recorded on it.
    :param params: The robot, the human strokes' asset, the sessions and rallies to solve, and the solver settings.
    :return: The report; ``passed`` is the verdict.
    """
    import antioch
    import rerun as rr

    from lib.assets import ASSETS
    from lib.robots import ROBOTS

    t0, logger = time.time(), antioch.Logger("retarget")
    run.set_blueprint(_blueprint())
    path = Path(antioch.fetch_asset(params.source, version=params.source_version))
    pinned = ASSETS[params.source].sha256 if params.source in ASSETS and ASSETS[params.source].version == params.source_version else None
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    human = load_mocap_strokes(path, params.sessions)
    robot = robot_model(ROBOTS[params.robot])
    logger.value(
        "events",
        rr.TextLog(f"{params.robot}: {robot.nq} joints, {robot.total_mass:.2f} kg; sessions {list(params.sessions)}, rallies {list(params.rallies) or 'all'}"),
    )

    def progress(r: RallyResult, done: int, total: int) -> None:
        kept = sum(row["kept"] for row in r.strokes)
        logger.value(
            "events",
            rr.TextLog(
                f"rally {done}/{total} (session {r.session}, {r.segment.strokes[0].id}): {len(r.strokes)} strokes, {kept} kept; {r.solution.seconds:.0f} s"
            ),
        )
        logger.scalar("rallies_done", done)

    results = retarget(human, robot, params.rallies, params.workers, params.adam_iters, params.lbfgs_iters, on_rally=progress)
    rows = [row for r in results for row in r.strokes]
    run_id = run.scenario_run_id or "local"
    out = Path(params.artifacts_dir) / run_id / f"{params.robot}-retargeted-strokes"
    manifest = {
        "robot": {
            "name": robot.name,
            "urdf_sha256": robot.urdf_sha256,
            "joints": robot.nq,
            "mass_kg": round(robot.total_mass, 4),
            "paddle_link": robot.links[robot.paddle_link],
        },
        "source": {
            "asset": f"{params.source}@{params.source_version}",
            "sha256": source_sha,
            "sessions": list(params.sessions),
            "rallies": list(params.rallies),
        },
        "solver": {
            "adam_iters": params.adam_iters,
            "lbfgs_iters": params.lbfgs_iters,
            "per_rally": "LM IK init, Adam, L-BFGS (strong Wolfe) over four balance rounds; float64, one CPU thread",
        },
        "gates": GATES.__dict__,
        "frame": "table frame: origin at the table-top centre, z up, the robot at -X, floor at z = -0.76 m; metres, radians, quaternions w, x, y, z; time on the camera clock",
        "paddle_frame": "body 'paddle': origin at the blade centre, x towards the tip, y the forehand rubber's normal (face 0 = +y, 1 = -y), z across",
        "run": run_id,
    }
    package = write_package(out, robot, kept_clips(results), rows, manifest)
    summary = summarise(rows)
    recheck = recheck_package(package, robot)
    run.add_artifact(package, name=package.name, description=f"Retargeted strokes for {params.robot}: motion.npz, strokes.csv, manifest.json")
    asset_name, asset_version = f"ping-pong/{params.robot}-retargeted-strokes", run_id[:8]
    asset: dict[str, str | bool] = {"name": asset_name, "version": asset_version}
    if summary["kept"]:
        try:
            antioch.save_asset(asset_name, version=asset_version, source=str(package), description=f"HIM strokes retargeted onto {params.robot}, run {run_id}")
            asset["ok"] = True
        except Exception as error:  # recorded, never hidden
            asset |= {"ok": False, "error": repr(error)}
    solved = [r for r in rows if r["source"] == "paddle_inverse"]
    raw_fail = [r["raw_blade_pos_max_m"] > GATES.blade_pos_m or r["raw_blade_face_max_deg"] > GATES.blade_face_deg for r in solved]
    playback(results, robot)

    expected = sum(
        1
        for k, src in human.sessions.items()
        for seg in segments(src)
        if not params.rallies or any(s.id.startswith(p) for s in seg.strokes for p in params.rallies)
    )
    checks = [
        run.check(
            "every selected rally solved and checked",
            len(results) == expected and expected > 0 and all(np.isfinite(r.solution.cost) for r in results),
            detail=f"{len(results)} of {expected} rallies, {len(rows)} strokes",
        ),
        run.check(
            "negative control: the capture's raw paddle fails the paddle-path check",
            bool(solved) and bool(np.mean(raw_fail) >= 0.5),
            detail=f"{sum(raw_fail)} of {len(solved)} solved strokes fail against the raw paddle (must be at least half)",
        ),
        run.check("at least one stroke kept", summary["kept"] > 0, detail=f"{summary['kept']} of {len(rows)} kept; causes {summary['causes']}"),
        run.check(
            "the packaged motion passes the checks when re-read", recheck["ok"], detail=", ".join(f"{k} {v:.4g}" for k, v in recheck.items() if k != "ok")
        ),
        run.check(
            "the kept strokes are published",
            bool(asset.get("ok")),
            detail=f"{asset_name}@{asset_version}" + ("" if asset.get("ok") else f": {asset.get('error', 'no stroke kept, nothing published')}"),
        ),
    ]
    if pinned is not None:
        checks.append(run.check("the human strokes are the pinned bundle", source_sha == pinned, detail=source_sha))
    report = RetargetReport(passed=all(checks), **summary, recheck=recheck, raw_control_fail=sum(raw_fail), asset=asset, seconds=round(time.time() - t0, 1))
    run.add_results(report.model_dump(exclude={"passed"}))
    return report


# ---------------------------------------------------------------------------------------------------- telemetry


def _blueprint():
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/retarget/world", name="Retargeted rallies (50 Hz): robot, blade, paddle path, contacts kept (green) / rejected (red)"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/retarget/blade_error_cm", name="Blade from the paddle path (cm)"),
                rrb.TextLogView(origin="/retarget/events", name="Events"),
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="sim_time"),
        collapse_panels=True,
    )


def playback(results: list[RallyResult], robot: RobotModel) -> float:
    """3D telemetry: up to :data:`PLAYBACK_RALLIES` solved rallies played back one after another on ``sim_time``, every
    frame: the robot's links, the blade, the paddle path and the contacts coloured by verdict. Returns the end time."""
    import rerun as rr

    from lib.telemetry.video import log_at

    pick = sorted(set(np.linspace(0, len(results) - 1, min(len(results), PLAYBACK_RALLIES)).round().astype(int).tolist())) if results else []
    log_at("retarget/world/table", rr.Boxes3D(centers=[[0, 0, -0.015]], half_sizes=[[1.37, 0.7625, 0.015]], colors=[(40, 80, 160)], fill_mode="solid"), 0.0)
    log_at("retarget/world/net", rr.Boxes3D(centers=[[0, 0, 0.076]], half_sizes=[[0.002, 0.915, 0.076]], colors=[(220, 220, 220)]), 0.0)
    bones = [(int(robot.parents[i]), i) for i in range(1, len(robot.links))] + [(robot.paddle_link, len(robot.links))]
    t_play = 0.0
    for r in (results[i] for i in pick):
        sol, tg = r.solution, r.targets
        _rot, pos = body_poses(robot, sol.root_p, sol.root_r, sol.q)
        error_cm = np.linalg.norm(pos[:, -1] - tg.paddle_p, axis=1) * 100
        contacts = np.array([row["blade_p"] for row in r.strokes])
        colours = [(40, 200, 60) if row["kept"] else (220, 40, 40) for row in r.strokes]
        labels = [f"{row['stroke_id']} {'kept' if row['kept'] else row['cause']}" for row in r.strokes]
        kept = sum(row["kept"] for row in r.strokes)
        log_at("retarget/events", rr.TextLog(f"playing session {r.session} rally {r.segment.strokes[0].id}: {kept} of {len(r.strokes)} strokes kept"), t_play)
        log_at("retarget/world/paddle_path", rr.LineStrips3D([tg.paddle_p], radii=0.003, colors=[(255, 170, 40)]), t_play)
        log_at("retarget/world/contacts", rr.Points3D(contacts, radii=0.03, colors=colours, labels=labels), t_play)
        for j in range(len(sol.t)):
            ts = t_play + sol.t[j] - sol.t[0]
            log_at("retarget/world/robot", rr.LineStrips3D([[pos[j, a], pos[j, b]] for a, b in bones], radii=0.012, colors=[(170, 170, 185)]), ts)
            log_at("retarget/world/blade", rr.Points3D([pos[j, -1]], radii=0.076, colors=[(200, 30, 30)]), ts)
            log_at("retarget/blade_error_cm", rr.Scalars(float(error_cm[j])), ts)
        t_play += sol.t[-1] - sol.t[0] + 0.5
    return t_play


__all__ = ["RallyResult", "RetargetReport", "kept_clips", "playback", "recheck_package", "retarget", "retarget_him_strokes", "summarise"]
