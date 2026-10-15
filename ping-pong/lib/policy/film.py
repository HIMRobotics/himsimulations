"""
Env 0 of a :class:`lib.robots.table_scene.TableScene` filmed at 24 fps in the HIM lab's digital twin: :class:`TwinFilm`.

The twin is render only (:func:`lib.him_lab.author_him_twin` refuses to change the stage's physics): the HIM room
around env 0's table, its calibrated ceiling lighting, and two cameras, each logged as an H.264 video stream at the
simulated time it shows:

- ``camera``: the ``cine_wide`` broadcast view, 1920 x 1080;
- ``camera_8f``: HIM's own overhead Orbbec 8F, calibrated, 640 x 480.

The ball (and the opponent's paddle, a red disc posed where the coupling has it) are drawn only while in play, and
only while the film runs. The other envs are hidden from the render (their physics runs), and so are the grid floor's
mesh and the dome light. Nothing here changes a physics result.

``camera`` is the viewport itself, so a run streamed to Mission Control shows exactly what the film records. So the
film stays the same whoever watches, its camera is locked against the viewer's mouse (``ROBOT_SIMULATION_CONFIG``
turns arrow-key flying off) and its size is pinned against the viewer's window: a capture under way when either
changes waits for a frame that never comes, and the film loses it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FPS: float = 24.0
"""Camera frames per simulated second."""

BALL_PRIM: str = "/World/FilmBall"
OPPONENT_PRIM: str = "/World/FilmOpponentBlade"
PARKED_BELOW_M: float = -1.0
"""A blade below this height in the table frame is parked out of play, and not drawn."""
CAMERA_LOCK: str = "omni:kit:cameraLock"
"""Kit's camera lock: the viewport's mouse navigation leaves a camera carrying it ``True`` where it is."""


def hide_other_envs(stage: Any, num_envs: int, keep: int = 0) -> int:
    """Make every env prim but ``keep`` invisible (render only), in one change block; returns how many."""

    from pxr import Sdf

    layer = stage.GetRootLayer()
    with Sdf.ChangeBlock():
        for i in range(num_envs):
            if i == keep:
                continue
            spec = Sdf.CreatePrimInLayer(layer, f"/World/envs/env_{i}")
            attr: Any = spec.attributes.get("visibility") or Sdf.AttributeSpec(spec, "visibility", Sdf.ValueTypeNames.Token)
            attr.default = "invisible"
    return max(num_envs - 1, 0)


class TwinFilm:
    """
    Films env 0 in the HIM twin at 24 fps; call :meth:`frame` after every policy step and :meth:`close` at the end.

    :param logger: ``antioch.Logger`` (its prefix joins the streams' paths).
    :param scene: The table scene (built with ``visuals=True``).
    :param orbbec_8f: Also stream HIM's Orbbec 8F view.
    """

    def __init__(self, logger: Any, scene: Any, *, orbbec_8f: bool = True) -> None:
        import antioch
        from isaaclab.sim.utils.stage import get_current_stage
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, Sdf, UsdGeom

        from lib.him_lab import author_him_twin
        from lib.telemetry.cameras import CameraRig
        from lib.telemetry.video import CameraStreamSink

        self.scene = scene
        env = scene.env
        stage = get_current_stage()
        self.table_offset = env.table_offset[0].detach().cpu().numpy()
        self.twin = author_him_twin(stage, self.table_offset, orbbec_8f=orbbec_8f)
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/ground")).MakeInvisible()
        dome = stage.GetPrimAtPath("/World/dome")
        if dome.IsValid():
            dome.SetActive(False)
        hide_other_envs(stage, scene.num_envs)
        # env 0's ball and the opponent's blade: render-only prims, hidden until in play
        ball = stage.DefinePrim(Sdf.Path(BALL_PRIM), "Xform")
        if scene.files is not None:
            ball.GetReferences().AddReference(scene.files.ball)
        blade = UsdGeom.Xform.Define(stage, OPPONENT_PRIM)
        blade.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -10.0))
        blade.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        disc = UsdGeom.Cylinder.Define(stage, f"{OPPONENT_PRIM}/Disc")
        disc.CreateAxisAttr("Z")
        disc.CreateRadiusAttr(1.0)
        paddle = scene.robot.paddle
        disc.CreateHeightAttr(float(paddle.blade_thickness))
        disc.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.08, 0.08)])
        UsdGeom.XformCommonAPI(disc.GetPrim()).SetScale(Gf.Vec3f(float(paddle.blade_half_width), float(paddle.blade_half_length), 1.0))
        self.stage = stage
        self.ball_shown: bool | None = None
        self.blade_shown: bool | None = None
        self._show(BALL_PRIM, False)
        self._show(OPPONENT_PRIM, False)
        # The film is the viewport. Streamed, the viewport would follow the viewer's camera and window: pin both, so the
        # film is the same streamed or not, and a viewer can't cost it a frame.
        stage.GetPrimAtPath(self.twin.cine_wide.prim_path).CreateAttribute(CAMERA_LOCK, Sdf.ValueTypeNames.Bool).Set(True)
        viewport = get_active_viewport()
        viewport.camera_path = self.twin.cine_wide.prim_path
        viewport.fill_frame = False  # a stream fits the viewport to its window by resizing it, which would resize the film
        viewport.resolution = self.twin.cine_wide.resolution
        antioch.capture_viewport()  # the first capture after a camera move is stale
        self.sink = CameraStreamSink(logger, "camera", FPS, max_width=1920)
        # On the GPU: streamed, a CPU annotator's copy hangs the Kit update Isaac Lab runs every step (5 of 5 runs).
        self.rig = CameraRig({"orbbec_8F": self.twin.orbbec_8f}, device="cuda") if self.twin.orbbec_8f is not None else None
        self.sink_8f = CameraStreamSink(logger, "camera_8f", FPS, max_width=640) if self.rig is not None else None
        self.next_frame: float | None = None
        self.frames = 0
        self.frames_8f = 0
        self.missing_8f = 0
        self.failures: list[str] = []
        self.span = [float("inf"), -float("inf")]

    def _show(self, path: str, visible: bool) -> None:
        from pxr import UsdGeom

        imageable = UsdGeom.Imageable(self.stage.GetPrimAtPath(path))
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def frame(self, sim_time: float) -> None:
        """After a policy step: pose the ball and the opponent's blade, and capture a frame when the 24 fps clock is due."""

        import antioch
        from pxr import Gf, UsdGeom

        from lib.telemetry.video import run_clocks

        env = self.scene.env
        coupling = env.coupling
        live = bool(coupling.active[0])
        if live:
            position = coupling.ball[0, :3].detach().cpu().numpy() + self.table_offset
            UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(BALL_PRIM)).SetTranslate(Gf.Vec3d(*(float(v) for v in position)))
        if live != self.ball_shown:
            self._show(BALL_PRIM, live)
            self.ball_shown = live
        pose = getattr(env, "opponent_pose", None)
        drawn = False
        if pose is not None:
            p, q = pose[0][0].detach().cpu().numpy(), pose[1][0].detach().cpu().numpy()
            drawn = float(p[2]) > PARKED_BELOW_M
            if drawn:
                ops = {op.GetOpName(): op for op in UsdGeom.Xformable(self.stage.GetPrimAtPath(OPPONENT_PRIM)).GetOrderedXformOps()}
                ops["xformOp:translate"].Set(Gf.Vec3d(*(float(v) for v in p + self.table_offset)))
                ops["xformOp:orient"].Set(Gf.Quatd(*(float(v) for v in q)))
        if drawn != self.blade_shown:
            self._show(OPPONENT_PRIM, drawn)
            self.blade_shown = drawn
        run_sim, _wall = run_clocks()
        clock = run_sim if run_sim is not None else sim_time
        if self.next_frame is not None and clock + 1e-9 < self.next_frame:
            return
        self.next_frame = max((self.next_frame if self.next_frame is not None else clock) + 1.0 / FPS, clock)
        stamp = None if run_sim is not None else sim_time
        try:
            image = antioch.capture_viewport()
            if image is not None and self.sink.push(np.asarray(image)[:, :, :3].astype(np.uint8), stamp):
                self.frames += 1
                self.span = [min(self.span[0], clock), max(self.span[1], clock)]
        except Exception as error:  # a capture failure is counted and reported, never a silent pass
            self.failures.append(repr(error))
        if self.rig is not None and self.sink_8f is not None:
            view = self.rig.read().get("orbbec_8F")
            if view is not None and self.sink_8f.push(view, stamp):
                self.frames_8f += 1
            else:
                self.missing_8f += 1

    def close(self) -> dict[str, Any]:
        """Flush the streams; the film's account: frames, failures, the span it covers, each stream's statistics."""

        # No longer posed after this (the policy's blind pass follows), so not drawn either.
        self._show(BALL_PRIM, False)
        self._show(OPPONENT_PRIM, False)
        video = self.sink.close().as_dict()
        out: dict[str, Any] = {
            "frames": self.frames,
            "span_s": [round(v, 3) for v in self.span] if self.frames else None,
            "capture_failures": len(self.failures),
            "first_failure": self.failures[0] if self.failures else None,
            "video": video,
            "twin": {"room": self.twin.room_prim, "camera": self.twin.cine_wide.prim_path, "resolution": list(self.twin.cine_wide.resolution)},
        }
        if self.sink_8f is not None:
            out["camera_8f"] = {"frames": self.frames_8f, "missing": self.missing_8f, "video": self.sink_8f.close().as_dict()}
        if self.rig is not None:
            self.rig.close()
        return out

    def at_24_fps(self, account: dict[str, Any], seconds: float) -> tuple[bool, str]:
        """Whether the film holds a frame for every 1/24 s of ``seconds`` filmed, none dropped or failed."""

        expected = int(seconds * FPS)
        dropped = int(account["video"].get("dropped", 0) or 0)
        errors = account["video"].get("errors") or []
        ok = account["frames"] >= expected - 1 and account["capture_failures"] == 0 and dropped == 0 and not errors
        return ok, f"{account['frames']} frames for {seconds:.1f} s at {FPS:g} fps (expected {expected}); {dropped} dropped; {len(errors)} encoder errors"


__all__ = ["FPS", "TwinFilm", "hide_other_envs"]
