"""
The live ball and paddles: the lab's prop assets made into PhysX bodies, stepped by the calibrated ball physics.

:func:`attach_live_ball` loads the ball, the retired-ball pool and one paddle per player, authors their bodies and
returns :class:`lib.physics.engine.stepper.GamePhysics`, the per-step callback that integrates the ball (RK4 at the
game step, 3840 Hz) and resolves every contact with the reference impact law:

- the table, net, floor and room are **hybrid**: PhysX never meets the ball in play; the adapter sweeps each surface
  every game step and applies the calibrated contact law;
- each paddle is a dynamic, collision-free body the adapter drives along the player's hand, and its contact with the
  ball is imposed by the adapter (the rigid off-centre paddle impact);
- PhysX steps only the paddles and the retired balls, ten times per rendered frame (240 Hz).

When a new ball is placed, the old one carries on physically as a retired ball (the room's native colliders, no air
force) until it is collected out of every camera's view; it is never evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lib.him_lab import COLLIDER_ROOT, FLOOR_COLLIDER, NET_FIXTURE_COLLIDERS, ROOM_COLLIDERS, STAGE_ROOT, author_ball_colliders, load_lab_asset
from lib.physics.engine.settings import EngineSettings, canonical_settings
from lib.physics.params import BallTableParams

if TYPE_CHECKING:
    from pxr import Usd

    from lib.physics.engine.stepper import GamePhysics

GAME_PHYSICS_DT: float = 1.0 / 3840.0
"""The game step: the ball's RK4 and every hybrid sweep run at 3840 Hz, the lowest rate that passes every calibrated test."""
RENDER_DT: float = 1.0 / 24.0
ENGINE_SUBSTEPS: int = 10
"""PhysX steps per rendered frame (240 Hz): enough for the driven paddles, which land exactly on each frame's pose, and
for the retired balls' native bounces, which are visual only."""

GAME_ROOT: str = f"{STAGE_ROOT}/Game"
BALL_PATH: str = f"{GAME_ROOT}/ball"
PADDLE_ROOT: str = f"{GAME_ROOT}/Paddles"
PARK_POSITION_M: tuple[float, float, float] = (-4.6, 3.0, -0.7)
"""Where an out-of-play ball waits (a far corner of the room, on the floor, hidden)."""
SPARE_COUNT: int = 8
"""Retired balls: eight, so a ball a camera sees is never reused."""
SPARE_PATHS: tuple[str, ...] = tuple(f"{GAME_ROOT}/ball_retired_{index}" for index in range(SPARE_COUNT))
SPARE_PARK_M: tuple[tuple[float, float, float], ...] = tuple((-6.5, -3.5 + 1.0 * index, -0.7) for index in range(SPARE_COUNT))
"""Where a retired ball waits: behind the 8E end wall (the walls stand at |x| = 4.9 m), out of every camera."""


def engine_settings(params: BallTableParams) -> EngineSettings:
    """The adapter's settings: hybrid table, net and floor, the RK4 ball, PhysX at :data:`ENGINE_SUBSTEPS` per frame."""

    return canonical_settings(
        params,
        physics_dt_s=GAME_PHYSICS_DT,
        table_mode="hybrid",
        net_mode="hybrid",
        floor_mode="hybrid",
        ball_integrator="rk4",
        engine_dt_s=RENDER_DT / ENGINE_SUBSTEPS,
    )


def set_physics_dt(world: Any, physics_dt: float) -> None:
    """
    Set the Isaac Sim world's physics step, keeping its rendering step (an integer multiple of it).

    :raises RuntimeError: If the world does not take it.
    """

    if abs(float(world.get_physics_dt()) - physics_dt) <= 1e-12:
        return
    rendering_dt = float(world.get_rendering_dt())
    ratio = rendering_dt / physics_dt
    if abs(ratio - round(ratio)) > 1e-6:
        raise RuntimeError(f"rendering step {rendering_dt} s is not an integer multiple of the physics step {physics_dt} s")
    world.set_simulation_dt(physics_dt=physics_dt, rendering_dt=rendering_dt)
    if abs(float(world.get_physics_dt()) - physics_dt) > 1e-12:
        raise RuntimeError(f"the world did not take physics_dt {physics_dt} s (reads {world.get_physics_dt()} s)")


def _place(stage: Usd.Stage, path: str, position: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*position))
    xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0))


@dataclass(frozen=True)
class LiveBall:
    """
    The live props and their physics.

    :ivar physics: The per-step callback (register it, then step the world).
    :ivar paddles: Paddle prim per role.
    :ivar scene: The PhysX scene settings, read back.
    """

    physics: GamePhysics
    paddles: dict[str, str]
    scene: dict[str, object]


def attach_live_ball(world: Any, stage: Usd.Stage, params: BallTableParams, roles: tuple[str, ...]) -> LiveBall:
    """
    Load the ball, the retired balls and the paddles, and make them bodies the calibrated physics steps.

    Call on the composed lab, before ``world.reset()``; bind the physics after it (:func:`bind_live_ball`).

    :param world: The Isaac Sim ``World``.
    :param stage: The active stage, stopped.
    :param params: The ball physics the engine runs.
    :param roles: The players, one paddle each.
    :return: The live ball.
    """

    from pxr import UsdGeom

    from lib.physics.engine.physx import author_ball_body, author_paddle_body, configure_scene, make_passive_marker
    from lib.physics.engine.stepper import GamePhysics

    settings = engine_settings(params)
    set_physics_dt(world, settings.physx_dt_s)
    UsdGeom.Xform.Define(stage, GAME_ROOT)
    UsdGeom.Xform.Define(stage, PADDLE_ROOT)
    load_lab_asset("ball", BALL_PATH)
    _place(stage, BALL_PATH, PARK_POSITION_M)
    for path, park in zip(SPARE_PATHS, SPARE_PARK_M, strict=True):
        load_lab_asset("ball", path)
        _place(stage, path, park)
    author_ball_colliders(stage, params, (BALL_PATH, *SPARE_PATHS))
    paddles: dict[str, str] = {}
    for index, role in enumerate(roles):
        path = f"{PADDLE_ROOT}/{role}"
        load_lab_asset("paddle", path)
        _place(stage, path, (0.0, 2.0 + 0.3 * index, -0.6))
        paddles[role] = path
    scene = configure_scene(world, stage, settings.scene)
    # The ball in play meets the table, net (with its posts and clamps), floor and room only through the adapter's sweeps.
    hybrid = (f"{COLLIDER_ROOT}/table_top", f"{COLLIDER_ROOT}/net", *NET_FIXTURE_COLLIDERS, FLOOR_COLLIDER, *ROOM_COLLIDERS)
    author_ball_body(stage, BALL_PATH, f"{BALL_PATH}/collider", settings.ball, filtered=hybrid)
    # Under the RK4 integrator the adapter moves the ball in play; its body only shows where it is.
    make_passive_marker(stage, BALL_PATH, f"{BALL_PATH}/collider")
    for path in paddles.values():
        author_paddle_body(stage, path, settings.paddle)
    # Retired balls play out under PhysX alone (no filtered surfaces, no air force): they are no longer evidence.
    for path in SPARE_PATHS:
        author_ball_body(stage, path, f"{path}/collider", settings.ball)
    physics = GamePhysics(params, settings, BALL_PATH, paddles, spare_paths=SPARE_PATHS, spare_park=SPARE_PARK_M)
    physics.create_view()
    return LiveBall(physics=physics, paddles=paddles, scene=scene)


def bind_live_ball(world: Any, ball: LiveBall) -> None:
    """Bind the physics to its tensor view after ``world.reset()`` (the view can appear only after the first step)."""

    try:
        ball.physics.bind()
    except RuntimeError:
        world.step(render=False)
        ball.physics.bind()


def set_ball_scale(stage: Usd.Stage, path: str, scale: float) -> None:
    """
    Scale a ball's render geometry about its centre, its collider untouched: how a retired ball still on camera leaves
    the play over a few frames (a continuity device, logged as such).
    """

    from pxr import Gf, Usd, UsdGeom

    root = stage.GetPrimAtPath(path)
    for prim in Usd.PrimRange(root):
        if prim == root or "collider" in prim.GetName() or not prim.IsA(UsdGeom.Gprim):
            continue
        xformable = UsdGeom.Xformable(prim)
        ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpName() == "xformOp:scale:collect"]
        op = ops[0] if ops else xformable.AddScaleOp(opSuffix="collect")
        op.Set(Gf.Vec3f(scale, scale, scale))


def set_ball_visible(stage: Usd.Stage, visible: bool) -> None:
    """Show or hide the ball in play's render mesh (a parked ball is hidden)."""

    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(stage.GetPrimAtPath(BALL_PATH))
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


__all__ = [
    "BALL_PATH",
    "ENGINE_SUBSTEPS",
    "GAME_PHYSICS_DT",
    "PARK_POSITION_M",
    "RENDER_DT",
    "SPARE_PATHS",
    "LiveBall",
    "attach_live_ball",
    "bind_live_ball",
    "engine_settings",
    "set_ball_scale",
    "set_ball_visible",
    "set_physics_dt",
]
