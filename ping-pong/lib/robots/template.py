"""
The starting point for your own controller: a humanoid with its paddle at the HIM table, the calibrated ball, and
nothing else. :func:`load_humanoid_at_table` runs it as a scenario and checks the scene is sound.

What it does, in every env at once:

1. builds the scene (:func:`lib.robots.table_scene.build_table_scene`): the robot from its spec, the table and net,
   the calibrated ball, and env 0 filmed in the HIM lab twin at 24 fps;
2. drops a ball 30 cm onto the table (ITTF's table test) and measures its rebound;
3. launches HIM's recorded feeds at the robot, one after another;
4. while each ball is in play, steps the robot with :meth:`~lib.robots.table_scene.TableScene.hold_default_pose`.
   **Replace that line with your controller.**

**Writing a controller.** An *env* is one copy of the scene; they all step together. An action is one number per
joint per env, a ``(num_envs, num_joints)`` tensor in the order of ``scene.joint_names``. Each joint's target is
``default_pose + action_scale * action``, which its PD gains (how stiffly and with how much damping the motor holds a
target) drive it toward; zeros hold the ready stance. ``scene.step(actions)`` returns the next observation, one row per
env, whose columns :func:`lib.robots.table_scene.observation_terms` lists in order (116 for the G1; columns 109-111 are
the ball's position in the pelvis frame). The README's "Your first experiments" walks through a first one.

The preset pins the pelvis (``fixed_base``): with zero actions the robot only holds its pose through its PD gains,
and a G1 with no balance controller falls. Unpin it once your controller balances, with a preset in
``lib/config.py`` such as ``"unitree-g1-unpinned": TemplateParams(fixed_base=False)``, or in code::

    params = TEMPLATE_PRESETS["unitree-g1"].model_copy(update={"fixed_base": False})
    load_humanoid_at_table(run, params)

or ``build_table_scene(robot, num_envs, fixed_base=False)`` directly. Unpinned, a fallen robot is reset to its stance.

**Checks** (each recorded on the run):

- the scene's meshes are the pinned versions;
- the drop rebounds into ITTF's band, 23-26 cm, in every env;
- nothing went non-finite;
- PhysX's GPU buffers kept headroom;
- the film holds every 1/24 s;
- every ball ended, at rest or off the table, before the time cap.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from lib.config import TemplateParams

if TYPE_CHECKING:
    import antioch

ITTF_DROP_M: float = 0.30
"""The ball's bottom is released this far above the table."""

ITTF_REBOUND_M: tuple[float, float] = (0.23, 0.26)
"""ITTF's table test: the rebound of the ball's bottom, m."""

DROP_XY_M: tuple[float, float] = (0.685, 0.0)
"""Where the drop lands, table frame: the middle of the far half, clear of the robot."""

RECORDED_FEEDS: tuple[tuple[float, ...], ...] = (
    (0.6308, 0.3601, 0.2228, -4.269, -0.0503, 1.3366, 0.0, 0.0, 0.0),
    (0.6233, 0.1676, 0.1684, -4.4368, -1.5753, 1.525, 0.0, 0.0, 0.0),
    (0.8011, 0.0678, 0.178, -5.3711, -0.4361, 1.0831, 0.0, 0.0, 0.0),
)
"""Three of HIM's opponent's recorded opening shots toward the robot (table frame: position m, velocity m/s, spin
rad/s), 4.5-5.5 m/s, each bouncing on the robot's half.

The table frame's origin is the centre of the table top, z up, and the robot stands at the -x end. These feeds travel
toward -x, so a negative y spin (the eighth number) is topspin and a positive one backspin; z spin is sidespin. To try
spin, append rows here and launch more balls with a preset's ``balls`` (balls after the drop cycle through these)."""

BALL_CAP_S: float = 4.0
"""The longest any one ball is followed, s."""


class TemplateReport(BaseModel):
    """What one run measured; ``passed`` is its verdict."""

    model_config = ConfigDict(frozen=True, extra="allow")

    passed: bool


class BallWatch:
    """
    The template's hooks (:class:`lib.robots.table_scene.TableHooks`): after every physics step, each env's drop
    rebound, read as the highest the ball's bottom reaches after its first table bounce, sampled every 5 ms (within
    0.03 mm of the true apex).
    """

    def __init__(self) -> None:
        self.watching = False
        self.bounced: Any = None
        self.apex: Any = None
        self.done: Any = None

    def watch_drop(self, num_envs: int, device: Any) -> None:
        import torch

        self.watching = True
        self.bounced = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.done = torch.zeros_like(self.bounced)
        self.apex = torch.full((num_envs,), -float("inf"), dtype=torch.float64, device=device)

    def physics_step(self, env: Any, contacts: Any) -> None:
        import torch

        from lib.physics.vectorized.geometry import TABLE_TOP

        if not self.watching or contacts is None:
            return
        flat = contacts.flat()
        self.bounced[flat["env"][flat["surface"] == TABLE_TOP]] = True
        ball = env.coupling.ball.to(torch.float64)
        bottom = ball[:, 2] - env.coupling.physics.params.radius.to(torch.float64)
        tracking = self.bounced & ~self.done
        self.apex = torch.where(tracking, torch.maximum(self.apex, bottom), self.apex)
        self.done |= tracking & (ball[:, 5] <= 0.0)


class LaunchedBall:
    """One ball in every env: :meth:`in_play` until every env's ball is at rest or off the table (the drop: until read)."""

    def __init__(self, scene: Any, watch: BallWatch, name: str, started_s: float) -> None:
        self.scene, self.watch, self.name, self.started_s = scene, watch, name, started_s
        self.ended = False

    def in_play(self) -> bool:
        import torch

        coupling = self.scene.coupling
        if self.name == "drop" and bool(self.watch.done.all()):
            coupling.park(torch.arange(self.scene.num_envs, device=coupling.device))
        self.ended = not bool(coupling.active.any())
        return not self.ended and float(coupling.t) - self.started_s < BALL_CAP_S


def launch_balls(scene: Any, watch: BallWatch, count: int) -> Any:
    """Yield ``count`` balls, one at a time, each launched in every env: the ITTF drop, then HIM's recorded feeds."""

    import torch

    coupling = scene.coupling
    rows = torch.arange(scene.num_envs, device=coupling.device)
    radius = float(coupling.physics.params.radius[0])
    for k in range(count):
        if k == 0:
            state = (DROP_XY_M[0], DROP_XY_M[1], radius + ITTF_DROP_M, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            watch.watch_drop(scene.num_envs, coupling.device)
        else:
            state = RECORDED_FEEDS[(k - 1) % len(RECORDED_FEEDS)]
            watch.watching = False
        coupling.launch(rows, torch.tensor(state, dtype=coupling.dtype, device=coupling.device).expand(scene.num_envs, 9))
        yield LaunchedBall(scene, watch, "drop" if k == 0 else f"feed-{k}", float(coupling.t))


def load_humanoid_at_table(run: antioch.ScenarioRun, params: TemplateParams) -> TemplateReport:
    """
    A humanoid with its paddle at the HIM table, balls launched at it, holding its default pose.

    One loop per ball: the drop test, then each recorded feed. The line marked ``<- replace with your controller``
    is where your actions go (see the module notes for what an action and an observation are).

    :param run: The active scenario run; the checks are recorded on it.
    :param params: The robot, the number of envs, whether the pelvis is pinned, how many balls, the time cap, the film.
    :return: The report; ``passed`` is the verdict.
    """

    import antioch

    from lib.policy.film import TwinFilm
    from lib.robots import ROBOTS
    from lib.robots.table_scene import build_table_scene

    watch = BallWatch()
    scene = build_table_scene(
        ROBOTS[params.robot], params.num_envs, hooks=watch, fixed_base=params.fixed_base, reset_spread=False, episode_s=params.max_sim_s + 5.0
    )
    film = TwinFilm(antioch.Logger("template"), scene) if params.film else None
    finite, steps, endings, rebound = True, 0, {}, None
    for ball in launch_balls(scene, watch, params.balls):  # terminal: every ball at rest or off the table
        while ball.in_play() and steps * scene.step_dt < params.max_sim_s:
            # scene.step returns the next observation, (num_envs, 116) for the G1: see observation_terms.
            scene.step(actions=scene.hold_default_pose())  # <- replace with your controller
            steps += 1
            finite &= state_is_finite(scene)
            if film is not None:
                film.frame(steps * scene.step_dt)
        endings[ball.name] = ball.ended
        if ball.name == "drop":
            rebound = watch.apex.clone()
        run.raise_if_cancelled()
    seconds = steps * scene.step_dt
    account = film.close() if film is not None else None
    return check_scene(run, scene, rebound_m=rebound, finite=finite, endings=endings, film=film, film_account=account, seconds=seconds, cap_s=params.max_sim_s)


def state_is_finite(scene: Any) -> bool:
    """Whether the robot's root and joints and every live ball are finite."""

    import torch

    from lib.robots.table_scene import data_tensor

    data = scene.env.robot.data
    ball = scene.coupling.ball[scene.coupling.active]
    return bool(torch.isfinite(data_tensor(data.root_link_pose_w)).all() and torch.isfinite(data_tensor(data.joint_pos)).all() and torch.isfinite(ball).all())


def check_scene(
    run: antioch.ScenarioRun,
    scene: Any,
    *,
    rebound_m: Any,
    finite: bool,
    endings: dict[str, bool],
    film: Any,
    film_account: dict[str, Any] | None,
    seconds: float,
    cap_s: float,
) -> TemplateReport:
    """The template's checks (see the module notes), each recorded with :meth:`antioch.ScenarioRun.check`."""

    import numpy as np

    from lib.robots.table_scene import physx_ok

    pinned = assets_pinned(scene, room=film is not None)
    low, high = ITTF_REBOUND_M
    rebound = rebound_m.detach().cpu().numpy() if rebound_m is not None else np.full(scene.num_envs, np.nan)
    physx = scene.physx_buffers()
    results = [
        run.check(
            "assets pinned", bool(pinned) and all(pinned.values()), detail=", ".join(f"{name} {'ok' if ok else 'MISMATCH'}" for name, ok in pinned.items())
        ),
        run.check(
            "drop rebound in the ITTF band",
            bool(((rebound >= low) & (rebound <= high)).all()),
            detail=f"{1e3 * np.nanmin(rebound):.1f}-{1e3 * np.nanmax(rebound):.1f} mm over {rebound.size} envs, band [{1e3 * low:.0f}, {1e3 * high:.0f}] mm",
        ),
        run.check("no NaN", finite, detail="robot root, joints and live balls finite at every step" if finite else "a non-finite state"),
        run.check(
            "PhysX buffers not full",
            physx_ok(physx),
            detail=f"full {physx['full']}, {physx['overflow_errors']} overflow errors, {physx['errors']} PhysX errors",
        ),
        run.check("every ball ended", bool(endings) and all(endings.values()) and seconds < cap_s, detail=f"{endings} in {seconds:.2f} s (cap {cap_s:g} s)"),
    ]
    if film is not None and film_account is not None:
        ok, detail = film.at_24_fps(film_account, seconds)
        results.append(run.check("film at 24 fps", ok, detail=detail))
    run.add_results(
        {
            "drop_rebound_mm": [round(1e3 * float(v), 2) if np.isfinite(v) else None for v in rebound],
            "falls": scene.falls,
            "physx_buffer_use": {name: round(buffer["use"], 4) for name, buffer in physx["buffers"].items()},
            "sim_s": round(seconds, 3),
        }
    )
    measured = {"drop_rebound_mm": [1e3 * float(v) for v in rebound], "pinned": pinned, "film": film_account, "physx": physx, "falls": scene.falls}
    return TemplateReport.model_validate({"passed": all(results), **measured})


def assets_pinned(scene: Any, *, room: bool) -> dict[str, bool]:
    """Each mesh the scene drew, hashed against its pin in :data:`lib.assets.ASSETS`."""

    from lib.assets import ASSETS

    files = scene.files
    if files is None:
        return {}
    drawn = {"ping-pong/table": files.table, "ping-pong/net": files.net, "ping-pong/paddle": files.paddle, "ping-pong/ball": files.ball}
    if room:
        import antioch

        asset = ASSETS["ping-pong/him-lab-room"]
        drawn[asset.name] = str(antioch.fetch_asset(asset.name, version=asset.version))
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() == ASSETS[name].sha256 for name, path in drawn.items()}


__all__ = ["ITTF_REBOUND_M", "RECORDED_FEEDS", "BallWatch", "TemplateReport", "launch_balls", "load_humanoid_at_table"]
