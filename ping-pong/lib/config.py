"""
Every run's inputs, by name: one frozen preset per scenario case, plus the services, tags and simulator configs.

``scenarios.py``, ``main.py`` and ``demo.ipynb`` pick a preset here and hand it to the feature's one run function::

    replay_him_session(run, REPLAY_PRESETS["session-2-first-point"])    # lib/replay/run.py      (Isaac Sim)
    play_ping_pong_policy(run, PLAY_PRESETS["g1-rally-smoke"])           # lib/policy/run.py      (Isaac Lab)
    load_humanoid_at_table(run, TEMPLATE_PRESETS["unitree-g1"])          # lib/robots/template.py (Isaac Lab)
    retarget_him_strokes(run, RETARGET_PRESETS["unitree-g1-smoke"])      # lib/retarget/run.py    (CPU)

A preset's name is its scenario case's id. Presets are frozen: to try a variant, ``model_copy(update={...})`` one.
"""

from __future__ import annotations

from typing import Literal

from antioch import SimulationConfig
from pydantic import BaseModel, ConfigDict, Field

from lib.assets import G1_RALLY_POLICY, session_asset

# ---- services, tags and simulators ----------------------------------------------------------------------------------

REPLAY_SERVICE: str = "replay"
"""Isaac Sim 6.0.1 (``Dockerfile``): the recorded sessions, rendered. Also its profile's name."""

ROBOT_SERVICE: str = "robot"
"""Isaac Lab 3.0 (``robot.Dockerfile``): the humanoid at the table and the retargeter. Also its profile's name."""

TAGS: tuple[str, ...] = ("ping_pong", "him_robotics")
"""Tags every scenario carries; each scenario adds its own feature tag."""

REPLAY_PHYSICS_HZ: int = 3840
"""The replay's physics rate: the lowest at which the ball passes every calibrated physics test (``docs/physics.md``)."""

RENDER_HZ: int = 24
"""Every rendered frame is one frame of 24 fps camera video."""

REPLAY_SIMULATION_CONFIG = SimulationConfig(
    physics_dt=1.0 / REPLAY_PHYSICS_HZ,
    render_dt=1.0 / RENDER_HZ,
    # DLAA at native resolution: DLSS upscaling blurs the virtual Orbbec frames.
    renderer_quality="quality",
    timeout_s=3 * 3600,
)
"""Isaac Sim for the replay: 160 physics steps per rendered frame."""

ROBOT_SIMULATION_CONFIG = SimulationConfig(log_level="warning", timeout_s=3 * 3600)
"""Isaac Lab for the policy and the template; Lab sets its own physics step (``lib/robots/table_scene.py``)."""

# ---- replay_him_session (lib/replay) --------------------------------------------------------------------------------

TablePhysics = Literal["nominal", "dead-table"]

TABLE_PHYSICS: dict[str, dict[str, float]] = {
    "nominal": {},
    # Negative control: a dead table. The opening drop test rebounds far below the ITTF band, so the replay must fail.
    "dead-table": {"table_restitution": 0.30},
}
"""The engine's ball physics per name, as overrides of the calibrated priors (``lib/physics/data/priors.json``)."""

SESSION_DURATIONS_S: dict[int, float] = {1: 252.9, 2: 192.9, 3: 318.9, 4: 295.4, 5: 298.6, 6: 305.2, 7: 307.9}
"""Each recorded HIM session's length, s."""


class ReplayParams(BaseModel):
    """
    One recorded HIM session replayed in Isaac Sim with live ball physics.

    :ivar session: The recorded session, 1-7.
    :ivar points: Play only the session's first ``points`` points (``None``: all of them).
    :ivar table_physics: The ball physics the engine runs (:data:`TABLE_PHYSICS`); the players always believe nominal.
    :ivar expect_pass: Whether every replay test should pass; ``False`` marks a negative control.
    :ivar max_sim_s: Safety cap on simulated game time; reaching it fails the run.
    :ivar clip: Encode the broadcast camera's MP4 through the compositor.
    :ivar artifacts_dir: Where the run writes its reports inside the service.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: int = Field(ge=1, le=7)
    points: int | None = Field(default=None, ge=1)
    table_physics: TablePhysics = "nominal"
    expect_pass: bool = True
    max_sim_s: float = Field(gt=0.0)
    clip: bool = True
    artifacts_dir: str = Field(default="runs", min_length=1)

    @property
    def package(self) -> str:
        """The session's package, ``asset:<name>@<version>``."""

        return session_asset(self.session)


REPLAY_PRESETS: dict[str, ReplayParams] = {
    **{f"session-{k}": ReplayParams(session=k, max_sim_s=duration + 60.0) for k, duration in SESSION_DURATIONS_S.items()},
    "session-2-first-point": ReplayParams(session=2, points=1, max_sim_s=60.0),
    "nc-dead-table": ReplayParams(session=2, points=1, table_physics="dead-table", expect_pass=False, max_sim_s=60.0),
}

# ---- play_ping_pong_policy (lib/policy) and load_humanoid_at_table (lib/robots/template.py) -------------------------

RobotName = Literal["unitree-g1"]
"""A humanoid in :data:`lib.robots.ROBOTS`."""


class PlayParams(BaseModel):
    """
    The trained policy rallying against the HIM opponent in batched Isaac Lab worlds.

    :ivar robot: The humanoid.
    :ivar policy: The checkpoint, ``asset:<name>@<version>``.
    :ivar num_envs: Parallel worlds.
    :ivar max_sim_s: The policy's pass, simulated s.
    :ivar control_s: The blind control's pass (the same actor with the ball hidden), simulated s.
    :ivar film: Film env 0 in the HIM lab twin at 24 fps.
    :ivar seed: Seed of the opponent's and the worlds' draws.
    :ivar artifacts_dir: Where the run writes its reports inside the service.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    robot: RobotName = "unitree-g1"
    policy: str = G1_RALLY_POLICY
    num_envs: int = Field(ge=1)
    max_sim_s: float = Field(default=30.0, gt=0.0)
    control_s: float = Field(default=30.0, gt=0.0)
    film: bool = True
    seed: int = 0
    artifacts_dir: str = Field(default="runs", min_length=1)


PLAY_PRESETS: dict[str, PlayParams] = {
    "g1-rally-smoke": PlayParams(num_envs=64),
    "g1-rally": PlayParams(num_envs=2048),
}


class TemplateParams(BaseModel):
    """
    A humanoid with its paddle at the table, balls launched at it, holding its default pose: your starting point.

    :ivar robot: The humanoid.
    :ivar num_envs: Parallel worlds.
    :ivar fixed_base: Pin the pelvis to the world. Holding its pose with zero actions, a G1 with no balance controller
        falls; unpin it once your controller balances.
    :ivar balls: Balls launched in turn: ITTF's 30 cm drop onto the table, then HIM's recorded feeds at the robot.
    :ivar max_sim_s: Safety cap on simulated time.
    :ivar film: Film env 0 in the HIM lab twin at 24 fps.
    :ivar artifacts_dir: Where the run writes its reports inside the service.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    robot: RobotName = "unitree-g1"
    num_envs: int = Field(default=4, ge=1)
    fixed_base: bool = True
    balls: int = Field(default=4, ge=1)
    max_sim_s: float = Field(default=20.0, gt=0.0)
    film: bool = True
    artifacts_dir: str = Field(default="runs", min_length=1)


TEMPLATE_PRESETS: dict[str, TemplateParams] = {"unitree-g1": TemplateParams()}

# ---- retarget_him_strokes (lib/retarget) ----------------------------------------------------------------------------


class RetargetParams(BaseModel):
    """
    HIM's mocap strokes retargeted onto a humanoid: each segment solved on CPU, gated, and the kept strokes published.

    :ivar robot: The humanoid.
    :ivar source: The human strokes' store name.
    :ivar source_version: Their pinned version.
    :ivar sessions: The recorded sessions whose strokes are solved.
    :ivar rallies: Stroke-id prefixes to keep, e.g. ``s1_r001`` (empty: every rally of the sessions).
    :ivar workers: CPU worker processes.
    :ivar adam_iters: Adam iterations of each segment's initial solve.
    :ivar lbfgs_iters: L-BFGS iterations of each segment's final solve.
    :ivar artifacts_dir: Where the run writes its reports inside the service.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    robot: RobotName = "unitree-g1"
    source: str = "ping-pong/him-mocap-strokes"
    source_version: str = "v1"
    sessions: tuple[int, ...] = Field(default=(1, 2, 3, 4, 5, 6, 7), min_length=1)
    rallies: tuple[str, ...] = ()
    workers: int = Field(default=24, ge=1)
    adam_iters: int = Field(default=200, ge=0)
    lbfgs_iters: int = Field(default=6000, ge=1)
    artifacts_dir: str = Field(default="runs", min_length=1)


RETARGET_PRESETS: dict[str, RetargetParams] = {
    # Session 1's 10-stroke probe rally.
    "unitree-g1-smoke": RetargetParams(sessions=(1,), rallies=("s1_r001",)),
    "unitree-g1": RetargetParams(),
}
