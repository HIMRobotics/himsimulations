"""
The trained Unitree G1 rallies against HIM's distilled opponent in Isaac Lab, beside its blind control:
:func:`play_ping_pong_policy`.

A *policy* is the robot's controller as a function from observation to action; this one is a neural network trained
by reinforcement learning (``docs/policy.md`` explains both from scratch). Thousands of copies of the table, *envs*,
play at once on one GPU. The headline number is legal returns per receive: of the balls the opponent sends toward the
robot's half, the share it returns to land on the opponent's half. The run's checks test the harness, not the robot;
the policy's numbers are reported. The number of envs, the passes' lengths and the seed are a
:class:`~lib.config.PlayParams` preset in ``lib/config.py``.

**Play conditions.** These are the trained policy's grade conditions, except for the ball physics:

- **Ball physics: nominal.** The calibrated priors' centre, every rally, where the grades randomised it per rally.
- **Opening feeds:** the opponent's recorded feeds open each rally; a fresh rally starts 1 s after the last ends.
- **Shoves:** random shoves of up to 0.5 m/s every 5-8 s, as in training: the film's robot occasionally stumbles.
- **Resets:** a fallen robot is reset to its ready stance and stands 0.5 s before its next rally.
- **The forecast through a serve's own-half bounce** is always on; it changes nothing for the feeds, only for serves.

**Two passes.** Every env is reset at the start of each:

1. **The policy:** ``max_sim_s`` of rallies. Env 0 is filmed in the HIM twin.
2. **The blind control:** the same actor with the ball and its forecast hidden. It must collapse.

The grade is :func:`lib.policy.grade.grade`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from lib.config import PlayParams

if TYPE_CHECKING:
    import antioch

PUSH_SPEED_MPS: float = 0.5
"""The random shoves' largest horizontal speed kick, m/s (every 5-8 s), as in every grade of the trained policy."""


class PlayReport(BaseModel):
    """What one run measured; ``passed`` is its verdict."""

    model_config = ConfigDict(frozen=True, extra="allow")

    passed: bool


def play_ping_pong_policy(run: antioch.ScenarioRun, params: PlayParams) -> PlayReport:
    """
    The trained policy rallies against the HIM opponent in Isaac Lab, beside its blind control.

    Two passes over every env: the policy plays ``max_sim_s`` of rallies with env 0 filmed, then the same actor plays
    ``control_s`` with the ball hidden (the negative control, which must collapse). Then the grade.

    :param run: The active scenario run; the checks, results and ``play_summary.json`` are recorded on it.
    :param params: The robot, the checkpoint, the number of envs, the passes' lengths, the film and the seed.
    :return: The report; ``passed`` is the verdict.
    """

    import antioch

    from lib.policy.actor import hide_ball, load_rally_actor
    from lib.policy.film import TwinFilm
    from lib.policy.grade import grade
    from lib.policy.rally import CoupledRally
    from lib.robots import ROBOTS
    from lib.robots.table_scene import build_table_scene

    robot = ROBOTS[params.robot]
    rally = CoupledRally(seed=params.seed, forecast_through_serve=True)
    scene = build_table_scene(
        robot,
        params.num_envs,
        hooks=rally,
        opponent=True,
        push_speed_mps=PUSH_SPEED_MPS,
        episode_s=max(params.max_sim_s, params.control_s) + 2.0,
        seed=params.seed,
    )
    require_policy_joints(scene.joint_names)
    actor = load_rally_actor(params.policy, default_pose=robot.default_pose, device=scene.env.device)  # folded to 146 inputs
    film = TwinFilm(antioch.Logger("play"), scene) if params.film else None

    def filmed(step: int) -> None:
        if film is not None:
            film.frame((step + 1) * scene.step_dt)

    played = rally.play(scene, actor, params.max_sim_s, on_step=filmed)  # terminal: the pass's span; rallies in flight are not graded
    film_account = film.close() if film is not None else None
    run.raise_if_cancelled()
    blind = rally.play(scene, hide_ball(actor), params.control_s)  # negative control: must collapse
    spans = (params.max_sim_s, params.control_s)
    summary = grade(run, played, blind, spans_s=spans, step_s=scene.step_dt, physx=scene.physx_buffers(), film=film, film_account=film_account)
    write_summary(run, params, {**summary, "film": film_account})
    return PlayReport.model_validate(summary)


def require_policy_joints(joint_names: list[str]) -> None:
    """
    Refuse to play when the articulation's joint order is not the one the policy was trained in.

    :raises RuntimeError: On any difference.
    """

    from lib.policy.actor import POLICY_JOINTS

    if list(joint_names) != list(POLICY_JOINTS):
        raise RuntimeError(f"the articulation's joint order {joint_names} is not the policy's {list(POLICY_JOINTS)}")


def write_summary(run: Any, params: PlayParams, summary: dict[str, Any]) -> None:
    """Keep the run's whole account as an artifact, ``play_summary.json``."""

    out = Path(params.artifacts_dir) / "play_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": params.model_dump(), **summary}, indent=1, default=str))
    run.add_artifact(out, name=out.name, description="The policy's and the blind control's metrics, PhysX's buffers and the film's account")


__all__ = ["PUSH_SPEED_MPS", "PlayReport", "play_ping_pong_policy", "require_policy_joints"]
