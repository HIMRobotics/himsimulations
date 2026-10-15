"""
The four ping-pong scenarios: the entry points Antioch runs. New here? ``README.md`` explains all of this from scratch.

A *scenario* is one kind of experiment, and a *case* is one named set of inputs for it. Each scenario below is a thin
wrapper: it picks the case's frozen preset from ``lib/config.py`` (the preset's name is the case id) and calls the
feature's run function, which records its own checks on the run and returns a report. ``main.py`` and ``demo.ipynb``
call the same functions, so a run looks the same whichever entrypoint started it.

=========================  =======  ==============================================================
scenario                   service  cases
=========================  =======  ==============================================================
``replay_him_session``     replay   ``session-1`` .. ``session-7``, ``session-2-first-point``,
                                    ``nc-dead-table`` (negative control: must fail its drop test)
``play_ping_pong_policy``  robot    ``g1-rally-smoke`` (64 envs), ``g1-rally`` (2048 envs)
``load_humanoid_at_table`` robot    ``unitree-g1``
``retarget_him_strokes``   robot    ``unitree-g1-smoke`` (10 strokes), ``unitree-g1`` (all sessions)
=========================  =======  ==============================================================

From ``ping-pong/``, list them with ``uv run antioch scenario collect`` (local and free), run one with
``uv run antioch scenario run <scenario> --cases <case>``, or run a suite from ``antioch.yaml`` with
``uv run antioch suite run <suite>``. To add a case, add a preset to ``lib/config.py``: its key becomes the case id.
"""

from __future__ import annotations

import antioch

from lib.config import (
    PLAY_PRESETS,
    REPLAY_PRESETS,
    REPLAY_SERVICE,
    REPLAY_SIMULATION_CONFIG,
    RETARGET_PRESETS,
    ROBOT_SERVICE,
    ROBOT_SIMULATION_CONFIG,
    TAGS,
    TEMPLATE_PRESETS,
)


def _cases(presets: dict, tags: dict[str, list[str]]) -> list:
    """One case per preset, its id the preset's name."""

    return [antioch.case({"preset": name}, id=name, tags=tags.get(name, [])) for name in presets]


@antioch.scenario(
    config=REPLAY_SIMULATION_CONFIG,
    capture=False,
    tags=[*TAGS, "replay"],
    service=REPLAY_SERVICE,
    profile=REPLAY_SERVICE,
    cases=_cases(REPLAY_PRESETS, {"session-2-first-point": ["smoke"], "nc-dead-table": ["smoke", "negative-control"], "session-2": ["evidence"]}),
)
def replay_him_session(run: antioch.ScenarioRun, preset: str = "session-2-first-point") -> None:
    """
    Replay a recorded HIM session in Isaac Sim: live ball physics, both players bending their recorded strokes to meet it.
    """

    from lib.replay.run import replay_him_session as replay

    report = replay(run, REPLAY_PRESETS[preset])
    run.add_result("passed", report.passed)


@antioch.scenario(
    config=ROBOT_SIMULATION_CONFIG,
    capture=False,
    tags=[*TAGS, "policy"],
    service=ROBOT_SERVICE,
    profile=ROBOT_SERVICE,
    cases=_cases(PLAY_PRESETS, {"g1-rally-smoke": ["smoke"], "g1-rally": ["evidence"]}),
)
def play_ping_pong_policy(run: antioch.ScenarioRun, preset: str = "g1-rally-smoke") -> None:
    """
    The trained Unitree G1 rallies against the HIM opponent in Isaac Lab, beside its blind control.
    """

    from lib.policy.run import play_ping_pong_policy as play

    report = play(run, PLAY_PRESETS[preset])
    run.add_result("passed", report.passed)


@antioch.scenario(
    config=ROBOT_SIMULATION_CONFIG,
    capture=False,
    tags=[*TAGS, "template"],
    service=ROBOT_SERVICE,
    profile=ROBOT_SERVICE,
    cases=_cases(TEMPLATE_PRESETS, {"unitree-g1": ["smoke"]}),
)
def load_humanoid_at_table(run: antioch.ScenarioRun, preset: str = "unitree-g1") -> None:
    """
    A humanoid with its paddle at the HIM table, balls launched at it, holding its pose: the template for your controller.
    """

    from lib.robots.template import load_humanoid_at_table as load

    report = load(run, TEMPLATE_PRESETS[preset])
    run.add_result("passed", report.passed)


@antioch.scenario(
    # CPU only: the retargeter needs no simulator.
    config=None,
    capture=False,
    tags=[*TAGS, "retarget"],
    service=ROBOT_SERVICE,
    profile=ROBOT_SERVICE,
    cases=_cases(RETARGET_PRESETS, {"unitree-g1-smoke": ["smoke"]}),
)
def retarget_him_strokes(run: antioch.ScenarioRun, preset: str = "unitree-g1-smoke") -> None:
    """
    Retarget HIM's mocap strokes onto a humanoid, gate them for reach, speed, torque, balance and feet, and publish them.
    """

    from lib.retarget.run import retarget_him_strokes as retarget

    report = retarget(run, RETARGET_PRESETS[preset])
    run.add_result("passed", report.passed)
