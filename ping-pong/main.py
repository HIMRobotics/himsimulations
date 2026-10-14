"""
Run one ping-pong preset from inside its Antioch service, and exit nonzero if it fails::

    antioch service exec --service replay --no-stream -- python main.py replay session-2-first-point
    antioch service exec --service robot --no-stream -- python main.py play g1-rally-smoke
    antioch service exec --service robot --no-stream -- python main.py template unitree-g1
    antioch service exec --service robot --no-stream -- python main.py retarget unitree-g1-smoke
    python main.py list

Each command calls the same scenario as ``antioch scenario run``. ``list`` and ``--help`` start no simulator.
"""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable, Mapping
from typing import Any

import click

from lib.config import PLAY_PRESETS, REPLAY_PRESETS, RETARGET_PRESETS, TEMPLATE_PRESETS

ENGINES: dict[str, str] = {"replay": "isaac-sim-6.0.1", "play": "isaac-lab-3.0", "template": "isaac-lab-3.0", "retarget": "isaac-lab-3.0"}
"""The engine each command's service runs; the service's ``ANTIOCH_ENGINE`` names it."""


def _run(command: str, scenario: Callable[..., Mapping[str, Any]], preset: str, config: Any) -> None:
    """Start the simulator (when the scenario needs one), run the scenario, and exit with its verdict."""

    engine = os.environ.get("ANTIOCH_ENGINE")
    if engine != ENGINES[command]:
        raise click.UsageError(f"'{command}' runs on engine {ENGINES[command]}, but this service's ANTIOCH_ENGINE is {engine!r}")

    import antioch

    if config is not None:
        antioch.start_simulation(config)
    passed = False
    try:
        passed = scenario(preset=preset).get("passed") is True
    except BaseException:
        # Isaac's fast shutdown can end the process inside close() below, before the interpreter prints this.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if antioch.is_running():
            antioch.application().close(exit_code=0 if passed else 1)
    raise click.exceptions.Exit(0 if passed else 1)


@click.group(context_settings={"show_default": True})
def cli() -> None:
    """Ping-pong on Antioch: replay HIM's sessions, play the trained G1, load the template, retarget strokes."""


@cli.command()
@click.argument("preset", type=click.Choice(list(REPLAY_PRESETS)), default="session-2-first-point")
def replay(preset: str) -> None:
    """Replay a recorded HIM session in Isaac Sim (replay service)."""

    from lib.config import REPLAY_SIMULATION_CONFIG
    from scenarios import replay_him_session

    _run("replay", replay_him_session, preset, REPLAY_SIMULATION_CONFIG)


@cli.command()
@click.argument("preset", type=click.Choice(list(PLAY_PRESETS)), default="g1-rally-smoke")
def play(preset: str) -> None:
    """The trained G1 rallies against the HIM opponent in Isaac Lab (robot service)."""

    from lib.config import ROBOT_SIMULATION_CONFIG
    from scenarios import play_ping_pong_policy

    _run("play", play_ping_pong_policy, preset, ROBOT_SIMULATION_CONFIG)


@cli.command()
@click.argument("preset", type=click.Choice(list(TEMPLATE_PRESETS)), default="unitree-g1")
def template(preset: str) -> None:
    """A humanoid at the table, holding its pose, in Isaac Lab (robot service)."""

    from lib.config import ROBOT_SIMULATION_CONFIG
    from scenarios import load_humanoid_at_table

    _run("template", load_humanoid_at_table, preset, ROBOT_SIMULATION_CONFIG)


@cli.command()
@click.argument("preset", type=click.Choice(list(RETARGET_PRESETS)), default="unitree-g1-smoke")
def retarget(preset: str) -> None:
    """Retarget HIM's mocap strokes onto a humanoid, on CPU (robot service)."""

    from scenarios import retarget_him_strokes

    _run("retarget", retarget_him_strokes, preset, None)


@cli.command("list")
def list_presets() -> None:
    """List every preset without starting a simulator."""

    for name, params in REPLAY_PRESETS.items():
        click.echo(f"replay   {name:<24} {params.model_dump_json()}")
    for name, params in PLAY_PRESETS.items():
        click.echo(f"play     {name:<24} {params.model_dump_json()}")
    for name, params in TEMPLATE_PRESETS.items():
        click.echo(f"template {name:<24} {params.model_dump_json()}")
    for name, params in RETARGET_PRESETS.items():
        click.echo(f"retarget {name:<24} {params.model_dump_json()}")


if __name__ == "__main__":
    cli()
