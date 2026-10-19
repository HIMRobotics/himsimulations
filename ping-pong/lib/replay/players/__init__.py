"""
The two players: their ActorCore bodies, the recorded motion they play, and how they bend each stroke to meet the ball.

- :mod:`.bodies`: spawning a body with a live UsdSkel ``Pose`` prim;
- :mod:`.skeleton`, :mod:`.kinematics`, :mod:`.rotations`: the CC3 rest skeletons and forward kinematics;
- :mod:`.playback`: :class:`~.playback.CharacterAnimator`, the recorded track played at any time, with the planned
  stroke's adaptation blended in and the paddle held in the hand;
- :mod:`.strokes`: planning a stroke from the live ball within human adaptation bounds;
- :mod:`.arm_ik`, :mod:`.grip`, :mod:`.clearance`, :mod:`.anatomy`, :mod:`.limits`: the arm IK, the shakehand grip,
  keeping the arms out of the torso, and human joint ranges and speeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd

    from lib.replay.players.bodies import CharacterHandle
    from lib.replay.players.playback import CharacterAnimator
    from lib.replay.session import LoadedSessionPackage


@dataclass(frozen=True)
class Players:
    """
    Both players on the stage.

    :ivar bodies: Each role's spawned body.
    :ivar animators: Each role's motion.
    """

    bodies: dict[str, CharacterHandle]
    animators: dict[str, CharacterAnimator]


def spawn_players(stage: Usd.Stage, package: LoadedSessionPackage) -> Players:
    """
    Spawn every role's body (after ``world.reset()``) and the motion that plays its recorded track.

    :param stage: The live stage.
    :param package: The session package.
    :return: The players.
    """

    from lib.replay.players.bodies import spawn_character
    from lib.replay.players.playback import CharacterAnimator

    manifest = package.manifest
    bodies = {
        track.role: spawn_character(
            stage,
            role=track.role,
            asset=track.character.asset,
            version=track.character.version,
            look=track.character.look,
            body_scale=track.character.body_scale,
        )
        for track in manifest.tracks
    }
    animators = {
        role: CharacterAnimator(skeleton=body.skeleton, track=manifest.track(role), arrays=package.track_arrays(role))  # type: ignore[arg-type]
        for role, body in bodies.items()
    }
    return Players(bodies=bodies, animators=animators)


__all__ = ["Players", "spawn_players"]
