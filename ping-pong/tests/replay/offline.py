"""
Stand-ins for the simulator, so the replay loop runs end to end on CPU in a test.

:class:`ReferenceBodies` replaces PhysX behind the stepper's ``Bodies`` interface for the paddles and the retired balls
(under the replay's RK4 integrator PhysX never touches the game ball: the stepper's own integrator moves it).
:class:`OfflineWorld` steps the registered physics callback, then the bodies, in the order PhysX runs them; cameras
return grey frames and the bodies' skins are inert. Everything else (the stepper and its hybrid contacts, the stroke
planner, the referee, the players' motion and the tests) is the production code. This is not a simulation run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lib.him_lab import PinholeSpec
from lib.physics.engine.stepper import GamePhysics
from lib.physics.params import BallTableParams, ContactMaterial
from lib.physics.reference.impact import resolve_impact
from lib.replay.ball import ENGINE_SUBSTEPS, RENDER_DT, SPARE_COUNT, engine_settings
from lib.replay.loop import ReplayLoop
from lib.replay.players.playback import CharacterAnimator
from lib.replay.players.skeleton import embedded_skeleton
from lib.replay.recorder import Recorder
from lib.replay.session import LoadedSessionPackage, Role
from lib.telemetry.video import StreamStats

Array = NDArray[np.float64]
FIXTURE = Path(__file__).with_name("data") / "session-2-first-point.zip"
"""The first point of HIM session 2, its tracks cut to session time 5.0-22.5 s, without the recorded frames."""


def _constant(material: ContactMaterial) -> ContactMaterial:
    return material.model_copy(update={"restitution_curve": None, "friction_curve": None})


class ReferenceBodies:
    """The ball and the paddles and retired balls, integrated without PhysX (``filtered``: the adapter's hybrid surfaces)."""

    def __init__(self, params: BallTableParams, paddles: int, *, filtered: tuple[str, ...] = ("table", "net"), spares: int = 0) -> None:
        count = 1 + paddles + spares
        self.filtered = filtered
        self.paddles = paddles
        self.spares = spares
        self.params = params
        self.position = np.zeros((count, 3))
        self.orientation = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
        self.linear = np.zeros((count, 3))
        self.angular = np.zeros((count, 3))
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.table = _constant(params.table_contact)
        self.net = _constant(params.net_contact)
        self.floor = _constant(params.floor_contact)

    def read(self) -> tuple[Array, Array, Array, Array]:
        return self.position.copy(), self.orientation.copy(), self.linear.copy(), self.angular.copy()

    def write_ball(self, position: Array, velocity: Array, spin: Array) -> None:
        self.position[0], self.linear[0], self.angular[0] = position, velocity, spin

    def push_ball(self, force: Array, torque: Array, position: Array) -> None:
        self.force, self.torque = np.asarray(force, float), np.asarray(torque, float)
        del position

    def drive_paddles(self, velocities: Array) -> None:
        self.linear[1 : 1 + self.paddles] = velocities[:, :3]
        self.angular[1 : 1 + self.paddles] = velocities[:, 3:]

    def place_paddles(self, poses: Array) -> None:
        self.position[1 : 1 + self.paddles] = poses[:, :3]
        self.orientation[1 : 1 + self.paddles] = poses[:, 3:]
        self.linear[1 : 1 + self.paddles] = 0.0
        self.angular[1 : 1 + self.paddles] = 0.0

    def write_spare(self, index: int, position: Array, velocity: Array, spin: Array) -> None:
        row = 1 + self.paddles + index
        self.position[row], self.linear[row], self.angular[row] = position, velocity, spin

    def _bounce(self, start: Array, velocity: Array, spin: Array, dt: float, *, filtered: tuple[str, ...] | None = None) -> tuple[Array, Array, Array]:
        """Advance the ball over one step, resolving the first table, net or floor contact on the way."""

        params = self.params
        radius = params.ball.radius
        end = start + velocity * dt
        half_length, half_width = 0.5 * params.table.length, 0.5 * params.table.width
        candidates: list[tuple[float, Array, ContactMaterial]] = []
        if velocity[2] < 0.0 and start[2] >= radius > end[2]:
            fraction = (start[2] - radius) / (start[2] - end[2])
            point = start + fraction * (end - start)
            if abs(point[0]) <= half_length and abs(point[1]) <= half_width and "table" not in (self.filtered if filtered is None else filtered):
                candidates.append((fraction, np.array([0.0, 0.0, 1.0]), self.table))
        floor = -params.table.height + radius
        if velocity[2] < 0.0 and start[2] >= floor > end[2]:
            candidates.append(((start[2] - floor) / (start[2] - end[2]), np.array([0.0, 0.0, 1.0]), self.floor))
        face = 0.5 * params.net.thickness + radius
        for side in (1.0, -1.0):
            plane = side * face
            if (start[0] - plane) * side >= 0.0 > (end[0] - plane) * side:
                fraction = (start[0] - plane) / (start[0] - end[0])
                point = start + fraction * (end - start)
                if (
                    abs(point[1]) <= params.net_half_span
                    and 0.0 <= point[2] <= params.net.height + radius
                    and "net" not in (self.filtered if filtered is None else filtered)
                ):
                    candidates.append((fraction, np.array([side, 0.0, 0.0]), self.net))
        if not candidates:
            return end, velocity, spin
        fraction, normal, material = min(candidates, key=lambda item: item[0])
        impact = resolve_impact(velocity, spin, params.ball, normal, material)
        contact = start + fraction * dt * velocity
        return contact + impact.velocity * (1.0 - fraction) * dt, impact.velocity, impact.spin

    def integrate(self, dt: float) -> None:
        params = self.params
        gravity = np.array([0.0, 0.0, -params.aero.gravity])
        velocity = self.linear[0] + (gravity + self.force / params.ball.mass) * dt
        spin = self.angular[0] + self.torque / params.ball.inertia * dt
        self.position[0], self.linear[0], self.angular[0] = self._bounce(self.position[0], velocity, spin, dt)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        for row in range(1 + self.paddles, len(self.position)):
            # A retired ball: gravity, and every surface resolved by the engine (nothing filtered).
            spare_velocity = self.linear[row] + gravity * dt
            self.position[row], self.linear[row], self.angular[row] = self._bounce(self.position[row], spare_velocity, self.angular[row], dt, filtered=())
        for row in range(1, 1 + self.paddles):
            self.position[row] = self.position[row] + self.linear[row] * dt
            angle = float(np.linalg.norm(self.angular[row])) * dt
            if angle > 0.0:
                axis = self.angular[row] / np.linalg.norm(self.angular[row])
                delta = np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * axis])
                w1, x1, y1, z1 = delta
                w2, x2, y2, z2 = self.orientation[row]
                product = np.array(
                    [
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    ]
                )
                self.orientation[row] = product / np.linalg.norm(product)


class OfflineWorld:
    """Steps the physics callback, then the bodies, ``substeps`` times per rendered frame; ``render`` steps nothing."""

    def __init__(self, bodies: ReferenceBodies) -> None:
        self.bodies = bodies
        self.physics_dt = RENDER_DT / ENGINE_SUBSTEPS
        self.callbacks: dict[str, Callable[[float], None]] = {}
        self.renders = 0

    def add_physics_callback(self, name: str, callback: Callable[[float], None]) -> None:
        self.callbacks[name] = callback

    def remove_physics_callback(self, name: str) -> None:
        self.callbacks.pop(name, None)

    def render(self) -> None:
        self.renders += 1

    def step(self, render: bool = True) -> None:
        del render
        for _ in range(ENGINE_SUBSTEPS):
            for callback in list(self.callbacks.values()):
                callback(self.physics_dt)
            self.bodies.integrate(self.physics_dt)


class GreyRig:
    """A camera rig of flat grey frames."""

    def __init__(self, sizes: dict[str, tuple[int, int]]) -> None:
        self.sizes = sizes

    def read(self) -> dict[str, NDArray[np.uint8]]:
        return {name: np.full((height, width, 3), 120, dtype=np.uint8) for name, (width, height) in self.sizes.items()}

    def close(self) -> None:
        return None


class InertHandle:
    """A body whose skin is not drawn."""

    def __init__(self, skeleton: Any) -> None:
        self.skeleton = skeleton
        self.writes = 0

    def write_pose(self, *args: Any) -> None:
        del args
        self.writes += 1


class CountingSink:
    """A camera stream that counts its frames."""

    def __init__(self) -> None:
        self.frames = 0

    def push(self, *args: Any) -> None:
        self.frames += 1

    def push_jpeg(self, *args: Any) -> None:
        self.frames += 1

    def close(self) -> StreamStats:
        return StreamStats(entity_path="offline", pushed=self.frames, logged=self.frames)


class OfflineRun:
    """The scenario-run surface the loop uses."""

    name = "offline"
    scenario_run_id = None
    sim_s = None

    def raise_if_cancelled(self) -> None:
        return None


def offline_replay(package: LoadedSessionPackage, physics: BallTableParams, belief: BallTableParams, *, max_sim_s: float = 60.0) -> ReplayLoop:
    """Replay a package's points offline: the drop tests, then every point. The engine runs ``physics``; the players believe ``belief``."""

    manifest = package.manifest
    roles: tuple[Role, ...] = tuple(track.role for track in manifest.tracks)
    bodies = ReferenceBodies(physics, len(roles), spares=SPARE_COUNT)
    spares = tuple(f"/spare/{index}" for index in range(SPARE_COUNT))
    parks = tuple((-6.5, -3.5 + 1.0 * index, -0.7) for index in range(SPARE_COUNT))
    game = GamePhysics(
        physics, engine_settings(physics), "/ball", {role: f"/paddle/{role}" for role in roles}, bodies=bodies, spare_paths=spares, spare_park=parks
    )
    handles = {role: InertHandle(embedded_skeleton(manifest.track(role).character.body)) for role in roles}
    animators = {role: CharacterAnimator(skeleton=handles[role].skeleton, track=manifest.track(role), arrays=package.track_arrays(role)) for role in roles}
    camera = PinholeSpec("orbbec_8E", 640, 480, 463.0, 463.0, 319.5, 239.5, np.eye(3), np.array([0.0, 0.0, 3.0]), True)
    recorder = Recorder({"orbbec_8E": camera}, roles, GreyRig({"orbbec_8E": (640, 480)}), GreyRig({"broadcast": (1280, 720)}), directory=Path("."))  # type: ignore[arg-type]
    recorder.open_video(lambda _channel, _crf: CountingSink())
    replay = ReplayLoop(
        OfflineRun(),  # type: ignore[arg-type]
        package,
        belief,
        game,
        handles,  # type: ignore[arg-type]
        animators,
        recorder,
        max_sim_s=max_sim_s,
        world=OfflineWorld(bodies),
        stage=object(),
        show_ball=lambda _visible: None,
        check_usdskel=False,
        scale_spare=lambda _index, _scale: None,
    )
    replay.drop_tests()
    replay.play()
    return replay
