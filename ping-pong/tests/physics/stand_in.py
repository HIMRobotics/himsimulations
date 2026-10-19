"""
A stand-in for PhysX behind the replay stepper's ``Bodies`` interface, so the adapter runs in a test.

The ball moves by semi-implicit Euler (PhysX's scheme) under gravity and the force the stepper applies, and bounces off
the table, net and floor by the reference impact law with the surface's constant restitution. Under the replay's
``rk4`` integrator PhysX never touches the ball, so the ball's physics there is exactly the production code's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from lib.physics.params import BallTableParams, ContactMaterial
from lib.physics.reference.impact import resolve_impact

Array = NDArray[np.float64]


def _constant(material: ContactMaterial) -> ContactMaterial:
    return material.model_copy(update={"restitution_curve": None, "friction_curve": None})


class ReferenceBodies:
    """The ball (row 0) and the paddles, integrated without PhysX; ``filtered`` surfaces are the adapter's (hybrid)."""

    def __init__(self, params: BallTableParams, paddles: int, *, filtered: tuple[str, ...] = ("table", "net")) -> None:
        self.params, self.paddles, self.filtered = params, paddles, filtered
        count = 1 + paddles
        self.position = np.zeros((count, 3))
        self.orientation = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
        self.linear = np.zeros((count, 3))
        self.angular = np.zeros((count, 3))
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.table, self.net, self.floor = (_constant(m) for m in (params.table_contact, params.net_contact, params.floor_contact))

    def read(self) -> tuple[Array, Array, Array, Array]:
        return self.position.copy(), self.orientation.copy(), self.linear.copy(), self.angular.copy()

    def write_ball(self, position: Array, velocity: Array, spin: Array) -> None:
        self.position[0], self.linear[0], self.angular[0] = position, velocity, spin

    def push_ball(self, force: Array, torque: Array, position: Array) -> None:
        del position
        self.force, self.torque = np.asarray(force, float), np.asarray(torque, float)

    def drive_paddles(self, velocities: Array) -> None:
        self.linear[1:], self.angular[1:] = velocities[:, :3], velocities[:, 3:]

    def place_paddles(self, poses: Array) -> None:
        self.position[1:], self.orientation[1:] = poses[:, :3], poses[:, 3:]
        self.linear[1:] = self.angular[1:] = 0.0

    def write_spare(self, index: int, position: Array, velocity: Array, spin: Array) -> None:
        del index, position, velocity, spin

    def integrate(self, dt: float) -> None:
        params = self.params
        radius = params.ball.radius
        velocity = self.linear[0] + (np.array([0.0, 0.0, -params.aero.gravity]) + self.force / params.ball.mass) * dt
        spin = self.angular[0] + self.torque / params.ball.inertia * dt
        start = self.position[0]
        end = start + velocity * dt
        candidates: list[tuple[float, Array, ContactMaterial]] = []
        if velocity[2] < 0.0 and start[2] >= radius > end[2] and "table" not in self.filtered:
            point = start + (start[2] - radius) / (start[2] - end[2]) * (end - start)
            if abs(point[0]) <= 0.5 * params.table.length and abs(point[1]) <= 0.5 * params.table.width:
                candidates.append(((start[2] - radius) / (start[2] - end[2]), np.array([0.0, 0.0, 1.0]), self.table))
        floor = -params.table.height + radius
        if velocity[2] < 0.0 and start[2] >= floor > end[2]:
            candidates.append(((start[2] - floor) / (start[2] - end[2]), np.array([0.0, 0.0, 1.0]), self.floor))
        if candidates:
            fraction, normal, material = min(candidates, key=lambda item: item[0])
            impact = resolve_impact(velocity, spin, params.ball, normal, material)
            end = start + fraction * dt * velocity + impact.velocity * (1.0 - fraction) * dt
            velocity, spin = impact.velocity, impact.spin
        self.position[0], self.linear[0], self.angular[0] = end, velocity, spin
        self.force, self.torque = np.zeros(3), np.zeros(3)
        self.position[1:] += self.linear[1:] * dt


class StepWorld:
    """Runs the registered physics callback once per engine step, then the stand-in engine."""

    def __init__(self, engine: ReferenceBodies, dt: float) -> None:
        self.engine, self.dt = engine, dt
        self.callbacks: dict[str, Any] = {}

    def add_physics_callback(self, name: str, fn: Any) -> None:
        self.callbacks[name] = fn

    def remove_physics_callback(self, name: str) -> None:
        self.callbacks.pop(name, None)

    def step(self, render: bool = False) -> None:
        del render
        for fn in list(self.callbacks.values()):
            fn(self.dt)
        self.engine.integrate(self.dt)
