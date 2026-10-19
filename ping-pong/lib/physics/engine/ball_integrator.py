"""
The game ball integrated by the adapter itself (``ball_integrator="rk4"``, the replay's), not by PhysX.

Under the ``physx`` integrator PhysX advances the game ball at the game step
(1/3840 s) by semi-implicit Euler under gravity plus the air force the stepper
applies, so its flight error is first order in the step: it does not converge
with the step the way the oracle does, and the step has to be small (3840 Hz)
for the calibrated flight, drag and parity tests to pass. Each PhysX step then
costs 0.5-1.5 ms of engine time, beyond the adapter's own work.

Every ball contact is already the adapter's own (the hybrid paddle, table, net,
posts and clamps, undercarriage, floor, room and body sweeps), so PhysX never
touches the game ball. :class:`IntegratedBodies` therefore keeps the ball's
state in float64 and advances it with the reference model's classical RK4
(:func:`lib.physics.reference.flight.rk4_step`, the oracle's own derivative):

- a free step integrates from the step-start state over the whole step;
- a step with a hybrid contact integrates the post-impact state from the
  contact time (``f dt``) to the step's end, which is the event-driven solution
  the oracle computes (the semi-implicit start-state rewrite of
  :func:`lib.physics.engine.contacts.impose_contact` is not needed);
- a parked ball is held where it was put.

The stepper's sweeps, impact law, sustained contacts and bookkeeping are
unchanged and still run at the game step; only the integration moves. PhysX
keeps everything else (the driven paddles and the retired balls) at a coarser
engine step, and the ball's PhysX body becomes a passive marker (no gravity, no
collisions) whose pose is written once per engine step for rendering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from lib.physics.reference.flight import FlightModel, rk4_step

Array = NDArray[np.float64]


class IntegratedBodies:
    """
    The stepper's ``Bodies`` with the game ball integrated in Python and the rest (paddles, retired balls) in the engine.

    :param engine: The engine side for the paddles and retired balls (:class:`lib.physics.engine.stepper.TensorBodies`,
        or a stand-in); its row 0 is the ball's passive marker body.
    :param model: The flight model (gravity, drag, Magnus, spin decay).
    """

    def __init__(self, engine: Any, model: FlightModel) -> None:
        self.engine = engine
        self.model = model
        self.paddles = getattr(engine, "paddles", 0)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.spin = np.zeros(3)
        self.hold = False
        """A parked ball is held (not integrated) until it is placed."""
        self._contact: tuple[Array, float] | None = None
        self._cache: tuple[Array, Array, Array, Array] | None = None
        self.engine_reads = 0
        self.marker_writes = 0
        self.integrated_s = 0.0
        self.trace: list[Array] | None = None
        """When a list, every integration appends the ball's state after it (to log every game step)."""

    # ---- engine lifecycle ------------------------------------------------------------------

    def bind(self) -> None:
        """Bind the engine view (after ``world.reset()``) and start the ball where its body is, at rest."""

        bind = getattr(self.engine, "bind", None)
        if bind is not None:
            bind()
        positions, _orientations, _linear, _angular = self.engine.read()
        self.write_ball(positions[0], np.zeros(3), np.zeros(3))
        self._cache = None

    def invalidate(self) -> None:
        """The engine stepped: the next :meth:`read` reads it again."""

        self._cache = None

    # ---- Bodies protocol ----------------------------------------------------------------------

    def read(self) -> tuple[Array, Array, Array, Array]:
        """Positions, orientations (wxyz), linear and angular velocities: the integrated ball, then the engine's rows."""

        cache = self._cache
        if cache is None:
            cache = self._cache = self.engine.read()
            self.engine_reads += 1
        positions, orientations, linear, angular = (array.copy() for array in cache)
        positions[0], linear[0], angular[0] = self.position, self.velocity, self.spin
        orientations[0] = (1.0, 0.0, 0.0, 0.0)
        return positions, orientations, linear, angular

    def ball(self) -> Array:
        """The ball's state, 9 floats."""

        return np.concatenate([self.position, self.velocity, self.spin])

    def write_ball(self, position: Array, velocity: Array, spin: Array) -> None:
        """Set the ball's state at the step start (a placement or the parking spot)."""

        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.asarray(velocity, dtype=np.float64).copy()
        self.spin = np.asarray(spin, dtype=np.float64).copy()
        self._contact = None

    def impose(self, after: Array, fraction: float) -> None:
        """A hybrid contact this step: the ball leaves with ``after`` (centre, velocity, spin) at ``fraction`` of the step."""

        after = np.asarray(after, dtype=np.float64)
        self.position, self.velocity, self.spin = after[0:3].copy(), after[3:6].copy(), after[6:9].copy()
        self._contact = (after.copy(), min(max(float(fraction), 0.0), 1.0))

    def push_ball(self, force: Array, torque: Array, position: Array) -> None:
        """No engine force: the integrator applies the flight model itself."""

        del force, torque, position

    def drive_paddles(self, velocities: Array) -> None:
        """See ``Bodies``: the paddles stay engine bodies."""

        self.engine.drive_paddles(velocities)

    def place_paddles(self, poses: Array) -> None:
        """See ``Bodies``."""

        self.engine.place_paddles(poses)
        self._cache = None

    def write_spare(self, index: int, position: Array, velocity: Array, spin: Array) -> None:
        """See ``Bodies``: retired balls stay engine bodies."""

        self.engine.write_spare(index, position, velocity, spin)
        self._cache = None

    # ---- integration ---------------------------------------------------------------------------

    def integrate(self, dt: float) -> None:
        """
        Advance the ball over ``dt``: RK4 from the step start, or from the contact time after a hybrid contact.

        :param dt: The game step, s, or a clear-flight stretch of several (no contact can fall inside one).
        """

        contact = self._contact
        self._contact = None
        if self.hold:
            if self.trace is not None:
                self.trace.append(self.ball())
            return
        if contact is not None:
            state, fraction = contact
            h = (1.0 - fraction) * dt
            y = [float(v) for v in state]
        else:
            h = dt
            y = [*map(float, self.position), *map(float, self.velocity), *map(float, self.spin)]
        if h > 0.0:
            y = rk4_step(self.model, y, h)
            self.integrated_s += h
        self.position = np.array(y[0:3])
        self.velocity = np.array(y[3:6])
        self.spin = np.array(y[6:9])
        if self.trace is not None:
            self.trace.append(np.array(y))

    def write_marker(self) -> None:
        """Put the ball's passive engine body (what the cameras render) at the integrated pose, at rest."""

        write = getattr(self.engine, "write_ball", None)
        if write is not None:
            write(self.position, np.zeros(3), np.zeros(3))
            self.marker_writes += 1


__all__ = ["IntegratedBodies"]
