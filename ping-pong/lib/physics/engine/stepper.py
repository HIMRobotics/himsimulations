"""
What runs every physics step of a game: air forces, the paddle drive, the hybrid contacts, and contact bookkeeping.

One physics callback (Isaac Sim ``World.add_physics_callback``) reads the
ball and both paddles once through the raw physics-tensor view with
preallocated buffers (on CPU the warp arrays' numpy views are zero-copy, so a
step allocates nothing; the experimental ``RigidPrim`` wrapper costs about 1.5 ms
per step against 0.4 ms for this path, measured on production), then:

1. recognises the engine-resolved contact of the step just taken (table, net,
   floor) from its velocity change, and checks for tunnelling;
2. sweeps the ball against both paddles (and the table and net in hybrid mode) over
   the coming step and, at a touch, writes the reference impact model's
   outcome (:mod:`lib.physics.engine.contacts`);
3. applies the reference model's aerodynamic force and torque
   (:meth:`lib.physics.reference.flight.FlightModel.wrench`);
4. commands each paddle the velocity that lands it on its target pose at the
   end of the step (a dead-beat velocity drive of a dynamic body); the target
   comes from the paddle segment the game loop sets once per render tick.

Exceptions inside a Kit physics callback are swallowed by Kit, so the
callback records the first one and the game loop raises it.

**Ball integrator.** With ``ball_integrator="physx"`` (the default) PhysX
integrates the ball at the game step and the callback runs once per PhysX step.
With ``"rk4"`` the adapter integrates the game ball itself
(:class:`lib.physics.engine.ball_integrator.IntegratedBodies`): PhysX steps
only the paddles and retired balls, at a coarser engine step, and each PhysX
callback runs the same per-step logic for every game step the engine step
covers, integrating the ball between them with the reference model's RK4 and
resuming from the contact time after a hybrid contact.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from lib.physics.engine.ball_integrator import IntegratedBodies
from lib.physics.engine.body_capsules import BODY_CONTACT, BodyMotion, sweep_body
from lib.physics.engine.colliders import left_room, room_gap
from lib.physics.engine.contacts import (
    NATIVE_MERGE_S,
    REST_SPEED_MPS,
    BladePose,
    ContactEpisodes,
    ContactEvent,
    FixtureGeometry,
    FrameGeometry,
    ImposedState,
    SweptContact,
    box_distance,
    classify_native,
    crossed_floor,
    crossed_frame,
    crossed_net,
    crossed_table,
    extend_event,
    impose_contact,
    kinetic_energy,
    sweep_blade,
    sweep_fixtures,
    sweep_floor,
    sweep_frame,
    sweep_net,
    sweep_room,
    sweep_table_box,
    table_board,
)
from lib.physics.engine.settings import EngineSettings
from lib.physics.params import BallTableParams
from lib.physics.reference.blade import blade_touch
from lib.physics.reference.flight import FlightModel
from lib.physics.reference.surfaces import Net, net_fixtures

Array = NDArray[np.float64]


def _quat_multiply(a: Array, b: Array) -> Array:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _quat_rotvec(q: Array) -> Array:
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return 2.0 * q[1:]
    return q[1:] / s * (2.0 * math.atan2(s, float(q[0])))


def _quat_slerp(a: Array, b: Array, u: float) -> Array:
    dot = float(a @ b)
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        out = a + u * (b - a)
        return out / np.linalg.norm(out)
    theta = math.acos(min(dot, 1.0))
    return (math.sin((1.0 - u) * theta) * a + math.sin(u * theta) * b) / math.sin(theta)


def quat_matrix(q: Array) -> Array:
    """Column-convention rotation matrix of a ``(w, x, y, z)`` quaternion."""

    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


PADDLE_RECONTACT_COOLDOWN_S: float = 0.03125
"""After a hybrid paddle hit, the same blade cannot touch the ball again for this long (30 steps at 960 Hz)."""
PARK_REWRITE_S: float = 1.0 / 24.0
"""A parked (hidden, out-of-play) ball is written back to its parking spot this often, not every step: between
rewrites it falls at most 9 mm, far from everything, and the engine calls it saves dominate the step's cost."""
CLEAR_FLIGHT_STEPS: tuple[int, ...] = (16, 4)
"""Under the ``rk4`` integrator, a stretch of this many game steps in which the ball cannot reach any surface is one RK4
step (largest first): no sweep can fire there, and RK4's error over 16 game steps (4.2 ms) is below 1e-9 m."""
CLEAR_FLIGHT_MARGIN_M: float = 0.01
"""Extra clearance a clear-flight stretch keeps from every surface, beyond the ball's and the blade's reach over it."""
KNOT_SPLIT_MARGIN_S: float = 1e-9
"""A contact knot closer than this to a step's end is on the boundary: the step is swept whole."""
NATIVE_FIXTURE_GAP_M: float = 0.015
"""A native contact this close to a net post or clamp (the reference model's ``NetFixture``) is tagged with its region."""


@dataclass(frozen=True)
class PaddleSegment:
    """
    A paddle's target over one render tick: cubic Hermite position, slerped orientation.

    :ivar t0: Segment start, game time s.
    :ivar t1: Segment end, s.
    :ivar p0: Blade-centre position at ``t0``, m.
    :ivar v0: Velocity at ``t0``, m/s.
    :ivar p1: Position at ``t1``.
    :ivar v1: Velocity at ``t1``.
    :ivar q0: Orientation at ``t0``, ``(w, x, y, z)``.
    :ivar q1: Orientation at ``t1``.
    :ivar knot: An interior knot ``(t, position, velocity, orientation)`` (a planned contact inside the tick), so the
        blade passes the contact with the animated velocity rather than a cubic's approximation of it.
    """

    t0: float
    t1: float
    p0: Array
    v0: Array
    p1: Array
    v1: Array
    q0: Array
    q1: Array
    knot: tuple[float, Array, Array, Array] | None = None

    def at(self, t: float) -> tuple[Array, Array, Array, Array]:
        """Position, orientation, velocity and angular velocity at ``t`` (clamped to the segment)."""

        if self.knot is not None:
            tm, pm, vm, qm = self.knot
            if self.t0 < tm < self.t1:
                if t <= tm:
                    return PaddleSegment(t0=self.t0, t1=tm, p0=self.p0, v0=self.v0, p1=pm, v1=vm, q0=self.q0, q1=qm).at(t)
                return PaddleSegment(t0=tm, t1=self.t1, p0=pm, v0=vm, p1=self.p1, v1=self.v1, q0=qm, q1=self.q1).at(t)
        h = self.t1 - self.t0
        u = min(max((t - self.t0) / h, 0.0), 1.0)
        u2, u3 = u * u, u * u * u
        position = (2 * u3 - 3 * u2 + 1) * self.p0 + (u3 - 2 * u2 + u) * h * self.v0 + (-2 * u3 + 3 * u2) * self.p1 + (u3 - u2) * h * self.v1
        velocity = ((6 * u2 - 6 * u) * self.p0 + (3 * u2 - 4 * u + 1) * h * self.v0 + (-6 * u2 + 6 * u) * self.p1 + (3 * u2 - 2 * u) * h * self.v1) / h
        rotation = _quat_slerp(self.q0, self.q1, u)
        conj = self.q0 * np.array([1.0, -1.0, -1.0, -1.0])
        angular = _quat_rotvec(_quat_multiply(self.q1, conj)) / h
        return position, rotation, velocity, angular


@dataclass
class StepStats:
    """Per-run counters of the physics callback."""

    callbacks: int = 0
    """Game steps (one per PhysX step under the ``physx`` integrator; several per engine step under ``rk4``)."""
    engine_steps: int = 0
    """PhysX steps whose callback ran."""
    clear_flight_steps: int = 0
    """Game steps integrated inside clear-flight stretches (``rk4`` only), without sweeps."""
    seconds: float = 0.0
    max_seconds: float = 0.0
    aero_seconds: float = 0.0
    engine_io_seconds: float = 0.0
    engine_calls: int = 0
    contacts_hybrid: int = 0
    contacts_edge: int = 0
    """Hybrid paddle contacts on a blade's rounded edge (``settings.paddle.blade_edge``; region ``blade_edge``)."""
    contacts_engine: int = 0
    tunnel_table: int = 0
    tunnel_net: int = 0
    tunnel_floor: int = 0
    tunnel_paddle: int = 0
    tunnel_frame: int = 0
    contacts_frame: int = 0
    contacts_body: int = 0
    body_embedded_steps: int = 0
    left_room: int = 0
    paddle_error_m: list[float] = field(default_factory=list)
    paddle_error_deg: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class BallPlacement:
    """
    A requested ball state, written at the start of the physics step nearest ``at_time``.

    :ivar position: Ball centre, m.
    :ivar velocity: m/s.
    :ivar spin: rad/s.
    :ivar label: What it is (``serve 0``, ``continuity stroke 3``, ``drop 1``).
    :ivar at_time: Game time to place it, s (``None``: the next step).
    """

    position: Array
    velocity: Array
    spin: Array
    label: str
    at_time: float | None = None
    retire: bool = False


class Bodies(Protocol):
    """The engine side of the stepper: read the ball and paddles, write what the step decided."""

    def read(self) -> tuple[Array, Array, Array, Array]:
        """Positions ``(N, 3)``, orientations ``(N, 4)`` wxyz, linear and angular velocities ``(N, 3)``: ball, then paddles."""
        ...

    def write_ball(self, position: Array, velocity: Array, spin: Array) -> None:
        """Set the ball's pose and velocities."""
        ...

    def push_ball(self, force: Array, torque: Array, position: Array) -> None:
        """Apply a world-frame force and torque to the ball for this step."""
        ...

    def drive_paddles(self, velocities: Array) -> None:
        """Set the paddles' linear and angular velocities, ``(paddles, 6)``."""
        ...

    def place_paddles(self, poses: Array) -> None:
        """Teleport the paddles to ``(paddles, 7)`` position + wxyz, at rest."""
        ...


class TensorBodies:
    """
    The ball and paddles through Isaac's raw physics-tensor view, with preallocated buffers (CPU, zero-copy).

    :param paths: Ball prim, then paddle prims.
    """

    def __init__(self, paths: tuple[str, ...], paddles: int | None = None) -> None:
        from isaacsim.core.experimental.prims import RigidPrim

        self.paths = paths
        self.paddles = len(paths) - 1 if paddles is None else paddles
        self._view = RigidPrim(list(paths))
        self._rb: Any = None

    def bind(self) -> None:
        """
        Bind the raw view and allocate its buffers (after ``world.reset()``).

        :raises RuntimeError: If the view is not valid or not on CPU
        """

        import warp as wp

        self._rb = self._view._physics_rigid_body_view
        if self._rb is None:
            raise RuntimeError("the physics tensor view is not valid after reset")
        device = self._rb.get_transforms().device
        if not str(device).startswith("cpu"):
            raise RuntimeError(f"the game adapter expects CPU physics tensors, got {device}")
        count = len(self.paths)
        self._index_ball = wp.array(np.array([0], np.int32), dtype=wp.int32, device=device)
        self._index_paddles = wp.array(np.arange(1, 1 + self.paddles, dtype=np.int32), dtype=wp.int32, device=device)
        self._index_spare = [wp.array(np.array([row], np.int32), dtype=wp.int32, device=device) for row in range(1 + self.paddles, count)]
        self._velocity_wp = wp.zeros((count, 6), dtype=wp.float32, device=device)
        self._transform_wp = wp.zeros((count, 7), dtype=wp.float32, device=device)
        self._force_wp = wp.zeros((count, 3), dtype=wp.float32, device=device)
        self._torque_wp = wp.zeros((count, 3), dtype=wp.float32, device=device)
        self._position_wp = wp.zeros((count, 3), dtype=wp.float32, device=device)
        self._velocity = self._velocity_wp.numpy()
        self._transform = self._transform_wp.numpy()
        self._force = self._force_wp.numpy()
        self._torque = self._torque_wp.numpy()
        self._position = self._position_wp.numpy()

    def read(self) -> tuple[Array, Array, Array, Array]:
        """See :class:`Bodies`."""

        transforms = np.asarray(self._rb.get_transforms().numpy(), dtype=np.float64)
        velocities = np.asarray(self._rb.get_velocities().numpy(), dtype=np.float64)
        return transforms[:, :3].copy(), transforms[:, [6, 3, 4, 5]].copy(), velocities[:, :3].copy(), velocities[:, 3:].copy()

    def write_ball(self, position: Array, velocity: Array, spin: Array) -> None:
        """See :class:`Bodies`."""

        self._transform[0, :3] = position
        self._transform[0, 3:] = (0.0, 0.0, 0.0, 1.0)
        self._velocity[0, :3] = velocity
        self._velocity[0, 3:] = spin
        self._rb.set_transforms(self._transform_wp, self._index_ball)
        self._rb.set_velocities(self._velocity_wp, self._index_ball)

    def push_ball(self, force: Array, torque: Array, position: Array) -> None:
        """See :class:`Bodies`."""

        self._force[0] = force
        self._torque[0] = torque
        self._position[0] = position
        self._rb.apply_forces_and_torques_at_position(self._force_wp, self._torque_wp, self._position_wp, indices=self._index_ball, is_global=True)

    def drive_paddles(self, velocities: Array) -> None:
        """See :class:`Bodies`."""

        self._velocity[1 : 1 + self.paddles] = velocities
        self._rb.set_velocities(self._velocity_wp, self._index_paddles)

    def write_spare(self, index: int, position: Array, velocity: Array, spin: Array) -> None:
        """See :class:`Bodies`."""

        row = 1 + self.paddles + index
        self._transform[row, :3] = position
        self._transform[row, 3:] = (0.0, 0.0, 0.0, 1.0)
        self._velocity[row, :3] = velocity
        self._velocity[row, 3:] = spin
        self._rb.set_transforms(self._transform_wp, self._index_spare[index])
        self._rb.set_velocities(self._velocity_wp, self._index_spare[index])

    def place_paddles(self, poses: Array) -> None:
        """See :class:`Bodies`."""

        self._transform[1 : 1 + self.paddles, :3] = poses[:, :3]
        self._transform[1 : 1 + self.paddles, 3:] = poses[:, [4, 5, 6, 3]]
        self._velocity[1 : 1 + self.paddles] = 0.0
        self._rb.set_transforms(self._transform_wp, self._index_paddles)
        self._rb.set_velocities(self._velocity_wp, self._index_paddles)


class GamePhysics:
    """
    The per-step physics of one game: one ball and the paddles.

    :param params: The engine's physics parameters (what the simulator runs).
    :param settings: Canonical adapter settings.
    :param ball_path: Ball prim.
    :param paddle_paths: Paddle prim per actor name (``suited``, ``opponent``).
    :param bodies: The engine side; ``None`` creates the Isaac tensor view in :meth:`create_view`.
    """

    CALLBACK_NAME = "table_tennis_game_physics"

    def __init__(
        self,
        params: BallTableParams,
        settings: EngineSettings,
        ball_path: str,
        paddle_paths: dict[str, str],
        *,
        bodies: Bodies | None = None,
        spare_paths: tuple[str, ...] = (),
        spare_park: tuple[tuple[float, float, float], ...] = (),
        ball_integrator: str | None = None,
        room: bool = True,
    ) -> None:
        ball_integrator = ball_integrator or settings.ball_integrator
        if ball_integrator not in ("physx", "rk4"):
            raise ValueError(f"unknown ball integrator {ball_integrator!r}: physx or rk4")
        if ball_integrator == "rk4" and "engine" in (settings.table.mode, settings.net.mode, settings.floor.mode):
            raise ValueError("the rk4 ball integrator needs every ball contact hybrid (table, net and floor): PhysX never touches the game ball")
        self.ball_integrator = ball_integrator
        self.room = room
        """Sweep the room's walls, ceiling and furniture with the hybrid floor (off for calibration trials, which may leave the room)."""
        self.params = params
        self.settings = settings
        self.model = FlightModel.from_params(params)
        self.ball_path = ball_path
        self.actors = tuple(paddle_paths)
        self.paths = (ball_path, *paddle_paths.values(), *spare_paths)
        self.spares = spare_paths
        self.spare_park = [
            np.asarray(spare_park[index] if index < len(spare_park) else (-6.5, -1.0 + 2.0 * index, -0.7), dtype=np.float64)
            for index in range(len(spare_paths))
        ]
        self.spare_parked = [True] * len(spare_paths)
        self.spare_visible = [False] * len(spare_paths)
        """Whether a camera sees each retired ball (kept by the game loop): a visible one is never reused."""
        self.visible_reuses = 0
        self.placement_blade_distance: list[tuple[float, str, float]] = []
        """Each placement's distance from the nearest blade at the step it was written: ``(time, label, m)``."""
        self.tunnel_log: list[tuple[float, str, Array]] = []
        """Every tunnelling or room exit: ``(time, surface, ball position)``."""
        self.collect_request: str | None = None
        self.collect_park: tuple[float, ...] = (0.0, 0.0, 0.0)
        self.retired: list[tuple[float, int, str]] = []
        self._spare_written = [-(10**9)] * len(spare_paths)
        self._spare_order: list[int] = []
        self.dt = settings.scene.physics_dt_s
        self.time_offset_s = 0.0
        self.step_index = 0
        step = settings.scene.physics_dt_s
        self._recontact_steps = max(1, round(PADDLE_RECONTACT_COOLDOWN_S / step))
        self._park_rewrite_steps = max(1, round(PARK_REWRITE_S / step))
        self._park_written_step = -(10**9)
        self._park_written: Array | None = None
        self._driven: dict[str, PaddleSegment] = {}
        self._drive_until: dict[str, float] = {}
        self._pose_cache: dict[str, tuple[PaddleSegment, float, tuple[Array, Array, Array, Array]]] = {}
        self._mass = params.ball.mass
        self._inertia = params.ball.inertia
        self._gravity = params.aero.gravity
        self.segments: dict[str, PaddleSegment] = {}
        self.events: list[ContactEvent] = []
        self.net = Net.from_params(params)
        self.frame = FrameGeometry.from_params(params)
        self.fixtures = net_fixtures(params)
        self.fixture_geometry = FixtureGeometry.from_params(params)
        self.event_regions: dict[int, str] = {}
        """The part of the net each hybrid net event touched (``mesh`` or ``cord``), by event index."""
        self.tick_listener: Callable[[Array | None], object] | None = None
        """Called once per render tick, in the step whose end is the tick's end, with the ball state the tick renders
        (position and velocity extrapolated over that step; ``None`` while parked). The ball-shadow decal uses it."""
        self.episodes = ContactEpisodes()
        self.body_motion: dict[str, BodyMotion] = {}
        self.body_excluded: dict[str, frozenset[str]] = {}
        self.stats = StepStats()
        self.clear_flight = ball_integrator == "rk4"
        """Under ``rk4``: integrate stretches where the ball cannot reach any surface as one step (:data:`CLEAR_FLIGHT_STEPS`)."""
        self._static_boxes = (
            table_board(0.5 * params.table.length, 0.5 * params.table.width),
            ((-0.5 * params.net.thickness, -self.net.half_span, 0.0), (0.5 * params.net.thickness, self.net.half_span, params.net.height)),
            *(box for box, _members in self.fixture_geometry.sides),
            self.frame.bounds,
        )
        self.error: str | None = None
        self.parked_at: Array | None = None
        self.pending: BallPlacement | None = None
        self.placements: list[tuple[float, BallPlacement]] = []
        self.cooldown_until: dict[str, int] = {}
        self.bodies: Any = (
            IntegratedBodies(bodies, self.model) if ball_integrator == "rk4" and bodies is not None and not isinstance(bodies, IntegratedBodies) else bodies
        )
        self.apexes: dict[int, float] = {}
        self._registered = False
        self._in_callback = False
        self._paddle_velocity = np.zeros((len(self.actors), 6))
        self._previous: Array | None = None
        self._previous_dv = np.zeros(3)
        self._wrote_ball = True
        self._written = np.zeros(9)
        self._written_valid = False
        self._touch_sustained = False
        self._last_engine: dict[str, int] = {}
        self._apex_event: int | None = None
        self._apex_z = 0.0

    # ---- lifecycle -------------------------------------------------------

    def create_view(self) -> None:
        """Create the rigid-body view over the ball and paddles (before ``world.reset()``)."""

        if self.bodies is None:
            engine = TensorBodies(self.paths, paddles=len(self.actors))
            self.bodies = IntegratedBodies(engine, self.model) if self.ball_integrator == "rk4" else engine

    def bind(self) -> None:
        """Bind the engine view (after ``world.reset()``)."""

        if self.bodies is None:
            raise RuntimeError("create_view() must run before world.reset()")
        if isinstance(self.bodies, TensorBodies | IntegratedBodies):
            self.bodies.bind()

    def register(self, world: Any) -> None:
        """Start running every physics step."""

        world.add_physics_callback(self.CALLBACK_NAME, self._on_step)
        self._registered = True

    def unregister(self, world: Any) -> None:
        """Stop running."""

        if self._registered:
            world.remove_physics_callback(self.CALLBACK_NAME)
            self._registered = False

    # ---- state access ----------------------------------------------------

    def read(self) -> tuple[Array, Array, Array, Array]:
        """Positions ``(N, 3)``, orientations ``(N, 4)`` wxyz, linear and angular velocities ``(N, 3)`` of ball then paddles."""

        if isinstance(self.bodies, IntegratedBodies) and not self._in_callback:
            # Between engine steps (the game loop): the engine has stepped the paddles and retired balls since.
            self.bodies.invalidate()
        return self.bodies.read()

    def ball_state(self) -> Array:
        """The ball's current state, 9 floats."""

        positions, _orientations, linear, angular = self.read()
        return np.concatenate([positions[0], linear[0], angular[0]])

    @property
    def time_s(self) -> float:
        """Game time of the next step's start, s."""

        return self.time_offset_s + self.step_index * self.dt

    def place_ball(self, position: Any, velocity: Any, spin: Any, *, label: str, at_time: float | None = None, retire: bool = False) -> None:
        """
        Put the ball into play at the step nearest ``at_time`` (a serve, a continuity ball, a drop test); a parked ball is
        released then. With ``retire``, a ball still in play is not taken away: a retired ball carries on from its state.
        """

        self.pending = BallPlacement(np.asarray(position, float), np.asarray(velocity, float), np.asarray(spin, float), label, at_time, retire)

    def park_spare(self, index: int) -> None:
        """Send a retired ball back to its waiting place (off camera)."""

        if not self.spare_parked[index]:
            self.spare_parked[index] = True
            self._spare_written[index] = -(10**9)

    def spare_rows(self) -> list[int]:
        """The rows of the retired balls in :meth:`read`."""

        return [1 + len(self.actors) + index for index in range(len(self.spares))]

    def collect_ball(self, label: str, park: Any) -> None:
        """
        Take the (dead) ball in play out of play at the next step: it carries on as a retired ball from where it is, so
        the caller can collect that one out of view, and the ball parks at ``park`` until the next placement.
        """

        self.collect_request = label
        self.collect_park = tuple(float(v) for v in park)

    def park_ball(self, position: Any) -> None:
        """Hold the ball at a position, out of play (rewritten every :data:`PARK_REWRITE_S`), until it is placed."""

        self.parked_at = np.asarray(position, dtype=np.float64)

    def place_paddles(self, poses: dict[str, tuple[Array, Array]]) -> None:
        """Teleport paddles to poses (``(position, wxyz)``) with zero velocity, before the game starts."""

        positions, orientations, _linear, _angular = self.read()
        rows = np.zeros((len(self.actors), 7))
        for row, actor in enumerate(self.actors):
            if actor in poses:
                rows[row, :3], rows[row, 3:] = poses[actor]
            else:
                rows[row, :3], rows[row, 3:] = positions[row + 1], orientations[row + 1]
        self.bodies.place_paddles(rows)

    def take_events(self, since: int) -> list[ContactEvent]:
        """Events with index at least ``since``."""

        return [event for event in self.events if event.index >= since]

    # ---- the step ----------------------------------------------------------

    def _retire(self, position: Array, velocity: Array, spin: Array, t: float, label: str) -> None:
        """
        Hand the ball in play to a retired ball, which carries on from its state: a parked one if any, else the
        longest-retired one no camera sees (:attr:`spare_visible`), else (never expected with eight) the longest-retired.
        """

        free = [index for index, parked in enumerate(self.spare_parked) if parked]
        hidden = [index for index in self._spare_order if index < len(self.spare_visible) and not self.spare_visible[index]]
        index = free[0] if free else (hidden[0] if hidden else (self._spare_order[0] if self._spare_order else 0))
        if not free and not hidden:
            self.visible_reuses += 1
        self._engine(self.bodies.write_spare, index, np.asarray(position, float), np.asarray(velocity, float), np.asarray(spin, float))
        self.spare_parked[index] = False
        if index in self._spare_order:
            self._spare_order.remove(index)
        self._spare_order.append(index)
        self.retired.append(
            (t, index, label if free else (f"{label} (reused a retired ball out of view)" if hidden else f"{label} (reused a retired ball still on camera)"))
        )

    def _write_ball(self, position: Array, velocity: Array, spin: Array, *, contact: tuple[Array, float] | None = None) -> None:
        """
        Write the ball's state for the coming step. ``contact`` is a hybrid contact's post-impact state and its fraction of
        the step: the ``rk4`` integrator resumes from it at the contact time; PhysX takes the semi-implicit start state.
        """

        if contact is not None and isinstance(self.bodies, IntegratedBodies):
            self.bodies.impose(contact[0], contact[1])
        else:
            self._engine(self.bodies.write_ball, position, velocity, spin)
        self._wrote_ball = True
        self._written = np.concatenate([position, velocity, spin])
        self._written_valid = True

    def _record(self, **fields: Any) -> None:
        event = ContactEvent(index=len(self.events), **fields)
        self.events.append(event)
        if event.surface == "table":
            self._apex_event = event.index
            self._apex_z = float(event.after[2])
        elif event.surface in ("paddle", "floor", "net"):
            self._apex_event = None

    def _on_step(self, step_size: float) -> None:
        started = time.perf_counter()
        self._in_callback = True
        bodies = self.bodies
        if isinstance(bodies, IntegratedBodies):
            # One engine step covers whole game steps; the ball is integrated between them, and its marker body is
            # put where the engine step ends (the pose the next rendered frame shows when this is the tick's last).
            steps = max(1, round(float(step_size) / self.dt))
            bodies.invalidate()
            done = 0
            while done < steps:
                count = 1
                try:
                    stretch = self._clear_stretch(steps - done) if self.clear_flight else 0
                    if stretch:
                        self._clear_flight(stretch)
                        count = stretch
                    else:
                        self._step(self.dt)
                        bodies.hold = self.parked_at is not None
                        bodies.integrate(self.dt)
                except Exception as exc:
                    if self.error is None:
                        self.error = f"{type(exc).__name__}: {exc}"
                self.stats.callbacks += count
                self.step_index += count
                done += count
            try:
                self._engine(bodies.write_marker)
            except Exception as exc:
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                self._step(float(step_size))
            except Exception as exc:
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"
            self.stats.callbacks += 1
            self.step_index += 1
        self._in_callback = False
        elapsed = time.perf_counter() - started
        self.stats.engine_steps += 1
        self.stats.seconds += elapsed
        self.stats.max_seconds = max(self.stats.max_seconds, elapsed)

    def _engine(self, call: Any, *args: Any) -> Any:
        """One engine (tensor-view) call, timed: at 3840 Hz their fixed cost dominates the step."""

        started = time.perf_counter()
        try:
            return call(*args)
        finally:
            self.stats.engine_io_seconds += time.perf_counter() - started
            self.stats.engine_calls += 1

    def _clear_stretch(self, remaining: int) -> int:
        """
        The longest :data:`CLEAR_FLIGHT_STEPS` stretch (at most ``remaining`` game steps) over which no step would do
        anything but integrate the ball: nothing to place, collect, drive or stop, no apex to catch, and every surface
        (table, net and fixtures, undercarriage, floor, room, bodies, blades) out of the ball's reach. 0 when none.
        """

        t = self.time_s
        dt = self.dt
        if self.pending is not None or self.collect_request is not None:
            return 0
        segments = self.segments
        if any(self._driven.get(actor) is not segment for actor, segment in segments.items()):
            return 0
        parked = self.parked_at is not None
        if parked and self._park_written is not self.parked_at:
            return 0
        bodies = self.bodies
        radius = self.params.ball.radius
        for count in CLEAR_FLIGHT_STEPS:
            if count > remaining or count < 2:
                continue
            horizon = count * dt
            if any(until - 0.5 * dt <= t + horizon - dt for until in self._drive_until.values()):
                continue
            if parked:
                return count
            p = bodies.position
            v = bodies.velocity
            speed = math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)
            if self._apex_event is not None and 0.0 < float(v[2]) <= 2.0 * self.params.aero.gravity * horizon:
                continue
            d = self.model.derivative([0.0, 0.0, 0.0, float(v[0]), float(v[1]), float(v[2]), *map(float, bodies.spin)])
            accel = math.sqrt(d[3] * d[3] + d[4] * d[4] + d[5] * d[5])
            reach = speed * horizon + accel * horizon * horizon + CLEAR_FLIGHT_MARGIN_M
            if not self._clear_of_surfaces(p, radius, reach, t, horizon):
                continue
            return count
        return 0

    def _clear_of_surfaces(self, p: Array, radius: float, reach: float, t: float, horizon: float) -> bool:
        """Whether a ball at ``p`` that moves at most ``reach`` cannot touch any surface (or blade) within ``horizon``."""

        need = radius + reach
        for box in self._static_boxes:
            if box_distance(p, box)[0] <= need:
                return False
        if float(p[2]) - radius + self.params.table.height <= reach:
            return False
        if self.settings.floor.mode == "hybrid" and self.room and room_gap(p, radius)[0] <= reach:
            return False
        for motion in self.body_motion.values():
            lx, ly, lz, hx, hy, hz = motion.bounds
            if box_distance(p, ((lx, ly, lz), (hx, hy, hz)))[0] <= need:
                return False
        paddle = self.params.paddle
        blade = math.hypot(paddle.blade_half_width, paddle.blade_half_length) + 0.5 * paddle.thickness
        for segment in self.segments.values():
            c0, _q0, v0, _w0 = segment.at(t)
            c1, _q1, v1, _w1 = segment.at(t + horizon)
            travel = float(np.linalg.norm(c1 - c0)) + 0.5 * (float(np.linalg.norm(v0)) + float(np.linalg.norm(v1))) * horizon
            if float(np.linalg.norm(p - c0)) - travel - blade <= need:
                return False
        return True

    def _clear_flight(self, count: int) -> None:
        """
        A clear-flight stretch of ``count`` game steps: one RK4 step of the ball (none while parked), the retired balls'
        parking rewrites, the invariant checks over the stretch, the apex and the render-tick listener.
        """

        dt = self.dt
        horizon = count * dt
        t = self.time_s
        bodies = self.bodies
        for index, parked in enumerate(self.spare_parked):
            if parked and self.step_index + count - 1 - self._spare_written[index] >= self._park_rewrite_steps:
                self._engine(self.bodies.write_spare, index, self.spare_park[index], np.zeros(3), np.zeros(3))
                self._spare_written[index] = self.step_index
        live = self.parked_at is None
        bodies.hold = not live
        before = bodies.ball()
        bodies.integrate(horizon)
        after = bodies.ball()
        self.stats.clear_flight_steps += count
        if live:
            params = self.params
            if crossed_table(before, after, params):
                self.stats.tunnel_table += 1
                self.tunnel_log.append((t, "table", after[:3].copy()))
            if crossed_net(before, after, params):
                self.stats.tunnel_net += 1
                self.tunnel_log.append((t, "net", after[:3].copy()))
            if crossed_floor(after, params):
                self.stats.tunnel_floor += 1
                self.tunnel_log.append((t, "floor", after[:3].copy()))
            if left_room(before, after):
                self.stats.left_room += 1
                self.tunnel_log.append((t, "room", after[:3].copy()))
            if self._apex_event is not None:
                if after[5] < 0.0 and after[2] < self._apex_z:
                    self.apexes[self._apex_event] = self._apex_z
                    self._apex_event = None
                else:
                    self._apex_z = max(self._apex_z, float(after[2]))
        # The next step starts from the integrated state, as after a write: nothing to classify across the stretch.
        self._wrote_ball = True
        self._written = after.copy()
        self._written_valid = True
        self._previous = after.copy()
        self._previous_dv = np.zeros(3)
        segments = self.segments
        if self.tick_listener is not None and segments and abs(t + horizon - min(segment.t1 for segment in segments.values())) < 0.5 * dt:
            self.tick_listener(after[:6].copy() if live else None)

    def _segment_pose(self, actor: str, segment: PaddleSegment, t: float) -> tuple[Array, Array, Array, Array]:
        """A segment's pose at ``t``, reusing the previous step's end pose (the same instant) when it is the same segment."""

        cached = self._pose_cache.get(actor)
        if cached is not None and cached[0] is segment and abs(cached[1] - t) < 1e-12:
            return cached[2]
        return segment.at(t)

    def _wrench(self, velocity: Array, spin: Array) -> tuple[Array, Array]:
        """:meth:`FlightModel.wrench` through the model's scalar fast path (the same expressions, without array overhead)."""

        d = self.model.derivative([0.0, 0.0, 0.0, float(velocity[0]), float(velocity[1]), float(velocity[2]), float(spin[0]), float(spin[1]), float(spin[2])])
        mass = self._mass
        force = np.array([mass * d[3], mass * d[4], mass * (d[5] + self._gravity)])
        torque = self._inertia * np.array([d[6], d[7], d[8]])
        return force, torque

    def _step(self, dt: float) -> None:
        params = self.params
        t = self.time_s
        segments = self.segments
        drive = [actor for actor, segment in segments.items() if self._driven.get(actor) is not segment]
        stop = [actor for actor in self._drive_until if actor not in drive and t >= self._drive_until[actor] - 0.5 * dt]
        parked = self.parked_at is not None
        state: Array | None = None
        positions = orientations = linear = angular = None
        if not parked or drive:
            positions, orientations, linear, angular = self._engine(self.read)
            state = np.concatenate([positions[0], linear[0], angular[0]])
        wrote_previous = self._wrote_ball
        self._wrote_ball = False

        # 1. The engine contact of the step just taken, and tunnelling. After a step that wrote the ball (a hybrid
        #    contact, a sustained roll, a placement) the engine integrated from the written state, not from the
        #    previous read: classify against that, so a native contact in such a step (a rolling ball into a net
        #    clamp's jaw) is still recorded.
        previous = self._written.copy() if wrote_previous and self._written_valid else (None if wrote_previous else self._previous)
        if previous is not None and state is not None and not parked:
            native = classify_native(previous, state, dt, params, applied_dv=self._previous_dv)
            if native is not None:
                self._note_engine_contact(native, previous, state, t - dt)
            if crossed_table(previous, state, params):
                self.stats.tunnel_table += 1
                self.tunnel_log.append((t, "table", state[:3].copy()))
            if crossed_net(previous, state, params) and (native is None or native.surface != "net"):
                self.stats.tunnel_net += 1
                self.tunnel_log.append((t, "net", state[:3].copy()))
            if crossed_floor(state, params):
                self.stats.tunnel_floor += 1
                self.tunnel_log.append((t, "floor", state[:3].copy()))
            if left_room(previous, state):
                self.stats.left_room += 1
                self.tunnel_log.append((t, "room", state[:3].copy()))
            if crossed_frame(previous, state, self.frame, params.ball.radius):
                self.stats.tunnel_frame += 1
                self.tunnel_log.append((t, "frame", state[:3].copy()))

        # 2. Placements and parking (a parked ball is rewritten every PARK_REWRITE_S, not every step). A dead ball being
        #    collected becomes a retired ball where it is (so it can leave view without vanishing) and the ball parks.
        if (
            self.collect_request is not None
            and self.pending is None
            and self.parked_at is None
            and positions is not None
            and linear is not None
            and angular is not None
        ):
            if self.spares:
                self._retire(positions[0], linear[0], angular[0], t, self.collect_request)
            self.collect_request = None
            self.parked_at = np.asarray(self.collect_park, dtype=np.float64)
            self._park_written = None
            state = np.concatenate([self.parked_at, np.zeros(6)])
            parked = True
        if self.pending is not None and (self.pending.at_time is None or t >= self.pending.at_time - 0.5 * dt):
            placement = self.pending
            self.pending = None
            if placement.retire and self.spares and self.parked_at is None and positions is not None and linear is not None and angular is not None:
                self._retire(positions[0], linear[0], angular[0], t, placement.label)
            self.parked_at = None
            self._apex_event = None
            self.episodes.close()
            self._write_ball(placement.position, placement.velocity, placement.spin)
            self.placements.append((t, placement))
            if positions is not None and len(positions) > 1 and not placement.label.startswith("drop"):
                blades = positions[1 : 1 + len(self.actors)]
                self.placement_blade_distance.append((t, placement.label, float(np.min(np.linalg.norm(blades - np.asarray(placement.position)[None], axis=1)))))
            state = np.concatenate([placement.position, placement.velocity, placement.spin])
        elif self.parked_at is not None:
            if self._park_written is not self.parked_at or self.step_index - self._park_written_step >= self._park_rewrite_steps:
                self._write_ball(self.parked_at, np.zeros(3), np.zeros(3))
                self._park_written = self.parked_at
                self._park_written_step = self.step_index
            state = np.concatenate([self.parked_at, np.zeros(6)])
        assert state is not None

        for index, parked in enumerate(self.spare_parked):
            if parked and self.step_index - self._spare_written[index] >= self._park_rewrite_steps:
                self._engine(self.bodies.write_spare, index, self.spare_park[index], np.zeros(3), np.zeros(3))
                self._spare_written[index] = self.step_index

        # 3. Hybrid contacts over the coming step. The blades are swept along their analytic segments (the bodies are
        #    driven once per tick, so only their tick-end pose, the rendered one, is the segment's). A ball written this
        #    step (a placement) is swept from the written state: skipping it could miss a touch.
        live = self.parked_at is None
        ends = {actor: segment.at(t + dt) for actor, segment in segments.items()}
        if live:
            starts = {actor: self._segment_pose(actor, segment, t) for actor, segment in segments.items()}
            imposed = self._sweep_paddles(state, starts, ends, t, dt)
            if imposed is None:
                # The static and body surfaces in turn. A bounce ends the step's contacts; a sustained touch (a ball
                # resting or rolling on the table, the floor or the cord) does not: the rest are swept from the
                # constrained state, so a ball rolling on the floor still meets a wall or a leg, and one rolling on the
                # table still meets the net.
                sweeps = []
                if self.settings.table.mode == "hybrid":
                    sweeps.append(self._sweep_table)
                if self.settings.net.mode == "hybrid":
                    sweeps.append(self._sweep_net)
                sweeps.append(self._sweep_frame)
                if self.settings.floor.mode == "hybrid":
                    sweeps.append(self._sweep_floor)
                    if self.room:
                        sweeps.append(self._sweep_room)
                if self.body_motion:
                    sweeps.append(self._sweep_bodies)
                current = state
                for sweep in sweeps:
                    self._touch_sustained = False
                    result = sweep(current, t, dt)
                    if result is None:
                        continue
                    imposed = result
                    if not self._touch_sustained:
                        break
                    current = result
            if imposed is not None:
                state = imposed
        self._pose_cache = {actor: (segments[actor], t + dt, pose) for actor, pose in ends.items()}

        # 4. Air forces from the reference model.
        if live:
            aero_started = time.perf_counter()
            force, torque = self._wrench(state[3:6], state[6:9])
            self.stats.aero_seconds += time.perf_counter() - aero_started
            self._engine(self.bodies.push_ball, force, torque, state[:3])
            self._previous_dv = force / self._mass * dt
        else:
            self._previous_dv = np.zeros(3)

        # 5. Each paddle body is driven once per tick: a constant velocity that lands it on its segment's end (the
        #    pose the tick renders), and stopped when a segment ends without a successor.
        if drive or stop:
            limit = self.settings.paddle.max_drive_speed_mps
            for actor in stop:
                self._paddle_velocity[self.actors.index(actor)] = 0.0
                del self._drive_until[actor]
            for actor in drive:
                assert positions is not None and orientations is not None
                row = self.actors.index(actor) + 1
                segment = segments[actor]
                remaining = segment.t1 - t
                current = orientations[row] / np.linalg.norm(orientations[row])
                if remaining <= 0.5 * dt:
                    self._paddle_velocity[row - 1] = 0.0
                else:
                    command = (segment.p1 - positions[row]) / remaining
                    speed = float(np.linalg.norm(command))
                    if speed > limit:
                        command *= limit / speed
                    self._paddle_velocity[row - 1, :3] = command
                    self._paddle_velocity[row - 1, 3:] = _quat_rotvec(_quat_multiply(segment.q1, current * np.array([1.0, -1.0, -1.0, -1.0]))) / remaining
                    self._drive_until[actor] = segment.t1
                self.stats.paddle_error_m.append(float(np.linalg.norm(positions[row] - segment.p0)))
                dot = abs(float(current @ segment.q0))
                self.stats.paddle_error_deg.append(math.degrees(2.0 * math.acos(min(1.0, dot))))
                self._driven[actor] = segment
            self._engine(self.bodies.drive_paddles, self._paddle_velocity)
        if self.tick_listener is not None and segments and abs(t + dt - min(segment.t1 for segment in segments.values())) < 0.5 * dt:
            self.tick_listener(np.concatenate([state[:3] + state[3:6] * dt, state[3:6]]) if live else None)
        self._previous = state.copy()
        if self._apex_event is not None:
            if state[5] < 0.0 and state[2] < self._apex_z:
                self.apexes[self._apex_event] = self._apex_z
                self._apex_event = None
            else:
                self._apex_z = max(self._apex_z, float(state[2]))

    def _sweep_paddles(
        self,
        state: Array,
        starts: dict[str, tuple[Array, Array, Array, Array]],
        ends: dict[str, tuple[Array, Array, Array, Array]],
        t: float,
        dt: float,
    ) -> Array | None:
        params = self.params
        acceleration: Array | None = None
        for actor in self.actors:
            target = ends.get(actor)
            if target is None or self.cooldown_until.get(actor, -1) > self.step_index:
                continue
            begin = starts[actor]
            if acceleration is None:
                force, _torque = self.model.wrench(state[3:6], state[6:9])
                acceleration = np.array([0.0, 0.0, -params.aero.gravity]) + np.asarray(force, dtype=np.float64) / params.ball.mass
            contact = self._blade_touch(actor, state, acceleration, t, dt)
            tunnelled = False
            if contact is None:
                # No touch: the swept test still flags a ball whose centre crossed the blade's mid-plane inside it.
                _swept, tunnelled = self._sweep_blade_across_knot(actor, state, begin, target, t, dt)
            if tunnelled:
                self.stats.tunnel_paddle += 1
                self.tunnel_log.append((t, f"paddle:{actor}", state[:3].copy()))
            if contact is None:
                continue
            imposed = impose_contact(
                contact,
                params.rim_contact if contact.edge else params.rubber_contact,
                params,
                dt,
                effective_mass=self.settings.paddle.effective_mass_kg,
                rigid_body=self.settings.paddle.rigid_body_impact,
                flight=self.model,
            )
            if imposed is None:
                continue
            self._write_ball(imposed.start_position, imposed.velocity, imposed.spin, contact=(imposed.after, contact.fraction))
            self.cooldown_until[actor] = self.step_index + self._recontact_steps
            relative_before = float((imposed.before[3:6] - contact.surface_velocity) @ contact.normal)
            relative_after = float((imposed.after[3:6] - contact.surface_velocity) @ contact.normal)
            if contact.edge:
                self.event_regions[len(self.events)] = "blade_edge"
                self.stats.contacts_edge += 1
            self._record(
                time_s=t + contact.fraction * dt,
                step=self.step_index,
                surface="paddle",
                mode="hybrid",
                actor=actor,
                position=contact.ball[:3].copy(),
                normal=contact.normal.copy(),
                before=imposed.before.copy(),
                after=imposed.after.copy(),
                approach_speed_mps=max(-relative_before, 0.0),
                normal_restitution=-relative_after / relative_before if relative_before < -1e-6 else math.nan,
                energy_before_j=kinetic_energy(imposed.before, params.ball.mass, params.ball.inertia),
                energy_after_j=kinetic_energy(imposed.after, params.ball.mass, params.ball.inertia),
                regime=imposed.regime,
                blade_offset_m=contact.blade_offset_m,
                surface_velocity=contact.surface_velocity.copy(),
            )
            self.stats.contacts_hybrid += 1
            return np.concatenate([imposed.start_position, imposed.velocity, imposed.spin])
        return None

    def _blade_touch(self, actor: str, state: Array, acceleration: Array, t: float, dt: float) -> SweptContact | None:
        """
        The ball's first touch on one blade inside the step, resolved with the oracle's own blade state
        (:func:`lib.physics.reference.blade.blade_touch`, physics-ref): the blade exact on its segment (knot included)
        at the bisected touch time, the ball on its constant-acceleration path from the step's start, the face or the
        rim of the outline, or (``settings.paddle.blade_edge``, the default) the blade's rounded edge, as the oracle's
        ``simulate`` has it since physics Addendum 13: an edge touch is resolved with ``rim_contact`` and recorded with
        region ``blade_edge``. A linear sweep between the step's two blade poses put the material point's velocity at the
        touch off by up to 3.3 mm/s (the Hermite velocity is quadratic in time), and the outgoing ball by 2.4 mm/s
        (datagen-core's ``ce654ebd9c366945``); with the shared state it agrees with the oracle to about 1e-7 m/s.

        :return: The contact (its fraction of the step), or ``None`` without a touch or when the ball is not approaching.
        """

        segment = self.segments.get(actor)
        if segment is None:
            return None
        paddle = self.params.paddle
        found = blade_touch(
            segment,
            t,
            dt,
            state,
            radius=self.params.ball.radius,
            half_thickness=0.5 * paddle.thickness,
            half_length=paddle.blade_half_length,
            half_width=paddle.blade_half_width,
            acceleration=acceleration,
            edge=self.settings.paddle.blade_edge,
        )
        if found is None:
            return None
        touch_time, centre, contact = found
        lead = touch_time - t
        if float((state[3:6] + acceleration * lead - contact.surface_velocity) @ contact.normal) >= 0.0:
            return None
        return SweptContact(
            fraction=min(max(lead / dt, 0.0), 1.0),
            ball=np.concatenate([centre, state[3:9]]),
            normal=np.asarray(contact.normal, dtype=np.float64),
            surface_velocity=np.asarray(contact.surface_velocity, dtype=np.float64),
            blade_offset_m=(float(contact.blade_offset[0]), float(contact.blade_offset[1])),
            contact_offset=np.asarray(contact.offset, dtype=np.float64),
            surface_rotation=np.asarray(contact.axes, dtype=np.float64),
            edge=contact.touch == "edge",
        )

    def _sweep_blade_across_knot(
        self, actor: str, state: Array, begin: tuple[Array, Array, Array, Array], end: tuple[Array, Array, Array, Array], t: float, dt: float
    ) -> tuple[SweptContact | None, bool]:
        """
        Sweep one blade over the step ``[t, t + dt]``. A step that holds the paddle segment's contact knot is swept as two
        pieces, ``[t, t_knot]`` and ``[t_knot, t + dt]``, with the blade at the knot between them: the two sides of a knot
        turn at different angular velocities, and one linear sweep across it put the blade's normal and surface velocity
        at contact off by up to 2e-5 rad and 1e-3 m/s (physics-ref's oracle replay of game-2, game-5 and smoke: outgoing
        velocities 1.6e-3 to 2.5e-2 m/s from the oracle's whenever the contact step held a knot, 1e-7 to 2e-4 otherwise).
        The returned contact's fraction is of the whole step.
        """

        paddle = self.params.paddle
        geometry = {
            "radius": self.params.ball.radius,
            "half_thickness": 0.5 * paddle.thickness,
            "half_width": paddle.blade_half_width,
            "half_length": paddle.blade_half_length,
        }

        def blade(pose: tuple[Array, Array, Array, Array]) -> BladePose:
            return BladePose(position=pose[0], rotation=quat_matrix(pose[1]), velocity=pose[2], angular_velocity=pose[3])

        segment = self.segments.get(actor)
        knot = segment.knot if segment is not None else None
        split = (
            knot is not None and segment is not None and segment.t0 < knot[0] < segment.t1 and t + KNOT_SPLIT_MARGIN_S < knot[0] < t + dt - KNOT_SPLIT_MARGIN_S
        )
        if not split or segment is None or knot is None:
            return sweep_blade(state, blade(begin), blade(end), dt, **geometry)
        first = knot[0] - t
        middle = segment.at(knot[0])
        contact, tunnelled = sweep_blade(state, blade(begin), blade(middle), first, **geometry)
        if contact is not None:
            return dataclasses.replace(contact, fraction=contact.fraction * first / dt), tunnelled
        moved = np.concatenate([state[:3] + state[3:6] * first, state[3:]])
        second = dt - first
        contact, crossed = sweep_blade(moved, blade(middle), blade(end), second, **geometry)
        if contact is not None:
            contact = dataclasses.replace(contact, fraction=(first + contact.fraction * second) / dt)
        return contact, tunnelled or crossed

    def _record_touch(self, key: str, imposed: ImposedState, contact: SweptContact, t: float, dt: float, **record: Any) -> None:
        """Record a hybrid static or body touch, folding a sustained contact's later steps into its first event."""

        params = self.params
        time_s = t + contact.fraction * dt
        extended = self.episodes.touch(key, self.step_index, imposed.sustained, len(self.events))
        if extended is not None:
            self.events[extended] = extend_event(self.events[extended], imposed.after, time_s)
            return
        normal = contact.normal
        before_n = float((imposed.before[3:6] - contact.surface_velocity) @ normal)
        after_n = float((imposed.after[3:6] - contact.surface_velocity) @ normal)
        self._record(
            time_s=time_s,
            step=self.step_index,
            mode="hybrid",
            position=imposed.after[:3].copy(),
            normal=normal.copy(),
            before=imposed.before.copy(),
            after=imposed.after.copy(),
            approach_speed_mps=max(-before_n, 0.0),
            normal_restitution=-after_n / before_n if before_n < -1e-6 else math.nan,
            energy_before_j=kinetic_energy(imposed.before, params.ball.mass, params.ball.inertia),
            energy_after_j=kinetic_energy(imposed.after, params.ball.mass, params.ball.inertia),
            regime=imposed.regime,
            surface_velocity=contact.surface_velocity.copy(),
            **record,
        )
        self.stats.contacts_hybrid += 1

    def _impose_static(self, key: str, contact: SweptContact, material: Any, t: float, dt: float, *, rolling: bool = False) -> Array | None:
        imposed = impose_contact(contact, material, self.params, dt, effective_mass=math.inf, flight=self.model, rest_speed=REST_SPEED_MPS, rolling=rolling)
        if imposed is None:
            return None
        self._touch_sustained = imposed.sustained
        self._write_ball(imposed.start_position, imposed.velocity, imposed.spin, contact=(imposed.after, contact.fraction))
        self._record_touch(key, imposed, contact, t, dt, surface=key, actor=None)
        return np.concatenate([imposed.start_position, imposed.velocity, imposed.spin])

    def _sweep_table(self, state: Array, t: float, dt: float) -> Array | None:
        """The hybrid table: the playing surface as before, plus the board's edges, side and end faces and underside."""

        params = self.params
        contact = sweep_table_box(state, dt, radius=params.ball.radius, half_length=0.5 * params.table.length, half_width=0.5 * params.table.width)
        if contact is None:
            return None
        # Rolling on the playing surface takes its rolling resistance and can stop; the board's edges never hold a ball.
        return self._impose_static("table", contact, params.table_contact, t, dt, rolling=True)

    def _sweep_net(self, state: Array, t: float, dt: float) -> Array | None:
        """
        The hybrid net assembly: the calibration harness's sweep of the ball against the net box and its posts and clamps,
        the earliest touch winning, resolved by the reference impact law (the net's region law; ``post_contact`` on the
        steel posts, ``fixture_contact`` on the clamps). The posts' and clamps' native colliders are filtered for the ball.
        """

        params = self.params
        radius = params.ball.radius
        fixture, fixture_index, fixture_tunnelled = sweep_fixtures(state, dt, radius=radius, fixtures=self.fixture_geometry)
        if fixture_tunnelled:
            self.stats.tunnel_net += 1
            self.tunnel_log.append((t, "net", state[:3].copy()))
        contact = None
        if abs(float(state[0])) <= 0.5 * params.net.thickness + radius + abs(float(state[3])) * dt + 1e-3:
            contact, tunnelled = sweep_net(
                state, dt, radius=radius, half_thickness=0.5 * params.net.thickness, half_span=params.net_half_span, height=params.net.height
            )
            if tunnelled:
                self.stats.tunnel_net += 1
                self.tunnel_log.append((t, "net", state[:3].copy()))
        if fixture is not None and fixture_index is not None and (contact is None or fixture.fraction < contact.fraction):
            count = len(self.events)
            imposed = self._impose_static("net", fixture, self.fixture_geometry.materials[fixture_index], t, dt)
            if len(self.events) > count:
                self.event_regions[count] = self.fixture_geometry.regions[fixture_index]
            return imposed
        if contact is None:
            return None
        # The mesh face below the top tape catches the ball; the tape, top edge and ends stay rigid (with the
        # sustained regime for a ball rolling on the cord). The reference model decides which part was touched.
        centre = contact.ball[:3] + contact.normal * contact.depth
        net = self.net
        point = np.clip(centre, [-net.half_thickness, -net.half_span, 0.0], [net.half_thickness, net.half_span, net.height])
        region = net.region(point)
        count = len(self.events)
        imposed = self._impose_static("net", contact, net.material_at(point), t, dt)
        if len(self.events) > count:
            self.event_regions[count] = region
        return imposed

    def _sweep_floor(self, state: Array, t: float, dt: float) -> Array | None:
        """The floor plane (hybrid): the reference impact law with the contract's ``floor_contact``, sustained while rolling."""

        params = self.params
        contact = sweep_floor(state, dt, radius=params.ball.radius, floor_z=-params.table.height)
        if contact is None:
            return None
        return self._impose_static("floor", contact, params.floor_contact, t, dt, rolling=True)

    def _sweep_room(self, state: Array, t: float, dt: float) -> Array | None:
        """The room's walls, ceiling and furniture (hybrid): the reference impact law with the wall material."""

        contact, name, tunnelled = sweep_room(state, dt, radius=self.params.ball.radius)
        if tunnelled:
            self.stats.left_room += 1
            self.tunnel_log.append((t, "room", state[:3].copy()))
        if contact is None:
            return None
        count = len(self.events)
        imposed = self._impose_static("wall", contact, self.params.wall_contact, t, dt)
        if len(self.events) > count and name is not None:
            self.event_regions[count] = name
        return imposed

    def _sweep_frame(self, state: Array, t: float, dt: float) -> Array | None:
        """
        The table's undercarriage (apron, legs, feet, casters, stretchers, spine, struts): the reference model's 38 parts,
        hybrid, with the contract's ``fixture_contact`` (painted steel); one bounds check while the ball is above the table.
        """

        params = self.params
        contact, region, _tunnelled = sweep_frame(state, dt, radius=params.ball.radius, frame=self.frame)
        if contact is None:
            return None
        count = len(self.events)
        imposed = self._impose_static("frame", contact, params.fixture_contact, t, dt)
        if len(self.events) > count and region is not None:
            self.event_regions[count] = region
            self.stats.contacts_frame += 1
        return imposed

    def _sweep_bodies(self, state: Array, t: float, dt: float) -> Array | None:
        """The players' body capsules (``body_motion``, set per tick by the game loop): a soft, heavy, moving surface."""

        params = self.params
        for role, motion in self.body_motion.items():
            touch, embedded = sweep_body(state, motion, t, dt, radius=params.ball.radius, excluded=self.body_excluded.get(role, frozenset()))
            self.stats.body_embedded_steps += int(embedded > 0)
            if touch is None:
                continue
            imposed = impose_contact(touch.contact, BODY_CONTACT, params, dt, effective_mass=math.inf, flight=self.model, rest_speed=REST_SPEED_MPS)
            if imposed is None:
                continue
            self._touch_sustained = imposed.sustained
            self._write_ball(imposed.start_position, imposed.velocity, imposed.spin, contact=(imposed.after, touch.contact.fraction))
            self._record_touch(f"body:{role}", imposed, touch.contact, t, dt, surface="body", actor=role)
            self.stats.contacts_body += 1
            return np.concatenate([imposed.start_position, imposed.velocity, imposed.spin])
        return None

    def _note_engine_contact(self, native: Any, before: Array, after: Array, t: float) -> None:
        params = self.params
        last_step = self._last_engine.get(native.surface)
        if last_step is not None and (self.step_index - last_step) * self.settings.scene.physics_dt_s <= NATIVE_MERGE_S and self.events:
            index = next((i for i in range(len(self.events) - 1, -1, -1) if self.events[i].surface == native.surface and self.events[i].mode == "engine"), None)
            if index is not None:
                event = self.events[index]
                vn0 = float(event.before[3:6] @ event.normal)
                vn1 = float(after[3:6] @ event.normal)
                self.events[index] = dataclasses.replace(
                    event,
                    after=after.copy(),
                    normal_restitution=-vn1 / vn0 if vn0 < -1e-6 else math.nan,
                    energy_after_j=kinetic_energy(after, params.ball.mass, params.ball.inertia),
                )
                self._last_engine[native.surface] = self.step_index
                return
        # A native contact at a net post or clamp belongs to the net assembly (as in the reference model): surface
        # ``net``, region ``post`` or ``clamp``; on the net itself, ``mesh`` or ``cord``.
        surface = native.surface
        region = None
        if surface in ("net", "other"):
            position = 0.5 * (before[:3] + after[:3])
            point = [float(value) for value in position]
            gap, fixture_region = min(((fixture.gap(t, point), fixture.region) for fixture in self.fixtures), key=lambda item: item[0])
            if gap < NATIVE_FIXTURE_GAP_M:
                surface, region = "net", fixture_region
            elif surface == "net":
                net = self.net
                region = net.region(np.clip(position, [-net.half_thickness, -net.half_span, 0.0], [net.half_thickness, net.half_span, net.height]))
        self._last_engine[native.surface] = self.step_index
        if region is not None:
            self.event_regions[len(self.events)] = region
        self._record(
            time_s=t,
            step=self.step_index,
            surface=surface,
            mode="engine",
            actor=None,
            position=0.5 * (before[:3] + after[:3]),
            normal=native.normal,
            before=before.copy(),
            after=after.copy(),
            approach_speed_mps=native.approach_speed_mps,
            normal_restitution=native.normal_restitution,
            energy_before_j=kinetic_energy(before, params.ball.mass, params.ball.inertia),
            energy_after_j=kinetic_energy(after, params.ball.mass, params.ball.inertia),
        )
        self.stats.contacts_engine += 1


__all__ = ["BallPlacement", "Bodies", "GamePhysics", "PaddleSegment", "StepStats", "TensorBodies", "quat_matrix"]
