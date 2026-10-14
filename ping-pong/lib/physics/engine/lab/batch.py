"""
What one Isaac Lab batch simulates, environment by environment, and what it returns.

Pure: numpy, pydantic and :mod:`lib.physics` only. A :class:`BatchSpec` is the
whole input of one engine build: the global engine configuration (step,
solver, device), the named parameter sets the environments draw from, and one
:class:`EnvSpec` per environment (initial ball state, paddle script, contact
modes, stop condition). :class:`BatchRecord` is everything the harness measured,
in numpy, in each environment's local frame (the table frame of
:mod:`lib.physics`: Z up, origin at the centre of the playing surface).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lib.physics.params import BallTableParams
from lib.physics.reference.surfaces import ScriptedPaddle, blade_rotation, constant_twist_paddle

_FROZEN = ConfigDict(frozen=True, extra="forbid")

Vec3 = tuple[float, float, float]

PaddleMode = Literal["none", "heavy", "pd", "kinematic"]
"""How the environment's paddle body is driven.

- ``none``: no paddle; the body is parked far below the table.
- ``heavy``: a 1000 kg dynamic body whose pose and velocity are rewritten every step (the default drive).
- ``pd``: a dynamic body of the paddle's effective mass, pulled to the script by a stiff PD wrench.
- ``kinematic``: a kinematic body written with pose and velocity; PhysX GPU ignores its velocity (a negative control).
"""

PaddleContact = Literal["hybrid", "native"]
"""``hybrid``: the paddle has no ball collider and the reference impact model resolves the contact the adapter
sweeps. ``native``: PhysX resolves it with the rubber material (no tangential restitution, no spin retention)."""

TableContact = Literal["native", "hybrid"]
"""``native``: PhysX resolves the bounce with a constant restitution. ``hybrid``: the table has no ball collider
and the reference impact model resolves the swept bounce, speed curves included."""

NetContact = Literal["native", "hybrid"]
"""``native``: PhysX resolves net contacts (its speculative contact fixes the normal at the step start, which
mis-deflects balls clipping the top edge at speed). ``hybrid``: the net has no ball collider and the reference
impact model resolves the swept touch, with the normal from the box's closest point."""

SURFACE_CODES: dict[str, int] = {"none": 0, "table": 1, "paddle": 2, "net": 3, "other": 4, "frame": 5}
"""Contact-kind codes in :attr:`BatchRecord.contact_kind`."""

SURFACE_NAMES: dict[int, str] = {code: name for name, code in SURFACE_CODES.items()}

PARK_POSITION: Vec3 = (0.0, 0.0, -3.0)
"""Where an unused paddle waits, environment frame, m: under the floor, out of every flight."""


class PaddleScript(BaseModel):
    """
    A paddle moving with constant linear and angular velocity from ``t = 0`` (the reference's constant-twist paddle).

    :ivar position: Blade centre at ``t = 0``, m, table frame.
    :ivar face_normal: Forehand face normal at ``t = 0`` (any length).
    :ivar handle_direction: Rough direction of the blade's local +X (towards the handle).
    :ivar velocity: Linear velocity, m/s.
    :ivar angular_velocity: Angular velocity, rad/s, table frame.
    """

    model_config = _FROZEN

    position: Vec3
    face_normal: Vec3
    handle_direction: Vec3 = (0.0, 0.0, -1.0)
    velocity: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)

    def rotation(self) -> NDArray[np.float64]:
        """Blade rotation at ``t = 0``: columns X (towards the handle), Y, Z (forehand normal)."""

        return blade_rotation(self.face_normal, self.handle_direction)

    def scripted(self, name: str = "paddle") -> ScriptedPaddle:
        """The same motion as a reference-model :class:`ScriptedPaddle`."""

        return constant_twist_paddle(self.position, self.rotation(), self.velocity, self.angular_velocity, name=name)


class EnvSpec(BaseModel):
    """
    One environment: one ball, at most one paddle, and when it is finished.

    :ivar label: Unique name within the batch.
    :ivar params: Key of the engine parameter set in :attr:`BatchSpec.param_sets`.
    :ivar initial_state: Ball position (m), velocity (m/s) and spin (rad/s) at ``t = 0``, table frame.
    :ivar paddle: The paddle's script; ``None`` with ``paddle_mode="none"``.
    :ivar paddle_mode: How the paddle body is driven.
    :ivar paddle_contact: How a paddle contact is resolved.
    :ivar table_contact: How a table bounce is resolved.
    :ivar net_contact: How a net contact is resolved.
    :ivar aero: Apply the reference model's aerodynamic wrench every step.
    :ivar max_angular_velocity_deg_s: The ball's PhysX angular-velocity cap (the engine default is 5729.58).
    :ivar t_end_s: Simulated time after which the environment is finished.
    :ivar stop_after_contacts: Finish ``hold_s`` after this many contacts; 0 runs to ``t_end_s``.
    :ivar hold_s: Simulated time kept after the stopping contact, s.
    """

    model_config = _FROZEN

    label: str = Field(min_length=1)
    params: str = Field(min_length=1)
    initial_state: tuple[float, float, float, float, float, float, float, float, float]
    paddle: PaddleScript | None = None
    paddle_mode: PaddleMode = "none"
    paddle_contact: PaddleContact = "hybrid"
    table_contact: TableContact = "native"
    net_contact: NetContact = "native"
    aero: bool = True
    max_angular_velocity_deg_s: float = Field(default=100000.0, gt=0.0)
    t_end_s: float = Field(gt=0.0)
    stop_after_contacts: int = Field(default=0, ge=0)
    hold_s: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _paddle_matches_mode(self) -> Self:
        if (self.paddle is None) != (self.paddle_mode == "none"):
            raise ValueError(f"environment {self.label!r}: a paddle script needs a paddle mode and the reverse (mode {self.paddle_mode!r})")
        if not all(math.isfinite(x) for x in self.initial_state):
            raise ValueError(f"environment {self.label!r}: initial state is not finite")
        return self


PaddleMassModel = Literal["human_intrinsic", "rigid_apparent"]
"""How the hybrid paddle impact treats the cited rubber restitution (see :mod:`.contact_model`)."""


class EngineConfig(BaseModel):
    """
    The global engine configuration of one build.

    :ivar physics_dt_s: Physics step, s.
    :ivar solver: PhysX solver; PGS (TGS loses restitution when contact starts early in a step).
    :ivar device: ``cuda:0`` for the GPU pipeline.
    :ivar env_spacing_m: Spacing of the environments, m.
    :ivar layout: ``lattice`` places the environments on a centred cube (collisions between environments are
        filtered, so stacking is harmless) to keep world coordinates small: PhysX GPU holds positions in float32, and
        a slow ball far from the origin drifts systematically by up to half a float32 ulp per step (measured on
        production: about 4 mm in 0.4 s at 126 m). ``grid`` is Isaac Lab's flat grid.
    :ivar replicate_physics: Let PhysX replicate environment 0; only legal when every environment's USD is identical.
    :ivar contact_offset_m: PhysX speculative contact distance per collider.
    :ivar rest_offset_m: PhysX rest offset per collider.
    :ivar position_iterations: Solver position iterations per body.
    :ivar velocity_iterations: Solver velocity iterations per body.
    :ivar bounce_threshold_mps: Relative normal speed below which PhysX applies no restitution.
    :ivar heavy_paddle_mass_kg: Mass of the ``heavy`` paddle body.
    :ivar pd_omega_dt: PD paddle natural frequency times the step.
    :ivar contact_dv_threshold_mps: A step whose velocity change departs from free flight by more than this held a contact.
    :ivar contact_dw_threshold_rad_s: The same for spin.
    :ivar log_dt_s: Interval of the decimated ball log, s; a whole number of steps.
    :ivar max_sim_s: Safety cap on simulated time; reaching it with unfinished environments fails the batch.
    :ivar table_thickness_m: Thickness of the table slab under the playing surface.
    :ivar max_contacts: Contact slots recorded per environment.
    :ivar hybrid_write: How a hybrid contact's outcome reaches PhysX: ``pose_velocity`` teleports the ball to where the
        event-driven solution starts the step and writes its velocities; ``velocity`` writes only the velocities (the ball
        leaves from where it is, at most ``|dv| dt`` from the touch); ``impulse`` adds ``m dv / dt`` and ``I dw / dt`` to the
        step's wrench and writes nothing.
    :ivar paddle_mass_model: How the hybrid impact treats the rubber's cited (rigid-hold) restitution.
    :ivar rigid_hold_mass_kg: Effective mass of the rig the rubber restitution was measured on (``inf``: rigid).
    """

    model_config = _FROZEN

    physics_dt_s: float = Field(default=1.0 / 4000.0, gt=0.0, le=0.01)
    solver: Literal["pgs", "tgs"] = "pgs"
    device: str = "cuda:0"
    env_spacing_m: float = Field(default=4.0, gt=0.0)
    layout: Literal["lattice", "grid"] = "lattice"
    replicate_physics: bool = False
    contact_offset_m: float = Field(default=0.01, gt=0.0)
    rest_offset_m: float = Field(default=0.0, ge=0.0)
    position_iterations: int = Field(default=16, ge=1, le=255)
    velocity_iterations: int = Field(default=4, ge=0, le=255)
    bounce_threshold_mps: float = Field(default=0.1, ge=0.0)
    heavy_paddle_mass_kg: float = Field(default=1000.0, gt=0.0)
    pd_omega_dt: float = Field(default=0.3, gt=0.0, le=1.0)
    contact_dv_threshold_mps: float = Field(default=0.05, gt=0.0)
    contact_dw_threshold_rad_s: float = Field(default=2.0, gt=0.0)
    log_dt_s: float = Field(default=1e-3, gt=0.0)
    max_sim_s: float = Field(default=3.0, gt=0.0)
    table_thickness_m: float = Field(default=0.025, gt=0.0)
    max_contacts: int = Field(default=6, ge=1, le=64)
    hybrid_write: Literal["pose_velocity", "velocity", "impulse"] = "pose_velocity"
    paddle_mass_model: PaddleMassModel = "human_intrinsic"
    rigid_hold_mass_kg: float = Field(default=math.inf, gt=0.0)

    @property
    def log_every(self) -> int:
        """Physics steps per log sample."""

        return max(1, round(self.log_dt_s / self.physics_dt_s))

    @property
    def log_interval_s(self) -> float:
        """The realised log interval, s."""

        return self.log_every * self.physics_dt_s


class BatchSpec(BaseModel):
    """
    Everything one engine build simulates.

    :ivar name: Name of the build, for records.
    :ivar engine: Global configuration.
    :ivar param_sets: Engine parameter sets by key. Geometry (table, net, paddle) and the lift form
        must agree across them: the adapter authors one table, net and blade shape per build.
    :ivar envs: One specification per environment, in environment order.
    """

    model_config = _FROZEN

    name: str = Field(min_length=1)
    engine: EngineConfig
    param_sets: dict[str, BallTableParams]
    envs: tuple[EnvSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        labels = [env.label for env in self.envs]
        if len(set(labels)) != len(labels):
            raise ValueError(f"batch {self.name!r} repeats environment labels")
        missing = sorted({env.params for env in self.envs} - set(self.param_sets))
        if missing:
            raise ValueError(f"batch {self.name!r} references unknown parameter sets {missing}")
        first = next(iter(self.param_sets.values()))
        for key, params in self.param_sets.items():
            if params.table != first.table or params.net != first.net:
                raise ValueError(f"parameter set {key!r} changes the table or net geometry; one build authors one table")
            shape = (params.paddle.blade_half_length, params.paddle.blade_half_width, params.paddle.thickness)
            if shape != (first.paddle.blade_half_length, first.paddle.blade_half_width, first.paddle.thickness):
                raise ValueError(f"parameter set {key!r} changes the blade shape; one build authors one blade")
            if params.aero.lift.form != first.aero.lift.form:
                raise ValueError(f"parameter set {key!r} changes the lift form; one build evaluates one form")
        if self.engine.replicate_physics and len({self._usd_signature(env) for env in self.envs}) > 1:
            raise ValueError(f"batch {self.name!r} varies per-environment USD (materials, masses, caps, drives); it cannot replicate physics")
        return self

    def _usd_signature(self, env: EnvSpec) -> tuple[object, ...]:
        params = self.param_sets[env.params]
        return (
            params.ball,
            params.table_contact.restitution,
            params.table_contact.friction,
            params.rubber_contact.restitution,
            params.rubber_contact.friction,
            params.net_contact.restitution,
            params.net_contact.friction,
            params.paddle.effective_mass if env.paddle_mode == "pd" else None,
            env.paddle_mode,
            env.paddle_contact,
            env.table_contact,
            env.net_contact,
            env.max_angular_velocity_deg_s,
        )

    @property
    def geometry(self) -> BallTableParams:
        """A parameter set carrying the build's shared geometry."""

        return next(iter(self.param_sets.values()))

    @property
    def num_envs(self) -> int:
        """Number of environments."""

        return len(self.envs)


@dataclass
class BatchRecord:
    """
    What one engine build measured, environment-local (table frame), in numpy.

    :ivar name: The build's name.
    :ivar engine: Its configuration.
    :ivar times: Log sample times, s, ``(S,)``.
    :ivar ball: Decimated ball states, ``(S, N, 9)``: position, velocity, spin.
    :ivar contact_count: Contacts recorded per environment, ``(N,)`` (may exceed the slots).
    :ivar contact_time: Contact time, s, ``(N, K)``: the exact touch for hybrid contacts, the step start for native ones.
    :ivar contact_step: Physics step index of each contact, ``(N, K)``.
    :ivar contact_kind: Surface code (:data:`SURFACE_CODES`), ``(N, K)``.
    :ivar contact_hybrid: Whether the reference impact model imposed the contact, ``(N, K)``.
    :ivar contact_before: Ball state before, ``(N, K, 9)``: at the touch (hybrid) or the step start (native).
    :ivar contact_after: Ball state after, ``(N, K, 9)``: at the touch (hybrid) or the step end (native).
    :ivar tunnels: Crossings without a contact per environment, ``(N, 3)``: paddle, table, net.
    :ivar done_time: When each environment finished, s; ``nan`` if it never did, ``(N,)``.
    :ivar env_origins: Environment origins in the world, m, ``(N, 3)``.
    :ivar terminal: The named condition that ended the batch.
    :ivar sim_s: Simulated seconds.
    :ivar steps: Physics steps taken.
    :ivar timing: Wall-clock breakdown, s.
    :ivar readback: Values read back from the engine after the build (materials, masses, caps).
    :ivar contact_sustained: Whether each recorded contact was a sustained (resting or rolling) episode, ``(N, K)``;
        ``None`` for records that predate sustained contacts.
    """

    name: str
    engine: EngineConfig
    times: NDArray[np.float64]
    ball: NDArray[np.float32]
    contact_count: NDArray[np.int64]
    contact_time: NDArray[np.float64]
    contact_step: NDArray[np.int64]
    contact_kind: NDArray[np.int64]
    contact_hybrid: NDArray[np.bool_]
    contact_before: NDArray[np.float64]
    contact_after: NDArray[np.float64]
    tunnels: NDArray[np.int64]
    done_time: NDArray[np.float64]
    env_origins: NDArray[np.float64]
    terminal: str
    sim_s: float
    steps: int
    timing: dict[str, float] = field(default_factory=dict)
    readback: dict[str, object] = field(default_factory=dict)
    contact_sustained: NDArray[np.bool_] | None = None

    @property
    def num_envs(self) -> int:
        """Number of environments."""

        return int(self.ball.shape[1])

    def contacts(self, env: int) -> list[tuple[str, bool, float, NDArray[np.float64], NDArray[np.float64]]]:
        """
        The recorded contacts of one environment, in order.

        :param env: Environment index.
        :return: ``(surface, hybrid, time, before, after)`` per contact.
        """

        count = min(int(self.contact_count[env]), self.contact_kind.shape[1])
        return [
            (
                SURFACE_NAMES[int(self.contact_kind[env, k])],
                bool(self.contact_hybrid[env, k]),
                float(self.contact_time[env, k]),
                self.contact_before[env, k],
                self.contact_after[env, k],
            )
            for k in range(count)
        ]


TERMINAL_ALL_COMPLETE: str = "every environment reached its stopping contact or its end time (all experiments complete)"
"""The named terminal condition of a batch."""

TERMINAL_CAP: str = "the max_sim_s safety cap was reached with environments unfinished (a recorded failure)"


__all__ = [
    "PARK_POSITION",
    "SURFACE_CODES",
    "SURFACE_NAMES",
    "TERMINAL_ALL_COMPLETE",
    "TERMINAL_CAP",
    "BatchRecord",
    "BatchSpec",
    "EngineConfig",
    "EnvSpec",
    "NetContact",
    "PaddleContact",
    "PaddleMassModel",
    "PaddleMode",
    "PaddleScript",
    "TableContact",
    "Vec3",
]
