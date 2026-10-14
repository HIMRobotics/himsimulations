"""
Run one :class:`BatchSpec` on Isaac Lab 3.0 (PhysX GPU) and return its :class:`BatchRecord`.

Each physics step, for every environment at once (torch on the GPU):

1. **Hybrid contacts.** Sweep the ball against its paddle's faces (and the
   table, in hybrid table mode) over the coming step; at a touch, resolve the
   impact with the reference impact model (:func:`.vectorized.resolve_impacts`)
   and write the post-impact velocity and spin, starting the ball where the
   event-driven solution puts it (the game adapter's scheme). The hybrid
   surfaces have no ball collider, so PhysX never sees those pairs.
2. **Aerodynamics.** Apply :meth:`FlightModel.wrench` (batched) as a world-frame
   force and torque, computed from the state the step starts with.
3. **Paddle drive.** Rewrite the pose and velocity of every ``heavy`` or
   ``kinematic`` paddle from its script; pull every ``pd`` paddle to its
   script with a stiff PD wrench.
4. **Step** PhysX (PGS, no CCD on the GPU pipeline).
5. **Native contacts.** A step whose velocity or spin change departs from free
   flight beyond a threshold held an engine contact; classify its surface by
   proximity and record the states either side. Record tunnelling: a ball
   centre crossing a solid surface inside its outline.
6. **Stop.** An environment finishes ``hold_s`` after its stopping contact or
   at ``t_end_s``; the batch ends when every environment has finished (the
   named terminal condition), or fails at the ``max_sim_s`` cap.

Simulator imports stay inside functions: this module imports without Isaac.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any

import numpy as np

from lib.physics.engine.lab import vectorized as vz
from lib.physics.engine.lab.authoring import apply_env_physics, build_scene, lattice_origins, make_authors, place_envs, read_back
from lib.physics.engine.lab.batch import PARK_POSITION, SURFACE_CODES, SURFACE_NAMES, TERMINAL_ALL_COMPLETE, TERMINAL_CAP, BatchRecord, BatchSpec
from lib.physics.engine.lab.materials import EnvPhysics, ball_body_settings, ball_friction, engine_params, env_physics
from lib.physics.params import BallTableParams, ContactMaterial
from lib.physics.reference.flight import FlightModel

FALLEN_Z: float = -0.5
"""A ball centre this far below the playing surface (m) can reach nothing else: its environment is finished."""

BOARD_SWEEP_MARGIN_M: float = 0.005
"""The table board and the net fixtures are swept only when some ball is within one step's travel, its radius and
this margin of them (the playing surface and the net box are always swept)."""

SUSTAINED_GAP_STEPS: int = 2
"""A sustained contact on the same surface within this many steps of the last is the same episode (the game adapter's
``ContactEpisodes.max_gap_steps``)."""

CHECK_EVERY_STEPS: int = 200
"""Steps between checks of the batch's completion and of cancellation."""


def shared_curve(laws: list[ContactMaterial], field: str) -> vz.CurveSpec | None:
    """
    The one speed curve every environment's law shares for ``field`` (``restitution_curve`` or ``friction_curve``).

    Scalars may differ per environment (they scale the curve); the curve's knots and reference may not.

    :param laws: Each environment's impact law for one surface.
    :param field: Which curve.
    :return: The shared curve, or ``None`` when no law has one.
    :raises ValueError: If the environments disagree on the curve
    """

    curves = {getattr(law, field) for law in laws}
    if len(curves) > 1:
        raise ValueError(f"the {field} differs across environments; one build shares one curve (scalars may differ)")
    curve = curves.pop()
    if curve is None:
        return None
    return vz.CurveSpec(speeds=tuple(curve.speeds), values=tuple(curve.values), reference=curve.reference)


class LabBallHarness:
    """
    One engine build: compose the batch on a fresh Lab simulation, step it to its terminal condition, release it.

    Use as a context manager so the Lab singleton is always cleared::

        with LabBallHarness(spec) as harness:
            record = harness.run()

    :param spec: The batch.
    :param log: Progress callback.
    :param fresh_stage: Create a new USD stage first (every build after the first in one process).
    """

    def __init__(self, spec: BatchSpec, *, log: Callable[[str], None] = print, fresh_stage: bool = False) -> None:
        self.spec = spec
        self.log = log
        self.fresh_stage = fresh_stage
        self.simulation: Any = None
        self.scene: Any = None
        self.params: dict[str, BallTableParams] = engine_params(spec)
        self.values: list[EnvPhysics] = [env_physics(spec, env, self.params[env.params]) for env in spec.envs]
        self.timing: dict[str, float] = {}
        self.readback: dict[str, object] = {}

    def __enter__(self) -> LabBallHarness:
        self.build()
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None) -> None:
        self.close()

    # ----------------------------------------------------------------------------------
    # Build
    # ----------------------------------------------------------------------------------

    def build(self) -> None:
        """Author every environment, write each one's physics, reset, and prepare the tensors."""

        from isaaclab.sim.utils import create_new_stage, get_current_stage

        started = time.perf_counter()
        if self.fresh_stage:
            create_new_stage()
        authors = make_authors(self.spec, self.values[0], ball_friction(self.spec), ball_body_settings(self.spec))
        self.simulation, self.scene = build_scene(self.spec, authors)
        stage = get_current_stage()
        written = 0 if self.spec.engine.replicate_physics else apply_env_physics(stage, self.values)
        self.lattice: list[tuple[float, float, float]] | None = None
        if self.spec.engine.layout == "lattice":
            if self.spec.engine.replicate_physics:
                raise ValueError("the lattice layout re-places environments in USD, which replicated physics ignores")
            self.lattice = lattice_origins(self.spec.num_envs, self.spec.engine.env_spacing_m)
            place_envs(stage, self.lattice)
        authored = time.perf_counter()
        sample = sorted({0, self.spec.num_envs - 1, *self._first_of_each_variant()})[:24]
        self.readback["usd"] = read_back(stage, sample)
        self.readback["usd_attributes_written"] = written
        self.simulation.reset()
        self.timing["author_s"] = authored - started
        self.timing["reset_s"] = time.perf_counter() - authored
        self._prepare_tensors()
        self.timing["build_s"] = time.perf_counter() - started
        self.log(
            f"built {self.spec.name}: {self.spec.num_envs} envs, dt 1/{round(1 / self.spec.engine.physics_dt_s)}, {self.spec.engine.solver}, {self.timing['build_s']:.1f} s"
        )

    def _first_of_each_variant(self) -> list[int]:
        seen: dict[EnvPhysics, int] = {}
        for index, values in enumerate(self.values):
            seen.setdefault(values, index)
        return list(seen.values())

    def close(self) -> None:
        """Release the Lab simulation (its singleton and stage)."""

        if self.simulation is not None:
            from isaaclab.sim import SimulationContext

            SimulationContext.clear_instance()
            self.simulation = None
            self.scene = None

    def _prepare_tensors(self) -> None:
        import torch  # pyright: ignore[reportMissingImports]
        from scipy.spatial.transform import Rotation

        spec = self.spec
        device = self.simulation.device
        self.device = device
        self.torch = torch

        def tensor(rows: Any, dtype: Any = torch.float32) -> Any:
            return torch.tensor(np.asarray(rows), dtype=dtype, device=device)

        envs = spec.envs
        params = [self.params[env.params] for env in envs]
        self.ball_arrays = vz.BallArrays(
            mass=tensor([v.ball_mass for v in self.values]),
            radius=tensor([v.ball_radius for v in self.values]),
            inertia=tensor([v.ball_inertia for v in self.values]),
        )
        models = [FlightModel.from_params(p) for p in params]
        aero_on = [env.aero for env in envs]
        self.aero_arrays = vz.AeroArrays(
            mass=self.ball_arrays.mass,
            inertia=self.ball_arrays.inertia,
            radius=self.ball_arrays.radius,
            aero_gain=tensor([m.aero_gain if on else 0.0 for m, on in zip(models, aero_on, strict=True)]),
            drag_coefficient=tensor([m.drag_coefficient for m in models]),
            drag_spin_slope=tensor([m.drag_spin_slope for m in models]),
            lift_slope=tensor([m.lift.slope for m in models]),
            lift_c_max=tensor([m.lift.c_max for m in models]),
            spin_decay_gain=tensor([m.spin_decay_gain if on else 0.0 for m, on in zip(models, aero_on, strict=True)]),
            lift_form=models[0].lift.form,
        )
        self.gravity = tensor([0.0, 0.0, -spec.geometry.aero.gravity])

        def material(label: str) -> vz.MaterialArrays:
            laws: list[ContactMaterial] = [getattr(p, label) for p in params]
            return vz.MaterialArrays(
                restitution=tensor([law.restitution for law in laws]),
                friction=tensor([law.friction for law in laws]),
                tangential_restitution=tensor([law.tangential_restitution for law in laws]),
                normal_spin_retention=tensor([law.normal_spin_retention for law in laws]),
                restitution_curve=shared_curve(laws, "restitution_curve"),
                friction_curve=shared_curve(laws, "friction_curve"),
                tangential_damping=tensor([law.tangential_damping for law in laws]),
                spin_damping=tensor([law.spin_damping for law in laws]),
            )

        self.table_material = material("table_contact")
        self.rubber_material = material("rubber_contact")
        self.net_material = material("net_contact")
        self.fixture_material = material("fixture_contact")
        # Steel posts take ``post_contact`` once the contract has it (physics-ref addendum 3); before, the net's rigid law.
        self.post_material = material("post_contact") if all(hasattr(p, "post_contact") for p in params) else self.net_material
        self.zero_inverse_mass = tensor([0.0] * len(envs))
        self.inverse_paddle_mass = tensor([0.0 if math.isinf(p.paddle.effective_mass) else 1.0 / p.paddle.effective_mass for p in params])
        # The hybrid paddle is the reference's free rigid body (off-centre hits recoil and turn it); an infinitely
        # heavy paddle stays a point mass that does not move.
        self.inverse_paddle_moments = tensor(
            [[0.0, 0.0, 0.0] if math.isinf(p.paddle.effective_mass) else [1.0 / moment for moment in p.paddle.inertia] for p in params]
        )
        geometry = spec.geometry
        self.blade = vz.BladeGeometry(
            half_thickness=0.5 * geometry.paddle.thickness, half_length=geometry.paddle.blade_half_length, half_width=geometry.paddle.blade_half_width
        )
        self.half_length = 0.5 * geometry.table.length
        self.half_width = 0.5 * geometry.table.width
        self.net_half_thickness = 0.5 * geometry.net.thickness
        self.net_half_span = geometry.net_half_span
        self.net_height = geometry.net.height
        self.net_tape_height = geometry.net.tape_height
        self.board_thickness = geometry.table.board_thickness
        # The reference's net posts (the net's material, rigid) and clamps (the fixture material), hybrid with the net.
        from lib.physics.reference.surfaces import net_fixtures

        self.fixture_shapes: list[tuple[str, vz.Shape]] = []
        for fixture in net_fixtures(geometry):
            if fixture.shape == "box":
                c, h = fixture.centre, fixture.size
                shape: vz.Shape = ("box", (c[0] - h[0], c[1] - h[1], c[2] - h[2]), (c[0] + h[0], c[1] + h[1], c[2] + h[2]), (0.0, 0.0))
            else:
                shape = ("cylinder", fixture.centre, fixture.size, fixture.z_range)
            self.fixture_shapes.append((fixture.region, shape))
        # The table's undercarriage (apron, legs, stretchers, struts, feet, casters), hybrid with the table, fixture material.
        from lib.physics.reference.undercarriage import TableFrame, undercarriage_parts

        self.frame_parts: list[tuple[str, vz.Shape]] = [(part.name, vz.part_shape(part)) for part in undercarriage_parts(geometry)]
        frame_low, frame_high = TableFrame.from_params(geometry).bounds
        self.frame_bounds: vz.Shape = ("box", frame_low, frame_high, (0.0, 0.0))
        lows = [f[1][1] if f[1][0] == "box" else (f[1][1][0] - f[1][2][0], f[1][1][1] - f[1][2][0], f[1][3][0]) for f in self.fixture_shapes]
        highs = [f[1][2] if f[1][0] == "box" else (f[1][1][0] + f[1][2][0], f[1][1][1] + f[1][2][0], f[1][3][1]) for f in self.fixture_shapes]
        self.fixture_bounds: list[vz.Shape] = []
        for sign in (-1.0, 1.0):
            side = [(lo, hi) for lo, hi in zip(lows, highs, strict=True) if (lo[1] + hi[1]) * sign > 0]
            self.fixture_bounds.append(
                (
                    "box",
                    tuple(min(lo[axis] for lo, _hi in side) for axis in range(3)),  # type: ignore[arg-type]
                    tuple(max(hi[axis] for _lo, hi in side) for axis in range(3)),  # type: ignore[arg-type]
                    (0.0, 0.0),
                )
            )

        modes = [env.paddle_mode for env in envs]
        self.has_paddle = tensor([m != "none" for m in modes], torch.bool)
        self.paddle_hybrid = tensor([m != "none" and env.paddle_contact == "hybrid" for m, env in zip(modes, envs, strict=True)], torch.bool)
        self.paddle_native = tensor([m != "none" and env.paddle_contact == "native" for m, env in zip(modes, envs, strict=True)], torch.bool)
        self.table_hybrid = tensor([env.table_contact == "hybrid" for env in envs], torch.bool)
        self.net_hybrid = tensor([env.net_contact == "hybrid" for env in envs], torch.bool)
        self.any_paddle = any(m != "none" for m in modes)
        self.any_driven = any(m != "none" and env.paddle_contact == "native" for m, env in zip(modes, envs, strict=True))
        self.any_hybrid_paddle = bool(self.paddle_hybrid.any().item())
        self.any_hybrid_table = bool(self.table_hybrid.any().item())
        self.any_hybrid_net = bool(self.net_hybrid.any().item())
        self.pd_ids = tensor([i for i, m in enumerate(modes) if m == "pd"], torch.long)
        # Only paddles that collide (native contacts) are driven. Rewriting, every step, the pose of a dynamic body whose
        # only collider is disabled corrupted unrelated ball-table contacts on PhysX GPU (measured on production: a 19-env
        # batch's balls fell through their tables until those writes stopped); hybrid and absent paddles need no engine body.
        self.drive_ids = tensor(
            [i for i, (m, env) in enumerate(zip(modes, envs, strict=True)) if m in ("heavy", "kinematic") and env.paddle_contact == "native"], torch.long
        )
        self.pd_mass = tensor([v.paddle_mass for v in self.values])
        self.pd_inertia = tensor([v.paddle_inertia for v in self.values])

        positions, rotations, quaternions, velocities, spins = [], [], [], [], []
        for env in envs:
            if env.paddle is None:
                rotation = np.eye(3)
                positions.append(PARK_POSITION)
                velocities.append((0.0, 0.0, 0.0))
                spins.append((0.0, 0.0, 0.0))
            else:
                rotation = env.paddle.rotation()
                positions.append(env.paddle.position)
                velocities.append(env.paddle.velocity)
                spins.append(env.paddle.angular_velocity)
            rotations.append(rotation)
            quaternions.append(Rotation.from_matrix(rotation).as_quat())
        self.script = vz.ScriptArrays(
            position=tensor(positions), rotation=tensor(rotations), quaternion=tensor(quaternions), velocity=tensor(velocities), angular_velocity=tensor(spins)
        )
        self.t_end = tensor([env.t_end_s for env in envs])
        self.stop_after = tensor([env.stop_after_contacts for env in envs], torch.long)
        self.hold = tensor([env.hold_s for env in envs])
        self.initial = tensor([env.initial_state for env in envs])
        self.origins = tensor(self.lattice) if self.lattice is not None else self.scene.env_origins.to(device).float()
        self.ball = self.scene["ball"]
        self.paddle = self.scene["paddle"]
        try:
            masses = self.ball.data.body_mass.torch.reshape(-1).float()
            self.readback["engine_ball_mass_min_max"] = [float(masses.min().item()), float(masses.max().item())]
            paddle_masses = self.paddle.data.body_mass.torch.reshape(-1).float()
            self.readback["engine_paddle_mass_min_max"] = [float(paddle_masses.min().item()), float(paddle_masses.max().item())]
            usd_ball = torch.tensor([v.ball_mass for v in self.values], device=masses.device, dtype=masses.dtype)
            self.readback["engine_ball_mass_max_error"] = float((masses - usd_ball).abs().max().item())
        except Exception as exc:  # diagnostics only: a readback attribute must never fail the batch
            self.readback["engine_mass_readback_error"] = repr(exc)

    # ----------------------------------------------------------------------------------
    # Engine I/O
    # ----------------------------------------------------------------------------------

    def _read_ball(self) -> Any:
        torch = self.torch
        pose = self.ball.data.root_link_pose_w.torch
        velocity = self.ball.data.root_com_vel_w.torch
        return torch.cat([pose[:, 0:3] - self.origins, velocity[:, 0:3], velocity[:, 3:6]], -1).clone()

    def _write_ball(self, state: Any, ids: Any) -> None:
        torch = self.torch
        quaternion = torch.zeros((state.shape[0], 4), device=self.device)
        quaternion[:, 3] = 1.0
        pose = torch.cat([state[:, 0:3] + self.origins, quaternion], -1)
        self.ball.write_root_pose_to_sim_index(root_pose=pose[ids], env_ids=ids)
        self.ball.write_root_velocity_to_sim_index(root_velocity=state[ids, 3:9].contiguous(), env_ids=ids)

    def _write_ball_motion(self, position: Any, velocity: Any, spin: Any, ids: Any) -> None:
        """Write position and velocity, keeping each ball's current orientation (spin integrates it)."""

        torch = self.torch
        orientation = self.ball.data.root_link_pose_w.torch[:, 3:7]
        pose = torch.cat([position + self.origins, orientation], -1)
        self.ball.write_root_pose_to_sim_index(root_pose=pose[ids].contiguous(), env_ids=ids)
        self.ball.write_root_velocity_to_sim_index(root_velocity=torch.cat([velocity, spin], -1)[ids].contiguous(), env_ids=ids)

    def _drive_paddles(self, blade: vz.BladeState) -> None:
        torch = self.torch
        if self.drive_ids.numel():
            pose = torch.cat([blade.position + self.origins, blade.quaternion], -1)
            twist = torch.cat([blade.velocity, blade.angular_velocity], -1)
            ids = self.drive_ids
            self.paddle.write_root_pose_to_sim_index(root_pose=pose[ids].contiguous(), env_ids=ids)
            self.paddle.write_root_velocity_to_sim_index(root_velocity=twist[ids].contiguous(), env_ids=ids)
        if self.pd_ids.numel():
            pose = self.paddle.data.root_link_pose_w.torch
            velocity = self.paddle.data.root_com_vel_w.torch
            dt = self.spec.engine.physics_dt_s
            omega = self.spec.engine.pd_omega_dt / dt
            position = pose[:, 0:3] - self.origins
            force = self.pd_mass[:, None] * (omega * omega * (blade.position - position) + 2.0 * omega * (blade.velocity - velocity[:, 0:3]))
            current = pose[:, 3:7]
            conjugate = current * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=self.device)
            error = vz.quat_multiply(torch, blade.quaternion, conjugate)
            sign = torch.where(error[:, 3:4] < 0, -1.0, 1.0)
            rotation_error = 2.0 * error[:, 0:3] * sign
            demand = omega * omega * rotation_error + 2.0 * omega * (blade.angular_velocity - velocity[:, 3:6])
            body = (blade.rotation.transpose(1, 2) @ demand[:, :, None])[:, :, 0] * self.pd_inertia
            torque = (blade.rotation @ body[:, :, None])[:, :, 0]
            mask = torch.zeros_like(force[:, 0:1])
            mask[self.pd_ids] = 1.0
            self.paddle.permanent_wrench_composer.set_forces_and_torques_index(
                forces=(force * mask)[:, None, :].contiguous(), torques=(torque * mask)[:, None, :].contiguous(), is_global=True
            )

    # ----------------------------------------------------------------------------------
    # Classification
    # ----------------------------------------------------------------------------------

    def _blade_distance(self, position: Any, blade: vz.BladeState) -> tuple[Any, Any]:
        """Signed distance of each ball centre from its blade's mid-plane, and whether it lies over the blade box."""

        offset = position - blade.position
        local = (blade.rotation.transpose(1, 2) @ offset[:, :, None])[:, :, 0]
        over = (local[:, 0].abs() <= self.blade.half_length + self.ball_arrays.radius) & (local[:, 1].abs() <= self.blade.half_width + self.ball_arrays.radius)
        return local[:, 2], over & self.has_paddle

    def _net_gap(self, position: Any) -> Any:
        torch = self.torch
        x = position[:, 0].clamp(-self.net_half_thickness, self.net_half_thickness)
        y = position[:, 1].clamp(-self.net_half_span, self.net_half_span)
        z = position[:, 2].clamp(0.0, self.net_height)
        closest = torch.stack([x, y, z], -1)
        return vz.norm(position - closest) - self.ball_arrays.radius

    def _classify(self, before: Any, after: Any, blade_before: vz.BladeState, blade_after: vz.BladeState) -> Any:
        """Surface code of an engine contact: the nearest surface over the step, if it is within reach."""

        torch = self.torch
        radius = self.ball_arrays.radius
        margin = self.spec.engine.contact_offset_m + 0.01
        far = torch.full_like(radius, 1e9)
        inside = (before[:, 0].abs() <= self.half_length + margin) & (before[:, 1].abs() <= self.half_width + margin)
        table_gap = torch.where(inside & (before[:, 2] > -0.5 * radius), torch.minimum(before[:, 2], after[:, 2]) - radius, far)
        d0, over0 = self._blade_distance(before[:, 0:3], blade_before)
        d1, over1 = self._blade_distance(after[:, 0:3], blade_after)
        reach = radius + self.blade.half_thickness
        paddle_gap = torch.where(over0 | over1, torch.minimum(d0.abs(), d1.abs()) - reach, far)
        net_gap = torch.minimum(self._net_gap(before[:, 0:3]), self._net_gap(after[:, 0:3]))
        gaps = torch.stack([table_gap, paddle_gap, net_gap], -1)
        nearest = gaps.argmin(-1)
        codes = torch.tensor([SURFACE_CODES["table"], SURFACE_CODES["paddle"], SURFACE_CODES["net"]], device=self.device)
        kind = codes[nearest]
        return torch.where(gaps.min(-1).values < margin, kind, torch.full_like(kind, SURFACE_CODES["other"]))

    def _tunnels(self, before: Any, after: Any, blade_before: vz.BladeState, blade_after: vz.BladeState) -> Any:
        """Crossings without a contact this step: ``(N, 3)`` for paddle (native), table (native) and net."""

        torch = self.torch
        thickness = self.spec.engine.table_thickness_m
        inside = (after[:, 0].abs() <= self.half_length) & (after[:, 1].abs() <= self.half_width)
        table = inside & (before[:, 2] > 0.5 * self.ball_arrays.radius) & (after[:, 2] < -thickness) & ~self.table_hybrid
        d0, over0 = self._blade_distance(before[:, 0:3], blade_before)
        d1, over1 = self._blade_distance(after[:, 0:3], blade_after)
        paddle = ((d0 >= 0) != (d1 >= 0)) & over0 & over1 & self.paddle_native
        x0, x1 = before[:, 0], after[:, 0]
        crossing = (x0 >= 0) != (x1 >= 0)
        denominator = torch.where((x0 - x1).abs() > 1e-12, x0 - x1, torch.ones_like(x0))
        fraction = (x0 / denominator).clamp(0.0, 1.0)
        at = before[:, 0:3] + fraction[:, None] * (after[:, 0:3] - before[:, 0:3])
        radius = self.ball_arrays.radius
        net = crossing & (at[:, 1].abs() < self.net_half_span - radius) & (at[:, 2] > 0.0) & (at[:, 2] < self.net_height - radius) & ~self.net_hybrid
        return torch.stack([paddle, table, net], -1).long()

    # ----------------------------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------------------------

    def run(self, cancelled: Callable[[], None] | None = None, progress: Callable[[float, float], None] | None = None) -> BatchRecord:
        """
        Step until every environment has finished, or the ``max_sim_s`` cap.

        :param cancelled: Called every few hundred steps; it raises to stop the batch (``run.raise_if_cancelled``).
        :param progress: Called every few hundred steps with ``(sim_s, finished fraction)``.
        :return: The measured record.
        """

        torch = self.torch
        spec = self.spec
        engine = spec.engine
        dt = engine.physics_dt_s
        n = spec.num_envs
        device = self.device
        cap_steps = math.ceil(engine.max_sim_s / dt - 1e-9)
        log_every = engine.log_every
        samples = cap_steps // log_every + 2
        slots = engine.max_contacts
        rows = torch.arange(n, device=device)

        log = torch.zeros((samples, n, 9), dtype=torch.float32, device=device)
        contact_count = torch.zeros(n, dtype=torch.long, device=device)
        surface_count = torch.zeros(n, dtype=torch.long, device=device)
        contact_time = torch.full((n, slots), float("nan"), dtype=torch.float64, device=device)
        contact_step = torch.full((n, slots), -1, dtype=torch.long, device=device)
        contact_kind = torch.zeros((n, slots), dtype=torch.long, device=device)
        contact_hybrid = torch.zeros((n, slots), dtype=torch.bool, device=device)
        contact_before = torch.full((n, slots, 9), float("nan"), dtype=torch.float32, device=device)
        contact_after = torch.full((n, slots, 9), float("nan"), dtype=torch.float32, device=device)
        tunnels = torch.zeros((n, 3), dtype=torch.long, device=device)
        stop_time = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        done_time = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
        threshold_v = engine.contact_dv_threshold_mps
        threshold_w = engine.contact_dw_threshold_rad_s

        contact_sustained = torch.zeros((n, slots), dtype=torch.bool, device=device)
        # A sustained (resting or rolling) contact is one event per episode: the surface and the last step it touched.
        episode_code = torch.zeros(n, dtype=torch.long, device=device)
        episode_step = torch.full((n,), -10, dtype=torch.long, device=device)

        def record(mask: Any, time_s: Any, step: int, kind: Any, hybrid: bool, before: Any, after: Any, sustained: Any = None) -> None:
            slot = contact_count.clamp(max=slots - 1)
            keep = mask[:, None]
            if sustained is not None:
                contact_sustained[rows, slot] = torch.where(mask, sustained, contact_sustained[rows, slot])
            contact_time[rows, slot] = torch.where(mask, time_s.double(), contact_time[rows, slot])
            contact_step[rows, slot] = torch.where(mask, torch.full_like(slot, step), contact_step[rows, slot])
            contact_kind[rows, slot] = torch.where(mask, kind, contact_kind[rows, slot])
            contact_hybrid[rows, slot] = torch.where(mask, torch.full_like(mask, hybrid), contact_hybrid[rows, slot])
            contact_before[rows, slot] = torch.where(keep, before, contact_before[rows, slot])
            contact_after[rows, slot] = torch.where(keep, after, contact_after[rows, slot])
            contact_count.add_(mask.long())
            surface = mask & (kind != SURFACE_CODES["other"])
            surface_count.add_(surface.long())
            stopping = surface & (self.stop_after > 0) & (surface_count == self.stop_after)
            stop_time.copy_(torch.where(stopping, time_s.float() + self.hold, stop_time))

        all_ids = rows
        zero_wrench = torch.zeros((n, 3), dtype=torch.float32, device=device)
        self._write_ball(self.initial, all_ids)
        blade_now = vz.script_pose(torch, self.script, 0.0)
        if self.any_driven:
            self._drive_paddles(blade_now)
        state = self.initial.clone()
        log[0] = state
        log_index = 1
        terminal = TERMINAL_CAP
        hybrid_counts: dict[int, int] = {}
        painleve = 0
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        loop_started = time.perf_counter()
        steps = 0
        for k in range(cap_steps):
            t = k * dt
            blade_next = vz.script_pose(torch, self.script, t + dt) if self.any_paddle else blade_now
            position, velocity, spin = state[:, 0:3], state[:, 3:6], state[:, 6:9]

            # 1. Hybrid contacts over the coming step.
            impulse_force = zero_wrench
            impulse_torque = zero_wrench
            if self.any_hybrid_paddle or self.any_hybrid_table or self.any_hybrid_net:
                candidates: list[tuple[Any, vz.SweepBatch, vz.MaterialArrays, Any, int, str]] = []
                radius = self.ball_arrays.radius
                reach = vz.norm(velocity) * dt + radius + BOARD_SWEEP_MARGIN_M
                if self.any_hybrid_paddle:
                    sweep_p = vz.sweep_blades(torch, position, velocity, radius, blade_now, blade_next, dt, self.blade)
                    tunnels[:, 0] += (sweep_p.tunnelled & self.paddle_hybrid).long()
                    candidates.append(
                        (sweep_p.contact & self.paddle_hybrid, sweep_p, self.rubber_material, self.inverse_paddle_mass, SURFACE_CODES["paddle"], "paddle")
                    )
                if self.any_hybrid_table:
                    # The playing surface first; the rest of the board (edges, ends, sides, underside) only for balls near it.
                    over_top = (position[:, 2] >= 0) & (abs(position[:, 0]) <= self.half_length) & (abs(position[:, 1]) <= self.half_width)
                    board_gap, _normal, _inside = vz.shape_gap(
                        torch, position, 0 * radius, vz.table_board_shape(self.half_length, self.half_width, self.board_thickness)
                    )
                    if bool((self.table_hybrid & ~over_top & (board_gap <= reach)).any()):
                        sweep_t = vz.sweep_table_box(torch, position, velocity, radius, dt, self.half_length, self.half_width, self.board_thickness)
                        tunnels[:, 1] += (sweep_t.tunnelled & self.table_hybrid).long()
                    else:
                        sweep_t = vz.sweep_table(torch, position, velocity, radius, dt, self.half_length, self.half_width)
                    candidates.append(
                        (sweep_t.contact & self.table_hybrid, sweep_t, self.table_material, self.zero_inverse_mass, SURFACE_CODES["table"], "table")
                    )
                    frame_gap, _normal, _inside = vz.shape_gap(torch, position, 0 * radius, self.frame_bounds)
                    if bool((self.table_hybrid & (frame_gap <= reach)).any()):
                        for part_name, shape in self.frame_parts:
                            part_gap, _normal, _inside = vz.shape_gap(torch, position, 0 * radius, shape)
                            if not bool((self.table_hybrid & (part_gap <= reach)).any()):
                                continue
                            sweep_f = vz.sweep_shape(torch, position, velocity, radius, dt, shape)
                            tunnels[:, 1] += (sweep_f.tunnelled & self.table_hybrid).long()
                            candidates.append(
                                (sweep_f.contact & self.table_hybrid, sweep_f, self.fixture_material, self.zero_inverse_mass, SURFACE_CODES["frame"], part_name)
                            )
                if self.any_hybrid_net:
                    sweep_n = vz.sweep_net(torch, position, velocity, radius, dt, self.net_half_thickness, self.net_half_span, self.net_height)
                    tunnels[:, 2] += (sweep_n.tunnelled & self.net_hybrid).long()
                    candidates.append((sweep_n.contact & self.net_hybrid, sweep_n, self.net_material, self.zero_inverse_mass, SURFACE_CODES["net"], "net"))
                    near_fixture = torch.zeros_like(self.net_hybrid)
                    for bounds in self.fixture_bounds:
                        bound_gap, _normal, _inside = vz.shape_gap(torch, position, 0 * radius, bounds)
                        near_fixture = near_fixture | (bound_gap <= reach)
                    if bool((self.net_hybrid & near_fixture).any()):
                        for region, shape in self.fixture_shapes:
                            sweep_f = vz.sweep_shape(torch, position, velocity, radius, dt, shape)
                            tunnels[:, 2] += (sweep_f.tunnelled & self.net_hybrid).long()
                            material_f = self.post_material if region == "post" else self.fixture_material
                            candidates.append((sweep_f.contact & self.net_hybrid, sweep_f, material_f, self.zero_inverse_mass, SURFACE_CODES["net"], region))
                # The earliest touch of the step wins; the others are re-swept next step from the new state.
                fractions = torch.stack([torch.where(use, sweep.fraction, torch.full_like(sweep.fraction, 2.0)) for use, sweep, *_ in candidates], -1)
                earliest = fractions.argmin(-1)
                imposed = torch.zeros(n, dtype=torch.bool, device=device)
                chosen = []
                for index, (use, sweep, material, inverse_mass, code, kind) in enumerate(candidates):
                    use = use & (earliest == index)
                    chosen.append((use, sweep, material, inverse_mass, code, kind))
                    imposed = imposed | use
                ids = torch.nonzero(imposed).flatten()
                if ids.numel():
                    new_position, new_velocity, new_spin = position.clone(), velocity.clone(), spin.clone()
                    # The impact meets the ball at the contact time, after the step's first f dt of flight.
                    force_0, torque_0 = vz.aero_wrench(torch, velocity, spin, self.aero_arrays)
                    acceleration_0 = self.gravity + force_0 / self.ball_arrays.mass[:, None]
                    angular_acceleration_0 = torque_0 / self.ball_arrays.inertia[:, None]
                    for use, sweep, material, inverse_mass, code, kind in chosen:
                        offset, inverse_inertia, catch = None, None, None
                        if code == SURFACE_CODES["paddle"] and sweep.offset is not None and sweep.axes is not None:
                            offset, inverse_inertia = sweep.offset, vz.surface_inverse_inertia(torch, sweep.axes, self.inverse_paddle_moments)
                        # A static touch that started inside (gravity pressing a resting ball in) is moved out by its depth.
                        centre = sweep.centre if sweep.depth is None else sweep.centre + sweep.normal * sweep.depth[:, None]
                        if kind == "net":
                            # The net mesh below the top tape catches the ball; the tape, edge and ends stay a rigid cord.
                            catch = vz.net_mesh(torch, centre, sweep.normal, self.ball_arrays.radius, self.net_half_span, self.net_height, self.net_tape_height)
                        lead = (sweep.fraction * dt)[:, None]
                        contact_velocity = velocity + acceleration_0 * lead
                        contact_spin = spin + angular_acceleration_0 * lead
                        impact = vz.resolve_impacts(
                            torch,
                            contact_velocity,
                            contact_spin,
                            self.ball_arrays,
                            sweep.normal,
                            material,
                            sweep.surface_velocity,
                            inverse_mass,
                            offset,
                            inverse_inertia,
                            catch,
                        )
                        use = use & impact.approaching
                        if impact.degenerate is not None and offset is not None:
                            painleve += int((impact.degenerate & use).sum().item())
                        sustained = torch.zeros_like(use)
                        if code in (SURFACE_CODES["table"], SURFACE_CODES["net"], SURFACE_CODES["frame"]) and impact.separation_speed is not None:
                            # Rebounding slower than the rest speed off a surface gravity presses the ball onto: a sustained
                            # contact (the reference simulation stops there; the game adapter resolves it as a constraint).
                            sustained = use & (impact.separation_speed < vz.REST_SPEED_MPS) & (sweep.normal[:, 2] > 0)
                            if catch is not None:
                                sustained = sustained & ~catch
                            if bool(sustained.any()):
                                resting = vz.resolve_impacts(
                                    torch,
                                    contact_velocity,
                                    contact_spin,
                                    self.ball_arrays,
                                    sweep.normal,
                                    vz.sustained_material(torch, material),
                                    sweep.surface_velocity,
                                    inverse_mass,
                                )
                                impact = dataclasses.replace(
                                    impact,
                                    velocity=torch.where(sustained[:, None], resting.velocity, impact.velocity),
                                    spin=torch.where(sustained[:, None], resting.spin, impact.spin),
                                )
                        force_1, torque_1 = vz.aero_wrench(torch, impact.velocity, impact.spin, self.aero_arrays)
                        start, start_velocity, start_spin = vz.imposed_state(
                            centre,
                            impact.velocity,
                            impact.spin,
                            self.gravity + force_1 / self.ball_arrays.mass[:, None],
                            torque_1 / self.ball_arrays.inertia[:, None],
                            sweep.fraction,
                            dt,
                        )
                        new_position = torch.where(use[:, None], start, new_position)
                        new_velocity = torch.where(use[:, None], start_velocity, new_velocity)
                        new_spin = torch.where(use[:, None], start_spin, new_spin)
                        before = torch.cat([centre, contact_velocity, contact_spin], -1)
                        after = torch.cat([centre, impact.velocity, impact.spin], -1)
                        continuing = sustained & (episode_code == code) & (k - episode_step <= SUSTAINED_GAP_STEPS)
                        recorded = use & ~continuing
                        record(recorded, t + sweep.fraction * dt, k, torch.full_like(contact_count, code), True, before, after, sustained)
                        closing = use & ~sustained & (episode_code == code)
                        episode_code = torch.where(sustained, torch.full_like(episode_code, code), torch.where(closing, 0 * episode_code, episode_code))
                        episode_step = torch.where(sustained, torch.full_like(episode_step, k), episode_step)
                        hybrid_counts[code] = hybrid_counts.get(code, 0) + int(recorded.sum().item())
                    mode = engine.hybrid_write
                    if mode == "pose_velocity":
                        self._write_ball_motion(new_position, new_velocity, new_spin, ids)
                    elif mode == "velocity":
                        new_position = position
                        self.ball.write_root_velocity_to_sim_index(root_velocity=torch.cat([new_velocity, new_spin], -1)[ids].contiguous(), env_ids=ids)
                    else:
                        new_position = position
                        impulse_force = self.ball_arrays.mass[:, None] * (new_velocity - velocity) / dt
                        impulse_torque = self.ball_arrays.inertia[:, None] * (new_spin - spin) / dt
                    state = torch.cat([new_position, new_velocity, new_spin], -1)
                    position, velocity, spin = state[:, 0:3], state[:, 3:6], state[:, 6:9]

            # 2. Aerodynamics from the state the step starts with.
            force, torque = vz.aero_wrench(torch, velocity, spin, self.aero_arrays)
            self.ball.permanent_wrench_composer.set_forces_and_torques_index(
                forces=(force + impulse_force)[:, None, :].contiguous(), torques=(torque + impulse_torque)[:, None, :].contiguous(), is_global=True
            )

            # 3. Paddle drive at the step start (colliding paddles only).
            if self.any_driven:
                self._drive_paddles(blade_now)

            # 4. Step.
            self.scene.write_data_to_sim()
            self.simulation.step(render=False)
            self.scene.update(dt)
            steps = k + 1
            new_state = self._read_ball()

            # 5. Native contacts and tunnelling.
            expected_dv = (self.gravity + force / self.ball_arrays.mass[:, None]) * dt
            expected_dw = (torque / self.ball_arrays.inertia[:, None]) * dt
            dv = new_state[:, 3:6] - velocity - expected_dv
            dw = new_state[:, 6:9] - spin - expected_dw
            native = (vz.norm(dv) > threshold_v) | (vz.norm(dw) > threshold_w)
            kind = self._classify(state, new_state, blade_now, blade_next)
            record(native, torch.full((n,), t, device=device, dtype=torch.float64), k, kind, False, state, new_state)
            tunnels += self._tunnels(state, new_state, blade_now, blade_next)

            # 6. Stop conditions.
            now = (k + 1) * dt
            finished = (now >= torch.minimum(self.t_end, stop_time) - 1e-9) | (new_state[:, 2] < FALLEN_Z)
            newly = finished & ~done
            done_time = torch.where(newly, torch.full_like(done_time, now), done_time)
            done = done | finished
            if (k + 1) % log_every == 0 and log_index < samples:
                log[log_index] = new_state
                log_index += 1
            state = new_state
            blade_now = blade_next
            if (k + 1) % CHECK_EVERY_STEPS == 0:
                if cancelled is not None:
                    cancelled()
                fraction_done = float(done.float().mean().item())
                if progress is not None:
                    progress(now, fraction_done)
                if fraction_done >= 1.0:
                    terminal = TERMINAL_ALL_COMPLETE
                    break
        else:
            if bool(done.all().item()):
                terminal = TERMINAL_ALL_COMPLETE
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        loop_s = time.perf_counter() - loop_started
        if terminal == TERMINAL_CAP and bool(done.all().item()):
            terminal = TERMINAL_ALL_COMPLETE
        self.timing.update(loop_s=loop_s, steps=float(steps), env_steps_per_s=n * steps / max(loop_s, 1e-9), wall_s_per_sim_s=loop_s / max(steps * dt, 1e-12))
        self.readback["hybrid_contacts"] = {SURFACE_NAMES[code]: count for code, count in hybrid_counts.items()}
        self.readback["hybrid_paddle_painleve"] = painleve
        times = np.arange(log_index, dtype=np.float64) * engine.log_interval_s
        return BatchRecord(
            name=spec.name,
            engine=engine,
            times=times,
            ball=log[:log_index].cpu().numpy(),
            contact_count=contact_count.cpu().numpy(),
            contact_time=contact_time.cpu().numpy(),
            contact_step=contact_step.cpu().numpy(),
            contact_kind=contact_kind.cpu().numpy(),
            contact_hybrid=contact_hybrid.cpu().numpy(),
            contact_before=contact_before.double().cpu().numpy(),
            contact_after=contact_after.double().cpu().numpy(),
            contact_sustained=contact_sustained.cpu().numpy(),
            tunnels=tunnels.cpu().numpy(),
            done_time=done_time.double().cpu().numpy(),
            env_origins=self.origins.double().cpu().numpy(),
            terminal=terminal,
            sim_s=steps * dt,
            steps=steps,
            timing=dict(self.timing),
            readback=dict(self.readback),
        )


def run_batch(
    spec: BatchSpec,
    *,
    log: Callable[[str], None] = print,
    fresh_stage: bool = False,
    cancelled: Callable[[], None] | None = None,
    progress: Callable[[float, float], None] | None = None,
) -> BatchRecord:
    """
    Build, run and release one batch.

    :param spec: The batch.
    :param log: Progress callback.
    :param fresh_stage: Create a new USD stage first (every build after the first in one process).
    :param cancelled: Raises to stop the batch.
    :param progress: Called every few hundred steps with ``(sim_s, finished fraction)``.
    :return: Its record.
    """

    with LabBallHarness(spec, log=log, fresh_stage=fresh_stage) as harness:
        return harness.run(cancelled=cancelled, progress=progress)


__all__ = ["CHECK_EVERY_STEPS", "FALLEN_Z", "SUSTAINED_GAP_STEPS", "LabBallHarness", "run_batch"]
