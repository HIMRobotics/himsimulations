# `lib/physics`: the parameter contract and the reference model

This domain holds the physics parameters every part of the demo shares, and
an engine-independent model of table-tennis ball physics written in numpy and
scipy only. The model is the oracle: every simulator backend (PhysX or Newton,
under Isaac Sim 6 or Isaac Lab 3) and every trajectory fit to the recorded
video is checked against it.

## Why `lib/physics`

The demo-creation contract allows an optional `lib/physics/` domain for
another solver behind a narrow adapter. Here the other solver is the
reference model, and the domain exists because no engine is trusted as
ground truth. Both harnesses must agree with one set of equations and one set
of parameters, and the fits to real camera data need the same equations. No
canonical folder owns that job: it is not scene, actor, hardware or scenario.

The seam is narrow:

- `params.BallTableParams` is the input every consumer takes.
- `FlightModel.wrench(v, omega)` is the aerodynamic force and torque an engine
  adapter applies each physics step.
- `resolve_impact(...)` is what a contact must produce.
- `simulate(...)` and `fit_flight(...)` are what a simulated run or a real
  trajectory is compared against.
- `canonical_events(...)` turns the oracle's events and the game adapter's
  events into one contact sequence, so the two can be compared.
- `blade.blade_touch(...)` is the blade's state at a paddle touch, which the
  oracle and the game adapter both resolve the impact with.

The oracle and the game adapter must describe the same touch in the same way,
so the event mapping and the blade contact state live here, where both sides
import them and neither imports the other.

Nothing here imports `pxr`, `omni`, `carb` or `isaacsim`. `params.py` imports
only numpy and pydantic, so the cheap `main.py --help` path never loads scipy.

## Layout

| Module | Contents |
|---|---|
| `params.py` | `BallTableParams` and its frozen pydantic sub-models, flat parameter names and their provisional defaults and ranges, the `priors.json` loader, randomised sampling |
| `data/priors.json` | Cited priors, keyed by flat name. Written by the calibration work, not by this module |
| `reference/flight.py` | Flight ODE, RK4 (scalar and vectorised), event detection, `propagate` |
| `reference/impact.py` | Rigid-ball impact on a flat, possibly moving surface of finite effective mass |
| `reference/surfaces.py` | Table top, net, floor and scripted-paddle contact geometry |
| `reference/simulate.py` | Event-driven trajectories and the drop-test restitution inversion |
| `reference/paddle_track.py` | `PaddleTrack`: the game's per-tick paddle segments as a scripted paddle |
| `reference/blade.py` | The blade's touch condition, touch time and contact state, shared with the game adapter |
| `reference/inverse.py` | The paddle impact inverted: the face velocity that sends a ball off with a wanted velocity |
| `canonical_events.py` | One canonical contact sequence for oracle and game-adapter event lists |
| `vectorized/` | The oracle's ball physics in torch, batched over environments on a GPU (optional; needs torch) |
| `engine/lab/robot_ball.py` | The robot's paddle and the GPU ball coupled both ways, once per PhysX step (Isaac Lab) |
| `reference/projection.py` | OpenCV pinhole cameras and robust physics-constrained flight fits |

## Conventions

- **Units.** SI throughout: m, kg, s, rad. Every docstring states its units.
- **Table frame (the world).** Z-up. The origin is the centre of the playing
  surface and +X runs along the table length. The table top is the rectangle
  `|x| <= 1.37`, `|y| <= 0.7625` in the plane `z = 0`. The net is the plane
  `x = 0`, and the floor is `z = -table.height`.
- **State vector.** 9 floats: position (m), velocity (m/s), then angular
  velocity (rad/s), all in the table frame.
- **Spin sign.** Topspin on a ball moving along +X is angular velocity along
  +Y: the top of the ball moves forward. A ball rolling along +X has
  `omega_y = v_x / R`.
- **Paddle.** `PaddlePose.rotation` has the blade axes as columns: local X
  points towards the handle, and local Z is the forehand face normal. The pose
  is the blade centre on its mid-plane. Both faces collide. The game's blade
  axes (`PaddleSegment`, `quat_matrix`, and `reference/blade.py`) are X across
  the blade, Y towards the tip and Z the forehand normal; a `PaddlePose`
  rotation is `[-Y, X, Z]` of them.
- **Cameras.** OpenCV convention: camera +X right, +Y down, +Z forward, and
  `x_cam = R x_world + t`. Detection times are on one master clock.

## Parameters

Each scalar has a flat name. `data/priors.json` is keyed by these names, and
run records and randomisation use them too. The cited values below come from
that file; its `source`, `confidence` and `notes` fields carry the citations,
and `docs/physics-priors.md` explains them. Any name the file lacks falls back
to its fallback. A fallback's provenance is either the literal string
`provisional` (a placeholder) or a `derived: ...` statement (computed here
from cited inputs). `load_priors().provisional_names()` lists the
placeholders still in use. With the committed file these are the net-mesh
and fixture parameters below.

The three paddle inertia moments are derived, not cited. They treat the
cited 157 x 150 x 13.6 mm head as a uniform slab at the cited 0.17 kg
effective mass, about the blade centre (`I = m (a^2 + b^2) / 12`), which is
the slab the calibration harness uses. Their ranges are judgement: a grip
can only add inertia. They are not in `priors.json`; if they move there,
the keys are `paddle_inertia_xx`, `paddle_inertia_yy` and
`paddle_inertia_zz`, in `kg m^2`.

The net-mesh parameters are judgement, stated as such: `net_tangential_damping`
(0.8, range 0.5 to 0.95), `net_spin_damping` (0.7, range 0.4 to 0.95) and
`net_tape_height` (15 mm, range 10 to 15 mm). The literature has no
ball-net measurement; the priors list the net among the quantities "not
pinned down". ITTF T2 allows a top tape of up to 15 mm, per the priors'
`net_thickness` notes. The mesh's normal restitution is the cited
`net_restitution` (0.2).

The table board and the net clamps are judgement too:
- **`table_board_thickness`** (25 mm) is derived from the table asset, whose top mesh spans 0.735 to 0.76 m above the floor.
- **The board's ends, sides, edges and underside** use the table's own law (`table_contact`). No measurement of a ball on a table's edge or side exists; it is the same chipboard board, struck in-plane where it is stiffer. The top's e already sits at the ball's own rigid-target restitution.
- **The clamps** use `fixture_contact`: `fixture_restitution` 0.85 (range 0.75 to 0.92) and `fixture_friction` 0.25 (range 0.15 to 0.40). The game adapter's `FIXTURE_CONTACT` is painted metal or hard plastic, below the ball's 0.92 to 0.95 rigid-target restitution.
- **The posts** are steel and use `post_contact`, which is cited.
  - `post_restitution` is 0.946, range 0.921 to 0.970. It comes from ITTF T3 B.7: a non-celluloid ball dropped 305 mm onto an S235 steel block must rebound 240 to 265 mm. Ignoring drag that is 0.887 to 0.932; inverted through this model with `restitution_from_drop_test` at the cited C_D 0.50 and rho 1.19, it is 0.921 to 0.970, with 0.946 for the mid-band 252.5 mm.
  - A 15 mm post may flex where a 20 mm block does not, so this is an upper bound.
  - `post_friction` is 0.20, range 0.10 to 0.35, judgement.
  - Both are in `priors.json`, with the source.
- **The undercarriage** (apron, legs, stretchers, struts, feet and casters) uses `fixture_contact`, painted steel. This is judgement.
- **The room** (walls, ceiling and solid furniture) uses `wall_contact`: `wall_restitution` 0.80 (range 0.70 to 0.90) and `wall_friction` 0.35 (range 0.20 to 0.50).
  - Both are judgement, in `priors.json`, and equal to the game's `WALL_CONTACT`.
  - The restitution sits below the ball's rigid-target bound (Rémond 2022), because a stud wall and sheet-metal cabinets dissipate more.
- **Rolling resistance.** `floor_rolling_resistance` is 0.010 (range 0.003 to 0.030) and `table_rolling_resistance` is 0.005 (range 0.002 to 0.015).
  - Both are judgement, in `priors.json`. No measured value for a table-tennis ball was found.
  - Hard balls on hard, smooth surfaces roll at well below 0.001, and high-bounce balls roll more easily than low-bounce ones (Cross 2015, 2016). A thin plastic shell and a concrete floor's roughness raise the value.
  - The convention is `C_rr = F/N`: the centre force that keeps the ball rolling, or equivalently the couple `C_rr N R`. A free ball therefore decelerates at `C_rr g / (1 + k)`.
  - With drag, a ball rolled at 1 m/s on the floor stops after about 5 m.

| Name | Field | Unit | Nominal | Range | Origin | Fallback | Fallback basis |
|---|---|---|---|---|---|---|---|
| `ball_radius` | `ball.radius` | m | 0.02005 | 0.02 to 0.02025 | cited | 0.02 (0.02 to 0.0202) | 40 mm ball; 40+ balls measure up to about 40.4 mm |
| `ball_mass` | `ball.mass` | kg | 0.0027 | 0.00267 to 0.00277 | cited | 0.0027 (0.00267 to 0.00277) | 2.7 g nominal, 2.67-2.77 g tolerance band |
| `ball_shell_thickness` | `ball.shell_thickness` | m | 0.0005 | 0.00038 to 0.00055 | cited | 0.0004 (0.0003 to 0.0006) | placeholder wall thickness; moves I/(mR^2) between 0.65 and 0.66 |
| `air_density` | `aero.air_density` | kg/m^3 | 1.19 | 1.1 to 1.23 | cited | 1.2 (1.12 to 1.25) | indoor air, 15-30 C at 0-600 m altitude |
| `gravity` | `aero.gravity` | m/s^2 | 9.80665 | fixed | cited | 9.81 | standard gravity; not randomised |
| `drag_coefficient` | `aero.drag_coefficient` | 1 | 0.5 | 0.4 to 0.58 | cited | 0.45 (0.35 to 0.6) | subcritical smooth sphere; placeholder until fitted |
| `drag_spin_slope` | `aero.drag_spin_slope` | 1 | 0 | fixed | cited | 0 | spin-dependent drag off until a prior exists |
| `lift_slope` | `aero.lift.slope` | 1 | 1.2 | 0.9 to 1.6 | cited | 1 (0.5 to 1.5) | S/(1+2S) form; placeholder |
| `lift_max` | `aero.lift.c_max` | 1 | 0.4 | 0.28 to 0.5 | cited | 0.5 (0.3 to 0.6) | S/(1+2S) form; placeholder |
| `spin_decay_coefficient` | `aero.spin_decay_coefficient` | 1 | 0.02 | 0.007 to 0.055 | cited | 0.02 (0 to 0.1) | about 8 % spin loss per second at 10 m/s; placeholder, unobservable here |
| `table_length` | `table.length` | m | 2.74 | 2.735 to 2.745 | cited | 2.74 | regulation table |
| `table_width` | `table.width` | m | 1.525 | 1.522 to 1.528 | cited | 1.525 | regulation table |
| `table_height` | `table.height` | m | 0.76 | 0.757 to 0.763 | cited | 0.76 | regulation table |
| `table_board_thickness` | `table.board_thickness` | m | 0.025 | fixed | derived | 0.025 | the table asset's top board |
| `net_height` | `net.height` | m | 0.1525 | 0.148 to 0.157 | cited | 0.1525 | regulation net |
| `net_overhang` | `net.overhang` | m | 0.1525 | 0.1505 to 0.1545 | cited | 0.1525 | regulation net |
| `net_thickness` | `net.thickness` | m | 0.003 | 0.002 to 0.006 | judgement (priors file) | 0.003 | placeholder mesh-plus-cord thickness |
| `net_tape_height` | `net.tape_height` | m | 0.015 | 0.01 to 0.015 | provisional | 0.015 (0.01 to 0.015) | judgement: ITTF T2 allows a top tape up to 15 mm (per priors notes) |
| `paddle_blade_half_length` | `paddle.blade_half_length` | m | 0.0785 | 0.075 to 0.08 | cited | 0.0785 | placeholder 157 mm blade |
| `paddle_blade_half_width` | `paddle.blade_half_width` | m | 0.075 | 0.0725 to 0.077 | cited | 0.075 | placeholder 150 mm blade |
| `paddle_thickness` | `paddle.thickness` | m | 0.0136 | 0.011 to 0.0146 | cited | 0.014 | placeholder: about 6 mm wood plus two 4 mm rubbers |
| `paddle_effective_mass` | `paddle.effective_mass` | kg | 0.17 | 0.1 to 0.4 | cited | 0.3 (0.15 to 1) | placeholder: 0.17-0.19 kg paddle plus partial hand coupling |
| `paddle_inertia_xx` | `paddle.inertia_xx` | kg m^2 | 0.0003214 | 0.00025 to 0.00065 | derived | 0.0003214 (0.00025 to 0.00065) | slab about the handle axis |
| `paddle_inertia_yy` | `paddle.inertia_yy` | kg m^2 | 0.0003518 | 0.00027 to 0.0007 | derived | 0.0003518 (0.00027 to 0.0007) | slab about the cross-blade axis |
| `paddle_inertia_zz` | `paddle.inertia_zz` | kg m^2 | 0.0006679 | 0.0005 to 0.0013 | derived | 0.0006679 (0.0005 to 0.0013) | slab about the face normal |
| `table_restitution` | `table_contact.restitution` | 1 | 0.94 | 0.92 to 0.955 | cited | 0.94 (0.92 to 0.955) | as priors.json; drag-corrected drop test (0.876 ignores drag) |
| `table_friction` | `table_contact.friction` | 1 | 0.25 | 0.15 to 0.4 | cited | 0.25 (0.1 to 0.5) | placeholder |
| `table_tangential_restitution` | `table_contact.tangential_restitution` | 1 | 0 | fixed | cited | 0 | rigid Coulomb table |
| `rubber_restitution` | `rubber_contact.restitution` | 1 | 0.72 | 0.58 to 0.86 | cited | 0.85 (0.7 to 0.92) | placeholder |
| `rubber_friction` | `rubber_contact.friction` | 1 | 1 | 0.6 to 1.5 | cited | 0.8 (0.4 to 1.2) | placeholder: inverted rubber is grippy |
| `rubber_tangential_restitution` | `rubber_contact.tangential_restitution` | 1 | 0.7 | 0.5 to 0.85 | cited | 0.7 (0.5 to 0.85) | as priors.json: rubber reverses contact slip |
| `rubber_normal_spin_retention` | `rubber_contact.normal_spin_retention` | 1 | 0.805 | 0.7 to 0.95 | cited | 1 | rigid model: no torque about the face normal |
| `net_restitution` | `net_contact.restitution` | 1 | 0.2 | 0.05 to 0.45 | judgement (priors file) | 0.2 (0.05 to 0.4) | placeholder: the net is soft |
| `net_friction` | `net_contact.friction` | 1 | 0.3 | 0.1 to 0.6 | judgement (priors file) | 0.3 (0.1 to 0.6) | placeholder |
| `net_tangential_damping` | `net_contact.tangential_damping` | 1 | 0.8 | 0.5 to 0.95 | provisional | 0.8 (0.5 to 0.95) | judgement: no ball-net measurement exists; the mesh catches the ball |
| `net_spin_damping` | `net_contact.spin_damping` | 1 | 0.7 | 0.4 to 0.95 | provisional | 0.7 (0.4 to 0.95) | judgement: no ball-net measurement exists |
| `floor_restitution` | `floor_contact.restitution` | 1 | 0.88 | 0.75 to 0.94 | cited | 0.85 (0.7 to 0.92) | placeholder: concrete |
| `floor_friction` | `floor_contact.friction` | 1 | 0.35 | 0.2 to 0.6 | judgement (priors file) | 0.3 (0.15 to 0.6) | placeholder |
| `fixture_restitution` | `fixture_contact.restitution` | 1 | 0.85 | 0.75 to 0.92 | provisional | 0.85 (0.75 to 0.92) | judgement: no measurement; below the ball's 0.92-0.95 rigid-target restitution (Remond 2022) |
| `fixture_friction` | `fixture_contact.friction` | 1 | 0.25 | 0.15 to 0.4 | provisional | 0.25 (0.15 to 0.4) | judgement: painted metal or hard plastic |
| `post_restitution` | `post_contact.restitution` | 1 | 0.946 | 0.921 to 0.97 | cited | 0.946 (0.921 to 0.97) | ITTF T3 steel-block bounce 240-265 mm from 305 mm, drag-corrected through this model (C_D 0.50, rho 1.19) |
| `post_friction` | `post_contact.friction` | 1 | 0.2 | 0.1 to 0.35 | judgement (priors file) | 0.2 (0.1 to 0.35) | judgement: ABS on painted or polished steel |
| `wall_restitution` | `wall_contact.restitution` | 1 | 0.8 | 0.7 to 0.9 | judgement (priors file) | 0.8 (0.7 to 0.9) | judgement: below the ball's rigid-target bound (Remond 2022); a stud wall and sheet-metal cabinets dissipate more |
| `wall_friction` | `wall_contact.friction` | 1 | 0.35 | 0.2 to 0.5 | judgement (priors file) | 0.35 (0.2 to 0.5) | judgement: painted or coated surface |
| `floor_rolling_resistance` | `floor_contact.rolling_resistance` | 1 | 0.01 | 0.003 to 0.03 | judgement (priors file) | 0.01 (0.003 to 0.03) | judgement: no measurement for a hollow plastic ball on concrete; high-bounce balls roll easily (Cross 2015) |
| `table_rolling_resistance` | `table_contact.rolling_resistance` | 1 | 0.005 | 0.002 to 0.015 | judgement (priors file) | 0.005 (0.002 to 0.015) | judgement: smoother and stiffer than the floor |

The priors file also carries entries that are not scalars:

- **`lift_form`** takes one of `rational`, `linear` or `constant`.
- **Speed curves.** `<surface>_<restitution|friction>_vs_speed` (or `_curve`)
  takes `{"speeds": [...], "values": [...]}`: a piecewise-linear coefficient
  against the normal approach speed in m/s, held constant past both ends. The
  committed file has curves for the table, the rubber and the floor.
- **How curves and scalars combine.** A curve never replaces its scalar. On
  load, each curve's `reference` is set to the nominal value of its scalar,
  and the material evaluates `curve(v) * scalar / reference`, clipped to
  [0, 1]. At the nominal scalar this is exactly the curve; a randomised
  `table_restitution` moves the whole curve, so "vary bounciness" still
  works. `nominal(speed_curves=False)` gives constant coefficients, for
  comparison against an engine without speed-dependent restitution.

**Priors file format.** Entries may sit at the top level, under a
`parameters` or `priors` key, or in a list of objects that each carry a
`name`. Each entry is either a bare number or an object with these fields:

- `value`, or one of `nominal`, `default`, `mean` or `median`.
- Optionally a range: `range: [low, high]`, `low`/`high`, or `min`/`max`.
- Optionally a `unit`. `mm`, `cm` and `g` convert to SI; any other mismatch
  is an error.
- Optionally a `source` or `citation`.

An entry with no range is fixed at its value. A value outside its own range
is an error, and keys the contract does not know are listed in
`PriorTable.unrecognised`.

**Randomisation.** `sample_params(rng, names=None)`, or
`load_priors().sample(...)`, draws each named parameter uniformly within its
range. Every other parameter stays nominal. The draw is reproducible from the
generator's seed, and the result is validated, so an impossible combination
raises an error. Ranges are uniform bounds, not standard deviations.

## Equations

### Ball

The ball is a uniform spherical shell of outer radius `R`, inner radius
`a = R - t` and mass `m`, with the exact inertia

```
I = (2/5) m (R^5 - a^5) / (R^3 - a^3)          (k = I / (m R^2): 2/3 thin, 2/5 solid)
```

At the cited 0.5 mm wall, `k = 0.6503`; at a 0.4 mm wall it is 0.6535. The
code evaluates the factored form so a very thin shell keeps its precision.

### Flight (`reference/flight.py`)

With `A = pi R^2`, `k_a = rho A / (2 m)`, `omega_perp` the spin component
perpendicular to the velocity and `S = R |omega_perp| / |v|`:

```
dp/dt     = v
dv/dt     = -g z_hat
            - k_a C_D(S) |v| v                                      drag
            + k_a C_L(S) |v|^2 (omega_perp x v_hat) / |omega_perp|  Magnus
domega/dt = -(C_tau rho A R^2 / (2 I)) |v| omega                     spin decay
C_D(S)    = C_D0 + (dC_D/dS) S
C_L(S)    = slope S / (1 + slope S / c_max)                         rational (default)
          | min(slope S, c_max)                                     linear
          | c_max                                                   constant
```

The Magnus force is therefore `F_M = 1/2 rho A C_L(S) |v|^2 e_M`, with unit
direction `e_M = (omega_perp x v_hat) / |omega_perp|`. When spin is
perpendicular to velocity, `e_M` equals `omega_hat x v_hat`.

Using the perpendicular spin means spin about the direction of travel
neither lifts the ball nor raises `S`. With a linear `C_L = C_M S` the force
is exactly the form `F_M = 1/2 rho A R C_M (omega x v)` (Tebbe's `C_M`).

**This lift model is the source of truth for Magnus.** The cited curve is the
rational form with `slope = 1.2` and `c_max = 0.40`, fitted to Sony AI's
measured data (see `docs/physics-priors.md`). Other conventions differ by
factors of 2 to 3. The Japanese papers, Sony AI and MuJoCo write
`F = C_M rho V (omega x v)` with `V` the ball's volume, which is
`C_L = (8/3) C_M S`. MuJoCo's default `C_M = 1`, used unmeasured by DeepMind
in 2024, is therefore `C_L = 2.67 S`: 2.7 times stronger than any measured
table-tennis ball at small `S`, and up to 10 times stronger near `S = 1`.
MuJoCo's "blunt drag" also leaves out the 1/2, so its 0.235 is a standard
`C_D` of 0.47.

An engine adapter must therefore switch off any engine-native fluid or
Magnus model and apply `FlightModel.wrench` instead. Any engine coefficient
has to be converted to this convention and checked against `wrench` before
it is trusted. The provisional fallback (`slope = 1`, `c_max = 0.5`, that is
`S / (1 + 2S)`) is uncited and 10 to 40 % above Sony AI's data at
`S >= 0.8`.

The spin-decay torque is `tau = -C_tau (1/2) rho A R^2 |v| omega`. The air is
still: there is no wind term.

**Integrator.** Classical RK4 at a fixed step, 1e-4 s (10 kHz) by default,
on a grid anchored at the segment start. Measured convergence is fourth
order. Over 0.8 s of a 12 m/s spinning flight, a 1 ms step differs from
1e-5 s by less than a micron.

**Events.** Each surface supplies a signed gap. An event fires when the gap
goes from `>= 0` at a step's start to `< 0` at its end. Its root is
bracketed to 1e-10 s by bisection, re-taking one RK4 step from the step start
for each trial. The event state is the last trial still on the non-negative
side, so it is exactly a state the integrator would have produced.

**Speed without changing answers.** `FlightModel.rk4_step` is the RK4 step
with the derivative inlined: the same operations in the same order, so bit for
bit the same state. A static surface's gap is evaluated only when the ball
could have changed its sign:
- each static surface has a `clearance(state, gap)`, the distance the centre
  can move without the sign changing (`|gap|` for a distance);
- the integrator sums the ball's step-end travel and evaluates the gap again
  only when that sum reaches the clearance;
- the event logic reads only signs at step ends, so the steps, roots and
  states are unchanged.

On 10,376 planner calls and 5,499 calibration trials the trajectories are
bit-identical, and the planner's calls run 2.4 times faster.

**Adaptive steps (opt-in, `max_step`).** Near anything the ball could touch,
steps are `dt`. A longer step, up to `max_step`, is taken only where every
watched surface and blade is out of the ball's reach over it, plus 1 mm. No
contact can start inside a long step, so touches are resolved at `dt`.
`dt = 1e-4` with `max_step = 2e-3` gives the same events as a fixed `1e-4`
step on every test corpus, within about 1e-9 m. The only exception is a
net-post graze that amplifies differences, which agrees to 3e-5 m. It runs at
about the cost of a fixed 1 ms step.

**Graze search (opt-in, `graze`).** A fixed step misses a touch that enters
and leaves between two step ends, which happens when the ball penetrates less
than about `(v dt)^2 / (8 R)`. At 1 ms that is a 1 mm graze of the cord one
time in ten. With `graze`, a step that ends clear of a surface but came within
its travel of it is searched for a dip below zero (three-point parabolic
steps), and the root is bracketed before the dip. With it, no graze from 1 µm
to 1 mm deep was missed.

### Impact (`reference/impact.py`)

The surface has unit normal `n` (pointing from the surface into the ball),
velocity `u_s` at the contact point and effective mass `M`. `M` is infinite
for the table, net and floor, and `paddle.effective_mass` for a paddle.

```
r_c = -R n,  u = v + omega x r_c - u_s,  u_n = u . n (< 0 approaching),  u_t = u - u_n n
1/m_n = 1/m + 1/M,   1/m_t = 1/m + 1/M + 1/(k m)

normal (Newton):        P_n = -(1 + e) m_n u_n
tangential (Brach):     P_stick = -(1 + e_t) m_t u_t
                        stick if |P_stick| <= mu P_n, else slip: P_t = -mu P_n u_t / |u_t|
ball update:            v' = v + (P_n n + P_t) / m,   omega' = omega + (r_c x P_t) / I - (1 - s_n)(omega . n) n
surface update:         u_s' = u_s - (P_n n + P_t) / M
energy change:          1/2 m_n (u_n'^2 - u_n^2) + 1/2 m_t (|u_t'|^2 - |u_t|^2) - 1/2 I (1 - s_n^2)(omega . n)^2 <= 0
rolling threshold:      mu_crit = (1 + e_t) m_t |u_t| / P_n   (= (k/(1+k)) |u_t| / ((1+e)|u_n|) for a fixed surface)
```

The normal and tangential problems decouple because the normal passes
through the centre of a sphere. For the same reason Newton's, Poisson's and
Stronge's restitution agree, and the slip direction cannot change during the
impact.

With `e_t = 0` this is the rigid Coulomb (Routh) model: a sticking contact
leaves rolling. With `e_t > 0` the contact slip reverses to `-e_t u_t`. That
is the tangential-compliance (Garwin, Cross) variant, which returns overspin
from a grippy rubber. The table is rigid (`e_t = 0`); the cited rubber value
is `e_t = 0.70`.

No impulse has a moment about `n`, so the rigid model keeps the spin about
the contact normal. `normal_spin_retention` `s_n` scales that component to
`s_n (omega . n)`, standing in for pivoting friction. The cited value for
rubber is 0.805 (Sony AI's `e_s`); every other surface keeps 1. This adds
`-1/2 I (1 - s_n^2) (omega . n)^2` to the energy change.

`e` and `mu` may follow speed curves keyed on the approach speed `-u_n`,
scaled by their scalars (see Parameters). A contact that is not approaching
(`u_n >= 0`) is left unchanged.

**Off-centre impacts on a rigid paddle.** `resolve_impact` takes an optional
`contact_offset` `r`, the contact point relative to the paddle's centre of
mass, and `surface_inertia` `I_p`, its inertia tensor. With both, the paddle
is a free rigid body of mass `M` instead of a point mass:

```
K_s = (1/M) 1 + [r]x^T I_p^-1 [r]x                (contact-point compliance; (1/M) 1 when r = 0)
1/M_eff = n^T K_s n = 1/M + (r x n)^T I_p^-1 (r x n)
stick:  P = (K_ball + K_s)^-1 ( -(1+e) u_n n - (1+e_t) u_t )      if inside the friction cone
slip:   P = P_n (n - mu u_t_hat),  P_n = -(1+e) u_n / (n^T (K_ball + K_s)(n - mu u_t_hat))
paddle: dV = -P/M,  d omega_p = -I_p^-1 (r x P)
```

With the derived inertia, a 0.17 kg paddle meets the ball with 0.077 kg 5 cm
along the blade and 0.073 kg 5 cm across it, which reproduces calibration's
numbers. An in-plane offset couples the normal and tangential impulse. The
coupling is about 1 kg^-1 against the ball's own 370 kg^-1, so it is tiny.
Over 2000 random paddle-like impacts the tests hold linear and angular
momentum to 1e-12 and confirm energy never rises.

Without an offset the law is the point-mass closed form above, unchanged.
Scripted paddles in `simulate` are rigid bodies by default:
`ScriptedPaddle.rigid_body`, centre of mass at the blade centre, inertia
from `paddle.inertia_*`. A paddle with infinite mass stays immovable.

**The net mesh: a catch law.** A rigid net face keeps the ball's motion
along the face and, through friction, turns spin into it. In game 7 of the
game videos that turned −200 rad/s of spin into +1.27 m/s of lift (0.79 to
2.05 m/s), and dead balls dribbled beside the net for 1 to 2.6 s
(`game-fixes` report, section 1.4). A real mesh wraps round the ball and
catches it. Under `ContactMaterial.law = "catch"`, against an immovable
surface:

```
v_r'   = -e (v_r . n) n + (1 - d_t) v_r,t       (v_r = v - u_s; v_r,t its part along the surface)
omega' = (1 - d_w) omega
dKE    = 1/2 m (e^2 - 1) v_n^2 + 1/2 m ((1 - d_t)^2 - 1) |v_t|^2 + 1/2 I ((1 - d_w)^2 - 1) |omega|^2 <= 0
```

There is no contact-point friction, so spin cannot become lift. The law is
phenomenological: the net takes up the momentum and angular momentum. With
the defaults (`e = 0.2`, `d_t = 0.8`, `d_w = 0.7`), the game-7 ball (5 m/s
into the face, rising at 0.79 m/s) leaves the mesh at 1 m/s, rising at
0.16 m/s instead of 2.05 m/s. In the simulated replay it lands 0.15 m from
the net on its own side and never rises to the cord.

**Apparent and intrinsic rubber restitution.** The cited rubber curves were
measured on robot-held rackets, so they are apparent values. With the
finite `paddle.effective_mass` (0.17 kg), the simulated rebound comes out
about 0.02 to 0.03 lower than measured. A paddle with `effective_mass = inf`
reproduces the measurements directly. Which of the two to use is a
calibration decision.

### Surfaces (`reference/surfaces.py`)

- **Table top.** Gap `z - R`. The root is a contact only when the point below
  the ball is on the table rectangle. Otherwise `simulate` records
  `table_plane_outside`, unless the centre is within one radius of the board's
  top edge; that ball is the board's to resolve. Contacts have region `top`.
- **Table board** (`TableBoard`). This is the box `|x| <= length/2`,
  `|y| <= width/2`, `-board_thickness <= z <= 0`: the game adapter's
  `table_board`, swept by its `sweep_table_box`.
  - Its gap is the distance from the ball centre to the box minus `R`, so
    ends, sides, edges, corners and the underside are covered. An edge or
    corner deflects the ball by its geometry: the normal runs from the
    closest point to the centre.
  - Over the playing surface's footprint the gap is infinite, because the
    top face belongs to `TableTop`. The top plane is therefore unchanged, and
    no top bounce gives two contacts.
  - The material is `table_contact`.
  - Each contact's region comes from
    `classify_table_touch(normal)`:
    - `top`: normal exactly +z;
    - `edge`: the top edge, normal tilted up;
    - `side`: an end or side face, horizontal normal;
    - `underside`: normal pointing down.
  - `is_edge_ball(region)` follows the Laws: `top` and `edge` count, while
    `side` and `underside` do not.
- **Net posts and clamps** (`NetFixture`, `net_fixtures(params)`). They match
  `scene/colliders.py`:
  - **Posts** are vertical cylinders of radius 7.5 mm, standing 4.5 mm beyond
    each net end, from z −0.03 to 0.1585 m. They use `post_contact` (steel).
  - **Clamps** are boxes at `|x| <= 0.016`, `0.7275 <= |y| <= 0.927`,
    `-0.089 <= z <= 0.016`. They use `fixture_contact`.
  - All are `kind = "net"`, because the Laws' net assembly includes them,
    with region `post` or `clamp` and names such as `net_post_pos_y`.
- **Net.** The box `|x| <= thickness/2`, `|y| <= width/2 + overhang`,
  `0 <= z <= height`. The gap is the distance from the ball centre to the box
  minus `R`, so the net face, the top cord and the ends are all covered. The
  normal runs from the closest point to the ball centre. Each contact has a
  region (`Contact.region`):
  - **`mesh`**: on a face, within the span, below the top tape (`z <
    height - tape_height`). It uses `net_contact` under the catch law.
  - **`cord`**: the tape band, the top edge and the ends. It uses
    `net_contact` as a rigid contact: the cord behaviour, unchanged.

  `Net.material_at(point)` and `Net.region(point)` give the same answer to an
  engine adapter, as does `BallTableParams.net_mesh_contact` for the mesh
  law.
- **Floor.** Gap `z + table.height - R`, unbounded.
- **Room** (`reference/room.py`, `Room`). This is opt-in: `simulate(..., room=True)`.
  - It holds the published room's four walls and ceiling: interior x from -5 to 5 m, y from -4.7 to 3.3 m, 3 m high, walls 0.10 m thick.
  - It also holds the solid furniture along the +Y wall: two cabinets, lockers, a pedestal and a cart.
  - These are the game adapter's `ROOM_BOXES`, pinned by a test.
  - Contacts are kind `wall`, with the box's name as region. The material is `wall_contact`.
- **Undercarriage** (`reference/undercarriage.py`, `TableFrame`). This is the
  table asset's steel frame, converted to the table frame from
  `reconstruction/scene/config.py` and `build_scene.py`. Per half it has:
  - **Apron:** two long rails and two cross rails, 25 mm tube, 55 mm deep
    under the board, inset 6 cm.
  - **Outer legs:** a pair at `|x| = 1.19`, `|y| = 0.62`, on 22 mm feet, with
    a stretcher 16 cm above the floor.
  - **Centre legs:** a pair at `|x| = 0.36`, `|y| = 0.52`, on 40 mm casters,
    with a stretcher.
  - **Struts and spine:** two diagonal struts, and a spine joining the leg
    pairs.

  That makes 38 rigid primitives. The asset's 8- to 16-sided cylinders are
  exact cylinders here, within about 1 mm, and the struts are capsules. The
  gap is the smallest over the parts, with bounding-box pruning. Contacts
  are kind `frame`, so nothing reads a leg as a table bounce, and their
  region names the part. The material is `fixture_contact` (judgement).
- **Paddle.** The gap is `blade.blade_gap`,
  `max(|n . (c - p)| - (R + thickness/2), (e - 1) min(a, b))`, against the
  scripted pose at time `t`, where `e` is the touching point's elliptical
  radius in the outline's semi-axes. It is negative exactly when the ball is
  within reach of the face plane and the touching point (the centre less `R`
  along the normal) is inside the outline. A touch therefore comes either
  through the face plane (a face touch) or through the outline while the ball
  is within reach of the plane (a rim touch), as the game adapter's sweep takes
  them.
  - Until 2026-10-07 the gap was the face plane alone, and a ball that reached
    the plane beside the blade passed through its rim.
  - The contact frame is `blade.blade_contact`. The surface velocity is
    `v_p + omega_p x (q - p)`, and the contact passes the offset `q - p` and
    the world-frame inertia `R diag(I) R^T`, so an off-centre hit meets less
    paddle.
  - The script carries on after the hit, as a stiffly driven paddle would;
    mass and inertia enter only the impulse.
  - For a `PaddleTrack`, a ball beyond a sphere bounding the blade in the
    segment in force gets a positive gap without evaluating the pose. Every
    touch lies inside that sphere, so no answer changes.
  - `ScriptedPaddle.windows` (opt-in) watches the paddle only inside given
    intervals. `paddle_track.contact_windows` builds them around planned
    contacts; a touch outside every window is not detected.

### Simulation (`reference/simulate.py`)

The simulation integrates flight until an event, applies the impact, and
repeats. It stops at the first terminal event:

- a contact kind or paddle name listed in `terminal_events` (default: the floor);
- `table_plane_outside`, if that is listed;
- `rest`: a static-surface rebound below `rest_speed` (0.05 m/s) on a
  surface gravity presses the ball onto (an upward contact normal). A slow
  rebound from the vertical net mesh is not rest: the ball falls away;
- `out_of_bounds`, `max_time` or `max_events`.

**Sustained rolling** (`reference/rolling.py`). With `rolling=True` (the
default), a ball that rebounds from the floor or the playing surface slower
than `rest_speed` does not stop the run. It gets a `roll` event and stays on
the plane (`v_z = 0`, normal force `m g` adjusted for aerodynamic lift or
downforce):
- **Sliding.** Coulomb friction acts at the contact. Sliding ends three
  steps' worth of slip before zero, `3 mu g (1 + 1/k) dt` (2.6 mm/s on the floor
  at 0.1 ms), and a stick impulse removes the rest.
  - A fixed-step integrator chatters on the friction's discontinuity near
    zero, so the margin is needed.
  - A ball that arrives with a slip already inside the margin is stuck at once.
  - The stick never adds energy and conserves angular momentum about the
    contact point, so the rolling state does not depend on when it is applied:
    a spinless ball rolls on at `v0 / (1 + k)`.
  - This was a defect found by game-fixes: a slip between 1e-6 m/s and the
    margin once slid forever with no rolling resistance. Game 6's event 632 is
    now a regression test.
- **Rolling.** The motion obeys
  `m (1 + k) dv/dt = F_air,h - C_rr N v_hat + (tau_air,h x n) / R`.
  - In vacuum the ball stops after `v0^2 (1 + k) / (2 C_rr g)`.
  - With air the decay is `dv/dt = -(a + b v^2)`, where drag, Magnus
    downforce and spin decay make up `b`. The tests match the closed-form
    stopping distance to 1e-5.
- **Stop.** The ball is at `rest` below 1 mm/s.
- **Leaving.** Rolling off the table hands the ball back to flight. Any
  contact with another surface (a wall, a leg, the net) returns it to flight
  through the impact law.
- **Edge pivot** (`pivot_off_edge`). A ball crossing the edge slower than
  `sqrt(g R)` (0.44 m/s) pivots about it first, with a `pivot` event.
  - Its centre turns about the edge line: `m R theta'' = m g sin(theta) - F`
    and `I Omega' = F R`.
  - It sticks while the needed friction is under `mu N` and slides otherwise,
    and it leaves when `N = m (g cos(theta) - R theta'^2)` vanishes.
  - Without slip this is `cos(theta_r) = ((1 + k) v^2/(g R) + 2)/(3 + k)`; the
    table's μ of 0.25 makes it slide off sooner.
  - Air and rolling resistance are left out of this phase, which lasts at most
    a few tenths of a second.
  - It agrees with the game adapter's pivot to 0.2 ms and 0.2 mm at 0.2–0.4 m/s.
    Without it, the oracle landed up to 53 ms sooner and 3.9 cm nearer.
  - `edge_pivot=False` releases the ball as its centre passes the edge.

A slow rebound from another upward-facing surface (the net's top or cord, a
frame rail) is `rest` at once, as before. A slow rebound from the table
board's edge is not, since that edge cannot hold a ball rolling off the
table. `rolling=False` keeps the earlier rule: any slow rebound
from an upward-facing surface is `rest`.

`simulate(..., max_step=..., graze=...)` takes the opt-in adaptive steps and
graze search of `reference/flight.py`. `simulate_many(states, params, ...)`
simulates a batch with the same settings, such as a Gauss-Newton
finite-difference batch. It runs the scalar path per ball, because a numpy
lockstep (`rk4_step_array`) breaks even only at about 70 balls. A parameter
set's static surfaces are built once and reused (`static_surfaces`).

`simulate(..., fixtures=True)` watches the board, the net fixtures and the
undercarriage, as well as the top, the net and the floor.
- `undercarriage=False` drops the undercarriage, matching the game adapter's
  board-only sweep.
- `fixtures=False` is the earlier oracle, kept for comparison.

Board contacts are events of kind `table`; their `contact.region` says top,
edge, side or underside. Undercarriage contacts are kind `frame`, with the
part as their region.

The record holds times, states and events. An impact appears as two samples
at one time, before and after.

`restitution_from_drop_test(drop, bounce, params)` inverts the vertical drop
test through this model. The ITTF table test drops a ball from 30 cm and
expects about 23 cm, with an approval band of 230 to 260 mm. The vacuum
formula `sqrt(23/30) = 0.876` ignores drag. Reading the drop through this
model with drag on gives these values:

- 0.908 at the cited `C_D = 0.50` and `rho = 1.19`;
- 0.912 at `C_D = 0.55`, which matches the priors work's independent figure.

The cited table rebounds 23.6 cm with a constant `e = 0.92`, and 24.2 cm
with its speed curve. With `e = 0.876` it rebounds only 21.4 cm.
`tests/physics/test_simulate.py` holds the rebound inside the band with drag
on, and requires `e = 0.876` to fail.

### Paddle tracks (`reference/paddle_track.py`)

`PaddleTrack(segments)` drives a scripted paddle along the game's per-tick
segments (`PaddleSegment`), the canonical paddle of data generation. Each
segment has:
- a cubic Hermite position;
- an orientation slerped from `q0` to `q1`, at the constant angular velocity
  `rotvec(q1 q0*) / h`;
- an optional contact knot, which splits it in two.

The segment in force at `t` is the last one starting at or before `t`,
clamped to its span. The track uses the game's own expressions and matches
`PaddleSegment.at` to 1e-12. Segments are read by their attributes or mapping
keys, so the reference does not import the adapter. Pass it as
`ScriptedPaddle(trajectory=PaddleTrack(segments))`.

### Blade contact state (`reference/blade.py`)

The oracle and the game adapter must resolve a paddle touch with the same
blade state. Otherwise the same touch leaves at different velocities: the
adapter's blade, interpolated linearly inside its step, once gave up to
2.4 mm/s on the outgoing ball.
- `blade_gap` is the touch condition above.
- `blade_touch_time(blade, t0, h, ball, ...)` is the first root of that gap
  inside a step, with the blade exact on its segment and the ball on a
  constant-acceleration path.
- `segment_contact` and `blade_contact` give a `BladeContact`, with these
  fields:
  - `touch`: `face` or `rim`;
  - the geometry: `normal`, `point`, `offset`, `blade_offset`, `outline`;
  - `surface_velocity`, the axes, and the paddle's `inertia(...)`.
- `blade_touch` combines them for an adapter's step.

With this state, the adapter's outgoing ball matches the oracle's to about
1e-7 m/s.

**The rounded edge** (the oracle's blade since 2026-10-09). The face-only gap
lets a ball whose centre passes up to one radius beyond the outline enter the
blade's edge and pass through. In the generated v1 rallies that was 2.1 per
1,000 shots, plus 0.7 per 1,000 that met the edge before the face law fired
(robot-env's §13.6; `reports/physics-ref.md` Addendum 12). The blade is now the
outline's disc with a rounded rim: a torus of tube radius `h` (half the
thickness) around the outline in the mid-plane.
- `ellipse_closest(x, y, a, b)` is the outline's point nearest a point of its
  plane, by a fixed number of iterations (`EDGE_ITERATIONS`, the same in the
  batched engine). Three iterations reach rounding on the contract's outline
  (2.8e-17 m). A fourth only lengthened the engine's float64 root loop: about
  +50 % of the ball's cost at 8,192 environments.
- `blade_edge_gap` is `sqrt(D^2 + d^2) - (R + h)`, with `D` the in-plane
  distance from the centre's projection to the outline.
- `blade_solid_gap` is `blade_gap` itself wherever that is negative or the
  projection is strictly inside the outline (the edge can never be nearer
  there), and otherwise the smaller of the two. Every value the face governs
  is therefore the face-only gap's own float: face touches, their roots and
  the graze search are unchanged bit for bit.
- `blade_solid_contact` gives the face's frame where the face governs, and
  otherwise an `edge` touch with the normal from the outline's nearest point
  to the ball centre.

`PaddleFaces.from_params` uses it, with the contract's `rim_contact`: rigid
Coulomb, restitution 0.5 (0.3 to 0.7), friction 0.3 (0.15 to 0.5). That is an
**unvalidated judgement prior**: there is no measurement, and the capture
cannot see an edge hit. Its range is open to randomisation. An edge touch is
resolved with the rigid paddle's mass at the edge point, and its canonical
region is `blade_edge`. `canonical_events.blade_edge_strikes` counts them, the
QA figure. `simulate(..., blade_edge=False)` is the face-only blade, an A/B
and a negative control.

The paddle's reach and point-speed bound now cover the rim (`max(a, b) + h`).
So at a fixed step, with or without the graze search, every trajectory without
an edge touch is bit-identical to the face-only blade's. With adaptive long
steps, the long steps near a blade end 6.7 mm earlier, so such trajectories
agree to rounding instead.

**The game adapter has the edge too** (stage 4). `engine/stepper.py` finds each
paddle touch with `blade_touch(..., edge=True)`. That bisects the solid gap on
the blade's exact segment, and an edge touch is resolved with `rim_contact`.
The event is recorded with region `blade_edge` (`GamePhysics.event_regions`;
`canonical_events.adapter_touches` keeps it), and `stats.contacts_edge`
counts them. The switch is `EngineSettings.paddle.blade_edge`
(`canonical_settings(..., blade_edge=True)`, the default).
- **Face-only data replays face-only.** The replay tier sets the switch from
  the episode: `ReplayEpisode.blade_edge` is true when its world physics names
  the edge's material. Datagen's face-only configurations (v1, v1.1) leave
  that material out of their records.
- **Face shots are unchanged.** A face shot gives the same events, bit for
  bit, with the edge on or off.
- **Parity with the oracle.** On edge-aimed balls, the adapter matches the
  oracle's edge strikes to 9.4e-11 s, 4.6e-10 m and 7.8e-8 m/s
  (`tests/physics/test_adapter_edge.py`).

### Inverse (`reference/inverse.py`)

`paddle_velocity_for(params, v_in, spin_in, v_out, normal, ...)` inverts
`resolve_impact` for the face velocity at the contact point, for a
point-mass or free rigid-body paddle.
- **Impulse.** The wanted change fixes the impulse `P = m (v_out - v_in)`.
- **Stick law.** With the contact compliance `K`, the stick law
  `K P = -(1 + e) u_n n - (1 + e_t) u_t` gives the relative contact velocity.
- **Restitution.** The approach speed is the root of `a (1 + e(a)) = (K P) . n`.
- **Unreachable.** An impulse outside the friction cone, or not pushing out
  along the normal, has no answer (`None`).

`blade_velocity_for` converts a face velocity to the blade-centre velocity.
The paddle reconstruction keeps its own point-mass version in
`reconstruction/paddle/inverse.py`.

### Canonical events (`canonical_events.py`)

`canonical_events(events, ...)` maps either an oracle event list or the game
adapter's events to one sequence of `CanonicalEvent` records, each with
`kind`, `region`, `player`, `t`, `point`, `normal`, `before`, `after`,
`merged`, `t_end` and `end`. It:
- drops the oracle's bookkeeping events (`roll`, `pivot`,
  `table_plane_outside`, `rest`, `out_of_bounds`, `max_time`, `max_events`);
- merges consecutive touches of the same surface within 20 ms of each other
  (sustained touches);
- merges a blade's touches within the adapter's recontact cooldown
  (0.03125 s) of its first touch.

The table's region comes from the contact normal for both sources.
`compare_sequences` compares two sequences. `KIND_CODES` and `REGION_NAMES`
are the data set's codes.

### Vectorised ball physics (`vectorized/`)

The same ball physics as the oracle, one ball per environment, stepped in
lockstep on a GPU, so training and large data runs need not loop over balls
in Python. It imports torch only when used (the demo's environment does not
ship it). `VecParams.from_params(params or a list, n, device=, dtype=)` holds
the contract's values as per-environment tensors; `with_values({...})`
randomises flat scalars per environment and recomputes the derived gains.
`BallBatch(params, states)` steps every live ball by `dt` with `step(dt,
paddles)`, where `paddles` is a `PaddleSegments` (`BladeTrack.from_states`:
each blade's pose and velocity at the step's two ends, the game's
`PaddleSegment` model). It covers flight (the oracle's RK4 operation for
operation), the table top and board, the net (catch law and cord), posts,
clamps and the floor, and the off-centre rigid paddle with the shared blade
contact state. Events go to an `EventLog`; `canonical(env)` gives the
`canonical_events` sequence. `reset(rows, states)` restarts ended balls (an
RL auto-reset). Rolling, the room and the undercarriage are not
modelled: a slow rebound off a surface gravity presses the ball onto ends it.

One deliberate difference from the oracle: over the playing surface less than
one radius up, the board's gap is the finite box distance rather than
infinite, so a ball that crosses the table's edge below one radius always
gets its edge touch. The oracle finds that touch only when one of its step
ends falls inside the crossing; its grid restarts at the top's off-table
root, so it usually does, while the batch's global grid often does not.

`torch.compile` fuses each step's work into a few kernels on CUDA (on by
default there). It needs a C compiler; without one the step falls back to
eager mode with a warning.

`DenseBallBatch` (`vectorized/dense.py`) is the same physics laid out for
CUDA graphs, for training loops. `BallBatch.step` compacts rows with
`nonzero` and checks convergence on the host, so each step costs dozens of
host syncs and hundreds of launches. `DenseBallBatch` keeps every array at a
fixed size in persistent buffers. Each phase of a step is compiled once and
captured as a graph: the prologue, a detect round, a block of root steps,
the contacts and the apply. The host reads one small tensor per round.
`vectorized/pairs.py` writes the root loop one column per quantity, so a
compiled Illinois step is a single kernel. Fired pairs are rooted in batches
of N/64, or N/4 for a burst of contacts (synchronised strokes or resets). The
batch also takes an articulated paddle compliance per blade
(`set_compliance`), and keeps a per-step on-device log (`log_view`,
`clear_log`). Use float64 on the GPU: the step
is bound by latency there, not arithmetic, so float32 buys little.

`vectorized/predict.py` (`BouncePredictor`) predicts flight plus one table
bounce for decisions. For each environment it takes the ball now and K
candidate delays, and returns:
- the first contact, which must be a table-top bounce on the receiver's half;
- the ball just after it;
- the ball at bounce + delay, with a `clean` flag when no other contact came
  first.

It runs on the batch's 1 ms grid as one fixed-iteration CUDA graph: one root
(the bounce's) and no host sync. Contacts are detected through
square-root-free quantities with the gaps' signs.

`vectorized/inverse.py` is the batched stick-regime inverse of
`reference/inverse.py`. `paddle_velocity_for` gives the face velocity that
sends the ball off at the wanted velocity, plus a validity mask (pushes out,
inside the friction cone). `GraphedInverse` replays it as one graph.

`vectorized/build.py` (`BallPhysics.create(..., precision=...)`) builds the
step, the predictor and the inverse with one `float32`/`float64` switch.

**The rounded edge in the batch.** The blades have the oracle's rounded edge
(`VecParams.blade_edge`, on by default; `BallPhysics.create(..., blade_edge=False)`
or `dataclasses.replace(params, blade_edge=False)` gives the face-only blade).
`geometry.blade_gap` and the root loop's per-column gap (`pairs.py`) compute
the face gap exactly as before. They then take the edge as
`blade_solid_gap` does, with `geometry.ellipse_closest`: the oracle's fixed
iterations, element by element, so the gap stays one fused, captured kernel.
`blade_contact` gives an edge touch's frame (`BladeContact.edge`), which
`resolve` takes with the `rim_contact` material (`MAT_RIM`). Its log region
is `blade_edge`, so `EventLog.canonical` reports it as the oracle does.
`rim_restitution` and `rim_friction` can be randomised per environment
(`VecParams.with_values`); they are the unvalidated prior.

**Face contacts carry over.** With `blade_edge=False` the engine is the engine
before the edge, bit for bit, compiled or not. With the edge on and the
engine uncompiled, an environment whose ball never touches an edge is also
bit-identical, with two exceptions. First, a ball the face-only blade let
through beside the outline is now caught, on the face. Second, the root
loop's convergence test is batch-wide: another environment's edge contact can
keep it iterating, which refines a root further within `root_tolerance`.
Compiled, the fused root-loop kernel is generated differently once the edge
is in it, so roots move in their last digits. The event sequences are the
same; the first changed event moves by about 1e-13 s. The standard for the
compiled engine is the identity suite's: identical event sequences against
the oracle on well-conditioned episodes. The drift is reported, not gated,
since bitwise identity under compiler code generation is not a meaningful
requirement (coordinator ruling, 2026-10-09). Canonical merging
folds an edge re-contact into the hit it follows. The coupling's QA count
reads the raw contacts.

### The robot-ball coupling (`engine/lab/robot_ball.py`)

`RobotBallCoupling(robot, env_origins, physics_dt, CouplingConfig(...),
chain=...)` couples an Isaac Lab articulation's `paddle` body (origin at the
blade centre) to the ball physics. Call `step()` once per PhysX step, first in
`_apply_action`. Over the step that just ended, it:
- reads the paddle's position and orientation, and its linear velocity only for
  the Hermite track (never the angular velocity);
- builds the blade's motion, a Hermite curve with slerp (or PhysX's straight
  line); a reset env's blade is parked;
- computes the arm's compliance at the blade, `J M^-1 J^T` with the actuators'
  armature (PhysX's mass matrix leaves it out), for the environments whose ball
  is within reach;
- flies the ball in sub-steps of the batch's `dt`;
- slices that step's contacts from the batch's on-device log;
- applies `F = -sum J / h` at the paddle's centre of mass, with the torque
  `sum (r_i - com) x F_i` about it (no `-L/h` term), together with any wrench
  given to `add_external(force_w, torque_w)` (a world-frame force at the
  centre of mass and a torque about it, per environment). Both are computed in
  the environment's frame in float64; a float32 world-position lever would lose
  about 3e-5 m at 360 m from the world origin. `coupling.reaction` holds the
  ball's force, that torque and the centre of mass; `impulse` and `moment` the
  ball's side.

With `compliance="articulated"`, the contact law takes the surface
compliance `K(r) = A + B[r]x - [r]xB^T - [r]xC[r]x` at each contact's offset
`r` instead of the contract's paddle mass and free-paddle inertia. For a free
rigid paddle the two are the same law; a test holds them equal. The
negative controls and A/Bs (`track="line"`, `blade_lag_steps`, sub-step `= h`,
`angular_torque_term`) run through the same path. `coupling.blade` is the
blade at the last step with the track's end velocity and spin (the values the
law used), with no PhysX read; `blade_now()` reads PhysX.

**Where `J` comes from.** `jacobian="physx"` reads Lab's
`body_link_jacobian_w`. `jacobian="torch"` computes it with `PaddleJacobian`,
in float64, from the root's orientation and the joint positions along the
chain from the root link to the paddle (`chain=`: `ChainJoint`s, or robot-env's
`lib/hardware/robots/g1/kinematics.chain_to`, read from the stage). It uses
PhysX's layout: on a floating base the six base columns come first and carry
the root's centre-of-mass velocity. Only orientations and positions relative
to the root enter, so it does not depend on the environment's world offset.
After the first step the coupling compares it with PhysX's in every
environment, once, and raises beyond `2e-5 max(1, |origin| / 16 m)`; PhysX's
own Jacobian is float32 in world coordinates. On the G1 the two agree to about
4e-6 within 21 m of the origin, and the effective mass at the blade agrees to a
relative 3.8e-6 (p99) in the origin env. A wrong elbow axis moves the
effective mass by 40 % or more and fails both checks.

**What a step reads from PhysX.** On PhysX, the first read in a physics step
that needs the links' post-step state (a link velocity, the Jacobian, or the
centre-of-mass position) refreshes it: about 10 ms per physics step at 4,096
environments and 21 ms at 8,192. Link poses, joint state and the mass matrix
do not trigger it. `track="line"` with `jacobian="torch"` reads nothing that
does. But Lab's wrench composer reads `body_com_pos_w` in `write_data_to_sim`
whenever it is active, so through the composer the env step pays the same
refresh there.

**Edge strikes** (the QA count). `StepContacts.robot_edge_hit` marks the
environments whose robot blade struck its rounded edge in the step, and
`counters.edge_contacts` counts those env-steps. `CouplingConfig(blade_edge=False)`
is the face-only blade, an A/B and negative control.

**Warm-up** (`CouplingConfig.warm_up`, on by default). The coupling compiles
and captures every phase when it is built: its own `prepare` and `collect`,
and each of the ball batch's phases and pair tiers
(`DenseBallBatch.warm_up`). On CUDA that takes about four minutes. Left to
first use, it landed inside the first steps that met a ball: robot-train's
rally pass read 208 ms per physics step there, against 15 ms once compiled.
Each phase runs once out of turn and every buffer is put back, so no result
changes; a CUDA test holds a warmed and a cold batch bit-identical.
- `CouplingConfig.compile_cache` (`off` by default) restores torch.compile's
  cache from an Antioch asset before the warm-up, per torch, CUDA and GPU
  architecture (`vectorized/compile_cache.py`). `restore+publish` also
  publishes it after a warm-up that compiled.
- Each session starts with an empty cache, so without this every run
  recompiles.

**Wrench routes** (`CouplingConfig.wrench`). PhysX applies only the last force
write to an articulation's links before a physics step, and clears it after
the step (measured on production, by this module's probe and robot-env's).
- `"composer"` (the default and the reference): the reaction and every
  `add_external` are added to Lab's instantaneous composer, and Lab writes
  their sum once.
- `"direct"` (opt-in): one write per physics step, for every environment, to
  `robot.root_view.apply_forces_and_torques_at_position` (a global force at the
  link's centre of mass and a global torque; zeros where nothing pushed). It is
  made inside the articulation's `write_data_to_sim`, after Lab's own write and
  after every `add_external` of the step, so their order in `_apply_action`
  does not matter. The composer must stay unused: a step on which it was
  written raises, as does a step whose write never happened. `coupling.route`
  says which route the last step took.
- `"direct_unguarded"` is a negative control: the reaction is written at once,
  and a later composer write erases it.

On the G1 the two routes give the same G0.4 balances, contacts, effective
masses and paddle response. Their residual statistics differ at the float32
floor; the direct route is slightly tighter. The env step gains only where
nothing else reads the links' post-step state: with `jacobian="torch"`,
`track="line"` and `wrench="direct"` it falls from 156 to 121 ms per policy
step at 4,096 environments, and from 239 to 154 at 8,192, against the default
path in the same suite. The Hermite track still reads PhysX's link velocity,
which pays the refresh, so with `"hermite"` the direct route saves nothing.
Taking that velocity from the torch Jacobian and the joint velocities (a cheap
read) would remove it; that is not built.

**Far from the world origin.** The coupling's own algebra is env-local float64:
the blade, the contact, the lever and the torque. PhysX's state is float32 in
world coordinates. At the far corner of an 8,192 × 2.5 m grid, (113, -113) m,
the same 10 m/s stroke as the origin env gives the blade's velocity at contact
within 2.0e-3 m/s (Hermite) or 4.7e-3 m/s (line), the ball's exit velocity
within 2.2e-3 or 5.9e-3 m/s, and the face within 4e-5 rad. The L0 contact
budgets are 0.27 m/s and 0.041 rad.

### Projection and fitting (`reference/projection.py`)

```
x' = X/Z, y' = Y/Z, r^2 = x'^2 + y'^2,  radial = 1 + k1 r^2 + k2 r^4 + k3 r^6
x'' = x' radial + 2 p1 x'y' + p2 (r^2 + 2x'^2),  y'' = y' radial + p1 (r^2 + 2y'^2) + 2 p2 x'y'
u = fx x'' + cx,  v = fy y'' + cy
```

Undistortion is OpenCV's fixed-point iteration.

`fit_flight` estimates a free-flight segment's state at `t0`: position and
velocity, and optionally spin. It minimises the sigma-normalised pixel
reprojection error of timestamped detections from any number of cameras,
integrating the flight ODE at a 1 ms step. The fit runs in up to three steps:

1. A linear ballistic solve gives the start point. With gravity only, each
   detection gives two equations that are linear in position and velocity;
   rows are reweighted by depth with a Huber weight.
2. A robust stage (`soft_l1` by default) finds the inliers.
3. A plain least-squares refit uses only those inliers.

The bounded-influence loss alone still lets a gross outlier bias the estimate
by about `f_scale` sigma, which is why the third step exists.

The covariance is the Gauss-Newton inverse over the inliers. It is scaled up
by the reduced chi-square when that exceeds 1, and never scaled down. Fitting
spin adds a Gaussian prior (500 rad/s by default), because spin along the
velocity is unobservable.

## What the oracle does not model

- Net sag. The mesh is the phenomenological catch law, not a deforming
  membrane, and the cord, posts and clamps are rigid.
- Paddle edges. Only the faces collide: a rim touch is a face touch at the
  outline, with the face normal.
- A realistic paddle centre of mass. It is taken at the blade centre; a real
  one sits nearer the handle, which would lower `M_eff` at the centre of the
  face too. The script also ignores the impulse after the hit.
- Ball buckling above about 5.5 m/s normal speed, except through the
  restitution speed curves.
- Sustained contact on anything other than the floor and the playing surface:
  a slow rebound there is `rest`. On the table board's edge it is more falling.
- Wind, and the ball's deformation beyond what `e`, `mu` and `e_t` encode.
- A per-face rubber. Both faces share `rubber_contact`.

## References

- R. M. Brach, "Rigid body collisions", *J. Appl. Mech.* 56 (1989) 133-138;
  *Mechanical Impact Dynamics*, Wiley, 1991.
- R. Garwin, "Kinematics of an ultraelastic rough ball", *Am. J. Phys.* 37
  (1969) 88-92.
- R. Cross, "Grip-slip behavior of a bouncing ball", *Am. J. Phys.* 70 (2002)
  1093-1102.
- W. J. Stronge, *Impact Mechanics*, Cambridge University Press, 2000.
- ITTF Laws of Table Tennis and the ITTF technical leaflets for the table and
  the ball (dimensions, the 30 cm to 23 cm drop test, 2.7 g).
- OpenCV camera calibration model (`calib3d`), radial-tangential distortion.

## Tests

`tests/physics/` runs without a simulator: `uv run pytest tests/physics`
from `table-tennis/`. It covers:

- closed forms: the vacuum parabola, quadratic drag to terminal velocity, and
  horizontal drag;
- RK4 order;
- the Magnus direction;
- event timing for the table, the net face and cord, and the floor;
- impact: energy, the elastic limit, the `e^2` rebound, the rolling
  threshold, spin effects, moving walls, change of frame, rotation
  equivariance, overspin, normal-spin retention, the off-centre rigid paddle
  and the net-mesh catch law;
- the net: a face hit is caught and drops beside the net with no lift from
  backspin, the old rigid face gave over 1 m/s of lift, and the cord and tape
  stay rigid;
- the board and fixtures:
  - 400 random board impacts at the game's 1/3840 s step agree with the game
    adapter's `sweep_table_box` plus the impact law: time to 6e-8 s, position
    to 0.4 µm, normal to 0.001°, and velocity to 1e-4 of speed;
  - the top plane is byte-identical with and without the board;
  - edge balls, side hits, the underside, posts and clamps;
  - energy never increases;
  - the geometry is pinned to the adapter's and the scene's constants;
  - the post restitution equals this model's inversion of the ITTF steel-block band;
- rolling:
  - the vacuum and air stopping distances against their closed forms;
  - sliding into rolling at `v0 / (1 + k)`;
  - mechanical energy is monotone;
  - a dead ball rolls off the table and comes to rest on the floor;
  - the slip-margin regression from game 6's event 632;
  - the edge pivot's no-slip release angle, energy, stick-slip and its effect on landing;
  - a rolling ball bounces off a wall;
  - the room's geometry and wall material are pinned to the game adapter;
- the undercarriage:
  - its geometry is pinned to the table asset's configuration;
  - legs, the apron seen from below, and a dead ball landing on a stretcher instead of the floor;
  - frame impacts never gain energy;
- the seam with `data/priors.json`: it loads, curves scale with their
  scalars, randomising `table_restitution` changes the bounce, and the cited
  table rebounds 23 to 26 cm from 30 cm with drag on while `e = 0.876` fails;
- simulation records and terminal events;
- speed without changing answers:
  - the inlined RK4 step is bit-identical;
  - static-gap skipping gives bit-identical segments;
  - each static surface's clearance keeps its gap's sign;
- adaptive steps and the graze search:
  - the fine step's events on a rally shot and on a swinging blade;
  - cord grazes that a fixed step misses;
- paddle tracks:
  - equal to `PaddleSegment.at`;
  - velocity and angular velocity consistent with the motion;
  - bounding spheres, the far test, windows and the blade axes;
  - a rim touch;
- the blade contact state:
  - datagen's two paddle-contact cases agree with the oracle to 1e-6 m/s
    with the shared state, and not without it;
  - the shared gap and contact frame;
  - a rim touch found inside the step;
- the inverse:
  - round trips through `resolve_impact`, point-mass and rigid;
  - agreement with the reconstruction's version;
- canonical events:
  - mapped from both sources;
  - the smoke rally through the oracle and the rk4 game adapter gives
    identical canonical sequences;
- the room and floor against the game adapter's sweeps: faces, edges and
  corners;
- the vectorised physics, with torch installed (skipped otherwise), on the
  CPU in float64:
  - flight, gaps, contact frames, blades and the impact law against the
    oracle's term by term;
  - per-environment parameters against the contract's derived values;
  - over 450 calibration trials (every board and fixture trial, a stride
    through the rest, scripted paddles included) give the oracle's canonical
    sequences with positions within 1e-6 m;
  - a 2 % change in table restitution is detected (negative control);
  - `DenseBallBatch` on the same trials, with default and with tiny pair and
    log capacities;
  - the root loop's per-column gap equals the tensor gap for every surface
    kind and both blades;
  - the predictor against the oracle's own predictions: validity on every
    launch, the bounce to 1e-9 s, states to 1e-6 m, `clean` never claimed
    where the oracle had another contact; a changed table restitution moves
    every bounce (negative control);
  - the batched inverse round-trips through the batched forward law, and
    matches the scalar reference row by row to 1e-11;
  - the precision switch sets every part's dtype;
- the robot-ball coupling on a stand-in free paddle:
  - its articulated compliance is the law's rigid paddle exactly;
  - the reaction balances the ball's impulse and moment (G0.4's bookkeeping),
    applied at the paddle's centre of mass with the torque about it;
  - every contact re-resolves from its recorded blade state and `K`;
  - the log is sliced per step and a reset parks the blade;
  - the negative controls change what they should;
  - each track reads only the PhysX velocity it needs;
- the torch paddle Jacobian:
  - it equals central differences of its own kinematics on a random 7-joint
    chain, fixed and floating base;
  - on a stand-in arm, the coupling's startup check passes against the
    reported Jacobian, and the effective mass equals the `jacobian="physx"`
    path's to 1e-6;
  - negative controls: a wrong joint axis breaks the match and fails the
    startup check, and a chain that does not end at the paddle is refused;
- the batch's blade against the oracle's, face-only and with the rounded
  edge (gap to 1e-12, frame, touching point and edge flag), and the rim QA
  count in the coupling (edge-aimed balls counted, face shots not, and the
  face-only blade letting the edge-aimed balls through);
- the rim prior against the oracle's impact law, with `rim_restitution` and
  `rim_friction` drawn per environment over their ranges, the rubber face
  unchanged;
- the warm-up: a warmed dense batch is bit-identical to a cold one on
  calibration trials, with both pair tiers warmed (CUDA only);
- the compile cache: it restores file for file, and a missing asset is
  reported rather than raised;
- the wrench routes, against a stand-in PhysX view that keeps only a step's
  last force write and clears it after the step (as measured on production):
  - on the composer and direct routes, PhysX receives exactly one write a
    step, equal to the ball's reaction plus `add_external`'s wrench;
  - the direct route raises when the composer was also written that step,
    and when a step's write never happened;
  - negative control: the reaction written to the view at once and a test
    wrench through the composer after it leave PhysX the test wrench alone;
- projection;
- two-camera, 20 fps flight fits with dropouts and an outlier, including a
  check that the fit's stated uncertainty is honest.
