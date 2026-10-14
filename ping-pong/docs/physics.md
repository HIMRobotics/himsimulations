# Ball physics

`lib/physics` holds the calibrated table-tennis ball physics that every part of this project shares:

- one set of physical parameters, with their units and priors;
- an engine-independent **reference model** (the oracle) in numpy and scipy;
- the replay's adapter, which makes Isaac Sim's ball behave like the oracle;
- a GPU batch of the same physics in torch, which the robot plays against in Isaac Lab.

No engine is trusted as ground truth. The simulators are checked against the oracle, and the oracle's parameters are
checked against HIM's recorded bounces.

## Layout

| Module | Contents |
|---|---|
| `params.py` | `BallTableParams` and its frozen pydantic sub-models, the flat parameter names, the `priors.json` loader |
| `data/priors.json` | The cited priors, keyed by flat name, with their sources |
| `reference/flight.py` | The flight ODE, RK4 (scalar and vectorised) and event detection |
| `reference/impact.py` | Rigid-ball impact on a flat, possibly moving surface of finite effective mass |
| `reference/surfaces.py`, `room.py`, `undercarriage.py` | Contact geometry: table top and board, net, posts and clamps, floor, room, table frame, scripted paddles |
| `reference/rolling.py` | Sustained sliding and rolling on the floor and the playing surface |
| `reference/simulate.py` | Event-driven trajectories |
| `reference/blade.py` | The blade's touch condition, touch time and contact state, shared by the oracle and the adapters |
| `reference/inverse.py` | The paddle impact inverted: the face velocity that sends a ball off with a wanted velocity |
| `reference/projection.py` | OpenCV pinhole cameras (the virtual Orbbecs) |
| `engine/` | The replay's Isaac Sim adapter: settings, hybrid contacts, the RK4 ball integrator, the per-step callback |
| `vectorized/` | The oracle batched over environments in torch, with CUDA graphs |
| `robot_coupling.py` | A robot's paddle and the GPU ball, coupled both ways once per PhysX step (Isaac Lab) |
| `canonical_events.py` | One canonical contact sequence from raw touches |

Nothing here imports Isaac at module scope; torch is imported on first use.

## Conventions

- **Units.** SI throughout: m, kg, s, rad. Every docstring states its units.
- **Table frame (the world).** Z up. The origin is the centre of the playing surface and +X runs along the table's
  length. The table top is the rectangle `|x| <= 1.37`, `|y| <= 0.7625` in the plane `z = 0`. The net is the plane
  `x = 0`, and the floor is `z = -table.height` (-0.76 m).
- **State vector.** 9 floats: position (m), velocity (m/s), angular velocity (rad/s), all in the table frame.
- **Spin sign.** Topspin on a ball moving along +X is angular velocity along +Y: the top of the ball moves forward.
  A ball rolling along +X has `omega_y = v_x / R`.
- **Blade axes.** The adapters' blade axes (`PaddleSegment`, `reference/blade.py`, `vectorized/`) are X across the
  blade, Y towards the tip and Z the forehand face normal; the pose is the blade centre on its mid-plane. The
  oracle's `PaddlePose` rotation is `[-Y, X, Z]` of them (local X towards the handle). Both faces collide.
- **Quaternions** are `(w, x, y, z)`, except where Isaac Lab's `(x, y, z, w)` is named.
- **Cameras.** OpenCV convention: camera +X right, +Y down, +Z forward, and `x_cam = R x_world + t`.

## Parameters and `priors.json`

Each scalar has a flat name (`PARAMETER_SPECS`). `data/priors.json` is keyed by these names; each entry carries its
value, range, unit and source. A name the file lacks falls back to a default defined in `params.py`, whose
provenance says `provisional` (a placeholder) or `derived: ...` (computed from cited inputs), so no report can mistake
it for a measured value. `load_priors().provisional_names()` lists the placeholders in use.

```python
from lib.physics.params import nominal_params

params = nominal_params()  # the priors' nominal values
dead = params.with_values({"table_restitution": 0.30})  # the replay's negative control
```

| Name | Nominal | Range | Origin |
|---|---|---|---|
| `ball_radius` | 0.02005 m | 0.02-0.02025 | cited (40+ ball) |
| `ball_mass` | 2.7 g | 2.67-2.77 g | cited |
| `ball_shell_thickness` | 0.5 mm | 0.38-0.55 mm | cited |
| `air_density` | 1.19 kg/m^3 | 1.10-1.23 | cited (indoor air) |
| `drag_coefficient` | 0.50 | 0.40-0.58 | cited |
| `lift_slope`, `lift_max` | 1.2, 0.40 | 0.9-1.6, 0.28-0.50 | cited (rational form) |
| `spin_decay_coefficient` | 0.02 | 0.007-0.055 | cited |
| `table_length`, `table_width`, `table_height` | 2.74, 1.525, 0.76 m | regulation tolerances | cited (ITTF) |
| `table_board_thickness` | 25 mm | fixed | derived from the table asset |
| `net_height`, `net_overhang` | 0.1525 m each | regulation tolerances | cited (ITTF) |
| `net_thickness`, `net_tape_height` | 3 mm, 15 mm | 2-6 mm, 10-15 mm | judgement |
| `paddle_blade_half_length`, `_half_width` | 78.5, 75 mm | 75-80, 72.5-77 mm | cited |
| `paddle_thickness` | 13.6 mm | 11-14.6 mm | cited |
| `paddle_effective_mass` | 0.17 kg | 0.10-0.40 | cited |
| `paddle_inertia_xx/yy/zz` | 3.21, 3.52, 6.68 e-4 kg m^2 | +-~100 % | derived (uniform slab of the blade) |
| `table_restitution` | 0.94 | 0.92-0.955 | **set by 1117 recorded HIM bounces** |
| `table_friction` | 0.25 | 0.15-0.40 | cited |
| `rubber_restitution` | 0.72 | 0.58-0.86 | cited |
| `rubber_friction` | 1.0 | 0.6-1.5 | cited |
| `rubber_tangential_restitution` | 0.70 | 0.50-0.85 | cited |
| `rubber_normal_spin_retention` | 0.805 | 0.70-0.95 | cited |
| `net_restitution`, `net_friction` | 0.2, 0.3 | 0.05-0.45, 0.1-0.6 | judgement |
| `net_tangential_damping`, `net_spin_damping` | 0.8, 0.7 | 0.5-0.95, 0.4-0.95 | judgement (no ball-net measurement exists) |
| `floor_restitution`, `floor_friction` | 0.88, 0.35 | 0.75-0.94, 0.2-0.6 | cited, judgement |
| `post_restitution`, `post_friction` | 0.946, 0.20 | 0.921-0.970, 0.10-0.35 | cited (ITTF steel-block test, drag-corrected), judgement |
| `fixture_restitution`, `fixture_friction` | 0.85, 0.25 | 0.75-0.92, 0.15-0.40 | judgement |
| `wall_restitution`, `wall_friction` | 0.80, 0.35 | 0.70-0.90, 0.20-0.50 | judgement |
| `floor_rolling_resistance`, `table_rolling_resistance` | 0.010, 0.005 | 0.003-0.030, 0.002-0.015 | judgement |
| `rim_restitution`, `rim_friction` | 0.5, 0.3 | 0.3-0.7, 0.15-0.5 | judgement (the blade's edge; unvalidated) |

**Speed curves.** `<surface>_<restitution|friction>_vs_speed` takes `{"speeds": [...], "values": [...]}`: a
piecewise-linear coefficient against the normal approach speed, held constant past both ends. The committed file has
curves for the table, the rubber and the floor. A curve never replaces its scalar: the material evaluates
`curve(v) * scalar / reference`, clipped to [0, 1], where `reference` is the scalar's nominal. At the nominal scalar
this is exactly the curve, and randomising `table_restitution` moves the whole curve.

**Randomisation.** `with_values({...})` replaces named scalars; `vectorized.params.VecParams.with_values` does the
same per environment on the GPU. Ranges are uniform bounds, not standard deviations.

**Apparent and intrinsic rubber restitution.** The cited rubber curves were measured on robot-held rackets, so they
are apparent values. With the finite `paddle_effective_mass` the simulated rebound comes out about 0.02-0.03 lower
than measured; a paddle with infinite effective mass reproduces the measurements directly.

## The reference model

### Ball

A uniform spherical shell of outer radius `R`, inner radius `a = R - t` and mass `m`, with the exact inertia
`I = (2/5) m (R^5 - a^5) / (R^3 - a^3)`. At the cited 0.5 mm wall, `k = I / (m R^2) = 0.6503`.

### Flight (`reference/flight.py`)

With `A = pi R^2`, `k_a = rho A / (2 m)`, `omega_perp` the spin perpendicular to the velocity and
`S = R |omega_perp| / |v|`:

```
dp/dt     = v
dv/dt     = -g z_hat
            - k_a C_D(S) |v| v                                      drag
            + k_a C_L(S) |v|^2 (omega_perp x v_hat) / |omega_perp|  Magnus
domega/dt = -(C_tau rho A R^2 / (2 I)) |v| omega                     spin decay
C_L(S)    = slope S / (1 + slope S / c_max)                         rational (default)
```

**This lift model is the source of truth for Magnus.** Engine-native fluid models use other conventions: MuJoCo's
`F = C_M rho V (omega x v)` is `C_L = (8/3) C_M S`, so its default `C_M = 1` is 2.7 to 10 times stronger than measured
table-tennis balls. An adapter switches engine-native fluid forces off and applies `FlightModel.wrench` instead. The
air is still: there is no wind term.

**Integrator.** Classical RK4 at a fixed step (1e-4 s by default); measured convergence is fourth order. Each surface
supplies a signed gap. An event fires when a gap goes from `>= 0` to `< 0` across a step; its root is bracketed to
1e-10 s by bisection, re-taking one RK4 step from the step start for each trial, so the event state is exactly a state
the integrator produces. Static gaps are evaluated only when the ball could have changed their sign (bit-identical
results, about 2.4x faster). Adaptive long steps (`max_step`) and a graze search (`graze`) are opt-in.

### Impact (`reference/impact.py`)

The surface has unit normal `n` (from the surface into the ball), velocity `u_s` at the contact point and effective
mass `M` (infinite for the table, net and floor):

```
r_c = -R n,  u = v + omega x r_c - u_s,  u_n = u . n (< 0 approaching),  u_t = u - u_n n
1/m_n = 1/m + 1/M,   1/m_t = 1/m + 1/M + 1/(k m)
normal (Newton):     P_n = -(1 + e) m_n u_n
tangential (Brach):  P_stick = -(1 + e_t) m_t u_t;  stick if |P_stick| <= mu P_n, else P_t = -mu P_n u_t / |u_t|
ball:                v' = v + (P_n n + P_t) / m,   omega' = omega + (r_c x P_t) / I - (1 - s_n)(omega . n) n
energy change:       1/2 m_n (u_n'^2 - u_n^2) + 1/2 m_t (|u_t'|^2 - |u_t|^2) - 1/2 I (1 - s_n^2)(omega . n)^2 <= 0
```

- With `e_t = 0` (the table) this is the rigid Coulomb model: a sticking contact leaves rolling. With `e_t > 0` (the
  rubber, 0.70) the contact slip reverses, which returns overspin from a grippy rubber.
- `normal_spin_retention` `s_n` (rubber 0.805, every other surface 1) stands in for pivoting friction.
- **Off-centre hits on a rigid paddle.** With the contact offset `r` and the paddle's inertia `I_p`, the paddle is a
  free rigid body: its contact-point compliance is `K_s = (1/M) 1 + [r]x^T I_p^-1 [r]x`, so 5 cm off-centre a 0.17 kg
  blade meets the ball with about 0.075 kg. Momentum is conserved to 1e-12 and energy never rises.
- **The net mesh: a catch law.** A rigid net face would turn backspin into lift (over 1 m/s). The mesh wraps round
  the ball instead: `v_r' = -e (v_r . n) n + (1 - d_t) v_r,t`, `omega' = (1 - d_w) omega`, with no contact-point
  friction. The cord and tape stay rigid.

`reference/inverse.py` inverts the stick regime in closed form: from the ball's wanted outgoing velocity to the face
velocity at the contact point (`None` outside the friction cone).

### Surfaces and simulation

- **Table top** (gap `z - R`, a contact only over the rectangle) and **table board** (the 25 mm board's ends, sides,
  edges and underside, as a box; contact regions `top`, `edge`, `side`, `underside`).
- **Net** (a box: `mesh` under the catch law, `cord` rigid), **posts** (steel cylinders) and **clamps** (boxes).
- **Floor**, the **room** (walls, ceiling, the furniture along the +Y wall; opt-in) and the table's steel
  **undercarriage** (apron, legs, stretchers, struts, feet, casters; contacts of kind `frame`).
- **Paddle.** The touch condition is `blade_gap = max(|n . (c - p)| - (R + h), (e - 1) min(a, b))`: the ball within
  reach of the face plane **and** its touching point inside the elliptical outline. With `blade_edge` the blade also
  has a rounded rim (a torus of tube radius `h` around the outline), resolved with the `rim_contact` prior.
- **Simulation** (`simulate`) integrates flight until an event, applies the impact and repeats, until a terminal
  contact, `rest`, leaving the bounds or a cap. A ball that rebounds from the floor or the playing surface slower than
  0.05 m/s enters sustained **sliding and rolling** (`reference/rolling.py`), and a ball rolling off the table pivots
  about its edge before it falls.

**The ITTF drop test.** A ball dropped from 30 cm must rebound 23-26 cm. With drag on, the cited table (0.94 on its
speed curve) rebounds 24.2 cm; the drag-free formula's `e = 0.876` rebounds only 21.4 cm and fails.

Not modelled: net sag (the mesh is the phenomenological catch law), a realistic paddle centre of mass (it is taken at
the blade centre), ball buckling beyond the restitution curves, wind, per-face rubbers.

## The replay's engine adapter (`engine/`)

The replay runs Isaac Sim 6.0.1 with PhysX on the CPU (CCD, the PGS solver) at **3840 Hz**, 160 physics steps per
24 fps rendered frame. Why each setting (`engine/settings.py`):

- **PGS, not TGS.** Under TGS a contact whose first touch falls early in a step loses most of its restitution.
- **CCD on the CPU**, no tunnelling at 20-50 m/s; stabilisation, sleep and damping off; the angular-velocity cap raised
  (the default clips a 150 rev/s topspin to a tenth).
- **Hybrid contacts.** No rigid-contact engine represents the rubber's tangential restitution or normal-spin
  retention, and PhysX's native table applies friction per axis and mis-deflects net-cord hits. So the ball never
  collides with the paddles, table, net, posts, clamps, floor, room or players in PhysX: each physics step the adapter
  sweeps the ball against every surface over the coming step (`engine/contacts.py`, `engine/sweeps.py`) and, at the
  first touch, imposes the oracle's impact law at the contact time. Slow touches are resolved as sustained contact.
- **The players' bodies** are moving capsules sized from anthropometric means (`engine/body_capsules.py`), so a ball
  bounces off a player instead of passing through the skinned mesh.
- **Paddles** are dynamic bodies driven by velocity every step along the replay's per-tick Hermite segments.
- **The ball is integrated by RK4** (`engine/ball_integrator.py`): with every contact the adapter's own, PhysX never
  touches the ball, so the adapter advances it with the oracle's RK4 at the game step and restarts from the contact
  time after a hit. PhysX steps only the paddles and the retired balls, ten times per rendered frame.

**Why 3840 Hz.** It is the lowest rate, dividing the 24 fps render, at which the adapter passes every calibrated
test; 960 and 1920 Hz fail the drop, drag and parity tests. Under RK4 the adapter matches the oracle on the canonical
trials to 0.00015 ms and 0.0006 mm (the parity budget is 2 game steps and 10 mm), and its step-halving error is
0.002 mm (PhysX's own integrator: 1.5 mm, first order).

## The GPU ball (`vectorized/`) and the robot coupling

`vectorized/` is the oracle's physics as torch tensors of shape `(N, ...)`, one ball per environment, so a robot can
train and play against the calibrated ball in thousands of parallel Isaac Lab worlds:

- `VecParams` holds the parameters per environment (domain randomisation without parameter objects);
- `BallBatch` steps every live ball by `dt` as `simulate` does: RK4, every watched gap, Illinois roots, the batched
  impact law;
- `DenseBallBatch` is the same physics laid out for CUDA graphs: every phase of a step is compiled with
  `torch.compile` and captured once, so a step costs a few graph replays and no host syncs;
- `BouncePredictor` predicts flight plus one table bounce in one graph (the policy's forecast and the opponent's
  decisions);
- `vectorized/inverse.py` is the batched stick-regime inverse;
- `BallPhysics.create(..., precision="float64" | "float32")` builds all three with one dtype switch.

In float64 the batch reproduces the oracle's canonical contact sequences with positions within 1e-6 m. It does not
model rolling, the room or the undercarriage: a training episode ends when the ball leaves play.

`robot_coupling.RobotBallCoupling` couples an Isaac Lab articulation's `paddle` body to the batch, once per PhysX
step (call `step()` first in `_apply_action`). Over the step that just ended it reads the paddle's pose, builds the
blade's motion (Hermite with slerp, or PhysX's straight line), computes the arm's compliance at the blade
(`J M^-1 J^T` with the actuators' armature, which PhysX's mass matrix leaves out), flies the ball in sub-steps, slices
that step's contacts, and pushes back on the paddle with `F = -sum J / h` at its centre of mass and the matching
torque, computed env-locally in float64. The robot feels a contact one step later. On a CUDA device the coupling
compiles and captures every phase when it is built (about four minutes; `CouplingConfig.warm_up`).

## What is calibrated against what

| Check | Against | Result |
|---|---|---|
| ITTF 30 cm drop, engine apex | the oracle | 0.5 mm (budget 2 mm); every apex inside 230-260 mm |
| Spinless and spinning flights with drag, 5-30 m/s, at 1 s | the oracle | within 0.6 of the error envelope |
| Oblique spinning table bounces; the hybrid rigid paddle | the oracle | 0.014 and 0.005 of the tolerance |
| Net mesh and cord; posts and clamps; table board, edges and undercarriage | the oracle | no outcome or region mismatches, no tunnels |
| Step convergence (step against half step) | itself | 0.002 mm under RK4 (budget 1 mm) |
| The replay at 3840 Hz, canonical trials | the oracle | 0.00015 ms, 0.0006 mm |
| Table restitution | 1117 recorded HIM bounces | 0.94 (folds 0.940-0.944); range narrowed to 0.92-0.955 |

The replay's own run checks the drop test again at the start of every session (`lib/replay`), and its dead-table
negative control (restitution 0.30) must fail it.

## References

- R. M. Brach, "Rigid body collisions", *J. Appl. Mech.* 56 (1989) 133-138; *Mechanical Impact Dynamics*, Wiley, 1991.
- R. Garwin, "Kinematics of an ultraelastic rough ball", *Am. J. Phys.* 37 (1969) 88-92.
- R. Cross, "Grip-slip behavior of a bouncing ball", *Am. J. Phys.* 70 (2002) 1093-1102.
- W. J. Stronge, *Impact Mechanics*, Cambridge University Press, 2000.
- ITTF Laws of Table Tennis and the ITTF technical leaflets for the table and the ball.
- OpenCV camera calibration model (`calib3d`), radial-tangential distortion.
