# Retargeting HIM's strokes onto a humanoid

`lib/retarget` turns HIM's motion-captured table-tennis strokes into motion for a humanoid robot holding a paddle. It
checks every stroke for what the robot can really do and packages the strokes that pass. The Unitree G1 is the worked
example. Any humanoid with a URDF and a `RobotSpec` can be the target.

## The short version

**Motion capture** (mocap) records how a person moves. HIM's player wore a Rokoko suit of motion sensors, which
recorded the positions and orientations of 21 key joints (hips, spine, shoulders, hands, feet and so on) 30 times a
second, through all seven sessions.

**Retargeting** moves that motion onto a different body. You can't copy joint angles across: the G1's arm is 0.75 the
length of the human's and its legs 0.84, its joints bend differently, and its motors have limits. So the retargeter
works from what matters, the paddle. At each of the 737 moments the human's paddle touched the ball, it asks: what
joint angles put the robot's paddle in the same place, facing the same way, moving the same way? Working backwards
from where a hand must be to the joint angles that put it there is called **inverse kinematics** (IK). Here it's
solved as an optimisation over a whole rally at once, so the motion is smooth, the feet stay planted, and the robot
stays balanced.

Then every stroke is **checked** against the robot's real abilities: can it reach, are its joints fast enough, are
its motors strong enough, does it keep its balance, do its feet stay put? A stroke that fails is rejected with its
reason, never faked. Of HIM's 737 strokes, the G1 reference output keeps 251. Balance and the waist motors' torque
reject more strokes than reach does.

## The scenario

The scenario is `retarget_him_strokes`, on the `robot` service (CPU only; no simulator starts). In the table, "blade
p50" is the median distance (the 50th percentile) between the robot's blade and the human's paddle path:

| Case | What it solves | Local result (G1) |
|---|---|---|
| `unitree-g1-smoke` | Session 1's probe rally: 10 strokes, 18 s | 5 of 10 strokes kept; blade p50 0.35 cm through ±0.1 s of contact; every run check passes; 4–5 min on one CPU thread |
| `unitree-g1` | Every rally of the 7 sessions: 191 rallies, 737 strokes | Reference output `ping-pong/g1-retargeted-strokes@v2`: 251 of 737 kept, 214 in its `motion.npz` |

**This code reproduces the reference.** The reference output was solved with the G1's earlier ready stance, right
elbow at 0 rad. With that stance, this code keeps the reference's 6 of 10 on the probe rally: the same six strokes,
blade p50 0.405 cm (reference 0.405 cm).

The G1 here stands with the elbow raised 0.1 rad, so the paddle clears the table at every reset. Through the solver's
weak pull to the stance, that raises one backhand's capture-point shift from 2.84 to 3.35 cm, past the 3 cm balance
check. (The *capture point* is the spot on the floor where a robot would have to step to stop itself; the balance
check limits how far a stroke can push it.) That one stroke is the difference between 6 and 5.

```bash
uv run antioch scenario run retarget_him_strokes --cases unitree-g1-smoke
```

The run publishes its kept strokes to HIM's asset store as `ping-pong/unitree-g1-retargeted-strokes`, versioned by
the first 8 characters of its run ID. To retarget a different rally, see the README's
[experiments](../README.md#4-retarget-a-different-rally). No simulator means no 3D window either, so there's nothing
to watch live in Mission Control: the run's telemetry, on its console page, plays the strokes back on a stick-figure G1.

## The algorithm

Read this sketch as four steps, from the human's strokes to a package of robot motion:

```python
human = load_mocap_strokes(asset)  # HIM's strokes on the table frame, the paddle state at each contact
robot = robot_model(ROBOTS["unitree-g1"])  # URDF kinematics, joint limits, motors, the paddle on the wrist
rallies = retarget(human, robot)  # reach -> targets -> whole-rally IK -> quality checks, per rally
package = write_package(kept_clips(rallies))  # the kept motion and every stroke's check values
```

`retarget` (`lib/retarget/run.py`) runs in three steps:

1. **Reach** (`optimize.solve_reach`), per session. For every contact, the waist and the paddle arm try to hold the
   blade on its path at five instants through ±0.1 s, with the legs in the ready stance.
   - It is solved once from the human's own stance, then again with a free body offset.
   - Where the robot's shorter arm cannot reach from the human's stance, that offset becomes the body's move.
2. **Targets** (`targets.build_targets`), per rally. The human's motion is mapped onto the robot's proportions (the
   joint map below).
   - The stance moves where the reach needs it.
   - A planted foot whose stance moves more than 5 cm takes a 0.24 s step instead of sliding.
   - The paddle path passes each contact at the paddle state solved from the ball.
3. **Solve and check**, per rally, in its own process.
   - The solve (`optimize.Problem.solve`): a per-frame Levenberg-Marquardt IK, then 200 Adam steps and 6000 L-BFGS
     evaluations over the whole rally, all in float64 on one CPU thread. (Levenberg-Marquardt, Adam and L-BFGS are
     standard numerical optimisers: each repeatedly adjusts the joint angles to lower a cost.)
   - The quality checks (`validate.grade_segment`) then grade every stroke and every frame.

**The joint map** (human key joint to robot body):

| Human | Robot |
|---|---|
| `Hip` | Pelvis. Its horizontal path is unscaled, so contacts keep their place at the table. Its height is scaled by the leg ratio, and its orientation is taken relative to the rest pose. |
| `Spine02` | Torso orientation, relative to the rest pose. It is followed less closely near contacts, so the shorter arm can reach with the waist. |
| `L_Foot`/`R_Foot`, `L_ToeBase`/`R_ToeBase` | The feet. The ankle is placed about the pelvis at the leg ratio, and the foot's yaw runs from heel to toe. While planted, the sole is flat and does not move. |
| `L_Hand` − `L_Upperarm` | The free hand about its shoulder, in the torso frame, at the arm ratio. |
| The paddle | The robot's paddle body. Its blade centre and face normal follow the path. |

For the G1, the leg ratio is 0.84 and the arm ratio 0.75.

**The cost**, in priority order:
- the blade on the paddle path (1000× harder near contacts, exact at the contact);
- pelvis and torso;
- feet: planted, flat and rigid;
- balance: the zero-moment point inside the planted soles;
- smoothing: joint accelerations weighted by inertia over peak torque, a speed margin, the free hand, and coarse-sphere
  self, table and floor penalties.

Joints are kept inside 0.90 of their ranges, Isaac Lab's soft limits.

## The quality checks

Every threshold, and its reason, is in `validate.py`. No threshold is loosened to make a stroke pass. A failing stroke
is rejected with its first failing check, in this order:

| Check | Passes when |
|---|---|
| **reach** | **Paddle path:** through ±0.1 s of the contact, graded every 1 ms, the blade centre stays within 1 cm of the path and the face normal within 5°. The same holds at the contact against the paddle state solved from the ball. **Joint limits:** every joint stays within 0.95 of its range. |
| **speed** | Every joint stays below 0.8 of its speed limit. |
| **torque** | Inverse dynamics (`dynamics.py`) stay within each motor's torque at its speed: the peak up to the knee speed, falling linearly to zero at the no-load speed. Foot contact forces are solved in a μ 0.7 friction cone at the sole corners. |
| **balance** | The wrench the feet cannot supply moves the capture point at most 3 cm over any 0.2 s. That is half the G1's sole width, a push it absorbs without stepping. |
| **feet** | **Foot skate:** no planted sole slides more than 1 cm. **Floor:** no sole sinks more than 5 mm into the floor. |

Speed, torque and balance are graded from 0.5 s before to 0.3 s after each contact. Every frame the package keeps must
pass too: a rally is cut 0.6 s before to 0.4 s after every rejected stroke, and 0.3 s either side of any failing frame.
Only pieces of at least 1 s are kept.

**Each run checks itself:**
- every selected rally is solved and checked;
- **negative control:** the same motion graded against the capture's raw paddle must fail the paddle-path check on at
  least half the strokes. The raw paddle is the suit's hand roll, 11 to 30 cm off on the probe rally, and this shows
  the check is live;
- the written `motion.npz` is re-read and re-checked: joint limits, joint speed, foot skate, the floor, and the
  packaged strokes' paddle-path errors;
- the kept strokes are published as `ping-pong/<robot>-retargeted-strokes`, versioned by the first 8 characters of
  the run ID.

The tests (`tests/retarget`) keep a stroke the G1 can make, then reject four corruptions of it: the robot 2 cm off its
path, the feet skating 5 cm, a joint at 0.98 of its range, and a joint past its speed limit.

## What a new robot must provide

Write a module like `lib/robots/unitree_g1.py` and register its `RobotSpec` in `lib/robots/__init__.py`. The
retargeter reads these fields:

1. **`urdf`, `urdf_sha256`.** The kinematic tree with revolute joints, joint limits and link inertias. Every link
   frame is the URDF's. The URDF's joint names must be the spec's `joint_names`.
2. **`actuators`.** Per motor group, for every joint:
   - peak torque;
   - speed limit;
   - knee speed: the peak torque holds up to it;
   - no-load speed: the torque falls to zero there;
   - rotor armature.
3. **`default_pose`.** The ready stance: the solver's weak pull, and the reach solve's legs.
4. **`paddle`.**
   - The wrist link it is fixed to, and the blade centre's offset along that link's +x.
   - The paddle frame is the link's own axes: x towards the tip, y the forehand face normal, z across. If your wrist
     link's x does not run along the forearm, add a link (or a fixed joint) whose axes do.
   - Mass, centre of mass and inertia; they ride on the link.
5. **`links`.** `pelvis` (the floating base), `torso`, and `feet` (left, right).
6. **`retarget`** (`RetargetSpec`):
   - **`racket_arm`, `free_arm`.** Shoulder, elbow and wrist links of the paddle arm and the other arm. HIM's player
     holds the paddle in the right hand, so the free arm takes the human's left hand.
   - **`urdf_drop_links`.** URDF links to leave out, such as the hand the paddle replaces.
   - **`sole_corners_m`.** The sole's contact points in the foot link's frame: two heel points, then two toe points.
     They define a planted, flat foot, the support polygon, and the contact forces.
   - **`collision_spheres`.** Coarse spheres per link: (centre in the link frame, radius). The self, table and floor
     penalties use them. The trunk, legs and forearms are found from the URDF chains and the arms above.

Everything else comes from the kinematic tree:
- **Reach joints:** the chain to the paddle link (G1: waist and right arm).
- **Leg joints:** the chain to each foot.
- **Scale ratios:** the hip-joint-to-ankle and shoulder-to-wrist lengths at zero joint angles.

Then add the robot's name to `RobotName` and a case to `RETARGET_PRESETS` in `lib/config.py`, run that case, and read
`strokes.csv` in its package. Its `cause` column says what the robot could not do.

## The human strokes: `ping-pong/him-mocap-strokes@v1`

**Contents.** A zip of `manifest.json` and `mocap.npz`. `lib/retarget/source.py` reads it.
- **Who:** HIM's suited player in all 7 recorded sessions: 59,154 capture frames and 737 contacts.
- **Paddle state at each contact:**
  - 446 contacts have the state solved from the ball (`paddle_inverse`): blade centre, struck-face normal and face
    velocity;
  - 291 have no ball solution and keep the capture's own paddle (`capture`), labelled as such.

**Frame and clock.**
- **Frame:** robot-canonical table frame. The origin is the table-top centre, z is up, and the player stands at -X;
  sessions recorded from the other end are yawed 180°, never mirrored. The floor is at z = -0.76 m.
- **Clock:** time is the camera clock (real time). Frames come at the capture's 30 Hz of mocap time, which runs 2–4 %
  fast, so `time_s` steps are 32–35 ms and every velocity is in real time.

| Array (per session `s<k>/`) | Shape | Meaning |
|---|---|---|
| `time_s`, `clip`, `donor_frame` | (N,) | Camera time; continuous capture stretch; the capture frame |
| `joint_pos`, `joint_quat` | (N, 21, 3), (N, 21, 4) | The 21 key joints (`joint_names`): world position and orientation (w, x, y, z) |
| `paddle_pos`, `paddle_quat` | (N, 3), (N, 4) | The capture's paddle (axes: X across, Y tip, Z forehand normal) |
| `foot_contact` | (N, 2) | Left and right foot planted |
| `stroke_id`, `stroke_source`, `stroke_t`, `stroke_face` | (S,) | Each contact: time, and rubber (0 forehand, 1 backhand) |
| `stroke_blade_p`, `stroke_normal`, `stroke_vel`, `stroke_donor_blade_p` | (S, 3) | The paddle state, and the capture's own blade centre there |
| `stroke_speed_mps`, `stroke_contact_sd_m`, `stroke_normal_sd_deg`, `stroke_ball_v_out` | (S,), (S, 3) | Solved stroke speed, the solution's uncertainty, and the predicted ball exit velocity (NaN for `capture`) |

Global arrays: `joint_names` (21), and `rest_quat` (21, 4), the key joints at the body's rest pose. The manifest adds
the body's rest lengths (leg, arm, hip and ankle height) and the lineage of every input by sha256.

**How it was built.** `tools/build_mocap_strokes.py` built it from four inputs:
- the donor library: HIM's capture retargeted onto the suited player's game body and conditioned onto a shakehand grip;
- the ActorCore rest skeleton;
- the mocap-to-table registration;
- the paddle states solved from the ball.

**Verification.** The bundle matches the inputs the reference output was solved from. Times, stroke ids, rubbers,
sources and contact states are identical. Poses agree to 1.2 × 10⁻⁷ m and 1 × 10⁻⁷ rad (float32 storage).

## The package: `ping-pong/<robot>-retargeted-strokes`

**Contents.** A zip of `manifest.json`, `motion.npz` and `strokes.csv`:
- `strokes.csv` has every stroke, with its check values, verdict and first failing check;
- the manifest records the robot, its URDF hash, the source bundle's sha256, the solver settings and the thresholds.

**`motion.npz`** has the layout of `ping-pong/g1-retargeted-strokes@v2`. Its arrays are flat over every kept frame
at 50 Hz, with per-clip offsets `clip_start`:
- **Joints:** `joint_pos` and `joint_vel` (central differences).
- **Bodies:** `body_pos_w`, `body_quat_w`, `body_lin_vel_w` and `body_ang_vel_w`, covering every link, then `paddle`.
- **Root and paddle:** `root_*` and `paddle_*`, the same values for the pelvis and the paddle.
- **Per frame:** `foot_contact`, `contact` (the frame nearest a stroke), `t_to_contact_s` and `time_s`.
- **Per clip:** `clip_session`, `clip_robot_end` and `clip_t0_s`.
- **Per kept stroke:** `stroke_id`, `stroke_clip`, `stroke_frame`, the contact target `stroke_blade_p/v/n`, and its
  check values.

`joint_names` and `body_names` name the columns; reorder by name when loading. The frame is the table frame above.

## Running it yourself

The retargeter is plain Python on the CPU, so it also runs on your own computer, with no Antioch session. First
download the human strokes (33 MB) into `outputs/`, which git and Antioch both ignore (run from `ping-pong/`, signed in
to the HIM Robotics organization):

```bash
mkdir -p outputs
uv run antioch asset pull ping-pong/him-mocap-strokes -v v1 -o outputs/
```

Then, in `uv run python`, solve the probe rally and write its package:

```python
from pathlib import Path

from lib.robots import ROBOTS
from lib.retarget.model import robot_model
from lib.retarget.source import load_mocap_strokes
from lib.retarget.run import retarget, kept_clips
from lib.retarget.package import write_package

human = load_mocap_strokes("outputs/him-mocap-strokes-v1.zip", sessions=(1,))
robot = robot_model(ROBOTS["unitree-g1"])
rallies = retarget(human, robot, rallies=("s1_r001",), workers=1)
write_package(Path("outputs/g1-strokes"), robot, kept_clips(rallies), [s for r in rallies for s in r.strokes], {})
```

On one thread of a workstation CPU, the probe rally's 899 frames (50 Hz) take 4 minutes, about 0.27 s per frame. The
full run is 69,196 frames in 191 rallies, spread over `workers` processes; the longest rally, 1,857 frames, sets its
shortest wall time.

## Limits

- **Rigid-body checks, not simulation.** Torque and balance use the URDF's inertias, rigid feet and 100 Hz finite
  differences. A tracking policy in PhysX is the real dynamic test.
- **The reference.** The paddle states solved from the ball are uncertain: contact p50 3.7 cm, normal p50 13°. That is
  wider than the 1 cm and 5° checks, so the check measures fidelity to the solved state, not to the truth.
- **Coarse collision.** Self, table and floor contact use coarse spheres in the solve, not the simulator's colliders.
- **Face velocity.** It is matched by the construction of the path but not checked.
- **No time-warp.** A stroke too fast for the robot is rejected, not slowed down.
- **What makes strokes infeasible for the G1.** Its arm is 0.75 of the human's, so 77 % of the solved strokes need a step. Balance
  and the waist motors' torque reject more strokes than the blade does.
