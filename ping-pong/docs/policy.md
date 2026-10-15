# The trained G1 policy

A Unitree G1 humanoid holding HIM's red paddle rallies against a copy of HIM's opponent, learnt from the recordings.
It plays from the ball alone: nothing plans its strokes or tells it where to hit. This page covers what the policy
reads and does, how a run grades it, how play differs from the conditions it was graded under, and its limits.

## The short version

**A policy** is a robot's controller written as a function: every time step it reads an *observation* (numbers about
the robot and the world) and returns an *action* (a command for each motor). This one runs 50 times a second. It reads
146 numbers, including the robot's joint angles, its paddle, and where the ball is and will be. It returns 29 numbers,
a target angle for each joint.

The function is a small neural network, an *MLP* (multi-layer perceptron): three layers of 512, 256 and 128 neurons,
with the ELU activation between them. Nobody wrote its strokes by hand. It learned them.

**Reinforcement learning** (RL) is how. The robot plays in simulation, a *reward* function scores each moment (a ball
returned legally scores high; leaning on the table costs), and a training algorithm nudges the network's weights so
high-scoring actions become more likely. Repeat for millions of tries, in thousands of parallel worlds at once, and the
network gets good. This one was trained with **PPO** (proximal policy optimization), a standard RL algorithm that makes
small, careful updates so learning doesn't collapse. During training PPO also learns a second network, a *critic*
that predicts how well things are going; together they're called an *actor-critic*. Only the actor plays.

**How this one was produced**, in three stages:

1. **Imitation.** It learned to track HIM's human strokes, [retargeted](retargeting.md) onto the G1's body.
2. **Striking.** It learned to hit the ball at commanded contact points.
3. **Rallying.** It played this opponent from the ball alone, paid mostly for returns that land legally.

The training code isn't part of this repository. What ships is the result, the *checkpoint* (the network's saved
weights) `ping-pong/g1-rally-policy@e0e0bb45-it1500`, from iteration 1500 of training. The scene, the opponent and
the observation that a training loop of your own would build on are all here (see
[training your own](#training-your-own)).

The code is in `lib/policy/`:

- `actor.py`: loads the policy and folds it;
- `rally.py`: the rally against the opponent, and the ball forecast;
- `grade.py`: the grade;
- `film.py`: the 24 fps film;
- `run.py`: the scenario.

The robot and its scene are in `lib/robots/`, the opponent in `lib/opponent/`.

## Running it

```bash
uv run antioch scenario run play_ping_pong_policy --cases g1-rally-smoke   # 64 envs, 30 s
uv run antioch scenario run play_ping_pong_policy --cases g1-rally         # 2048 envs, 30 s
```

An *env* (environment) is one copy of the table scene, with its own robot, ball and opponent. Each run plays two
passes, every env reset at the start of each:

1. **The policy.** It plays 30 s of rallies, and env 0 is filmed in the HIM lab twin at 24 fps.
2. **The blind control.** The same actor plays 30 s with the ball and its forecast hidden from it. A policy that plays
   from the ball must collapse here. This is a *negative control*: a test that must fail, to prove the measurement
   can tell the difference.

Add `--stream` to watch env 0 live in Mission Control, through the film's own camera. Streamed, the run films that
side view only, without the overhead 8F view (a streamed Isaac Lab run hangs with it attached). The film stops when
the policy's pass ends, so the blind control plays on in the same view with no ball drawn. The README's
[Watch it live in Mission Control](../README.md#watch-it-live-in-mission-control) has the details.

## The policy in detail

The checkpoint is `ping-pong/g1-rally-policy@e0e0bb45-it1500`: an MLP actor, 512-256-128 with ELU, trained with PPO
(rsl_rl 5.0.1) in the three stages above: tracking HIM's strokes retargeted onto the G1, striking commanded contacts,
and playing rallies against this opponent from the ball alone.

The final stage hid the stroke reference and the commands from the actor, and paid for returns that land legally.
Smaller terms rewarded reach, contact and the return's quality, and the pelvis, torso and legs paid for leaning on the
table.

**What it reads: 146 values, 50 times a second.** The first 116 are `3 J + 29` for `J = 29` joints, all from
`lib/robots/table_scene.py`'s `observe`:

| Inputs | Values |
|---|---|
| The pelvis's linear and angular velocity, and gravity, in its own frame | 9 |
| The pelvis's height, and its position and heading at the table | 4 |
| Joint angles relative to the default pose, joint velocities, and the last action | 3 × 29 |
| The paddle's position, face normal and velocity, in the pelvis frame | 9 |
| The ball's position and velocity in the pelvis frame, and whether it is in play | 7 |
| The ball forecast (`lib/policy/rally.py`) | 30 |

**The ball forecast** is physics only. It holds:

- the ball's predicted bounce on the robot's half: the point, the time to it, and the velocity and spin just after;
- the ball's position 0.1, 0.2, 0.3 and 0.4 s ahead;
- the ball's live spin.

The bounce and each look-ahead carry a valid flag. A flight is predicted once, when it starts, with the calibrated
free flight and table law.

A serve bounces first on the server's own half. The forecast predicts it through that bounce to its bounce on the
robot's half, and is always on. Without it the policy returns 22 % of HIM's recorded serves; with it, 98 %. The
opponent's rally feeds are not serves, so the fix does not change them.

**What it does: 29 joint targets.** Each target is `default_pose + action_scale * action`, and the joint's PD gains
drive the joint there. PD gains set how stiffly (P, proportional) and with how much damping (D, derivative) a motor
holds its target. The action scale is a quarter of each joint's peak torque over its stiffness: 0.031 rad for the
wrist's small motor, 0.156 rad for the arms, 0.11-0.35 rad for the legs and waist.

**Why it is folded.** During training the network had extra inputs that play doesn't use, and *folding* removes them
without changing a single output. The checkpoint's first layer reads 248 values. The other 102 are the stroke reference (88) and the
strike command (14), and at play both are constant:

- the hidden reference reads its normaliser mean;
- the hidden command reads zeros.

Their contribution is folded exactly into the first layer's bias, so play needs neither, and loading needs only torch.
`tests/robot/test_actor_fold.py` checks the folded actor against rsl_rl's own model on observations built by the
source pipeline, serves included. The largest difference is 5.7e-6 in action units.

## The opponent

`lib/opponent/` is HIM's opponent, distilled from HIM's recorded sessions and batched in torch.

**How it plays.**
- At each robot hit it perceives the ball and predicts its bounce. A trained mixture network draws its intent: when to
  meet the ball, where to land it and with what spin.
- Rules fit to the recordings decide whether and how it reaches the ball: the reaction time, the footwork, the arm's
  reach, a body-feasibility model and the execution noise.
- Its paddle is a kinematic blade through the contact, and the calibrated ball physics meets it like the robot's.

**Rally openings.** Each rally opens with one of the opponent's recorded feeds, about 4.5 m/s with no spin, aimed at the
robot.

**Its model.** It loads from the `ping-pong/him-opponent-model@v2` asset.

## The grade

`lib/policy/grade.py` reads, per pass:

- **Legal returns per receive, the headline.** A receive is a ball the opponent sent that is bound for the robot's half.
  Balls going out are let go, and are the opponent's error. A legal return is the robot's hit whose first table contact
  lands on the opponent's half; net touches are allowed.
- **The hit rate.** Also legal given a hit, rally length in legal shots, why rallies ended, and falls per 10 s.

Rallies cut by a fall, and rallies still in flight when the pass ends, are left out. The intervals are Wilson 95 %
confidence intervals, the range the true rate most likely lies in. They are too narrow: receives in one rally are not
independent.

**The run's checks** are the harness's and the physics'. The policy's own numbers are reported, never gated: a
policy is judged on its evidence, and yours will read differently.

| Check | Passes when |
|---|---|
| The blind control collapses | at most 0.02 legal per receive with the ball hidden |
| PhysX buffers not full | every GPU buffer kept headroom and PhysX reported no overflow |
| Both passes ran their span | the policy's and the control's passes each simulated their full 30 s |
| Film at 24 fps | a frame for every 1/24 s of the policy's pass, none dropped |

The run records legal per receive with its interval, beside the source's readings below.

**What the source grades read.** These were measured where the policy was trained, as legal per receive:

- 30 s of rallies (2048 envs): 0.963, and 0.962 on a second seed;
- a 120 s pass: 0.969;
- the opponent's feeds as single receives, under nominal physics: 0.958;
- with the ball hidden: 0.000.

Its hit rate was 0.98–0.99, its rallies ran 7.5 legal shots on average, and it fell 0.002–0.006 times per 10 s.

## Play against the source's grade conditions

Play keeps the grade's robot, opponent, feeds and timing, and its random shoves: every 5–8 s each robot's pelvis gets
a horizontal kick of up to 0.5 m/s, as in training. That is why the robot in the film occasionally stumbles and
recovers. Three things differ:

- **Ball physics is nominal.** Every rally uses the calibrated priors' centre. The grades drew each rally's physics
  from the training distribution.
- **A reset starts at the ready stance.** A reset after a fall, or at a pass's start, puts the robot at its default pose
  within its stance spread. The grades reset it to a held-out frame of its training strokes. The robot stands 0.5 s
  before its next rally either way.
- **The driver alone calls legal returns.** The grades also ran an independent referee beside the driver, whose calls
  agreed within 0.004.

## Limits

- **Privileged observation.** The policy reads information a real robot couldn't have: the ball's exact state and
  spin, and a forecast from the same physics that flies the ball. A real robot would need perception (cameras and
  software to find the ball) and a predictor.
- **Hardware.** The G1's actuators are Unitree's published motor table and deployed gains, not validated against a
  real G1. Its ankles and waist roll and pitch use the actuator table's 50 N m and 37 rad/s, where Unitree's URDF writes
  35 N m and 30 rad/s.
- **Heavy backspin fails.**
- **Fast balls.** With little spin, its return rate falls off from about 9 m/s, where it is about 0.7. From 11 m/s it is
  0.50 or below, at any spin.
- **Rally length.** It is bounded by the opponent as much as by the robot.
- **One robot.** The policy is the G1's alone. Another humanoid needs its own policy; start from
  `load_humanoid_at_table` (`lib/robots/template.py`) and `lib/robots/spec.py`, which lists what a new humanoid
  provides.

## Training your own

The pieces a training loop needs are all here:

- **The scene:** `build_table_scene(robot, num_envs, hooks=..., opponent=True)`. It holds the robot, the calibrated
  ball coupled to its paddle every physics step, and the table.
- **The opponent and its feeds:** `CoupledRally`, as the scene's hooks.
- **The observation:** `observe`, and the forecast.

The PPO, curriculum and reward code that trained this policy is not part of this repository. To train your own you
would add an RL library (this policy used rsl_rl), a reward function that scores what you want the robot to do, and
usually a curriculum: a sequence of tasks from easy to hard, like the three stages above. The template,
`load_humanoid_at_table` in `lib/robots/template.py`, is the simplest scene to start from.
