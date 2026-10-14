# Ping pong

Table tennis on [Antioch](https://antioch.com), built from HIM Robotics' recorded sessions of 2026-08-15: two overhead
Orbbec RGB-D cameras, a Rokoko motion-capture suit and a paddle IMU in HIM's lab.

It comes with everything a ping-pong robot needs in simulation:

- **the HIM lab as a digital twin**: the room, table, net, calibrated lighting and the two overhead cameras at their
  calibrated poses;
- **calibrated ball physics**: drag, Magnus, spin decay and the impact laws, checked against an independent reference
  model and HIM's recorded bounces;
- **the seven recorded sessions**, replayed with live physics;
- **a trained Unitree G1** that rallies against a distilled copy of HIM's opponent;
- **a bare humanoid-at-table template** for your own controller;
- **HIM's mocap strokes retargeted onto the G1**, with the retargeter to do it for your humanoid.

| Scenario | Engine | What it does |
|---|---|---|
| `replay_him_session` | Isaac Sim | Replays a recorded HIM session. The ball is live physics from the first serve; both players follow their recorded motion and bend each stroke, within human bounds, to meet it. The virtual cameras render beside the recorded frames. |
| `play_ping_pong_policy` | Isaac Lab | The trained G1 rallies against the HIM opponent in thousands of parallel worlds, beside a blind control. |
| `load_humanoid_at_table` | Isaac Lab | A humanoid with its paddle at the table and the calibrated ball, holding its pose: replace one line with your controller. |
| `retarget_him_strokes` | CPU | Retargets HIM's mocap strokes onto a humanoid, checks each for reach, speed, torque, balance and feet, and publishes the kept strokes. |

## Quick start

You need [uv](https://docs.astral.sh/uv/) and the Antioch CLI signed in to HIM Robotics' organization. The project is
locked to it (`organization_id` in `antioch.yaml`), so a run can never land anywhere else.

```bash
cd ping-pong
uv sync --locked --group dev
uv run antioch suite collect                 # the four suites and their runs, without running anything
uv run antioch suite run replay-smoke        # Isaac Sim: session 2's first point, and the dead-table control
uv run antioch suite run robot-smoke         # Isaac Lab: the policy, the template and the retargeting probe
```

One scenario case at a time:

```bash
uv run antioch scenario run replay_him_session --case session-2-first-point
uv run antioch scenario run play_ping_pong_policy --case g1-rally-smoke
uv run antioch scenario run load_humanoid_at_table --case unitree-g1
uv run antioch scenario run retarget_him_strokes --case unitree-g1-smoke
```

Inside a session, `main.py` runs the same scenarios (`python main.py replay session-2-first-point`), and
`demo.ipynb` walks through the template scene and the trained policy step by step (`antioch session new --profile
robot`, then open the notebook).

## Suites and recorded runs

Each suite stays on one engine, so a session boots exactly one simulator.

| Suite | Engine | Runs | What passing means |
|---|---|---|---|
| `replay-smoke` | Isaac Sim | `replay_him_session[session-2-first-point]`, `[nc-dead-table]` | The first point passes every replay test; the dead table (restitution 0.30) fails its drop tests, as a negative control must |
| `robot-smoke` | Isaac Lab | `play_ping_pong_policy[g1-rally-smoke]`, `load_humanoid_at_table[unitree-g1]`, `retarget_him_strokes[unitree-g1-smoke]` | The policy rallies and its blind control collapses; the template scene is sound; the probe rally retargets |
| `replay-evidence` | Isaac Sim | `replay_him_session[session-2]` | All of session 2 passes every replay test |
| `robot-evidence` | Isaac Lab | `play_ping_pong_policy[g1-rally]` | The policy at 2048 worlds, 30 s, against the opponent |

| Run | Result | Record |
|---|---|---|
| `replay-smoke` | pending | pending |
| `robot-smoke` | pending | pending |
| `replay-evidence` | pending | pending |
| `robot-evidence` | pending | pending |

## The replay

```python
def replay_him_session(run, params):  # lib/replay/run.py
    session = load_him_session(params)  # mocap, serves, strokes, camera epochs, real frames
    physics = engine_physics(params)  # the calibrated priors, or the negative control's dead table
    build_him_lab(stage, physics)  # room, table, net, colliders, calibrated lighting
    cameras = author_him_cameras(stage, camera_calibrations(session))
    ball = attach_live_ball(world, stage, physics, roles)  # hybrid contacts, calibrated impact law, RK4 at 3840 Hz
    players = spawn_players(stage, session)  # the recorded player and the inferred opponent
    recorder = open_recorder(run, params, cameras, session, physics)  # simulated beside real at 24 fps, and the MP4
    replay = ReplayLoop(run, session, nominal_params(), ball.physics, players.bodies, players.animators, recorder, ...)
    replay.drop_tests()  # ITTF 30 cm drops
    replay.play()  # every point, until the last ends physically
    return grade(run, params, session, replay)  # motion QA, physical invariants, telemetry, terminal
```

Each render tick (24 Hz) the referee reads the ball's contacts and applies the rules of a point; each player is posed
from the recording, and a stroke due soon is planned from the ball's predicted flight: a time warp, a blade offset and a
blade tilt within bounds set by the uncertainty of the recorded contact and how a human can re-plan. The arm IK carries
the paddle to the plan. When a simulated point ends before the recorded one, the next recorded stroke is played with a
labelled **continuity ball**, never counted as evidence.

The replay passes when all its tests pass and it ends on its terminal condition:

- **Motion QA**: no non-finite pose, no frame outside the recorded track, at most one glitch per minute, and the
  paddle never leaves the hand (1 cm on every frame).
- **Physical invariants**: no tunnelling, no ball leaving the room, no bounce gaining energy, at most four net
  touches per point, and the opening drop tests inside the ITTF band, 23-26 cm.
- **Telemetry completeness**: every rendered frame of every camera at 24 fps, every recorded Orbbec frame at its own
  time, 3D telemetry at 12 Hz, and an MP4 with every frame.

How many recorded strokes the simulation realises with a physical contact is reported, not graded: a live return
diverges from the recorded one.

## The robot

`lib/robots/` describes a humanoid as data, a `RobotSpec` (`spec.py`): its USD and URDF, joints, motors and PD gains,
action scale, ready stance, paddle mount, fall rule and starting stance. `unitree_g1.py` is the G1's.
`table_scene.py` builds any spec into an Isaac Lab scene with the table, net and calibrated GPU ball, coupled to the
paddle every physics step; `template.py` is the scenario that loads it and holds the pose.

**To put your own humanoid at the table**, write a module like `unitree_g1.py` with its `RobotSpec`, register it in
`lib/robots/__init__.py`, and add its name to `RobotName` and a preset in `lib/config.py`. The template, the retargeter and the ball forecast then
work for it; [`docs/retargeting.md`](docs/retargeting.md) covers the retargeting fields.

[`docs/policy.md`](docs/policy.md) covers the trained G1: what it reads, how it was trained, how a run grades it, and
how to train your own on this scene.

## Layout

```text
ping-pong/
  antioch.yaml         services (replay + compositor, robot), profiles, suites; locked to HIM Robotics
  scenarios.py         the four scenarios, thin wrappers over the run functions below
  main.py              the same scenarios from a shell inside a session
  demo.ipynb           the template scene and the trained policy, step by step
  lib/config.py        every preset (one per scenario case), services, simulator configs
  lib/assets.py        every asset by name, version and sha256
  lib/physics/         the calibrated ball: parameters, reference model, Isaac Sim adapter, GPU batch, robot coupling
  lib/him_lab.py       the HIM lab twin: room, table, net, lighting, colliders, cameras
  lib/telemetry/       24 fps H.264 camera streams, render products, the MP4 compositor client
  lib/replay/          replay_him_session: session, players, referee, loop, tests, recorder
  lib/robots/          RobotSpec, the G1, the table scene, the template
  lib/policy/          play_ping_pong_policy: the actor, the rally, the grade, the film
  lib/opponent/        the HIM opponent the policy plays
  lib/retarget/        retarget_him_strokes
  compositor/          the MP4 encoder service
  tools/               one-off builders for the published assets
  tests/               contract, physics, replay, robot and retargeting tests (CPU only)
  docs/                assets, physics, policy, retargeting
```

## Limits

- The policy reads privileged state: the ball's exact position, velocity and spin, and a physics-derived forecast of
  its flight. A real robot would need perception for those.
- The G1's motors are Unitree's published limits and deployed gains; they are not validated against a real G1.
- The policy fails on heavy backspin, and returns at most half the balls from 11 m/s at any spin.
- The replay realises about 41 % of recorded strokes with a physical contact; the rest are missed or played with
  continuity balls. It is reported, not graded.

## Development

```bash
uv sync --locked --group dev
uv run ruff check . && uv run ruff format --check . && uv run pyright
uv run pytest                                # CPU only: no simulator, no GPU, no remote calls
python ../tools/check_org_lock.py            # every project locked to HIM Robotics
```

CI (`.github/workflows/ping-pong.yml`) runs the same on every push. `tests/test_contract.py` keeps the project's
shape: the organization lock, exactly four scenarios and four single-engine suites, engine images pinned to the SDK,
simulator modules imported lazily, every asset pinned, and nothing internal shipped. The replay tests play session
2's first point end to end on a stand-in engine, and its dead-table control.

The documentation is in [`docs/`](docs): [assets](docs/assets.md), [physics](docs/physics.md),
[policy](docs/policy.md) and [retargeting](docs/retargeting.md).
