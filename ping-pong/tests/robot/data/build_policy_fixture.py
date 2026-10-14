"""
One-off: build ``policy_observations.npz``, the evidence ``tests/robot/test_actor_fold.py`` and
``tests/robot/test_forecast.py`` check the policy's port against.

It runs against a checkout of the table-tennis source pipeline (the code the policy was trained and graded with), never
against this repository, and the tests never import it. It writes:

- **Flights**: recorded HIM serves and rally shots and designed sweep shots from the source's novel-shot set, and the
  opponent's recorded feeds, each flown at nominal physics (1 ms, CPU, float64) and sampled at a few instants, one of
  them after a serve's own-half bounce and the last after many balls are dead;
- **Robot poses**: one per flight, from play's reset distribution (pelvis 0.35-0.75 m behind the robot's end line);
- **The source's ball forecast** at every instant, with the serve fix (``forecast_fixed``) and without it
  (``forecast_unfixed``), from the source's own ``predict_flight`` and ``forecast_columns_from``;
- **The checkpoint's 248 inputs** (``obs``, float32), as the source's grade built them: the robot's 116 values (the
  proprioception drawn from the actor's own normaliser, the pose and the ball from the flight), the forecast, the hidden
  stroke reference at the normaliser's mean, the hidden command at zeros, velocities clipped;
- **``actions_rsl_rl``**: rsl_rl 5.0.1's own ``MLPModel`` (deterministic output) with the checkpoint's actor loaded, on
  those inputs.

Run (CPU; torch, numpy and rsl_rl 5.0.1 with tensordict importable)::

    python -I tests/robot/data/build_policy_fixture.py <source checkout> <model_final.pt> <rsl_rl site dir> [extra sys.path ...]
"""

# The imports inside main() are the source checkout's `lib` and rsl_rl, not this repository's: neither sort nor resolve here.
# ruff: noqa: I001
# pyright: reportMissingImports=false

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

SEED = 20261014
"""The draws' seed: which shots, the robot's poses, the proprioception."""

COUNTS = {"serve": 24, "rally": 12, "sweep": 4, "feed": 8}
"""Flights per kind."""

INSTANTS_S = (0.0, 0.1, 0.25, 0.4, 0.55, 1.2)
"""Times after the launch at which the live ball is sampled (one more is added after the serves' own-half bounce); by
1.2 s many balls are dead on the floor."""

BALL_STEP_S = 1e-3


def main(source: Path, checkpoint: Path, rsl_rl_site: Path, extra: list[Path], out: Path) -> None:
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(rsl_rl_site))
    sys.path.extend(str(p) for p in extra)

    import numpy as np
    import torch

    torch.set_num_threads(2)
    from lib.environment.actors.opponent import RuleBook
    from lib.physics.params import nominal_params
    from lib.physics.vectorized.params import VecParams
    from lib.physics.vectorized.sim import BallBatch
    from lib.scenarios.robot.env import OBSERVATION_TERMS
    from lib.scenarios.robot.training.rally import feed_states, forecast_columns_from, predict_flight
    from lib.scenarios.robot.training.runner import observation_clip_bounds
    from lib.scenarios.robot.training.shots import REAL, SWEEP
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    f64 = torch.float64
    rng = np.random.default_rng(SEED)

    # ---- the flights ---------------------------------------------------------------------------------------------
    shots = np.load(source / "lib/scenarios/robot/training/data/incoming_shots_v1.npz", allow_pickle=False)
    kind, serve_col = shots["kind"], shots["serve"].astype(bool)
    pools = {
        "serve": np.flatnonzero((kind == REAL) & serve_col),
        "rally": np.flatnonzero((kind == REAL) & ~serve_col),
        "sweep": np.flatnonzero(kind == SWEEP),
    }
    picked = {name: np.sort(rng.choice(pool, COUNTS[name], replace=False)) for name, pool in pools.items()}
    rules = RuleBook.load(device="cpu", dtype=f64)
    feeds = feed_states(rules, COUNTS["feed"], np.random.default_rng(SEED + 1))
    balls = np.concatenate([shots["states"][picked["serve"]], shots["states"][picked["rally"]], shots["states"][picked["sweep"]], feeds])
    names = ("serve", "rally", "sweep", "feed")
    flight_kind = np.concatenate([np.full(COUNTS[name], name, dtype="<U5") for name in names])
    serve = flight_kind == "serve"
    n = balls.shape[0]
    rows = torch.arange(n)
    params = VecParams.from_params(nominal_params(), n, device="cpu", dtype=f64)
    start = torch.as_tensor(balls, dtype=f64)

    # each flight predicted once, at its start, as the source's forecast stores it
    fixed = predict_flight(params, start, rows, 0.0, torch.as_tensor(serve))
    unfixed = predict_flight(params, start, rows, 0.0, None)
    t_pre = fixed.t_pre.numpy()[serve]
    t_pre = t_pre[np.isfinite(t_pre)]
    after_own_bounce = round(float(np.median(t_pre)) + 0.03, 3)
    instants = sorted({*INSTANTS_S, after_own_bounce})

    # the live ball at each instant, flown at nominal physics
    batch = BallBatch(params, start, t0=0.0, terminal_kinds=("floor",), graze=True, compile=False)
    live, active, done = [], [], 0
    for t in instants:
        while done < round(t / BALL_STEP_S):
            batch.step(BALL_STEP_S)
            done += 1
        live.append(batch.state.clone())
        active.append(batch.active.clone())

    # ---- the robot's poses -----------------------------------------------------------------------------------------
    behind = rng.uniform(0.35, 0.75, n)
    yaw = rng.uniform(-0.15, 0.15, n)
    position = np.stack([-(1.37 + behind), rng.uniform(-0.35, 0.35, n), np.full(n, 0.78)], -1)
    rotation = np.zeros((n, 3, 3))
    rotation[:, 0, 0], rotation[:, 0, 1], rotation[:, 1, 0], rotation[:, 1, 1], rotation[:, 2, 2] = np.cos(yaw), -np.sin(yaw), np.sin(yaw), np.cos(yaw), 1.0
    table_offset = np.tile([0.0, 0.0, 0.76], (n, 1))
    pos_t, rot_t, off_t = (torch.as_tensor(x, dtype=f64) for x in (position, rotation, table_offset))

    # ---- the source's forecast -------------------------------------------------------------------------------------
    forecast_fixed, forecast_unfixed = [], []
    for t, ball, live_rows in zip(instants, live, active, strict=True):
        forecast_fixed.append(
            forecast_columns_from(
                params,
                ball,
                live_rows,
                t,
                fixed.t_bounce,
                fixed.after,
                fixed.ok,
                fixed.toward_robot,
                pos_t,
                rot_t,
                off_t,
                t_pre=fixed.t_pre,
                pre_after=fixed.pre_after,
            )
        )
        forecast_unfixed.append(
            forecast_columns_from(params, ball, live_rows, t, unfixed.t_bounce, unfixed.after, unfixed.ok, unfixed.toward_robot, pos_t, rot_t, off_t)
        )

    # ---- the checkpoint's 248 inputs -------------------------------------------------------------------------------
    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    actor_state = saved["actor_state_dict"]
    mean = actor_state["obs_normalizer._mean"][0].double()
    std = actor_state["obs_normalizer._std"][0].double()
    offsets, cursor = {}, 0
    for name, size in OBSERVATION_TERMS:
        offsets[name] = slice(cursor, cursor + size)
        cursor += size
    blocks = []
    to_base = rot_t.transpose(-1, -2)
    for k, (ball, live_rows) in enumerate(zip(live, active, strict=True)):
        env = mean[:116] + std[:116] * torch.as_tensor(np.random.default_rng(SEED + 10 + k).standard_normal((n, 116)), dtype=f64)
        env[:, offsets["base_height"]] = pos_t[:, 2:3]
        env[:, offsets["base_pose_table_xy_yaw"]] = torch.stack([pos_t[:, 0] - off_t[:, 0], pos_t[:, 1] - off_t[:, 1], torch.as_tensor(yaw)], -1)
        env[:, offsets["ball_pos_b"]] = (to_base @ (ball[:, :3] + off_t - pos_t)[..., None])[..., 0]
        env[:, offsets["ball_vel_b"]] = (to_base @ ball[:, 3:6][..., None])[..., 0]
        env[:, offsets["ball_active"]] = live_rows.to(f64)[:, None]
        reference = mean[146:234].expand(n, 88)
        command = torch.zeros(n, 14, dtype=f64)
        blocks.append(torch.cat([env, forecast_fixed[k], reference, command], -1))
    obs = torch.cat(blocks)
    bound = torch.tensor(observation_clip_bounds(248), dtype=f64)
    obs = torch.maximum(torch.minimum(obs, bound), -bound).float()

    model = MLPModel(
        TensorDict({"policy": obs}, batch_size=[obs.shape[0]]),
        {"actor": ["policy"]},
        "actor",
        29,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    model.load_state_dict(actor_state, strict=True)
    model.eval()
    with torch.no_grad():
        actions = model(TensorDict({"policy": obs}, batch_size=[obs.shape[0]]))

    np.savez_compressed(
        out,
        flight_kind=flight_kind,
        serve=serve,
        balls=balls,
        instants=np.asarray(instants),
        live=torch.stack(live).numpy(),
        active=torch.stack(active).numpy(),
        position=position,
        rotation=rotation,
        table_offset=table_offset,
        forecast_fixed=torch.stack(forecast_fixed).numpy(),
        forecast_unfixed=torch.stack(forecast_unfixed).numpy(),
        obs=obs.numpy(),
        actions_rsl_rl=actions.float().numpy(),
        checkpoint_sha256=np.asarray(hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
    )
    print(f"{out}: {n} flights x {len(instants)} instants {instants}; {obs.shape[0]} observation rows")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    here = Path(__file__).resolve().parent
    main(
        Path(sys.argv[1]).resolve(),
        Path(sys.argv[2]).resolve(),
        Path(sys.argv[3]).resolve(),
        [Path(p).resolve() for p in sys.argv[4:]],
        here / "policy_observations.npz",
    )
