# Ping pong

Table tennis on [Antioch](https://antioch.com), built from HIM Robotics' recorded sessions of 2026-08-15.

It comes with the HIM lab as a digital twin, calibrated ball physics, the seven recorded sessions, a trained Unitree G1
policy, and HIM's mocap strokes retargeted onto the G1. Start from it to train ping-pong policies for your own
humanoids.

| Scenario | Engine | What it does |
|---|---|---|
| `replay_him_session` | Isaac Sim | Replays a recorded HIM session with live ball physics. Both players bend their recorded strokes to meet the ball. |
| `play_ping_pong_policy` | Isaac Lab | The trained G1 rallies against the HIM opponent. |
| `load_humanoid_at_table` | Isaac Lab | Loads a humanoid with its paddle at the table, holding its pose. It is the template for your controller. |
| `retarget_him_strokes` | CPU | Retargets HIM's mocap strokes onto a humanoid. |

## Quick start

```bash
cd ping-pong
uv sync --locked --group dev
uv run antioch suite collect          # the four suites, without running anything
uv run antioch suite run replay-smoke
uv run antioch suite run robot-smoke
```
