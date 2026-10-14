# HIM Robotics simulations

Simulations built on [Antioch](https://antioch.com). Each simulation is a self-contained Antioch project in its own
directory.

| Directory | What it is |
|---|---|
| [`ping-pong/`](ping-pong/README.md) | Table tennis: HIM's recorded sessions replayed with calibrated ball physics, a trained Unitree G1, a humanoid-at-table template and a stroke retargeter. |

Every project is locked to HIM Robotics' Antioch organization; `python tools/check_org_lock.py` checks it, and CI runs
it with each project's lint, type checks and tests.
