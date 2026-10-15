# HIM Robotics simulations

Robot simulations built from HIM Robotics' real lab recordings, which you can run, change and learn from. They run on
[Antioch](https://antioch.com), a cloud platform for robot simulation: you edit code on your laptop, and the
simulators run on Antioch's cloud GPUs. You don't need a GPU of your own.

Each directory is one self-contained project:

| Directory | What it is |
|---|---|
| [`ping-pong/`](ping-pong/README.md) | Table tennis. HIM's recorded games replayed with measured ball physics, a Unitree G1 humanoid that learned to rally, a template for your own robot controller, and human strokes moved onto the robot. |

**Start here:** [`ping-pong/README.md`](ping-pong/README.md). It explains what's going on from scratch, and walks you
from installing the tools to your first runs and experiments.

You'll need an Antioch account in the HIM Robotics organization: every project here is locked to it.
`python tools/check_org_lock.py` checks that lock, and CI runs it with each project's lint, type checks and tests.
