"""
The calibrated reference ball physics, batched over environments in torch (GPU), for robot training and play.

The numpy oracle (:mod:`lib.physics.reference`) simulates one ball at a time. A humanoid trains against a live ball in
thousands of parallel Isaac Lab environments, and that ball must obey the same calibrated physics, which the native
PhysX ball does not (it does not converge with the step, and has no tangential restitution, normal-spin retention,
net catch law or rigid-paddle impact). This package is the oracle's physics as torch tensors of shape ``(N, ...)``,
one row per environment:

- :mod:`.params`: the contract (:mod:`lib.physics.params`) as per-environment tensors, so every environment can carry
  its own domain-randomised parameters;
- :mod:`.flight`: drag, Magnus and spin decay, and the RK4 step of :func:`lib.physics.reference.flight.rk4_step`;
- :mod:`.geometry`: the table top and board, the net (mesh and cord), the posts and clamps, the floor, and the blade
  (:mod:`lib.physics.reference.blade`'s touch condition), as signed gaps and contact frames;
- :mod:`.impact`: :func:`lib.physics.reference.impact.resolve_impact` (point mass, the off-centre rigid paddle, the
  net-mesh catch law), batched;
- :mod:`.sim`: :class:`BallBatch`, the event-driven step: integrate, find every environment's first touch inside the
  step to the oracle's root, resolve it, and carry on from the touch, with an event log.

torch is imported by these modules; the ``robot`` service image provides it (CUDA), and the dev environment installs
the CPU build for the tests.
"""

from lib.physics.vectorized.params import VecParams
from lib.physics.vectorized.sim import BallBatch, EventLog

__all__ = ["BallBatch", "EventLog", "VecParams"]
