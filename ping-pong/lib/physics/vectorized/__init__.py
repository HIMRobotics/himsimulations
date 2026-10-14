"""
The calibrated reference ball physics, batched over environments in torch (GPU), for robot training.

The numpy oracle (:mod:`lib.physics.reference`) simulates one ball at a time.
Phase two trains a humanoid against a live ball in thousands of parallel Isaac
Lab environments; that ball must obey the same calibrated physics, which the
native PhysX ball does not (it fails the C8 convergence gate and has no
tangential restitution, normal-spin retention, net catch law or rigid-paddle
impact). This package is the oracle's physics as torch tensors of shape
``(N, ...)``, one row per environment:

- :mod:`.params`: the contract (:mod:`lib.physics.params`) as per-environment tensors, so every environment can carry
  its own domain-randomised parameters;
- :mod:`.flight`: drag, Magnus and spin decay, and the RK4 step of :func:`lib.physics.reference.flight.rk4_step`;
- :mod:`.geometry`: the table top and board, the net (mesh and cord), the posts and clamps, the floor, and the blade
  (:mod:`lib.physics.reference.blade`'s touch condition), as signed gaps and contact frames;
- :mod:`.impact`: :func:`lib.physics.reference.impact.resolve_impact` (point mass, the off-centre rigid paddle, the
  net-mesh catch law), batched;
- :mod:`.sim`: :class:`BallBatch`, the event-driven step: integrate, find every environment's first touch inside the
  step to the oracle's root, resolve it, and carry on from the touch, with an event log that maps to
  :func:`lib.physics.canonical_events.canonical_events`.

torch is imported by these modules; the demo's own environment does not install it (the ``lab`` service image has it),
and the tests skip where it is absent.
"""

from lib.physics.vectorized.params import VecParams
from lib.physics.vectorized.sim import BallBatch, EventLog, PaddleSegments

__all__ = ["BallBatch", "EventLog", "PaddleSegments", "VecParams"]
