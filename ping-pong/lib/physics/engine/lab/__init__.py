"""
The Isaac Lab batch adapter: thousands of ball worlds on PhysX GPU, checked against the reference model.

The calibration harness runs every experiment as one environment of a
vectorised Isaac Lab 3.0 scene (PhysX GPU, PGS solver, ``dt = 1/4000``).
This package realises the :mod:`lib.physics` contract there:

- :mod:`.batch` (pure): what one batch simulates, environment by environment,
  and what the adapter returns.
- :mod:`.vectorized` (pure, array-generic): the reference model's aerodynamic
  wrench, its impact law and the hybrid contact sweeps, written once for
  numpy (the tests compare them with :mod:`lib.physics.reference`) and run on
  torch tensors in the engine.
- :mod:`.materials` (pure): per-environment PhysX materials, masses and caps.
- :mod:`.contact_model` (pure): the paddle effective-mass decision.
- :mod:`.authoring` (USD, after Kit boots) and :mod:`.harness` (the stepping
  loop) are the only modules that touch the simulator.

The game harness's adapter (Isaac Sim 6.0.1, PhysX CPU) lives beside this
package in :mod:`lib.physics.engine`; the two share
:func:`lib.physics.engine.settings.canonical_settings` so the parity check
compares like with like.
"""
