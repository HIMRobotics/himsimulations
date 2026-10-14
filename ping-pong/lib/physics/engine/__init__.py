"""
The replay's engine adapter: how Isaac Sim 6.0.1 (PhysX CPU, CCD, PGS) realises the :mod:`lib.physics` contract.

The reference model (:mod:`lib.physics.reference`) is the oracle. The adapter configures PhysX so its bodies behave
like it, and imposes the contacts PhysX cannot represent:

- :mod:`.settings`: the canonical engine settings derived from a parameter set (pure);
- :mod:`.contacts` and :mod:`.sweeps`: contact detection, the hybrid table, net and paddle contacts (pure);
- :mod:`.ball_integrator`: the oracle's RK4 for the ball at the replay's physics step;
- :mod:`.physx`: USD/PhysX authoring;
- :mod:`.stepper`: the per-physics-step callback.
"""
