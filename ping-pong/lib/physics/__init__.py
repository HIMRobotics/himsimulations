"""
The calibrated table-tennis ball physics.

- :mod:`.params` is the single source of truth for the physical parameters, their units and priors
  (``data/priors.json``).
- :mod:`.reference` is the pure numpy and scipy oracle: flight, impact law, contact geometry and event-driven
  simulation. Every simulator backend is checked against it.
- :mod:`.engine` realises the oracle in Isaac Sim for the replay (the per-step callback, hybrid contacts, rk4).
- :mod:`.vectorized` is the oracle batched in torch on the GPU, and :mod:`.robot_coupling` couples it to a robot's
  paddle in Isaac Lab.

Nothing here imports Isaac at module scope. ``docs/physics.md`` explains the equations and conventions.
"""
