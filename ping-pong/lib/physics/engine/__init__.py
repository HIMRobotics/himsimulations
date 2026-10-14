"""
Engine adapters: how the simulators realise the :mod:`lib.physics` contract.

The reference model in :mod:`lib.physics.reference` is the oracle; an adapter
configures an engine so its ball flies and bounces like it, and imposes the
contacts the engine cannot represent. The game harness's adapter (Isaac Sim
6.0.1, PhysX CPU with CCD and the PGS solver) lives in the modules beside this
one: :mod:`lib.physics.engine.settings` (pure, canonical settings derived from a
parameter set, shared with the calibration harness for parity),
:mod:`lib.physics.engine.contacts` (pure contact detection and the hybrid paddle
contact), :mod:`lib.physics.engine.physx` (USD/PhysX authoring) and
:mod:`lib.physics.engine.stepper` (the per-physics-step callback). The Isaac Lab
batch harness's adapter lives in ``lab/``.
"""
