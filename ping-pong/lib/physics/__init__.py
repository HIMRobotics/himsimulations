"""
Physics: the parameter contract and the engine-independent reference model.

``params`` is the single source of truth for physics parameters, units, priors
and randomisation ranges. ``reference`` is a pure numpy and scipy oracle (flight,
impact, event-driven simulation, camera projection and flight fits) that every
simulator backend and every trajectory fit is checked against. Nothing here
imports Isaac; see ``README.md`` for the equations and conventions.
"""
