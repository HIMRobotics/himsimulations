"""
The ball physics a robot environment builds, with one precision switch: the step, the predictor and the inverse.

    physics = BallPhysics.create(params, n, device="cuda", precision="float64", blades=2)
    physics.batch.step()                                     # 1 ms, CUDA graphs
    prediction = physics.predictor.predict(state, tau)       # flight + one bounce, K candidates
    face, ok = physics.inverse(env, v_in, w_in, v_out, n, m_eff, offset, inertia)

``precision`` is ``float64`` (the oracle's answers: identical canonical sequences, positions to nanometres) or
``float32`` (2 to 3x cheaper on a GPU whose FP64 runs at 1/64 rate, with micrometre to sub-millimetre drift). Every
part of a build uses the same one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from lib.physics.params import BallTableParams
from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.dense import DenseBallBatch
from lib.physics.vectorized.inverse import GraphedInverse
from lib.physics.vectorized.params import VecParams
from lib.physics.vectorized.predict import BouncePredictor

Tensor = Any

Precision = Literal["float32", "float64"]


@dataclass
class BallPhysics:
    """
    One environment set's ball physics.

    :ivar params: The environments' physics tensors.
    :ivar batch: The step (:class:`DenseBallBatch`).
    :ivar predictor: Flight plus one bounce (:class:`BouncePredictor`).
    :ivar inverse: The stick-regime inverse impact law for ``inverse_rows`` rows (:class:`GraphedInverse`).
    :ivar precision: ``float32`` or ``float64``.
    """

    params: VecParams
    batch: DenseBallBatch
    predictor: BouncePredictor
    inverse: GraphedInverse
    precision: Precision

    @classmethod
    def create(
        cls,
        params: BallTableParams | Sequence[BallTableParams],
        n: int | None = None,
        *,
        device: Any = "cuda",
        precision: Precision = "float64",
        state: Tensor | None = None,
        dt: float = 1e-3,
        blades: int = 2,
        candidates: int = 5,
        predict_dt: float = 2e-3,
        horizon: float = 1.0,
        tau_max: float = 0.7,
        inverse_rows: int | None = None,
        log: bool = False,
        blade_edge: bool = True,
        **batch_options: Any,
    ) -> BallPhysics:
        """
        :param params: One parameter set for all ``n`` environments, or one per environment.
        :param precision: The dtype of every tensor of the build.
        :param state: The balls' initial states ``(N, 9)`` (default: at rest at the origin, to be reset).
        :param dt: The step, s.
        :param blades: Blades per environment (the robot's and the opponent's: 2).
        :param candidates: Delays per prediction.
        :param predict_dt: The predictor's step, s. On 500 states (rally returns and launches) predictions at 2 ms and
            4 ms steps agreed with the oracle's at 1 ms: the same valid bounces, positions within 1e-9 m; a 2 ms step
            halves the cost of a 1 ms one.
        :param horizon: How far ahead a prediction looks for the bounce, s.
        :param tau_max: The longest delay after the bounce, s.
        :param inverse_rows: Rows per inverse call (default ``N``).
        :param log: Keep the step's event log.
        :param blade_edge: Blades with their rounded edge (:attr:`VecParams.blade_edge`, the oracle's default); False is
            the face-only blade (an A/B and negative control).
        :param batch_options: Passed to :class:`DenseBallBatch`.
        """

        if precision not in ("float32", "float64"):
            raise ValueError(f"precision is float32 or float64, not {precision!r}")
        dtype = getattr(torch, precision)
        vec = VecParams.from_params(params, n, device=device, dtype=dtype)
        if not blade_edge:
            vec = dataclasses.replace(vec, blade_edge=False)
        start = torch.zeros(vec.n, 9, dtype=dtype, device=vec.device) if state is None else state
        batch = DenseBallBatch(vec, start, dt=dt, blades=blades, log=log, **batch_options)
        predictor = BouncePredictor(vec, candidates=candidates, dt=predict_dt, horizon=horizon, tau_max=tau_max)
        inverse = GraphedInverse(vec, inverse_rows if inverse_rows is not None else vec.n, rigid=True)
        return cls(vec, batch, predictor, inverse, precision)


__all__ = ["BallPhysics", "Precision"]
