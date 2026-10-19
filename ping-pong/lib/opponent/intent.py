"""
The shot intent h2: what a receiver means to do with an incoming ball, before the rules make it concrete.

The generator's intent is ``PlayerModels.draw_shot`` (the offline data generator's, not shipped here): conditioned on the
predicted own-half bounce (x, y, velocity) and the receiver's stance, in the hitter frame, it draws the contact time
after the bounce, the landing target, the flight time, the outgoing topspin and sidespin, the rubber presented, the
blade offset from the hips, the grip roll (the donor's blade-tip direction) and the start of the normal solve.

- :class:`MdnIntent` draws it: a mixture density network over the same 13 continuous outputs with a Bernoulli rubber
  face per component, trained on the generator's own draws at the receive opportunities of the v1 release's body-first
  shards (``intent_mdn_v2`` in the ``him-opponent-model`` asset). It is the only learned part of the decision; the
  body-first feasibility's acceptance and nudge are fitted on the body runtime's own calls (:mod:`.body`).
- :class:`OutOfDistribution` is the generator's own test of an incoming ball against the recorded donors (each draw
  carries it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from lib.physics.vectorized._torch import torch

from .rules import PROFILES, RuleBook

Tensor = Any

INTENT_VERSION = 2
"""v2: distilled from the v1 release's body-first shards (the nominal and randomised shards)."""

TARGETS: tuple[str, ...] = (
    "after_bounce_s",
    "landing_x",
    "landing_y",
    "flight_s",
    "spin_top",
    "spin_side",
    "offset_x",
    "offset_y",
    "tip_x",
    "tip_y",
    "tip_z",
    "normal_a",
    "normal_b",
)
"""The intent's continuous outputs (hitter frame), in the MDN's order; the rubber face is the 14th, categorical."""

TAU_MAX_S = 0.70
"""The learned intent's contact time after the bounce is held to 0.05-0.70 s (review H4; the rules search 0.06-0.70 s)."""

FEATURES: tuple[str, ...] = ("bounce_x", "bounce_y", "bounce_vx", "bounce_vy", "bounce_vz", "stance_x", "stance_y")
"""The conditioning: the generator's draw features (``players.FEATURES``), hitter frame."""


@dataclass
class IntentDraws:
    """
    Intent draws, rows ``(M, D)`` (``D`` draws per receiver), hitter frame.

    :ivar guess: A launch-velocity guess (the donor's recorded outgoing velocity), ``nan`` where the model has none.
    """

    after_bounce_s: Tensor
    landing: Tensor
    flight_s: Tensor
    spin: Tensor
    face: Tensor
    offset: Tensor
    tip: Tensor
    normal_rel: Tensor
    guess: Tensor
    donor: Tensor
    ood: Tensor

    def flat(self) -> IntentDraws:
        """Rows ``(M * D, ...)``."""

        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.reshape((-1, *tuple(v.shape[2:])))
        return IntentDraws(**out)

    @classmethod
    def from_targets(cls, y: Tensor, face: Tensor, guess: Tensor | None = None, donor: Tensor | None = None, ood: Tensor | None = None) -> IntentDraws:
        tip = y[..., 8:11]
        tip = tip / torch.sqrt((tip * tip).sum(-1, keepdim=True)).clamp_min(1e-9)
        shape = y.shape[:-1]
        return cls(
            after_bounce_s=y[..., 0].clamp_min(0.05),
            landing=y[..., 1:3],
            flight_s=y[..., 3].clamp_min(0.15),
            spin=y[..., 4:6],
            face=face.long(),
            offset=y[..., 6:8],
            tip=tip,
            normal_rel=y[..., 11:13],
            guess=torch.full((*shape, 3), float("nan"), dtype=y.dtype, device=y.device) if guess is None else guess,
            donor=torch.full(shape, -1, dtype=torch.long, device=y.device) if donor is None else donor,
            ood=torch.zeros(shape, dtype=torch.bool, device=y.device) if ood is None else ood,
        )


def categorical(probs: Tensor, count: int, generator: Any = None) -> Tensor:
    """``count`` draws per row from unnormalised probabilities ``(M, K)`` by inverse CDF (no host synchronisation)."""

    cdf = torch.cumsum(probs, 1)
    u = torch.rand((probs.shape[0], count), dtype=cdf.dtype, device=cdf.device, generator=generator) * cdf[:, -1:]
    return torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True).clamp(max=probs.shape[1] - 1)


class OutOfDistribution:
    """The generator's out-of-distribution rule against the recorded donors (``PlayerModels.draw_shot``'s pool)."""

    def __init__(self, rules: RuleBook) -> None:
        a = rules.arrays
        h = rules.header
        dev, dt = rules.device, rules.dtype

        def t(x: Any) -> Tensor:
            return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dt, device=dev)

        self.scale = t(h["feature_scale"])
        features = np.column_stack([a["donor_bounce"][:, [0, 1, 3, 4, 5]], a["donor_stance"][:, :2]])
        self.tables = []
        for role in PROFILES:
            index = a[f"draw_index_{role}"]
            self.tables.append((torch.as_tensor(index, device=dev), t(features[index] / np.asarray(h["feature_scale"])), float(h["profiles"][role]["ood_cut"])))

    def out_of_distribution(self, profile: Tensor, query: Tensor) -> Tensor:
        """The generator's out-of-distribution rule: the nearest donor farther than the donors' own p99 spacing."""

        out = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
        for p, (_index, z, cut) in enumerate(self.tables):
            out = torch.where(profile == p, torch.cdist(query / self.scale, z).min(1).values > cut, out)
        return out


# ---- the mixture density network ------------------------------------------------------------------------------

MDN_COMPONENTS = 12
MDN_HIDDEN = 256


@cache
def _mdn_class() -> Any:
    """The MDN module class (defined on first use: torch is imported lazily)."""

    nn = torch.nn

    class _MDN(nn.Module):
        def __init__(self, inputs: int, outputs: int, components: int, hidden: int, profiles: int = 2) -> None:
            super().__init__()
            self.d, self.k = outputs, components
            tri = outputs * (outputs + 1) // 2
            self.body = nn.Sequential(
                nn.Linear(inputs + profiles, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
            )
            self.logits = nn.Linear(hidden, components)
            self.means = nn.Linear(hidden, components * outputs)
            self.chol = nn.Linear(hidden, components * tri)
            self.face = nn.Linear(hidden, components)
            rows, cols = torch.tril_indices(outputs, outputs)
            self.register_buffer("tri_rows", rows, persistent=False)
            self.register_buffer("tri_cols", cols, persistent=False)

        def forward(self, x: Tensor, profile_onehot: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            hdn = self.body(torch.cat([x, profile_onehot], -1))
            n = x.shape[0]
            logits = torch.log_softmax(self.logits(hdn), -1)
            means = self.means(hdn).reshape(n, self.k, self.d)
            raw = self.chol(hdn).reshape(n, self.k, -1)
            scale = torch.zeros((n, self.k, self.d, self.d), dtype=x.dtype, device=x.device)
            scale[:, :, self.tri_rows, self.tri_cols] = raw
            diag = torch.nn.functional.softplus(torch.diagonal(scale, dim1=-2, dim2=-1)) + 1e-3
            scale = torch.tril(scale, -1) + torch.diag_embed(diag)
            return logits, means, scale, self.face(hdn)

    return _MDN


class MdnIntent:
    """The distilled intent: a trained mixture density network (weights in ``data/intent_mdn_v<k>.npz``)."""

    def __init__(self, module: Any, x_mean: Tensor, x_std: Tensor, y_mean: Tensor, y_std: Tensor, meta: dict) -> None:
        self.module = module
        self.x_mean, self.x_std, self.y_mean, self.y_std = x_mean, x_std, y_mean, y_std
        self.meta = meta

    @classmethod
    def create(
        cls, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, *, device: Any = "cpu", dtype: Any = None, meta: dict | None = None
    ) -> MdnIntent:
        dtype = dtype or torch.float32
        module = _mdn_class()(len(FEATURES), len(TARGETS), MDN_COMPONENTS, MDN_HIDDEN).to(device=device, dtype=dtype)

        def t(v: np.ndarray) -> Tensor:
            return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=dtype, device=device)

        return cls(module, t(x_mean), t(x_std), t(y_mean), t(y_std), dict(meta or {}))

    @classmethod
    def load(cls, path: Path | str, *, device: Any = "cpu", dtype: Any = None) -> MdnIntent:
        """The trained network at ``path`` (``intent_mdn_v<k>.npz``, its ``.json`` beside it) on ``device``."""

        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
        model = cls.create(arrays.pop("x_mean"), arrays.pop("x_std"), arrays.pop("y_mean"), arrays.pop("y_std"), device=device, dtype=dtype, meta=meta)
        state = {key[len("param/") :]: torch.as_tensor(value) for key, value in arrays.items() if key.startswith("param/")}
        model.module.load_state_dict(state, strict=False)
        model.module.eval()
        return model

    def inputs(self, profile: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        x = ((query.to(self.x_mean.dtype)) - self.x_mean) / self.x_std
        onehot = torch.nn.functional.one_hot(profile.long(), 2).to(x.dtype)
        return x, onehot

    def draw(self, profile: Tensor, query: Tensor, count: int, generator: Any = None) -> IntentDraws:
        with torch.no_grad():
            return self._draw(profile, query, count, generator)

    def _draw(self, profile: Tensor, query: Tensor, count: int, generator: Any = None) -> IntentDraws:
        x, onehot = self.inputs(profile, query)
        logits, means, scale, face_logits = self.module(x, onehot)
        m = x.shape[0]
        component = categorical(torch.exp(logits), count, generator)
        mu = means.gather(1, component[..., None].expand(m, count, means.shape[-1]))
        sc = scale.gather(1, component[..., None, None].expand(m, count, scale.shape[-2], scale.shape[-1]))
        eps = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        z = mu + (sc @ eps[..., None])[..., 0]
        y = z * self.y_std + self.y_mean
        p_face = torch.sigmoid(face_logits.gather(1, component))
        face = (torch.rand(p_face.shape, dtype=p_face.dtype, device=p_face.device, generator=generator) < p_face).long()
        draws = IntentDraws.from_targets(y.to(query.dtype), face)
        # The intent's contact time after the bounce spans the contact window the rules search, 0.05-0.70 s.
        draws.after_bounce_s = draws.after_bounce_s.clamp(max=TAU_MAX_S)
        return draws


__all__ = ["FEATURES", "INTENT_VERSION", "TARGETS", "IntentDraws", "MdnIntent", "OutOfDistribution"]
