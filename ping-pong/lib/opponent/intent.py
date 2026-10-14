"""
The shot intent h2: what a receiver means to do with an incoming ball, before the rules make it concrete.

The generator's intent is ``PlayerModels.draw_shot`` (the offline data generator's, not shipped here): conditioned on the
predicted own-half bounce (x, y, velocity) and the receiver's stance, in the hitter frame, it draws the contact time
after the bounce, the landing target, the flight time, the outgoing topspin and sidespin, the rubber presented, the
blade offset from the hips, the grip roll (the donor's blade-tip direction) and the start of the normal solve.

Two batched models of it:

- :class:`KnnIntent`, the generator's own draw ported exactly (k = 20 nearest donors, Gaussian kernel at the k-th
  distance, the shrunk smoothed bootstrap, the out-of-distribution pool). It is the reference the learned model is
  checked against and a drop-in alternative.
- :class:`MdnIntent`, the distilled model: a mixture density network over the same 13 continuous outputs with a
  Bernoulli rubber face per component, trained on the generator's own draws at the receive opportunities of the v1
  release's body-first shards (``intent_mdn_v2`` in the ``him-opponent-model`` asset). It is the
  only learned part of the decision; the body-first feasibility's acceptance and nudge are fitted on the body
  runtime's own calls (:mod:`.body`).
"""

from __future__ import annotations

import json
import math
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

    def as_targets(self) -> Tensor:
        """``(..., 13)`` in :data:`TARGETS` order."""

        return torch.cat([self.after_bounce_s[..., None], self.landing, self.flight_s[..., None], self.spin, self.offset, self.tip, self.normal_rel], -1)

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


class KnnIntent:
    """The generator's conditional draw (``PlayerModels.draw_shot``), batched and exact in distribution."""

    def __init__(self, rules: RuleBook) -> None:
        self.rules = rules
        a = rules.arrays
        h = rules.header
        dev, dt = rules.device, rules.dtype

        def t(x: Any) -> Tensor:
            return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dt, device=dev)

        self.scale = t(h["feature_scale"])
        features = np.column_stack([a["donor_bounce"][:, [0, 1, 3, 4, 5]], a["donor_stance"][:, :2]])
        self.k = int(h["k_neighbours"])
        self.tables = []
        for role in PROFILES:
            index = a[f"draw_index_{role}"]
            self.tables.append((torch.as_tensor(index, device=dev), t(features[index] / np.asarray(h["feature_scale"])), float(h["profiles"][role]["ood_cut"])))
        self.after = t(a["donor_after_bounce_s"])
        self.landing = t(a["donor_landing"])
        self.flight = t(a["donor_flight_s"])
        self.spin = t(a["donor_spin_rel"])
        self.normal_rel = t(a["donor_normal_rel"])
        self.offset = t(a["donor_blade_centre"][:, :2] - a["donor_pelvis"][:, :2])
        self.face = torch.as_tensor(a["donor_blade_face"].astype(np.int64), device=dev)
        self.tip = t(a["donor_tip"])
        self.v_out = t(a["donor_v_out"])
        self.bounce = t(a["donor_bounce"])
        bw = h["output_bandwidth"]
        spread = h["output_spread"]
        self.groups = {
            name: (float(bw[name]), t(np.asarray(spread[name], dtype=np.float64).reshape(-1))) for name in ("after_bounce", "landing", "flight", "spin")
        }

    def out_of_distribution(self, profile: Tensor, query: Tensor) -> Tensor:
        """The generator's out-of-distribution rule: the nearest donor farther than the donors' own p99 spacing."""

        out = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
        for p, (_index, z, cut) in enumerate(self.tables):
            out = torch.where(profile == p, torch.cdist(query / self.scale, z).min(1).values > cut, out)
        return out

    def draw(self, profile: Tensor, query: Tensor, count: int, generator: Any = None) -> IntentDraws:
        """
        ``count`` draws for each query ``(M, 7)`` of profile ``(M,)``.

        Every row is computed against both profiles' donors and the row's own profile selected, so the draw has no row
        compaction and no host synchronisation (it can be captured in a CUDA graph).
        """

        m = query.shape[0]
        dev, dt = query.device, query.dtype
        out_index = torch.zeros((m, count), dtype=torch.long, device=dev)
        centres = {name: torch.zeros((m, n.numel()), dtype=dt, device=dev) for name, (_h, n) in self.groups.items()}
        ood = torch.zeros(m, dtype=torch.bool, device=dev)
        q = query / self.scale
        for p, (index, z, cut) in enumerate(self.tables):
            mine = profile == p
            d = torch.cdist(q, z)
            dn, nearest = torch.topk(d, self.k, dim=1, largest=False, sorted=True)
            radius = dn[:, -1].clamp_min(1e-9)
            weights = torch.exp(-0.5 * (dn / radius[:, None]) ** 2)
            weights = weights / weights.sum(1, keepdim=True)
            far = dn[:, 0] > cut
            # Out of distribution: uniform over donors on the same side and within 0.3 m of depth.
            donor_bounce = self.bounce[index]
            side = torch.sign(donor_bounce[None, :, 1]) == torch.sign(q[:, 1:2] * self.scale[1])
            depth = (donor_bounce[None, :, 0] - q[:, 0:1] * self.scale[0]).abs() < 0.3
            pool = (side & depth).to(dt)
            pool = torch.where(pool.sum(1, keepdim=True) > 0, pool, torch.ones_like(pool))
            pool = pool / pool.sum(1, keepdim=True)
            dense = torch.zeros((m, index.numel()), dtype=dt, device=dev)
            dense.scatter_(1, nearest, weights)
            dense = torch.where(far[:, None], pool, dense)
            pick = categorical(dense, count, generator)
            out_index = torch.where(mine[:, None], index[pick], out_index)
            for name, centre in centres.items():
                centres[name] = torch.where(mine[:, None], dense @ self._group(name)[index], centre)
            ood = torch.where(mine, far, ood)
        draws = {}
        for name, (h, spread) in self.groups.items():
            values = self._group(name)[out_index]
            centre = centres[name][:, None, :]
            noise = torch.randn(values.shape, dtype=dt, device=dev, generator=generator) * h * spread
            draws[name] = centre + (values - centre + noise) / math.sqrt(1.0 + h * h)
        return IntentDraws(
            after_bounce_s=draws["after_bounce"][..., 0].clamp_min(0.05),
            landing=draws["landing"],
            flight_s=draws["flight"][..., 0].clamp_min(0.15),
            spin=draws["spin"],
            face=self.face[out_index],
            offset=self.offset[out_index],
            tip=self.tip[out_index],
            normal_rel=self.normal_rel[out_index],
            guess=self.v_out[out_index],
            donor=out_index,
            ood=ood[:, None].expand(m, count),
        )

    def _group(self, name: str) -> Tensor:
        return {"after_bounce": self.after[:, None], "landing": self.landing, "flight": self.flight[:, None], "spin": self.spin}[name]


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


def mdn_log_likelihood(logits: Tensor, means: Tensor, scale: Tensor, face_logits: Tensor, y: Tensor, face: Tensor) -> Tensor:
    """``log p(y, face | x)`` per row under the mixture (``scale`` is each component's Cholesky factor)."""

    diff = y[:, None, :] - means
    solved = torch.linalg.solve_triangular(scale, diff[..., None], upper=False)[..., 0]
    maha = (solved * solved).sum(-1)
    logdet = torch.log(torch.diagonal(scale, dim1=-2, dim2=-1)).sum(-1)
    d = y.shape[-1]
    gauss = -0.5 * maha - logdet - 0.5 * d * math.log(2.0 * math.pi)
    f = face[:, None].to(y.dtype)
    bern = f * torch.nn.functional.logsigmoid(face_logits) + (1.0 - f) * torch.nn.functional.logsigmoid(-face_logits)
    return torch.logsumexp(logits + gauss + bern, -1)


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

    def save(self, path: Path | str, meta: dict | None = None) -> None:
        path = Path(path)
        arrays = {f"param/{k}": v.detach().cpu().numpy() for k, v in self.module.state_dict().items()}
        for name in ("x_mean", "x_std", "y_mean", "y_std"):
            arrays[name] = getattr(self, name).detach().cpu().numpy()
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps({**self.meta, **(meta or {})}, indent=1) + "\n")

    def inputs(self, profile: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        x = ((query.to(self.x_mean.dtype)) - self.x_mean) / self.x_std
        onehot = torch.nn.functional.one_hot(profile.long(), 2).to(x.dtype)
        return x, onehot

    def log_likelihood(self, profile: Tensor, query: Tensor, y: Tensor, face: Tensor) -> Tensor:
        x, onehot = self.inputs(profile, query)
        logits, means, scale, face_logits = self.module(x, onehot)
        z = (y.to(x.dtype) - self.y_mean) / self.y_std
        return mdn_log_likelihood(logits, means, scale, face_logits, z, face) - torch.log(self.y_std).sum()

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


__all__ = ["FEATURES", "INTENT_VERSION", "TARGETS", "IntentDraws", "KnnIntent", "MdnIntent", "mdn_log_likelihood"]
