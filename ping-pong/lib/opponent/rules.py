"""
The generator's receiver rules, batched: the fitted stop hazard, the cited human limits, the contact-phase envelopes
and the paddle-state plausibility density.

Every number here is the offline data generator's, exported to ``opponent_rules_v2.{npz,json}`` in the
``ping-pong/him-opponent-model`` asset (:func:`lib.opponent.fetch_opponent_model`). The export was checked value by
value against the generator, and the batched functions against the generator's own on the same inputs.

These are **rules, not learned decisions**: the 4-feature logistic hazard (``players.py`` ``stop_probability``; its
flight-time and speed coefficients are negative, so a faster ball makes a stop *less* likely), the footwork envelope
(latency 0.30 s, 4 m/s^2, 1.8 m/s), the reaction floor (0.12 s), the contact window (0.06-0.70 s after the bounce),
the recorded blade-offset envelope and the capture's body-reach model. Being closed-form, they extrapolate the way
their formulas do to faster or spinnier balls, which :mod:`.validate` checks (O3).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from lib.physics.vectorized._torch import torch

from .ops import yaw_vectors, yaw_xy

Tensor = Any

RULES_VERSION = 2
"""v2: the v1 release's fit (noise 0.7, hazard 0.8) and its body-first feasibility (``body_*``, :mod:`.body`)."""
PROFILES: tuple[str, ...] = ("suited", "opponent")
"""Profile 0 is the suited player (the robot's seat in the generator), 1 the opponent."""

ENVELOPE_GRID_M = 0.004
"""Cell of the precomputed blade-offset envelope (log density), m: looked up bilinearly."""


@cache
def load_rules_document(directory: Path, version: int = RULES_VERSION) -> tuple[dict, dict[str, np.ndarray]]:
    """The exported rules in ``directory`` (the opponent model's files): ``(header, arrays)``."""

    header = json.loads((directory / f"opponent_rules_v{version}.json").read_text())
    with np.load(directory / f"opponent_rules_v{version}.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return header, arrays


def footwork_reach(warning_s: Tensor, latency: float, speed: float, accel: float) -> Tensor:
    """``HumanLimits.footwork_reach``: how far the pelvis can move ``warning_s`` after the other player's hit."""

    tau = (warning_s - latency).clamp_min(0.0)
    rise = speed / accel
    return torch.where(tau <= rise, 0.5 * accel * tau * tau, 0.5 * speed * rise + speed * (tau - rise))


@dataclass
class RuleBook:
    """The generator's receiver rules as tensors on one device."""

    header: dict
    arrays: dict[str, np.ndarray]
    device: Any
    dtype: Any
    hazard_centre: Tensor
    hazard_scale_vec: Tensor
    hazard_coefficients: Tensor
    """``(2, 5)``: intercept and the four feature coefficients per profile."""
    shoulder_low: Tensor
    shoulder_high: Tensor
    arm_reach: Tensor
    blade_reach: Tensor
    hips_front: Tensor
    envelope_origin: Tensor
    envelope_grid: Tensor
    """``(2, H, W)``: the offset envelope's log density on a grid, per profile."""
    envelope_threshold: Tensor
    plausibility_points: Tensor
    plausibility_centre: Tensor
    plausibility_scale: Tensor
    plausibility_bandwidth: float
    plausibility_threshold: float
    shoulder_offset: Tensor

    @classmethod
    def load(cls, directory: Path, *, device: Any = "cpu", dtype: Any = None, version: int = RULES_VERSION) -> RuleBook:
        """The rules in ``directory`` (the opponent model's files, :func:`lib.opponent.fetch_opponent_model`) on ``device``."""

        header, arrays = load_rules_document(Path(directory), version)
        dtype = dtype or torch.float64

        def t(values: Any) -> Tensor:
            return torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=dtype, device=device)

        roles = [header["profiles"][role] for role in PROFILES]
        grids = []
        thresholds = []
        origin = None
        for role, entry in zip(PROFILES, roles, strict=True):
            grid, origin = _envelope_grid(arrays[f"envelope_points_{role}"], np.asarray(entry["envelope_bandwidth"]), device)
            grids.append(grid)
            thresholds.append(entry["envelope_threshold"])
        plausibility = header["plausibility"]
        points = (arrays["donor_plausibility"] - np.asarray(plausibility["centre"])) / np.asarray(plausibility["scale"])
        return cls(
            header=header,
            arrays=arrays,
            device=device,
            dtype=dtype,
            hazard_centre=t(header["hazard"]["centre"]),
            hazard_scale_vec=t(header["hazard"]["scale"]),
            hazard_coefficients=t([entry["hazard_coefficients"] for entry in roles]),
            shoulder_low=t([entry["shoulder_box_m"][0] for entry in roles]),
            shoulder_high=t([entry["shoulder_box_m"][1] for entry in roles]),
            arm_reach=t([entry["arm_reach_m"] for entry in roles]),
            blade_reach=t([entry["blade_reach_m"] for entry in roles]),
            hips_front=t([entry["hips_front_m"] for entry in roles]),
            envelope_origin=t(origin),
            envelope_grid=t(np.stack(grids)),
            envelope_threshold=t(thresholds),
            plausibility_points=t(points),
            plausibility_centre=t(plausibility["centre"]),
            plausibility_scale=t(plausibility["scale"]),
            plausibility_bandwidth=float(plausibility["bandwidth"]),
            plausibility_threshold=float(plausibility["threshold"]),
            shoulder_offset=t(header["constants"]["shoulder_offset_m"]),
        )

    # ---- constants --------------------------------------------------------------------------

    @property
    def constants(self) -> dict:
        return self.header["constants"]

    @property
    def limits(self) -> dict:
        return self.header["limits"]

    @property
    def fitted(self) -> dict:
        return self.header["fitted"]

    # ---- the stop hazard (players.py stop_probability) --------------------------------------------

    def hazard_features(self, bounce_local: Tensor, flight_s: Tensor, speed_after: Tensor, stance_local: Tensor) -> Tensor:
        """``players.hazard_features``: lateral distance stance-bounce, depth from the end line, flight time, speed after."""

        return torch.stack([(bounce_local[..., 1] - stance_local[..., 1]).abs(), bounce_local[..., 0] + 1.37, flight_s, speed_after], -1)

    def stop_probability(self, profile: Tensor, bounce_local: Tensor, flight_s: Tensor, speed_after: Tensor, stance_local: Tensor) -> Tensor:
        """The fitted chance that a receiver stops the rally (before the hazard scale)."""

        x = self.hazard_features(bounce_local, flight_s, speed_after, stance_local)
        z = (x - self.hazard_centre) / self.hazard_scale_vec
        c = self.hazard_coefficients[profile]
        return torch.sigmoid(c[..., 0] + (z * c[..., 1:]).sum(-1))

    # ---- footwork, the end line and reach ---------------------------------------------------------

    def footwork_reach(self, warning_s: Tensor) -> Tensor:
        lim = self.limits
        return footwork_reach(warning_s, lim["footwork_latency_s"], lim["footwork_speed_mps"], lim["footwork_accel_mps2"])

    def behind_line(self, pelvis: Tensor, flip: Tensor, limit_local_x: Tensor) -> Tensor:
        """``Rally._behind_line``: the pelvis (table frame, ``(..., 2)``) no further forward than ``limit`` (hitter frame x)."""

        local = yaw_xy(pelvis, flip)
        clipped = torch.stack([torch.minimum(local[..., 0], limit_local_x), local[..., 1]], -1)
        return yaw_xy(clipped, flip)

    def envelope_log_density(self, profile: Tensor, offset_local: Tensor) -> Tensor:
        """The recorded blade-offset envelope's log density at offsets ``(..., 2)`` (hitter frame), bilinear on the grid."""

        grid = self.envelope_grid
        h, w = grid.shape[1], grid.shape[2]
        u = (offset_local - self.envelope_origin) / ENVELOPE_GRID_M
        inside = (u[..., 0] >= 0.0) & (u[..., 0] <= h - 1) & (u[..., 1] >= 0.0) & (u[..., 1] <= w - 1)
        uc0 = u[..., 0].clamp(0.0, h - 1.000001)
        uc1 = u[..., 1].clamp(0.0, w - 1.000001)
        i0, j0 = uc0.floor().long(), uc1.floor().long()
        fi, fj = uc0 - i0, uc1 - j0
        p = profile.expand(offset_local.shape[:-1]) if profile.dim() < offset_local.dim() - 1 else profile
        if p.shape != offset_local.shape[:-1]:
            p = p.reshape(p.shape + (1,) * (offset_local.dim() - 1 - p.dim())).expand(offset_local.shape[:-1])
        v00 = grid[p, i0, j0]
        v01 = grid[p, i0, j0 + 1]
        v10 = grid[p, i0 + 1, j0]
        v11 = grid[p, i0 + 1, j0 + 1]
        value = (1 - fi) * ((1 - fj) * v00 + fj * v01) + fi * ((1 - fj) * v10 + fj * v11)
        return torch.where(inside, value, torch.full_like(value, -math.inf))

    def envelope_ok(self, profile: Tensor, offset_local: Tensor) -> Tensor:
        threshold = self.envelope_threshold[profile]
        threshold = threshold.reshape(threshold.shape + (1,) * (offset_local.dim() - 1 - threshold.dim()))
        return self.envelope_log_density(profile, offset_local) >= threshold

    def arm_excess(self, profile: Tensor, points: Tensor, pelvis_xy: Tensor, flip: Tensor, reach: Tensor) -> Tensor:
        """``motion.arm_excess``: distance of each point beyond the shoulder box around the hips, less ``reach``."""

        height = self.constants["pelvis_height_m"]
        hips = torch.cat([pelvis_xy, torch.full_like(pelvis_xy[..., :1], height)], -1)
        local = yaw_vectors(points - hips, flip)
        shape = local.shape[:-1]
        low = self.shoulder_low[profile].reshape(profile.shape + (1,) * (len(shape) - profile.dim()) + (3,))
        high = self.shoulder_high[profile].reshape(profile.shape + (1,) * (len(shape) - profile.dim()) + (3,))
        gap = torch.maximum(torch.maximum(low - local, torch.zeros_like(local)), local - high)
        return torch.sqrt((gap * gap).sum(-1)) - reach

    def shoulder(self, pelvis_xy: Tensor, flip: Tensor) -> Tensor:
        """``Player.shoulder``: the paddle-arm shoulder (table frame) of hips at ``pelvis_xy``."""

        height = self.constants["pelvis_height_m"]
        offset = yaw_vectors(self.shoulder_offset.expand((*pelvis_xy.shape[:-1], 3)), flip)
        return torch.cat([pelvis_xy, torch.full_like(pelvis_xy[..., :1], height)], -1) + offset

    # ---- plausibility -------------------------------------------------------------------------

    def plausibility(self, coordinates: Tensor) -> Tensor:
        """``Plausibility.log_density`` of paddle states ``(M, 7)`` (hitter frame): the 891-state kernel density."""

        z = (coordinates - self.plausibility_centre) / self.plausibility_scale
        points = self.plausibility_points
        d2 = ((z * z).sum(-1, keepdim=True) + (points * points).sum(-1)[None, :] - 2.0 * z @ points.T).clamp_min(0.0)
        d2 = d2 / (self.plausibility_bandwidth**2)
        peak = d2.min(-1).values
        return -0.5 * peak + torch.log(torch.exp(-0.5 * (d2 - peak[:, None])).mean(-1))

    # ---- episode set-up draws (stances, feeds) ------------------------------------------------------

    def draw_stance(self, profile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """``PlayerModels.draw_stance`` (hitter frame), one per entry of ``profile``."""

        a = self.arrays
        out = np.empty((len(profile), 2))
        for i, p in enumerate(np.asarray(profile)):
            index = np.flatnonzero(a["stance_profile"] == p)
            out[i] = a["stance_xy"][int(rng.choice(index))] + rng.normal(0.0, self.header["stance_bandwidth_m"], 2)
        return out

    def draw_feed(self, profile: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """``PlayerModels.draw_feed`` (feeder's hitter frame): ``(feed indices, states)``."""

        a = self.arrays
        jitter = self.header["feed_jitter"]
        index = np.empty(len(profile), dtype=np.int64)
        states = np.empty((len(profile), 9))
        for i, p in enumerate(np.asarray(profile)):
            pool = np.flatnonzero(a["feed_profile"] == p)
            feed = int(rng.choice(pool))
            state = a["feed_state"][feed].astype(np.float64).copy()
            state[:3] += rng.normal(0.0, jitter["position_m"], 3)
            state[3:6] += rng.normal(0.0, jitter["velocity_mps"], 3)
            index[i], states[i] = feed, state
        return index, states


def _envelope_grid(points: np.ndarray, bandwidth: np.ndarray, device: Any = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """``OffsetEnvelope.log_density`` on a grid covering the shared offset range (both profiles use one origin)."""

    origin = np.array([-0.6, -1.4])
    extent = np.array([1.8, 2.8])
    xs = origin[0] + ENVELOPE_GRID_M * np.arange(round(extent[0] / ENVELOPE_GRID_M) + 1)
    ys = origin[1] + ENVELOPE_GRID_M * np.arange(round(extent[1] / ENVELOPE_GRID_M) + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    q = torch.as_tensor(np.column_stack([gx.ravel(), gy.ravel()]) / bandwidth, dtype=torch.float64, device=device)
    zp = torch.as_tensor(points / bandwidth, dtype=torch.float64, device=device)
    out = []
    for chunk in torch.split(q, 65536):
        d2 = torch.cdist(chunk, zp) ** 2
        peak = d2.min(1).values
        out.append(-0.5 * peak + torch.log(torch.exp(-0.5 * (d2 - peak[:, None])).mean(1)))
    return torch.cat(out).reshape(len(xs), len(ys)).cpu().numpy(), origin


__all__ = ["PROFILES", "RULES_VERSION", "RuleBook", "footwork_reach", "load_rules_document"]
