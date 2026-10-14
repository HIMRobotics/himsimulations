"""
The receiver's blade before a stroke: :class:`BladePlan`, a blade state per row, and :class:`ReadyPoses`, the recorded
ready poses each receiver returns to (``episode.draw_ready``: a capture's ready pose at the stance, within the
capture's reach, height and body-reach limits, and clear of the board).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lib.physics.vectorized._torch import torch

from .ops import quat_to_matrix, yaw_quaternions, yaw_vectors

Tensor = Any
TICK_S = 1.0 / 24.0


@dataclass
class BladePlan:
    """A blade plan state per row: position, velocity, acceleration, orientation (wxyz), angular velocity."""

    p: Tensor
    v: Tensor
    a: Tensor
    q: Tensor
    w: Tensor

    def take(self, index: Tensor) -> BladePlan:
        return BladePlan(self.p[index], self.v[index], self.a[index], self.q[index], self.w[index])

    @classmethod
    def at_rest(cls, p: Tensor, q: Tensor) -> BladePlan:
        z = torch.zeros_like(p)
        return cls(p, z, z.clone(), q, z.clone())


class ReadyPoses:
    """The capture's ready poses per profile, and the body limits a ready pose must keep."""

    def __init__(self, rules: Any) -> None:
        self.rules = rules
        h = rules.header
        dev, dt = rules.device, rules.dtype
        self.c = h["constants"]["stroke"]
        self.motion = [h["profiles"][role]["motion"] for role in ("suited", "opponent")]

        def t(x: Any) -> Tensor:
            return torch.as_tensor(x, dtype=dt, device=dev)

        a = rules.arrays
        self.ready = [(t(a[f"ready_{role}_offset"]), t(a[f"ready_{role}_quat"])) for role in ("suited", "opponent")]
        self.grip_offset = torch.stack([t(m["grip_axes"]).T @ t(m["grip_centre"]) for m in self.motion])
        self.reach_p99 = t([m["reach_p99_m"] for m in self.motion])
        self.height = t([m["height_above_floor_m"] for m in self.motion])

    def hands(self, profile: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """``motion.hands``: the hand from blade poses (``p`` ``(..., 3)``, ``q`` ``(..., 4)``) through the grip."""

        offset = self.grip_offset[profile].to(p.dtype)
        offset = offset.reshape(offset.shape[:1] + (1,) * (p.dim() - 2) + (3,))
        return p - (quat_to_matrix(q) @ offset.expand(p.shape)[..., None])[..., 0]

    # ---- ready poses -------------------------------------------------------------------------------------------

    def draw_ready(self, profile: Tensor, stance: Tensor, flip: Tensor, floor_z: Tensor, half: tuple[Tensor, Tensor], generator: Any = None) -> BladePlan:
        """``episode.draw_ready``: a recorded ready pose at the stance within the limits (up to 50 draws, else the last)."""

        r = profile.shape[0]
        k = int(self.c["ready_draws"])
        dev = stance.device
        offsets = torch.zeros((r, k, 3), dtype=stance.dtype, device=dev)
        quats = torch.zeros((r, k, 4), dtype=stance.dtype, device=dev)
        for p in (0, 1):
            rows = torch.nonzero(profile == p).flatten()
            if rows.numel() == 0:
                continue
            off, qu = self.ready[p]
            pick = torch.randint(0, off.shape[0], (rows.numel(), k), device=dev, generator=generator)
            offsets[rows] = off[pick]
            quats[rows] = qu[pick]
        flip_k = flip[:, None].expand(r, k)
        local = yaw_vectors(offsets, flip_k)
        height_m = self.rules.constants["pelvis_height_m"]
        position = torch.stack([stance[:, None, 0] + local[..., 0], stance[:, None, 1] + local[..., 1], height_m + local[..., 2]], -1)
        orientation = yaw_quaternions(quats, flip_k)
        profile_k = profile[:, None].expand(r, k)
        height = position[..., 2] - floor_z[:, None]
        reach = torch.sqrt(((position[..., 0:2] - stance[:, None, :]) ** 2).sum(-1))
        hand = self.hands(profile, position, orientation)
        within = self.rules.arm_excess(profile_k, hand, stance[:, None, :].expand(r, k, 2), flip_k, self.rules.arm_reach[profile][:, None]) <= 0.0
        over = (hand[..., 0].abs() <= half[0][:, None]) & (hand[..., 1].abs() <= half[1][:, None])
        clear = ~over | (hand[..., 2] >= self.c["plan_board_clearance_m"])
        ok = (
            (height >= self.height[profile, 0][:, None])
            & (height <= self.height[profile, 1][:, None])
            & (reach <= self.reach_p99[profile][:, None])
            & within
            & clear
        )
        index = torch.arange(k, device=dev)[None, :].expand(r, k)
        first = torch.where(ok, index, torch.full_like(index, k - 1)).min(1).values
        sel = first[:, None, None]
        return BladePlan.at_rest(position.gather(1, sel.expand(r, 1, 3))[:, 0], orientation.gather(1, sel.expand(r, 1, 4))[:, 0])


__all__ = ["BladePlan", "ReadyPoses"]
