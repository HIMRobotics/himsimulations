"""
The arms against the body: how deep either arm's distal part reaches into the torso. Numpy only; the one definition
for the arm hold's clearance cost (:func:`~lib.replay.players.arm_ik.solve_arm`) and the playback's runtime clearance.

**The torso** is four elliptic cylinders, one per spine segment (Pelvis-Waist, Waist-Spine01, Spine01-Spine02,
Spine02-NeckTwist01), each in a frame built from joint positions alone: the axis along the segment, the lateral
direction from the thighs (the lowest), the shoulders (the upper two) or their mean (the middle). Their centres and
semi-axes were fitted per rig to the skinned torso mesh (bind pose, p1-p99 of each segment's torso vertices, metres at
body scale 1) and are scaled by :data:`TORSO_SCALE`.

**The arm** is sampled along forearm-to-hand (7 points, radius 4 cm) and hand-to-middle-knuckle (3 points, 3 cm). A
point's depth is ``(1 - rho) * min(semi-axes) + radius``, ``rho`` its normalised elliptic radius, within the segment's
length; the deepest point and segment count.

**Calibration** against a skinned-mesh measure of the arms in the torso, on frames of both rigs: precision 0.85 and
recall 0.86 at 1 cm. The capture retargeted onto each body (every 10th valid frame, 5,425 per body) is deeper than 1 cm
on 0.13% (Party) and 0.04% (Uniform) of its frames; over 5 s windows its share is 0% at p95, 6.7% / 0% at p99 and
13.3% / 6.7% at most.

It is a visual-realism measure: it shapes how the arm is held, never the physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[Any]

TORSO_SCALE: float = 0.95
"""The fitted semi-axes are scaled by this (the calibration against the skinned mesh)."""
SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("Pelvis", "Waist", "hips"),
    ("Waist", "Spine01", "mid"),
    ("Spine01", "Spine02", "shoulders"),
    ("Spine02", "NeckTwist01", "shoulders"),
)
"""The torso's segments: ``(from joint, to joint, where the lateral direction comes from)``."""
ARM_SEGMENTS: dict[str, tuple[tuple[str, str, float, int], ...]] = {
    side: ((f"{side}_Forearm", f"{side}_Hand", 0.04, 7), (f"{side}_Hand", f"{side}_Mid2", 0.03, 3)) for side in ("R", "L")
}
"""Per arm, the sampled parts: ``(from joint, to joint, radius at body scale 1 in metres, samples)``."""
TORSO_PROXY: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "ActorCore_Party_M_0001@2.0.2": (
        (0.0089, 0.1634, 0.0092, 0.2244),
        (0.0130, 0.1598, 0.0100, 0.2212),
        (0.0291, 0.1701, 0.0124, 0.2461),
        (0.0055, 0.1383, 0.0037, 0.2167),
    ),
    "ActorCore_Uniform_M_0001@1.0.1": (
        (-0.0145, 0.1629, -0.0030, 0.1988),
        (-0.0253, 0.1609, -0.0001, 0.1881),
        (-0.0293, 0.1554, -0.0030, 0.1855),
        (-0.0142, 0.1452, -0.0116, 0.2047),
    ),
}
"""Per rig, per segment: the forward centre, forward semi-axis, lateral centre and lateral semi-axis (m, body scale 1)."""


@dataclass(frozen=True)
class TorsoFrames:
    """The torso's elliptic cylinders posed, per segment ``S`` and frame ``F``: origins, unit axes, lengths, forward and
    lateral directions ``(S, F, 3)`` / ``(S, F)``, and the scaled centres and semi-axes ``(S,)``, metres."""

    origin: Array
    axis: Array
    length: Array
    forward: Array
    lateral: Array
    forward_centre: Array
    forward_axis: Array
    lateral_centre: Array
    lateral_axis: Array


def has_torso(rig: str) -> bool:
    """Whether the rig has a fitted torso."""

    return rig in TORSO_PROXY


def torso_frames(joints: tuple[str, ...] | list[str], positions: ArrayLike, scale: float, rig: str) -> TorsoFrames:
    """
    The torso's cylinders for joint positions ``(F, J, 3)`` in ``joints`` order (any frame of reference: only the
    joints' relative positions matter), body scale ``scale``.
    """

    idx = {name: k for k, name in enumerate(joints)}
    p = np.asarray(positions, dtype=np.float64)
    hips = p[:, idx["R_Thigh"]] - p[:, idx["L_Thigh"]]
    shoulders = p[:, idx["R_Upperarm"]] - p[:, idx["L_Upperarm"]]
    sides = {"hips": hips, "shoulders": shoulders, "mid": 0.5 * (hips + shoulders)}
    origins, axes, lengths, forwards, laterals = [], [], [], [], []
    for a, b, lateral in SEGMENTS:
        start = p[:, idx[a]]
        along = p[:, idx[b]] - start
        length = np.linalg.norm(along, axis=1)
        u = along / length[:, None]
        side = sides[lateral]
        lat = side - np.sum(side * u, axis=1)[:, None] * u
        lat = lat / np.linalg.norm(lat, axis=1)[:, None]
        origins.append(start)
        axes.append(u)
        lengths.append(length)
        forwards.append(np.cross(u, lat))
        laterals.append(lat)
    proxy = np.asarray(TORSO_PROXY[rig], dtype=np.float64)
    return TorsoFrames(
        origin=np.stack(origins),
        axis=np.stack(axes),
        length=np.stack(lengths),
        forward=np.stack(forwards),
        lateral=np.stack(laterals),
        forward_centre=proxy[:, 0] * scale,
        forward_axis=proxy[:, 1] * scale * TORSO_SCALE,
        lateral_centre=proxy[:, 2] * scale,
        lateral_axis=proxy[:, 3] * scale * TORSO_SCALE,
    )


def points_depth(torso: TorsoFrames, points: ArrayLike, radius: ArrayLike) -> Array:
    """
    How deep spheres at ``points`` (``(F, ..., 3)``, the frames first) of ``radius`` (broadcasting against the points'
    shape without its last axis) reach into the torso, m: the deepest segment's, ``-inf`` where no segment's length
    spans the point (negative values are clearance).
    """

    p = np.asarray(points, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    middle = (1,) * (p.ndim - 2)
    best = np.full(p.shape[:-1], -np.inf)
    for s in range(len(torso.length)):
        d = p - torso.origin[s].reshape(len(p), *middle, 3)
        t = np.sum(d * torso.axis[s].reshape(len(p), *middle, 3), axis=-1) / torso.length[s].reshape(len(p), *middle)
        forward = (np.sum(d * torso.forward[s].reshape(len(p), *middle, 3), axis=-1) - torso.forward_centre[s]) / torso.forward_axis[s]
        lateral = (np.sum(d * torso.lateral[s].reshape(len(p), *middle, 3), axis=-1) - torso.lateral_centre[s]) / torso.lateral_axis[s]
        inside = (1.0 - np.hypot(forward, lateral)) * min(torso.forward_axis[s], torso.lateral_axis[s]) + r
        best = np.maximum(best, np.where((t >= 0.0) & (t <= 1.0), inside, -np.inf))
    return best


def push_out(torso: TorsoFrames, points: ArrayLike, radius: float, margin: float) -> Array:
    """
    Points ``(F, 3)`` (spheres of ``radius``) moved out of the torso to ``margin`` clear of it, along the deepest
    segment's elliptic normal; points already that clear are kept.
    """

    p = np.array(points, dtype=np.float64, copy=True)
    best = np.full(len(p), -np.inf)
    direction = np.zeros_like(p)
    for s in range(len(torso.length)):
        d = p - torso.origin[s]
        t = np.sum(d * torso.axis[s], axis=-1) / torso.length[s]
        f = np.sum(d * torso.forward[s], axis=-1) - torso.forward_centre[s]
        lat = np.sum(d * torso.lateral[s], axis=-1) - torso.lateral_centre[s]
        fa, la = torso.forward_axis[s], torso.lateral_axis[s]
        depth = (1.0 - np.hypot(f / fa, lat / la)) * min(fa, la) + radius
        normal = torso.forward[s] * (f / fa**2)[:, None] + torso.lateral[s] * (lat / la**2)[:, None]
        normal = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        deeper = (t >= 0.0) & (t <= 1.0) & (depth > best)
        best = np.where(deeper, depth, best)
        direction = np.where(deeper[:, None], normal, direction)
    need = best > -margin
    p[need] += direction[need] * (best[need] + margin)[:, None]
    return p


def arm_points(joints: tuple[str, ...] | list[str], positions: ArrayLike, scale: float, side: str) -> tuple[Array, Array]:
    """One arm's sampled points ``(F, P, 3)`` and their radii ``(P,)`` (:data:`ARM_SEGMENTS`)."""

    idx = {name: k for k, name in enumerate(joints)}
    p = np.asarray(positions, dtype=np.float64)
    points, radii = [], []
    for j0, j1, radius, samples in ARM_SEGMENTS[side]:
        w = np.linspace(0.0, 1.0, samples)
        points.append(p[:, idx[j0], None] * (1.0 - w)[None, :, None] + p[:, idx[j1], None] * w[None, :, None])
        radii.append(np.full(samples, radius * scale))
    return np.concatenate(points, axis=1), np.concatenate(radii)


def arm_body_depth(joints: tuple[str, ...] | list[str], positions: ArrayLike, scale: float, rig: str, sides: tuple[str, ...] = ("R", "L")) -> Array:
    """
    ``(F,)`` how deep (m; negative is clearance) either arm's forearm or hand reaches into the torso, for joint
    positions ``(F, J, 3)`` in ``joints`` order (the module docstring's definition; ``sides`` limits it to one arm).
    """

    p = np.asarray(positions, dtype=np.float64)
    torso = torso_frames(joints, p, scale, rig)
    best = np.full(len(p), -np.inf)
    for side in sides:
        points, radii = arm_points(joints, p, scale, side)
        best = np.maximum(best, np.max(points_depth(torso, points, radii), axis=1))
    return best


__all__ = [
    "ARM_SEGMENTS",
    "SEGMENTS",
    "TORSO_PROXY",
    "TORSO_SCALE",
    "TorsoFrames",
    "arm_body_depth",
    "arm_points",
    "has_torso",
    "points_depth",
    "push_out",
    "torso_frames",
]
