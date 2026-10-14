"""
Batched frame and rotation helpers (torch, rows first) shared by the opponent's modules.

Conventions are the offline data generator's: quaternions ``(w, x, y, z)``; the hitter frame
is a 180-degree yaw about +Z (never a mirror); reference blade rotations have columns (towards the handle, across,
forehand normal) and the game's blade axes are ``R_ref @ GAME_FROM_REFERENCE`` (X across, Y towards the tip, Z the
forehand normal).
"""

from __future__ import annotations

from typing import Any

from lib.physics.vectorized._torch import torch

Tensor = Any


def axis_like(like: Tensor, index: int) -> Tensor:
    """A unit axis shaped like ``like`` (``(..., 3)``), built on the device (no host-to-device copy)."""

    out = torch.zeros_like(like)
    out[..., index] = 1.0
    return out


def rows(value: Tensor, like: Tensor) -> Tensor:
    """``value`` of shape ``(M,)`` viewed to broadcast against ``like`` of shape ``(M, ...)``."""

    return value.reshape(value.shape + (1,) * (like.dim() - value.dim()))


def flip_sign(flip: Tensor, dtype: Any) -> Tensor:
    """``-1`` where ``flip``, else ``+1``."""

    return 1.0 - 2.0 * flip.to(dtype)


def yaw_vectors(v: Tensor, flip: Tensor) -> Tensor:
    """The 180-degree yaw of points or vectors ``(M, ..., 3)`` on rows where ``flip``."""

    s = rows(flip_sign(flip, v.dtype), v[..., 0])
    return torch.stack([v[..., 0] * s, v[..., 1] * s, v[..., 2]], -1)


def yaw_xy(v: Tensor, flip: Tensor) -> Tensor:
    """The yaw of planar points ``(M, ..., 2)``."""

    s = rows(flip_sign(flip, v.dtype), v[..., 0])
    return v * s[..., None]


def yaw_states(s: Tensor, flip: Tensor) -> Tensor:
    """The yaw of ball states ``(M, ..., 9)`` (position, velocity, spin)."""

    return torch.cat([yaw_vectors(s[..., 0:3], flip), yaw_vectors(s[..., 3:6], flip), yaw_vectors(s[..., 6:9], flip)], -1)


def yaw_quaternions(q: Tensor, flip: Tensor) -> Tensor:
    """Pre-multiply ``(w, x, y, z)`` orientations by the 180-degree yaw ``(0, 0, 0, 1)`` on rows where ``flip``."""

    turned = torch.stack([-q[..., 3], -q[..., 2], q[..., 1], q[..., 0]], -1)
    return torch.where(rows(flip, q[..., 0])[..., None], turned, q)


def unit(v: Tensor, eps: float = 1e-300) -> Tensor:
    n = torch.sqrt((v * v).sum(-1, keepdim=True))
    return v / n.clamp_min(eps)


def norm(v: Tensor) -> Tensor:
    return torch.sqrt((v * v).sum(-1))


def dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(-1)


def cross(a: Tensor, b: Tensor) -> Tensor:
    return torch.cross(a, b, dim=-1)


def quat_multiply(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def quat_conjugate(q: Tensor) -> Tensor:
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], -1)


def quat_from_rotvec(r: Tensor) -> Tensor:
    angle = norm(r)
    small = angle < 1e-12
    safe = torch.where(small, torch.ones_like(angle), angle)
    half = 0.5 * angle
    s = torch.where(small, 0.5 * torch.ones_like(angle), torch.sin(half) / safe)
    q = torch.cat([torch.cos(half)[..., None], s[..., None] * r], -1)
    return q / norm(q)[..., None]


def quat_to_rotvec(q: Tensor) -> Tensor:
    q = q / norm(q)[..., None]
    q = torch.where((q[..., 0] < 0.0)[..., None], -q, q)
    s = norm(q[..., 1:])
    small = s < 1e-12
    safe = torch.where(small, torch.ones_like(s), s)
    scale = torch.where(small, 2.0 * torch.ones_like(s), 2.0 * torch.atan2(s, q[..., 0]) / safe)
    return q[..., 1:] * scale[..., None]


def rotation_from_rotvec(r: Tensor) -> Tensor:
    """Rotation matrices ``(..., 3, 3)`` of rotation vectors (Rodrigues)."""

    angle = norm(r)
    small = angle < 1e-15
    safe = torch.where(small, torch.ones_like(angle), angle)
    k = r / safe[..., None]
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    skew = torch.stack([torch.stack([zero, -kz, ky], -1), torch.stack([kz, zero, -kx], -1), torch.stack([-ky, kx, zero], -1)], -2)
    c = torch.cos(angle)[..., None, None]
    s = torch.sin(angle)[..., None, None]
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(skew.shape)
    out = eye + s * skew + (1.0 - c) * (skew @ skew)
    return torch.where(small[..., None, None], eye, out)


def rotate_about(axis_angle: Tensor, v: Tensor) -> Tensor:
    """Rotate ``v`` by the rotation vector ``axis_angle`` (Rodrigues), rows first."""

    angle = norm(axis_angle)
    small = angle < 1e-15
    safe = torch.where(small, torch.ones_like(angle), angle)
    k = axis_angle / safe[..., None]
    c = torch.cos(angle)[..., None]
    s = torch.sin(angle)[..., None]
    out = v * c + cross(k, v) * s + k * dot(k, v)[..., None] * (1.0 - c)
    return torch.where(small[..., None], v, out)


def quat_from_matrix(m: Tensor) -> Tensor:
    """``(w, x, y, z)`` of rotation matrices (columns are the body axes), ``w >= 0`` (the generator's branches)."""

    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    trace = m00 + m11 + m22
    one = torch.ones_like(trace)

    def safe_sqrt(x: Tensor) -> Tensor:
        return torch.sqrt(torch.clamp_min(x, 1e-30))

    s0 = 2.0 * safe_sqrt(trace + 1.0)
    q0 = torch.stack([0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0], -1)
    s1 = 2.0 * safe_sqrt(one + m00 - m11 - m22)
    q1 = torch.stack([(m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1], -1)
    s2 = 2.0 * safe_sqrt(one + m11 - m00 - m22)
    q2 = torch.stack([(m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2], -1)
    s3 = 2.0 * safe_sqrt(one + m22 - m00 - m11)
    q3 = torch.stack([(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3], -1)
    c0 = trace > 0.0
    c1 = ~c0 & (m00 > m11) & (m00 > m22)
    c2 = ~c0 & ~c1 & (m11 > m22)
    q = torch.where(c0[..., None], q0, torch.where(c1[..., None], q1, torch.where(c2[..., None], q2, q3)))
    q = q / norm(q)[..., None]
    return torch.where((q[..., 0] < 0.0)[..., None], -q, q)


def quat_to_matrix(q: Tensor) -> Tensor:
    q = q / norm(q)[..., None]
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def reference_from_game(rotation: Tensor) -> Tensor:
    """The reference blade rotation of game blade axes: columns (handle, across, normal) = (-Y_game, X_game, Z_game)."""

    return torch.stack([-rotation[..., :, 1], rotation[..., :, 0], rotation[..., :, 2]], -1)


def blade_rotation(face_normal: Tensor, handle: Tensor) -> Tensor:
    """:func:`lib.physics.reference.surfaces.blade_rotation` batched: local +Z the normal, +X towards ``handle``."""

    z = unit(face_normal)
    x = handle - dot(handle, z)[..., None] * z
    degenerate = norm(x) < 1e-9
    fallback_hint = torch.where((z[..., 0].abs() < 0.9)[..., None], axis_like(z, 0), axis_like(z, 1))
    fallback = fallback_hint - dot(fallback_hint, z)[..., None] * z
    x = unit(torch.where(degenerate[..., None], fallback, x))
    y = cross(z, x)
    return torch.stack([x, y, z], -1)


def strike_frame(direction: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The generator's frame about ``direction``: ``e1 = helper x d`` (horizontal unless ``d`` is near vertical), ``e2 = d x e1``."""

    d = unit(direction)
    ez = axis_like(d, 2)
    ex = axis_like(d, 0)
    helper = torch.where((d[..., 2].abs() < 0.95)[..., None], ez, ex)
    e1 = unit(cross(helper, d))
    return d, e1, cross(d, e1)


def frame_exp(frame: tuple[Tensor, Tensor, Tensor], ab: Tensor) -> Tensor:
    """The unit vector at tangent coordinates ``ab`` (rad) about the frame's direction."""

    d, e1, e2 = frame
    angle = torch.sqrt(ab[..., 0] ** 2 + ab[..., 1] ** 2)
    small = angle < 1e-12
    safe = torch.where(small, torch.ones_like(angle), angle)
    axis = (ab[..., 0:1] * e1 + ab[..., 1:2] * e2) / safe[..., None]
    out = torch.cos(angle)[..., None] * d + torch.sin(angle)[..., None] * axis
    return torch.where(small[..., None], d, out)


def frame_log(frame: tuple[Tensor, Tensor, Tensor], v: Tensor) -> Tensor:
    """The tangent coordinates ``(a, b)`` of a unit vector about the frame's direction (geodesic)."""

    d, e1, e2 = frame
    v = unit(v)
    cosine = dot(v, d).clamp(-1.0, 1.0)
    tangent = v - cosine[..., None] * d
    n = norm(tangent)
    small = n < 1e-12
    angle = torch.acos(cosine)
    scale = torch.where(small, torch.zeros_like(n), angle / torch.where(small, torch.ones_like(n), n))
    return torch.stack([dot(tangent, e1) * scale, dot(tangent, e2) * scale], -1)


def spin_frame(velocity: Tensor) -> tuple[Tensor, Tensor]:
    """The generator's spin axes about an outgoing velocity: ``e1 = z x v`` (topspin), ``e2 = v x e1`` (sidespin)."""

    v = unit(velocity)
    ez = axis_like(v, 2)
    e1 = cross(ez, v)
    e1 = e1 / norm(e1).clamp_min(1e-12)[..., None]
    return e1, cross(v, e1)


__all__ = [
    "blade_rotation",
    "cross",
    "dot",
    "flip_sign",
    "frame_exp",
    "frame_log",
    "norm",
    "quat_conjugate",
    "quat_from_matrix",
    "quat_from_rotvec",
    "quat_multiply",
    "quat_to_matrix",
    "quat_to_rotvec",
    "reference_from_game",
    "rotate_about",
    "rotation_from_rotvec",
    "rows",
    "spin_frame",
    "strike_frame",
    "unit",
    "yaw_quaternions",
    "yaw_states",
    "yaw_vectors",
    "yaw_xy",
]
