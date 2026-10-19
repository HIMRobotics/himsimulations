"""
Unit-quaternion and rotation helpers for character motion, vectorised over leading axes.

Conventions, shared with the motion pipeline that produced the session tracks:
quaternions are ``(w, x, y, z)``, Hamilton product, active rotations, so
``rotate(q, v)`` equals ``matrix(q) @ v``; matrices are column convention
(``v_parent = R @ v_local``); rotation vectors are angle times unit axis, in
radians.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_EPS = 1e-12

Array = NDArray[Any]


def _f64(value: ArrayLike) -> Array:
    return np.asarray(value, dtype=np.float64)


def _cross(a: Array, b: Array) -> Array:
    """
    ``np.cross`` of ``(..., 3)`` operands with exactly its arithmetic (``a1 b2 - a2 b1``, ``a2 b0 - a0 b2``,
    ``a0 b1 - a1 b0``, each product rounded, then the difference) but without its axis bookkeeping, which dominated
    the per-tick posing (``moveaxis`` and ``normalize_axis_tuple``: 14-19 % of a replay tick's profile).
    """

    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1)


def quat_product(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """:func:`multiply` of two single quaternions as Python floats (the same IEEE arithmetic, in the same order)."""

    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """:func:`rotate` of one vector by one quaternion as Python floats (the same IEEE arithmetic, in the same order)."""

    w, ux, uy, uz = q
    vx, vy, vz = v
    tx = 2.0 * (uy * vz - uz * vy)
    ty = 2.0 * (uz * vx - ux * vz)
    tz = 2.0 * (ux * vy - uy * vx)
    return (vx + w * tx + (uy * tz - uz * ty), vy + w * ty + (uz * tx - ux * tz), vz + w * tz + (ux * ty - uy * tx))


def multiply(a: ArrayLike, b: ArrayLike) -> Array:
    """Hamilton product ``a * b``: apply ``b`` first, then ``a``."""

    a = _f64(a)
    b = _f64(b)
    if a.shape == (4,) and b.shape == (4,):
        return np.array(quat_product(a.tolist(), b.tolist()))
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def conjugate(q: ArrayLike) -> Array:
    """The inverse of a unit quaternion."""

    return _f64(q) * np.array([1.0, -1.0, -1.0, -1.0])


def normalize(q: ArrayLike) -> Array:
    """Scale to unit length (a zero quaternion becomes the identity)."""

    q = _f64(q)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    out = q / np.maximum(norm, _EPS)
    if np.any(norm < _EPS):
        out = np.where(norm < _EPS, identity(q.shape[:-1]), out)
    return out


def identity(shape: tuple[int, ...] = ()) -> Array:
    """Identity quaternions of the given leading shape."""

    q = np.zeros((*shape, 4))
    q[..., 0] = 1.0
    return q


def rotate(q: ArrayLike, v: ArrayLike) -> Array:
    """Rotate vectors ``v`` (..., 3) by unit quaternions ``q`` (..., 4)."""

    q = _f64(q)
    v = _f64(v)
    if q.shape == (4,) and v.shape == (3,):
        return np.array(quat_rotate(q.tolist(), v.tolist()))
    u = q[..., 1:]
    w = q[..., :1]
    t = 2.0 * _cross(u, v)
    return v + w * t + _cross(u, t)


def to_matrix(q: ArrayLike) -> Array:
    """Column-convention rotation matrices (..., 3, 3) of unit quaternions."""

    n = normalize(q)
    w, x, y, z = n[..., 0], n[..., 1], n[..., 2], n[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def from_matrix(m: ArrayLike) -> Array:
    """Unit quaternions (w >= 0) of column-convention rotation matrices (..., 3, 3)."""

    m = _f64(m)
    shape = m.shape[:-2]
    flat = m.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for index, r in enumerate(flat):
        trace = r[0, 0] + r[1, 1] + r[2, 2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
        out[index] = q
    out = normalize(out)
    out = np.where(out[:, :1] < 0.0, -out, out)
    return out.reshape(*shape, 4)


def from_axis_angle(axis: ArrayLike, angle: ArrayLike) -> Array:
    """Rotation by ``angle`` (rad) about ``axis`` (normalised here)."""

    axis = _f64(axis)
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), _EPS)
    half = 0.5 * _f64(angle)[..., None]
    return np.concatenate([np.cos(half), np.sin(half) * axis], axis=-1)


def from_rotvec(r: ArrayLike) -> Array:
    """Quaternion of a rotation vector (angle times axis, rad)."""

    r = _f64(r)
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * angle
    small = angle < 1e-9
    scale = np.where(small, 0.5 - angle**2 / 48.0, np.sin(half) / np.maximum(angle, _EPS))
    return np.concatenate([np.cos(half), scale * r], axis=-1)


def to_rotvec(q: ArrayLike) -> Array:
    """Rotation vector (shortest arc) of a unit quaternion."""

    q = normalize(q)
    q = np.where(q[..., :1] < 0.0, -q, q)
    vec = q[..., 1:]
    s = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    scale = np.where(s < 1e-12, 2.0, angle / np.maximum(s, _EPS))
    return scale * vec


def angle(q: ArrayLike) -> Array:
    """Rotation angle of a quaternion, rad, in [0, pi]."""

    q = _f64(q)
    return 2.0 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.abs(q[..., 0]))


def slerp(a: ArrayLike, b: ArrayLike, u: ArrayLike) -> Array:
    """Spherical interpolation from ``a`` (u = 0) to ``b`` (u = 1), shortest arc."""

    a = normalize(a)
    b = normalize(b)
    u = _f64(u)[..., None] if np.ndim(u) else _f64(u)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-6
    wa = np.where(near, 1.0 - u, np.sin((1.0 - u) * theta) / np.maximum(sin_theta, _EPS))
    wb = np.where(near, u, np.sin(u * theta) / np.maximum(sin_theta, _EPS))
    return normalize(wa * a + wb * b)


def min_arc(a: ArrayLike, b: ArrayLike) -> Array:
    """The smallest rotation taking direction ``a`` onto direction ``b``."""

    first = _f64(a)
    second = _f64(b)
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), _EPS)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), _EPS)
    dot = np.sum(first * second, axis=-1, keepdims=True)
    cross = _cross(first, second)
    q = np.concatenate([1.0 + dot, cross], axis=-1)
    opposite = (1.0 + dot) < 1e-9
    if np.any(opposite):
        helper = np.where(np.abs(first[..., :1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        axis = _cross(first, helper)
        q = np.where(opposite, np.concatenate([np.zeros_like(dot), axis], axis=-1), q)
    return normalize(q)


def continuous(q: ArrayLike, axis: int = 0) -> Array:
    """Flip signs along ``axis`` so consecutive quaternions stay in one hemisphere."""

    q = np.moveaxis(_f64(q).copy(), axis, 0)
    for index in range(1, q.shape[0]):
        flip = np.sum(q[index] * q[index - 1], axis=-1, keepdims=True) < 0.0
        q[index] = np.where(flip, -q[index], q[index])
    return np.moveaxis(q, 0, axis)


def swing_twist(q: ArrayLike, axis: ArrayLike) -> tuple[Array, Array]:
    """
    Split ``q`` into ``swing * twist``, with ``twist`` about ``axis`` (in the frame ``q`` acts on).

    :return: ``(swing, twist)``.
    """

    q = normalize(q)
    a = _f64(axis)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), _EPS)
    projection = np.sum(q[..., 1:] * a, axis=-1, keepdims=True) * a
    twist = normalize(np.concatenate([q[..., :1], projection], axis=-1))
    swing = multiply(q, conjugate(twist))
    return swing, twist


def yaw(angle_rad: float) -> Array:
    """Rotation about +Z (table frame up)."""

    return from_axis_angle((0.0, 0.0, 1.0), angle_rad)


def smoothstep(u: ArrayLike) -> Array:
    """Cubic ease from 0 at ``u <= 0`` to 1 at ``u >= 1`` with zero end slopes."""

    u = np.clip(_f64(u), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


__all__ = [
    "angle",
    "conjugate",
    "continuous",
    "from_axis_angle",
    "from_matrix",
    "from_rotvec",
    "identity",
    "min_arc",
    "multiply",
    "normalize",
    "quat_product",
    "quat_rotate",
    "rotate",
    "slerp",
    "smoothstep",
    "swing_twist",
    "to_matrix",
    "to_rotvec",
    "yaw",
]
