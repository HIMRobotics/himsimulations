"""
Pinhole cameras with OpenCV distortion: the virtual Orbbec cameras' model.

Camera convention (OpenCV): camera +X right, +Y down, +Z forward;
``x_cam = R x_world + t``. A world point projects through::

    x' = X/Z, y' = Y/Z, r^2 = x'^2 + y'^2
    radial = 1 + k1 r^2 + k2 r^4 + k3 r^6
    x'' = x' radial + 2 p1 x' y' + p2 (r^2 + 2 x'^2)
    y'' = y' radial + p1 (r^2 + 2 y'^2) + 2 p2 x' y'
    u = fx x'' + cx,  v = fy y'' + cy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation


class CameraIntrinsics(BaseModel):
    """
    Pinhole intrinsics with OpenCV radial-tangential distortion.

    :param fx: Focal length along u, px.
    :param fy: Focal length along v, px.
    :param cx: Principal point u, px.
    :param cy: Principal point v, px.
    :param k1: Radial distortion, dimensionless.
    :param k2: Radial distortion, dimensionless.
    :param p1: Tangential distortion, dimensionless.
    :param p2: Tangential distortion, dimensionless.
    :param k3: Radial distortion, dimensionless (OpenCV's fifth coefficient; usually 0).
    :param width: Image width, px.
    :param height: Image height, px.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fx: float = Field(gt=0.0)
    fy: float = Field(gt=0.0)
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)

    @property
    def matrix(self) -> NDArray[np.float64]:
        """The 3x3 intrinsic matrix ``K``."""

        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def distortion(self) -> NDArray[np.float64]:
        """OpenCV distortion vector ``(k1, k2, p1, p2, k3)``."""

        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    def distort(self, normalized: ArrayLike) -> NDArray[np.float64]:
        """Apply distortion to normalised image coordinates of shape ``(..., 2)``."""

        xy = np.asarray(normalized, dtype=float)
        x = xy[..., 0]
        y = xy[..., 1]
        r2 = x * x + y * y
        radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return np.stack([xd, yd], axis=-1)

    def undistort(self, distorted: ArrayLike, iterations: int = 30) -> NDArray[np.float64]:
        """Invert :meth:`distort` by fixed-point iteration (OpenCV's method)."""

        xy_d = np.asarray(distorted, dtype=float)
        xd = xy_d[..., 0]
        yd = xy_d[..., 1]
        x = xd.copy()
        y = yd.copy()
        for _ in range(iterations):
            r2 = x * x + y * y
            radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
            dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
            dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        return np.stack([x, y], axis=-1)

    def pixels_to_normalized(self, pixels: ArrayLike) -> NDArray[np.float64]:
        """Undistorted normalised coordinates of pixels, shape ``(..., 2)``."""

        uv = np.asarray(pixels, dtype=float)
        distorted = np.stack([(uv[..., 0] - self.cx) / self.fx, (uv[..., 1] - self.cy) / self.fy], axis=-1)
        return self.undistort(distorted)


@dataclass(frozen=True)
class PinholeCamera:
    """
    A calibrated camera.

    :param intrinsics: Intrinsics and distortion.
    :param rotation: World-to-camera rotation ``R`` (3x3).
    :param translation: World-to-camera translation ``t``, m: ``x_cam = R x_world + t``.
    :param name: Label, e.g. the device serial.
    """

    intrinsics: CameraIntrinsics
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    name: str = ""

    def __post_init__(self) -> None:
        rot = np.array(self.rotation, dtype=float).reshape(3, 3)
        if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-9) or np.linalg.det(rot) < 0.0:
            raise ValueError("camera rotation must be a proper rotation matrix")
        object.__setattr__(self, "rotation", rot)
        object.__setattr__(self, "translation", np.array(self.translation, dtype=float).reshape(3))

    @classmethod
    def look_at(cls, eye: ArrayLike, target: ArrayLike, intrinsics: CameraIntrinsics, *, up: ArrayLike = (0.0, 0.0, 1.0), name: str = "") -> PinholeCamera:
        """
        A camera at ``eye`` (m) whose optical axis passes through ``target`` (m).

        ``up`` (world) maps to image-up (negative v).
        """

        centre = np.asarray(eye, dtype=float)
        forward = np.asarray(target, dtype=float) - centre
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(up, dtype=float))
        if np.linalg.norm(right) < 1e-9:
            raise ValueError("look_at: up is parallel to the viewing direction")
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        rot = np.asarray(np.vstack([right, down, forward]), dtype=np.float64)
        return cls(intrinsics, rot, np.asarray(-rot @ centre, dtype=np.float64), name)

    @classmethod
    def from_rvec_tvec(cls, rvec: ArrayLike, tvec: ArrayLike, intrinsics: CameraIntrinsics, *, name: str = "") -> PinholeCamera:
        """A camera from OpenCV's Rodrigues rotation vector and translation (world to camera)."""

        rot = Rotation.from_rotvec(np.asarray(rvec, dtype=float).reshape(3)).as_matrix()
        return cls(intrinsics, rot, np.asarray(tvec, dtype=float).reshape(3), name)

    @property
    def centre(self) -> NDArray[np.float64]:
        """Optical centre in the world, m."""

        return -self.rotation.T @ self.translation

    def to_camera(self, points: ArrayLike) -> NDArray[np.float64]:
        """World points (m), shape ``(..., 3)``, in camera coordinates."""

        return np.asarray(points, dtype=float) @ self.rotation.T + self.translation

    def project(self, points: ArrayLike) -> NDArray[np.float64]:
        """
        Pixels of world points.

        :param points: World points, m, shape ``(..., 3)``.
        :return: Pixels, shape ``(..., 2)``; NaN for points not in front of the camera.
        """

        cam = self.to_camera(points)
        z = cam[..., 2]
        front = z > 1e-9
        safe = np.where(front, z, 1.0)
        normalized = np.stack([cam[..., 0] / safe, cam[..., 1] / safe], axis=-1)
        distorted = self.intrinsics.distort(normalized)
        k = self.intrinsics
        uv = np.stack([k.fx * distorted[..., 0] + k.cx, k.fy * distorted[..., 1] + k.cy], axis=-1)
        return np.where(front[..., None], uv, np.nan)

    def in_image(self, pixels: ArrayLike) -> NDArray[np.bool_]:
        """Whether pixels lie inside the image."""

        uv = np.asarray(pixels, dtype=float)
        k = self.intrinsics
        return (uv[..., 0] >= 0.0) & (uv[..., 0] < k.width) & (uv[..., 1] >= 0.0) & (uv[..., 1] < k.height)
