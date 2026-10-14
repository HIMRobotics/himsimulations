"""
Pinhole cameras with OpenCV distortion, and physics-constrained flight fits to 2D detections.

Camera convention (OpenCV): camera +X right, +Y down, +Z forward;
``x_cam = R x_world + t``. A world point projects through::

    x' = X/Z, y' = Y/Z, r^2 = x'^2 + y'^2
    radial = 1 + k1 r^2 + k2 r^4 + k3 r^6
    x'' = x' radial + 2 p1 x' y' + p2 (r^2 + 2 x'^2)
    y'' = y' radial + p1 (r^2 + 2 y'^2) + 2 p2 x' y'
    u = fx x'' + cx,  v = fy y'' + cy

The flight fit estimates a segment's initial state (position, velocity and
optionally spin at ``t0``) from timestamped detections in any number of
cameras with unsynchronised clocks already mapped to one master time. The
model is the reference flight ODE, so the fit is physics-constrained; the cost
is the pixel reprojection error over each detection's sigma with a robust loss
(scipy ``least_squares``). A linear ballistic solve (gravity only) supplies the
starting point when none is given.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import OptimizeResult, least_squares
from scipy.spatial.transform import Rotation

from lib.physics.params import BallTableParams
from lib.physics.reference.flight import FlightModel, propagate


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


@dataclass(frozen=True)
class Detections:
    """
    Timestamped 2D ball detections from one or more cameras.

    :param camera: Camera index per detection, shape ``(N,)``.
    :param time: Master-clock time, s, shape ``(N,)``.
    :param pixel: Ball centre in pixels, shape ``(N, 2)``.
    :param sigma: One-sigma pixel noise per detection, px, shape ``(N,)``.
    """

    camera: NDArray[np.int64]
    time: NDArray[np.float64]
    pixel: NDArray[np.float64]
    sigma: NDArray[np.float64]

    @classmethod
    def build(cls, camera: ArrayLike, time: ArrayLike, pixel: ArrayLike, sigma: ArrayLike = 1.0) -> Detections:
        """Validated detections; ``sigma`` may be a scalar."""

        cam = np.asarray(camera, dtype=np.int64).reshape(-1)
        t = np.asarray(time, dtype=float).reshape(-1)
        uv = np.asarray(pixel, dtype=float).reshape(-1, 2)
        sig = np.broadcast_to(np.asarray(sigma, dtype=float), t.shape).copy()
        if not cam.size == t.size == uv.shape[0]:
            raise ValueError("camera, time and pixel must have the same length")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(uv))):
            raise ValueError("detections must be finite")
        if np.any(sig <= 0.0):
            raise ValueError("sigma must be positive")
        return cls(cam, t, uv, sig)

    def __len__(self) -> int:
        return int(self.time.size)

    def subset(self, mask: ArrayLike) -> Detections:
        """The detections selected by a boolean mask or index array."""

        index = np.asarray(mask)
        return Detections(self.camera[index], self.time[index], self.pixel[index], self.sigma[index])


def project_trajectory(positions: ArrayLike, camera: PinholeCamera) -> NDArray[np.float64]:
    """Pixels of a trajectory's positions (m, shape ``(N, 3)``); NaN where behind the camera."""

    return camera.project(positions)


def ballistic_initial_guess(detections: Detections, cameras: Sequence[PinholeCamera], t0: float, gravity: float, *, iterations: int = 4) -> NDArray[np.float64]:
    """
    Linear estimate of position and velocity at ``t0`` assuming gravity only.

    With ``x(t) = p0 + v0 tau + 1/2 g tau^2`` each detection's normalised ray
    gives two equations linear in ``(p0, v0)``. Rows are reweighted by the
    current depth estimate and a Huber weight on the pixel residual.

    :return: ``[p0 (m), v0 (m/s)]``, 6 floats.
    """

    if len(detections) < 3:
        raise ValueError("a ballistic solve needs at least 3 detections")
    g_vec = np.array([0.0, 0.0, -gravity])
    rows: list[NDArray[np.float64]] = []
    rhs: list[float] = []
    focal: list[float] = []
    for i in range(len(detections)):
        cam = cameras[int(detections.camera[i])]
        xn, yn = cam.intrinsics.pixels_to_normalized(detections.pixel[i])
        tau = float(detections.time[i] - t0)
        gravity_term = 0.5 * g_vec * tau * tau
        for row_index, coord in ((0, xn), (1, yn)):
            a = cam.rotation[row_index] - coord * cam.rotation[2]
            b = cam.translation[row_index] - coord * cam.translation[2]
            rows.append(np.concatenate([a, a * tau]))
            rhs.append(-b - float(a @ gravity_term))
            focal.append((cam.intrinsics.fx if row_index == 0 else cam.intrinsics.fy) / detections.sigma[i])
    a_mat = np.array(rows)
    b_vec = np.array(rhs)
    focal_arr = np.array(focal)
    weights = np.ones(len(rows))
    solution = np.zeros(6)
    for _ in range(max(1, iterations)):
        solution, *_ = np.linalg.lstsq(a_mat * weights[:, None], b_vec * weights, rcond=None)
        depth = np.empty(len(rows))
        for i in range(len(detections)):
            cam = cameras[int(detections.camera[i])]
            tau = float(detections.time[i] - t0)
            point = solution[:3] + solution[3:] * tau + 0.5 * g_vec * tau * tau
            depth[2 * i : 2 * i + 2] = max(float(cam.to_camera(point)[2]), 1e-3)
        scale = focal_arr / depth
        residual = np.abs((a_mat @ solution - b_vec) * scale)
        huber = np.where(residual <= 3.0, 1.0, 3.0 / np.maximum(residual, 1e-12))
        weights = scale * huber
    return np.asarray(solution, dtype=np.float64)


@dataclass(frozen=True)
class FlightFit:
    """
    A fitted flight segment.

    :param t0: Reference time of the fitted state, s.
    :param state0: Fitted state at ``t0``: position (m), velocity (m/s), spin (rad/s).
    :param fitted_spin: Whether spin was estimated (else held at its fixed value).
    :param covariance: Covariance of the fitted parameters (6 or 9), SI units squared: the
        Gauss-Newton inverse over the detections the final stage used (the inliers, or all of them
        with a ``linear`` loss), scaled up by their reduced chi-square when it exceeds 1 (never
        scaled down). NaN when fewer than 3 inliers remain.
    :param std: One-sigma uncertainty of the fitted parameters.
    :param residuals: Reprojection residuals (prediction minus detection), px, shape ``(N, 2)``.
    :param inliers: Detections whose sigma-normalised residual norm was under the inlier threshold
        after the robust stage; the final least-squares stage and the covariance use only these.
    :param rms_px: Root-mean-square inlier residual per axis, px.
    :param success: The optimiser's convergence flag (false with fewer than 3 inliers).
    :param message: The optimiser's message.
    :param nfev: Residual evaluations over both stages.
    """

    t0: float
    state0: NDArray[np.float64]
    fitted_spin: bool
    covariance: NDArray[np.float64]
    std: NDArray[np.float64]
    residuals: NDArray[np.float64]
    inliers: NDArray[np.bool_]
    rms_px: float
    success: bool
    message: str
    nfev: int


def _predict_pixels(
    model: FlightModel, states0: NDArray[np.float64], t0: float, detections: Detections, cameras: Sequence[PinholeCamera], dt: float
) -> NDArray[np.float64]:
    """Predicted pixels for a batch of initial states: shape ``(B, N, 2)``."""

    unique_times, inverse = np.unique(detections.time, return_inverse=True)
    # The scalar integrator beats the vectorised one below a few dozen states.
    positions = np.stack([propagate(model, state, t0, unique_times, dt=dt)[:, 0:3] for state in states0])
    per_detection = positions[:, inverse]
    out = np.empty((states0.shape[0], len(detections), 2))
    for index, cam in enumerate(cameras):
        mask = detections.camera == index
        if np.any(mask):
            out[:, mask] = cam.project(per_detection[:, mask])
    return out


def fit_flight(
    detections: Detections,
    cameras: Sequence[PinholeCamera],
    params: BallTableParams,
    *,
    t0: float | None = None,
    initial_state: ArrayLike | None = None,
    fit_spin: bool = False,
    spin: ArrayLike = (0.0, 0.0, 0.0),
    spin_prior_sigma: float | None = 500.0,
    loss: str = "soft_l1",
    f_scale: float = 2.0,
    inlier_threshold: float = 5.0,
    dt: float = 1e-3,
    max_nfev: int = 200,
) -> FlightFit:
    """
    Fit a free-flight segment's initial state to 2D detections.

    :param detections: Detections of one flight segment (no contact inside it).
    :param cameras: Cameras indexed by ``detections.camera``.
    :param params: Physics parameters of the flight model.
    :param t0: Reference time of the fitted state, s; default the earliest detection. Must not follow it.
    :param initial_state: Starting guess (9 floats, or 6 without spin); default a linear ballistic solve.
    :param fit_spin: Estimate the spin as well as position and velocity.
    :param spin: Spin, rad/s: held fixed when ``fit_spin`` is false, else the prior mean and start.
    :param spin_prior_sigma: One-sigma Gaussian prior on each spin component, rad/s, when fitting spin.
        Spin about the direction of travel has no effect on flight, so some prior is needed; ``None`` disables it.
    :param loss: scipy ``least_squares`` loss for the first stage (``linear``, ``soft_l1``, ``huber``,
        ``cauchy``, ``arctan``). Unless it is ``linear``, a second plain least-squares stage refits the inliers.
    :param f_scale: Robust-loss soft margin in units of each detection's sigma.
    :param inlier_threshold: Sigma-normalised residual norm under which a detection counts as an inlier.
    :param dt: Integration step for the fit, s. RK4 at 1 ms differs from 0.1 ms by far less than a micron here.
    :param max_nfev: Residual evaluation cap.
    :return: The fit with its covariance and residuals.
    """

    if len(detections) < 3:
        raise ValueError("a flight fit needs at least 3 detections")
    if int(detections.camera.max()) >= len(cameras) or int(detections.camera.min()) < 0:
        raise ValueError("detection camera index out of range")
    reference = float(detections.time.min()) if t0 is None else float(t0)
    if reference > float(detections.time.min()) + 1e-12:
        raise ValueError("t0 must not follow the earliest detection")
    model = FlightModel.from_params(params)
    spin_vec = np.asarray(spin, dtype=float).reshape(3)
    if initial_state is None:
        start6 = ballistic_initial_guess(detections, cameras, reference, params.aero.gravity)
    else:
        start = np.asarray(initial_state, dtype=float).reshape(-1)
        start6 = start[:6]
        if start.size == 9 and fit_spin:
            spin_vec = start[6:9]
    x0 = np.concatenate([start6, spin_vec]) if fit_spin else start6.copy()
    n_params = x0.size
    prior_mean = np.asarray(spin, dtype=float).reshape(3)
    prior_sigma = spin_prior_sigma if fit_spin else None
    steps = np.array([1e-6] * 6 + ([1e-4] * 3 if fit_spin else []))

    def states_from(xs: NDArray[np.float64]) -> NDArray[np.float64]:
        if fit_spin:
            return xs
        return np.concatenate([xs, np.broadcast_to(spin_vec, (xs.shape[0], 3))], axis=1)

    def residual_batch(xs: NDArray[np.float64], dets: Detections) -> NDArray[np.float64]:
        pixels = _predict_pixels(model, states_from(xs), reference, dets, cameras, dt)
        res = ((pixels - dets.pixel[None]) / dets.sigma[None, :, None]).reshape(xs.shape[0], -1)
        res = np.where(np.isfinite(res), res, 1e6)
        if prior_sigma is not None:
            res = np.concatenate([res, (xs[:, 6:9] - prior_mean) / prior_sigma], axis=1)
        return res

    def jacobian(x: NDArray[np.float64], dets: Detections) -> NDArray[np.float64]:
        res = residual_batch(np.vstack([x, x + np.diag(steps)]), dets)
        return ((res[1:] - res[0]) / steps[:, None]).T

    def solve(dets: Detections, start: NDArray[np.float64], loss_name: str) -> OptimizeResult:
        def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return residual_batch(x[None], dets)[0]

        def jac(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return jacobian(x, dets)

        # scipy's least_squares is unannotated, so its jac parameter infers as str; pass options untyped.
        options: dict[str, Any] = {"jac": jac, "loss": loss_name, "f_scale": f_scale, "x_scale": "jac", "max_nfev": max_nfev, "method": "trf"}
        return least_squares(residual, start, **options)

    def pixel_residuals(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return residual_batch(x[None], detections)[0, : 2 * len(detections)].reshape(-1, 2)

    # Stage 1 finds the inliers under the robust loss. A bounded-influence loss still lets a gross
    # outlier pull the estimate by about f_scale sigma, so stage 2 refits the inliers by plain least
    # squares; its Gauss-Newton covariance is then the honest one.
    result = solve(detections, x0, loss)
    nfev = int(result.nfev)
    inliers = np.linalg.norm(pixel_residuals(result.x), axis=1) < inlier_threshold
    success = bool(result.success)
    message = str(result.message)
    if int(inliers.sum()) < 3:
        success = False
        message = f"only {int(inliers.sum())} inliers; {message}"
    elif loss != "linear":
        result = solve(detections.subset(inliers), result.x, "linear")
        nfev += int(result.nfev)
        success = bool(result.success)
        message = str(result.message)
    x = result.x
    residuals_px = pixel_residuals(x) * detections.sigma[:, None]
    rms = float(np.sqrt(np.mean(residuals_px[inliers] ** 2))) if np.any(inliers) else math.nan

    if int(inliers.sum()) >= 3:
        used = detections if loss == "linear" else detections.subset(inliers)
        j_used = jacobian(x, used)
        final = residual_batch(x[None], used)[0]
        dof = final.size - n_params
        scale = float(np.sum(final**2)) / dof if dof > 0 else 1.0
        covariance = np.asarray(np.linalg.pinv(j_used.T @ j_used), dtype=np.float64) * max(scale, 1.0)
    else:
        covariance = np.full((n_params, n_params), np.nan)
    state0 = np.concatenate([x[:6], x[6:9] if fit_spin else spin_vec])
    return FlightFit(
        t0=reference,
        state0=state0,
        fitted_spin=fit_spin,
        covariance=covariance,
        std=np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        residuals=residuals_px,
        inliers=inliers,
        rms_px=rms,
        success=success,
        message=message,
        nfev=nfev,
    )
