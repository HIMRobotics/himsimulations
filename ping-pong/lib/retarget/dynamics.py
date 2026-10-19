"""
Inverse dynamics of a retargeted motion against the robot's motors and feet (numpy; a rigid-body analysis on the URDF's
inertias, not a simulation).

The motion is splined (joints, pelvis position, sign-continuous pelvis quaternion) and evaluated at :data:`ID_HZ`.
Per link the Newton-Euler wrench is ``F = m (a - g)``, ``M = I w' + w x I w`` (the paddle merged into its link).

* **Joint torque** is the axis component of the wrench the subtree beyond the joint needs, less the moments of the foot
  contacts inside it, plus armature times joint acceleration.
* **Contacts.** A foot within :data:`CONTACT_HEIGHT_M` of the floor carries its sole corners' forces in a linearised
  friction cone (:data:`MU`). The forces minimise the leg torques relative to their peaks (non-negative least squares,
  the whole-body wrench weighted heavily). What they cannot supply is the **unsupported wrench**.
* **Balance** is the capture-point shift that wrench integrates to over any 0.2 s (:meth:`Dynamics.capture_shift`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation

from lib.retarget.model import RobotModel, body_poses
from lib.retarget.source import FLOOR_Z_M

ID_HZ = 100.0
GRAVITY = np.array([0.0, 0.0, -9.81])
MU = 0.7
CONTACT_HEIGHT_M = 0.015


@dataclass
class Dynamics:
    t: np.ndarray  # (n,) evaluation times
    qd: np.ndarray  # (n, J) joint speed
    tau: np.ndarray  # (n, J) joint torque, with the contact forces
    tau_avail: np.ndarray  # (n, J) the motor's torque at that speed
    force_res: np.ndarray  # (n, 3) N, the force the contacts cannot supply
    moment_res: np.ndarray  # (n, 3) N m, the moment they cannot supply, about the centre of mass
    com: np.ndarray  # (n, 3)
    mass: float

    @property
    def torque_ratio(self) -> np.ndarray:
        """|torque| over the motor's torque at the joint's speed (inf where the motor has none left)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(self.tau) / self.tau_avail
        return np.where(self.tau_avail > 0, r, np.where(np.abs(self.tau) > 1e-6, np.inf, 0.0))

    def capture_shift(self, t0: float = -np.inf, t1: float = np.inf, span: float = 0.2) -> float:
        """The largest shift of the capture point (m) the unsupported wrench causes over any ``span`` seconds inside
        [t0, t1]: the COM velocity error it integrates to (force |int F dt| / m, moment |int M dt| / (m h)), times
        sqrt(h / g). Short alternating spikes cancel; a sustained one does not."""
        sel = (self.t >= t0) & (self.t <= t1)
        if sel.sum() < 2:
            return 0.0
        return float(self._shift(sel, float(np.mean(self.com[sel, 2]) - FLOOR_Z_M), span).max())

    def capture_shift_series(self, span: float = 0.2) -> np.ndarray:
        """Per evaluation time, the capture-point shift over the ``span`` centred on it."""
        dt = float(np.median(np.diff(self.t)))
        h = np.maximum(self.com[:, 2] - FLOOR_Z_M, 0.3)
        w, n = max(1, round(span / dt)), len(self.t)
        cf = np.cumsum(np.r_[np.zeros((1, 2)), self.force_res[:, :2]], 0) * dt
        cm = np.cumsum(np.r_[np.zeros((1, 2)), self.moment_res[:, :2]], 0) * dt
        a, b = np.maximum(0, np.arange(n) - w // 2), np.minimum(n, np.arange(n) + w // 2 + 1)
        dv = np.linalg.norm(cf[b] - cf[a], axis=1) / self.mass + np.linalg.norm(cm[b] - cm[a], axis=1) / (self.mass * h)
        return dv * np.sqrt(h / 9.81)

    def _shift(self, sel: np.ndarray, h: float, span: float) -> np.ndarray:
        dt = float(np.median(np.diff(self.t)))
        w = max(1, round(span / dt))
        cf = np.cumsum(np.r_[np.zeros((1, 2)), self.force_res[sel, :2]], 0) * dt
        cm = np.cumsum(np.r_[np.zeros((1, 2)), self.moment_res[sel, :2]], 0) * dt
        if len(cf) <= w:
            dv = np.atleast_1d(np.linalg.norm(cf[-1]) / self.mass + np.linalg.norm(cm[-1]) / (self.mass * h))
        else:
            dv = np.linalg.norm(cf[w:] - cf[:-w], axis=1) / self.mass + np.linalg.norm(cm[w:] - cm[:-w], axis=1) / (self.mass * h)
        return dv * np.sqrt(h / 9.81)


def inverse_dynamics(m: RobotModel, t: np.ndarray, root_p: np.ndarray, root_r: np.ndarray, q: np.ndarray) -> Dynamics:
    """Inverse dynamics of one continuous motion (see the module docstring)."""
    te = np.arange(t[0], t[-1] + 1e-9, 1.0 / ID_HZ)
    spline = lambda y, nu=0: CubicSpline(t, y, axis=0, bc_type="natural")(te, nu)  # noqa: E731
    xyzw = Rotation.from_matrix(root_r).as_quat()
    quat = np.c_[xyzw[:, 3], xyzw[:, :3]]
    for k in range(1, len(quat)):  # sign-continuous
        if quat[k] @ quat[k - 1] < 0:
            quat[k] = -quat[k]
    quat = spline(quat)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    qd, qdd = spline(q, 1), spline(q, 2)
    rot, pos = body_poses(m, spline(root_p), Rotation.from_quat(np.c_[quat[:, 1:], quat[:, :1]]).as_matrix(), spline(q))
    rot, pos = rot[:, :-1], pos[:, :-1]  # the links (the paddle rides on its link)
    n, nl, dt = len(te), len(m.links), 1.0 / ID_HZ
    com = pos + np.einsum("nlij,lj->nli", rot, m.com)
    acc = np.gradient(np.gradient(com, dt, axis=0), dt, axis=0)
    w = np.zeros((n, nl, 3))  # angular velocity from consecutive rotations, then its derivative
    rel = np.einsum("nlij,nlkj->nlik", rot[2:], rot[:-2])
    w[1:-1] = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(n - 2, nl, 3) / (2 * dt)
    w[0], w[-1] = w[1], w[-2]
    wd = np.gradient(w, dt, axis=0)
    iw = np.einsum("nlij,ljk,nlmk->nlim", rot, m.inertia, rot)
    force = m.mass[None, :, None] * (acc - GRAVITY)
    moment = np.einsum("nlij,nlj->nli", iw, wd) + np.cross(w, np.einsum("nlij,nlj->nli", iw, w))
    axis_w = np.einsum("nlij,lj->nli", rot, m.axis)
    tau0 = np.zeros((n, m.nq))
    for i in range(1, nl):
        sub = [k for k in range(nl) if i in m.chain(k)]
        mom = (moment[:, sub] + np.cross(com[:, sub] - pos[:, i][:, None], force[:, sub])).sum(1)
        tau0[:, i - 1] = np.einsum("ni,ni->n", mom, axis_w[:, i]) + m.armature[i - 1] * qdd[:, i - 1]
    # the whole-body wrench the contacts must supply (about the world origin)
    wf, wm = force.sum(1), (moment + np.cross(com, force)).sum(1)
    feet = list(m.feet)
    corners = np.einsum("nfij,kj->nfki", rot[:, feet], m.sole) + pos[:, feet][:, :, None]
    in_contact = corners[..., 2].min(-1) - FLOOR_Z_M <= CONTACT_HEIGHT_M
    gens = np.array([[MU * np.cos(a), MU * np.sin(a), 1.0] for a in (0, np.pi / 2, np.pi, 3 * np.pi / 2)])
    gens /= np.linalg.norm(gens, axis=1, keepdims=True)
    legs = [m.joints_moving(f) for f in feet]
    tau, com_all = tau0.copy(), (com * m.mass[None, :, None]).sum(1) / m.total_mass
    fres, mres = np.zeros((n, 3)), np.zeros((n, 3))
    for k in range(n):
        cols, owners, points = [], [], []
        for f in (0, 1):
            if in_contact[k, f]:
                for c in range(len(m.sole)):
                    for g in gens:
                        cols.append(g)
                        owners.append(f)
                        points.append(corners[k, f, c])
        wrench = np.r_[wf[k], wm[k]]
        if not cols:  # airborne: nothing is supported
            fres[k], mres[k] = wf[k], wm[k] - np.cross(com_all[k], wf[k])
            continue
        gm, pts, own = np.array(cols).T, np.array(points), np.array(owners)
        a_w = np.r_[gm, np.cross(pts, gm.T).T]  # (6, g): force and moment of a unit generator
        rows, rhs = [], []  # leg torque rows: a foot's force acts on its own leg
        for f in (0, 1):
            for j in legs[f]:
                rows.append(-np.einsum("gi,i->g", np.cross(pts - pos[k, j + 1], gm.T), axis_w[k, j + 1]) * (own == f) / m.peak_torque[j])
                rhs.append(-tau0[k, j] / m.peak_torque[j])
        lam, _ = nnls(np.r_[30.0 * a_w, np.array(rows), 1e-3 * np.eye(len(cols))], np.r_[30.0 * wrench, np.array(rhs), np.zeros(len(cols))], maxiter=2000)
        res = wrench - a_w @ lam
        fres[k], mres[k] = res[:3], res[3:] - np.cross(com_all[k], res[:3])
        f_all = gm * lam
        for f in (0, 1):
            if (own == f).any():
                for j in legs[f]:
                    tau[k, j] = tau0[k, j] - np.cross(pts[own == f] - pos[k, j + 1], f_all[:, own == f].T).sum(0) @ axis_w[k, j + 1]
    return Dynamics(t=te, qd=qd, tau=tau, tau_avail=m.torque_available(qd), force_res=fres, moment_res=mres, com=com_all, mass=m.total_mass)


__all__ = ["CONTACT_HEIGHT_M", "ID_HZ", "MU", "Dynamics", "inverse_dynamics"]
