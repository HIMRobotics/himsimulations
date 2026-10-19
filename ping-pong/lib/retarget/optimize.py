"""
The inverse kinematics: a static reach per contact, and one whole-segment trajectory optimisation (torch, float64, CPU).

**Static reach** (:func:`solve_reach`). Per contact, with the legs in the ready stance and the pelvis on its target,
the reach joints (inside :data:`REACH_COMFORT` of their ranges) put the blade on the paddle path with the struck normal
at five instants through +-0.1 s. It is solved twice: from the human's stance, and with a free body offset (shift,
yaw, crouch; lightly penalised). The offset of a contact unreachable from the stance becomes the body's move there.

**Trajectory** (:class:`Problem`). Variables per 50 Hz frame: pelvis position, a rotation-vector residual on the
pelvis target, and every joint angle, boxed inside :data:`JOINT_BOX` of its range by ``q = mid + half sin(u)``. The
cost, in priority order:

1. **paddle**: the blade centre and forehand normal on the path, weights rising 1000x near a contact; at the exact
   contact time the solved position and struck normal (the roll about the normal is only weakly held);
2. **body**: pelvis and torso orientation (the torso freer near contacts), pelvis height and position;
3. **feet**: a planted sole flat on the floor and rigid (heel and toe on the plant's mean: no slide, no pivot), only
   weakly drawn to the scaled human placement;
4. **balance**: the zero-moment point (each link's horizontal acceleration at its own height) inside the planted soles,
   and a friction cap;
5. **smoothing and style**: joint accelerations weighted by (inertia / peak torque)^2, pelvis acceleration, joint speed
   above 0.75 of the limit, the free hand about its shoulder, a weak pull to the ready stance, and coarse-sphere
   penalties for self, table and floor contact.

**Solver** (:meth:`Problem.solve`): a per-frame Levenberg-Marquardt IK from the stance; 200 Adam steps without the
balance and plant terms; then L-BFGS (strong Wolfe) in four rounds of rising balance weight (x0.1, 1, 10, 30).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from lib.retarget.model import RobotModel, TorchKinematics, body_poses, rotvec_to_matrix
from lib.retarget.source import FLOOR_Z_M
from lib.retarget.targets import FPS, HOLD_S, Targets

if TYPE_CHECKING:
    import torch

JOINT_BOX = 0.90  # the solve keeps every joint inside 0.9 of its range (Isaac Lab's soft joint limits), the gate is 0.95
REACH_COMFORT = 0.85  # the static reach keeps room for the trajectory solve
TABLE_HALF_M = (1.37, 0.7625, 0.38)  # the table as a box from the top (z = 0) to the floor
SOLE_HALF_WIDTH_M = 0.025
POS_TOL_M, FACE_TOL_DEG = 0.01, 5.0


@dataclass(frozen=True)
class Weights:
    blade_pos: float = 300.0
    blade_pos_window: float = 3e5
    blade_nrm: float = 30.0
    blade_nrm_window: float = 3e4
    blade_tip: float = 1.0
    contact_pos: float = 1e6
    contact_nrm: float = 3e4
    pelvis_rot: float = 20.0
    torso_rot: float = 20.0
    pelvis_xy: float = 1000.0
    pelvis_z: float = 300.0
    foot_xy: float = 10.0
    foot_z: float = 1e5
    foot_flat: float = 3e4
    foot_yaw: float = 20.0
    foot_slide: float = 3e5
    lhand: float = 10.0
    acc_q: float = 30.0
    acc_root: float = 0.3
    speed: float = 1e3
    speed_fraction: float = 0.75
    stance: float = 0.02
    collide: float = 3e4
    zmp: float = 1e4
    friction: float = 1e3
    plant_rigid: float = 1e7


@dataclass
class Solution:
    t: np.ndarray
    root_p: np.ndarray  # (T, 3) pelvis position
    root_r: np.ndarray  # (T, 3, 3) pelvis orientation
    q: np.ndarray  # (T, J) joint angles
    cost: float
    seconds: float


def _seg_dist(p, a, b):
    ab = b - a
    u = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)).clamp(0.0, 1.0)
    return (p - (a + u[..., None] * ab)).norm(dim=-1)


def _tri_dist(p, a, b, c):
    """2-D distance from p to triangle abc, 0 inside."""
    import torch

    cross = lambda o, x, y: (x[..., 0] - o[..., 0]) * (y[..., 1] - o[..., 1]) - (x[..., 1] - o[..., 1]) * (y[..., 0] - o[..., 0])  # noqa: E731
    d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
    inside = ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))
    d = torch.minimum(torch.minimum(_seg_dist(p, a, b), _seg_dist(p, b, c)), _seg_dist(p, c, a))
    return torch.where(inside, torch.zeros_like(d), d)


def support_distance(z, heel, toe, planted):
    """Distance (T,) of the ZMP z (T, 2) from the support: one planted foot its heel-toe line, both the hull of the four
    points, none 0."""
    import torch

    hl, tl, hr, tr = heel[:, 0], toe[:, 0], heel[:, 1], toe[:, 1]
    both = torch.stack([_tri_dist(z, *tri) for tri in ((hl, tl, hr), (hl, tl, tr), (hr, tr, hl), (hr, tr, tl))], -1).min(-1).values
    pl, pr = planted[:, 0], planted[:, 1]
    return torch.where(pl & pr, both, torch.where(pl, _seg_dist(z, hl, tl), torch.where(pr, _seg_dist(z, hr, tr), torch.zeros_like(both))))


class Problem:
    """One segment's trajectory optimisation (see the module docstring)."""

    def __init__(self, tg: Targets, m: RobotModel, w: Weights | None = None) -> None:
        import torch

        self.m, self.w, self.tg, self.n = m, w or Weights(), tg, len(tg.t)
        self.g = TorchKinematics(m)
        f64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
        self.pelvis_p, self.pelvis_r, self.torso_r, self.lhand = f64(tg.pelvis_p), f64(tg.pelvis_r), f64(tg.torso_r), f64(tg.lhand_rel)
        self.foot_p, self.foot_yaw, self.planted = f64(tg.foot_p), f64(tg.foot_yaw), torch.as_tensor(tg.planted)
        self.paddle_p, self.paddle_r, self.near = f64(tg.paddle_p), f64(tg.paddle_r), f64(tg.near)
        # rho: 1 inside the paddle window (hold + one frame), falling smoothly to 0 at +-0.35 s
        self.rho = f64(np.clip(tg.near / (0.5 * (1 + np.cos(np.pi * (HOLD_S + 0.02) / 0.35))), 0.0, 1.0))
        mid, half = (m.lower + m.upper) / 2, (m.upper - m.lower) / 2
        self.lo, self.hi = f64(mid - JOINT_BOX * half), f64(mid + JOINT_BOX * half)
        self.vlim, self.q_default, self.sole = f64(m.speed_limit), f64(m.default_q), f64(m.sole)
        self.heel, self.toe = f64(m.sole[:2].mean(0)), f64(m.sole[2:].mean(0))
        self.feet = torch.as_tensor(m.feet)
        # contacts on the 50 Hz grid: frame j and fraction u of the exact time
        x = (np.array([c["t"] for c in tg.contacts]) - tg.t[0]) * FPS
        j = np.clip(np.floor(x), 0, self.n - 2).astype(int)
        self.c_j, self.c_u = torch.as_tensor(j), f64(np.clip(x - j, 0.0, 1.0))[:, None]
        self.c_p, self.c_n = f64([c["p"] for c in tg.contacts]), f64([c["n"] for c in tg.contacts])
        self.c_sign = f64([1.0 if c["face"] == 0 else -1.0 for c in tg.contacts])
        self._collision_setup()
        self._dynamics_setup()

    def _collision_setup(self) -> None:
        import torch

        m = self.m
        on = lambda links: [k for k, b in enumerate(m.sphere_body) if b in links]  # noqa: E731
        trunk = on(m.chain(m.torso))
        legs = on(m.chain(m.feet[0])[1:-1]) + on(m.chain(m.feet[1])[1:-1])
        forearm = lambda arm: on(m.chain(arm[2])[m.chain(arm[2]).index(arm[1]) :])  # noqa: E731
        paddle = on([len(m.links)])
        pairs = [(a, b) for a in paddle for b in trunk + legs + forearm(m.free_arm)]
        pairs += [(a, b) for a in forearm(m.racket_arm) + forearm(m.free_arm) for b in trunk + legs]
        pairs += [(a, b) for a in on(m.chain(m.feet[0])[1:-1]) for b in on(m.chain(m.feet[1])[1:-1])]
        self.pairs = torch.as_tensor(pairs)
        self.floor_spheres = torch.as_tensor([k for k in range(len(m.sphere_body)) if m.sphere_body[k] not in m.feet])

    def _dynamics_setup(self) -> None:
        """Per-joint acceleration weights ((subtree inertia about the axis / peak torque)^2 in the stance, so smoothing
        approximates torque use), link masses for the ZMP, and the plants."""
        import torch

        m = self.m
        rot, pos = body_poses(m, np.zeros((1, 3)), np.eye(3)[None], m.default_q[None])
        rot, pos = rot[0, :-1], pos[0, :-1]
        com = np.einsum("lij,lj->li", rot, m.com) + pos
        inertia = np.einsum("lij,ljk,lmk->lim", rot, m.inertia, rot)
        eff = np.zeros(m.nq)
        for i in range(1, len(m.links)):
            a = rot[i] @ m.axis[i]
            sub = [k for k in range(len(m.links)) if i in m.chain(k)]
            eff[i - 1] = m.armature[i - 1] + sum(a @ inertia[k] @ a + m.mass[k] * np.linalg.norm(np.cross(com[k] - pos[i], a)) ** 2 for k in sub)
        self.acc_weight = torch.as_tensor(np.maximum((eff / m.peak_torque) ** 2, 3e-6))
        self.link_mass, self.link_com = torch.as_tensor(m.mass), torch.as_tensor(m.com)
        # plants: each (foot, run of planted frames) an id; frames off a plant go to a spare bin
        ids, k = np.full((self.n, 2), -1, dtype=np.int64), 0
        for f in (0, 1):
            edges = np.diff(np.r_[0, self.tg.planted[:, f].astype(np.int8), 0])
            for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True):
                ids[a:b, f], k = k, k + 1
        self.plant_valid = torch.as_tensor(ids >= 0, dtype=torch.float64)
        self.plant_bin = torch.as_tensor(np.where(ids >= 0, ids, k))
        self.plant_count = torch.zeros(k + 1, dtype=torch.float64).index_add(0, self.plant_bin.reshape(-1), self.plant_valid.reshape(-1)).clamp(min=1.0)
        self.n_plants = k

    # ------------------------------------------------------------------------------------------- the cost

    def at_contacts(self, root_p, root_rv, q):
        """Blade frame and position at the exact contact times (the variables interpolated linearly)."""
        import torch

        j, u = self.c_j, self.c_u
        rp, rv, qq = (x[j] * (1 - u) + x[j + 1] * u for x in (root_p, root_rv, q))
        pr = torch.where((u < 0.5)[..., None], self.pelvis_r[j], self.pelvis_r[j + 1])
        rots, poss = self.g.fk(rp, pr @ rotvec_to_matrix(rv), qq)
        return rots[:, -1], poss[:, -1]

    def terms(self, root_p, root_rv, q) -> dict[str, torch.Tensor]:
        """Every cost term, summed over the segment."""
        import torch

        w, dt, g = self.w, 1.0 / FPS, 9.81
        root_r = self.pelvis_r @ rotvec_to_matrix(root_rv)
        rots, poss = self.g.fk(root_p, root_r, q)
        t: dict[str, torch.Tensor] = {}
        # 1. paddle
        pr, pp = rots[:, -1], poss[:, -1]
        t["blade_pos"] = (w.blade_pos * (w.blade_pos_window / w.blade_pos) ** self.rho * ((pp - self.paddle_p) ** 2).sum(-1)).sum()
        t["blade_nrm"] = (w.blade_nrm * (w.blade_nrm_window / w.blade_nrm) ** self.rho * ((pr[..., :, 1] - self.paddle_r[..., :, 1]) ** 2).sum(-1)).sum()
        t["blade_tip"] = (w.blade_tip * ((pr[..., :, 0] - self.paddle_r[..., :, 0]) ** 2).sum(-1)).sum()
        if len(self.c_j):
            r0, p0 = self.at_contacts(root_p, root_rv, q)
            nn = self.c_sign[:, None] * r0[..., :, 1]
            t["contact"] = (w.contact_pos * ((p0 - self.c_p) ** 2).sum(-1) + w.contact_nrm * ((nn - self.c_n) ** 2).sum(-1)).sum()
        # 2. body
        t["pelvis_rot"] = w.pelvis_rot * ((root_r - self.pelvis_r) ** 2).sum()
        t["torso_rot"] = (w.torso_rot * (1 - 0.8 * self.near) * ((rots[:, self.m.torso] - self.torso_r) ** 2).sum((-1, -2))).sum()
        t["pelvis_xy"] = w.pelvis_xy * ((root_p[:, :2] - self.pelvis_p[:, :2]) ** 2).sum()
        t["pelvis_z"] = w.pelvis_z * ((root_p[:, 2] - self.pelvis_p[:, 2]) ** 2).sum()
        # 3. feet
        fr, fp, pl = rots[:, self.feet], poss[:, self.feet], self.planted.to(torch.float64)
        t["foot_xy"] = w.foot_xy * ((fp[..., :2] - self.foot_p[..., :2]) ** 2).sum()
        ez = fp[..., 2] - self.foot_p[..., 2]
        t["foot_z"] = (w.foot_z * pl * ez * ez + 0.01 * w.foot_z * (1 - pl) * ez * ez).sum()
        t["foot_flat"] = (w.foot_flat * pl * (1 - fr[..., 2, 2]) + 0.1 * w.foot_flat * (1 - fr[..., 2, 2])).sum()
        yaw = torch.atan2(fr[..., 1, 0], fr[..., 0, 0])
        t["foot_yaw"] = w.foot_yaw * (1 - torch.cos(yaw - self.foot_yaw)).sum()
        both = pl[1:] * pl[:-1]
        t["foot_slide"] = (w.foot_slide * both * ((fp[1:] - fp[:-1]) ** 2).sum(-1) + w.foot_slide * 0.01 * both * (1 - torch.cos(yaw[1:] - yaw[:-1]))).sum()
        corners = fp[..., None, :] + (fr[..., None, :, :] @ self.sole[:, :, None])[..., 0]
        t["sole_floor"] = 10 * w.foot_z * (torch.clamp(FLOOR_Z_M - corners[..., 2], min=0) ** 2).sum()
        # 5. style (the free hand about its shoulder in the torso frame) and joint smoothing
        hand, shoulder = self.m.free_arm[2], self.m.free_arm[0]
        lrel = torch.einsum("tji,tj->ti", rots[:, self.m.torso], poss[:, hand] - poss[:, shoulder])
        t["lhand"] = w.lhand * ((lrel - self.lhand) ** 2).sum()
        qdd = (q[2:] - 2 * q[1:-1] + q[:-2]) / dt**2
        t["acc_q"] = w.acc_q * (self.acc_weight * qdd**2).sum()
        # 4. balance: the multi-body linear-inverted-pendulum ZMP inside the planted soles, and friction
        com = poss[:, :-1] + (rots[:, :-1] @ self.link_com[:, :, None])[..., 0]
        c = (com * self.link_mass[:, None]).sum(1) / self.link_mass.sum()
        add = (com[2:] - 2 * com[1:-1] + com[:-2]) / dt**2
        h = com[1:-1, :, 2:3] - FLOOR_Z_M
        zmp = c[1:-1, :2] - (self.link_mass[None, :, None] * h * add[..., :2]).sum(1) / (self.link_mass.sum() * g)
        heel = (fp + (fr @ self.heel[:, None])[..., 0])[1:-1, :, :2]
        toe = (fp + (fr @ self.toe[:, None])[..., 0])[1:-1, :, :2]
        out = support_distance(zmp, heel, toe, self.planted[1:-1]) - SOLE_HALF_WIDTH_M
        t["zmp"] = w.zmp * (torch.clamp(out, min=0) ** 2).sum()
        cdd = (c[2:] - 2 * c[1:-1] + c[:-2]) / dt**2
        t["friction"] = w.friction * (torch.clamp(cdd[:, :2].norm(dim=-1) / g - 0.5, min=0) ** 2).sum()
        # 3. feet, continued: every planted frame's heel and toe on its plant's mean
        pts = torch.cat([fp + (fr @ self.heel[:, None])[..., 0], fp + (fr @ self.toe[:, None])[..., 0]], -1)
        sums = torch.zeros(self.n_plants + 1, 6, dtype=fp.dtype).index_add(0, self.plant_bin.reshape(-1), (pts * self.plant_valid[..., None]).reshape(-1, 6))
        t["plant_rigid"] = w.plant_rigid * (((pts - (sums / self.plant_count[:, None])[self.plant_bin]) ** 2).sum(-1) * self.plant_valid).sum()
        # 5. smoothing, continued
        rdd = (root_p[2:] - 2 * root_p[1:-1] + root_p[:-2]) / dt**2
        vdd = (root_rv[2:] - 2 * root_rv[1:-1] + root_rv[:-2]) / dt**2
        t["acc_root"] = w.acc_root * ((rdd**2).sum(-1) + 0.3 * (vdd**2).sum(-1)).sum()
        over = torch.clamp(((q[2:] - q[:-2]) / (2 * dt)).abs() / self.vlim - w.speed_fraction, min=0)
        t["speed"] = w.speed * (over**2).sum()
        t["stance"] = w.stance * ((q - self.q_default) ** 2).sum()
        sph, rad = self.g.spheres(rots, poss), self.g.sphere_radius
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        t["self"] = w.collide * (torch.clamp(rad[a] + rad[b] + 0.005 - (sph[:, a] - sph[:, b]).norm(dim=-1), min=0) ** 2).sum()
        d = (sph - torch.tensor([0.0, 0.0, -TABLE_HALF_M[2]], dtype=torch.float64)).abs() - torch.tensor(TABLE_HALF_M, dtype=torch.float64)
        sdf = torch.clamp(d, min=0).norm(dim=-1) + torch.clamp(d.max(-1).values, max=0)
        t["table"] = w.collide * (torch.clamp(rad + 0.005 - sdf, min=0) ** 2).sum()
        fs = self.floor_spheres
        t["floor"] = w.collide * (torch.clamp(FLOOR_Z_M + rad[fs] - sph[:, fs, 2], min=0) ** 2).sum()
        return t

    def cost(self, x: torch.Tensor) -> torch.Tensor:
        """The total cost of ``x`` (T, J + 6): [u (J), pelvis position (3), pelvis rotation residual (3)]."""
        import torch

        nj = self.m.nq
        return torch.stack(list(self.terms(x[:, nj + 0 : nj + 3], x[:, nj + 3 : nj + 6], self.q_of(x[:, :nj])).values())).sum()

    def q_of(self, u):
        import torch

        return (self.hi + self.lo) / 2 + (self.hi - self.lo) / 2 * torch.sin(u)

    def u_of(self, q):
        import torch

        return torch.asin(((q - (self.hi + self.lo) / 2) / ((self.hi - self.lo) / 2)).clamp(-0.999, 0.999))

    # ------------------------------------------------------------------------------------------- the solve

    def initial_ik(self, iters: int = 25) -> torch.Tensor:
        """Per-frame Levenberg-Marquardt IK (frames independent, batched) from the stance on the frame-local terms: paddle,
        pelvis, torso, feet held on their targets and flat, the free hand, the stance pull. The trajectory solve then
        only reconciles frames instead of discovering poses."""
        import torch
        from torch.func import jacfwd, vmap

        w, g, m = self.w, self.g, self.m
        pl = self.planted.to(torch.float64)
        mid, half = (self.hi + self.lo) / 2, (self.hi - self.lo) / 2
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        tg = {
            "pelvis_r": self.pelvis_r,
            "pelvis_p": self.pelvis_p,
            "paddle_p": self.paddle_p,
            "paddle_r": self.paddle_r,
            "torso_r": self.torso_r,
            "foot_p": self.foot_p,
            "foot_yaw": self.foot_yaw,
            "lhand": self.lhand,
            "wpos": (w.blade_pos * (w.blade_pos_window / w.blade_pos) ** self.rho).sqrt(),
            "wn": (w.blade_nrm * (w.blade_nrm_window / w.blade_nrm) ** self.rho).sqrt(),
            "wt": (w.torso_rot * (1 - 0.8 * self.near)).sqrt(),
            "wfz": (w.foot_z * pl + 0.01 * w.foot_z * (1 - pl)).sqrt(),
            "wff": (w.foot_flat * pl + 0.1 * w.foot_flat * (1 - pl)).sqrt(),
            "wfxy": torch.full_like(pl, 300.0).sqrt(),
        }

        def residual(x, tg):
            rp, rv, q = x[:3], x[3:6], mid + half * torch.sin(x[6:])
            rots, poss = g.fk(rp, tg["pelvis_r"] @ rotvec_to_matrix(rv), q)
            res = [
                tg["wpos"] * (poss[-1] - tg["paddle_p"]),
                tg["wn"] * (rots[-1][:, 1] - tg["paddle_r"][:, 1]),
                (w.blade_tip**0.5) * (rots[-1][:, 0] - tg["paddle_r"][:, 0]),
                torch.stack(
                    [
                        w.pelvis_xy**0.5 * (rp[0] - tg["pelvis_p"][0]),
                        w.pelvis_xy**0.5 * (rp[1] - tg["pelvis_p"][1]),
                        w.pelvis_z**0.5 * (rp[2] - tg["pelvis_p"][2]),
                    ]
                ),
                (w.pelvis_rot**0.5) * rv,
                tg["wt"] * (rots[m.torso] - tg["torso_r"]).reshape(-1),
            ]
            for k, f in enumerate(m.feet):
                res += [
                    tg["wfz"][k] * (poss[f][2:] - tg["foot_p"][k, 2:]),
                    tg["wfxy"][k] * (poss[f][:2] - tg["foot_p"][k, :2]),
                    tg["wff"][k] * (rots[f][:, 2] - up),
                ]
                res.append((w.foot_yaw**0.5) * (rots[f][:2, 0] - torch.stack([torch.cos(tg["foot_yaw"][k]), torch.sin(tg["foot_yaw"][k])])))
            res.append((w.lhand**0.5) * (rots[m.torso].T @ (poss[m.free_arm[2]] - poss[m.free_arm[0]]) - tg["lhand"]))
            res.append((w.stance**0.5) * (q - self.q_default))
            return torch.cat([r.reshape(-1) for r in res])

        x = torch.cat([self.pelvis_p, torch.zeros(self.n, 3, dtype=torch.float64), self.u_of(self.q_default.expand(self.n, -1))], 1)
        jac, res_f = vmap(jacfwd(residual)), vmap(residual)
        for a in range(0, self.n, 4096):
            sl = slice(a, min(self.n, a + 4096))
            tgs, xs = {k: v[sl] for k, v in tg.items()}, x[sl]
            lam = torch.full((xs.shape[0], 1, 1), 1e-2, dtype=torch.float64)
            r = res_f(xs, tgs)
            cost = (r * r).sum(-1)
            eye = torch.eye(xs.shape[1], dtype=torch.float64)
            for _ in range(iters):
                jm = jac(xs, tgs)
                h = jm.transpose(1, 2) @ jm
                step = torch.linalg.solve(h + lam * (torch.diagonal(h, dim1=1, dim2=2)[..., None] * eye + 1e-6 * eye), -(jm.transpose(1, 2) @ r[..., None]))[
                    ..., 0
                ]
                rn = res_f(xs + step, tgs)
                cn = (rn * rn).sum(-1)
                better = cn < cost
                xs, r, cost = torch.where(better[:, None], xs + step, xs), torch.where(better[:, None], rn, r), torch.where(better, cn, cost)
                lam = torch.where(better[:, None, None], lam * 0.3, lam * 10.0).clamp(1e-7, 1e7)
            x[sl] = xs
        return torch.cat([self.u_of(self.q_of(x[:, 6:])), x[:, :3], x[:, 3:6]], 1)  # [u, pelvis position, rotation residual]

    def solve(self, adam_iters: int = 200, lbfgs_iters: int = 6000, lr: float = 0.02) -> Solution:
        """LM init, Adam (one-cycle schedule) without the balance and plant terms, then L-BFGS over four rounds of rising
        balance weight."""
        import torch

        t0, full, nj = time.time(), self.w, self.m.nq
        x = self.initial_ik().detach()
        params = [x[:, :nj].clone(), x[:, nj : nj + 3].clone(), x[:, nj + 3 :].clone()]
        for p in params:
            p.requires_grad_(True)
        joined = lambda: torch.cat(params, 1)  # noqa: E731
        self.w = replace(full, zmp=0.0, friction=0.0, plant_rigid=0.0)
        m1, m2 = [torch.zeros_like(p) for p in params], [torch.zeros_like(p) for p in params]
        rates, warm = (lr, lr * 0.25, lr * 0.5), max(1, int(0.1 * adam_iters))
        for it in range(adam_iters):
            if it < warm:
                f = 1 / 25 + (1 - 1 / 25) * (1 - np.cos(np.pi * it / warm)) / 2
            else:
                f = 1 / 25e4 + (1 - 1 / 25e4) * (1 + np.cos(np.pi * (it - warm) / max(1, adam_iters - warm))) / 2
            grads = torch.autograd.grad(self.cost(joined()), params)
            with torch.no_grad():
                for k, (p, gk) in enumerate(zip(params, grads, strict=True)):
                    m1[k].mul_(0.9).add_(gk, alpha=0.1)
                    m2[k].mul_(0.999).addcmul_(gk, gk, value=0.001)
                    p.sub_(rates[k] * f * (m1[k] / (1 - 0.9 ** (it + 1))) / ((m2[k] / (1 - 0.999 ** (it + 1))).sqrt() + 1e-8))
        for scale in (0.1, 1.0, 10.0, 30.0):
            self.w = replace(full, zmp=full.zmp * scale, friction=full.friction * scale)
            opt = torch.optim.LBFGS(
                params, lr=1.0, max_iter=max(1, lbfgs_iters // 4), history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12
            )

            def closure(opt=opt):
                opt.zero_grad()
                c = self.cost(joined())
                c.backward()
                return c

            opt.step(closure)
        self.w = full
        with torch.no_grad():
            x = joined()
            cost = float(self.cost(x))
            root_r = self.pelvis_r @ rotvec_to_matrix(x[:, nj + 3 :])
            return Solution(
                t=self.tg.t, root_p=x[:, nj : nj + 3].numpy().copy(), root_r=root_r.numpy(), q=self.q_of(x[:, :nj]).numpy(), cost=cost, seconds=time.time() - t0
            )


# ---------------------------------------------------------------------------------------------------- static reach


def _rz(a):
    import torch

    c, s, o, one = torch.cos(a), torch.sin(a), torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([torch.stack([c, -s, o], -1), torch.stack([s, c, o], -1), torch.stack([o, o, one], -1)], -2)


def solve_reach(m: RobotModel, problem: dict[str, np.ndarray], iters: int = 400) -> dict[str, np.ndarray]:
    """See the module docstring. ``problem`` (``targets.reach_problem``): ``p``/``n`` (k, S, 3) the paddle path's blade
    centre and struck normal at S instants through each contact, ``face`` (k,), ``pelvis_p``/``pelvis_r`` (k, S, 3) and
    (k, S, 3, 3) the pelvis targets then. One body offset per contact serves its instants. Returns per contact the body
    offset ``xy`` (k, 2), whether it is reachable ``from_stance`` and at all (``reachable``), and the worst instant's
    errors."""
    import torch

    p, n, face, pelvis_p, pelvis_r = (problem[k] for k in ("p", "n", "face", "pelvis_p", "pelvis_r"))
    k0, ns = p.shape[:2]
    f64 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64).reshape(k0 * ns, *np.shape(a)[2:]))  # noqa: E731
    blade, normal, pp, pr = f64(p), f64(n), f64(pelvis_p), f64(pelvis_r)
    sign = torch.as_tensor(np.repeat(np.where(np.asarray(face) == 0, 1.0, -1.0), ns))
    own = torch.as_tensor(np.repeat(np.arange(k0), ns))
    g, k = TorchKinematics(m), k0 * ns
    free = torch.as_tensor(m.joints_moving(m.paddle_link))
    trunk = torch.as_tensor(m.joints_moving(m.torso))
    lo, hi = m.lower[free.numpy()], m.upper[free.numpy()]
    mid, half = torch.as_tensor((lo + hi) / 2), torch.as_tensor((hi - lo) / 2 * REACH_COMFORT)
    q0 = torch.as_tensor(m.default_q).expand(k, -1).clone()

    def run(offset_on: bool):
        u = torch.zeros(k, len(free), dtype=torch.float64, requires_grad=True)
        off = torch.zeros(k0, 4, dtype=torch.float64, requires_grad=True)  # dx, dy, yaw, dz per contact

        def cost():
            q = q0.clone()
            q[:, free] = mid + half * torch.sin(u)
            oc = off if offset_on else off * 0.0
            o = oc[own]
            rots, poss = g.fk(pp + torch.stack([o[:, 0], o[:, 1], o[:, 3]], -1), _rz(o[:, 2]) @ pr, q)
            e, en = poss[:, -1] - blade, sign[:, None] * rots[:, -1, :, 1] - normal
            c = (1e4 * (e * e).sum(-1) + 1e3 * (en * en).sum(-1) + 0.05 * (q[:, trunk] ** 2).sum(-1)).sum()
            c = c + (100.0 * (oc[:, :2] ** 2).sum(-1) + 5.0 * oc[:, 2] ** 2 + 20.0 * oc[:, 3] ** 2 + 5.0 * torch.relu(oc[:, 3]) ** 2).sum()
            return c, e, rots

        best = None
        for seed in range(3):  # three arm starts against elbow-flip minima; the best per contact
            with torch.no_grad():
                gen = torch.Generator(device="cpu").manual_seed(seed)
                u.copy_((torch.rand(k, len(free), generator=gen, dtype=torch.float64) - 0.5) * (0.0 if seed == 0 else 1.5))
                off.zero_()
            opt = torch.optim.LBFGS([u, off], lr=1.0, max_iter=iters, history_size=30, line_search_fn="strong_wolfe")

            def closure(opt=opt):
                opt.zero_grad()
                c, *_ = cost()
                c.backward()
                return c

            opt.step(closure)
            with torch.no_grad():
                _c, e, rots = cost()
                pos = e.norm(dim=-1).reshape(k0, ns).max(1).values
                ang = torch.rad2deg(torch.arccos((sign[:, None] * rots[:, -1, :, 1] * normal).sum(-1).clamp(-1, 1))).reshape(k0, ns).max(1).values
                score = pos / POS_TOL_M + ang / FACE_TOL_DEG + (off[:, :2].norm(dim=-1) / 0.05 if offset_on else 0.0)
                cand = (score.clone(), pos.clone(), ang.clone(), off.detach().clone())
            better = cand[0] < best[0] if best is not None else torch.ones(k0, dtype=torch.bool)
            best = cand if best is None else tuple(torch.where(better.reshape(-1, *([1] * (c.dim() - 1))), c, b) for c, b in zip(cand, best, strict=True))
        assert best is not None
        return best[1].numpy(), best[2].numpy(), best[3].numpy()

    pos_s, ang_s, _ = run(False)
    pos_f, ang_f, off = run(True)
    from_stance = (pos_s <= POS_TOL_M) & (ang_s <= FACE_TOL_DEG)
    off[from_stance] = 0.0  # reachable where the human stood: the body stays there
    pos_f, ang_f = np.where(from_stance, pos_s, pos_f), np.where(from_stance, ang_s, ang_f)
    return {
        "xy": off[:, :2],
        "from_stance": from_stance,
        "reachable": (pos_f <= POS_TOL_M) & (ang_f <= FACE_TOL_DEG),
        "pos_err_stance_m": pos_s,
        "face_err_stance_deg": ang_s,
    }


__all__ = ["JOINT_BOX", "Problem", "Solution", "Weights", "solve_reach", "support_distance"]
