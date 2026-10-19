"""
The robot as the retargeter sees it: a kinematic tree from the URDF, joint limits, motors and mass properties, the
paddle on its wrist, feet and coarse collision spheres. Everything comes from one :class:`~lib.robots.spec.RobotSpec`.

**What a robot provides** (``RobotSpec``; the G1's is ``lib/robots/unitree_g1.py``):

* ``urdf`` and ``urdf_sha256``: kinematics, joint limits and link inertias (every frame the URDF's; revolute joints);
* ``actuators``: per joint the peak torque, speed limit, torque-speed knee and no-load speed, rotor armature;
* ``default_pose``: the ready stance (the solver's weak pull and the reach solve's legs);
* ``paddle``: the wrist link and the blade centre's offset along its +x; the paddle frame is that link's axes (x towards
  the tip, y the forehand normal), and its mass rides on the link;
* ``links``: pelvis (the floating base), torso and the two feet;
* ``retarget``: the racket and free arms (shoulder, elbow, wrist links), the URDF links to leave out, the sole's
  contact corners in the foot frame, and coarse collision spheres per link.

Joint groups come from the tree: the **reach joints** move the paddle link (G1: waist and right arm), the **trunk
joints** move the torso, a **leg** is the chain to a foot.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from lib.robots.spec import RobotSpec

PADDLE_SPHERES = (((0.0, 0.0, 0.0), 0.076), ((-0.045, 0.0, 0.0), 0.06), ((-0.12, 0.0, 0.0), 0.025), ((-0.16, 0.0, 0.0), 0.022))
"""The paddle's coarse spheres in its frame (blade, blade root, handle), m."""


@dataclass(frozen=True)
class RobotModel:
    """
    The tree: ``links[0]`` is the pelvis and link ``i > 0`` turns on revolute joint ``joint_names[i - 1]``. Bodies are
    the links plus the paddle (index ``len(links)``), whose frame is its wrist link's, at the blade centre.
    """

    name: str
    links: list[str]
    parents: np.ndarray  # (L,) parent link, -1 for the pelvis
    origin_xyz: np.ndarray  # (L, 3) joint origin in the parent frame
    origin_rot: np.ndarray  # (L, 3, 3)
    axis: np.ndarray  # (L, 3) joint axis in the link frame
    joint_names: list[str]  # URDF order
    lower: np.ndarray  # (J,) rad
    upper: np.ndarray
    speed_limit: np.ndarray  # (J,) rad/s
    peak_torque: np.ndarray  # (J,) N m
    knee_speed: np.ndarray  # (J,) rad/s, the peak holds up to here
    no_load_speed: np.ndarray  # (J,) rad/s, zero torque here
    armature: np.ndarray  # (J,) kg m^2
    mass: np.ndarray  # (L,) kg, fixed children and the paddle merged
    com: np.ndarray  # (L, 3) in the link frame
    inertia: np.ndarray  # (L, 3, 3) about the COM, link axes
    paddle_link: int
    paddle_offset: np.ndarray  # (3,) blade centre in the paddle link's frame
    default_q: np.ndarray  # (J,)
    sole: np.ndarray  # (4, 3) heel pair then toe pair, foot frame
    sphere_body: np.ndarray  # (S,) body of each collision sphere
    sphere_centre: np.ndarray  # (S, 3) in the body frame
    sphere_radius: np.ndarray  # (S,)
    torso: int
    feet: tuple[int, int]
    racket_arm: tuple[int, int, int]  # shoulder, elbow, wrist
    free_arm: tuple[int, int, int]
    urdf_sha256: str

    @property
    def nq(self) -> int:
        return len(self.joint_names)

    @property
    def body_names(self) -> list[str]:
        return [*self.links, "paddle"]

    @property
    def total_mass(self) -> float:
        return float(self.mass.sum())

    def chain(self, link: int) -> list[int]:
        """Link indices from the pelvis down to ``link``, inclusive."""
        out = [link]
        while self.parents[out[-1]] >= 0:
            out.append(int(self.parents[out[-1]]))
        return out[::-1]

    @property
    def hips(self) -> tuple[int, int]:
        """The first link of each leg (the leg length runs from its joint to the ankle)."""
        return self.chain(self.feet[0])[1], self.chain(self.feet[1])[1]

    def joints_moving(self, link: int) -> list[int]:
        """The joints on the chain to ``link`` (joint ``i - 1`` turns link ``i``)."""
        return [i - 1 for i in self.chain(link)[1:]]

    def torque_available(self, qd: np.ndarray) -> np.ndarray:
        """Each motor's torque at joint speed ``qd`` (..., J): the peak up to the knee speed, falling linearly to zero at the
        no-load speed; zero beyond the speed limit, where the simulator clamps."""
        s = np.abs(qd)
        falling = self.peak_torque * np.clip((self.no_load_speed - s) / (self.no_load_speed - self.knee_speed), 0.0, 1.0)
        return np.where(s <= self.speed_limit, np.where(s <= self.knee_speed, self.peak_torque, falling), 0.0)


def _rpy(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _vec(el: ET.Element | None, attr: str) -> np.ndarray:
    s = None if el is None else el.get(attr)
    return np.zeros(3) if s is None else np.array([float(v) for v in s.split()])


def _child(el: ET.Element, tag: str) -> ET.Element:
    found = el.find(tag)
    if found is None:
        raise ValueError(f"URDF element <{el.tag} name={el.get('name')!r}> has no <{tag}>")
    return found


def _combine(parts: list[tuple[float, np.ndarray, np.ndarray]]) -> tuple[float, np.ndarray, np.ndarray]:
    """One rigid body from (mass, com, inertia about the com) parts in one frame."""
    m = sum(p[0] for p in parts)
    if m <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    c = np.sum([p[0] * p[1] for p in parts], axis=0) / m
    inertia = np.zeros((3, 3))
    for pm, pc, pi in parts:
        d = pc - c
        inertia += pi + pm * (d @ d * np.eye(3) - np.outer(d, d))
    return m, c, inertia


def robot_model(spec: RobotSpec) -> RobotModel:
    """The retargeting model of a robot (see the module docstring)."""
    if spec.retarget is None:
        raise ValueError(f"{spec.name} has no RetargetSpec: it cannot be retargeted onto yet")
    raw = spec.urdf.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec.urdf_sha256:
        raise ValueError(f"{spec.urdf} is not the pinned URDF (sha256 {spec.urdf_sha256[:16]}...)")
    root = ET.fromstring(raw)
    inertial = {}
    for link in root.findall("link"):
        el = link.find("inertial")
        if el is None:
            inertial[str(link.get("name"))] = (0.0, np.zeros(3), np.zeros((3, 3)))
            continue
        rot = _rpy(_vec(el.find("origin"), "rpy"))
        a = {k: float(v) for k, v in _child(el, "inertia").attrib.items()}
        i = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]])
        inertial[str(link.get("name"))] = (float(_child(el, "mass").get("value", "0")), _vec(el.find("origin"), "xyz"), rot @ i @ rot.T)
    joints = []
    for j in root.findall("joint"):
        limit = j.find("limit")
        joints.append(
            {
                "name": str(j.get("name")),
                "type": j.get("type"),
                "parent": str(_child(j, "parent").get("link")),
                "child": str(_child(j, "child").get("link")),
                "xyz": _vec(j.find("origin"), "xyz"),
                "rot": _rpy(_vec(j.find("origin"), "rpy")),
                "axis": _vec(j.find("axis"), "xyz"),
                "range": (float(limit.get("lower", "0")), float(limit.get("upper", "0"))) if limit is not None else (0.0, 0.0),
            }
        )
    # the movable tree, parents first (URDF order is depth-first from the pelvis)
    links, parents, xyz, rots, axes, names, lower, upper = [spec.links.pelvis], [-1], [np.zeros(3)], [np.eye(3)], [np.zeros(3)], [], [], []
    for j in joints:
        if j["type"] == "revolute":
            links.append(j["child"])
            parents.append(links.index(j["parent"]))
            xyz.append(j["xyz"])
            rots.append(j["rot"])
            axes.append(j["axis"])
            names.append(j["name"])
            lower.append(j["range"][0])
            upper.append(j["range"][1])
    if set(names) != set(spec.joint_names):
        raise ValueError(f"{spec.name}: the URDF's revolute joints are not the spec's joint_names")
    # fixed children (head, sensors) merge into their movable ancestor; the dropped links (the hands) go
    parts = {name: [inertial[name]] for name in links}
    for j in joints:
        if j["type"] != "fixed" or j["child"] in spec.retarget.urdf_drop_links:
            continue
        parent, rot, tr = j["parent"], j["rot"], j["xyz"]
        while parent not in parts:
            up = next(k for k in joints if k["child"] == parent)
            rot, tr, parent = up["rot"] @ rot, up["rot"] @ tr + up["xyz"], up["parent"]
        m, c, i = inertial[j["child"]]
        parts[parent].append((m, rot @ c + tr, rot @ i @ rot.T))
    pad = spec.paddle
    offset = np.array([pad.offset_m, 0.0, 0.0])
    parts[pad.link].append((pad.mass_kg, offset + np.asarray(pad.com_m), np.diag(pad.inertia)))
    bodies = [_combine(parts[name]) for name in links]
    motors = [spec.motor(n) for n in names]
    default_q = np.array([next((v for e, v in spec.default_pose.items() if re.fullmatch(e, n)), 0.0) for n in names])
    rt = spec.retarget
    spheres = [(links.index(link), c, r) for link, items in rt.collision_spheres.items() for c, r in items]
    spheres += [(len(links), c, r) for c, r in PADDLE_SPHERES]
    return RobotModel(
        name=spec.name,
        links=links,
        parents=np.array(parents),
        origin_xyz=np.array(xyz),
        origin_rot=np.array(rots),
        axis=np.array(axes),
        joint_names=names,
        lower=np.array(lower),
        upper=np.array(upper),
        speed_limit=np.array([m.speed_limit for m in motors]),
        peak_torque=np.array([m.peak_torque for m in motors]),
        knee_speed=np.array([m.knee_speed for m in motors]),
        no_load_speed=np.array([m.no_load_speed for m in motors]),
        armature=np.array([m.armature for m in motors]),
        mass=np.array([b[0] for b in bodies]),
        com=np.array([b[1] for b in bodies]),
        inertia=np.array([b[2] for b in bodies]),
        paddle_link=links.index(pad.link),
        paddle_offset=offset,
        default_q=default_q,
        sole=np.array(rt.sole_corners_m, dtype=np.float64),
        sphere_body=np.array([s[0] for s in spheres]),
        sphere_centre=np.array([s[1] for s in spheres], dtype=np.float64),
        sphere_radius=np.array([s[2] for s in spheres], dtype=np.float64),
        torso=links.index(spec.links.torso),
        feet=(links.index(spec.links.feet[0]), links.index(spec.links.feet[1])),
        racket_arm=tuple(links.index(n) for n in rt.racket_arm),  # type: ignore[arg-type]
        free_arm=tuple(links.index(n) for n in rt.free_arm),  # type: ignore[arg-type]
        urdf_sha256=spec.urdf_sha256,
    )


# ---------------------------------------------------------------------------------------------------- kinematics


def _axis_angle(axis, angle):
    """Rotations (..., 3, 3) about a fixed unit axis (3,) by angles (...) (torch)."""
    import torch

    x, y, z = axis[0], axis[1], axis[2]
    c, s = torch.cos(angle), torch.sin(angle)
    t = 1 - c
    return torch.stack(
        [
            torch.stack([c + x * x * t, x * y * t - z * s, x * z * t + y * s], -1),
            torch.stack([y * x * t + z * s, c + y * y * t, y * z * t - x * s], -1),
            torch.stack([z * x * t - y * s, z * y * t + x * s, c + z * z * t], -1),
        ],
        -2,
    )


def rotvec_to_matrix(v):
    """Rodrigues (..., 3) -> (..., 3, 3), smooth at zero (torch)."""
    import torch

    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-24)
    k = v / theta
    zero = torch.zeros_like(k[..., 0])
    kmat = torch.stack(
        [torch.stack([zero, -k[..., 2], k[..., 1]], -1), torch.stack([k[..., 2], zero, -k[..., 0]], -1), torch.stack([-k[..., 1], k[..., 0], zero], -1)], -2
    )
    th = theta[..., None]
    small = theta2[..., None] < 1e-12
    a = torch.where(small, torch.ones_like(th), torch.sin(th) / th)
    b = torch.where(small, 0.5 * torch.ones_like(th), (1 - torch.cos(th)) / (th * th))
    vx = kmat * th
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand((*v.shape[:-1], 3, 3))
    return eye + a * vx + b * (vx @ vx)


class TorchKinematics:
    """Batched forward kinematics of a :class:`RobotModel` in torch (float64), differentiable."""

    def __init__(self, m: RobotModel, device: str = "cpu") -> None:
        import torch

        self.m = m
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)  # noqa: E731
        self.origin_xyz, self.origin_rot, self.axis, self.paddle_offset = t(m.origin_xyz), t(m.origin_rot), t(m.axis), t(m.paddle_offset)
        self.sphere_centre, self.sphere_radius = t(m.sphere_centre), t(m.sphere_radius)
        self.sphere_body = torch.as_tensor(m.sphere_body, device=device)

    def fk(self, root_p, root_r, q):
        """Body rotations (..., B, 3, 3) and positions (..., B, 3): the links in model order, then the paddle."""
        import torch

        rots, poss = [root_r], [root_p]
        for i in range(1, len(self.m.links)):
            p = int(self.m.parents[i])
            rots.append(rots[p] @ (self.origin_rot[i] @ _axis_angle(self.axis[i], q[..., i - 1])))
            poss.append(poss[p] + (rots[p] @ self.origin_xyz[i][:, None])[..., 0])
        k = self.m.paddle_link
        rots.append(rots[k])
        poss.append(poss[k] + (rots[k] @ self.paddle_offset[:, None])[..., 0])
        return torch.stack(rots, -3), torch.stack(poss, -2)

    def spheres(self, rots, poss):
        """World centres (..., S, 3) of the collision spheres."""
        return poss[..., self.sphere_body, :] + (rots[..., self.sphere_body, :, :] @ self.sphere_centre[:, :, None])[..., 0]


def body_poses(m: RobotModel, root_p: np.ndarray, root_r: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:meth:`TorchKinematics.fk` on numpy arrays: rotations (n, B, 3, 3) and positions (n, B, 3), float64."""
    import torch

    with torch.no_grad():
        r, p = TorchKinematics(m).fk(*(torch.as_tensor(np.asarray(a, dtype=np.float64)) for a in (root_p, root_r, q)))
    return r.numpy(), p.numpy()


__all__ = ["PADDLE_SPHERES", "RobotModel", "TorchKinematics", "body_poses", "robot_model", "rotvec_to_matrix"]
