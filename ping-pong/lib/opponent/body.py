"""
Body-first feasibility: the generator's body runtime as the opponent's stroke check, distilled.

In the v1 release every stroke is a whole body (the offline generator's body runtime, not shipped here):
``Rally._stroke_body`` asks ``BodyRuntime.plan_stroke`` for the drawn contact; the runtime places a captured donor
stroke on the hips plan, corrects its contact with the arm and holds the joints to their limits, and either holds the
contact (its blade centre and face within 2 cm and 5 degrees of the request, its velocity near the donor's swing) or
rejects it (``no_donor``, ``nudge``, ``reach``, ``speed``, ``board``, ``limits``). At execution the noisy contact is
re-held on the first plan's body (``BodyRuntime.renudge``). The runtime (a donor library, inverse kinematics, joint-
speed limits) is far too large to run per environment, so the opponent carries what decides its outcomes:

1. **The reach rule, exact.** The runtime's donor prefilter (``BodyRuntime._rank``): some library contact, turned by up
   to 15 degrees about the hips and moved by up to 10 cm (both shrinking with less than 0.35 s to the contact), must lie
   within 0.135 m of the requested blade centre with its face within 27 degrees of the requested face (either rubber).
   Else the runtime rejects the stroke (``no_donor``). And the least lead, 0.10 s from the replan tick.
2. **Tabulated acceptance** of the rest (``nudge``, ``speed``, ``reach``, ``limits``, ``board``): P(held | features),
   binned and Beta-smoothed, from the runtime's logged calls (:class:`AcceptTable`).
3. **The nudge model**: the held contact against the request (blade velocity, centre, face tilt, angular velocity, in
   the receiver's frame) as a conditional Gaussian mixture (:class:`NudgeModel`).
4. **The same-outcome rule, exact** (the generator's ``_lands_as_requested``): a held contact is kept only if, under
   the player's belief, it sends the predicted ball where the request does (``in``, ``net``, ``own_half``, ``out``).

At execution each attempt (the noisy contact, then the generator's fallbacks) is re-held on the first plan's body
(``renudge``: its acceptance and nudge given how far the request is from the first held contact), and if that fails
or changes the outcome, with at least 0.10 s to go, re-planned with the same donor (``plan_stroke(donor=...)``: its own
acceptance and nudge); the same-outcome rule applies to each.

Every table and weight is in the rules export (``opponent_rules_v<k>.npz``, arrays ``body_*``), fitted offline on the
runtime's own logged calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lib.physics.vectorized._torch import torch
from lib.physics.vectorized.params import VecParams

from .ballistics import compiled, fly
from .ops import dot, norm, quat_from_rotvec, quat_multiply, quat_to_matrix, reference_from_game, yaw_vectors
from .strike import forward

Tensor = Any

CLASS_NAMES: tuple[str, ...] = ("in", "net", "own_half", "out")
"""Where a contact sends the predicted ball (``episode.landing_class``)."""

DECISION_FEATURES: tuple[str, ...] = (
    "lead",
    "rel_x",
    "rel_y",
    "rel_z",
    "n_elev",
    "n_azim",
    "v_x",
    "v_y",
    "v_z",
    "speed",
    "face",
    "profile",
    "hand_x",
    "hand_y",
    "hand_z",
    "hand_dist",
    "hand_speed",
    "move",
)
"""A contact request at a decision, in the receiver's frame (x towards the net): the lead from the replan tick (s), the
blade centre against the hips at contact (x, y; z the height), the struck face's normal (elevation, azimuth, rad), the
blade velocity, the rubber, the profile, the hand (blade) at the replan tick against the contact, its speed, and the
hips' move to the contact."""

EXECUTION_FEATURES: tuple[str, ...] = ("lead", "ask_x", "ask_y", "ask_z", "ask_m", "ask_rad", "dv_x", "dv_y", "dv_z", "dv_m", "speed", "face", "profile")
"""A noisy contact against the first plan's held contact at the same time (``renudge``'s ask), in the receiver's frame."""

NUDGE_TARGETS: tuple[str, ...] = ("dv_x", "dv_y", "dv_z", "dp_x", "dp_y", "dp_z", "tilt_a", "tilt_b", "dw_x", "dw_y", "dw_z")
"""The held contact minus the request (receiver's frame): blade velocity (m/s), centre (m), the struck face's tilt
about the blade's across and tip axes (rad), angular velocity (rad/s). A drawn nudge carries a twelfth column, the
rubber flip (1: the runtime met the ball with the other rubber, its ``face_fallback``; physics treats both alike)."""


@dataclass
class Contact:
    """A blade contact state per row: time (s), centre (m), orientation (game axes, ``(w, x, y, z)``), velocity, angular velocity."""

    t: Tensor
    position: Tensor
    orientation: Tensor
    velocity: Tensor
    angular_velocity: Tensor

    def where(self, mask: Tensor, other: Contact) -> Contact:
        def pick(a: Tensor, b: Tensor) -> Tensor:
            return torch.where(mask.reshape(mask.shape + (1,) * (a.dim() - 1)), a, b)

        return Contact(*(pick(getattr(self, k), getattr(other, k)) for k in ("t", "position", "orientation", "velocity", "angular_velocity")))

    def take(self, index: Tensor) -> Contact:
        return Contact(self.t[index], self.position[index], self.orientation[index], self.velocity[index], self.angular_velocity[index])


# ---- the same-outcome rule ----------------------------------------------------------------------------------------


def landing_class(params: VecParams, env: Tensor, ball_in: Tensor, contact: Contact, *, steps: int | None = None) -> Tensor:
    """
    ``episode.landing_class`` per row: where the contact sends the predicted ball ``ball_in`` under the row's physics
    (the impact law, then the flight): 0 ``in`` (the far half), 1 ``net``, 2 ``own_half``, 3 ``out``.
    """

    rotation = reference_from_game(quat_to_matrix(contact.orientation))
    z = rotation[:, :, 2]
    normal = torch.where((dot(z, ball_in[:, 0:3] - contact.position) >= 0.0)[:, None], z, -z)
    v_out, spin_out = forward(params, env, ball_in, contact.position, rotation, contact.velocity, contact.angular_velocity, normal)
    flight = fly(params, torch.cat([ball_in[:, 0:3], v_out, spin_out], -1), env, steps=steps)
    return classify_landing(params, env, ball_in, flight)


def classify_landing(params: VecParams, env: Tensor, ball_in: Tensor, flight: Any) -> Tensor:
    """The class of a flight (:func:`lib.opponent.ballistics.fly`) from a contact at ``ball_in``."""

    x, y = flight.landing[:, 0], flight.landing[:, 1]
    on_table = (x.abs() <= params.table_half_length[env]) & (y.abs() <= params.table_half_width[env])
    own = on_table & (torch.sign(x) == torch.sign(ball_in[:, 0]))
    out = torch.full_like(env, 3)
    out = torch.where(own, torch.full_like(out, 2), out)
    out = torch.where(flight.on_far_half, torch.zeros_like(out), out)
    return torch.where(flight.net, torch.ones_like(out), out)


# ---- features -----------------------------------------------------------------------------------------------------


def struck_normal(orientation: Tensor, face: Tensor) -> Tensor:
    """The struck rubber's outward normal: the blade's forehand normal (game ``Z``), reversed for the backhand rubber."""

    z = quat_to_matrix(orientation)[:, :, 2]
    return z * (1.0 - 2.0 * face.to(z.dtype))[:, None]


def decision_features(
    profile: Tensor,
    flip: Tensor,
    lead: Tensor,
    contact: Contact,
    face: Tensor,
    pelvis_contact: Tensor,
    hips_from: Tensor,
    hand_position: Tensor,
    hand_velocity: Tensor,
) -> Tensor:
    """:data:`DECISION_FEATURES` per row ``(R, F)``."""

    dtype = contact.position.dtype
    hips3 = torch.cat([pelvis_contact, torch.zeros_like(pelvis_contact[:, :1])], -1)
    rel = yaw_vectors(contact.position - hips3, flip)
    n = yaw_vectors(struck_normal(contact.orientation, face), flip)
    v = yaw_vectors(contact.velocity, flip)
    hand = yaw_vectors(hand_position - contact.position, flip)
    return torch.stack(
        [
            lead,
            rel[:, 0],
            rel[:, 1],
            contact.position[:, 2],
            torch.asin(n[:, 2].clamp(-1.0, 1.0)),
            torch.atan2(n[:, 1], n[:, 0]),
            v[:, 0],
            v[:, 1],
            v[:, 2],
            norm(v),
            face.to(dtype),
            profile.to(dtype),
            hand[:, 0],
            hand[:, 1],
            hand[:, 2],
            norm(hand),
            norm(hand_velocity),
            norm(pelvis_contact - hips_from),
        ],
        -1,
    )


def execution_features(profile: Tensor, flip: Tensor, lead: Tensor, request: Contact, first: Contact, face: Tensor) -> Tensor:
    """
    :data:`EXECUTION_FEATURES` per row: the request against the first plan's held contact carried to the request's time
    (its centre moved at its velocity), as ``renudge`` asks it (centre, face angle, velocity).
    """

    dtype = request.position.dtype
    old = first.position + first.velocity * (request.t - first.t)[:, None]
    ask = yaw_vectors(request.position - old, flip)
    z_req = quat_to_matrix(request.orientation)[:, :, 2]
    z_old = quat_to_matrix(first.orientation)[:, :, 2]
    angle = torch.acos(dot(z_req, z_old).abs().clamp(max=1.0))
    dv = yaw_vectors(request.velocity - first.velocity, flip)
    return torch.stack(
        [
            lead,
            ask[:, 0],
            ask[:, 1],
            ask[:, 2],
            norm(ask),
            angle,
            dv[:, 0],
            dv[:, 1],
            dv[:, 2],
            norm(dv),
            norm(request.velocity),
            face.to(dtype),
            profile.to(dtype),
        ],
        -1,
    )


def rubber_flipped(nudge: Tensor) -> Tensor:
    """Whether a drawn nudge meets the ball with the other rubber (its twelfth column; none without one)."""

    if nudge.shape[1] <= len(NUDGE_TARGETS):
        return torch.zeros(nudge.shape[0], dtype=torch.bool, device=nudge.device)
    return nudge[:, len(NUDGE_TARGETS)] > 0.5


def apply_nudge(request: Contact, nudge: Tensor, flip: Tensor) -> Contact:
    """
    The held contact: ``request`` moved by ``nudge`` (:data:`NUDGE_TARGETS`, receiver's frame). With a rubber flip the
    blade is first turned half a turn about its tip axis (the other rubber faces where the requested one did).
    """

    m = quat_to_matrix(request.orientation)
    rotvec = nudge[:, 6:7] * m[:, :, 0] + nudge[:, 7:8] * m[:, :, 1]
    half_turn = torch.zeros_like(request.orientation)
    half_turn[:, 2] = 1.0
    base = torch.where(rubber_flipped(nudge)[:, None], quat_multiply(request.orientation, half_turn), request.orientation)
    return Contact(
        request.t,
        request.position + yaw_vectors(nudge[:, 3:6], flip),
        quat_multiply(quat_from_rotvec(rotvec), base),
        request.velocity + yaw_vectors(nudge[:, 0:3], flip),
        request.angular_velocity + yaw_vectors(nudge[:, 8:11], flip),
    )


# ---- the reach rule -------------------------------------------------------------------------------------------------


def _reach_core(
    px: Tensor,
    py: Tensor,
    pz: Tensor,
    zx: Tensor,
    zy: Tensor,
    zz: Tensor,
    smax: Tensor,
    cx: Tensor,
    cy: Tensor,
    cz: Tensor,
    nx: Tensor,
    ny: Tensor,
    nz: Tensor,
    d2: float,
    cos_min: float,
) -> Tensor:
    """Any library contact within reach of each (rotated) request: ``(R,)`` from requests ``(R,)`` and candidates ``(M,)``."""

    dx = px[:, None] - cx[None, :]
    dy = py[:, None] - cy[None, :]
    dxy = torch.sqrt(dx * dx + dy * dy)
    gap = torch.clamp_min(dxy - smax[:, None], 0.0)
    dz = pz[:, None] - cz[None, :]
    close = gap * gap + dz * dz <= d2
    cosn = zx[:, None] * nx[None, :] + zy[:, None] * ny[None, :] + zz[:, None] * nz[None, :]
    return (close & (cosn.abs() >= cos_min)).any(1)


_reach_fast = compiled(_reach_core)
"""The reach core compiled on CUDA, with the opponent's shared fallback: if ``torch.compile`` fails (an image without
a C compiler), it runs eagerly with one ``RuntimeWarning``."""


# ---- the tables and the mixture -------------------------------------------------------------------------------------


class Library:
    """
    Logged rows drawn by up to two binned conditions: rows sorted by cell (``a``'s quantile bin, then ``b``'s), ``starts``
    the cell bounds, ``redirect`` each cell to itself or (empty) its nearest filled neighbour. A draw takes a row of the
    query's cell uniformly at random.
    """

    def __init__(self, rows: Tensor, edges_a: Tensor, edges_b: Tensor, starts: Tensor, redirect: Tensor) -> None:
        self.rows, self.edges_a, self.edges_b, self.starts, self.redirect = rows, edges_a, edges_b, starts, redirect

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], prefix: str, device: Any, dtype: Any) -> Library:
        def t(name: str, kind: Any = dtype) -> Tensor:
            return torch.as_tensor(np.asarray(arrays[f"{prefix}_{name}"]), dtype=kind, device=device)

        return cls(t("rows"), t("edges_a"), t("edges_b"), t("starts", torch.long), t("redirect", torch.long))

    def draw(self, a: Tensor, b: Tensor | None, u: Tensor) -> Tensor:
        ia = torch.bucketize(a.contiguous().to(self.edges_a.dtype), self.edges_a, right=True)
        cell = ia * (self.edges_b.numel() + 1)
        if b is not None and self.edges_b.numel():
            cell = cell + torch.bucketize(b.contiguous().to(self.edges_b.dtype), self.edges_b, right=True)
        cell = self.redirect[cell]
        lo, hi = self.starts[cell], self.starts[cell + 1]
        pick = torch.minimum(lo + (u * (hi - lo).to(u.dtype)).long(), hi - 1)
        return self.rows[pick]


class AcceptTable:
    """
    P(the runtime holds the contact | features): a table over quantile bins of a few features (``columns`` into the
    feature vector, ``edges`` their inner bin edges), each cell Beta-smoothed towards its parent (the table without the
    last feature) and monotone in the lead where fitted so.
    """

    def __init__(self, columns: Tensor, edges: Tensor, sizes: Tensor, probability: Tensor) -> None:
        self.columns, self.edges, self.sizes, self.probability = columns, edges, sizes, probability
        strides = torch.ones_like(sizes)
        for i in range(sizes.numel() - 2, -1, -1):
            strides[i] = strides[i + 1] * sizes[i + 1]
        self.strides = strides

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], prefix: str, device: Any, dtype: Any) -> AcceptTable:
        def t(name: str, kind: Any) -> Tensor:
            return torch.as_tensor(np.asarray(arrays[f"{prefix}_{name}"]), dtype=kind, device=device)

        return cls(t("columns", torch.long), t("edges", dtype), t("sizes", torch.long), t("probability", dtype))

    def __call__(self, features: Tensor) -> Tensor:
        cell = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
        for i in range(self.columns.numel()):
            x = features[:, self.columns[i]].contiguous()
            inner = self.edges[i, : int(self.sizes[i]) - 1].contiguous().to(x.dtype)
            cell = cell + torch.bucketize(x, inner, right=True) * self.strides[i]
        return self.probability[cell].to(features.dtype)


class AcceptNet:
    """
    P(the runtime holds the contact | features) as a small logistic network: the pre-registered fallback where a table's held-out log-loss is more than 5 % worse than a gradient-boosted tree's.
    """

    def __init__(self, module: Any, x_mean: Tensor, x_std: Tensor) -> None:
        self.module, self.x_mean, self.x_std = module, x_mean, x_std

    @staticmethod
    def build(inputs: int, hidden: int, device: Any, dtype: Any) -> Any:
        nn = torch.nn
        return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)).to(device=device, dtype=dtype)

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], prefix: str, hidden: int, device: Any, dtype: Any) -> AcceptNet:
        x_mean = np.asarray(arrays[f"{prefix}_x_mean"])
        module = cls.build(len(x_mean), hidden, device, dtype)
        state = {k[len(prefix) + 7 :]: torch.as_tensor(v) for k, v in arrays.items() if k.startswith(f"{prefix}_param/")}
        module.load_state_dict(state)
        module.eval()

        def t(name: str) -> Tensor:
            return torch.as_tensor(np.asarray(arrays[f"{prefix}_{name}"], dtype=np.float64), dtype=dtype, device=device)

        return cls(module, t("x_mean"), t("x_std"))

    def logit(self, features: Tensor) -> Tensor:
        return self.module((features.to(self.x_mean.dtype) - self.x_mean) / self.x_std)[:, 0]

    def __call__(self, features: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.logit(features)).to(features.dtype)


def accept_model(arrays: dict[str, np.ndarray], header: dict, name: str, device: Any) -> AcceptTable | AcceptNet:
    """The fitted acceptance ``name`` (``decide_accept`` / ``execute_accept``): a table, or the network fallback."""

    kind = header["fit"][name].get("kind", "table")
    if kind == "net":
        return AcceptNet.from_arrays(arrays, f"body_{name}", int(header["networks"]["accept_hidden"]), device, torch.float32)
    return AcceptTable.from_arrays(arrays, f"body_{name}", device, torch.float64)


class NudgeModel:
    """The held contact against the request: a mixture density network over :data:`NUDGE_TARGETS` given the features."""

    def __init__(self, module: Any, x_mean: Tensor, x_std: Tensor, y_mean: Tensor, y_std: Tensor) -> None:
        self.module, self.x_mean, self.x_std, self.y_mean, self.y_std = module, x_mean, x_std, y_mean, y_std

    @staticmethod
    def build(inputs: int, components: int, hidden: int, device: Any, dtype: Any) -> Any:
        from .intent import _mdn_class

        return _mdn_class()(inputs, len(NUDGE_TARGETS), components, hidden, profiles=0).to(device=device, dtype=dtype)

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], prefix: str, components: int, hidden: int, device: Any, dtype: Any) -> NudgeModel:
        x_mean = np.asarray(arrays[f"{prefix}_x_mean"])
        module = cls.build(len(x_mean), components, hidden, device, dtype)
        state = {k[len(prefix) + 7 :]: torch.as_tensor(v) for k, v in arrays.items() if k.startswith(f"{prefix}_param/")}
        module.load_state_dict(state, strict=False)
        module.eval()

        def t(name: str) -> Tensor:
            return torch.as_tensor(np.asarray(arrays[f"{prefix}_{name}"], dtype=np.float64), dtype=dtype, device=device)

        return cls(module, t("x_mean"), t("x_std"), t("y_mean"), t("y_std"))

    def standardise(self, features: Tensor) -> Tensor:
        return (features.to(self.x_mean.dtype) - self.x_mean) / self.x_std

    def draw(self, features: Tensor, generator: Any = None) -> Tensor:
        from .intent import categorical

        with torch.no_grad():
            x = self.standardise(features)
            logits, means, scale, _face = self.module(x, x[:, :0])
            m = x.shape[0]
            k = categorical(torch.exp(logits), 1, generator)[:, 0]
            mu = means.gather(1, k[:, None, None].expand(m, 1, means.shape[-1]))[:, 0]
            sc = scale.gather(1, k[:, None, None, None].expand(m, 1, scale.shape[-2], scale.shape[-1]))[:, 0]
            eps = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
            z = mu + (sc @ eps[..., None])[..., 0]
            p_flip = torch.sigmoid(_face.gather(1, k[:, None])[:, 0])
            flip = (torch.rand(p_flip.shape, dtype=p_flip.dtype, device=p_flip.device, generator=generator) < p_flip).to(z.dtype)
            return torch.cat([z * self.y_std + self.y_mean, flip[:, None]], -1).to(features.dtype)


# ---- the feasibility ------------------------------------------------------------------------------------------------


class BodyFeasibility:
    """
    The body runtime's verdict on contact requests, per row, from the rules export (``body_*``): see the module text.

    :param arrays: The rules export's arrays.
    :param header: Its header's ``body`` section (constants, table and network shapes, lineage).
    """

    def __init__(self, arrays: dict[str, np.ndarray], header: dict, device: Any, dtype: Any) -> None:
        self.header = header
        c = header["constants"]
        self.min_lead_s = float(c["min_lead_s"])
        self.place_full_lead_s = float(c["place_full_lead_s"])
        self.yaw_max = math.radians(float(c["yaw_max_deg"]))
        self.yaw_steps = int(c["yaw_steps"])
        self.shift_max = float(c["shift_max_m"])
        self.reach2 = (float(c["nudge_max_m"]) * float(c["prefilter"])) ** 2
        self.cos_min = math.cos(math.radians(float(c["nudge_max_deg"]) * float(c["prefilter"])))
        self.device, self.dtype = device, dtype
        self.candidates = []
        for role in ("suited", "opponent"):
            pos = torch.as_tensor(np.asarray(arrays[f"body_cand_pos_{role}"], dtype=np.float32), device=device)
            nrm = torch.as_tensor(np.asarray(arrays[f"body_cand_normal_{role}"], dtype=np.float32), device=device)
            self.candidates.append(
                (pos[:, 0].contiguous(), pos[:, 1].contiguous(), pos[:, 2].contiguous(), nrm[:, 0].contiguous(), nrm[:, 1].contiguous(), nrm[:, 2].contiguous())
            )
        self.start_hands = [
            torch.as_tensor(np.asarray(arrays[f"body_start_hand_{role}"], dtype=np.float64), dtype=dtype, device=device) for role in ("suited", "opponent")
        ]
        self.rally_hands = [Library.from_arrays(arrays, f"body_rally_hand_{role}", device, dtype) for role in ("suited", "opponent")]
        self.realisations = [Library.from_arrays(arrays, f"body_realise_{role}", device, dtype) for role in ("suited", "opponent")]
        self.tick_s = float(header["constants"]["tick_s"])
        net = header["networks"]
        self.decide_accept = accept_model(arrays, header, "decide_accept", device)
        self.execute_accept = accept_model(arrays, header, "execute_accept", device)
        self.decide_nudge = NudgeModel.from_arrays(arrays, "body_decide_nudge", int(net["components"]), int(net["hidden"]), device, torch.float32)
        self.execute_nudge = NudgeModel.from_arrays(arrays, "body_execute_nudge", int(net["components"]), int(net["hidden"]), device, torch.float32)
        self.replan_accept = accept_model(arrays, header, "replan_accept", device)
        self.replan_nudge = NudgeModel.from_arrays(arrays, "body_replan_nudge", int(net["components"]), int(net["hidden"]), device, torch.float32)

    def start_hand(self, profile: Tensor, stance: Tensor, flip: Tensor, generator: Any = None) -> Tensor:
        """
        The hand (blade centre) before a player's first stroke: the runtime's start pose (``BodyRuntime.start``, a
        donor 0.5-0.7 s before one of its contacts, at the stance, turned to the net), drawn from its library.
        """

        out = torch.zeros((profile.shape[0], 3), dtype=stance.dtype, device=stance.device)
        u = torch.rand(profile.shape[0], dtype=stance.dtype, device=stance.device, generator=generator)
        for role in (0, 1):
            library = self.start_hands[role]
            pick = (u * library.shape[0]).long().clamp(max=library.shape[0] - 1)
            out = torch.where((profile == role)[:, None], library[pick].to(stance.dtype), out)
        offset = yaw_vectors(out, flip)
        return torch.stack([stance[:, 0] + offset[:, 0], stance[:, 1] + offset[:, 1], offset[:, 2]], -1)

    def rally_hand(self, profile: Tensor, hips: Tensor, flip: Tensor, since: Tensor, generator: Any = None) -> tuple[Tensor, Tensor]:
        """
        The hand after a stroke, ``since`` seconds after its contact: the runtime's bodies carry on through their
        donors' follow-through and recovery, so it is drawn from the hands the runtime's bodies had at rally decisions
        (relative to the hips, with their velocity), among those the same time since their contact (binned).
        """

        dtype = hips.dtype
        u = torch.rand(profile.shape[0], dtype=dtype, device=hips.device, generator=generator)
        row = torch.where((profile == 0)[:, None], self.rally_hands[0].draw(since, None, u).to(dtype), self.rally_hands[1].draw(since, None, u).to(dtype))
        o = yaw_vectors(row[:, 1:4], flip)
        position = torch.stack([hips[:, 0] + o[:, 0], hips[:, 1] + o[:, 1], o[:, 2]], -1)
        return position, yaw_vectors(row[:, 4:7], flip)

    def realise(self, profile: Tensor, flip: Tensor, knot: Contact, ball_in: Tensor, generator: Any = None) -> tuple[Tensor, ...]:
        """
        How the oracle's paddle passes the held knot: in body-first it is the hand's, a cubic Hermite per 1/24 s render
        tick with the knot inside its tick (``SCHEMA.md``), so the ball, meeting the blade a few milliseconds off the knot,
        sees the hand's velocity there. The contact tick's two states (against the knot's constant-velocity line, in
        the player's frame, with the knot's phase in the tick) are drawn from the v1 release's contacts and the robot-ball
        receives with the nearest knot speed and incoming ball speed. Returns ``(t0, p0, v0, t1, p1, v1)``.
        """

        dtype = knot.position.dtype
        u = torch.rand(profile.shape[0], dtype=dtype, device=knot.position.device, generator=generator)
        speed = norm(knot.velocity)
        ball_speed = norm(ball_in[:, 3:6])
        row = torch.where(
            (profile == 0)[:, None], self.realisations[0].draw(speed, ball_speed, u).to(dtype), self.realisations[1].draw(speed, ball_speed, u).to(dtype)
        )
        t0 = knot.t - row[:, 0] * self.tick_s
        t1 = t0 + self.tick_s
        p0 = knot.position + knot.velocity * (t0 - knot.t)[:, None] + yaw_vectors(row[:, 1:4], flip)
        v0 = knot.velocity + yaw_vectors(row[:, 4:7], flip)
        p1 = knot.position + knot.velocity * (t1 - knot.t)[:, None] + yaw_vectors(row[:, 7:10], flip)
        v1 = knot.velocity + yaw_vectors(row[:, 10:13], flip)
        return t0, p0, v0, t1, p1, v1

    def reach(
        self,
        profile: Tensor,
        flip: Tensor,
        lead: Tensor,
        contact: Contact,
        face: Tensor,
        pelvis_contact: Tensor,
        chunk: int = 8192,
        roles: tuple[int, ...] = (0, 1),
    ) -> Tensor:
        """
        The runtime's donor prefilter (``BodyRuntime._rank``, the body at its plan's placement): some library contact
        within reach of the request (``no_donor`` otherwise). Exact for a start without a carried placement. ``roles``:
        the profiles that can occur in ``profile`` (a static hint: the robot's opponent is profile 1 only).
        """

        hips3 = torch.cat([pelvis_contact, torch.zeros_like(pelvis_contact[:, :1])], -1)
        p = yaw_vectors(contact.position - hips3, flip).float()
        z = yaw_vectors(quat_to_matrix(contact.orientation)[:, :, 2], flip).float()
        room = (lead / self.place_full_lead_s).clamp(0.0, 1.0).float()
        smax = self.shift_max * room
        ymax = self.yaw_max * room
        fn = _reach_fast
        out = torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
        for role in roles:
            cx, cy, cz, nx, ny, nz = self.candidates[role]
            mine = profile == role
            for k in range(2 * self.yaw_steps + 1):
                yaw = ymax * (k / self.yaw_steps - 1.0)
                c, s = torch.cos(yaw), torch.sin(yaw)
                # The candidates turned by ``yaw`` against the request is the request turned by ``-yaw``.
                px = c * p[:, 0] + s * p[:, 1]
                py = -s * p[:, 0] + c * p[:, 1]
                zx = c * z[:, 0] + s * z[:, 1]
                zy = -s * z[:, 0] + c * z[:, 1]
                hits = [
                    fn(
                        px[i : i + chunk],
                        py[i : i + chunk],
                        p[i : i + chunk, 2],
                        zx[i : i + chunk],
                        zy[i : i + chunk],
                        z[i : i + chunk, 2],
                        smax[i : i + chunk],
                        cx,
                        cy,
                        cz,
                        nx,
                        ny,
                        nz,
                        self.reach2,
                        self.cos_min,
                    )
                    for i in range(0, p.shape[0], chunk)
                ]
                out = out | (mine & torch.cat(hits))
        return out

    def hold_decision(
        self,
        params: VecParams,
        env: Tensor,
        profile: Tensor,
        flip: Tensor,
        t_from: Tensor,
        request: Contact,
        face: Tensor,
        ball_in: Tensor,
        wanted: Tensor,
        pelvis_contact: Tensor,
        hips_from: Tensor,
        hand_position: Tensor,
        hand_velocity: Tensor,
        generator: Any = None,
        steps: int | None = None,
        roles: tuple[int, ...] = (0, 1),
    ) -> tuple[Tensor, Contact, Tensor, Tensor]:
        """
        ``plan_stroke`` then the same-outcome rule for requests at a decision: ``(held (R,), the held contact, cause, the
        rubber it presents)`` with ``cause`` 0 held, 1 ``no_donor``/lead, 2 the runtime's other rejections, 3 the outcome
        changed. ``wanted`` is the request's own landing class (the decision's intended flight).
        """

        lead = request.t - t_from
        reach = self.reach(profile, flip, lead, request, face, pelvis_contact, roles=roles) & (lead >= self.min_lead_s)
        features = decision_features(profile, flip, lead, request, face, pelvis_contact, hips_from, hand_position, hand_velocity)
        p = self.decide_accept(features)
        u = torch.rand(p.shape, dtype=p.dtype, device=p.device, generator=generator)
        accepted = reach & (u < p)
        nudge = self.decide_nudge.draw(features.float(), generator).to(request.position.dtype)
        held = apply_nudge(request, nudge, flip)
        same = landing_class(params, env, ball_in, held, steps=steps) == wanted
        ok = accepted & same
        cause = torch.where(
            ~reach, torch.ones_like(env), torch.where(~accepted, torch.full_like(env, 2), torch.where(~same, torch.full_like(env, 3), torch.zeros_like(env)))
        )
        return ok, held, cause, torch.where(rubber_flipped(nudge), 1 - face, face)

    def hold_execution(
        self,
        params: VecParams,
        env: Tensor,
        profile: Tensor,
        flip: Tensor,
        t_from: Tensor,
        request: Contact,
        first: Contact,
        face: Tensor,
        ball_in: Tensor,
        generator: Any = None,
        first_face: Tensor | None = None,
    ) -> tuple[Tensor, Contact, Tensor, Tensor]:
        """
        ``Rally._stroke_body`` at execution for a noisy contact against the first plan's held one: ``renudge`` (on the
        first plan's body: its rubber, ``first_face``) and the same-outcome rule, else (with at least the least lead) the
        re-plan with the same donor and the rule again. Returns ``(held, contact, the request's landing class, the
        rubber presented)``.
        """

        lead = request.t - t_from
        features = execution_features(profile, flip, lead, request, first, face)
        dtype = request.position.dtype
        wanted = landing_class(params, env, ball_in, request)
        u = torch.rand((2, features.shape[0]), dtype=features.dtype, device=features.device, generator=generator)
        kept = face if first_face is None else first_face
        nudge_r = self.execute_nudge.draw(features.float(), generator).to(dtype)[:, : len(NUDGE_TARGETS)]
        nudge_r = torch.cat([nudge_r, (kept != face).to(dtype)[:, None]], -1)
        renudged = apply_nudge(request, nudge_r, flip)
        ok_r = (u[0] < self.execute_accept(features)) & (landing_class(params, env, ball_in, renudged) == wanted)
        nudge_p = self.replan_nudge.draw(features.float(), generator).to(dtype)
        replanned = apply_nudge(request, nudge_p, flip)
        ok_p = (lead >= self.min_lead_s) & (u[1] < self.replan_accept(features)) & (landing_class(params, env, ball_in, replanned) == wanted)
        face_p = torch.where(rubber_flipped(nudge_p), 1 - face, face)
        return ok_r | ok_p, renudged.where(ok_r, replanned), wanted, torch.where(ok_r, kept, face_p)


__all__ = [
    "CLASS_NAMES",
    "DECISION_FEATURES",
    "EXECUTION_FEATURES",
    "NUDGE_TARGETS",
    "AcceptNet",
    "AcceptTable",
    "BodyFeasibility",
    "Contact",
    "Library",
    "NudgeModel",
    "accept_model",
    "apply_nudge",
    "classify_landing",
    "decision_features",
    "execution_features",
    "landing_class",
    "rubber_flipped",
    "struck_normal",
]
