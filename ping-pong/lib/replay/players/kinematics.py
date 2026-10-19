"""
Forward kinematics and two-bone inverse kinematics on an ActorCore skeleton.

A pose is the local rotation of every joint plus the root (``Hip``)
translation in skeleton space, metres at body scale 1. A body placement maps
skeleton space into the table frame: a uniform scale (the body's height match)
and an origin on the floor. Skeleton space has the table frame's axes (the
character prim carries no rotation), so world rotations equal skeleton-space
rotations and world positions are ``origin + scale * p_skeleton``.

The same functions drive the rendered body (the poses written to the UsdSkel
``Pose`` prim) and the paddle (the hand joint's world transform), so the paddle
always sits in the rendered hand.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lib.replay.players import rotations as rot
from lib.replay.players.skeleton import ROOT_JOINT, SkeletonRest

Array = NDArray[Any]


@dataclass(frozen=True)
class BodyPlacement:
    """
    Where skeleton space sits in the table frame.

    :ivar scale: Uniform scale from body-scale-1 metres to table metres (0.9255 for the suited player).
    :ivar origin: Table-frame position of the skeleton origin, m (on the floor, below the table centre).
    """

    scale: float
    origin: tuple[float, float, float]

    def to_world(self, skeleton_points: ArrayLike) -> Array:
        """Skeleton-space points (body metres) to table-frame points."""

        return np.asarray(self.origin, dtype=np.float64) + self.scale * np.asarray(skeleton_points, dtype=np.float64)

    def to_skeleton(self, world_points: ArrayLike) -> Array:
        """Table-frame points to skeleton-space points (body metres)."""

        return (np.asarray(world_points, dtype=np.float64) - np.asarray(self.origin, dtype=np.float64)) / self.scale


def rest_pose(skeleton: SkeletonRest, batch: tuple[int, ...] = ()) -> tuple[Array, Array]:
    """
    The rest pose: rest local rotations and the rest root translation.

    :return: ``(local_rotation (..., J, 4), root_translation (..., 3))`` with leading ``batch`` dims.
    """

    local = np.broadcast_to(skeleton.rest_rotation, (*batch, *skeleton.rest_rotation.shape)).copy()
    root = np.broadcast_to(skeleton.rest_translation[skeleton.index(ROOT_JOINT)], (*batch, 3)).copy()
    return local, root


def forward_kinematics(
    skeleton: SkeletonRest, local_rotation: ArrayLike, root_translation: ArrayLike, placement: BodyPlacement, *, upto: int | None = None
) -> tuple[Array, Array]:
    """
    World rotations and positions of every joint, vectorised over leading batch dims.

    :param skeleton: The body's rest skeleton (bone offsets).
    :param local_rotation: Local rotations, shape ``(..., J, 4)``.
    :param root_translation: Root (``Hip``) local translation in skeleton space, body metres, shape ``(..., 3)``.
    :param placement: Skeleton space in the table frame.
    :param upto: Stop after this joint index (joints are ordered parents first); later rows stay zero.
    :return: World rotations ``(..., J, 4)`` and table-frame positions ``(..., J, 3)``.
    """

    local = np.asarray(local_rotation, dtype=np.float64)
    root = np.asarray(root_translation, dtype=np.float64)
    count = len(skeleton.joints)
    last = count if upto is None else upto + 1
    if local.ndim == 2 and root.shape == (3,):
        return _forward_rows(skeleton, local, root, placement, range(last))
    world_rotation = np.zeros(local.shape)
    skeleton_position = np.zeros((*local.shape[:-1], 3))
    root_index = skeleton.index(ROOT_JOINT)
    for index in range(last):
        parent = int(skeleton.parents[index])
        offset = root if index == root_index else skeleton.rest_translation[index]
        if parent < 0:
            world_rotation[..., index, :] = local[..., index, :]
            skeleton_position[..., index, :] = offset
        else:
            parent_rotation = world_rotation[..., parent, :]
            world_rotation[..., index, :] = rot.multiply(parent_rotation, local[..., index, :])
            skeleton_position[..., index, :] = skeleton_position[..., parent, :] + rot.rotate(parent_rotation, offset)
    return world_rotation, placement.to_world(skeleton_position)


def _forward_rows(skeleton: SkeletonRest, local: Array, root: Array, placement: BodyPlacement, rows: Any) -> tuple[Array, Array]:
    """
    :func:`forward_kinematics` of one pose (``local`` ``(J, 4)``) for the joints ``rows`` only (each joint's parent
    listed before it, every ancestor included); the other rows stay zero. The same IEEE arithmetic in the same order as
    the vectorised path (:func:`~lib.replay.players.rotations.quat_product` and ``quat_rotate``), on Python
    floats: per joint, numpy's call overhead on 4-element arrays was most of a replay tick's posing.
    """

    parents = skeleton.parents_list
    rest = skeleton.rest_translation_list
    root_index = skeleton.index(ROOT_JOINT)
    rotations_in = local.tolist()
    root_offset = tuple(root.tolist())
    world: dict[int, tuple[float, ...]] = {}
    position: dict[int, tuple[float, ...]] = {}
    for index in rows:
        parent = parents[index]
        offset = root_offset if index == root_index else rest[index]
        if parent < 0:
            world[index] = tuple(rotations_in[index])
            position[index] = tuple(offset)
        else:
            parent_rotation = world[parent]
            world[index] = rot.quat_product(parent_rotation, rotations_in[index])  # type: ignore[arg-type]
            px, py, pz = position[parent]
            rx, ry, rz = rot.quat_rotate(parent_rotation, offset)  # type: ignore[arg-type]
            position[index] = (px + rx, py + ry, pz + rz)
    world_rotation = np.zeros(local.shape)
    skeleton_position = np.zeros((local.shape[0], 3))
    keys = list(world)
    world_rotation[keys] = [world[key] for key in keys]
    skeleton_position[keys] = [position[key] for key in keys]
    return world_rotation, placement.to_world(skeleton_position)


def chain_kinematics(
    skeleton: SkeletonRest, chain: tuple[int, ...], local_rotation: ArrayLike, root_translation: ArrayLike, placement: BodyPlacement
) -> tuple[Array, Array]:
    """
    World rotations and positions along one root-to-joint chain only (cheap per-step evaluation).

    :param chain: Joint indices from the skeleton root, as :meth:`SkeletonRest.chain` returns.
    :return: World rotations ``(..., len(chain), 4)`` and positions ``(..., len(chain), 3)``.
    """

    local = np.asarray(local_rotation, dtype=np.float64)
    root = np.asarray(root_translation, dtype=np.float64)
    if local.ndim == 2 and root.shape == (3,):
        world_rotation, world_position = _forward_rows(skeleton, local, root, placement, chain)
        return world_rotation[list(chain)], world_position[list(chain)]
    root_index = skeleton.index(ROOT_JOINT)
    rotations: list[Array] = []
    positions: list[Array] = []
    for position_in_chain, index in enumerate(chain):
        offset = root if index == root_index else skeleton.rest_translation[index]
        if position_in_chain == 0:
            rotations.append(local[..., index, :])
            positions.append(np.broadcast_to(offset, (*local.shape[:-2], 3)).astype(np.float64))
        else:
            parent_rotation = rotations[-1]
            rotations.append(rot.multiply(parent_rotation, local[..., index, :]))
            positions.append(positions[-1] + rot.rotate(parent_rotation, offset))
    return np.stack(rotations, axis=-2), placement.to_world(np.stack(positions, axis=-2))


def set_world_rotation(skeleton: SkeletonRest, local_rotation: Array, world_rotation: Array, joint: int, desired_world: ArrayLike) -> None:
    """
    Change one joint's local rotation so its world rotation becomes ``desired_world``, in place.

    The joint's descendants keep their local rotations (so they follow). ``world_rotation``
    must hold the current world rotation of the joint's parent; the joint's own row is
    updated too.
    """

    parent = int(skeleton.parents[joint])
    desired = rot.normalize(desired_world)
    if parent < 0:
        local_rotation[..., joint, :] = desired
    else:
        local_rotation[..., joint, :] = rot.multiply(rot.conjugate(world_rotation[..., parent, :]), desired)
    world_rotation[..., joint, :] = desired


@dataclass(frozen=True)
class TwoBoneSolution:
    """
    World-frame deltas for a two-bone chain.

    :ivar upper_delta: Rotation applied (in the world frame, about the upper joint) to the upper bone and everything below it.
    :ivar elbow_delta: Rotation applied (in the world frame, about the middle joint) to the lower bone before ``upper_delta``.
    :ivar reached: Whether the target lay within the chain's reach.
    :ivar residual_m: Distance left between the end joint and the target, m.
    """

    upper_delta: Array
    elbow_delta: Array
    reached: bool
    residual_m: float


def soft_reach(wanted: float, longest: float, softness_m: float) -> float:
    """
    The distance a soft two-bone chain extends to for a wanted distance (soft IK).

    Below ``longest - softness_m`` the chain reaches exactly; beyond, the reach approaches ``longest`` exponentially,
    ``d = s + k (1 - exp(-(wanted - s) / k))`` with ``s = longest - k`` and ``k = softness_m``. The elbow never snaps
    straight: the hard clamp's elbow angle changes infinitely fast with the target as the arm locks
    (``d(angle)/d(distance) ~ 1 / sin(angle)``), and the soft curve's slope (``exp(-(wanted - s) / k)``) damps it to a
    bounded, vanishing rate. It is the damped solve of the chain's one singular degree of freedom.

    :param wanted: Shoulder-to-target distance, m.
    :param longest: The chain's full reach, m.
    :param softness_m: The soft band, m (0 is the hard clamp).
    :return: The distance to extend to, m.
    """

    if softness_m <= 0.0:
        return min(wanted, longest)
    start = longest - softness_m
    if wanted <= start:
        return wanted
    return start + softness_m * (1.0 - math.exp(-(wanted - start) / softness_m))


def two_bone_ik(
    upper: ArrayLike,
    middle: ArrayLike,
    end: ArrayLike,
    target: ArrayLike,
    *,
    reach_margin: float = 1e-3,
    bend_hint: ArrayLike | None = None,
    softness: float = 0.0,
) -> TwoBoneSolution:
    """
    Analytic two-bone IK that keeps the current bend plane.

    The middle joint first opens or closes in the plane of the chain so the
    upper-to-end distance equals the upper-to-target distance (law of cosines),
    then the whole chain swings about the upper joint by the smallest rotation
    that points it at the target.

    :param upper: Upper joint world position (shoulder, hip), m.
    :param middle: Middle joint world position (elbow, knee), m.
    :param end: End joint world position (wrist, ankle), m.
    :param target: Desired end-joint position, m.
    :param reach_margin: Fraction of the chain length kept back from full extension.
    :param bend_hint: Direction the lower bone leans away from the straight line (backward for a knee, forward
        for an elbow); when given it fixes the bend plane, otherwise the current plane is kept.
    :param softness: Soft-IK band as a fraction of the chain length (:func:`soft_reach`); 0 clamps hard.
    :return: The world-frame deltas.
    """

    a = np.asarray(upper, dtype=np.float64)
    b = np.asarray(middle, dtype=np.float64)
    c = np.asarray(end, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    l1 = float(np.linalg.norm(b - a))
    l2 = float(np.linalg.norm(c - b))
    wanted = float(np.linalg.norm(t - a))
    longest = (l1 + l2) * (1.0 - reach_margin)
    shortest = abs(l1 - l2) * (1.0 + reach_margin) + 1e-6
    d = max(soft_reach(wanted, longest, softness * (l1 + l2)), shortest)
    reached = shortest <= wanted <= longest
    u = (a - b) / max(l1, 1e-12)
    lower = c - b
    v = lower - float(lower @ u) * u
    if bend_hint is not None:
        hint = np.asarray(bend_hint, dtype=np.float64)
        hinted = hint - float(hint @ u) * u
        if float(np.linalg.norm(hinted)) > 1e-6:
            v = hinted
    if float(np.linalg.norm(v)) < 1e-9:
        helper = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        v = helper - float(helper @ u) * u
    v /= np.linalg.norm(v)
    cos_inner = np.clip((l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2), -1.0, 1.0)
    inner = float(np.arccos(cos_inner))
    new_lower = l2 * (np.cos(inner) * u + np.sin(inner) * v)
    elbow_delta = rot.min_arc(lower, new_lower)
    new_end = b + new_lower
    upper_delta = rot.min_arc(new_end - a, t - a)
    final_end = a + rot.rotate(upper_delta, new_end - a)
    return TwoBoneSolution(upper_delta=upper_delta, elbow_delta=elbow_delta, reached=reached, residual_m=float(np.linalg.norm(final_end - t)))


def apply_two_bone(
    skeleton: SkeletonRest,
    local_rotation: Array,
    root_translation: Array,
    placement: BodyPlacement,
    joints: tuple[str, str, str],
    target_position: ArrayLike,
    target_rotation: ArrayLike | None = None,
    *,
    bend_hint: ArrayLike | None = None,
    softness: float = 0.0,
) -> TwoBoneSolution:
    """
    Pose a limb so its end joint reaches ``target_position`` (and optionally ``target_rotation``), in place.

    :param skeleton: Rest skeleton.
    :param local_rotation: Local rotations ``(J, 4)``; modified in place.
    :param root_translation: Root translation, body metres.
    :param placement: Skeleton space in the table frame.
    :param joints: Upper, middle and end joint names, e.g. ``("R_Upperarm", "R_Forearm", "R_Hand")``.
    :param target_position: Desired end-joint position, table frame, m.
    :param target_rotation: Desired end-joint world rotation; ``None`` keeps the end joint's local rotation.
    :param bend_hint: See :func:`two_bone_ik`.
    :param softness: See :func:`two_bone_ik`.
    :return: The IK solution.
    """

    upper, middle, end = (skeleton.index(name) for name in joints)
    local = np.asarray(local_rotation, dtype=np.float64)
    root = np.asarray(root_translation, dtype=np.float64)
    if local.ndim == 2 and root.shape == (3,):
        # Only the limb's chain is read below (its joints' world rotations and positions, and their parents'): the same
        # rows as the whole forward pass, without the rest of the body.
        world_rotation, world_position = _forward_rows(skeleton, local, root, placement, skeleton.chain(joints[2]))
    else:
        world_rotation, world_position = forward_kinematics(skeleton, local_rotation, root_translation, placement, upto=end)
    solution = two_bone_ik(world_position[upper], world_position[middle], world_position[end], target_position, bend_hint=bend_hint, softness=softness)
    new_upper = rot.multiply(solution.upper_delta, world_rotation[upper])
    new_middle = rot.multiply(solution.upper_delta, rot.multiply(solution.elbow_delta, world_rotation[middle]))
    set_world_rotation(skeleton, local_rotation, world_rotation, upper, new_upper)
    _refresh_between(skeleton, local_rotation, world_rotation, upper, middle)
    set_world_rotation(skeleton, local_rotation, world_rotation, middle, new_middle)
    _refresh_between(skeleton, local_rotation, world_rotation, middle, end)
    if target_rotation is not None:
        set_world_rotation(skeleton, local_rotation, world_rotation, end, target_rotation)
    return solution


def twist_angle(q: ArrayLike, axis: ArrayLike) -> float:
    """The signed angle of ``q``'s twist about ``axis`` (swing-twist split), in ``(-pi, pi]``."""

    _swing, twist = rot.swing_twist(q, axis)
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    return float(2.0 * math.atan2(float(twist[1:] @ a), float(twist[0])))


FOREARM_TWIST_SHARES: tuple[tuple[str, float], ...] = (("ForearmTwist01", 0.28), ("ForearmTwist02", 0.67))
"""The forearm twist joints' share of a pronation, at their skin-weight centroids (as the motion pipeline distributes it)."""


def continuous_twist(raw: float, previous: float | None) -> float:
    """The ``2 pi`` branch of a wrapped twist angle nearest the previous frame's (``raw`` itself without one), rad."""

    if previous is None:
        return raw
    return raw + 2.0 * math.pi * round((previous - raw) / (2.0 * math.pi))


def soft_twist(wanted: float, previous: float | None, *, limit_rad: float, rate_rad: float) -> float:
    """
    The added twist to apply: softly saturated at ``limit_rad`` (``limit tanh(wanted / limit)``, the soft-IK curve's
    philosophy for the forearm's pronation range) and softly rate-limited to ``rate_rad`` from the previous frame's
    (``previous + rate tanh(change / rate)``), so a demand that jumps or winds never turns the forearm in one frame.
    """

    target = limit_rad * math.tanh(wanted / limit_rad) if limit_rad > 0.0 else wanted
    if previous is None or rate_rad <= 0.0:
        return target
    return previous + rate_rad * math.tanh((target - previous) / rate_rad)


def transfer_added_twist(
    skeleton: SkeletonRest,
    local_rotation: Array,
    before: Array,
    parent: str,
    child: str,
    *,
    shares: Mapping[str, float] | None = None,
    previous: tuple[float, float] | None = None,
    limit_rad: float = 0.0,
    rate_rad: float = 0.0,
) -> tuple[float, float]:
    """
    Move into the parent only the twist the IK *added* to the child, relative to its pose ``before`` the IK, in place.

    The recorded motion already shares the forearm's and the hand's twist naturally. Moving the child's whole twist
    jumps when the swing-twist split nears its singularity (a sharply bent wrist), which played as 40-160 degree pops
    of the forearm and hand under large reference corrections; the IK's own addition is continuous with the blend
    weight and moves without them.

    Only the IK's addition moves, so a track that already carries its pronation in the parent (the session tracks put
    it in the forearm, not the hand) is not twisted twice: with the hand at its recorded pose nothing moves. The parent's
    other children keep their world rotations except for their ``shares`` of the added twist (the forearm twist
    joints take 0.28 and 0.67 of it, as the motion pipeline distributes a pronation; the skin is on them, none on the
    forearm), so the added roll ramps along the forearm instead of wringing the skin at one end.

    Over time the added twist is continuous: the IK's demand takes the ``2 pi`` branch nearest the ``previous`` frame's
    demand (:func:`continuous_twist`; a wrapped twist flipping from +170 to -170 degrees moved the forearm through a
    full turn in one frame), and :func:`soft_twist` bounds what the forearm takes (``limit_rad``) and its change from
    the previous frame's (``rate_rad`` per call). The child keeps the rest: its world rotation (the IK's target) is
    unchanged whatever the forearm takes.

    :param before: The child's local rotation before the IK.
    :param shares: Share of the added twist each other child of ``parent`` takes, by joint name (missing: 0). ``None``
        uses :data:`FOREARM_TWIST_SHARES` with the child's side prefix.
    :param previous: The previous frame's ``(demand, applied)`` twists, rad (``None``: no history).
    :param limit_rad: Soft bound on the applied twist (0: none).
    :param rate_rad: Soft bound on its change from the previous applied twist (0: none).
    :return: This frame's ``(demand, applied)`` twists, rad (signed; the demand unwrapped).
    """

    parent_index, child_index = skeleton.index(parent), skeleton.index(child)
    axis = skeleton.rest_translation[child_index]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    delta = twist_angle(local_rotation[child_index], axis) - twist_angle(before, axis)
    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    demand = continuous_twist(delta, None if previous is None else previous[0])
    delta = soft_twist(demand, None if previous is None else previous[1], limit_rad=limit_rad, rate_rad=rate_rad)
    moved = rot.from_axis_angle(axis, delta)
    local_rotation[parent_index] = rot.multiply(local_rotation[parent_index], moved)
    local_rotation[child_index] = rot.multiply(rot.conjugate(moved), local_rotation[child_index])
    if shares is None:
        side = child.split("_", 1)[0]
        shares = {f"{side}_{name}": share for name, share in FOREARM_TWIST_SHARES}
    # The parent's other descendants, parents first: each takes its share of the added twist about the bone axis
    # (a joint without one keeps its world rotation if it hangs on the parent, else rides its own parent).
    parents = np.asarray(skeleton.parents)
    inherited = {parent_index: 1.0}
    chains = {parent_index: rot.identity()}
    queue = [int(k) for k in np.flatnonzero(parents == parent_index) if int(k) != child_index]
    while queue:
        index = queue.pop(0)
        above = int(parents[index])
        name = skeleton.joints[index]
        share = shares[name] if name in shares else (0.0 if above == parent_index else inherited[above])
        if share != inherited[above]:
            local_axis = rot.rotate(rot.conjugate(chains[above]), axis)
            local_rotation[index] = rot.multiply(rot.from_axis_angle(local_axis, (share - inherited[above]) * delta), local_rotation[index])
        inherited[index] = share
        chains[index] = rot.multiply(chains[above], local_rotation[index])
        queue.extend(int(k) for k in np.flatnonzero(parents == index))
    return demand, delta


def _refresh_between(skeleton: SkeletonRest, local_rotation: Array, world_rotation: Array, ancestor: int, joint: int) -> None:
    """Recompute world rotations of the joints strictly between ``ancestor`` and ``joint``'s parent, top down."""

    path: list[int] = []
    index = int(skeleton.parents[joint])
    while index != ancestor and index >= 0:
        path.append(index)
        index = int(skeleton.parents[index])
    for index in reversed(path):
        parent = int(skeleton.parents[index])
        world_rotation[..., index, :] = rot.multiply(world_rotation[..., parent, :], local_rotation[..., index, :])


__all__ = [
    "FOREARM_TWIST_SHARES",
    "BodyPlacement",
    "TwoBoneSolution",
    "apply_two_bone",
    "chain_kinematics",
    "continuous_twist",
    "forward_kinematics",
    "rest_pose",
    "set_world_rotation",
    "soft_reach",
    "soft_twist",
    "transfer_added_twist",
    "twist_angle",
    "two_bone_ik",
]
